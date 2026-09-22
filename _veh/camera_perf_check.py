#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""camera_perf_check.py —— T6 的性能判据（真机相机页的原始日志）。

## 数据来源（都在 hilog 里，不需要改代码）

- `RATE`  行：arrive/done fps、丢帧、`rt_hit_p50`（往返 p50，按是否检出分桶）、
             `prep_hit_p50`、thermal、battC
- `STAGE` 行：native 侧 `conv` / `infer` ms（每 30 帧一条抽样）
- 布局树    ：`roiSegText`（**只有 ROI 路径才有**），给出 `veh … + roi …` 两段耗时

## 判据

① `frameMs p50 = prep_hit_p50 + rt_hit_p50`，与 33.3 ms 帧预算对比，给余量
② 丢帧率 = `cum_dropped / cum_arrived`
③ 有 `roiSegText` 时拆出 veh / roi 两段，指出各自占比
④ 超预算时给出「减 ROI 数 / 降车辆检测分辨率」的量化空间（按每框约 17 ms 估）

⚠️ **只搬运不换算**：缺字段就跳过并写明，**不补 0**（补 0 会造出假的 0 ms）。
⚠️ 取**后 1/3** 窗口做统计：前段含相机爬坡与模型冷启动，混进去会把基线拉偏。

## 用法

    python _veh/camera_perf_check.py --log _veh/devlog_T6.txt \\
        [--layout-run _veh/t6_layout.txt] [--path roi|direct]
"""
import argparse
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

BUDGET_MS = 33.3
# 实测：ROI 路径下每个车框跑一次车牌检测约 17 ms（69 ms / 4 框）
MS_PER_ROI = 17.0


def parse_rate_lines(log_path):
    """抽出所有 RATE 窗口的字段字典。"""
    out = []
    for ln in open(log_path, encoding="utf-8", errors="replace"):
        if "RATE" not in ln or "arrive=" not in ln:
            continue
        body = ln[ln.find("RATE") + 4:]
        d = {}
        for tok in body.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        out.append(d)
    return out


def fnum(d, k):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return None


def parse_stage_lines(log_path):
    out = []
    for ln in open(log_path, encoding="utf-8", errors="replace"):
        if "STAGE" not in ln or "infer=" not in ln:
            continue
        body = ln[ln.find("STAGE") + 5:]
        d = {}
        for tok in body.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        out.append(d)
    return out


def parse_seg(text):
    """从 `veh 73.3 + roi 69.0 ms · 车框 4→3（跨类去重 1）· 牌去重丢 0` 抠两个数。"""
    m = re.search(r"veh\s+([\d.]+)\s*\+\s*roi\s+([\d.]+)", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="相机页 hilog")
    ap.add_argument("--layout-run", default="", help="相机运行时的布局树 dump")
    ap.add_argument("--path", default="", choices=["", "roi", "direct"],
                    help="本轮采集走的是哪条路径（用于标注结论口径）")
    a = ap.parse_args()

    if not os.path.exists(a.log):
        print(f"[t6perf] FAIL 日志不存在：{a.log}")
        return 1

    rates = parse_rate_lines(a.log)
    stages = parse_stage_lines(a.log)
    prob, note = [], []

    if not rates:
        prob.append("日志里没有 RATE 行 —— 相机未进入统计窗口，或采集时相机没跑")
    else:
        # 取后 1/3 窗口：前段含爬坡与冷启动
        keep = rates[len(rates) * 2 // 3:] or rates
        note.append(f"[t6perf] RATE 窗口 {len(rates)} 个，取后 {len(keep)} 个做稳态统计")

        prep = [v for v in (fnum(d, "prep_hit_p50") for d in keep) if v is not None]
        rt = [v for v in (fnum(d, "rt_hit_p50") for d in keep) if v is not None]
        if not prep or not rt:
            prob.append("RATE 行缺 prep_hit_p50 / rt_hit_p50 —— 无法算 frameMs")
        else:
            # 每个窗口各取 p50，再对这些窗口值取中位（跨窗口的稳健估计）
            import statistics
            prep_m = statistics.median(prep)
            rt_m = statistics.median(rt)
            frame_m = prep_m + rt_m
            margin = BUDGET_MS - frame_m
            note.append(f"[t6perf] prep_hit_p50 中位 = {prep_m:.2f} ms")
            note.append(f"[t6perf] rt_hit_p50   中位 = {rt_m:.2f} ms")
            note.append(f"[t6perf] frameMs p50  ≈ {frame_m:.2f} ms（= prep + rt）")
            note.append(f"[t6perf] 余量 = 33.3 − {frame_m:.2f} = {margin:+.2f} ms")
            if margin < 0:
                note.append("[t6perf] ⚠️ 超预算。量化空间：")
                note.append(f"[t6perf]    · 每减 1 个 ROI ≈ 省 {MS_PER_ROI:.0f} ms"
                            f"（实测 69 ms / 4 框）")
                n_need = int(-margin / MS_PER_ROI) + 1
                note.append(f"[t6perf]    · 要补回 {abs(margin):.1f} ms，约需减少 {n_need} 个 ROI")
                note.append("[t6perf]    · 或降车辆检测输入分辨率（需另测，本脚本不臆断收益）")
            else:
                note.append(f"[t6perf] 在预算内，余 {margin:.2f} ms")

        # 丢帧
        arr = [v for v in (fnum(d, "cum_arrived") for d in keep) if v is not None]
        drop = [v for v in (fnum(d, "cum_dropped") for d in keep) if v is not None]
        if arr and drop and arr[-1] > 0:
            rate_drop = drop[-1] / arr[-1]
            note.append(f"[t6perf] 累计到达 {int(arr[-1])} 帧 / 丢 {int(drop[-1])} 帧 "
                        f"→ 丢帧率 {rate_drop*100:.2f}%")
        # 温度（热污染会让后段数据不可比）
        th = [v for v in (fnum(d, "thermal") for d in keep) if v is not None]
        if th:
            note.append(f"[t6perf] 热档 {th[0]:.0f} → {th[-1]:.0f}（若跨档，前后段不可直接比）")

    if stages:
        conv = [v for v in (fnum(d, "conv") for d in stages) if v is not None]
        inf = [v for v in (fnum(d, "infer") for d in stages) if v is not None]
        if conv and inf:
            import statistics
            note.append(f"[t6perf] STAGE（native 侧，{len(stages)} 条抽样）："
                        f"conv 中位 {statistics.median(conv):.2f} ms，"
                        f"infer 中位 {statistics.median(inf):.2f} ms")

    # 分段（只有 ROI 路径的布局树里才有 roiSegText）
    segv = segr = None
    if a.layout_run and os.path.exists(a.layout_run):
        txt = open(a.layout_run, encoding="utf-8", errors="replace").read()
        m = re.search(r"veh\s+[\d.]+\s*\+\s*roi\s+[\d.]+\s*ms[^\n]*", txt)
        if m:
            segv, segr = parse_seg(m.group(0))
    if segv is not None:
        tot = segv + segr
        note.append(f"[t6perf] ROI 分段：veh {segv:.1f} ms（{segv/tot*100:.0f}%）"
                    f" + roi {segr:.1f} ms（{segr/tot*100:.0f}%）")
        note.append(f"[t6perf]   ⇒ 车辆检测占 {segv/tot*100:.0f}%，去重/top-N 只能省 roi 那段")
    else:
        note.append("[t6perf] 未从布局树取到 roiSegText（直检路径本就没有，属正常）")

    print()
    for ln in note:
        print(ln)
    print()
    if prob:
        print(f"[t6perf] ✗ 发现 {len(prob)} 个问题：")
        for p in prob:
            print("   - " + p)
        return 1
    print(f"[t6perf] ✓ 性能判据完成"
          + (f"（路径 = {a.path}）" if a.path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
