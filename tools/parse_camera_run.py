"""解析一轮相机复采的落盘文件，产出逐窗口 CSV 与逐档汇总。

## 为什么需要它

2026-09-21 定位到一个**读数陷阱**：界面上的「延迟 20–25 ms」与「有时 13 ms」
被当成抖动，实际是**档位 × 是否检出**两个变量混在同一个滑动窗口里：

    生产档  未检出 infer p50 20.30 ms (n=92)  vs  检出 30.32 ms (n=21)
    全 NPU  未检出 infer p50 13.64 ms (n=27)  vs  检出 33.14 ms (n=1)

未检出帧不跑矫正 / 识别 / CTC / 判色，所以便宜一大截。把两者混桶取中位数，
得到的是两种分布的混合物 —— 这个脚本的全部意义就是**永不混桶**。

同时它强制报**余量**（`33.3 ms − p50(frameMs)`）而不是 `1000/p50`：
后者是服务时间的倒数，不是 fps（见 CONTEXT.md 的「服务时间 / 到达率 / 完成率」），
在相机路上它读出的「40–50 帧」物理上不可达（相机上限实测 ~30 fps）。

## 输入

App 落盘的 `camera_run_<ts>.log`（由 `CameraPage.ets` 的 `dumpLine()` 写）。
行格式：

    CAMRUN BEGIN ts=... gear=0 analyze=640x480 ...
    STREAM START show=960x960 analyze=640x480 rot=90
    SETTLE done after 45 s gear=0
    GEAR SWITCH -> 1 (全 NPU) settle restarts
    RATE win=1 arrive=29.9 done=29.9 ... gear=0 hit_n=12 empty_n=48 \
         infer_hit_p50=30.32 infer_empty_p50=20.30 conv_hit_p50=3.68 \
         conv_empty_p50=2.88 thermal=2 battC=36.0
    STAGE n=30 conv=... infer=... count=0 wh=... stages=...

## 输出

    evidence/camera_windows.csv    逐窗口（含档位、热态、分桶 p50）
    evidence/camera_summary.md     逐档汇总 + 余量表 + 口径声明

用法：
    python tools/parse_camera_run.py evidence/camera_prod-r1.log
    python tools/parse_camera_run.py evidence/camera_prod-r1.log evidence/camera_npu-r1.log
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FRAME_BUDGET_MS = 1000.0 / 30.0      # 相机 30 fps 下的每帧预算 = 33.33 ms
MIN_HIT_SAMPLES = 5                  # 检出桶样本少于这个数就不报 p50（标「样本不足」）
GEAR_NAMES = {0: "生产 det=CPU rec=NPU cls=CPU", 1: "全 NPU", 2: "全 GPU(Vulkan)",
              3: "基准：只取帧不推理", 4: "只数 callback"}

RATE_RE = re.compile(
    r"^RATE win=(\d+) arrive=([\d.]+) done=([\d.]+) dropped_in_win=(\d+) "
    r"cum_arrived=(\d+) cum_done=(\d+) cum_dropped=(\d+) gear=(\d+) "
    r"hit_n=(\d+) empty_n=(\d+) rt_hit_p50=(-?[\d.]+) rt_empty_p50=(-?[\d.]+) "
    r"prep_hit_p50=(-?[\d.]+) prep_empty_p50=(-?[\d.]+) thermal=(-?\d+) battC=([\d.]+)"
)
# 中间版本：字段名曾叫 infer_*/conv_*，含义是「JS↔native 往返」/「JS 侧 prep」，
# 与 STAGE 行的 native conv=/infer= **同名却不同量**，已改名（见 CameraPage.record）。
# 这里保留解析以免已采到的证据作废，但汇总时**必须**标注它是往返口径。
RATE_MID_RE = re.compile(
    r"^RATE win=(\d+) arrive=([\d.]+) done=([\d.]+) dropped_in_win=(\d+) "
    r"cum_arrived=(\d+) cum_done=(\d+) cum_dropped=(\d+) gear=(\d+) "
    r"hit_n=(\d+) empty_n=(\d+) infer_hit_p50=(-?[\d.]+) infer_empty_p50=(-?[\d.]+) "
    r"conv_hit_p50=(-?[\d.]+) conv_empty_p50=(-?[\d.]+) thermal=(-?\d+) battC=([\d.]+)"
)
STAGE_RE = re.compile(
    r"^STAGE n=(\d+) conv=([\d.]+) infer=([\d.]+) count=(\d+) wh=(\d+)x(\d+) "
    r"rgbaSum=(\d+) stages=(\S+)"
)
# 旧格式（没有分桶字段）—— 也要能读，否则老证据无法与新证据并列对照。
# `gear=` 是 2026-09-21 才加的（在「callback 档不打 RATE」那次误读之后），
# 所以更早的日志没有它 —— 必须可选，否则那份证据整份读不出来。
RATE_OLD_RE = re.compile(
    r"^RATE arrive=([\d.]+) fps done=([\d.]+) fps dropped_in_win=(\d+) "
    r"cum_arrived=(\d+) cum_done=(\d+) cum_dropped=(\d+)(?: gear=(\d+))?"
)
# hilog 行前缀：`09-21 04:16:51.671  7173  7173 I A0D001/com.shusen.lprdemo/LprCamera: `
# 归档的旧证据是 hilog 原文，App 落盘的新证据没有前缀。两条都要能吃 ——
# 否则「新证据 vs 旧证据」的并列对照做不了，而那正是本轮要给出的东西。
HILOG_PREFIX_RE = re.compile(
    r"^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+\d+\s+\d+\s+[A-Z]\s+.*?:\s"
)


def strip_prefix(line: str) -> str:
    """剥掉 hilog 前缀（若有），返回从 tag 之后开始的消息体。"""
    m = HILOG_PREFIX_RE.match(line)
    return line[m.end():] if m else line


def parse(path: Path):
    """解析一份落盘文件 -> (header, windows, stages, switches)。"""
    header = {}
    windows = []
    stages = []
    switches = []
    gear = None
    settled_seen = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_prefix(raw.strip())
        if not line:
            continue
        if line.startswith("CAMRUN BEGIN"):
            for kv in line.split()[2:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    header[k] = v
        elif line.startswith("GEAR SWITCH"):
            m = re.search(r"-> (\d+)", line)
            if m:
                gear = int(m.group(1))
                switches.append({"gear": gear, "line": line})
        elif line.startswith("SETTLE done"):
            settled_seen = True
        elif line.startswith("STAGE "):
            m = STAGE_RE.match(line)
            if m:
                stages.append({
                    "n": int(m.group(1)), "conv": float(m.group(2)),
                    "infer": float(m.group(3)), "count": int(m.group(4)),
                    "w": int(m.group(5)), "h": int(m.group(6)),
                    "rgba_sum": int(m.group(7)), "stages": m.group(8),
                })
        else:
            m = RATE_RE.match(line)
            mid = False
            if not m:
                m = RATE_MID_RE.match(line)
                mid = True
            if m:
                g = int(m.group(8))
                gear = g
                windows.append({
                    "win": int(m.group(1)), "arrive": float(m.group(2)),
                    "done": float(m.group(3)), "dropped": int(m.group(4)),
                    "cum_arrived": int(m.group(5)), "cum_done": int(m.group(6)),
                    "cum_dropped": int(m.group(7)), "gear": g,
                    "hit_n": int(m.group(9)), "empty_n": int(m.group(10)),
                    # 两个版本里这两对字段的**含义相同**（都是往返 / 都是 JS prep），
                    # 只是名字从 infer/conv 改成了 rt/prep。统一存成 rt_*/prep_*。
                    "rt_hit_p50": float(m.group(11)),
                    "rt_empty_p50": float(m.group(12)),
                    "prep_hit_p50": float(m.group(13)),
                    "prep_empty_p50": float(m.group(14)),
                    "thermal": int(m.group(15)), "battC": float(m.group(16)),
                    "bucketed": True,
                    # 中间版本的字段名有歧义（与 STAGE 的 native 口径同名），
                    # 汇总时必须标出来，否则读者会把往返值当 native 值。
                    "ambiguous_names": mid,
                })
                continue
            m = RATE_OLD_RE.match(line)
            if m:
                # 缺 gear= 的老日志：记 -1，汇总时单列一档「未知（旧日志）」。
                # **不要**猜成 0 —— 那会把两轮不同条件的数据合并，正是本轮要根除的病。
                g = int(m.group(7)) if m.group(7) is not None else -1
                gear = g
                windows.append({
                    "win": len(windows) + 1, "arrive": float(m.group(1)),
                    "done": float(m.group(2)), "dropped": int(m.group(3)),
                    "cum_arrived": int(m.group(4)), "cum_done": int(m.group(5)),
                    "cum_dropped": int(m.group(6)), "gear": g,
                    "hit_n": -1, "empty_n": -1,
                    "rt_hit_p50": -1.0, "rt_empty_p50": -1.0,
                    "prep_hit_p50": -1.0, "prep_empty_p50": -1.0,
                    "thermal": -1, "battC": -1.0, "bucketed": False,
                    "ambiguous_names": False,
                })
    return header, windows, stages, switches, settled_seen


def fmt(v, nd=2):
    """通用数值：None/负 = 无数据。

    ⚠️ **不能用于「余量」** —— 余量为负是有意义的（超预算），
    用这个函数会把最关键的结论显示成 `-`。余量请用 `fmt_margin`。
    """
    return "-" if v is None or v < 0 else f"{v:.{nd}f}"


def fmt_margin(v):
    """余量专用：**负值必须显示出来**（= 我们比相机慢，吃超了帧预算）。

    2026-09-21 实测生产档检出帧 frame p50 = 40 ms > 33.33 ms 预算，
    余量 -6.7 ms —— 这正是「为什么到不了 30 fps」的答案。
    用通用 fmt() 会把它渲染成 `-`，等于把结论藏了。
    """
    if v is None:
        return "N/A"
    return f"{v:+.2f}"


def fmt_infer(v):
    """infer p50 的显示：0 表示**这一档不推理**（基准档），不是「0 ms」。

    基准档（gear=3）只取帧不推理。App 侧已不再把 0 推进桶（正常应读到 -1），
    但万一读到 0（旧构建），必须显示成 N/A —— 否则会被读成「推理只要 0 ms」，
    那是完全不同的一件事。
    """
    if v is None or v < 0:
        return "N/A"
    if v == 0:
        return "N/A（本档不推理）"
    return f"{v:.2f}"


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    paths = [Path(a) for a in argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"[FAIL] 找不到 {p}")
            return 2

    root = Path(__file__).resolve().parent.parent
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)

    all_windows = []
    per_file = []
    for p in paths:
        header, windows, stages, switches, settled = parse(p)
        per_file.append((p, header, windows, stages, switches, settled))
        for w in windows:
            w["source"] = p.name
            all_windows.append(w)

    # ── 逐窗口 CSV ──────────────────────────────────────────────────────────
    csv_path = evidence / "camera_windows.csv"
    cols = ["source", "gear", "win", "arrive", "done", "dropped", "cum_arrived",
            "cum_done", "cum_dropped", "hit_n", "empty_n", "rt_hit_p50",
            "rt_empty_p50", "prep_hit_p50", "prep_empty_p50", "thermal", "battC",
            "bucketed"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for w in all_windows:
            wr.writerow(w)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    out = []
    out.append("# 相机复采汇总（由 tools/parse_camera_run.py 生成）\n")
    out.append("## 口径声明\n")
    out.append(f"- 帧预算 = {FRAME_BUDGET_MS:.2f} ms（相机 30 fps）")
    out.append("- **余量 = 帧预算 − p50(frameMs)**。为正 = 我们没吃满相机预算；")
    out.append("  为负 = 我们比相机慢，到达率会被我们拖住。")
    out.append("- **不使用 `1000/p50(frameMs)`**：那是服务时间的倒数，不是 fps")
    out.append("  （CONTEXT.md：服务时间 / 到达率 / 完成率是三个量）。")
    out.append("- p50 **按检出 / 未检出分桶**。未检出帧不跑矫正/识别/判色，便宜得多；")
    out.append("  混桶得到的是两种分布的混合物（这正是「20–25 又 13 ms」的成因）。")
    out.append(f"- 检出桶样本 < {MIN_HIT_SAMPLES} 时标「样本不足」，**不给 p50**。\n")

    # 输入文件与条件
    out.append("## 输入与条件\n")
    for p, header, windows, stages, switches, settled in per_file:
        out.append(f"### {p.name}\n")
        if header:
            out.append("| 字段 | 值 |")
            out.append("|---|---|")
            for k in ("gear", "analyze", "show", "rotation", "settleMs",
                      "warmupFrames", "window", "detModel", "recModel",
                      "analyzeTargetW", "fpsTarget"):
                if k in header:
                    out.append(f"| {k} | `{header[k]}` |")
            out.append("")
        out.append(f"- 稳态门通过：**{'是' if settled else '否'}**")
        out.append(f"- 窗口数：{len(windows)}；档位切换：{len(switches)}")
        out.append(f"- STAGE 抽样行：{len(stages)}")
        if not settled:
            out.append("  ⚠️ 未看到 `SETTLE done` —— 要么时长不够，要么这份日志来自"
                       "改造前的构建，下面的数字含相机爬坡期。")
        out.append("")

    # 逐档汇总。**按 (文件, 档位) 分组，不跨文件合并** —— 每个文件是一轮独立运行，
    # 轮与轮之间热态不同（CONTEXT.md：热档 2→3 时 NPU 延迟 +15~20%），
    # 合并会把「热态差异」伪装成「测量噪声」。跨轮聚合只在最后那张余量表里做，
    # 并且并列给出各轮的热档，让读者自己判断能不能合并。
    out.append("## 逐档汇总（仅稳态窗口，按轮分开）\n")
    by_run = {}
    for w in all_windows:
        by_run.setdefault((w["source"], w["gear"]), []).append(w)

    if not by_run:
        out.append("_没有解析到任何 RATE 窗口。检查落盘是否生效（hilog 里的 `RUNLOG path`）。_\n")
    for (src, g) in sorted(by_run, key=lambda k: (k[0], k[1])):
        ws = by_run[(src, g)]
        name = "未知（旧日志无 gear= 字段）" if g < 0 else GEAR_NAMES.get(g, f"gear {g}")
        bucketed = [w for w in ws if w["bucketed"]]
        out.append(f"### {src} · gear={g} · {name}\n")
        out.append(f"- 稳态窗口：**{len(ws)}**")
        arr = [w["arrive"] for w in ws]
        done = [w["done"] for w in ws]
        out.append(f"- 到达 fps：p50 **{st.median(arr):.2f}**"
                   f"（{min(arr):.2f}–{max(arr):.2f}）")
        out.append(f"- 完成 fps：p50 **{st.median(done):.2f}**"
                   f"（{min(done):.2f}–{max(done):.2f}）")
        out.append(f"- 丢帧合计：**{sum(w['dropped'] for w in ws)}**")
        thermals = sorted({w["thermal"] for w in ws if w["thermal"] >= 0})
        batts = [w["battC"] for w in ws if w["battC"] >= 0]
        if thermals:
            out.append(f"- 热档：{thermals}（跨轮比较只允许同热档内做）")
        if batts:
            out.append(f"- 电池温度：{min(batts):.1f}–{max(batts):.1f} ℃")

        if not bucketed:
            out.append("\n  ⚠️ 本档窗口来自**旧格式**日志（无分桶字段）——"
                       "下面的分桶结论不适用，只能看到达/完成率。\n")
            out.append("")
            continue

        hit_n = sum(w["hit_n"] for w in ws)
        empty_n = sum(w["empty_n"] for w in ws)
        out.append(f"- 样本：检出 **{hit_n}** 帧 / 未检出 **{empty_n}** 帧\n")
        out.append("| 桶 | n | prep p50 (ms) | 往返 p50 (ms) | frame p50 (ms) | 余量 vs 33.33 ms |")
        out.append("|---|---|---|---|---|---|")

        # 未检出桶：conv 与 infer 都有分桶 p50；frame 只有 STAGE 行能给单帧值，
        # 所以这里用 conv+infer 作为该桶的单帧估计，并明确标注是「估计」。
        if empty_n > 0:
            ce = st.median([w["prep_empty_p50"] for w in ws if w["prep_empty_p50"] >= 0]) \
                if any(w["prep_empty_p50"] >= 0 for w in ws) else -1
            ie = st.median([w["rt_empty_p50"] for w in ws if w["rt_empty_p50"] >= 0]) \
                if any(w["rt_empty_p50"] >= 0 for w in ws) else -1
            # infer = 0 表示本档不推理（基准档），此时 frame 无意义 —— 不报。
            fe = (ce + ie) if (ce >= 0 and ie > 0) else -1
            margin = (FRAME_BUDGET_MS - fe) if fe >= 0 else -1
            out.append(f"| 未检出 | {empty_n} | {fmt(ce)} | {fmt_infer(ie)} | {fmt(fe)} | "
                       f"{fmt_margin(margin)} |")
        if hit_n > 0:
            ch = st.median([w["prep_hit_p50"] for w in ws if w["prep_hit_p50"] >= 0]) \
                if any(w["prep_hit_p50"] >= 0 for w in ws) else -1
            ih = st.median([w["rt_hit_p50"] for w in ws if w["rt_hit_p50"] >= 0]) \
                if any(w["rt_hit_p50"] >= 0 for w in ws) else -1
            fh = (ch + ih) if (ch >= 0 and ih > 0) else -1
            margin = (FRAME_BUDGET_MS - fh) if fh >= 0 else -1
            flag = "" if hit_n >= MIN_HIT_SAMPLES else " ⚠️样本不足"
            out.append(f"| 检出{flag} | {hit_n} | {fmt(ch)} | {fmt_infer(ih)} | {fmt(fh)} | "
                       f"{fmt_margin(margin)} |")
        out.append("")

    # 余量表：只对**新格式（有分桶）**的行给结论。旧格式的到达/完成率另列，
    # 因为它们无法回答「余量」这个问题（没有分桶的 frame p50）。
    out.append("## 余量表（生产档该看这张）\n")
    out.append("每行 = 一轮 × 一个档位。跨轮不合并：热态不同就不可比。\n")
    out.append("| 来源 | gear | 到达 fps | 完成 fps | 未检出 frame p50 | 检出 frame p50 | 检出余量 | 热档 | 判定 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (src, g) in sorted(by_run, key=lambda k: (k[0], k[1])):
        ws = by_run[(src, g)]
        if not any(w["bucketed"] for w in ws):
            continue
        arr = st.median([w["arrive"] for w in ws])
        done = st.median([w["done"] for w in ws])
        thermals = sorted({w["thermal"] for w in ws if w["thermal"] >= 0})
        thermal_s = ",".join(str(t) for t in thermals) if thermals else "-"

        def bucket_frame(key_c, key_i, n_key):
            if sum(w[n_key] for w in ws) < MIN_HIT_SAMPLES:
                return -1
            cs = [w[key_c] for w in ws if w[key_c] >= 0]
            is_ = [w[key_i] for w in ws if w[key_i] > 0]
            if not cs or not is_:
                return -1
            return st.median(cs) + st.median(is_)

        fe = bucket_frame("prep_empty_p50", "rt_empty_p50", "empty_n")
        fh = bucket_frame("prep_hit_p50", "rt_hit_p50", "hit_n")
        # margin 只在**确实算出了 frame** 时才有值。用 None 表示「没有余量可报」，
        # 不能借 -1 当哨兵 —— -1 在 fmt_margin 里是个合法的负余量（超预算 1 ms），
        # 会把「样本不足」渲染成「超预算 1 ms」，正好把结论说反。
        margin = (FRAME_BUDGET_MS - fh) if fh >= 0 else None
        if fh < 0:
            verdict = "检出桶样本不足"
        elif margin > 5:
            verdict = "有余量（相机是瓶颈）"
        elif margin > 0:
            verdict = "余量薄（<5 ms，易掉帧）"
        else:
            verdict = "**吃满/超预算**（我们在拖相机）"
        out.append(f"| {src} | {g} | {arr:.2f} | {done:.2f} | {fmt(fe)} | {fmt(fh)} | "
                   f"{fmt_margin(margin)} | {thermal_s} | {verdict} |")
    out.append("")

    # STAGE 明细：检出/未检出的分段
    all_stages = [s for _, _, _, sts, _, _ in per_file for s in sts]
    if all_stages:
        out.append("## STAGE 抽样行的分段（native 侧）\n")
        for label, sel in (("检出", [s for s in all_stages if s["count"] > 0]),
                           ("未检出", [s for s in all_stages if s["count"] == 0])):
            if not sel:
                out.append(f"- **{label}**：无样本")
                continue
            out.append(f"- **{label}**（n={len(sel)}）："
                       f"conv p50 {st.median([s['conv'] for s in sel]):.2f} ms · "
                       f"infer p50 {st.median([s['infer'] for s in sel]):.2f} ms · "
                       f"infer 范围 {min(s['infer'] for s in sel):.2f}–"
                       f"{max(s['infer'] for s in sel):.2f} ms")
        out.append("")

    md_path = evidence / "camera_summary.md"
    md_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    print("\n".join(out))
    print(f"\n[ok] 逐窗口 CSV -> {csv_path}")
    print(f"[ok] 汇总       -> {md_path}")

    # 非零退出表示证据不完整，便于脚本化把关
    incomplete = []
    for p, header, windows, stages, switches, settled in per_file:
        if len(windows) < 20:
            incomplete.append(f"{p.name}: 稳态窗口仅 {len(windows)} 个（期望 ≥20）")
        if not settled:
            incomplete.append(f"{p.name}: 未看到 SETTLE done")
        if windows and not any(w["bucketed"] for w in windows):
            incomplete.append(f"{p.name}: 旧格式日志（无分桶字段）")
    if incomplete:
        print("\n[warn] 证据不完整：")
        for s in incomplete:
            print(f"  - {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
