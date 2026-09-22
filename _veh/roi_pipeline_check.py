#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4 的判据脚本：去重自证 + ROI 路径一致性 + CCPD 召回。

三件事，对应 T4 票面的三条验收：

1. **合并去重生效**（`--dedupe-log`）
   设备侧 `LprDedupeSelfTest()` 构造了完全重合 / IoU 0.667 / IoU 0.333 / 链式重叠 /
   边界相接 / 空输入等已知输入，逐 case 断言"保留几条、保留的是哪一条"。
   这里要求**全部 ok=1**（设备自报 failed=0 也要对上）。

2. **ROI 路径端到端可跑且自洽**（`--pipe-log`）
   - `rawHits - count == dedupeDropped`（去重账要平：丢掉的数必须等于原始数减结果数）
   - `roiTried + roiSkipped == vehCount`（每个车框要么跑了要么被跳过，不许凭空少）
   - 每条车牌的 `owner` 必须落在 `[0, vehCount)` —— 归属连不出去就是界面画不出线
   - `colour` 只能取 blue/green/yellow/unknown（ADR-0005 的像素测量口径）
   - 检出车牌时 `code` 不能为空

3. **CCPD 召回**（`--recall-json`）
   票面要求 91.0% ± 1pp 且不得低于 90%。数据来自 `_veh/roi_vs_direct.py`
   （复用 `hlpr_reference.py` 的权威后处理口径，与 native 是同一份依据）。

用法：
    python _veh/roi_pipeline_check.py --dedupe-log _veh/devlog_T4DEDUPE.txt \
        --pipe-log _veh/devlog_T4PIPE.txt --recall-json _veh/roi_vs_direct_t4.json
"""
from __future__ import annotations

import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

COLOURS = {"blue", "green", "yellow", "unknown"}
RECALL_TARGET = 0.91
RECALL_TOL = 0.01


def parse_cases(path, tag):
    """解析 `case=<名>;ok=0/1;...` 形式的自证报告。"""
    cases, meta = {}, {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if tag not in line:
                continue
            body = line.split(tag, 1)[1].strip()
            parts = [p for p in body.split(";") if p]
            if not parts:
                continue
            if parts[0].startswith("case="):
                kv = {}
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        kv[k] = v
                cases[parts[0][len("case="):]] = kv
            else:
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        meta[k] = v
    return cases, meta


def parse_pipe(path):
    """解析 `T4PIPE <kv>` 与 `T4PLATE <kv>` / `T4VEH <kv>` 行；同名取最后一次（ArkTS 回显更全）。"""
    pipe, plates, vehs = None, {}, {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "T4PIPE" not in line and "T4PLATE" not in line and "T4VEH" not in line:
                continue
            body = line.split("LprNative", 1)[-1]
            for marker in ("T4PIPE ", "T4PLATE ", "T4VEH "):
                if marker in body:
                    body = body.split(marker, 1)[1].strip()
                    break
            kv = {}
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            # ⚠️ 同一个坑在 T3 的脚本里已经踩过一次，这里又踩了一遍：
            # hilog 行用的是**紧凑字段名**（`veh=` / `truncated=` / `dropped=`），
            # 而 NAPI 返回串用的是 `vehCount` / `vehTruncated` / `dedupeDropped`。
            # 只认一种就会 KeyError 或把字段读成 None。
            kv.setdefault("vehCount", kv.get("veh"))
            kv.setdefault("vehTruncated", kv.get("truncated"))
            kv.setdefault("dedupeDropped", kv.get("dropped"))
            # rect 的分隔符两处也不同：hilog 用逗号，返回串用竖线。统一成竖线再解析。
            if "rect" in kv and "|" not in kv["rect"] and kv["rect"].count(",") == 3:
                kv["rect"] = kv["rect"].replace(",", "|")
            if "T4PIPE " in line:
                if "count" in kv:
                    pipe = kv
            elif "T4PLATE " in line:
                if "idx" in kv:
                    plates[int(kv["idx"])] = kv
            elif "T4VEH " in line:
                if "idx" in kv:
                    vehs[int(kv["idx"])] = kv
    return pipe, plates, vehs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedupe-log")
    ap.add_argument("--pipe-log")
    ap.add_argument("--recall-json")
    a = ap.parse_args()

    prob = []

    # ---------- 1) 去重自证 ----------
    if a.dedupe_log:
        cases, meta = parse_cases(a.dedupe_log, "T4DEDUPE")
        print(f"[t4check] 去重自证 case 数 = {len(cases)}")
        n_ok = 0
        for name in sorted(cases):
            kv = cases[name]
            ok = kv.get("ok") == "1"
            n_ok += 1 if ok else 0
            print(f"[t4check] {'✓' if ok else '✗'} {name:<26} "
                  + "; ".join(f"{k}={v}" for k, v in kv.items() if k != "ok"))
            if not ok:
                prob.append(f"去重用例 {name} 断言失败：{kv}")
        print(f"[t4check] 去重自证 ok = {n_ok} / {len(cases)}；"
              f"设备自报 total={meta.get('total')} failed={meta.get('failed')}")
        if not cases:
            prob.append("去重自证没有任何 case 行")
        if meta.get("failed") not in (None, "0"):
            prob.append(f"去重自证设备自报 failed={meta.get('failed')}")
        if meta.get("total") and len(cases) != int(meta["total"]):
            prob.append(f"case 数与 total 对不上：{len(cases)} vs {meta['total']}")

    # ---------- 2) ROI 路径一致性 ----------
    if a.pipe_log:
        pipe, plates, vehs = parse_pipe(a.pipe_log)
        if pipe is None:
            prob.append("日志里没有 T4PIPE 汇总行")
        else:
            n_veh = int(pipe["vehCount"])
            n_tried = int(pipe["roiTried"])
            n_skip = int(pipe["roiSkipped"])
            raw = int(pipe["rawHits"])
            dropped = int(pipe["dedupeDropped"])
            count = int(pipe["count"])
            print(f"[t4check] ROI 路径：veh={n_veh} roiTried={n_tried} roiSkipped={n_skip} "
                  f"rawHits={raw} dropped={dropped} count={count}")
            print(f"[t4check] 耗时：vehInferMs={pipe.get('vehInferMs')} "
                  f"roiDetectMs={pipe.get('roiDetectMs')} totalMs={pipe.get('totalMs')}")
            if raw - count != dropped:
                prob.append(f"去重账不平：rawHits({raw}) - count({count}) = {raw - count} "
                            f"!= dedupeDropped({dropped})")
            if n_tried + n_skip != n_veh:
                prob.append(f"车框账不平：roiTried({n_tried}) + roiSkipped({n_skip}) "
                            f"!= vehCount({n_veh})")
            if len(plates) != count:
                prob.append(f"逐框行数({len(plates)}) 与 count({count}) 不符")
            if len(vehs) != n_veh:
                prob.append(f"车辆框行数({len(vehs)}) 与 vehCount({n_veh}) 不符")
            for i in sorted(plates):
                kv = plates[i]
                rect = kv.get("rect", "").split("|")
                if len(rect) != 4:
                    prob.append(f"车牌#{i} 的 rect 不是 4 个数：{kv.get('rect')}")
                    continue
                x1, y1, x2, y2 = (int(v) for v in rect)
                if not (x2 > x1 and y2 > y1):
                    prob.append(f"车牌#{i} 的框非法：{kv.get('rect')}")
                owner = int(kv.get("owner", -1))
                if not (0 <= owner < n_veh):
                    prob.append(f"车牌#{i} 的 owner={owner} 越界 [0,{n_veh})")
                if kv.get("colour") not in COLOURS:
                    prob.append(f"车牌#{i} 的颜色 '{kv.get('colour')}' 不在 {sorted(COLOURS)}")
                if kv.get("code") in (None, ""):
                    prob.append(f"车牌#{i} 检出了但没有车牌串（code 为空）")
                print(f"[t4check]   车牌#{i} rect={kv.get('rect')} score={kv.get('score')} "
                      f"owner={owner} colour={kv.get('colour')} code={kv.get('code')}")
            for i in sorted(vehs):
                print(f"[t4check]   车辆#{i} cls={vehs[i].get('cls')} "
                      f"score={vehs[i].get('score')} name={vehs[i].get('name')} "
                      f"rect={vehs[i].get('rect')}")

    # ---------- 3) CCPD 召回 ----------
    if a.recall_json:
        d = json.load(open(a.recall_json, encoding="utf-8"))
        cfg = d.get("config", {})
        print(f"[t4check] 召回测量配置：veh_conf={cfg.get('veh_conf')} "
              f"roi_max_boxes={cfg.get('roi_max_boxes')} n={cfg.get('n')}")
        for key, v in sorted(d.get("by_config", {}).items()):
            if not isinstance(v, dict):
                continue
            top1 = v.get("recall_top1", {})
            any_ = v.get("recall_any", {})
            print(f"[t4check] 配置 {key}：ROI top1={top1.get('roi')} any={any_.get('roi')} "
                  f"（直检 top1={top1.get('direct')}）；无车辆率="
                  f"{v.get('images_without_vehicle', {}).get('rate')}")
            # 票面：91.0% ± 1pp，且不得低于 90%
            for label, val in (("top1", top1.get("roi")), ("any", any_.get("roi"))):
                if val is None:
                    continue
                if val < 0.90:
                    prob.append(f"{key} ROI {label} 召回 {val} < 0.90（票面下限）")
                elif abs(val - RECALL_TARGET) > RECALL_TOL:
                    prob.append(f"{key} ROI {label} 召回 {val} 偏离 {RECALL_TARGET} 超过 "
                                f"{RECALL_TOL}（票面要求 ±1pp）")

    print()
    if prob:
        for p in prob:
            print(f"[t4check] ✗ {p}")
        return 2
    print("[t4check] ✓ T4 判据全过：去重自证 / ROI 路径账目与归属 / CCPD 召回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
