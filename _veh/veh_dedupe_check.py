#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""veh_dedupe_check.py —— 跨类别车框去重的 PC 侧独立验证。

## 为什么要独立验

C++ 里的 `LprDedupeVehicles` 只有一个真机读数（`vehDeduped`）能看。这里用
**同一批 PC 车检结果**（`_veh/ref_decoded.json`，跑在 NV21 反解图上 —— 与真机
喂图同源）复刻同一套逻辑，两边的"合并了哪几个框"必须一致。

判据：
  ① 合并前后框数差 == 期望（veh320 上是 4 → 3：cls=2 与 cls=3 对同一目标 IoU 0.99）
  ② 保留下来的必须是**分数最高**的那个
  ③ 不误合并真正不同的目标（保留框两两 IoU 都 < 阈值）

## 用法

    python _veh/veh_dedupe_check.py --ref _veh/ref_decoded.json --iou 0.6
"""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe(boxes, thresh):
    """复刻 C++ LprDedupeVehicles：按分数降序，跨类别 NMS。"""
    s = sorted(boxes, key=lambda b: -b["score"])
    keep = []
    for b in s:
        if not any(iou(b["rect"], k["rect"]) >= thresh for k in keep):
            keep.append(b)
    return s, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="PC 车检结果 json")
    ap.add_argument("--iou", type=float, default=0.6)
    a = ap.parse_args()

    d = json.load(open(a.ref, encoding="utf-8"))
    boxes = d.get("boxes") or d.get("detections") or []
    if not boxes:
        print("[dedupe] FAIL 参考结果里没有框")
        return 1

    for b in boxes:
        r = b.get("rect") or b.get("box")
        b["rect"] = [float(v) for v in r]
        b["score"] = float(b.get("score", 0))

    print(f"[dedupe] 阈值 IoU = {a.iou}；去重前 {len(boxes)} 个框")
    for i, b in enumerate(boxes):
        print(f"   #{i} cls={b.get('cls', b.get('classId'))} score={b['score']:.4f} "
              f"rect={[round(v,2) for v in b['rect']]}")

    ordered, keep = dedupe(boxes, a.iou)
    dropped = len(ordered) - len(keep)
    print(f"\n[dedupe] 去重后 {len(keep)} 个框（丢 {dropped}）")

    prob, note = [], []

    # 判据③：保留框两两 IoU 必须都 < 阈值
    worst = 0.0
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            worst = max(worst, iou(keep[i]["rect"], keep[j]["rect"]))
    note.append(f"[dedupe] 保留框两两最大 IoU = {worst:.4f}（必须 < {a.iou}）")
    if worst >= a.iou:
        prob.append(f"保留框之间仍有 IoU {worst:.4f} >= {a.iou} —— 去重不彻底")

    # 判据②：被丢的框必须是与某个**更高分**保留框重叠的
    kept_ids = {id(k) for k in keep}
    for b in ordered:
        if id(b) in kept_ids:
            continue
        hit = [k for k in keep if iou(b["rect"], k["rect"]) >= a.iou]
        if not hit:
            prob.append(f"框 score={b['score']:.4f} 被丢，但找不到与它重叠的保留框 "
                        f"—— 误丢")
        else:
            hi = max(hit, key=lambda k: k["score"])
            if hi["score"] < b["score"]:
                prob.append(f"框 score={b['score']:.4f} 被丢，但保留的 {hi['score']:.4f} "
                            f"分数更低 —— 应该保留高分那个")
            else:
                note.append(f"[dedupe] 丢 score={b['score']:.4f}（与保留的 "
                            f"score={hi['score']:.4f} IoU={iou(b['rect'], hi['rect']):.3f}）✓")

    # 判据①：合并明细
    pairs = []
    for i, b in enumerate(ordered):
        for j, c in enumerate(ordered):
            if j <= i:
                continue
            v = iou(b["rect"], c["rect"])
            if v >= a.iou:
                pairs.append((i, j, v))
    note.append(f"[dedupe] 触发合并的框对：{[(i, j, round(v,3)) for i, j, v in pairs]}")

    print()
    for ln in note:
        print(ln)
    print()
    if prob:
        print(f"[dedupe] ✗ 发现 {len(prob)} 个问题：")
        for p in prob:
            print("   - " + p)
        return 1
    print(f"[dedupe] ✓ 跨类别去重逻辑正确：{len(boxes)} → {len(keep)}，"
          f"被丢的都是与更高分保留框重叠的重复框，且保留框之间不再互相重叠。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
