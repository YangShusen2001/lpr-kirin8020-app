#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI 到底放大了多少？—— 解释 ROI 为何无收益的核心指标。

## 为什么这是核心

"截 ROI"这个动作之所以被期待有收益，唯一的物理机制就是**放大车牌**：
把车牌从整图 letterbox 后的 70x26 px 放大到更接近车牌检测器的舒适尺度。

`why_roi_fails.py`（320，n=200）已经量到：车辆框**包含**车牌（cover 中位 1.000，
中心在内 97.6%），但 **IoU 仅 0.051** —— 车辆框远大于车牌框。

后果是几何上的：车框越大，裁出来再 resize 到 320 时的放大倍数越小。
这个脚本把倍数算出来：

    zoom = r_roi / r_direct
    r_direct = min(320/图高, 320/图宽)          # 直检：整图 letterbox
    r_roi    = min(320/ROI高, 320/ROI宽)        # ROI：裁剪后 letterbox

zoom ≈ 1.0 意味着 ROI 与直检等价（白做一次检测）；
zoom < 1.0 意味着 ROI 反而把车牌缩得更小（纯亏）。

同时给出 ROI 相对整图**少掉了多少像素面积**（信息损失的来源）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default=os.path.join(HERE, "why_roi_fails_n200_320.json"))
    ap.add_argument("--img-w", type=int, default=720)
    ap.add_argument("--img-h", type=int, default=1160)
    ap.add_argument("--net", type=int, default=320)
    ap.add_argument("--out", default=os.path.join(HERE, "roi_zoom.json"))
    a = ap.parse_args()

    d = json.load(open(a.in_json, encoding="utf-8"))
    rows = [r for r in d["rows"] if r["n_veh"] > 0 and r["veh_box"]]

    r_direct = min(a.net / a.img_w, a.net / a.img_h)
    img_area = a.img_w * a.img_h

    zooms, area_keep = [], []
    per_row = []
    for r in rows:
        x1, y1, x2, y2 = r["veh_box"]
        w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        r_roi = min(a.net / h, a.net / w)
        z = r_roi / r_direct
        zooms.append(z)
        area_keep.append((w * h) / img_area)
        per_row.append({"file": r["file"], "veh_box": r["veh_box"],
                        "zoom_vs_direct": round(z, 3),
                        "roi_area_frac_of_image": round((w * h) / img_area, 3)})
    z = np.asarray(zooms)

    summary = {
        "n": len(rows),
        "net_input": a.net,
        "img_size": [a.img_w, a.img_h],
        "r_direct": round(r_direct, 5),
        "zoom_vs_direct": {
            "min": round(float(z.min()), 3),
            "p05": round(float(np.percentile(z, 5)), 3),
            "p25": round(float(np.percentile(z, 25)), 3),
            "median": round(float(np.percentile(z, 50)), 3),
            "p75": round(float(np.percentile(z, 75)), 3),
            "max": round(float(z.max()), 3),
            "mean": round(float(z.mean()), 3),
        },
        "roi_area_frac_of_image": {
            "median": round(float(np.percentile(area_keep, 50)), 3),
            "p95": round(float(np.percentile(area_keep, 95)), 3),
        },
        "n_zoom_ge_1_5": int((z >= 1.5).sum()),
        "n_zoom_le_1_1": int((z <= 1.1).sum()),
        "per_row": per_row,
    }
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    zz = summary["zoom_vs_direct"]
    print(f"n={len(rows)}  直检缩放 r={r_direct:.4f}（整图 -> {a.net}）")
    print(f"ROI 相对直检放大倍数：中位 {zz['median']}x  "
          f"p25 {zz['p25']}x  p75 {zz['p75']}x  最大 {zz['max']}x")
    print(f"  >=1.5x 的图 {summary['n_zoom_ge_1_5']} 张；<=1.1x（等于白做）{summary['n_zoom_le_1_1']} 张")
    print(f"ROI 占整图面积：中位 {summary['roi_area_frac_of_image']['median'] * 100:.0f}%  "
          f"p95 {summary['roi_area_frac_of_image']['p95'] * 100:.0f}%")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
