#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补测被漏掉的那一维：ROI 对**误检/冗余框**的影响（不只是召回）。

## 为什么补

`roi_vs_direct.py` 全部指标都是「召回」（与 GT 的 IoU≥0.5 命中与否），
**完全没有度量误检**。而 CCPD 这个数据集本身干扰极少（一图一车一牌、背景干净），
所以它天然无法体现「缩小搜索区域 → 减少误检」这个 ROI 最可能的价值。

用户指出真实场景是「摄像头实时 + 干扰多」，正好打在这个缺口上。
本脚本从**已有的 JSON** 里把误检相关的量算出来（零额外推理）：

  - 每图检出框数分布（0 / 1 / 2 / 3+）—— 一图单牌，所以 >1 即为冗余框
  - 「干净命中」= 恰好出 1 个框且命中 GT
  - 「命中但带冗余」= 命中 GT 的同时还出了别的框
  - 冗余框上界 = 总框数 − 命中图数（下界，因为命中图可能同时含冗余框）

注意：JSON 只记录了框数与命中与否，没有逐框的匹配标记，
所以冗余数只能给**区间**，不能给精确值。区间足以判断方向。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def dist(counts):
    d = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for c in counts:
        if c == 0:
            d["0"] += 1
        elif c == 1:
            d["1"] += 1
        elif c == 2:
            d["2"] += 1
        else:
            d["3+"] += 1
    return d


def describe(name, counts, hits):
    n = len(counts)
    total_boxes = sum(counts)
    hit_n = sum(hits)
    dd = dist(counts)
    return {
        "path": name,
        "n_images": n,
        "total_boxes": total_boxes,
        "boxes_per_image_mean": round(total_boxes / n, 3) if n else 0,
        "zero_detection": dd["0"],
        "zero_rate": round(dd["0"] / n, 4) if n else 0,
        "hits": hit_n,
        "recall": round(hit_n / n, 4) if n else 0,
        "box_count_histogram": dd,
        "multi_box_rate": round((dd["2"] + dd["3+"]) / n, 4) if n else 0,
        # 冗余框（一图只有一块牌，多出来的框即非真检）
        "redundant_boxes": total_boxes - hit_n,
        "redundant_per_hit": round((total_boxes - hit_n) / hit_n, 3) if hit_n else None,
        "clean_hit_rate": None,   # 下面补（需要每图的框数与命中配对）
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default=os.path.join(HERE, "roi_vs_direct_full.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "fp_analysis.json"))
    a = ap.parse_args()

    d = json.load(open(a.in_json, encoding="utf-8"))
    rows = d["rows"]

    out = {"source": a.in_json, "n": len(rows), "configs": {}}

    a_counts = [r["A_n"] for r in rows]
    a_hits = [r["A_any"] for r in rows]
    out["configs"]["direct_full_image"] = describe("direct", a_counts, a_hits)
    clean = sum(1 for r in rows if r["A_n"] == 1 and r["A_any"])
    out["configs"]["direct_full_image"]["clean_hit_rate"] = round(clean / len(rows), 4)
    out["configs"]["direct_full_image"]["clean_hits"] = clean

    for key, cfg in d["by_config"].items():
        b_counts = [r["B"][key]["n_det"] for r in rows]
        b_hits = [r["B"][key]["any"] for r in rows]
        s = describe(key, b_counts, b_hits)
        s["veh_imgsz"] = cfg["veh_imgsz"]
        s["roi_pad"] = cfg["roi_pad"]
        clean_b = sum(1 for r in rows if r["B"][key]["n_det"] == 1 and r["B"][key]["any"])
        s["clean_hit_rate"] = round(clean_b / len(rows), 4)
        s["clean_hits"] = clean_b
        # ROI 路的总检出框数应把「车辆漏检导致 ROI 未跑」的图算进去（那些贡献 0 框）
        out["configs"][f"roi_{key}"] = s

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"n={out['n']}  （一图一牌，故框数 >1 = 冗余框）\n")
    hdr = f"  {'配置':<24} {'召回':>7} {'框/图':>7} {'零检':>7} {'多框率':>7} {'冗余框/命中':>12} {'干净命中率':>10}"
    print(hdr)
    for name, s in out["configs"].items():
        print(f"  {name:<24} {s['recall']:>7.4f} {s['boxes_per_image_mean']:>7.3f} "
              f"{s['zero_rate']:>7.4f} {s['multi_box_rate']:>7.4f} "
              f"{(s['redundant_per_hit'] if s['redundant_per_hit'] is not None else float('nan')):>12.3f} "
              f"{s['clean_hit_rate']:>10.4f}")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
