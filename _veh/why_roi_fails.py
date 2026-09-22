#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为什么 ROI 会掉召回：把「车辆框没圈住车牌」这一几何原因量化。

`roi_vs_direct.py` 已经证明 ROI 掉召回；这个脚本回答**为什么** —— 不是靠推断，
而是直接量每一张图上「最高分车辆框」与「真值车牌框」的几何关系。

三个指标：
  cover  = 车牌框被车辆框覆盖的面积比例   —— 车牌到底在不在车框里
  cen_in = 车牌框中心是否落在车辆框内      —— 最宽松的"包含"判据
  iou    = 两框 IoU

若 cover / cen_in 本身就低，那就说明**通用车辆检测器框的不是整车**，
ROI 这一刀从根上切不出车牌 —— 与 ROI 里换成什么车牌检测器无关，
再怎么调 RO I内部也救不回来。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_roi import VEHICLE_CLASSES, decode_yolov5, letterbox, parse_plate_bbox  # noqa: E402

DEFAULT_IMG_DIR = os.path.join(
    HERE, "..", "LprDemo", "entry", "src", "main", "resources", "rawfile", "assets", "ccpd")


def inter_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def read_img(path):
    import cv2
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def run(args):
    img_dir = os.path.abspath(args.img_dir)
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    if args.limit:
        step = max(1, len(names) // args.limit)
        names = names[::step][:args.limit]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.veh_onnx, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rows = []
    for n in names:
        img = read_img(os.path.join(img_dir, n))
        gt = parse_plate_bbox(n)
        if img is None or gt is None:
            continue
        lb, r, px, py = letterbox(img, args.imgsz)
        blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        vd = sess.run([out_name], {in_name: blob})[0]
        veh = [d for d in decode_yolov5(vd, r, px, py, args.veh_conf, args.iou)
               if d["cls"] in VEHICLE_CLASSES]

        if not veh:
            rows.append({"file": n, "gt_box": list(gt), "n_veh": 0,
                         "cover": None, "cen_in": None, "iou": None,
                         "veh_box": None, "cls": None, "conf": None})
            continue

        top = max(veh, key=lambda d: d["conf"])
        vb = top["xyxy"]
        ia = inter_area(vb, gt)
        cover = ia / area(gt) if area(gt) > 0 else 0.0
        iou = ia / (area(vb) + area(gt) - ia) if (area(vb) + area(gt) - ia) > 0 else 0.0
        cx, cy = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
        cen_in = (vb[0] <= cx <= vb[2]) and (vb[1] <= cy <= vb[3])
        rows.append({"file": n, "gt_box": list(gt), "n_veh": len(veh),
                     "cover": round(float(cover), 4), "cen_in": bool(cen_in),
                     "iou": round(float(iou), 4), "veh_box": [round(v, 1) for v in vb],
                     "cls": top["name"], "conf": top["conf"]})

    with_veh = [r for r in rows if r["n_veh"] > 0]
    covers = np.array([r["cover"] for r in with_veh], dtype=np.float64)
    ious = np.array([r["iou"] for r in with_veh], dtype=np.float64)

    def pct(v, q):
        return round(float(np.percentile(v, q)), 4) if v.size else None

    summary = {
        "config": {"img_dir": img_dir, "n": len(rows), "veh_onnx": args.veh_onnx,
                   "imgsz": args.imgsz, "veh_conf": args.veh_conf},
        "n_with_vehicle": len(with_veh),
        "plate_covered_by_vehicle_box": {
            "median": pct(covers, 50), "p05": pct(covers, 5), "p25": pct(covers, 25),
            "mean": round(float(covers.mean()), 4) if covers.size else None,
        },
        "plate_center_inside_vehicle_box": {
            "n": int(sum(r["cen_in"] for r in with_veh)),
            "rate": round(sum(r["cen_in"] for r in with_veh) / len(with_veh), 4) if with_veh else None,
        },
        "iou_plate_vs_vehicle_box": {"median": pct(ious, 50),
                                     "mean": round(float(ious.mean()), 4) if ious.size else None},
        # 覆盖度分档：这是"ROI 能不能切到车牌"的直接判据
        "cover_histogram": {
            "0.00-0.25": int(((covers >= 0) & (covers < 0.25)).sum()),
            "0.25-0.50": int(((covers >= 0.25) & (covers < 0.50)).sum()),
            "0.50-0.75": int(((covers >= 0.50) & (covers < 0.75)).sum()),
            "0.75-1.00": int(((covers >= 0.75) & (covers <= 1.0)).sum()),
        },
        "examples_low_cover": sorted(with_veh, key=lambda r: r["cover"])[:8],
        "rows": rows,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--veh-onnx", default=os.path.join(HERE, "yolov5su_640.onnx"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--veh-conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "why_roi_fails.json"))
    a = ap.parse_args()

    s = run(a)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)

    c = s["plate_covered_by_vehicle_box"]
    print(f"n={s['config']['n']}  有车辆框 {s['n_with_vehicle']}")
    print(f"  车牌被车辆框覆盖：中位 {c['median']:.3f}  p25 {c['p25']:.3f}  p05 {c['p05']:.3f}")
    print(f"  车牌中心落在车辆框内：{s['plate_center_inside_vehicle_box']['n']}"
          f"/{s['n_with_vehicle']} "
          f"({s['plate_center_inside_vehicle_box']['rate'] * 100:.1f}%)")
    print(f"  两框 IoU 中位 {s['iou_plate_vs_vehicle_box']['median']:.3f}")
    print(f"  覆盖度分布 {s['cover_histogram']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
