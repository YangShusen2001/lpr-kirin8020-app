#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI 预裁剪 vs 整图直检：车牌检测召回 A/B（CCPD2019 整图）。

## 为什么做这个

用户提的流水线是「车辆检测 -> 截取 ROI -> 车牌检测 -> 识别+颜色」。
其中「截 ROI」这一步**默认被当成收益**，但没人验证过。这个脚本就是来证实/证伪的。

关键前提（`probe_roi.py` 实测，见 probe_roi.json）：
  * CCPD 整图 720x1160，车牌中位 253x95px，占图 2.92%；
  * letterbox 到 320 后车牌仍有 70x26px —— **不是小目标**；
  * 通用车辆检测器（YOLOv5su/COCO）在整图上约 86.7% 的图能框出车。

第 3 条意味着 ROI 路存在**串行瓶颈**：车辆没框到 -> ROI 里没车牌 -> 召回直接归零。

## 公平性设计（重要）

只测一档配置就宣布「ROI 不行」是不诚实的 —— 可能是我们没调好。所以本脚本
**一次扫多组配置**，给 ROI 路所有合理的帮助：
  * 车辆检测输入 320 vs 640（640 对小车辆友好得多）
  * ROI 外扩 0.15 / 0.40（外扩越大越可能把车牌包进来）
每张图只做一次车辆检测（按 imgsz 去重），避免重复开销。

## 两路怎么比

  A 直检   整车图 -> 车牌检测器（与手机端同一模型/同一后处理）
  B ROI    整车图 -> 车辆检测 -> 取最高分车辆框外扩 p% -> 裁 -> 车牌检测 -> 框映射回原图

命中判据：与文件名真值框的 IoU >= 0.5。同时报 top1（只认最高分框）与 any（任一框），
避免"靠多框蒙中"掩盖问题。

## 与手机端一致性

车牌检测的 letterbox / 后处理 / 角点还原**直接复用** `Desktop/Test/lpr-showcase/tools/
hlpr_reference.py`（native C++ 的逐行对照实现），不另写一套 —— 否则测出来的基线与
设备上的行为没有可比性。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
HLPR_TOOLS = r"C:\Users\26671\Desktop\Test\lpr-showcase\tools"
sys.path.insert(0, HLPR_TOOLS)
sys.path.insert(0, HERE)

import hlpr_reference as H          # noqa: E402  车牌检测的权威口径
from probe_roi import (             # noqa: E402
    VEHICLE_CLASSES, decode_yolov5, letterbox, parse_plate_bbox,
)

DEFAULT_IMG_DIR = os.path.join(
    HERE, "..", "LprDemo", "entry", "src", "main", "resources", "rawfile", "assets", "ccpd")
DEFAULT_PLATE_ONNX = r"C:\Users\26671\Desktop\车牌识别\_scratch\ms\y5fu_320x_sim.onnx"


# --------------------------------------------------------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def read_img(path):
    buf = np.fromfile(path, dtype=np.uint8)   # 非 ASCII 路径下 cv2.imread 会失败
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def expand_box(box, img_w, img_h, pad):
    """按框尺寸的比例外扩，并夹到图内。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad)),
            min(img_w, int(x2 + w * pad)), min(img_h, int(y2 + h * pad)))


def best_iou(boxes, gt):
    """返回 (最大IoU, 命中top1?, 命中any?)。boxes 需按分数降序。"""
    if len(boxes) == 0:
        return 0.0, False, False
    ious = [iou_xyxy(b, gt) for b in boxes]
    return max(ious), ious[0] >= 0.5, any(v >= 0.5 for v in ious)


def parse_veh_models(spec):
    """`320=path/a.onnx,640=path/b.onnx` -> [(320, abs_path), ...]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        imgsz, path = part.split("=", 1)
        out.append((int(imgsz), os.path.abspath(path.strip())))
    return out


# --------------------------------------------------------------------------
def run(args):
    img_dir = os.path.abspath(args.img_dir)
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    if args.limit:
        step = max(1, len(names) // args.limit)
        names = names[::step][:args.limit]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    det = ort.InferenceSession(args.plate_onnx, so, providers=["CPUExecutionProvider"])

    veh_models = parse_veh_models(args.veh_models)
    veh_sess = {}
    for imgsz, path in veh_models:
        if os.path.isfile(path):
            veh_sess[imgsz] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        else:
            print(f"[warn] 车辆模型不存在，跳过 imgsz={imgsz}: {path}", file=sys.stderr)

    pads = [float(p) for p in args.pads.split(",")]
    configs = [(imgsz, pad) for imgsz in sorted(veh_sess) for pad in pads]
    if not configs:
        print("[fail] 没有任何可用的车辆模型配置", file=sys.stderr)
        return 2

    rows = []
    t0 = time.perf_counter()
    for k, n in enumerate(names):
        img = read_img(os.path.join(img_dir, n))
        gt = parse_plate_bbox(n)
        if img is None or gt is None:
            continue
        ih, iw = img.shape[:2]

        # ---- A：整图直检 -------------------------------------------------
        tA = time.perf_counter()
        A = H.detect(det, img, args.conf, args.iou)
        tA = (time.perf_counter() - tA) * 1000
        A_sorted = A[np.argsort(-A[:, 4])] if len(A) else A
        a_iou, a_top1, a_any = best_iou([r[:4].tolist() for r in A_sorted], gt)

        # ---- B：车辆检测 -> ROI -> 车牌检测 -------------------------------
        row = {"file": n, "gt_box": list(gt), "gt_plate_px": [gt[2] - gt[0], gt[3] - gt[1]],
               "A_n": int(len(A)), "A_iou": round(a_iou, 4),
               "A_top1": a_top1, "A_any": a_any, "tA_ms": round(tA, 1), "B": {}}
        tB_all = 0.0
        for imgsz, sess in veh_sess.items():
            tb = time.perf_counter()
            lb, r, px, py = letterbox(img, imgsz)
            blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            vd = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: blob})[0]
            vdets = [d for d in decode_yolov5(vd, r, px, py, args.veh_conf, args.iou)
                     if d["cls"] in VEHICLE_CLASSES]
            t_veh = (time.perf_counter() - tb) * 1000

            # 取哪些车辆框进 ROI。真实场景一帧多车，**只取 top1 是错的实现** ——
            # 车牌可能落在非最高分的车框里。`--roi-max-boxes=-1` 遍历全部。
            if vdets:
                tops = sorted(vdets, key=lambda d: -d["conf"])
                if args.roi_max_boxes > 0:
                    tops = tops[: args.roi_max_boxes]
            else:
                tops = []

            for pad in pads:
                key = f"{imgsz}|{pad}"
                tC = time.perf_counter()
                if not tops:
                    row["B"][key] = {"n_veh": 0, "iou": 0.0, "top1": False, "any": False,
                                     "veh_conf": 0.0, "roi": None, "n_det": 0, "ms": 0.0,
                                     "n_roi_used": 0}
                    continue

                scored = []          # (score, box_global)
                n_det_total = 0
                roi_first = None
                n_roi_used = 0
                for top in tops:
                    roi = expand_box(top["xyxy"], iw, ih, pad)
                    rx1, ry1, rx2, ry2 = roi
                    if roi_first is None:
                        roi_first = roi
                    if rx2 - rx1 <= 8 or ry2 - ry1 <= 8:
                        continue
                    n_roi_used += 1
                    B = H.detect(det, img[ry1:ry2, rx1:rx2], args.conf, args.iou)
                    n_det_total += int(len(B))
                    B_sorted = B[np.argsort(-B[:, 4])] if len(B) else B
                    for r0 in B_sorted:
                        scored.append((float(r0[4]),
                                       [rx1 + r0[0], ry1 + r0[1], rx1 + r0[2], ry1 + r0[3]]))

                if scored:
                    scored.sort(key=lambda x: -x[0])
                    bi, b1, ba = best_iou([b for _, b in scored], gt)
                else:
                    bi, b1, ba = 0.0, False, False

                row["B"][key] = {
                    "n_veh": len(vdets), "iou": round(bi, 4), "top1": b1, "any": ba,
                    "veh_conf": tops[0]["conf"], "roi": list(roi_first),
                    "n_det": n_det_total, "n_roi_used": n_roi_used,
                    "ms": round(t_veh + (time.perf_counter() - tC) * 1000, 1),
                }
                tB_all += t_veh + (time.perf_counter() - tC) * 1000
        row["tB_ms"] = round(tB_all / max(1, len(configs)), 1)
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  ... {k + 1}/{len(names)}", flush=True)

    # ---- 汇总 --------------------------------------------------------------
    n = len(rows)
    bins = [(0, 150), (150, 220), (220, 300), (300, 10 ** 9)]
    by_config = {}
    for imgsz, pad in configs:
        key = f"{imgsz}|{pad}"
        a_t1 = sum(r["A_top1"] for r in rows) / n
        b_t1 = sum(r["B"][key]["top1"] for r in rows) / n
        a_an = sum(r["A_any"] for r in rows) / n
        b_an = sum(r["B"][key]["any"] for r in rows) / n
        sub_bins = {}
        for lo, hi in bins:
            sub = [r for r in rows if lo <= r["gt_plate_px"][0] < hi]
            if not sub:
                continue
            sub_bins[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = {
                "n": len(sub),
                "A_recall_top1": round(sum(r["A_top1"] for r in sub) / len(sub), 4),
                "B_recall_top1": round(sum(r["B"][key]["top1"] for r in sub) / len(sub), 4),
                "A_mean_iou": round(float(np.mean([r["A_iou"] for r in sub])), 4),
                "B_mean_iou": round(float(np.mean([r["B"][key]["iou"] for r in sub])), 4),
            }
        by_config[key] = {
            "veh_imgsz": imgsz, "roi_pad": pad,
            "recall_top1": {"direct": round(a_t1, 4), "roi": round(b_t1, 4),
                            "delta": round(b_t1 - a_t1, 4)},
            "recall_any": {"direct": round(a_an, 4), "roi": round(b_an, 4),
                           "delta": round(b_an - a_an, 4)},
            "mean_iou": {"direct": round(float(np.mean([r["A_iou"] for r in rows])), 4),
                         "roi": round(float(np.mean([r["B"][key]["iou"] for r in rows])), 4)},
            "images_without_vehicle": {
                "n": sum(1 for r in rows if r["B"][key]["n_veh"] == 0),
                "rate": round(sum(1 for r in rows if r["B"][key]["n_veh"] == 0) / n, 4)},
            "empty_roi_detect": sum(1 for r in rows if r["B"][key]["n_det"] == 0),
            "veh_boxes_per_image_mean": round(
                float(np.mean([r["B"][key]["n_veh"] for r in rows])), 3),
            "roi_used_per_image_mean": round(
                float(np.mean([r["B"][key].get("n_roi_used", 0) for r in rows])), 3),
            "plate_dets_per_image_mean": round(
                float(np.mean([r["B"][key]["n_det"] for r in rows])), 3),
            "detect_ms": {"direct": round(float(np.mean([r["tA_ms"] for r in rows])), 1),
                          "roi_total": round(float(np.mean([r["B"][key]["ms"] for r in rows])), 1)},
            "recall_by_plate_width_px": sub_bins,
        }

    summary = {
        "config": {"img_dir": img_dir, "n": n, "limit": args.limit,
                   "plate_onnx": args.plate_onnx, "veh_models": args.veh_models,
                   "conf": args.conf, "iou": args.iou, "veh_conf": args.veh_conf,
                   "roi_max_boxes": args.roi_max_boxes, "pads": pads},
        "by_config": by_config,
        "wall_s": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--plate-onnx", default=DEFAULT_PLATE_ONNX)
    ap.add_argument("--veh-models", default=f"320={os.path.join(HERE, 'yolov5su.onnx')}",
                    help="逗号分隔的 `imgsz=onnx路径`")
    ap.add_argument("--limit", type=int, default=0, help="0=全部 1000 张")
    ap.add_argument("--conf", type=float, default=0.25, help="车牌检测置信度")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--veh-conf", type=float, default=0.25)
    ap.add_argument("--roi-max-boxes", type=int, default=1,
                    help="每个尺度取前 N 个车辆框做 ROI；-1 = 遍历全部"
                         "（真实一帧多车场景必须用 -1，只取 top1 是错的实现）")
    ap.add_argument("--pads", default="0.15,0.4", help="逗号分隔的 ROI 外扩比例")
    ap.add_argument("--out", default=os.path.join(HERE, "roi_vs_direct.json"))
    a = ap.parse_args()

    if not os.path.isdir(a.img_dir):
        print(f"[fail] 图片目录不存在: {a.img_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(a.plate_onnx):
        print(f"[fail] 车牌模型不存在: {a.plate_onnx}", file=sys.stderr)
        return 2

    s = run(a)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)

    print(f"\nn={s['config']['n']}  用时 {s['wall_s']}s")
    for key, c in s["by_config"].items():
        print(f"\n[配置 车辆输入={c['veh_imgsz']} ROI外扩={c['roi_pad']}]")
        print(f"  召回(top1)  直检 {c['recall_top1']['direct']:.4f}  "
              f"ROI {c['recall_top1']['roi']:.4f}  delta {c['recall_top1']['delta']:+.4f}")
        print(f"  召回(any)   直检 {c['recall_any']['direct']:.4f}  "
              f"ROI {c['recall_any']['roi']:.4f}  delta {c['recall_any']['delta']:+.4f}")
        print(f"  平均IoU     直检 {c['mean_iou']['direct']:.4f}  ROI {c['mean_iou']['roi']:.4f}")
        print(f"  无车辆图 {c['images_without_vehicle']['n']} "
              f"({c['images_without_vehicle']['rate'] * 100:.1f}%)   "
              f"ROI内零检出 {c['empty_roi_detect']}")
        print(f"  车框/图 {c['veh_boxes_per_image_mean']}   "
              f"实际用ROI/图 {c['roi_used_per_image_mean']}   "
              f"牌检出/图 {c['plate_dets_per_image_mean']}")
        print(f"  耗时  直检 {c['detect_ms']['direct']}ms  ROI全程 {c['detect_ms']['roi_total']}ms")
        for k2, v in c["recall_by_plate_width_px"].items():
            print(f"    牌宽 {k2:>10s}px n={v['n']:<4d} 直检 {v['A_recall_top1']:.3f} "
                  f"ROI {v['B_recall_top1']:.3f}  IoU {v['A_mean_iou']:.3f}/{v['B_mean_iou']:.3f}")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
