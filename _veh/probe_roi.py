#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI 预裁剪假设的先行探测（纯离线，不碰设备）。

在写任何 spec 之前必须先回答两个事实问题，否则整条「车辆检测 -> ROI -> 车牌检测」
可能就是空中楼阁：

  Q1  CCPD 整图上的车牌到底多大？
      —— 真值只在文件名里（`<id>-<tilt>-<bbox>-<verts>-<lp>-<bright>-<blur>`），
         第 3 段就是车牌框 `x1&y1_x2&y2`。纯字符串解析，零推理成本。

  Q2  通用车辆检测器（YOLOv5su / COCO）能不能在 CCPD 整图上框出车？
      —— 决定 ROI 这一刀切不切得下去。切不下去就换数据，别硬做。

输出：probe_roi.json（UTF-8，供后续 Read / 喂进实验脚本）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort

# ---- COCO 里属于「车」的类：car / motorcycle / bus / truck ----
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# --------------------------------------------------------------------------
# Q1：从文件名解车牌框
# --------------------------------------------------------------------------
def parse_plate_bbox(name: str):
    """`00341954022988-90_90-371&464_490&503-...` -> (x1,y1,x2,y2) 或 None。"""
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        a, b = parts[2].split("_")
        x1, y1 = (int(v) for v in a.split("&"))
        x2, y2 = (int(v) for v in b.split("&"))
    except (ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def probe_q1(img_dir: str, limit: int):
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    if limit:
        names = names[:limit]

    # 单张图的尺寸只读一次（CCPD2019 全部同尺寸，但仍校验，不假设）
    sample = cv2.imread(os.path.join(img_dir, names[0]))
    if sample is None:
        raise RuntimeError(f"读不出图: {names[0]}")
    img_h, img_w = sample.shape[:2]

    pw, ph, frac = [], [], []
    off_img = 0
    for n in names:
        bb = parse_plate_bbox(n)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        w, h = x2 - x1, y2 - y1
        pw.append(w)
        ph.append(h)
        frac.append((w * h) / (img_w * img_h))
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            off_img += 1

    def stats(v):
        a = np.asarray(v, dtype=np.float64)
        return {
            "n": int(a.size),
            "min": float(a.min()),
            "p05": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }

    # letterbox 到 320 之后，车牌还剩多少像素？（预测「直检」的物理上限）
    r = min(320.0 / img_w, 320.0 / img_h)
    return {
        "img_dir": img_dir,
        "img_size": [img_w, img_h],
        "n_parsed": len(pw),
        "plate_w_px": stats(pw),
        "plate_h_px": stats(ph),
        "plate_area_frac_of_image": stats(frac),
        "letterbox_r_to_320": round(r, 4),
        "plate_w_px_after_letterbox": stats([w * r for w in pw]),
        "plate_h_px_after_letterbox": stats([h * r for h in ph]),
        "n_bbox_out_of_image": off_img,
    }


# --------------------------------------------------------------------------
# Q2：车辆检测
# --------------------------------------------------------------------------
def letterbox(img, new=320, color=114):
    """YOLOv5 风格 letterbox：等比缩放 + 居中灰边。返回 (图, 缩放比, 左pad, 上pad)。"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2.0, (new - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, left, top


def decode_yolov5(out, r, pad_x, pad_y, conf_th=0.25, iou_th=0.45):
    """ultralytics 导出的 YOLOv5u ONNX：输出 (1, 4+nc, N)，已解码、无 objectness。

    坐标是**输入尺度**的 xywh -> 反 letterbox 回原图 -> NMS。
    """
    pred = out[0]                       # (4+nc, N)
    if pred.shape[0] < pred.shape[1]:   # 期望 (C, N)，若反过来就转
        pred = pred.T
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_id = scores_all.argmax(1)
    conf = scores_all[np.arange(len(cls_id)), cls_id]

    keep = conf >= conf_th
    if not keep.any():
        return []

    xywh = boxes_xywh[keep]
    cls_id = cls_id[keep]
    conf = conf[keep]

    # 输入尺度 xywh -> 原图 xywh
    cx = (xywh[:, 0] - pad_x) / r
    cy = (xywh[:, 1] - pad_y) / r
    bw = xywh[:, 2] / r
    bh = xywh[:, 3] / r
    x1, y1 = cx - bw / 2, cy - bh / 2

    boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(boxes, conf.tolist(), conf_th, iou_th)
    if len(idxs) == 0:
        return []
    idxs = np.asarray(idxs).reshape(-1)
    return [
        {
            "cls": int(cls_id[i]),
            "name": VEHICLE_CLASSES.get(int(cls_id[i]), f"cls{int(cls_id[i])}"),
            "conf": round(float(conf[i]), 4),
            "xyxy": [round(float(cx[i] - bw[i] / 2), 1), round(float(cy[i] - bh[i] / 2), 1),
                     round(float(cx[i] + bw[i] / 2), 1), round(float(cy[i] + bh[i] / 2), 1)],
        }
        for i in idxs
    ]


def probe_q2(img_dir, names, veh_onnx, imgsz, conf_th, iou_th):
    sess = ort.InferenceSession(veh_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    in_shape = sess.get_inputs()[0].shape
    out_shape = sess.get_outputs()[0].shape

    n_with_vehicle = 0
    cls_counter = Counter()
    det_per_img = []
    empty_examples = []
    samples = []

    for n in names:
        img = cv2.imread(os.path.join(img_dir, n))
        if img is None:
            continue
        lb, r, px, py = letterbox(img, imgsz)
        blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = sess.run([out_name], {in_name: blob})[0]

        dets = decode_yolov5(out, r, px, py, conf_th, iou_th)
        veh = [d for d in dets if d["cls"] in VEHICLE_CLASSES]
        det_per_img.append(len(veh))
        if veh:
            n_with_vehicle += 1
            for d in veh:
                cls_counter[d["name"]] += 1
            if len(samples) < 8:
                samples.append({"file": n, "vehicles": veh})
        elif len(empty_examples) < 8:
            empty_examples.append({"file": n, "all_dets": dets})

    return {
        "veh_onnx": veh_onnx,
        "input_shape": str(in_shape),
        "output_shape": str(out_shape),
        "imgsz": imgsz,
        "conf_th": conf_th,
        "n_images": len(det_per_img),
        "n_with_vehicle": n_with_vehicle,
        "hit_rate": round(n_with_vehicle / len(det_per_img), 4) if det_per_img else 0.0,
        "vehicle_class_counts": dict(cls_counter),
        "det_per_image_mean": round(float(np.mean(det_per_img)), 2) if det_per_img else 0.0,
        "vehicle_samples": samples,
        "no_vehicle_samples": empty_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "LprDemo", "entry",
        "src", "main", "resources", "rawfile", "assets", "ccpd"))
    ap.add_argument("--veh-onnx", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "yolov5su.onnx"))
    ap.add_argument("--q1-limit", type=int, default=0)
    ap.add_argument("--q2-limit", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "probe_roi.json"))
    a = ap.parse_args()

    img_dir = os.path.abspath(a.img_dir)
    if not os.path.isdir(img_dir):
        print(f"[fail] 图片目录不存在: {img_dir}", file=sys.stderr)
        return 2

    print(f"[q1] 解析车牌框：{img_dir}")
    q1 = probe_q1(img_dir, a.q1_limit)
    print(f"     n={q1['n_parsed']}  图幅={q1['img_size']}  "
          f"车牌中位 {q1['plate_w_px']['p50']:.0f}x{q1['plate_h_px']['p50']:.0f}px  "
          f"占图 {q1['plate_area_frac_of_image']['p50'] * 100:.3f}%")
    print(f"     letterbox r={q1['letterbox_r_to_320']} -> "
          f"车牌中位 {q1['plate_w_px_after_letterbox']['p50']:.1f}x"
          f"{q1['plate_h_px_after_letterbox']['p50']:.1f}px")

    result = {"q1_plate_geometry": q1}

    veh_onnx = os.path.abspath(a.veh_onnx)
    if os.path.isfile(veh_onnx):
        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
        if a.q2_limit:
            # 均匀抽样，避免只看到某一类场景
            step = max(1, len(names) // a.q2_limit)
            names = names[::step][:a.q2_limit]
        print(f"[q2] 车辆检测：{os.path.basename(veh_onnx)} on n={len(names)}")
        result["q2_vehicle_detection"] = probe_q2(
            img_dir, names, veh_onnx, a.imgsz, a.conf, a.iou)
        q2 = result["q2_vehicle_detection"]
        print(f"     输出 {q2['output_shape']}  检出车辆 {q2['n_with_vehicle']}/{q2['n_images']} "
              f"({q2['hit_rate'] * 100:.1f}%)  {q2['vehicle_class_counts']}")
    else:
        result["q2_vehicle_detection"] = {"error": f"未找到 {veh_onnx}"}
        print(f"[q2] 跳过：未找到 {veh_onnx}")

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
