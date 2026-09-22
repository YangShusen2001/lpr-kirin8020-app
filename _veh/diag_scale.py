#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尺度对照诊断：我的 ONNX 后处理 vs ultralytics 原生推理。

## 为什么必须做这一步

一次探测里出现反常：车辆检测在 imgsz=640 下的检出率（45.0%）**远低于** 320（86.7%），
且类别分布从 car 主导变成 truck 大量出现。640 输入不可能比 320 差 —— 反常意味着
**要么我的 letterbox/后处理有 bug，要么 CN 模型导出有问题**。

在被测对象（ROI 实验）依赖这套后处理的前提下，必须先自证后处理无偏：
同一张图、同一阈值，三条路径对比：
  1. ultralytics 原生（`.pt`，它自己做 letterbox + NMS）—— 作为参考真值
  2. 我的后处理 + 320 ONNX
  3. 我的后处理 + 640 ONNX

若 2/3 与 1 在**同一 imgsz** 下一致，后处理可信，尺度差异就是真实现象；
若不一致，说明我的实现有问题，ROI 实验的结论要作废重跑。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_roi import VEHICLE_CLASSES, decode_yolov5, letterbox  # noqa: E402

IMG_DIR = os.path.join(HERE, "..", "LprDemo", "entry", "src", "main",
                       "resources", "rawfile", "assets", "ccpd")
VEH_CLASS_IDS = set(VEHICLE_CLASSES)


def read_img(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def my_onnx_detect(sess, img, imgsz, conf_th=0.25, iou_th=0.45):
    lb, r, px, py = letterbox(img, imgsz)
    blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: blob})[0]
    dets = decode_yolov5(out, r, px, py, conf_th, iou_th)
    return [d for d in dets if d["cls"] in VEH_CLASS_IDS]


def main():
    os.environ.setdefault("YOLO_VERBOSE", "false")
    from ultralytics import YOLO

    names = sorted(n for n in os.listdir(IMG_DIR) if n.lower().endswith(".jpg"))
    step = max(1, len(names) // 24)
    names = names[::step][:24]

    m = YOLO(os.path.join(HERE, "yolov5su.pt"))
    s320 = ort.InferenceSession(os.path.join(HERE, "yolov5su_320.onnx"),
                                providers=["CPUExecutionProvider"])
    s640 = ort.InferenceSession(os.path.join(HERE, "yolov5su_640.onnx"),
                                providers=["CPUExecutionProvider"])

    rows = []
    for n in names:
        img = read_img(os.path.join(IMG_DIR, n))
        rec = {"file": n}
        for imgsz, sess in ((320, s320), (640, s640)):
            r = m.predict(img, imgsz=imgsz, conf=0.25, iou=0.45, verbose=False)[0]
            u_veh = [m.names[int(c)] for c in r.boxes.cls.tolist()
                     if int(c) in VEH_CLASS_IDS]
            mine = my_onnx_detect(sess, img, imgsz)
            rec[f"ultra_{imgsz}"] = {"n": len(u_veh), "cls": sorted(u_veh)}
            rec[f"mine_{imgsz}"] = {"n": len(mine), "cls": sorted(d["name"] for d in mine),
                                    "conf_max": round(max((d["conf"] for d in mine), default=0), 3)}
        rows.append(rec)

    def agg(prefix):
        with_det = sum(1 for r in rows if r[f"{prefix}"]["n"] > 0)
        return {"images_with_vehicle": with_det, "n": len(rows),
                "rate": round(with_det / len(rows), 4),
                "total_boxes": sum(r[f"{prefix}"]["n"] for r in rows)}

    summary = {
        "img_dir": IMG_DIR,
        "n": len(rows),
        "ultralytics_320": agg("ultra_320"),
        "mine_320": agg("mine_320"),
        "ultralytics_640": agg("ultra_640"),
        "mine_640": agg("mine_640"),
        "rows": rows,
    }
    out = os.path.join(HERE, "diag_scale.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    for k in ("ultralytics_320", "mine_320", "ultralytics_640", "mine_640"):
        v = summary[k]
        print(f"  {k:18s} 有车图 {v['images_with_vehicle']}/{v['n']} "
              f"({v['rate'] * 100:.1f}%)  框总数 {v['total_boxes']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
