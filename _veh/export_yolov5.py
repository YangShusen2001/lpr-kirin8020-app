#!/usr/bin/env python3
"""导出 YOLOv5 COCO 预训练为 ONNX，并打印车辆类索引。

为什么用 YOLOv5 而不是 YOLOv8（有本机失败证据）
==============================================
`D:\\Tools\\mindspore-lite\\yolo_conv.txt` / `yolo_conv2.txt` 记录了两次独立尝试，
均因 YOLOv8 的 DFL 模块导致 MindSpore Lite 转换失败：

    InferShapeByNNACL for op: /model.22/dfl/conv/Conv failed
    InferSubgraph index: 0 failed, ret: -500

所以必须选 YOLOv5 系（架构与现有 y5fu_320x_head 一致，且 NPU 落点已验证 4.95 ms）。

用法
====
  python export_yolov5.py                      # 默认 320，输出 yolov5s_320.onnx
  python export_yolov5.py --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import sys

# COCO 里属于「车辆」的类（其余 76 类对我们无用）
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 YOLOv5s COCO 为 ONNX")
    ap.add_argument("--weights", default="yolov5s.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        print(f"[export] 找不到权重 {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(args.weights)

    names = model.names
    print(f"[export] 权重 = {args.weights}")
    print(f"[export] 类别数 = {len(names)}")
    veh = {i: n for i, n in names.items() if n in VEHICLE_CLASSES}
    print(f"[export] 车辆类索引 = {veh}")

    # 抽查几个常见类，确认 COCO 映射没变
    for probe in ("person", "car", "bus", "truck", "motorcycle"):
        idx = [i for i, n in names.items() if n == probe]
        print(f"[export]   {probe:12s} -> index {idx}")

    out = os.path.abspath(args.out or f"yolov5s_{args.imgsz}.onnx")
    if os.path.exists(out):
        print(f"[export] 目标已存在，将覆盖: {out}", file=sys.stderr)

    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                        simplify=False, dynamic=False)

    # ultralytics 的 export 自己决定落盘文件名（`<weights.stem>.onnx`），**不看我们的
    # --out**，导出完再搬一次才能让参数真正生效。
    # 2026-09-22 踩过：先导 320 再导 640，第二次把第一次的产物静默盖掉了 ——
    # 表面上两条命令都成功，实际上 320 版已经不存在。
    if os.path.abspath(path) != out:
        os.replace(path, out)

    print(f"[export] ONNX = {out}")
    print(f"[export] 大小 = {os.path.getsize(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
