#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI 的价值边界：按输入尺度扫描，找「直检 vs ROI」的交叉点。

## 为什么还要做这一步

`roi_vs_direct.py` 已经证明：在 CCPD 原生尺度上 ROI 掉召回（-17.5%），
且 `roi_zoom.py` 算出 ROI 确实把车牌放大了 1.624x —— 放大是真的，收益为零也是真的。

合起来只能推出一个解释：**直检时车牌已经在检测器的舒适尺度上，放大没有边际收益**。
如果这个解释成立，那么**把整图缩小**（等价于"拍摄距离变远、车牌变像素化"），
直检会先失效，ROI 的放大就会开始有收益 —— 两条召回曲线必然交叉。

本脚本就是去找这个交叉点，把"ROI 有没有用"从一个是非题变成一个**有边界的结论**：

    车牌 < X px 时，ROI 值得做；>= X px 时，ROI 是净亏损。

这比"ROI 没用"更有价值，也更接近工程真相。

## 设计

对每个缩放因子 s（图按 s 缩小，GT 框同步缩放）：
  A 直检   缩放图 -> 车牌检测
  B ROI    缩放图 -> 车辆检测(320) -> 取最高分车框外扩 -> 裁 -> 车牌检测 -> 映射回缩放图
判据仍是 IoU >= 0.5。

注意 ROI 路自身也有失效点：s 太小时车也变像素化 -> 车辆检测先崩 -> ROI 路归零。
两条曲线各有各的崩溃点，交叉区才是 ROI 的适用区间。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
HLPR_TOOLS = r"C:\Users\26671\Desktop\Test\lpr-showcase\tools"
sys.path.insert(0, HLPR_TOOLS)
sys.path.insert(0, HERE)

import hlpr_reference as H          # noqa: E402
from probe_roi import VEHICLE_CLASSES, decode_yolov5, letterbox, parse_plate_bbox  # noqa: E402
from roi_vs_direct import best_iou, expand_box, read_img  # noqa: E402

DEFAULT_IMG_DIR = os.path.join(
    HERE, "..", "LprDemo", "entry", "src", "main", "resources", "rawfile", "assets", "ccpd")
DEFAULT_PLATE_ONNX = r"C:\Users\26671\Desktop\车牌识别\_scratch\ms\y5fu_320x_sim.onnx"


def run(args):
    img_dir = os.path.abspath(args.img_dir)
    names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    if args.limit:
        step = max(1, len(names) // args.limit)
        names = names[::step][:args.limit]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    det = ort.InferenceSession(args.plate_onnx, so, providers=["CPUExecutionProvider"])
    veh = ort.InferenceSession(args.veh_onnx, so, providers=["CPUExecutionProvider"])
    vin, vout = veh.get_inputs()[0].name, veh.get_outputs()[0].name

    scales = [float(s) for s in args.scales.split(",")]
    rows = []

    for k, n in enumerate(names):
        full = read_img(os.path.join(img_dir, n))
        gt0 = parse_plate_bbox(n)
        if full is None or gt0 is None:
            continue
        ih, iw = full.shape[:2]

        rec = {"file": n, "gt_px_full": [gt0[2] - gt0[0], gt0[3] - gt0[1]], "s": {}}
        for s in scales:
            if args.mode == "inset":
                # 关键：**保留画布尺寸**，把内容缩小后居中嵌进去（周围填灰）。
                # 因为 letterbox 总会把长边缩到 net，若直接把整图 resize 小，
                # 车牌在输入里的像素数**不变**（r 会同步变大补偿掉）——
                # 那样测的是"缩小图片"，不是"车牌变小"（2026-09-22 踩过，首轮扫描
                # 5 档的"牌宽@输入"全是 71.8px，说明自变量根本没动）。
                nw2, nh2 = max(8, int(round(iw * s))), max(8, int(round(ih * s)))
                small = cv2.resize(full, (nw2, nh2),
                                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
                ox, oy = (iw - nw2) // 2, (ih - nh2) // 2
                canvas[oy:oy + nh2, ox:ox + nw2] = small
                img = canvas
                gt = [gt0[0] * s + ox, gt0[1] * s + oy, gt0[2] * s + ox, gt0[3] * s + oy]
            else:
                nw2, nh2 = max(16, int(round(iw * s))), max(16, int(round(ih * s)))
                if s == 1.0:
                    img = full
                else:
                    # 缩小用 INTER_AREA（等价于相机离远/降采样），放大用 INTER_LINEAR
                    img = cv2.resize(full, (nw2, nh2),
                                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                gt = [v * s for v in gt0]
            ihh, iww = img.shape[:2]

            A = H.detect(det, img, args.conf, args.iou)
            A_s = A[np.argsort(-A[:, 4])] if len(A) else A
            _, a_top1, a_any = best_iou([r[:4].tolist() for r in A_s], gt)

            lb, r, px, py = letterbox(img, args.veh_imgsz)
            blob = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            vd = veh.run([vout], {vin: blob})[0]
            vdets = [d for d in decode_yolov5(vd, r, px, py, args.veh_conf, args.iou)
                     if d["cls"] in VEHICLE_CLASSES]

            b_top1, b_any, n_det, roi = False, False, 0, None
            if vdets:
                top = max(vdets, key=lambda d: d["conf"])
                roi = expand_box(top["xyxy"], iww, ihh, args.pad)
                rx1, ry1, rx2, ry2 = roi
                if rx2 - rx1 > 8 and ry2 - ry1 > 8:
                    B = H.detect(det, img[ry1:ry2, rx1:rx2], args.conf, args.iou)
                    B_s = B[np.argsort(-B[:, 4])] if len(B) else B
                    _, b_top1, b_any = best_iou(
                        [[rx1 + r0[0], ry1 + r0[1], rx1 + r0[2], ry1 + r0[3]] for r0 in B_s], gt)
                    n_det = int(len(B))

            # 车牌在这条路径的 letterbox 输入里有多少像素（横轴的解释变量）
            r_direct = min(args.net / iww, args.net / ihh)
            plate_w_input = (gt[2] - gt[0]) * r_direct
            rec["s"][str(s)] = {
                "img_wh": [iww, ihh], "plate_w_in_img": round(gt[2] - gt[0], 1),
                "plate_w_at_net_input": round(plate_w_input, 1),
                "n_veh": len(vdets), "roi": list(roi) if roi else None, "roi_n_det": n_det,
                "A_top1": a_top1, "A_any": a_any, "B_top1": b_top1, "B_any": b_any,
            }
        rows.append(rec)
        if (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(names)}", flush=True)

    n = len(rows)
    curve = []
    for s in scales:
        k = str(s)
        sub = [r for r in rows if k in r["s"]]
        if not sub:
            continue
        pw = float(np.mean([r["s"][k]["plate_w_at_net_input"] for r in sub]))
        curve.append({
            "scale": s,
            "img_wh": sub[0]["s"][k]["img_wh"],
            "plate_w_at_net_input_px": round(pw, 1),
            "A_recall_top1": round(sum(r["s"][k]["A_top1"] for r in sub) / len(sub), 4),
            "B_recall_top1": round(sum(r["s"][k]["B_top1"] for r in sub) / len(sub), 4),
            "A_recall_any": round(sum(r["s"][k]["A_any"] for r in sub) / len(sub), 4),
            "B_recall_any": round(sum(r["s"][k]["B_any"] for r in sub) / len(sub), 4),
            "no_vehicle_rate": round(sum(1 for r in sub if r["s"][k]["n_veh"] == 0) / len(sub), 4),
            "roi_empty_rate": round(sum(1 for r in sub
                                       if r["s"][k]["n_veh"] > 0 and r["s"][k]["roi_n_det"] == 0)
                                   / len(sub), 4),
        })

    # 交叉点：ROI 召回首次 >= 直检召回 的最大缩放因子
    crossover = None
    for c in sorted(curve, key=lambda x: x["scale"]):
        if c["B_recall_top1"] >= c["A_recall_top1"]:
            crossover = {"at_scale": c["scale"], "plate_w_px": c["plate_w_at_net_input_px"],
                         "A": c["A_recall_top1"], "B": c["B_recall_top1"]}
        else:
            crossover = None   # 从大尺度往小走，最后一次持平/反超才算交叉点

    return {
        "config": {"img_dir": img_dir, "n": n, "scales": scales, "roi_pad": args.pad,
                   "mode": args.mode,
                   "net": args.net, "plate_onnx": args.plate_onnx, "veh_onnx": args.veh_onnx,
                   "veh_imgsz": args.veh_imgsz},
        "curve_top1": curve,
        "crossover": crossover,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--plate-onnx", default=DEFAULT_PLATE_ONNX)
    ap.add_argument("--veh-onnx", default=os.path.join(HERE, "yolov5su_320.onnx"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--scales", default="1.0,0.75,0.5,0.375,0.25")
    ap.add_argument("--mode", default="inset", choices=["inset", "resize"],
                    help="inset=保留画布、内容缩小后居中嵌入（真实模拟远距离/低分辨率）；"
                         "resize=直接缩小整图（会被 letterbox 补偿，自变量失效）")
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--net", type=int, default=320)
    ap.add_argument("--veh-imgsz", type=int, default=320)
    ap.add_argument("--veh-conf", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "scale_sweep.json"))
    a = ap.parse_args()

    s = run(a)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)

    print(f"\nn={s['config']['n']}  外扩={s['config']['roi_pad']}  车辆输入={s['config']['veh_imgsz']}")
    print(f"  {'scale':>6} {'图幅':>12} {'牌宽@输入':>10} {'直检':>8} {'ROI':>8} "
          f"{'差':>8} {'无车率':>8}")
    for c in s["curve_top1"]:
        d = c["B_recall_top1"] - c["A_recall_top1"]
        print(f"  {c['scale']:>6} {str(c['img_wh']):>12} {c['plate_w_at_net_input_px']:>10} "
              f"{c['A_recall_top1']:>8.4f} {c['B_recall_top1']:>8.4f} {d:>+8.4f} "
              f"{c['no_vehicle_rate']:>8.3f}")
    print(f"  交叉点: {s['crossover']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
