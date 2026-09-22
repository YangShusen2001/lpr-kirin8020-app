#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3 自证的 PC 侧独立复核。

设备侧（`LprRoiSelfTest`）把逐 case 报告写进 hilog（`T3ROI ...`）。本脚本做两件事：

1. **独立的第二实现**：用 Python `math` 重算 ROI 几何（外扩 → floor/ceil → clamp），
   与设备报的 `x0/y0/w/h/clamped/valid` 逐字段比对。两条实现分别写在 C++ 与 Python 里，
   不是同一份代码自说自话。
2. **跨实现像素证据**：用同一张 PNG 重算「整图 RGB 和」与「各 crop case 的 ROI 区域
   RGB 和」，与设备报的 `rgbSumAll` / `rgbSum` 比对。口径与 C++ 侧 `LprRgbSum`
   一致：跳过 alpha，R+G+B 直加。

用法：
    python _veh/roi_selftest_check.py --log _veh/devlog_T3ROI.txt \
        --png LprDemo/entry/src/main/resources/rawfile/assets/veh320/veh320.png
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

EXPAND = 0.15  # kRoiExpandDefault


def roi_from_box(box, img_w, img_h, expand):
    """Python 侧独立实现，规则同 spec：向外取整 + clamp 到图内。"""
    x1 = min(box[0], box[2])
    y1 = min(box[1], box[3])
    x2 = max(box[0], box[2])
    y2 = max(box[1], box[3])
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or img_w <= 0 or img_h <= 0 or expand < 0:
        return None
    ex, ey = expand * bw, expand * bh
    x0 = math.floor(x1 - ex)
    y0 = math.floor(y1 - ey)
    X1 = math.ceil(x2 + ex)
    Y1 = math.ceil(y2 + ey)
    clamped = False
    if x0 < 0:
        x0, clamped = 0, True
    if y0 < 0:
        y0, clamped = 0, True
    if X1 > img_w:
        X1, clamped = img_w, True
    if Y1 > img_h:
        Y1, clamped = img_h, True
    w, h = X1 - x0, Y1 - y0
    return {"x0": x0, "y0": y0, "w": w, "h": h,
            "valid": w > 0 and h > 0, "clamped": clamped}


def parse_log(path):
    meta, cases = {}, {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "T3ROI" not in line:
                continue
            body = line.split("T3ROI", 1)[1].strip()
            if body.startswith("VERDICT") or body.startswith("FAIL") or body.startswith("EXCEPTION"):
                continue
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
    return meta, cases


def rgb_sum(arr):
    return int(arr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--png", required=True)
    a = ap.parse_args()

    meta, cases = parse_log(a.log)
    if not cases:
        sys.exit("[t3check] 日志里没有 T3ROI case 行")

    img = np.asarray(Image.open(a.png).convert("RGB"), dtype=np.int64)
    H, W = img.shape[0], img.shape[1]

    print(f"[t3check] 日志 case 数 = {len(cases)}")
    print(f"[t3check] 设备 srcSize = {meta.get('srcSize')} / PNG = {W}x{H}")
    print(f"[t3check] 设备 geomFrame = {meta.get('geomFrame')}")

    prob = []

    if meta.get("srcSize") != f"{W}x{H}":
        prob.append(f"srcSize 与 PNG 不符：设备 {meta.get('srcSize')} vs PNG {W}x{H}")

    # ---- 跨实现像素证据：整图 RGB 和 ----
    dev_all = meta.get("rgbSumAll")
    py_all = rgb_sum(img)
    ok_all = dev_all is not None and int(dev_all) == py_all
    print(f"[t3check] 整图 rgbSum 设备={dev_all} PC={py_all} "
          f"{'✓' if ok_all else '✗'}")
    if not ok_all:
        prob.append(f"整图 rgbSum 不一致：设备 {dev_all} vs PC {py_all}")

    # ---- 逐 case ----
    n_ok = 0
    for name in sorted(cases):
        kv = cases[name]
        dev_ok = kv.get("ok") == "1"
        notes = []
        if not dev_ok:
            notes.append("设备侧断言 ok=0")
            prob.append(f"{name}: 设备侧断言失败（{kv}）")

        # 几何重算（有 box 就能做）。参数一律取日志里报的**原值** ——
        # 猜参数会误判：例如 roi-bad-src-size 传的是 0x0，猜成 320x320 就会把
        # 一个"应当无效"的用例算成有效，反过来冤枉设备。
        if "box" in kv and "x0" in kv:
            box = [float(v) for v in kv["box"].split(",")]
            fw = int(kv.get("imgW", 320))
            fh = int(kv.get("imgH", 320))
            expand = float(kv.get("expand", EXPAND))
            exp = roi_from_box(box, fw, fh, expand)
            if exp is None:
                if kv.get("valid") == "1":
                    notes.append("PC 认为应无效但设备给了有效 ROI")
                    prob.append(f"{name}: PC 判无效 / 设备判有效")
                else:
                    notes.append("PC 也判无效 ✓")
            else:
                got = {"x0": int(kv["x0"]), "y0": int(kv["y0"]),
                       "w": int(kv["w"]), "h": int(kv["h"]),
                       "valid": kv.get("valid") == "1",
                       "clamped": kv.get("clamped") == "1"}
                for k in ("x0", "y0", "w", "h", "valid", "clamped"):
                    if got[k] != exp[k]:
                        notes.append(f"{k}: 设备={got[k]} PC={exp[k]}")
                        prob.append(f"{name}: 字段 {k} 设备 {got[k]} != PC {exp[k]}")
                notes.append("几何: PC 重算一致")

        # 跨实现像素证据（crop case 带 rgbSum）
        if "rgbSum" in kv:
            x0, y0 = int(kv["x0"]), int(kv["y0"])
            w, h = int(kv["w"]), int(kv["h"])
            if 0 <= x0 and 0 <= y0 and x0 + w <= W and y0 + h <= H:
                py_sum = rgb_sum(img[y0:y0 + h, x0:x0 + w, :])
                dev_sum = int(kv["rgbSum"])
                if py_sum != dev_sum:
                    notes.append(f"rgbSum 设备={dev_sum} PC={py_sum}")
                    prob.append(f"{name}: rgbSum 设备 {dev_sum} != PC {py_sum}")
                else:
                    notes.append(f"rgbSum={dev_sum} ✓(PC 重算一致)")
            else:
                notes.append("ROI 越出 PNG，跳过像素复核")
                prob.append(f"{name}: ROI {x0},{y0},{w},{h} 越出 PNG")

        if dev_ok:
            n_ok += 1
        print(f"[t3check] {'✓' if dev_ok else '✗'} {name:<24} " + ("; ".join(notes) or "-"))

    print()
    print(f"[t3check] 设备侧 ok=1 的 case: {n_ok} / {len(cases)}")
    if meta.get("total") and meta.get("failed"):
        print(f"[t3check] 设备自报 total={meta['total']} failed={meta['failed']}")
    print()
    if prob:
        for p in prob:
            print(f"[t3check] ✗ {p}")
        return 2
    print("[t3check] ✓ T3 自证通过：几何与像素两项都由 PC 侧独立重算确认")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
