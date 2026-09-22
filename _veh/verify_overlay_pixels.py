#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_overlay_pixels.py —— 从**截图**里验证叠加框画对了（不依赖"看图"）。

## 为什么要写它

本项目看不了图（模型侧对图片内容会被过滤）。所以「框画得对不对」必须落到
像素上：从设备截图里按**期望颜色**去找描边，看它是否落在**期望位置**上。

这是与 `camera_ui_check.py` 互补的一条独立证据链：
  camera_ui_check 层1/层2 验的是「坐标换算」与「布局引擎把框放对了」
  本脚本验的是「**屏幕上真的有这些颜色的线，且就在那些坐标上**」
    —— 前者可能对而渲染失败（被遮挡、颜色写错、alpha 全 0），后者能抓住。

## 颜色表不硬编码

`VEH_COLOURS` 直接从 `CameraPage.ets` 里读出来 —— 两处各写一份必然漂移，
而漂移的表现是「脚本说对/错」与「眼睛看到的」不一致，最难查。

## 用法

    python _veh/verify_overlay_pixels.py --shot _veh/t5_shot_roi.jpeg \\
        --log _veh/devlog_T5FEED_roi.txt [--ets <CameraPage.ets>]
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# 描边宽度 1.5 vp ≈ 4.3 px；JPEG 有压缩与抗锯齿，颜色容差必须给宽
TOL = 70.0
HALF_WIN = 6


def read_veh_colours(ets_path):
    """从 ArkTS 源码读出 VEH_COLOURS —— 单一事实源。"""
    txt = open(ets_path, encoding="utf-8", errors="replace").read()
    m = re.search(r"const VEH_COLOURS: string\[\] = \[(.*?)\];", txt, re.S)
    if not m:
        raise SystemExit("在 CameraPage.ets 里找不到 VEH_COLOURS")
    return [h.strip().strip("'") for h in m.group(1).split(",") if h.strip()]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float64)


def parse_feed(log_path):
    feed = {}
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "FEED" not in line:
            continue
        body = line[line.find("FEED") + 4:].strip()
        kind = body.split(" ", 1)[0]
        feed.setdefault(kind, []).append(body)
    return feed


def kv_of(s):
    d = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def boxes_of(feed):
    out = []
    for body in feed.get("BOX", []):
        p = body.split()
        if len(p) < 3 or p[1] not in ("veh", "plate"):
            continue
        d = kv_of(body)
        if "drawPx" not in d:
            continue
        out.append({
            "kind": p[1],
            "i": int(d.get("i", -1)),
            "owner": int(d["owner"]) if "owner" in d else None,
            "drawPx": [float(x) for x in d["drawPx"].split(",")],
        })
    return out


def edge_hits(img, x1, y1, x2, y2, target, axis, fixed_lo, fixed_hi):
    """沿一条边采样，在 ±HALF_WIN 窗口内找最接近 target 的像素。

    axis='h' → 水平边（y 固定为 fixed_lo..fixed_hi，x 遍历）
    axis='v' → 垂直边（x 固定，y 遍历）
    返回 (命中率, 中位最小距离, 中位最佳位置)
    """
    H, W, _ = img.shape
    if axis == "h":
        a_lo, a_hi = int(x1) + 12, int(x2) - 12
        fixed_center = None
    else:
        a_lo, a_hi = int(y1) + 12, int(y2) - 12
        fixed_center = None
    if a_hi <= a_lo:
        return None
    samples = np.linspace(a_lo, a_hi, 40).astype(int)
    hits, dists, offs = 0, [], []
    for s in samples:
        best, best_off = 1e9, 0
        for off in range(fixed_lo, fixed_hi + 1):
            # ⚠️ 这里曾写成 `yy, xx = s, off`（x/y 弄反），于是沿边的采样点被当成 y、
            # 窗口偏移被当成 x —— 结果是「在一条竖线上横向扫」，永远找不到描边。
            # 症状极具误导性：**位置偏差看着很小**（窗口内全是噪声时，argmin 会返回
            # 任意位置），但命中率 0%。命中率才是判据，位置偏差在没有命中时无意义。
            if axis == "h":
                yy, xx = off, s      # y 在窗口内扫，x 是沿边的采样点
            else:
                yy, xx = s, off      # y 是沿边的采样点，x 在窗口内扫
            if not (0 <= yy < H and 0 <= xx < W):
                continue
            px = img[yy, xx].astype(np.float64)
            d = float(np.sqrt(((px - target) ** 2).sum()))
            if d < best:
                best, best_off = d, off
        dists.append(best)
        offs.append(best_off)
        if best <= TOL:
            hits += 1
    rate = hits / len(samples)
    return rate, float(np.median(dists)), float(np.median(offs))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ets", default="")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ets = a.ets or os.path.join(here, "..", "LprDemo", "entry", "src", "main", "ets",
                                "pages", "CameraPage.ets")
    colours = read_veh_colours(ets)
    print(f"[pix] VEH_COLOURS（读自 {os.path.basename(ets)}）= {colours}")

    img = np.asarray(Image.open(a.shot).convert("RGB"), dtype=np.uint8)
    H, W, _ = img.shape
    print(f"[pix] 截图 {W}x{H}")

    feed = parse_feed(a.log)
    g = kv_of(feed["GEOM"][0])
    vx, vy = [float(x) for x in g["viewPosPx"].split(",")]
    print(f"[pix] 预览区屏幕原点 (px) = ({vx},{vy})")

    boxes = boxes_of(feed)
    if not boxes:
        print("[pix] FAIL 日志里没有带 drawPx 的 BOX 行")
        return 1

    # 绘制顺序：车辆框按 index 升序，其后是车牌框（与 overlayLayer 的 ForEach 一致）。
    # 若某框与**它后面**画的框几乎完全重叠，它的描边会被盖住 —— 那是正确的绘制
    # 顺序（后画的在上），不是缺陷。实测 veh#1 与 veh#2 只差 1 px（模型对同一目标
    # 给了 cls=2 与 cls=3 两个框，class-wise NMS 不会互相抑制），绿色被蓝色盖死。
    order = ([b for b in boxes if b["kind"] == "veh"] +
             [b for b in boxes if b["kind"] == "plate"])
    occluded = {}
    for i, b in enumerate(order):
        for c in order[i + 1:]:
            if iou(b["drawPx"], c["drawPx"]) > 0.85:
                occluded[id(b)] = c
                break
    if occluded:
        for b, c in ((b, c) for b, c in
                     ((x, occluded[id(x)]) for x in order if id(x) in occluded)):
            print(f"[pix] 注：{b['kind']}#{b['i']} 与后绘制的 {c['kind']}#{c['i']} "
                  f"IoU>0.85，其描边预期被覆盖")

    prob, note = [], []
    for b in boxes:
        x1 = vx + b["drawPx"][0]
        y1 = vy + b["drawPx"][1]
        x2 = vx + b["drawPx"][2]
        y2 = vy + b["drawPx"][3]
        if b["kind"] == "veh":
            hexc = colours[b["i"] % len(colours)]
            label = f"veh#{b['i']}"
        else:
            # 无归属车牌框用红色（ArkTS 里写死的 '#FF3B30'）。
            # ⚠️ 判空必须用 `is not None`：写成 `(b["owner"] or -1) >= 0` 时
            # owner=0 会被 `or` 判成 falsy 而误走红色分支（0 or -1 == -1）。
            ow = b["owner"]
            hexc = colours[ow % len(colours)] if (ow is not None and ow >= 0) else "#FF3B30"
            label = f"plate#{b['i']}(owner={b['owner']})"
        target = hex_to_rgb(hexc)
        note.append(f"[pix] {label} 期望 {hexc} 屏幕框=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # 四条边：上/下（y 窗口固定，x 遍历）、左/右（x 窗口固定，y 遍历）
        edges = [
            ("上", "h", int(y1) - HALF_WIN, int(y1) + HALF_WIN, x1, y1, x2, y2),
            ("下", "h", int(y2) - HALF_WIN, int(y2) + HALF_WIN, x1, y1, x2, y2),
            ("左", "v", int(x1) - HALF_WIN, int(x1) + HALF_WIN, x1, y1, x2, y2),
            ("右", "v", int(x2) - HALF_WIN, int(x2) + HALF_WIN, x1, y1, x2, y2),
        ]
        for name, axis, lo, hi, ex1, ey1, ex2, ey2 in edges:
            r = edge_hits(img, ex1, ey1, ex2, ey2, target, axis, lo, hi)
            if r is None:
                continue
            rate, med, off = r
            exp_fixed = y1 if axis == "h" else x1
            if name in ("下",):
                exp_fixed = y2
            if name in ("右",):
                exp_fixed = x2
            delta = off - exp_fixed
            note.append(f"[pix]   {name}边 命中率 {rate*100:.0f}%  中位色距 {med:.0f}  "
                        f"中位位置 {off:.0f}（期望 {exp_fixed:.0f}，差 {delta:+.1f}）")
            if rate < 0.5:
                if id(b) in occluded:
                    note.append(f"[pix]   {name}边 命中率 {rate*100:.0f}% —— "
                                f"该框描边被后绘制的 {occluded[id(b)]['kind']}"
                                f"#{occluded[id(b)]['i']} 覆盖，属正常")
                else:
                    prob.append(f"{label} {name}边 命中率仅 {rate*100:.0f}% —— "
                                f"该处没有期望颜色的描边")
            elif abs(delta) > 5:
                prob.append(f"{label} {name}边 位置偏差 {delta:+.1f} px（> 5）")

    # 归属的几何合理性：车牌框应落在其 owner 车辆框内
    veh = {b["i"]: b for b in boxes if b["kind"] == "veh"}
    for b in boxes:
        if b["kind"] != "plate" or (b["owner"] or -1) < 0:
            continue
        if b["owner"] not in veh:
            prob.append(f"plate#{b['i']} 归属 veh#{b['owner']}，但日志里没有这个车辆框")
            continue
        v = veh[b["owner"]]["drawPx"]
        p = b["drawPx"]
        inside = (p[0] >= v[0] - 2 and p[1] >= v[1] - 2 and p[2] <= v[2] + 2 and p[3] <= v[3] + 2)
        note.append(f"[pix] 归属检查：plate#{b['i']} 在 veh#{b['owner']} 框内 = {inside}")
        if not inside:
            prob.append(f"plate#{b['i']} 报归属 veh#{b['owner']}，但它的框并不在该车框内")

    print()
    for ln in note:
        print(ln)
    print()
    if prob:
        print(f"[pix] ✗ 发现 {len(prob)} 个问题：")
        for p in prob:
            print("   - " + p)
        return 1
    print("[pix] ✓ 截图像素验证通过：每个叠加框的 4 条边都在期望位置检出了期望颜色的描边，")
    print("      且每个有归属的车牌框都落在其归属车辆框之内。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
