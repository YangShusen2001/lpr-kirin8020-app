#!/usr/bin/env python3
"""make_nv21_fixture.py —— 把一张 PNG 编成 NV21，供相机页「喂图自证」用。

## 为什么需要它

T5 要验的是**相机页的呈现层**（叠加框 / 结果区 / 归属 / 计时），而不是算法
（那由 T2/T3/T4 负责）。但真机上让相机**碰巧拍到车牌**不是可复现的验证 ——
设备朝向、光照、焦距都不可控。

于是把 `veh320.png` 预编码成 NV21 塞进 rawfile，相机页读它、调
`cameraFrameAsync` —— 与真实相机帧**同格式、同入口、同后处理**。这样被验证的
就是相机页本身，而不是一条旁路。

## 编码必须与 native 的解码同源

native 侧 `Nv21PixelToRgb`（lpr_pipeline.cpp）用的是**整数 BT.601 有限范围**：
    C = Y-16, D = U-128, E = V-128
    R = (298C + 409E + 128) >> 8
    G = (298C - 100D - 208E + 128) >> 8
    B = (298C + 516D + 128) >> 8
本脚本的编码用 libyuv 的标准整数逆变换，**反解直接 import native 同源的
`_yuv`**（来自 tools/verify_nv21_to_rgba.py），从而把「编码损失」量化出来，
而不是靠"看起来还行"。

⚠️ 编码-解码**不是无损**的：色度 2x2 子采样 + 8bit 量化。Y 通道全分辨率保留
（车牌检测主要靠亮度/纹理），但色度边界会糊。所以本脚本会报 PSNR 与最大差值，
并把「反解后的图」也落盘，便于在它上面跑 PC 参考检测 —— 若车框仍能检出，
说明保真度足够，设备侧再检不出就是别的原因。

## 用法

    python _veh/make_nv21_fixture.py \
        --png LprDemo/entry/src/main/resources/rawfile/assets/veh320/veh320.png \
        --out-nv21 LprDemo/entry/src/main/resources/rawfile/assets/veh320/veh320.nv21 \
        --out-meta _veh/veh320_nv21_meta.json \
        --out-decoded _veh/veh320_nv21_decoded.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

# native 同源的解码器 —— 不另写一份，否则"同源"就成了口号。
from verify_nv21_to_rgba import _yuv  # noqa: E402


def rgb_to_yuv601(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """libyuv 的标准整数 BT.601 有限范围编码（与 _yuv 的逆变换配对）。"""
    ri = r.astype(np.int32)
    gi = g.astype(np.int32)
    bi = b.astype(np.int32)
    y = ((66 * ri + 129 * gi + 25 * bi + 128) >> 8) + 16
    u = ((-38 * ri - 74 * gi + 112 * bi + 128) >> 8) + 128
    v = ((112 * ri - 94 * gi - 18 * bi + 128) >> 8) + 128
    return (np.clip(y, 0, 255).astype(np.uint8),
            np.clip(u, 0, 255).astype(np.uint8),
            np.clip(v, 0, 255).astype(np.uint8))


def encode_nv21(rgb: np.ndarray, stride: int) -> bytes:
    """RGB(H,W,3) -> NV21 字节串。stride 必须为偶数（色度按 2 采样）。"""
    h, w, _ = rgb.shape
    if stride < w or stride % 2 != 0:
        raise SystemExit(f"stride 必须 >= width 且为偶数，收到 stride={stride} w={w}")
    y, u, v = rgb_to_yuv601(rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])

    buf = np.zeros((stride * h + stride * (h // 2)), dtype=np.uint8)
    # Y plane：每行左侧 w 字节有效，右侧 padding 补 0
    for row in range(h):
        buf[row * stride:row * stride + w] = y[row]
    # VU plane：NV21 = V 在前、U 在后，色度 2x2 取平均后按 2 采样
    uu = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    vv = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    uu = np.clip(np.round(uu), 0, 255).astype(np.uint8)
    vv = np.clip(np.round(vv), 0, 255).astype(np.uint8)
    base = stride * h
    for row in range(h // 2):
        off = base + row * stride
        buf[off:off + w:2] = vv[row]   # V 在偶下标
        buf[off + 1:off + w:2] = uu[row]  # U 在奇下标
    return buf.tobytes()


def decode_nv21_via_native(nv21: bytes, w: int, h: int, stride: int) -> np.ndarray:
    """用 native 同一份算术反解，返回 RGB(H,W,3) uint8。"""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    y_plane = nv21
    uv_base = stride * h
    for row in range(h):
        for col in range(w):
            Y = nv21[row * stride + col]
            off = uv_base + (row // 2) * stride + (col // 2) * 2
            V = nv21[off]
            U = nv21[off + 1]
            r, g, b = _yuv(Y, U, V)
            rgb[row, col, 0] = r
            rgb[row, col, 1] = g
            rgb[row, col, 2] = b
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", required=True, help="源图（RGB，尺寸任意偶数）")
    ap.add_argument("--out-nv21", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--out-decoded", default="", help="反解回来的 PNG（可选，用于 PC 复检）")
    ap.add_argument("--stride", type=int, default=0, help="0 = 用 width")
    a = ap.parse_args()

    img = Image.open(a.png).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    h, w, _ = rgb.shape
    if h % 2 or w % 2:
        raise SystemExit(f"宽高必须为偶数（色度 2x2 采样），收到 {w}x{h}")
    stride = a.stride if a.stride > 0 else w

    nv21 = encode_nv21(rgb, stride)
    expect_len = stride * h + stride * (h // 2)
    if len(nv21) != expect_len:
        raise SystemExit(f"NV21 长度不符：{len(nv21)} != {expect_len}")

    with open(a.out_nv21, "wb") as f:
        f.write(nv21)

    dec = decode_nv21_via_native(nv21, w, h, stride)
    diff = np.abs(dec.astype(np.int32) - rgb.astype(np.int32))
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(255.0 * 255.0 / mse))

    # 亮度误差单列：车牌检测主要吃亮度，色度糊掉对它影响小得多。
    lum_src = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    lum_dec = (0.299 * dec[:, :, 0] + 0.587 * dec[:, :, 1] + 0.114 * dec[:, :, 2])
    lum_diff = np.abs(lum_src - lum_dec)

    if a.out_decoded:
        Image.fromarray(dec, "RGB").save(a.out_decoded)

    meta = {
        "source": os.path.basename(a.png),
        "width": w, "height": h, "stride": stride, "rotation": 0,
        "nv21Bytes": len(nv21),
        "maxAbsDiff": int(diff.max()),
        "meanAbsDiff": round(float(diff.mean()), 4),
        "psnrDb": round(psnr, 2),
        "lumMaxDiff": round(float(lum_diff.max()), 3),
        "lumMeanDiff": round(float(lum_diff.mean()), 4),
        "decodedPng": os.path.basename(a.out_decoded) if a.out_decoded else "",
    }
    with open(a.out_meta, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[nv21] {w}x{h} stride={stride} -> {len(nv21)} 字节")
    print(f"[nv21] 最大通道差 {meta['maxAbsDiff']}  平均 {meta['meanAbsDiff']}  PSNR {meta['psnrDb']} dB")
    print(f"[nv21] 亮度最大差 {meta['lumMaxDiff']}  平均 {meta['lumMeanDiff']}")
    print(f"[nv21] 落盘 {a.out_nv21}")
    print(f"[nv21] meta {a.out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
