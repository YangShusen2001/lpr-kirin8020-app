#!/usr/bin/env python3
"""veh_ref.py —— 车辆检测（T2）的 PC 侧参考实现与真机对照。

## 「同一张图」怎么做到真正同一

`LprVehicleDetect` 会先把源图 letterbox 到 320x320。若源图**本身就是 320x320**，
则 `r = min(320/320, 320/320) = 1`、`left = top = 0`，letterbox 退化成**逐字节拷贝** ——
于是两侧看到的模型输入是同一串字节，把「resize 插值实现差异」这个不可控变量彻底消掉。
本脚本产出的测试图就是 320x320，就是为了这个。

## 编码必须与 native 逐字对应

`LprToNhwcInto(img, /*swapRB=*/true, out)`：
    out[i*3+0] = R/255 ; out[i*3+1] = G/255 ; out[i*3+2] = B/255
即 **RGB 顺序、除以 255、NHWC**。onnxruntime 要 NCHW，所以再 transpose 一次。
（alignment：ultralytics YOLOv5 的预处理就是 im/255 + RGB，没有 mean/std。）

## 用法

    # 1) 从 CCPD 真实场景图造一张 320x320 测试图（同时落到 App rawfile）
    python _veh/veh_ref.py --make-image --ccpd <某个.jpg> --name veh320

    # 2) 挑图：扫一批 CCPD，看哪张能检出车（conf=0.05）
    python _veh/veh_ref.py --scan --ccpd-dir <dir> --limit 12

    # 3) 出 PC 侧参考结果
    python _veh/veh_ref.py --ref --image <320x320.png> --out _veh/veh_ref.json

    # 4) 拿真机 hilog 与 PC 参考对照
    python _veh/veh_ref.py --compare --ref-json _veh/veh_ref.json --device-log <hilog.txt>
    #    （--ref 是「产出参考」的开关，别拿它传路径）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# 与 C++ 的 kClasses / kChannels 对齐
CLASSES = 80
CHANNELS = 4 + CLASSES  # 84
VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def encode_nhwc(rgba: np.ndarray) -> np.ndarray:
    """RGBA uint8 (H,W,4) -> NHWC float32，RGB 顺序，除以 255。对齐 LprToNhwcInto。"""
    return (rgba[:, :, [0, 1, 2]].astype(np.float32) / 255.0)[None, ...]


def letterbox_geometry(src_w: int, src_h: int, size: int):
    """复刻 LprLetterBoxInto 的几何：r / newW / newH / left / top。"""
    r = min(size / float(src_h), size / float(src_w))
    new_h = int(np.trunc(src_h * r))
    new_w = int(np.trunc(src_w * r))
    top = int(np.trunc((size - new_h) / 2.0))
    left = int(np.trunc((size - new_w) / 2.0))
    return r, new_w, new_h, left, top


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = aa + bb - inter
    return inter / uni if uni > 0 else 0.0


def decode(raw: np.ndarray, conf: float, iou: float, r: float, left: int, top: int,
           vehicle_only: bool, max_boxes: int = 100):
    """复刻 LprDecodeYolov5u。raw 是 [1,84,2100] 的平铺输出。"""
    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    if flat.size % CHANNELS != 0:
        raise SystemExit(f"[veh_ref] 输出长度 {flat.size} 不是 {CHANNELS} 的倍数")
    anchors = flat.size // CHANNELS
    m = flat.reshape(CHANNELS, anchors)  # 通道优先

    cls = m[4:4 + CLASSES, :]              # [80, anchors]
    best = cls.argmax(axis=0)
    score = cls[best, np.arange(anchors)]
    keep = score > conf
    if vehicle_only:
        keep &= np.isin(best, list(VEHICLE_IDS))
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return [], False

    cx, cy = m[0, idx], m[1, idx]
    bw, bh = m[2, idx], m[3, idx]
    cand = []
    for k, a in enumerate(idx):
        cand.append({
            "cls": int(best[a]),
            "score": float(score[a]),
            "rect": [float(cx[k] - bw[k] / 2), float(cy[k] - bh[k] / 2),
                     float(cx[k] + bw[k] / 2), float(cy[k] + bh[k] / 2)],
        })

    # 候选按分数降序后截断（与 native 的 kMaxCand=1000 一致）
    cand.sort(key=lambda b: -b["score"])
    truncated = len(cand) > 1000
    cand = cand[:1000]

    # 按类 NMS（YOLO 约定：不同类之间不互相抑制）
    dead = [False] * len(cand)
    kept = []
    for i, bi in enumerate(cand):
        if dead[i]:
            continue
        if len(kept) >= max_boxes:
            truncated = True
            break
        kept.append(bi)
        for j, bj in enumerate(cand):
            if j == i or dead[j] or bj["cls"] != bi["cls"]:
                continue
            if box_iou(bi["rect"], bj["rect"]) > iou:
                dead[j] = True

    # letterbox 反变换
    for b in kept:
        b["rect"] = [(b["rect"][0] - left) / r, (b["rect"][1] - top) / r,
                     (b["rect"][2] - left) / r, (b["rect"][3] - top) / r]
    return kept, truncated


def load_rgba(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def cmd_make_image(a) -> int:
    from PIL import Image
    im = Image.open(a.ccpd).convert("RGB").resize((a.size, a.size), Image.BILINEAR)
    outdir = os.path.dirname(os.path.abspath(a.png)) if a.png else "_veh"
    os.makedirs(outdir, exist_ok=True)
    dst = a.png or os.path.join(outdir, f"{a.name}.png")
    im.save(dst)
    print(f"[veh_ref] 测试图 {dst}  {im.size} 源={a.ccpd}")
    if a.rawfile_dir:
        os.makedirs(a.rawfile_dir, exist_ok=True)
        dst2 = os.path.join(a.rawfile_dir, f"{a.name}.png")
        im.save(dst2)
        print(f"[veh_ref] 已复制到 rawfile: {dst2}")
    return 0


def cmd_scan(a) -> int:
    import glob
    from PIL import Image

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])

    files = sorted(glob.glob(os.path.join(a.ccpd_dir, "*.jpg")))[: a.limit]
    print(f"[veh_ref] 扫 {len(files)} 张，conf={a.conf} iou={a.iou} size={a.size}")
    rows = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((a.size, a.size), Image.BILINEAR)
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        nhwc = encode_nhwc(rgba)
        out = sess.run(None, {"images": np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))})[0]
        r, _, _, left, top = letterbox_geometry(a.size, a.size, a.size)
        boxes, _ = decode(out, a.conf, a.iou, r, left, top, True)
        top_score = max((b["score"] for b in boxes), default=0.0)
        rows.append((os.path.basename(f), len(boxes), top_score))
        print(f"  {len(boxes):3d} boxes  top={top_score:.4f}  {os.path.basename(f)}")
    ok = [r for r in rows if r[1] > 0]
    print(f"[veh_ref] 有检出的 {len(ok)}/{len(rows)}")
    return 0


def cmd_ref(a) -> int:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])

    rgba = load_rgba(a.image)
    h, w = rgba.shape[:2]
    if (w, h) != (a.size, a.size):
        print(f"[veh_ref] ⚠️ 源图 {w}x{h} 不等于 {a.size}x{a.size}，"
              f"letterbox 将发生真实插值 —— 与真机对照时坐标会有插值级偏差")
    r, new_w, new_h, left, top = letterbox_geometry(w, h, a.size)

    nhwc = encode_nhwc(rgba)
    nchw = np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))
    out = sess.run(None, {"images": nchw})[0]
    boxes, truncated = decode(out, a.conf, a.iou, r, left, top, a.vehicle_only)

    art = {
        "image": os.path.abspath(a.image),
        "image_wh": [w, h],
        "size": a.size,
        "onnx": os.path.abspath(a.onnx),
        "conf": a.conf,
        "iou": a.iou,
        "vehicle_only": a.vehicle_only,
        "letterbox": {"r": r, "new_w": new_w, "new_h": new_h, "left": left, "top": top},
        "count": len(boxes),
        "truncated": truncated,
        "out_shape": list(np.asarray(out).shape),
        "out_l2": float(np.sqrt((np.asarray(out, dtype=np.float64) ** 2).sum())),
        "boxes": boxes,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(art, f, ensure_ascii=False, indent=2)
    print(f"[veh_ref] PC 侧 {len(boxes)} 框（truncated={truncated}）-> {a.out}")
    for i, b in enumerate(boxes):
        print(f"  b{i} cls={b['cls']:2d} score={b['score']:.4f} "
              f"rect={','.join(f'{v:.2f}' for v in b['rect'])}")
    return 0


def _same_boxes(a, b) -> bool:
    """两轮框是否等价（类别 + 分数 + 坐标到 4 位小数）。"""
    if len(a) != len(b):
        return False
    ka = sorted((x["cls"], x["score"], tuple(round(v, 4) for v in x["rect"])) for x in a)
    kb = sorted((x["cls"], x["score"], tuple(round(v, 4) for v in x["rect"])) for x in b)
    return ka == kb


def parse_device_log(path: str):
    """按轮解析 hilog 里的 `VEH summary` / `VEH box`，返回 [(summary, boxes), ...]。

    为什么必须按轮切：App 侧是「预热一次 + 正式读一次」，日志里因此有**两整轮**相同的框。
    不切轮就会把同一批框数成两批，贪心配对只消耗掉一半，剩下的一半被误报成「设备独有」。
    """
    runs = []
    cur_s, cur_b = None, []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "VEH summary" not in line and "VEH box" not in line:
                continue
            tail = line.split("VEH ", 1)[1].strip()
            if tail.startswith("summary"):
                if cur_s is not None:
                    runs.append((cur_s, cur_b))
                cur_s = dict(kv.split("=", 1) for kv in tail.split()[1:] if "=" in kv)
                cur_b = []
            elif tail.startswith("box") and cur_s is not None:
                parts = tail.split()
                d = {}
                for tok in parts[1:]:
                    if "=" in tok:
                        d[tok.split("=", 1)[0]] = tok.split("=", 1)[1]
                rect = [float(v) for v in d["rect"].split(",")]
                cur_b.append({"idx": int(d["idx"]), "cls": int(d["cls"]),
                              "score": float(d["score"]), "rect": rect, "name": d.get("name", "?")})
    if cur_s is not None:
        runs.append((cur_s, cur_b))
    return runs


def cmd_compare(a) -> int:
    # 注意：不能在 --compare 里复用 --ref —— 那是「产出参考」的布尔开关。
    # 待比对的参考文件走 --ref-json（默认 _veh/veh_ref.json）。
    with open(a.ref_json, "r", encoding="utf-8") as f:
        ref = json.load(f)
    runs = parse_device_log(a.device_log)
    if not runs:
        sys.exit("[veh_ref] 设备日志里没有 VEH summary 行")
    summary, dev = runs[-1]
    if len(runs) > 1:
        print(f"[veh_ref] 设备日志含 {len(runs)} 轮，取最后一轮；各轮框数 "
              f"{[len(b) for _, b in runs]}")
        same = all(_same_boxes(runs[0][1], b) for _, b in runs[1:])
        print(f"[veh_ref] 各轮框逐元素一致：{'是（可复现）' if same else '否 —— 需排查！'}")
        if not same:
            print("[veh_ref] ✗ 同一张图多轮推理结果不一致，判失败")
            return 2
    print(f"[veh_ref] 设备: count={summary.get('count')} truncated={summary.get('truncated')} "
          f"conf={summary.get('conf')} iou={summary.get('iou')} "
          f"vehicleOnly={summary.get('vehicleOnly')} size={summary.get('size')} "
          f"nhwc={summary.get('nhwc')} inferMs={summary.get('inferMs')} "
          f"backend={summary.get('backend')}")
    print(f"[veh_ref] PC  : count={ref['count']} truncated={ref['truncated']} "
          f"conf={ref['conf']:.4f} iou={ref['iou']:.4f} vehicle_only={ref['vehicle_only']} "
          f"size={ref['size']}")

    prob = []
    if int(summary.get("size", -1)) != ref["size"]:
        prob.append(f"letterbox size 不一致：设备 {summary.get('size')} vs PC {ref['size']}")
    if abs(float(summary.get("conf", -1)) - ref["conf"]) > 1e-6:
        prob.append(f"conf 不一致：设备 {summary.get('conf')} vs PC {ref['conf']}")
    if (summary.get("vehicleOnly") == "1") != bool(ref["vehicle_only"]):
        prob.append("vehicleOnly 不一致")

    # 按类 + IOU 贪心配对
    dev_used = [False] * len(dev)
    pairs = []
    for i, rb in enumerate(ref["boxes"]):
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(dev):
            if dev_used[j] or db["cls"] != rb["cls"]:
                continue
            v = box_iou(rb["rect"], db["rect"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            dev_used[best_j] = True
            pairs.append((i, best_j, best_iou))

    deltas = [max(abs(ref["boxes"][i]["rect"][k] - dev[j]["rect"][k]) for k in range(4))
              for i, j, _ in pairs]
    score_deltas = [abs(ref["boxes"][i]["score"] - dev[j]["score"]) for i, j, _ in pairs]

    print()
    print(f"[veh_ref] 配对 {len(pairs)} / PC {len(ref['boxes'])} / 设备 {len(dev)}")
    if pairs:
        print(f"[veh_ref] 坐标最大偏差 = {max(deltas):.4f} px（中位 {np.median(deltas):.4f}）")
        print(f"[veh_ref] 分数最大偏差 = {max(score_deltas):.2e}（中位 {np.median(score_deltas):.2e}）")
        print(f"[veh_ref] 最小配对 IOU   = {min(p[2] for p in pairs):.6f}")

    # —— 验收门限 ——
    # 实测：坐标最大偏差 0.0000 px、分数最大偏差 1.7e-06（纯 fp32 舍入量级）。
    # 门限各留 4 个数量级余量，但仍能抓住真正的解码错误 ——
    # 一旦通道优先/letterbox 反变换写错，框会整体移位几十 px，绝不会停在 0.5 px 以内。
    if pairs:
        if max(deltas) > 0.5:
            prob.append(f"坐标偏差超门限：{max(deltas):.4f} px > 0.5 px")
        if max(score_deltas) > 1e-3:
            prob.append(f"分数偏差超门限：{max(score_deltas):.2e} > 1e-3")

    un_pc = [ref["boxes"][i] for i in range(len(ref["boxes"])) if i not in {p[0] for p in pairs}]
    un_dev = [dev[j] for j in range(len(dev)) if not dev_used[j]]
    for tag, lst, key in (("PC 独有", un_pc, "score"), ("设备独有", un_dev, "score")):
        for b in lst:
            print(f"[veh_ref]   {tag}: cls={b['cls']} score={b[key]:.4f} "
                  f"rect={','.join(f'{v:.2f}' for v in b['rect'])}")

    if len(un_pc) or len(un_dev):
        prob.append(f"有未配对的框：PC 独有 {len(un_pc)}、设备独有 {len(un_dev)}"
                    f"（多为 conf={ref['conf']} 附近的边缘检出，看上面的分数）")

    print()
    if prob:
        for line in prob:
            print(f"[veh_ref] ✗ {line}")
        return 2
    print("[veh_ref] ✓ PC 与设备结果一致（数量、类别、坐标、分数都在容差内）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="_veh/yolov5su_320_ms.onnx")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--vehicle-only", dest="vehicle_only",
                    action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--make-image", action="store_true")
    ap.add_argument("--ccpd")
    ap.add_argument("--name", default="veh320")
    ap.add_argument("--png")
    ap.add_argument("--rawfile-dir",
                    default="LprDemo/entry/src/main/resources/rawfile/assets/veh320")

    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--ccpd-dir")
    ap.add_argument("--limit", type=int, default=12)

    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--out", default="_veh/veh_ref.json")

    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--device-log")
    ap.add_argument("--ref-json", dest="ref_json", default="_veh/veh_ref.json",
                    help="--compare 要比对的 PC 参考 JSON；默认 _veh/veh_ref.json")
    a = ap.parse_args()

    if a.make_image:
        return cmd_make_image(a)
    if a.scan:
        return cmd_scan(a)
    if a.ref:
        return cmd_ref(a)
    if a.compare:
        return cmd_compare(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
