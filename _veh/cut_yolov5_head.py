#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cut_yolov5_head.py —— 把原版 YOLOv5 v7 的 Detect 头裁到最后一个 rank-4 张量（T8）。

## 为什么必须裁

原版 v7 裸导出的 ONNX 张这样（每层 anchor 都一样）：

    Conv(255ch, rank-4)
      → Reshape            rank-5
      → Transpose perm=[0,1,3,4,2]   <-- 硬门2 不过（只允许 [0,1,3,2]）
      → Sigmoid → Split → xy/wh 解码
      → Concat → Reshape_1 → cat → output0

`tools/scan_onnx.py` 的判词：

    [scan] ✗ 硬门1 不过：rank>=5 的常量 0 个、张量 3 个
    [scan] Transpose 共 3 个，其中 perm != [0,1,3,2] 的 3 个
             /model.24/Transpose perm=[0, 1, 3, 4, 2]   <-- NOT SUPPORTED

**关键**：sigmoid 与 anchor-grid 解码全在 **rank-5** 张量上跑，所以不能只删 Transpose，
必须把整段解码搬出计算图 —— 切在 `/model.24/m.{i}/Conv_output_0`（255 通道，rank-4）。
这正是 kirin-npu-model-porting 技能第二节「切在最后一个 rank-4 张量上」，
也对应技能参考数据表里「裁掉 decode 到 rank-4 → NPU，4.95 ms」那一行。

裁完输出三个张量（320 输入）：(1,255,40,40) / (1,255,20,20) / (1,255,10,10)。
sigmoid 与 grid/anchor 解码改由 Host（C++ / numpy）做 —— 与 ncnn/mnn 的 YOLOv5 部署一致。

## 切点怎么找到的（不靠猜）

沿 `/model.24/Transpose*` 的输入反查 producer，得到
`Transpose ← Reshape ← Conv`，所以 rank-4 的边界就是 Conv 的输出。

## 验证（唯一判据）

拿**原图**（含解码）跑 onnxruntime 得到 `output0`，
再拿**裁切图**跑 ORT，用**从原图里抽取的**解码常量在 numpy 里复现解码，逐元素对比。
数值一致才说明裁切没改变语义 —— 光看 shape 对不算数。

用法：
    python _veh/cut_yolov5_head.py --src _veh/yolov5s_v7_320.onnx --out _veh/yolov5s_v7_320_npu.onnx
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from onnx import ModelProto, TensorProto, helper

# 三个切点：Detect 头每层的 Conv 输出（rank-4，255 = 3 anchor × 85）
TARGETS = [
    "/model.24/m.0/Conv_output_0",
    "/model.24/m.1/Conv_output_0",
    "/model.24/m.2/Conv_output_0",
]


def parse(path):
    m = ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m


def cut_graph(src_path, out_path):
    m = parse(src_path)
    g = m.graph

    init_names = {i.name for i in g.initializer}
    input_names = {i.name for i in g.input}
    producer = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            producer[o] = i

    # ---- 反向可达 ----
    keep, stack, missing = set(), list(TARGETS), []
    while stack:
        t = stack.pop()
        if t in init_names or t in input_names:      # 常量/输入都是叶子
            continue
        i = producer.get(t)
        if i is None:
            missing.append(t)
            continue
        if i in keep:
            continue
        keep.add(i)
        stack.extend(g.node[i].input)
    if missing:
        raise SystemExit(f"[cut] 反向可达漏了 producer：{sorted(set(missing))[:5]}")

    kept = [n for i, n in enumerate(g.node) if i in keep]
    print(f"[cut] 节点 {len(g.node)} → {len(kept)}（删掉 {len(g.node) - len(kept)} 个解码节点）")

    # protobuf repeated field 不支持切片赋值 → del + extend
    del g.node[:]
    g.node.extend(kept)

    # ---- 输出改成三个切点 ----
    del g.output[:]
    for t in TARGETS:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))

    # ---- 常量按「仍被引用」过滤（rank-5 常量就在这里离开） ----
    used = set()
    for n in g.node:
        used.update(n.input)
    ki = [x for x in g.initializer if x.name in used]
    dropped = len(g.initializer) - len(ki)
    del g.initializer[:]
    g.initializer.extend(ki)

    g.value_info.clear()   # 残留 value_info 会引用已删张量

    with open(out_path, "wb") as f:
        f.write(m.SerializeToString())
    print(f"[cut] 常量 {len(ki) + dropped} → {len(ki)}（丢 {dropped}）")
    print(f"[cut] 输出 = {TARGETS}")
    return out_path


def extract_decode_consts(src_path):
    """从**原图**里抽出解码常量，供 numpy 复现解码 —— 避免手抄 anchor 抄错。

    ⚠ 不要自己解 `raw_data`（dtype/字节宽因节点而异，实测踩过
      `cannot reshape array of size 10 into shape (5,)`）——统一用 `numpy_helper.to_array`。

    也不能靠名字下标算层级：`/model.24/Constant_output_0`（view 用的 shape 常量）
    没有数字后缀，会污染下标。改**按形状 + 节点出现顺序**认：

      xy = (s·2 + C2) · C3         C2 = grid - 0.5，shape (1,3,ny,nx,2)
                                   C3 = stride，     shape (1,1,1,1,1)
      wh = (s·2)² · C6             C6 = anchor_grid，shape (1,3,1,1,2)
    """
    from onnx import numpy_helper

    m = parse(src_path)
    g = m.graph
    ordered = []
    for n in g.node:
        if n.op_type == "Constant" and n.name.startswith("/model.24/Constant"):
            for a in n.attribute:
                if a.name == "value":
                    ordered.append((n.name, numpy_helper.to_array(a.t)))
    print(f"[cut] 原图 /model.24/Constant* 共 {len(ordered)} 个")

    # 每层的常量序列（实测 dump 出来的）：
    #   [view_shape(int64,5), 2.0, grid-0.5, stride, 2.0, 2.0, anchor_grid, reshape_shape(int64,3)]
    # 注意 torch 导出器把 anchor_grid **展开成完整的 (1,3,ny,nx,2)**（不是 (1,3,1,1,2)），
    # 所以每层有**两个**同形状 rank-5 常量，严格交替：grid 在前、anchor 在后。
    # stride 是标量（shape=()），值 8/16/32。
    rank5 = [v for _, v in ordered
             if v.ndim == 5 and v.shape[0] == 1 and v.shape[1] == 3 and v.shape[4] == 2]
    grids = rank5[0::2]
    anchors = rank5[1::2]
    # 标量常量里既有 2.0（Mul/Pow 的指数与系数）也有 stride，按取值筛
    strides = [v for _, v in ordered
               if v.ndim == 0 and float(v) in (8.0, 16.0, 32.0)]

    print(f"[cut] rank-5 常量 {len(rank5)} 个 → grid×{len(grids)} anchor×{len(anchors)} stride×{len(strides)}")
    print(f"[cut] strides = {[float(s) for s in strides]}")
    # anchor 被导出器 tile 成了 (1,3,ny,nx,2)，同一 anchor 在每个 (y,x) 上取值相同
    # → 取 [:, 0, :] 还原成每 anchor 的 (w, h) 表（C++ 侧需要的正是这张表）
    print(f"[cut] anchors = {[a.reshape(3, -1, 2)[:, 0, :].tolist() for a in anchors]}")
    if not (len(grids) == len(anchors) == len(strides) == 3):
        raise SystemExit("[cut] 常量识别数量不为 3，认法失效，别硬跑")
    return grids, anchors, strides


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_veh/yolov5s_v7_320.onnx")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    if not os.path.isfile(src):
        print(f"[cut] 找不到 {src}", file=sys.stderr)
        return 2
    out = os.path.abspath(a.out or src.replace(".onnx", "_npu.onnx"))

    cut_graph(src, out)

    if a.no_verify:
        return 0

    # ---------------- 数值验证：原图 vs 裁切图 ----------------
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 320, 320), dtype=np.float32) * 0.5
    # 用随机输入而不是全 0：全 0 输入会让 sigmoid 全部落到 0.5，掩盖差异

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess_src = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    sess_cut = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])

    ref = sess_src.run(None, {"images": x})
    ref = np.asarray(ref[0], dtype=np.float32)          # (1, 6300, 85)
    print(f"[cut] 原图 output0 = {ref.shape}")

    outs = sess_cut.run(None, {"images": x})
    print(f"[cut] 裁切图输出 = {[tuple(o.shape) for o in outs]}")

    grids, anchors, strides = extract_decode_consts(src)

    # 在 numpy 里复现 yolov5 的 inplace 解码：
    #   p = conv_out.view(bs, na, no, ny, nx)
    #   y = cat((s[...,0:2]*2 + grid - 0.5) * stride, (s[...,2:4]*2)**2 * anchor, s[...,4:])
    #   → reshape(bs, na*ny*nx, no)
    levels = []
    for li, o in enumerate(outs):
        o = np.asarray(o, dtype=np.float32)
        _, c, ny, nx = o.shape
        na, no = 3, c // 3
        p = o.reshape(1, na, no, ny, nx)
        # 原图是 view(bs,na,no,ny,nx) → permute(0,1,3,4,2) → 即把 no 轴移到末尾，
        # 后面所有解码都作用在 (1,na,ny,nx,no) 上。这里显式复现这一步，
        # 否则切片拿到的会是 (1,3,2,ny,nx) 而不是 (1,3,ny,nx,2)。
        s = (1.0 / (1.0 + np.exp(-p))).transpose(0, 1, 3, 4, 2)

        grid, anchor, stride = grids[li], anchors[li], strides[li]
        if tuple(grid.shape) != (1, 3, ny, nx, 2):
            print(f"[cut] ✗ 第 {li} 层 grid shape {grid.shape} 与特征图 {(ny, nx)} 不匹配",
                  file=sys.stderr)
            return 3

        xy = (s[..., 0:2] * 2 + grid) * stride
        wh = (s[..., 2:4] * 2) ** 2 * anchor
        y = np.concatenate([xy, wh, s[..., 4:]], axis=4)
        levels.append(y.reshape(1, na * ny * nx, no))

    mine = np.concatenate(levels, axis=1)
    print(f"[cut] numpy 复现解码 = {mine.shape}")

    if mine.shape != ref.shape:
        print(f"[cut] ✗ shape 不一致 {mine.shape} vs {ref.shape}", file=sys.stderr)
        return 3

    diff = np.abs(mine - ref)
    scale = max(1e-6, float(np.abs(ref).mean()))
    rel = float(diff.max()) / scale
    print(f"[cut] maxAbsDiff = {diff.max():.6e}  meanAbsDiff = {diff.mean():.6e}")
    print(f"[cut] maxAbsDiff / mean|ref| = {rel:.3e}   (mean|ref| = {scale:.3e})")
    if rel > 1e-4:
        print("[cut] ✗ 数值不一致，裁切改变了语义", file=sys.stderr)
        return 3
    print("[cut] ✓ 裁切前后逐元素一致（相对误差 < 1e-4）")
    print(f"[cut] 产物 = {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
