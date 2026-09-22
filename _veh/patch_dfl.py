#!/usr/bin/env python3
"""patch_dfl.py —— 把 YOLOv5u(ultralytics) 的 DFL 子图改写成 MindSpore Lite 能吃的等价形式。

为什么必须做
============
`yolov5su_320.onnx` 的 DFL（Distribution Focal Loss）头里有两个 Transpose：

    /model.24/dfl/Transpose     perm=[0,3,1,2]
    /model.24/dfl/Transpose_1   perm=[0,3,2,1]

它们让 MindSpore Lite 的 `InferShapeByNNACL` 在
`/model.24/dfl/conv/Conv` 上直接失败：

    InferShapeByNNACL for op: /model.24/dfl/conv/Conv failed.
    Transform meta graph failed! ret = -500

于是**整个模型转不出 `.ms`**（不只是上不了 NPU —— 连 CPU 的 .ms 也转不出来）。
`tools/scan_onnx.py` 也独立印证了同一处（perm != [0,1,3,2] 的 Transpose 恰好是这两个）。

DFL 到底在算什么（已用本机实测复核，见文件末尾「实测记录」）
==========================================================
    Concat(box 分支, 64ch) -> Reshape [1,4,16,2100]      # 4 条边 × 16 个 bin × 2100 anchor
    Transpose[0,3,1,2]     -> [1,2100,4,16]
    Softmax                -> 对最后那 16 个 bin 归一化
    Transpose_1[0,3,2,1]   -> [1,16,4,2100]
    Conv 1x1 (weight=arange(16)) -> [1,1,4,2100]
    Reshape_1              -> [1,4,2100]

所以 DFL 等价于「**对 16 个 bin 做 softmax，再按 0..15 加权求和**」。
两个 Transpose 只是为了让 1x1 Conv 的通道维落在正确位置而做的搬运，本身没有数学含义。

改写（最小差分）
================
⚠️⚠️ 坑中之坑：**ONNX opset ≤ 12 的 Softmax 是「coerced-2D」语义**，不是「沿给定轴归一化」。
它把 `[0,axis)` 压成一维、`[axis,rank)` 压成另一维，然后对**后一维**做 softmax。
所以在 `[1,4,16,2100]` 上写 `Softmax(axis=2)`，实际算的是「4 行 × 33600 的 softmax」，
跟「对 16 个 bin 归一化」毫无关系。本机实测：与正确结果差 1.0（=完全错）。
**这也正是 ultralytics 当初非要插 `Transpose[0,3,1,2]` 的原因** —— opset 12 下只有把待
归一化的轴搬到最后一维才写得出 softmax。（opset 13 才改成沿轴语义。）

于是不用 Softmax 算子，改用等价的「指数 / 归一化」手工形式：

    加  ReduceMax(axes=[2], keepdims=1) -> 行的最大值 [1,4,1,2100]
    加  Sub                            -> [1,4,16,2100]
    加  Exp                            -> [1,4,16,2100]
    加  Mul(bins=arange(16))           -> [1,4,16,2100]
    加  ReduceSum(axes=[2], keepdims=0) -> 分子 [1,4,2100]
    加  ReduceSum(axes=[2], keepdims=0) -> 分母 [1,4,2100]
    加  Div(分子, 分母)                 -> [1,4,2100]   == sum_k softmax_k * k
    接  /model.24/dfl/Reshape_1 的 input[0] 由 conv/Conv_output_0 改成 Div 的输出
    删  /model.24/dfl/Transpose, Softmax, Transpose_1, conv/Conv

必须减最大值（ReduceMax/Sub）：DFL 的 logits 后面没有激活函数压着，本机实测在
0..255 的随机图上**裸 Exp 直接溢出成 inf**，随后 inf/inf = NaN。别图省两个算子。

保留 `/model.24/dfl/Constant_1` 与 `/model.24/dfl/Reshape_1` 不动，于是
**最终张量名 `/model.24/dfl/Reshape_1_output_0` 不变**，它的 3 个消费者
（`/model.24/Shape`、`/model.24/Slice`、`/model.24/Slice_1`）一个都不用碰。
输出 `output0: [1,84,2100]`（已解码）**格式不变**，并且两个不合规 Transpose 都消失
—— 顺带把 T8 的 NPU perm 硬门也清了。

为什么保住 opset 12 而不是把整图升到 opset 13：opset 13 里 Squeeze/Unsqueeze/ReduceSum/Split
的 axes 都从属性变成了输入，升版要逐个改写这 290 个节点再重验一遍；而这里只需要
等价替换 4 个节点。风险差太远。

⚠️ 本图 opset = 12，`ReduceSum` 的 `axes` 是**属性**（opset 13 才变成输入）。
写脚本时踩过这个坑，非必要别改。

⚠️ 正确性不靠"读图读对了"，靠**数值自证**：脚本会用 onnxruntime 在同一输入上跑
原图与改后图，逐元素比对全部输出，判据是 `|a-b| <= 1e-6 + 1e-5*|a|`，跑 3 个随机图。
比对不过就报错退出，**不落盘**。

用法
====
  python _veh/patch_dfl.py --in _veh/yolov5su_320.onnx --out _veh/yolov5su_320_ms.onnx
  python _veh/patch_dfl.py --in ... --dump        # 只打印 DFL 子图，不改

退出码：0 = 改写且数值自证通过；2 = 数值不符；3 = 图结构与预期不符；4 = 读取失败

实测记录（2026-09-22，本机 yolov5su_320.onnx）
=============================================
  ir_version = 7        opset ai.onnx = 12
  graph.input  = images  [1,3,320,320]
  graph.output = output0 [1,84,2100]
  n_nodes = 291         n_init = 151
  /model.24/dfl/Constant   = [1, 4, 16, 2100]
  /model.24/dfl/Constant_1 = [1, 4, 2100]
  model.24.dfl.conv.weight = arange(16)，shape (1,16,1,1)  -> 加权求和前提成立
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


# ---------------------------------------------------------------- 常量

# 要被删掉的那 4 个数学节点（拓扑顺序）
DFL_DROP = [
    "/model.24/dfl/Transpose",
    "/model.24/dfl/Softmax",
    "/model.24/dfl/Transpose_1",
    "/model.24/dfl/conv/Conv",
]
# 保留但需要重接输入的节点
N_RESHAPE_1 = "/model.24/dfl/Reshape_1"

DFL_IN = "/model.24/dfl/Reshape_output_0"          # 输入：形状 [1,4,16,2100]
CONV_OUT = "/model.24/dfl/conv/Conv_output_0"      # 要被顶掉的中间张量
DFL_OUT = "/model.24/dfl/Reshape_1_output_0"       # 最终输出：形状 [1,4,2100]（名字保持不变）
W_CONV = "model.24.dfl.conv.weight"

# 新张量名（带前缀以免和原图撞名）
T_MAX = "dfl_patched/rowmax"
T_SHIFT = "dfl_patched/shifted"
T_EXP = "dfl_patched/exp"
T_BINS_INIT = "dfl_patched/bins"
T_EK = "dfl_patched/exp_times_bin"
T_NUM = "dfl_patched/numerator"
T_DEN = "dfl_patched/denominator"
T_SUM = "dfl_patched/sum"


# ---------------------------------------------------------------- 图结构检查

def dump_dfl(g) -> None:
    print("--- DFL 相关节点 ---")
    for n in g.node:
        if "/model.24/dfl/" in n.name:
            extra = ""
            for a in n.attribute:
                if a.name == "perm":
                    extra = f" perm={list(helper.get_attribute_value(a))}"
                elif a.name == "axis":
                    extra = f" axis={helper.get_attribute_value(a)}"
            print(f"  {n.name:36s} {n.op_type:10s}{extra}")
            print(f"      in ={list(n.input)}")
            print(f"      out={list(n.output)}")


def _const_value(g, name: str):
    """取 Constant 节点的张量值；不是 Constant 或没有 value 属性则返回 None。"""
    by = {n.name: n for n in g.node}
    n = by.get(name)
    if n is None or n.op_type != "Constant":
        return None
    for a in n.attribute:
        if a.name == "value":
            return numpy_helper.to_array(a.t)
    return None


def assert_structure(g) -> int:
    """校验图结构与预期一致，返回 bin 轴（c1 所在的轴）。"""
    names = {n.name for n in g.node}
    missing = [n for n in DFL_DROP + [N_RESHAPE_1] if n not in names]
    if missing:
        sys.exit(f"[patch_dfl] 图结构与预期不符，缺少节点：{missing}")

    by = {n.name: n for n in g.node}
    if by["/model.24/dfl/Transpose"].op_type != "Transpose":
        sys.exit("[patch_dfl] /model.24/dfl/Transpose 不是 Transpose")
    if by["/model.24/dfl/conv/Conv"].op_type != "Conv":
        sys.exit("[patch_dfl] /model.24/dfl/conv/Conv 不是 Conv")

    # Conv 的权重必须是 arange(c1)，否则"加权求和"这个前提不成立
    init = {i.name: i for i in g.initializer}
    if W_CONV not in init:
        sys.exit(f"[patch_dfl] 找不到 Conv 权重初值 {W_CONV}")
    w = numpy_helper.to_array(init[W_CONV]).reshape(-1)
    c1 = int(w.shape[0])
    if not np.array_equal(w, np.arange(c1, dtype=w.dtype)):
        sys.exit(f"[patch_dfl] Conv 权重不是 arange({c1})，实际前几个={w[:4]}")
    print(f"[patch_dfl] Conv 权重 = arange({c1}) ✓（加权求和前提成立）")

    # 从 dfl/Reshape 的目标形状推出 bin 轴，而不是硬编码 axis=2
    target = _const_value(g, "/model.24/dfl/Constant")
    if target is None:
        sys.exit("[patch_dfl] 拿不到 /model.24/dfl/Constant（DFL 输入的目标形状）")
    target = [int(d) for d in target]
    hits = [i for i, d in enumerate(target) if d == c1 and i > 0]
    if len(hits) != 1:
        sys.exit(f"[patch_dfl] 目标形状 {target} 里维度 {c1} 出现 {len(hits)} 次，无法定位 bin 轴")
    axis = hits[0]
    print(f"[patch_dfl] DFL 输入形状 = {target}，bin 轴 = {axis} ✓")

    # 最终输出的目标形状（用于后面核对 ReduceSum 的输出形状是否吻合）
    out_target = _const_value(g, "/model.24/dfl/Constant_1")
    if out_target is None:
        sys.exit("[patch_dfl] 拿不到 /model.24/dfl/Constant_1（DFL 输出的目标形状）")
    out_target = [int(d) for d in out_target]
    expect = [d for i, d in enumerate(target) if i != axis]
    if expect != out_target:
        sys.exit(f"[patch_dfl] 去掉 bin 轴后的形状 {expect} 与 Reshape_1 目标 {out_target} 不一致")
    print(f"[patch_dfl] DFL 输出形状 = {out_target} ✓（与 ReduceSum 结果吻合）")

    # Conv 输出只应被 Reshape_1 消费 —— 这是「最小差分」能成立的前提
    eaters = [n.name for n in g.node if CONV_OUT in list(n.input)]
    if eaters != [N_RESHAPE_1]:
        sys.exit(f"[patch_dfl] {CONV_OUT} 的消费者是 {eaters}，预期恰好 [{N_RESHAPE_1}]")

    return axis


# ---------------------------------------------------------------- 拓扑排序

def toposort(nodes, g, extra_available=None) -> list:
    """Kahn 拓扑排序：保证每个节点的输入都先于它被产出。

    available 初值 = 图输入 + 所有 initializer + 无输入节点(Constant) 的输出
    + extra_available（调用方承诺已经就位的名字）。

    ⚠️ 踩过的坑：新加的 initializer 必须在**调用本函数之前**写进 `g.initializer`，
    否则 Mul 需要的 bins 不在 available 里，reduce 也够不着，直接级联卡死。
    """
    available = (
        {i.name for i in g.input}
        | {i.name for i in g.initializer}
        | set(extra_available or ())
    )
    pending = list(nodes)
    ordered: list = []
    while pending:
        ready, rest = [], []
        for n in pending:
            if all((i == "" or i in available) for i in n.input):
                ready.append(n)
            else:
                rest.append(n)
        if not ready:
            stuck = [n.name for n in rest][:6]
            sys.exit(f"[patch_dfl] 拓扑排序卡住，疑似有环或缺输入：{stuck}")
        for n in ready:
            available.update(o for o in n.output if o)
        ordered.extend(ready)
        pending = rest
    return ordered


# ---------------------------------------------------------------- 改写

def patch(model, c1: int, axis: int) -> None:
    """把 DFL 链替换掉：删 4 个节点、加 3 个节点、重接 1 处输入。

    ⚠️ 传进来的是 **model** 而不是 graph：收尾的 `onnx.checker` 必须拿到带
    `opset_import` 的原模型，否则 checker 按最新 opset 校验
    （opset 18+ 的 ReduceSum 要求 axes 是输入），会误报
    `Unrecognized attribute: axes for operator ReduceSum`。踩过。
    """
    g = model.graph
    # 1) 新节点 —— 手工 softmax。
    #    绝不能用 opset 12 的 Softmax 算子沿中间轴归一化（coerced-2D 语义，见文件头）。
    #    且必须**先减最大值**：DFL logits 后面没有激活函数压着，实测在 0..255 的随机图上
    #    裸 Exp 直接溢出成 inf，进而 inf/inf = NaN。（踩过。）
    mx = helper.make_node("ReduceMax", [DFL_IN], [T_MAX],
                          name="dfl_patched/ReduceMax", axes=[axis], keepdims=1)
    sub = helper.make_node("Sub", [DFL_IN, T_MAX], [T_SHIFT], name="dfl_patched/Sub")
    exp = helper.make_node("Exp", [T_SHIFT], [T_EXP], name="dfl_patched/Exp")

    # bins 形状：在 bin 轴上放 arange(c1)，其余维给 1，靠广播对齐
    shape = [1] * 4
    shape[axis] = c1
    bins = np.arange(c1, dtype=np.float32).reshape(shape)
    bins_init = numpy_helper.from_array(bins, T_BINS_INIT)
    ek = helper.make_node("Mul", [T_EXP, T_BINS_INIT], [T_EK], name="dfl_patched/Mul")

    # opset 12：ReduceSum 的 axes 是**属性**（opset 13 才变成输入）
    num = helper.make_node("ReduceSum", [T_EK], [T_NUM],
                           name="dfl_patched/ReduceSumNum", axes=[axis], keepdims=0)
    den = helper.make_node("ReduceSum", [T_EXP], [T_DEN],
                           name="dfl_patched/ReduceSumDen", axes=[axis], keepdims=0)
    div = helper.make_node("Div", [T_NUM, T_DEN], [T_SUM], name="dfl_patched/Div")
    new_nodes = [mx, sub, exp, ek, num, den, div]

    # 2) 删旧节点
    doomed = set(DFL_DROP)
    kept = [n for n in g.node if n.name not in doomed]
    if len(kept) != len(g.node) - len(DFL_DROP):
        sys.exit("[patch_dfl] 删除节点数量不符，检查 DFL_DROP 是否重名")

    # 3) 重接 /model.24/dfl/Reshape_1 的输入
    rewired = 0
    for n in kept:
        if n.name != N_RESHAPE_1:
            continue
        for i, inp in enumerate(n.input):
            if inp == CONV_OUT:
                n.input[i] = T_SUM
                rewired += 1
    if rewired != 1:
        sys.exit(f"[patch_dfl] {N_RESHAPE_1} 的输入重接次数 = {rewired}，预期 1")
    print(f"[patch_dfl] 重接 {N_RESHAPE_1} : {CONV_OUT} -> {T_SUM}")

    # 4) 初值：先加 bins（必须在 toposort 之前！），再摘掉不再被引用的 conv 权重
    old = list(g.initializer)
    g.ClearField("initializer")
    g.initializer.extend([i for i in old if i.name != W_CONV])
    g.initializer.extend([bins_init])
    print(f"[patch_dfl] 初值：+{T_BINS_INIT}（shape={shape}），-{W_CONV}")

    # 5) 重排为拓扑序
    #
    # ⚠️ onnx.checker 会校验 "Nodes in a graph must be topologically sorted"。
    # 直接把新节点 append 到末尾是不行的 —— 消费 DFL 输出的 /model.24/Shape
    # 等节点在列表里更靠前，就会报
    #   input 'dfl_patched/sum' ... is not output of any previous nodes
    # 所以这里做一次 Kahn 排序。（n≈290，O(n²) 完全够用。）
    g.ClearField("node")
    g.node.extend(toposort(kept + new_nodes, g))

    # 6) 收尾检查：两个不合规 Transpose 必须消失
    left = [n.name for n in g.node
            if n.op_type == "Transpose" and n.name in doomed]
    if left:
        sys.exit(f"[patch_dfl] 仍有残留 Transpose：{left}")

    onnx.checker.check_model(model)
    print("[patch_dfl] onnx.checker 通过 ✓")


# ---------------------------------------------------------------- 数值自证

def numeric_selfcheck(orig_path: str, patched_path: str, atol: float, rtol: float) -> None:
    """同一输入跑两图，逐元素比对全部输出。这是唯一的正确性判据。

    判据是标准的**相对**形式：对每个元素要求 |a-b| <= atol + rtol*|a|。

    为什么不能用绝对容差：`output0` 的 box 通道是**像素量纲**（box decode 会把 DFL 结果
    乘上 stride，最高 32），而 DFL 自身只有 0..15 的量级。手工 softmax 与算子 softmax 的
    fp32 舍入差 ~4e-6，乘 stride 后就成了 ~1.2e-4 —— 看着超标，实则只有 2~3 个 ULP。
    用绝对阈值会把"纯粹的浮点舍入"误判成错误，也会把量纲小的信号放得太松。
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s_orig = ort.InferenceSession(orig_path, so, providers=["CPUExecutionProvider"])
    s_new = ort.InferenceSession(patched_path, so, providers=["CPUExecutionProvider"])

    oi = s_orig.get_inputs()[0]
    ni = s_new.get_inputs()[0]
    if oi.name != ni.name:
        sys.exit(f"[patch_dfl] 输入名变了：{oi.name} vs {ni.name}")
    shape = [d if isinstance(d, int) else 1 for d in oi.shape]
    print(f"[patch_dfl] 输入 {oi.name} {shape}")

    outs_o = [(o.name, list(o.shape)) for o in s_orig.get_outputs()]
    outs_n = [(o.name, list(o.shape)) for o in s_new.get_outputs()]
    if outs_o != outs_n:
        sys.exit(f"[patch_dfl] 输出签名变了：{outs_o} vs {outs_n}")
    print(f"[patch_dfl] 输出签名一致：{outs_o}")

    rng = np.random.default_rng(0)
    bad = []
    for trial in range(3):
        # 用真实量纲的输入（0..255 的图像），别用 N(0,1) —— softmax 的数值行为不一样
        x = (rng.random(shape, dtype=np.float32) * 255.0).astype(np.float32)
        yo = s_orig.run(None, {oi.name: x})
        yn = s_new.run(None, {oi.name: x})
        for k, (a, b) in enumerate(zip(yo, yn)):
            # ⚠️ 必须先查 NaN/Inf 再比大小：`max(worst, nan)` 在 Python 里会返回 worst
            #    （因为 nan > worst 为假），"全 NaN"会被静默判成通过。踩过 —— 这是门禁说谎。
            nf_o, nf_n = int((~np.isfinite(a)).sum()), int((~np.isfinite(b)).sum())
            if nf_o or nf_n:
                bad.append(f"trial{trial} out[{k}] 非有限值：原图 {nf_o} 个 / 改后 {nf_n} 个")
                continue

            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            absdiff = np.abs(a64 - b64).max()
            scale = np.abs(a64).max()
            # 按元素归一化的误差：<=1 即通过，等价于逐元素的 allclose
            norm = float((np.abs(a64 - b64) / (atol + rtol * np.abs(a64))).max())
            status = "✓" if norm <= 1.0 else "✗"
            print(f"[patch_dfl] trial{trial} out[{k}] {a.shape} {status} "
                  f"maxAbsDiff={absdiff:.3e} max|ref|={scale:.3e} "
                  f"归一化误差={norm:.3f}（预算 1.000）")
            if norm > 1.0:
                bad.append(f"trial{trial} out[{k}] 归一化误差 {norm:.3f} > 1（abs={absdiff:.3e}）")

    if bad:
        for line in bad:
            print(f"[patch_dfl] ✗ {line}")
        sys.exit(2)
    print(f"[patch_dfl] 全部输出在 {atol:.1e} + {rtol:.1e}*|ref| 的预算内 ✓")


# ---------------------------------------------------------------- 入口

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst")
    ap.add_argument("--dump", action="store_true", help="只打印 DFL 子图")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    try:
        model = onnx.load(args.src)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[patch_dfl] 读取失败：{e}")

    g = model.graph
    if args.dump:
        dump_dfl(g)
        return 0

    axis = assert_structure(g)
    c1 = int(numpy_helper.to_array(
        {i.name: i for i in g.initializer}[W_CONV]).reshape(-1).shape[0])
    if not args.dst:
        sys.exit("[patch_dfl] 未给 --out")

    patch(model, c1, axis)

    # 先落临时文件、跑数值自证，过了才改名成正式产物
    tmp = args.dst + ".unverified.onnx"
    onnx.save(model, tmp)
    print(f"[patch_dfl] 改写后已存 {tmp}，开始数值自证…")
    try:
        numeric_selfcheck(args.src, tmp, args.atol, args.rtol)
    except SystemExit as e:
        print(f"[patch_dfl] ✗ 数值自证未通过（code={e.code}），**不落盘**")
        return int(e.code) if isinstance(e.code, int) else 2

    os.replace(tmp, args.dst)
    print(f"[patch_dfl] ✓ 数值自证通过，最终产物 {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
