#!/usr/bin/env python3
"""scan_onnx.py —— ONNX 静态扫描：判定模型能否上麒麟 NPU（麒麟 8020）

为什么需要
==========
官方「支持的算子」列表**不能**作为选型依据 —— `Reshape`/`Permute`/`Swish`
全在官方列表里，但真正判否的条件写在编译器内部检查里，文档上没有。

判据只有两条硬门（来自 kirin-npu-model-porting 技能，均为编译器原话）：

  1. 所有张量 rank ≤ 4     ← permute_check_support.cc IsSupport(21):: realdimCnt > 4
  2. Transpose 的 perm 只能是 [0,1,3,2]
                            ← permute_matmul_fusion_pass.cc: only support order 0132

先跑这个（几秒），再决定要不要花几分钟去转换。

用法
====
  python tools/scan_onnx.py model.onnx
  python tools/scan_onnx.py model.onnx --ops        # 附带算子清单

退出码：0 = 两条硬门都过；2 = 有硬门不过；3 = 无法读取
"""

from __future__ import annotations

import argparse
import collections
import sys


def load_graph(path: str):
    """读 ONNX。★ 不能用 onnx.load() —— 对 .onnx.json 之类会按扩展名当 JSON 解析。"""
    from onnx import ModelProto

    m = ModelProto()
    with open(path, "rb") as f:
        m.ParseFromString(f.read())
    return m.graph


def tensor_ranks(g) -> tuple:
    """返回 (常量 rank 分布, 常量中 rank>=5 的个数, rank>=5 的常量名)。"""
    ranks = collections.Counter()
    bad = []
    for init in g.initializer:
        r = len(init.dims)
        ranks[r] += 1
        if r >= 5:
            bad.append((init.name, list(init.dims)))
    return ranks, len(bad), bad


def transpose_perms(g) -> list:
    """列出所有 Transpose 的 perm，标记非 [0,1,3,2] 的。"""
    out = []
    for n in g.node:
        if n.op_type == "Transpose":
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            out.append((n.name, perm))
    return out


def value_ranks(g) -> dict:
    """张量（非常量）的 rank 分布 —— 从 value_info + input + output 推。"""
    ranks = collections.Counter()
    bad = []
    for coll, tag in ((g.input, "input"), (g.output, "output"), (g.value_info, "value")):
        for vi in coll:
            r = len(vi.type.tensor_type.shape.dim)
            if r > 0:
                ranks[r] += 1
                if r >= 5:
                    bad.append((f"{tag}:{vi.name}", r))
    return ranks, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="ONNX 静态扫描（麒麟 NPU 可行性）")
    ap.add_argument("onnx", help="ONNX 文件路径")
    ap.add_argument("--ops", action="store_true", help="打印算子清单")
    args = ap.parse_args()

    try:
        g = load_graph(args.onnx)
    except Exception as e:
        print(f"[scan] 读取失败：{e}", file=sys.stderr)
        return 3

    print(f"[scan] 文件 = {args.onnx}")
    print(f"[scan] 节点数 = {len(g.node)}，常量数 = {len(g.initializer)}")

    # 输入 / 输出形状
    for tag, coll in (("输入", g.input), ("输出", g.output)):
        for vi in coll:
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                    for d in vi.type.tensor_type.shape.dim]
            print(f"[scan] {tag} {vi.name} : {dims}")

    # ---- 硬门 1：rank ≤ 4 ----
    c_ranks, c_bad, c_names = tensor_ranks(g)
    v_ranks, v_bad = value_ranks(g)
    print(f"[scan] 常量 rank 分布 = {dict(sorted(c_ranks.items()))}")
    print(f"[scan] 张量 rank 分布 = {dict(sorted(v_ranks.items()))}")

    gate1 = (c_bad == 0 and len(v_bad) == 0)
    if not gate1:
        print(f"[scan] ✗ 硬门1 不过：rank>=5 的常量 {c_bad} 个、张量 {len(v_bad)} 个")
        for n, d in c_names[:8]:
            print(f"         常量 {n} dims={d}")
        for n, r in v_bad[:8]:
            print(f"         张量 {n} rank={r}")
    else:
        print("[scan] ✓ 硬门1 过：所有张量 rank ≤ 4")

    # ---- 硬门 2：Transpose perm ----
    perms = transpose_perms(g)
    bad_perms = [(n, p) for n, p in perms if p != [0, 1, 3, 2]]
    print(f"[scan] Transpose 共 {len(perms)} 个，其中 perm != [0,1,3,2] 的 {len(bad_perms)} 个")
    for n, p in bad_perms[:10]:
        print(f"         {n} perm={p}   <-- NOT SUPPORTED")
    gate2 = (len(bad_perms) == 0)

    # ---- 结论 ----
    print()
    if gate1 and gate2:
        print("[scan] ⇒ 两条硬门都过，可以尝试转 .ms")
    else:
        print("[scan] ⇒ 有硬门不过，直接转会失败；需先裁切/改写（见技能二、三节）")

    print(f"[scan] 算子 = {dict(sorted(collections.Counter(n.op_type for n in g.node).items()))}"
          if args.ops else
          f"[scan] 算子种类 = {len(set(n.op_type for n in g.node))}")

    return 0 if (gate1 and gate2) else 2


if __name__ == "__main__":
    sys.exit(main())
