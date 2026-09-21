#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三组配对 McNemar，用来把「后端差异」与「预处理实现差异」分开。

## 为什么需要三组

`crop_bare.log`（T11）里同一批 1000 张图上同时有：
  - 端侧 CPU 的读数 `code_cpu`（MS Lite CPU 后端）
  - 端侧 NPU 的读数 `code_np`（MS Lite NNRT→NPU 后端）

两者**共用同一套 C++ 预处理**（`LprEncodePlateInto`），所以
**CPU vs NPU 是纯粹的后端对比**。

而「端侧 NPU vs 主机 rpv3」除了后端不同，还多了
**C++ 移植 vs Python 参考实现**这一层预处理差异 —— 是个混淆对比。

所以：
  - CPU vs NPU（同预处理）→ 干净的后端差异
  - NPU vs host / CPU vs host → 后端 + 预处理 的合成差异

若 NPU-vs-host 显著而 CPU-vs-NPU 也显著，且方向一致，说明两层都在起作用。

用法：python tools/mcnemar_triplet.py <crop_bare.log> <lpr-showcase 根>
"""
from __future__ import annotations

import io
import os
import re
import sys
from math import comb


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)


def load_device(path: str) -> dict[str, tuple[str, str]]:
    """name -> (code_cpu, code_np)"""
    out: dict[str, tuple[str, str]] = {}
    pat = re.compile(r"BARE file=(\S+) gt=(\S+) code_cpu=(\S*) code_np=(\S*)")
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = (m.group(3), m.group(4))
    return out


def load_host(showcase: str) -> dict[str, str]:
    sys.path.insert(0, os.path.join(showcase, "tools"))
    os.chdir(showcase)
    import importlib
    H = importlib.import_module("hlpr_reference")
    crops = os.path.join(showcase, "_dataset", "real", "crops")
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    out: dict[str, str] = {}
    for n in sorted(x for x in os.listdir(crops) if x.lower().endswith(".jpg")):
        out[n] = H.recognize(rec, H.imread_u(os.path.join(crops, n)))[0]
    return out


def pair(common, get_a, get_b, label_a, label_b) -> None:
    a_ok_only = b_ok_only = both = neither = 0
    for n, gt in common:
        a = get_a(n) == gt
        b = get_b(n) == gt
        if a and b:
            both += 1
        elif a:
            a_ok_only += 1
        elif b:
            b_ok_only += 1
        else:
            neither += 1
    n = len(common)
    acc_a = (both + a_ok_only) / n
    acc_b = (both + b_ok_only) / n
    p = mcnemar_exact(a_ok_only, b_ok_only)
    print(f"\n=== {label_a}  vs  {label_b} ===")
    print(f"  {label_a:12s}: {both + a_ok_only}/{n} = {acc_a:.1%}")
    print(f"  {label_b:12s}: {both + b_ok_only}/{n} = {acc_b:.1%}")
    print(f"  差          : {100 * (acc_a - acc_b):+.1f} pp")
    print(f"  配对: 都对 {both} | 仅{label_a}对 {a_ok_only} | 仅{label_b}对 {b_ok_only} | 都错 {neither}")
    verdict = "显著" if p < 0.05 else "不显著"
    print(f"  McNemar p = {p:.4f}  → 差异{verdict}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    dev = load_device(sys.argv[1])
    host = load_host(sys.argv[2])
    print(f"[device] {len(dev)} 条   [host] {len(host)} 条")

    common = []
    for n in sorted(set(dev) & set(host)):
        gt = os.path.splitext(n)[0].split("-")[0].split("_")[0]
        common.append((n, gt))
    print(f"[pair]   {len(common)} 条共同")

    pair(common, lambda n: dev[n][1], lambda n: dev[n][0], "端侧NPU", "端侧CPU")
    pair(common, lambda n: dev[n][1], lambda n: host[n], "端侧NPU", "主机rpv3")
    pair(common, lambda n: dev[n][0], lambda n: host[n], "端侧CPU", "主机rpv3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
