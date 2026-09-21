#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端侧 NPU 裸识别 vs 主机 onnxruntime rpv3 —— 配对 McNemar 检验。

## 为什么要做

T11 报了「端侧 NPU 89.9% vs 主机 onnxruntime 90.6%，只差 0.7pp」，并自己标注：
「0.7 pp 在 1000 张上约 7 张的量级，**不能排除是抽样波动**」。
本轮把这个保留意见**做掉**。

关键：两次测量用的是**同一批 1000 张图**（`sirius-ai/LPRNet_Pytorch` test 集），
所以正确的检验是**配对** McNemar，而不是两个独立置信区间 ——
配对能消掉「哪些图本来就难」这个共同方差，检出力高得多。

## 两个数据源

- **端侧**：`evidence/crop_bare.log`（T11 落盘），每行有 `code_np=` 与 `gt=`。
- **主机**：本脚本**现场重跑** A16 的 `lprnet_real_accuracy.py` 里那条 rpv3 路径，
  因为 A16 的日志只打印了**分歧**，没有全量 1000 条的逐图预测，无法直接配对。
  重跑而不是手工抄，是为了让「主机侧数字」与 A16 报的 906/1000 可核对。

## 用法

    python tools/mcnemar_device_vs_host.py <crop_bare.log> <lpr-showcase 根目录>
"""
from __future__ import annotations

import io
import os
import re
import sys


def load_device(path: str) -> dict[str, str]:
    """从 T11 日志取 name -> code_np（端侧 NPU）。"""
    out: dict[str, str] = {}
    pat = re.compile(r"BARE file=(\S+) gt=(\S+) code_cpu=(\S*) code_np=(\S*)")
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out[m.group(1)] = m.group(4)
    return out


def load_host(showcase: str) -> dict[str, str]:
    """现场重跑主机 rpv3，返回 name -> code。

    复用 lpr-showcase 自己的 `hlpr_reference` 助手（`H.sess` / `H.imread_u` /
    `H.recognize`），**不复制预处理** —— 复制就会有两套实现，正是本项目反复
    踩的坑。真值口径也与 A16 一致：`stem.split('-')[0].split('_')[0]`。
    """
    sys.path.insert(0, os.path.join(showcase, "tools"))
    os.chdir(showcase)  # hlpr_reference 按相对路径找模型
    import importlib

    H = importlib.import_module("hlpr_reference")

    crops = os.path.join(showcase, "_dataset", "real", "crops")
    names = sorted(n for n in os.listdir(crops) if n.lower().endswith(".jpg"))
    rec = H.sess("rpv3_mdict_160_r3.onnx")
    out: dict[str, str] = {}
    for n in names:
        img = H.imread_u(os.path.join(crops, n))
        out[n] = H.recognize(rec, img)[0]
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """精确二项检验（b, c 为两个不对称格的计数）。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 双尾：2 * P(X <= k)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    log_path, showcase = sys.argv[1], sys.argv[2]

    dev = load_device(log_path)
    print(f"[device] {len(dev)} 条（crop_bare.log）")
    host = load_host(showcase)
    print(f"[host]   {len(host)} 条（现场重跑 rpv3）")

    common = sorted(set(dev) & set(host))
    print(f"[pair]   共同 {len(common)} 条")

    both_ok = dev_ok_only = host_ok_only = both_bad = 0
    for n in common:
        # 真值口径与 A16 一致：stem 去掉 '-' 与 '_' 后缀。
        gt = os.path.splitext(n)[0].split("-")[0].split("_")[0]
        d = dev[n] == gt
        h = host[n] == gt
        if d and h:
            both_ok += 1
        elif d:
            dev_ok_only += 1
        elif h:
            host_ok_only += 1
        else:
            both_bad += 1

    n = len(common)
    d_acc = (both_ok + dev_ok_only) / n
    h_acc = (both_ok + host_ok_only) / n
    print()
    print(f"  端侧 NPU 正确 : {both_ok + dev_ok_only}/{n} = {d_acc:.1%}")
    print(f"  主机 rpv3 正确: {both_ok + host_ok_only}/{n} = {h_acc:.1%}")
    print(f"  差            : {100 * (d_acc - h_acc):+.1f} pp")
    print()
    print("  配对表:")
    print(f"    两端都对     : {both_ok}")
    print(f"    仅端侧对     : {dev_ok_only}")
    print(f"    仅主机对     : {host_ok_only}")
    print(f"    两端都错     : {both_bad}")

    p = mcnemar_exact(dev_ok_only, host_ok_only)
    print()
    print(f"  McNemar 精确检验: b={dev_ok_only} c={host_ok_only}  p = {p:.4f}")
    if p < 0.05:
        print("  ⇒ 差异**统计显著**，不能归因于抽样波动。")
    else:
        print("  ⇒ 差异**不显著** —— 0.7pp 落在抽样波动内，移植保真这一说法成立。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
