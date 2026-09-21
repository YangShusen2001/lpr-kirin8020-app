#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CCPD 全量里分层抽 1000 张，连同真值一起搬到暂存目录。

为什么分层：CCPD2019-balance 仍严重偏斜（皖 4794 / 8900 = 54%）。
T10/T11 已经证明**省份位是替换错误的主战场**（且是数据集属性，换后端换不掉），
如果抽样继续让皖占一半，det A/B 的分歧率会被单一省份的分布绑架，
看不出「换后端在多大范围内影响输出」。按省份设上限再补足，样本才有代表性。

用法：
    python ccpd_select.py <srcDir> <gtTsv> <dstDir> <N> [--cap 每省上限]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("gt")
    ap.add_argument("dst")
    ap.add_argument("n", type=int)
    ap.add_argument("--cap", type=int, default=100, help="每个省份最多取几张")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()

    rows = []
    with open(a.gt, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, gt = line.split("\t")
            rows.append((name, gt))
    print(f"[read] {len(rows)} rows from {a.gt}")

    rng = random.Random(a.seed)
    by_prov: dict[str, list[str]] = {}
    for name, gt in rows:
        by_prov.setdefault(gt[0], []).append(name)
    print(f"[prov] {len(by_prov)} provinces")

    picked: list[str] = []
    for prov in sorted(by_prov):
        names = by_prov[prov][:]
        rng.shuffle(names)
        picked.extend(names[: a.cap])
    rng.shuffle(picked)

    if len(picked) > a.n:
        picked = picked[: a.n]
    print(f"[pick] {len(picked)} images (cap={a.cap}/province)")

    if len(picked) < a.n:
        print(f"[warn] 只有 {len(picked)} 张，少于请求的 {a.n}", file=sys.stderr)

    os.makedirs(a.dst, exist_ok=True)
    gt_of = dict(rows)
    out_rows = []
    for name in sorted(picked):
        src = os.path.join(a.src, name)
        if not os.path.isfile(src):
            print(f"[miss] {name}", file=sys.stderr)
            continue
        shutil.copyfile(src, os.path.join(a.dst, name))
        out_rows.append((name, gt_of[name]))

    with open(os.path.join(a.dst, "gt.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for name, gt in out_rows:
            f.write(f"{name}\t{gt}\n")

    provs: dict[str, int] = {}
    for _, gt in out_rows:
        provs[gt[0]] = provs.get(gt[0], 0) + 1
    top = sorted(provs.items(), key=lambda kv: -kv[1])[: 6]
    print(f"[done] copied {len(out_rows)} -> {a.dst}")
    print(f"[dist] {len(provs)} provinces; top: " +
          ", ".join(f"{p}×{c}" for p, c in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
