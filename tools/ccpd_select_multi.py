#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从多目录 CCPD-Green 里分层抽样，复制图片并写 gt.tsv。

与 `ccpd_select.py` 的差别：那个假设**所有图在一个目录**。CCPD-Green 分成
train/val/test 三个目录，且同名文件可能跨目录，所以这里显式接收多组
`目录:gt.tsv` 映射，先建全量索引再抽样。

用法：
    python ccpd_select_multi.py <outDir> <N> --cap 100 --seed 20260921 \
        --src <dir1> <gt1.tsv> --src <dir2> <gt2.tsv> ...

注意：**不要用 `<dir>:<tsv>` 这种冒号分隔写法**。在 PowerShell 下把
`D:\a:D:\b` 传给原生程序时，开头的 `D:` 会被当成 drive 限定符吃掉，
Python 收到的是 `\a:D:\b`（实测）。用 `--src` 重复两次参数即可绕开。
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dst")
    ap.add_argument("n", type=int)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--src", nargs=2, action="append", required=True,
                    metavar=("DIR", "GT_TSV"))
    a = ap.parse_args()

    # 建全量索引：文件名 -> (所在目录, 真值)
    index: dict[str, tuple[str, str]] = {}
    for d, tsv in a.src:
        n_before = len(index)
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                name, gt = parts
                index[name] = (d, gt)
        print(f"[load] {d}: +{len(index) - n_before} (累计 {len(index)})")

    rng = random.Random(a.seed)
    by_prov: dict[str, list[str]] = {}
    for name, (_, gt) in index.items():
        by_prov.setdefault(gt[0], []).append(name)
    print(f"[prov] {len(by_prov)} 个省")

    picked: list[str] = []
    for prov in sorted(by_prov):
        names = by_prov[prov][:]
        rng.shuffle(names)
        picked.extend(names[: a.cap])
    rng.shuffle(picked)
    if len(picked) > a.n:
        picked = picked[: a.n]
    print(f"[pick] {len(picked)} 张（cap={a.cap}/省）")
    if len(picked) < a.n:
        print(f"[warn] 只取到 {len(picked)} 张（少于请求的 {a.n}）"
              f" —— 省份偏斜 + cap 所致，如实记录", file=sys.stderr)

    os.makedirs(a.dst, exist_ok=True)
    out_rows = []
    for name in sorted(picked):
        d, gt = index[name]
        src = os.path.join(d, name)
        if not os.path.isfile(src):
            print(f"[miss] {src}", file=sys.stderr)
            continue
        shutil.copyfile(src, os.path.join(a.dst, name))
        out_rows.append((name, gt))

    with open(os.path.join(a.dst, "gt.tsv"), "w", encoding="utf-8", newline="\n") as f:
        for name, gt in out_rows:
            f.write(f"{name}\t{gt}\n")

    provs: dict[str, int] = {}
    lens: dict[int, int] = {}
    for _, gt in out_rows:
        provs[gt[0]] = provs.get(gt[0], 0) + 1
        lens[len(gt)] = lens.get(len(gt), 0) + 1
    print(f"[done] copied {len(out_rows)} -> {a.dst}")
    print(f"[dist] 省 {len(provs)} 个; 长度 {dict(sorted(lens.items()))}")
    print("[prov] " + ", ".join(f"{p}×{c}" for p, c in
                                sorted(provs.items(), key=lambda kv: -kv[1])[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
