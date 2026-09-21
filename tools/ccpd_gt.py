#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCPD 文件名 -> 真值车牌串。

CCPD 的文件名把车牌号编码成 7 个下标（`load_CCPD.py` 的口径）：
    <id>-<tilt>-<bbox>-<vertices>-<lp>-<brightness>-<blur>.jpg
    其中 <lp> 是 `p0_p1_..._p6`，p0 取 provinces，p1..p6 取 ads。

为什么要单独一个脚本：整车数据集的真值**只在文件名里**，没有单独标注文件。
解错一张表就会让全部 1000 个真值系统性偏移 —— 与 t6 踩过的
「文件名被工具改过、凭空制造 29 条长度错误」同类，但更隐蔽。
所以这里把三张表与解码规则集中一处，并带自检。

用法：
    python ccpd_gt.py <ccpd图片目录或zip解包目录> [--limit N] [--out out.tsv]
"""
from __future__ import annotations

import argparse
import os
import sys

# 三张表必须与 CCPD 官方 load_CCPD.py 逐字一致。
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
             "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
             "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
             "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q",
       "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
       "6", "7", "8", "9", "O"]


def decode_lp(field: str) -> str | None:
    """`1_0_32_32_4_32_25` -> `沪A88E81`。长度不是 7 或下标越界时返回 None。"""
    parts = field.split("_")
    if len(parts) != 7:
        return None
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        return None
    if idx[0] >= len(PROVINCES):
        return None
    if any(i >= len(ADS) for i in idx[1:]):
        return None
    return PROVINCES[idx[0]] + "".join(ADS[i] for i in idx[1:])


def gt_of_filename(name: str) -> str | None:
    """从完整文件名取真值。字段数不是 7 段（CCPD 标准）时返回 None。"""
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    return decode_lp(parts[4])


def self_check() -> int:
    """自检：表长、一条已知样例、越界拒绝。"""
    errs = []
    if len(PROVINCES) != 34:
        errs.append(f"PROVINCES 长度 {len(PROVINCES)} != 34")
    if len(ADS) != 35:
        errs.append(f"ADS 长度 {len(ADS)} != 35")
    # CCPD README 之外的独立锚点：下标 1/0/32/32/4/32/25 -> 沪 A 8 8 E 8 1
    got = decode_lp("1_0_32_32_4_32_25")
    if got != "沪A88E81":
        errs.append(f"样例解码错误: got {got!r} want '沪A88E81'")
    if decode_lp("1_0_32_32_4_32") is not None:
        errs.append("6 段应被拒绝")
    if decode_lp("99_0_0_0_0_0_0") is not None:
        errs.append("越界省份下标应被拒绝")
    if decode_lp("0_99_0_0_0_0_0") is not None:
        errs.append("越界 ads 下标应被拒绝")
    for e in errs:
        print(f"[FAIL] {e}")
    if errs:
        return 1
    print("[ok] 表长 34/35，样例解码正确，越界与短段均被拒绝")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", help="CCPD 图片目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0=全部）")
    ap.add_argument("--out", help="输出 TSV 路径（默认打到 stdout）")
    ap.add_argument("--self-check", action="store_true", help="只跑自检")
    a = ap.parse_args()

    if a.self_check or not a.dir:
        return self_check()

    import os
    names = sorted(n for n in os.listdir(a.dir) if n.lower().endswith(".jpg"))
    if a.limit:
        names = names[: a.limit]

    rows, bad = [], 0
    for n in names:
        gt = gt_of_filename(n)
        if gt is None or len(gt) not in (7, 8):
            bad += 1
            continue
        rows.append((n, gt))

    out = open(a.out, "w", encoding="utf-8", newline="\n") if a.out else None
    try:
        for n, gt in rows:
            line = f"{n}\t{gt}\n"
            if out:
                out.write(line)
            else:
                print(line, end="")
    finally:
        if out:
            out.close()

    print(f"[done] n={len(rows)} skipped={bad}", file=__import__('sys').stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
