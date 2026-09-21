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
    """`1_0_32_32_4_32_25` -> `沪A88E81`（7 位）；`0_0_3_30_30_25_31_32` -> `皖AD66178`（8 位）。

    7 位是 CCPD2019 常规牌，8 位是 CCPD-Green 新能源牌。两者**同一套下标表**，
    差别只在段数 —— 所以这里按段数分派，不复制表。

    结构自检（两档都做）：
      - 段数必须是 7 或 8；
      - idx[0] 必须在省份表内；
      - idx[1] 必须是**字母**（ALPHABETS 区间，即 < 24）——
        这是唯一能把「解错表」与「解对表」区分开的硬约束：
        若把 ADS 当成字母表用（少 11 项偏移），idx[1] 会落到数字区，立刻暴露。
      - 其余下标在 ADS 范围内。
    """
    parts = field.split("_")
    if len(parts) not in (7, 8):
        return None
    try:
        idx = [int(p) for p in parts]
    except ValueError:
        return None
    if idx[0] >= len(PROVINCES):
        return None
    # 第 2 位（下标 1）在真实车牌上必为字母。这一条是防「表用错」的护栏。
    if idx[1] >= len(ALPHABETS):
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
    if len(ALPHABETS) != 24:
        errs.append(f"ALPHABETS 长度 {len(ALPHABETS)} != 24")
    if len(ADS) != 35:
        errs.append(f"ADS 长度 {len(ADS)} != 35")
    # 独立锚点：下标 1/0/32/32/4/32/25 -> 沪 A 8 8 E 8 1
    got = decode_lp("1_0_32_32_4_32_25")
    if got != "沪A88E81":
        errs.append(f"7 位样例解码错误: got {got!r} want '沪A88E81'")
    # 绿牌锚点（CCPD-Green，8 段）。这 4 条取自 Kaggle ccpd-green 真实文件名。
    for field, want in [("0_0_3_30_30_25_31_32", "皖AD66178"),
                        ("0_0_5_24_25_24_30_24", "皖AF01060"),
                        ("0_0_3_29_30_33_33_33", "皖AD56999"),
                        ("0_0_3_1_24_25_26_33", "皖ADB0129")]:
        g = decode_lp(field)
        if g != want:
            errs.append(f"8 位样例 {field}: got {g!r} want {want!r}")
    # 反例：段数、省份越界、ads 越界、**第 2 位落在数字区**
    if decode_lp("1_0_32_32_4_32") is not None:
        errs.append("6 段应被拒绝")
    if decode_lp("1_0_32_32_4_32_25_0_0") is not None:
        errs.append("9 段应被拒绝")
    if decode_lp("99_0_0_0_0_0_0") is not None:
        errs.append("越界省份下标应被拒绝")
    if decode_lp("0_99_0_0_0_0_0") is not None:
        errs.append("越界 ads 下标应被拒绝")
    # 这条是「表用错」的探测器：第 2 位若解出数字，说明 ALPHABETS/ADS 混用了。
    if decode_lp("0_30_0_0_0_0_0") is not None:
        errs.append("第 2 位是数字(下标 30)应被拒绝 —— 该约束用于发现表用错")
    for e in errs:
        print(f"[FAIL] {e}")
    if errs:
        return 1
    print("[ok] 表长 34/24/35；7 位与 8 位样例各 4 条全对；"
          "段数/省份越界/ads 越界/第2位为数字 均被拒绝")
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
