#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查识别器的字符表是否覆盖 CCPD-Green 真值所需字符。

为什么必须先查这个：如果真值里出现了字符表装不下的字符，那么无论识别器多好，
**上限就已经被钉死**——此时把「绿牌准确率低」解释成模型问题是错的，
真正的原因是字符表覆盖不足。这类「上限由表决定」的坑本项目已经踩过一次
（`lpr_pipeline.cpp` 里 C 从字符表推导导致错位，见该文件 455-459 行的注释）。

用法：python tools/check_green_charset.py <gt.tsv> [...]
"""
from __future__ import annotations

import io
import sys

# 与 lpr_pipeline.cpp::LprToken() 一致（index 0 = CTC blank）。
# 注意：`lpr_pipeline.h` 的注释写「44 entries」，但实际表是 **77 项**
# （44 是「44 个真实类别」的旧说法，见该头文件 L208-209 的 token 表）。
# 这里以**代码里的实际表**为准，不以注释为准 —— 注释与表不一致本身就是
# 一个值得记录的坑。
LPR_TOKEN = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民", "沪",
    "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", "蒙", "藏",
    "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", "黑", "领", "使", "澳",
]


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2

    tok = set(LPR_TOKEN)
    print(f"LprToken: {len(LPR_TOKEN)} 项（含 blank）")

    all_chars: set[str] = set()
    for p in paths:
        chars: set[str] = set()
        n = 0
        with io.open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                n += 1
                chars |= set(parts[1])
        all_chars |= chars
        miss = chars - tok
        print(f"\n{p}")
        print(f"  n={n}  真值字符集 {len(chars)} 个: {''.join(sorted(chars))}")
        print(f"  字符表缺失: {''.join(sorted(miss)) if miss else '无 —— 全覆盖'}")

    miss_all = all_chars - tok
    print("\n" + "=" * 60)
    if miss_all:
        print(f"[WARN] 字符表缺 {len(miss_all)} 个字符: {''.join(sorted(miss_all))}")
        print("       ⇒ 绿牌准确率上限已被字符表钉死，不能归因于模型能力。")
        return 1
    print("[ok] 字符表完全覆盖 —— 绿牌准确率不受字符表限制。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
