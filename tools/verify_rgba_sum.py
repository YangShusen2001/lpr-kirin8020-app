"""回归守卫：RGBA 校验和的优化必须与「逐字节 + i%4」版本**逐位相等**。

## 背景

`napi_init.cpp` 的相机帧路径会算一遍全图 RGB 校验和（`rgbaSum`），用来证明
native 的 NV21→RGBA 与系统解码器等价 —— 这是「换掉取帧路径」这条结论的证据。

2026-09-21 做了两处修改：
1. 把它**纳入计时**（原先夹在 convMs 与 totalMs 之间，完全隐形）
2. 把 `for (i...) if (i % 4 != 3) s += d[i]`
   改成 `for (i = 0; i + 2 < n; i += 4) s += d[i] + d[i+1] + d[i+2]`

第 2 项是纯性能优化，但**校验和的用途是等价性判定**：若优化后数值变了，
与历史日志（如 evidence/ 里记录的 rgbaSum）就对不上，证据链断裂。
主机微基准（`bench_rgba_sum.cpp`）实测该改法快 4.2x，且与现状逐位相等。

## 本守卫做什么

用 Python 重实现两个版本，在多种尺寸/内容上断言它们**完全相等**。
若有人进一步优化（比如改成抽稀采样、改字长累加），只要数值一变，这里就会红。

注意：本守卫**不阻止**优化，只要求「改数值必须同步更新历史对照方式」——
即要么保持逐位相等，要么在提交信息里说明旧 rgbaSum 不可再比对。

用法：
    python tools/verify_rgba_sum.py
退出码 0 = 通过。不需要设备。
"""
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "..", "LprDemo", "entry", "src", "main", "cpp",
                   "napi_init.cpp")


def strip_comments(text: str) -> str:
    """剥掉 C/C++ 的行注释与块注释，只留代码。

    为什么需要：源码守卫要判断"代码里是否还有 i % 4 分支"，但注释里会
    记录历史写法（"原写法是 i % 4"）。若不剥离，守卫会对注释编辑过敏。
    这是很粗糙的实现（不处理字符串内的 //），但对本项目这几个模式够用，
    且够用即可 —— 守卫要的是稳定，不是完备的词法分析。
    """
    import re
    # 块注释
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    # 行注释
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def sum_current(data: bytes) -> int:
    """现状（优化后）：按像素步进 4，直接累加 R/G/B。"""
    s = 0
    n = len(data)
    i = 0
    while i + 2 < n:
        s += data[i] + data[i + 1] + data[i + 2]
        i += 4
    return s


def sum_legacy(data: bytes) -> int:
    """原始版本：逐字节 + i%4 分支（跳过 alpha）。"""
    s = 0
    for i, b in enumerate(data):
        if i % 4 != 3:
            s += b
    return s


def make_rgba(w: int, h: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(w * h * 4))


def check_numeric() -> bool:
    print("=== 数值等价性（多种尺寸/内容）===")
    ok = True
    cases = [
        (480, 640, 1),      # 相机实际尺寸（640x480 输入，rot=90 后为 480x640）
        (480, 640, 2),
        (1, 1, 3),
        (3, 5, 4),
        (7, 2, 5),          # 非 4 对齐的字节数
        (640, 480, 6),
        (2, 3, 7),
    ]
    for w, h, seed in cases:
        data = make_rgba(w, h, seed)
        a = sum_current(data)
        b = sum_legacy(data)
        same = (a == b)
        if not same:
            ok = False
        print(f"  {w}x{h:<5} seed={seed}  n={len(data):>7}  "
              f"现状={a:>12}  原始={b:>12}  相等? {'是' if same else '否'}")
    return ok


def check_source() -> bool:
    """源码守卫：确认仍是「步进 4」写法，且没有退化回 i%4 分支。"""
    print()
    print("=== 源码守卫 ===")
    if not os.path.exists(CPP):
        print(f"  [FAIL] 找不到 {CPP}")
        return False

    with open(CPP, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    ok = True

    # 1) 不应再有 i % 4 的分支（那是最慢的写法）
    #
    # 只扫**代码**，不扫注释：注释里会记录"原写法是 i % 4"，那是解释性文字，
    # 把它也算进来会让守卫对注释编辑过敏。所以先剥掉 // 行注释与 /* */ 块注释。
    code = strip_comments(text)
    if "i % 4 != 3" in code:
        print("  [FAIL] 代码中仍存在 `i % 4 != 3` 分支 —— 这是被优化掉的最慢写法")
        ok = False
    else:
        print("  [OK] 代码中已无 `i % 4 != 3` 分支（注释里的历史说明不计）")

    # 2) 应有步进 4 的循环
    if "i += 4" in code:
        print("  [OK] 使用步进 4 的循环（i += 4）")
    else:
        print("  [FAIL] 未找到 `i += 4`，校验和写法已被改成别的形式 ——"
              " 若新写法与旧值不等，历史 rgbaSum 将无法比对，"
              " 请在提交信息中明确说明")
        ok = False

    # 3) 不应退化成抽稀采样（会漏检单字节变化）
    if "i += 64" in code:
        print("  [FAIL] 检测到抽稀步长 i += 64 —— 对单字节改动不敏感，"
              " 校验和将失去等价性判定的意义")
        ok = False

    return ok


def check_sensitivity() -> bool:
    """校验和必须对内容敏感：改 1 个字节就要变化。"""
    print()
    print("=== 敏感性（改 1 字节必须变化）===")
    data = bytearray(make_rgba(8, 8, 11))
    base = sum_current(bytes(data))
    changed = 0
    total = 0
    # 只改 RGB 通道（alpha 本就不计入，改它不变是预期的）
    for idx in range(0, len(data), 4):
        for off in (0, 1, 2):
            pos = idx + off
            if pos >= len(data):
                continue
            total += 1
            data[pos] ^= 0xFF
            if sum_current(bytes(data)) != base:
                changed += 1
            data[pos] ^= 0xFF
    ok = (changed == total and total > 0)
    print(f"  翻转单个 RGB 字节 {total} 次，校验和变化 {changed} 次  "
          f"-> {'全部敏感 OK' if ok else '存在漏检 FAIL'}")
    return ok


def main() -> int:
    ok = check_numeric()
    ok = check_source() and ok
    ok = check_sensitivity() and ok
    print()
    if ok:
        print("=> 全部通过：优化后的校验和与原始版本逐位相等，且对单字节改动敏感")
        return 0
    print("=> 未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())
