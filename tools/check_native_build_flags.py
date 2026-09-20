#!/usr/bin/env python3
"""守卫：native 代码**必须以优化模式**编译。

## 为什么需要这个守卫

`entry/build-profile.json5` 里写了 `"cppFlags": "-O3"`，看起来已经开了优化。
但 hvigor 默认按 **debug** 构建，CMake 因此用 `CMAKE_BUILD_TYPE=Debug`，并把它
自己的 `-O0 -g` **追加在 `-O3` 之后**。编译器取**最后一个** `-O` 参数，于是
`-O3` 被静默废掉。实际编译行：

    ... -O3 -D__MUSL__ -O0 -g -fPIC

**没有任何报错、警告或日志**。唯一的后果是全部 native 代码以未优化状态运行。

真机实测（nova 14 Pro，生产档，同一场景）：

| | -O0（默认 debug） | -O2（-p buildMode=release） |
|---|---|---|
| conv（NV21→RGBA+旋转） | ~25 ms | **1.42 ms** |
| native 单帧合计 | ~86 ms | ~20 ms |
| 相机帧率 | 7 fps | **20 fps** |

`1.42 ms` 与主机 `-O2` 微基准（`bench_conv.cpp`，1.412 ms）同量级，互为佐证。

这个缺陷持续了很久且**极难发现**：帧率低看起来像「算法慢」或「设备弱」，
很容易引向优化算法，而真正的问题在构建配置里。所以把它做成守卫。

## 判据

读 hvigor 生成的 `build.ninja` 里**真实的** FLAGS，取**最后一个** `-O` 参数，
断言它不是 `-O0`。同时检查是否定义了 `NDEBUG`（debug 下缺失，会让 assert 生效）。

用法：
    python tools/check_native_build_flags.py            # 检查最新构建目录
    python tools/check_native_build_flags.py --mode release
退出码 0 = 通过，1 = 未优化（会导致性能数据失真）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CXX_DIR = REPO_ROOT / "LprDemo" / "entry" / ".cxx"


def open_text(path: Path) -> str:
    """build.ninja 可能是 UTF-8 带/不带 BOM，也可能是 UTF-16。"""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def effective_opt_level(flags: str) -> str | None:
    """返回生效的 -O 参数（最后一个胜出）。"""
    matches = re.findall(r"(?<![\w-])-O([0-3sg]|fast)?\b", flags)
    if not matches:
        return None
    return f"-O{matches[-1]}"


def find_build_ninjas(mode: str | None) -> list[Path]:
    if not CXX_DIR.is_dir():
        return []
    found = []
    for p in CXX_DIR.rglob("build.ninja"):
        # 目录形如 .cxx/default/default/<mode>/<abi>/build.ninja
        parts = p.parts
        if mode and mode not in parts:
            continue
        found.append(p)
    # 新的在前
    return sorted(found, key=lambda q: q.stat().st_mtime, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None,
                    help="只看某个构建模式（debug / release）")
    args = ap.parse_args()

    if not CXX_DIR.is_dir():
        print(f"找不到 {CXX_DIR}")
        print("先跑一次 native 构建（assembleHap）再执行本守卫。")
        return 0

    ninjas = find_build_ninjas(args.mode)
    if not ninjas:
        print(f"在 {CXX_DIR} 下没有 build.ninja"
              + (f"（模式 {args.mode}）" if args.mode else ""))
        return 0

    problems: list[str] = []
    print("native 优化标志检查")
    print("=" * 72)

    for path in ninjas:
        text = open_text(path)
        # 只取 C/C++ 的 FLAGS 行（rules.ninja 里还有链接等其他规则）
        flag_lines = re.findall(r"^\s*FLAGS = (.+)$", text, re.M)
        if not flag_lines:
            continue
        # 同一目录里多个 FLAGS 通常一致，取第一条即可，但要报告唯一形态
        distinct = sorted(set(flag_lines))
        opt = effective_opt_level(distinct[0])
        has_ndebug = any("-DNDEBUG" in f for f in distinct)
        has_o0 = any(effective_opt_level(f) == "-O0" for f in distinct)
        has_o3_but_o0 = has_o0 and any("-O3" in f for f in distinct)

        rel = path.relative_to(REPO_ROOT)
        mode_name = path.parts[-3] if len(path.parts) >= 3 else "?"
        mark = "OK  " if not has_o0 else "FAIL"
        print(f"[{mark}] {mode_name:<8} 生效 {opt or '(未指定)':<8} "
              f"NDEBUG={'有' if has_ndebug else '无'}   {rel}")

        if has_o3_but_o0:
            problems.append(
                f"{rel}\n"
                f"      编译行同时含 -O3 与 -O0，且 **-O0 在后、实际生效**。\n"
                f"      这是 hvigor 按 debug 构建时，CMake 把自己的 -O0 追加在\n"
                f"      build-profile.json5 的 -O3 之后造成的。\n"
                f"      修法：构建时显式指定 release ——\n"
                f"        hvigorw --mode module -p product=default "
                f"-p buildMode=release assembleHap --no-daemon"
            )
        elif has_o0:
            problems.append(
                f"{rel}\n"
                f"      生效优化等级是 -O0。性能数据（延迟 / 帧率）都会严重失真。\n"
                f"      用 -p buildMode=release 重新构建。"
            )

        if distinct:
            print(f"         FLAGS = {distinct[0][:150]}")

    print("=" * 72)
    if problems:
        print(f"\n发现 {len(problems)} 处未优化构建：\n")
        for p in problems:
            print(f"  - {p}\n")
        print("理由：-O0 下 native 单帧 ~86 ms / 相机 7 fps；-O2 下 ~20 ms / 20 fps。")
        print("所有延迟与帧率数字必须先确认本守卫通过，否则不可引用。")
        return 1

    print("\n通过：native 以优化模式构建，性能数据可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
