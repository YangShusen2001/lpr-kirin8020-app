#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本仓库脚本共用的路径解析。

## 为什么有这个文件

2026-09-21：为开源做脱敏时，`sanitize_paths.py` 把各脚本里硬编码的本机路径
替换成了 `<HOME>` / `<REPO>` 之类的占位符 —— **结果 `build.sh` 与 7 个 tools
脚本的默认路径全部失效**（`make_figures.py` 直接抛 WinError 123）。

这是"脱敏"这一动作本身造成的回归。正确的修法不是把路径改回去（那等于放弃开源），
而是**让路径从环境解析**：本机用环境变量或默认值，换一台机器改环境变量即可。

## 用法

    from paths import REPO, PRIOR_WORK, SCRATCH, MSLITE_DIR, DEVECO_HOME

    # 或直接取变量（未设置时返回默认值，并给出提示）
    from paths import resolve, hint
    print(hint())

## 环境变量一览

| 变量 | 含义 | 默认值 |
|---|---|---|
| `LPR_DOC_REPO` | 文档/论文仓库 | `../车牌识别`（与本仓库同级） |
| `LPR_PRIOR_WORK` | 前期工程根 | `~/Desktop/Test` |
| `LPR_SCRATCH` | 临时数据目录 | `~/lpr-data` |
| `MSLITE_DIR` | MindSpore Lite 转换器根 | 见下 |
| `DEVECO_HOME` | DevEco Studio 安装根 | 见下 |
| `LPR_DEVICE_LOG` | 设备日志取回目录 | `~/lpr-data` |

未设置且默认值不存在时，`resolve()` 会返回默认值并在 `hint()` 里列出**哪些路径不存在**，
便于一次性排查，而不是等某个脚本在深处崩掉。
"""
from __future__ import annotations

import os

HOME = os.path.expanduser("~")

# 本仓库根（tools/ 的上一级）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PARENT = os.path.dirname(REPO)

def _find_doc_repo() -> str:
    """定位文档/论文仓库。

    候选顺序（先命中先用）：
      1. 环境变量 `LPR_DOC_REPO`
      2. `<HOME>/Desktop/车牌识别`   ← 本机实际布局
      3. `<本仓库同级>/车牌识别`
      4. `<HOME>/车牌识别`
    找不到就返回 2（让 hint() 报出来，而不是静默取一个错路径）。
    """
    env = os.environ.get("LPR_DOC_REPO")
    if env:
        return env
    cands = [
        os.path.join(HOME, "Desktop", "车牌识别"),
        os.path.join(REPO_PARENT, "车牌识别"),
        os.path.join(HOME, "车牌识别"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


#: 文档/论文仓库。
DOC_REPO = _find_doc_repo()

#: 前期工程根。本工程继承其数据与结论，但不延续其代码库（ADR-0001）。
PRIOR_WORK = os.environ.get("LPR_PRIOR_WORK", os.path.join(HOME, "Desktop", "Test"))

#: 临时数据目录（中间产物、从设备取回的日志）。
SCRATCH = os.environ.get("LPR_SCRATCH", os.path.join(HOME, "lpr-data"))

#: MindSpore Lite 转换器（PC 端，无需华为账号）。
MSLITE_DIR = os.environ.get(
    "MSLITE_DIR", os.path.join("D:", os.sep, "Tools", "mindspore-lite"))

#: DevEco Studio 安装根。
DEVECO_HOME = os.environ.get(
    "DEVECO_HOME", os.path.join("D:", os.sep, "IDE", "DevEco_Studio"))
DEVECO_SDK = os.environ.get("DEVECO_SDK_HOME", os.path.join(DEVECO_HOME, "sdk"))
# ⚠️ 这里**故意不读环境里的 JAVA_HOME**：本机 PATH 上的 JAVA_HOME 指向系统 JDK 1.8，
# 而 hvigor 必须用 DevEco 自带的 jbr（JDK21）——否则读不了 JDK21 生成的 PKCS12
# 密钥库，报 `11014003 Init keystore failed`。用独立变量名避免被宿主的 JAVA_HOME 污染。
DEVECO_JBR = os.environ.get("LPR_JAVA_HOME", os.path.join(DEVECO_HOME, "jbr"))
DEVECO_NODE = os.environ.get("LPR_NODE_HOME", os.path.join(DEVECO_HOME, "tools", "node"))
HVIGOR = os.path.join(DEVECO_HOME, "tools", "hvigor")

#: 具体文件（沿用既有口径）
PRIOR_SHOWCASE = os.path.join(PRIOR_WORK, "lpr-showcase")
PRIOR_SHUSEN = os.path.join(PRIOR_WORK, "ShusenPaper")
PRIOR_LPR_HARMONY = os.path.join(HOME, "lpr-harmony")

_MAP = {
    "LPR_DOC_REPO": DOC_REPO,
    "LPR_PRIOR_WORK": PRIOR_WORK,
    "LPR_SCRATCH": SCRATCH,
    "MSLITE_DIR": MSLITE_DIR,
    "DEVECO_HOME": DEVECO_HOME,
    "DEVECO_SDK_HOME": DEVECO_SDK,
    "LPR_JAVA_HOME": DEVECO_JBR,
    "LPR_NODE_HOME": DEVECO_NODE,
}


def missing() -> list[tuple[str, str]]:
    """返回 (环境变量名, 路径) 中**在磁盘上不存在**的那些。"""
    return [(k, v) for k, v in _MAP.items() if not os.path.exists(v)]


def hint() -> str:
    """一行诊断：哪些路径需要用户用环境变量覆盖。"""
    miss = missing()
    if not miss:
        return "paths: 全部路径存在。"
    lines = ["paths: 以下默认路径不存在，请用环境变量覆盖（见 tools/paths.py 顶部表格）："]
    lines += [f"  {k} = {v}" for k, v in miss]
    return "\n".join(lines)


if __name__ == "__main__":
    for k, v in _MAP.items():
        mark = "OK " if os.path.exists(v) else "MISS"
        print(f"{mark} {k:18s} {v}")
    print()
    print(hint())
