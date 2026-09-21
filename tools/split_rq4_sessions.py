"""把一份多会话 RQ4 日志按 `RQ4 BEGIN` 标记切成**每次独立运行的独立文件**。

## 为什么需要它

`parse_rq4.py` 的键是**轮号 `r`**（`rows[r] = ...`）。一份日志里跑两次，
第二段的 `r=0..39` 会直接覆盖第一段的 `r=0..39`，**最后只剩一段**，
而且不会有任何报错 —— 这是「写固定键」这一类工具第 N 次咬人，
与 `parse_rq4.py` 曾覆盖固定文件名、`build.sh` 硬编码 debug 同族。

## 会话的定义（本脚本采用的口径）

一次「独立运行」= 一个 `RQ4 BEGIN` 到下一个 `RQ4 BEGIN`（或文件尾）之间的全部行。

**为什么用 `BEGIN` 当边界而不是 `END`**：那份日志里只出现了 1 个 `END`
（第一段 80 轮跑满后写的），第二段被打断时**没有写 `END`** ——
App 被系统冻结/杀掉时不会有机会写收尾行。以 `END` 为边界会把
「被打断的运行」整体丢掉，而以 `BEGIN` 为边界能把它们如实保留下来。

## 用法

    python tools/split_rq4_sessions.py <log> <outdir> [--prefix rq4_run]

输出：
    <outdir>/<prefix>_s1.log   ← 第 1 次运行（原始行，可再喂给 parse_rq4.py）
    <outdir>/<prefix>_s2.log
    ...
    <outdir>/<prefix>_manifest.json  ← 每次运行的 ts / 起止轮数 / 时长 / 是否跑满
"""
from __future__ import annotations

import json
import os
import re
import sys

RE_BEGIN = re.compile(r"RQ4 BEGIN minutes=(\d+) rounds=(\d+) intervalMs=(\d+) "
                      r"frame=(\S+) ts=(\d+)")
RE_END = re.compile(r"RQ4 END rounds=(\d+) minutes=([\d.]+) ts=(\d+)")
RE_R = re.compile(r"RQ4 r=(\d+) t=(\d+) ")


def open_log(path: str):
    """device 日志经 hdc 取回可能是 UTF-16（带 BOM）。"""
    with open(path, "rb") as fh:
        head = fh.read(2)
    enc = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return open(path, encoding=enc, errors="replace")


def split(raw: str, outdir: str, prefix: str = "rq4_run") -> list[dict]:
    lines = open_log(raw).read().splitlines()

    # 切分点 = 每条 BEGIN 所在的行号
    starts = [i for i, l in enumerate(lines) if RE_BEGIN.search(l)]
    if not starts:
        raise SystemExit(f"!! {raw} 里没有 `RQ4 BEGIN` 标记，无法切分。"
                         f"（该文件可能是单会话旧格式，直接用 parse_rq4.py）")
    bounds = starts + [len(lines)]

    os.makedirs(outdir, exist_ok=True)
    manifest = []
    for si in range(len(starts)):
        chunk = lines[bounds[si]:bounds[si + 1]]
        name = f"{prefix}_s{si + 1}"
        out = os.path.join(outdir, name + ".log")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(chunk) + "\n")

        b = RE_BEGIN.search(chunk[0])
        rounds = [int(m.group(1)) for m in
                  (RE_R.search(l) for l in chunk) if m]
        ts = [int(m.group(2)) for m in
              (RE_R.search(l) for l in chunk) if m]
        e = next((RE_END.search(l) for l in chunk if RE_END.search(l)), None)
        adv = (m.group(1), m.group(2)) if (m := RE_END.search(
            next((l for l in chunk if RE_END.search(l)), ""))) else None

        manifest.append({
            "session": si + 1,
            "file": os.path.basename(out),
            "ts_begin": int(b.group(5)),
            "planned_rounds": int(b.group(2)),
            "planned_minutes": int(b.group(1)),
            "interval_ms": int(b.group(3)),
            "frame": b.group(4),
            "rounds_captured": len(rounds),
            "round_first": rounds[0] if rounds else None,
            "round_last": rounds[-1] if rounds else None,
            "t_last_s": ts[-1] if ts else None,
            # 有 END 行 = 探针自己跑满并主动收尾；没有 = 被系统打断。
            "ended_cleanly": e is not None,
            "end_declared_rounds": int(e.group(1)) if e else None,
            "end_declared_minutes": float(e.group(2)) if e else None,
        })
        print(f"  {name}: 轮 {manifest[-1]['round_first']}~"
              f"{manifest[-1]['round_last']} "
              f"(n={len(rounds)}, t={manifest[-1]['t_last_s']}s)"
              f"{'' if e else '  ← 无 END，被打断'}")

    mf = os.path.join(outdir, f"{prefix}_manifest.json")
    with open(mf, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f"-> {mf}")
    return manifest


if __name__ == "__main__":
    args = list(sys.argv[1:])
    pre = "rq4_run"
    if "--prefix" in args:
        i = args.index("--prefix")
        pre = args[i + 1]
        del args[i:i + 2]
    if len(args) < 2:
        raise SystemExit(__doc__)
    print(f"=== 切分 {args[0]} ===")
    split(args[0], args[1], pre)
