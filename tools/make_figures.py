#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从证据文件生成论文与网页共用的图。

## 为什么要脚本而不是手工画

1. **可复现**：证据重跑后，图跟着重跑，不会出现「图是新主张、数据是旧实验」。
2. **数字同源**：每个画上去的点都从 evidence 里读，与 `verify_published_numbers.py`
   用同一批文件。手工画图是「表格与正文分头维护」那类错误的温床。
3. **一次出两版**：PNG（网页/PPT）+ PDF（LaTeX 矢量）。

**旧论文稿（`lpr-showcase/paper/figures/`）的 7 张图属于旧主张（移植保真），不可复用。**

## 用法

    python tools/make_figures.py                       # 全部，输出到默认目录
    python tools/make_figures.py --out <dir>           # 指定输出目录
    python tools/make_figures.py --only fig3 fig6      # 只出某几张
    python tools/make_figures.py --rq4 evidence/rq4_thermal_80r.csv

标签**一律英文** —— 论文是英文稿，且可避开 matplotlib 的 CJK 字体依赖。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认输出到**本仓库**的 `figures/`。
#
# 2026-09-21 改：原先硬编码了文档仓库的绝对路径，被 `sanitize_paths.py` 替换成
# `<REPO>/...` 占位符后**脚本直接跑不起来**（WinError 123）——脱敏本身造成的回归。
# 现在默认值不依赖本机布局；给论文出图时显式传 `--out`。
DEFAULT_OUT = os.environ.get("LPR_FIGURE_OUT", os.path.join(ROOT, "figures"))

# 统一风格：白底、无多余边框、字号偏大（论文缩印后仍可读）
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

# 全项目统一色板：NPU 蓝、CPU 灰、CANN 绿、强调橙
C_NPU, C_CPU, C_CANN, C_ACC, C_BAD = "#1f6feb", "#8b949e", "#2da44e", "#d29922", "#cf222e"


def load_rows(rel: str) -> list[str]:
    with io.open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


KV = re.compile(r"(\w+)=([^\s]*)")


def kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV.finditer(line)}


def save(fig, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ---------------------------------------------------------------- fig1 架构

def fig1(out: str) -> None:
    """四级流水线 + 落点自证。纯示意图，不含数据。"""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)

    stages = [
        ("Detection", "CPU", C_CPU, "19.61 ms"),
        ("Rectify", "CPU", C_CPU, "0.49 ms"),
        ("Recognition", "NPU", C_NPU, "3.98 ms"),
        ("Colour\n(pixel)", "CPU", C_CPU, "0.08 ms"),
    ]
    w, gap = 17.5, 4.0
    for i, (name, backend, colour, ms) in enumerate(stages):
        x = 3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 15), w, 11, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=1.2, edgecolor=colour,
                                    facecolor=colour + "1a"))
        ax.text(x + w / 2, 22.4, name, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, 19.4, backend, ha="center", va="center", fontsize=8.5, color=colour)
        ax.text(x + w / 2, 17.0, ms, ha="center", va="center", fontsize=8, color="#555555")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 20.5), (x + w + gap, 20.5),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.1, color="#444444"))

    # 落点自证条
    ax.add_patch(FancyBboxPatch((3, 3), 94, 8, boxstyle="round,pad=0.6,rounding_size=1.2",
                                linewidth=1.0, edgecolor="#444444", facecolor="#f6f8fa"))
    ax.text(5, 8.4, "Landing self-evidencing (logged per inference)",
            fontsize=8.5, weight="bold", va="center")
    ax.text(5, 5.4,
            'req=nnrt   LANDED=NNRT:NPU_ohos.boot.hardware.kirin8020_v2_0   fallback=   '
            'fingerprint  l2=4.0489',
            fontsize=7.6, va="center", family="DejaVu Sans Mono", color="#24292f")
    ax.text(50, 31.0, "4-stage Chinese licence-plate pipeline on Kirin 8020",
            ha="center", fontsize=10, weight="bold")
    save(fig, out, "fig1-pipeline-and-landing")


# ---------------------------------------------------------------- fig2 算子覆盖

def fig2(out: str) -> None:
    """算子覆盖：NNRT 通路（36 探针，部分被拒）vs CANN 通路（51 探针，全通过）。"""
    path = os.path.join(ROOT, "models_om_ops/op_collide.csv")
    with io.open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    cann_n = len(rows)
    cann_pass = sum(1 for r in rows if r["run_rc"] == "0" and r["build_rc"] == "0")

    # NNRT 侧来自 ShusenPaper 的 L1 矩阵（36 算子三信号判定）：独立可编译 20 / 被拒 16
    nnrt_total, nnrt_ok = 36, 20
    nnrt_rejected = nnrt_total - nnrt_ok

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax1.barh(["NNRT\n(MindSpore Lite)", "CANN\n(OMG .om)"],
             [nnrt_ok / nnrt_total * 100, cann_pass / cann_n * 100],
             color=[C_CPU, C_CANN], height=0.5)
    for y, (ok, tot) in enumerate([(nnrt_ok, nnrt_total), (cann_pass, cann_n)]):
        ax1.text(ok / tot * 100 + 2, y, f"{ok}/{tot}", va="center", fontsize=9, weight="bold")
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("Operators admitted to the NPU (%)")
    ax1.set_title("Same operator set, two toolchains")
    ax1.grid(axis="y", visible=False)

    # 右：被 NNRT 拒、CANN 通过的那 9 个
    rejected = ["relu", "sigmoid", "softmax", "maxpool", "pad",
                "cast_f16", "transpose", "resize", "tanh"]
    ax2.set_axis_off()
    ax2.set_title("Rejected by NNRT, accepted by CANN", fontsize=9.5)
    ax2.text(0.0, 0.86, "9 operators:", fontsize=8.5, weight="bold")
    for i, name in enumerate(rejected):
        ax2.text(0.02 + (i % 3) * 0.34, 0.66 - (i // 3) * 0.20, f"· {name}",
                 fontsize=8.2, family="DejaVu Sans Mono")
    ax2.text(0.0, 0.02,
             "ConvTranspose: cannot even be converted\non the NNRT side.",
             fontsize=8.0, color=C_ACC, weight="bold")
    save(fig, out, "fig2-operator-coverage")


# ---------------------------------------------------------------- fig3 落点敏感性

def fig3(out: str) -> None:
    """分层答案：换后端几乎不动输出，但后端本身是准确率变量；牌长代价更大。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：分歧率（真实整车场景，n=1000）
    labels = ["rec backend\n(recogniser)", "det backend\n(detector)"]
    div = [0.0, 0.1]
    bars = ax1.bar(labels, div, color=[C_NPU, C_NPU], width=0.45)
    for b, v in zip(bars, div):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.004,
                 f"{v}%\n({int(v * 10)}/1000)", ha="center", fontsize=8.5, weight="bold")
    ax1.set_ylim(0, 0.20)
    ax1.set_ylabel("Outputs changed by switching backend (%)")
    ax1.set_title("Real vehicle scenes (n=1000):\nbackend barely moves the output")
    ax1.grid(axis="x", visible=False)

    # 右：三档影响对比（百分点）
    items = [("rec NPU vs CPU\n(p<0.0001)", +3.2, C_NPU),
             ("device NPU vs host\n(p=0.0156)", -0.7, C_CPU),
             ("8-char vs 7-char plate\n(province-matched)", -6.6, C_ACC)]
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    colours = [i[2] for i in items]
    y = range(len(items))
    ax2.barh(list(y), vals, color=colours, height=0.5)
    ax2.axvline(0, color="#24292f", linewidth=0.8)
    for i, v in enumerate(vals):
        ax2.text(v + (0.25 if v > 0 else -0.25), i, f"{v:+.1f} pp",
                 va="center", ha="left" if v > 0 else "right",
                 fontsize=8.5, weight="bold")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(names, fontsize=7.8)
    ax2.set_xlim(-9.5, 6.5)
    ax2.set_xlabel("Accuracy delta (percentage points)")
    ax2.set_title("What actually moves accuracy")
    ax2.grid(axis="y", visible=False)
    ax2.invert_yaxis()

    save(fig, out, "fig3-backend-sensitivity")


# ---------------------------------------------------------------- fig4 错误位

def fig4(out: str) -> None:
    """逐位替换错误：省份位是主战场，三个工况一致。"""
    def per_position(rel: str, prefix: str, code_key: str) -> list[int]:
        sub = [0] * 8
        for line in load_rows(rel):
            if not line.startswith(prefix + " ") or "file=" not in line:
                continue
            d = kv(line)
            gt, code = d.get("gt", ""), d.get(code_key, "")
            if len(gt) != len(code) or not gt:
                continue
            for i, (a, b) in enumerate(zip(gt, code)):
                if a != b:
                    sub[i] += 1
        return sub

    a = per_position("evidence/crop_bare.log", "BARE", "code_np")          # T11 裁剪图裸喂
    b = per_position("evidence/scene_green_rec.log", "SCENE", "code_np")   # T13 8 字符真实场景

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # 左：T13 的逐位分布（绝对值）
    x = range(8)
    bars = ax1.bar(x, b, color=[C_ACC] + [C_CPU] * 7, width=0.62)
    for i, v in enumerate(b):
        if v:
            ax1.text(i, v + 0.6, str(v), ha="center", fontsize=8, weight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(["prov\n(0)", "1", "2", "3", "4", "5", "6", "7"], fontsize=8)
    ax1.set_xlabel("Character position")
    ax1.set_ylabel("Substitution errors")
    ax1.set_ylim(0, max(b) * 1.25)
    tot = sum(b)
    ax1.set_title(f"8-char new-energy plates (n=971 equal-length)\n"
                  f"province = {b[0]}/{tot} = {100 * b[0] / tot:.1f}%")
    ax1.grid(axis="x", visible=False)

    # 右：两个工况的归一化形状对比
    w = 0.38
    ax2.bar([i - w / 2 for i in x], [v / sum(a) * 100 for v in a], w,
            label="T11 cropped, bare-fed", color=C_CPU)
    ax2.bar([i + w / 2 for i in x], [v / sum(b) * 100 for v in b], w,
            label="T13 real scene, 8-char", color=C_ACC)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(["prov", "1", "2", "3", "4", "5", "6", "7"], fontsize=8)
    ax2.set_xlabel("Character position")
    ax2.set_ylabel("Share of substitution errors (%)")
    ax2.set_title("Province position dominates in both regimes")
    ax2.legend(loc="upper right")
    ax2.grid(axis="x", visible=False)

    save(fig, out, "fig4-error-position")


# ---------------------------------------------------------------- fig5 RQ4

def fig5(out: str, rel: str) -> None:
    """RQ4 持续负载：延迟时间线 + 热档/电池，指纹恒定。

    会**主动标出**轮间隔异常的停顿（> 60 s）。理由：这类停顿意味着探针被系统冻结过
    （`FROZEN_AFTER_ENTER_BG`），那段数据不是"持续负载"，画图时若不标出来，
    读者会把它当成真实的延迟尖峰。
    """
    with io.open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t_s"]) / 60 for r in rows]
    ms = [float(r["totalMs"]) for r in rows]
    batt = [float(r["battC"]) for r in rows]
    thermal = [int(r["thermal"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1]})

    # 标出异常停顿
    gaps = [(t[i], t[i + 1] - t[i]) for i in range(len(t) - 1) if t[i + 1] - t[i] > 1.0]
    for x, g in gaps:
        ax1.axvspan(x, x + g, color=C_BAD, alpha=0.12, zorder=0)
        ax1.text(x + g / 2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
                 f"stall {g * 60:.0f} s", fontsize=7, color=C_BAD,
                 ha="center", va="top")

    ax1.plot(t, ms, "-o", ms=2.5, linewidth=0.9, color=C_NPU)
    p50 = statistics.median(ms)
    ax1.axhline(p50, color=C_ACC, linewidth=1.0, linestyle="--",
                label=f"p50 = {p50:.1f} ms")
    ax1.set_ylabel("End-to-end latency (ms)")
    ax1.set_title(f"RQ4 sustained load: {len(rows)} rounds, "
                  f"{t[-1]:.0f} min, thermal level {sorted(set(thermal))}")
    ax1.legend(loc="upper left")
    fp = [k for k in ("det_l2", "rec_l2", "code") if k in rows[0]]
    const = all(len({r[k] for r in rows}) == 1 for k in fp)
    ax1.text(0.99, 0.03,
             f"fingerprint {'/'.join(fp)}\n"
             + ("constant in all rounds" if const else "NOT constant"),
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.6,
             color="#57606a" if const else C_BAD)

    ax2.plot(t, batt, "-o", ms=2.5, linewidth=0.9, color=C_CPU, label="battery (°C)")
    ax2.set_ylabel("Battery (°C)", color=C_CPU)
    ax2.set_xlabel("Elapsed (min)")
    ax2.set_ylim(min(batt) - 1.5, max(batt) + 1.5)
    ax2b = ax2.twinx()
    ax2b.step(t, thermal, where="post", color=C_BAD, linewidth=1.0, label="thermal level")
    ax2b.set_ylabel("Thermal level", color=C_BAD)
    ax2b.set_ylim(0, 6)
    ax2b.grid(False)
    save(fig, out, "fig5-rq4-sustained")


# ---------------------------------------------------------------- fig6 C8

def fig6(out: str) -> None:
    """C8：隔离基准 ≠ 流水线内成本。剂量-反应 + 忙等对照 + 线程数对照。"""
    pat = re.compile(r"LANDED=CPU:t(\d) gapMs=([\d.]+) polluteKB=(\d+) spinMs=([\d.]+) p50=([\d.]+)")
    # 只取 det 模型（y5fu_320x，基线最稳）的 10 条
    rows: list[tuple[int, float, int, float, float]] = []
    for line in load_rows("evidence/camera_gap_sweep.log"):
        if "y5fu_320x_head_fp32.ms" not in line:
            continue
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5))))

    def pick(threads: int, gap: float, spin: float = 0.0, pollute: int = 0) -> float:
        for th, g, p, s, p50 in rows:
            if th == threads and abs(g - gap) < 0.01 and abs(s - spin) < 0.01 and p == pollute:
                return p50
        raise KeyError((threads, gap, spin, pollute))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # 左：剂量-反应
    gaps = [0, 8, 33]
    t4 = [pick(4, g) for g in gaps]
    t1 = [pick(1, g) for g in gaps]
    ax1.plot(gaps, t4, "-o", color=C_NPU, label="4 threads (default)")
    ax1.plot(gaps, t1, "-o", color=C_CPU, label="1 thread")
    ax1.set_xlabel("Idle gap between calls, gapMs (ms)")
    ax1.set_ylabel("p50 (ms)")
    ax1.set_title("Dose–response: sparse calls are slower")
    ax1.legend(loc="upper left")
    ax1.set_xticks(gaps)
    # 流水线观测区间
    ax1.axhspan(15.26, 31.56, color=C_ACC, alpha=0.12, zorder=0)
    ax1.text(0.5, 30.4, "in-pipeline range", fontsize=7.4, color=C_ACC)

    # 右：忙等 + 线程数对照
    labels = ["gap 0\n(baseline)", "gap 33\n(sleep)", "spin 33\n(busy-wait)", "pollute\n1.2 MB"]
    t4v = [pick(4, 0), pick(4, 33), pick(4, 0, spin=33), pick(4, 0, pollute=1200)]
    t1v = [pick(1, 0), pick(1, 33), pick(1, 0, spin=33), pick(1, 0, pollute=1200)]
    x = range(len(labels))
    w = 0.36
    ax2.bar([i - w / 2 for i in x], t4v, w, color=C_NPU, label="4 threads")
    ax2.bar([i + w / 2 for i in x], t1v, w, color=C_CPU, label="1 thread")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=7.4)
    ax2.set_ylabel("p50 (ms)")
    ax2.set_title("Busy-wait fixes it only at 1 thread")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)
    ax2.annotate("", xy=(2 - w / 2, t4v[2]), xytext=(2 - w / 2, t4v[0]),
                 arrowprops=dict(arrowstyle="<->", color=C_BAD, linewidth=1.0))
    ax2.text(2 - w / 2 - 0.06, (t4v[0] + t4v[2]) / 2, f"+{t4v[2] - t4v[0]:.1f}",
             fontsize=7.6, color=C_BAD, ha="right", weight="bold")
    ax2.text(2 + w / 2 + 0.06, t1v[2] + 1.2, f"+{t1v[2] - t1v[0]:.1f}",
             fontsize=7.6, color=C_BAD, ha="left", weight="bold")

    save(fig, out, "fig6-c8-dose-response")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rq4", default="evidence/rq4_thermal.csv")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    print(f"输出目录: {args.out}")
    jobs = {"fig1": lambda: fig1(args.out),
            "fig2": lambda: fig2(args.out),
            "fig3": lambda: fig3(args.out),
            "fig4": lambda: fig4(args.out),
            "fig5": lambda: fig5(args.out, args.rq4),
            "fig6": lambda: fig6(args.out)}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        try:
            fn()
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name} 失败: {type(exc).__name__}: {exc}")
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
