#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从证据文件生成论文与网页共用的图。

## 为什么按 3.5 英寸设计（2026-09-22 返工）

第一版全部按 **7.0 英寸**宽设计，而 IEEEtran 单栏只有 **3.5 英寸**。
塞进去等于整体缩到 50%：9pt 的字实际印出 **4.5pt**，于是标题互相压、
文字冲出框、相邻子图挤在一起。

**根因不是「字号没调够」，是「按错误的尺寸设计」。**
现在统一按 `COL_W = 3.5` 英寸设计、1:1 呈现，字号即为最终字号。

## 为什么自带 linter

图我看不了，所以**排版问题必须程序化检出**。matplotlib 知道每个 `Text` 的
bbox，因此「文字溢出图框」与「文字互相重叠」都能自动断言 —— 见 `lint()`。
每次 `save()` 都会跑，有问题就打印并让脚本退出码非零。

### 这个 linter 我写错了三版（2026-09-22，务必别再踩）

**一版错在坐标系**：拿 `get_window_extent()`（屏幕坐标）去比 `fig.bbox`，
于是把 `ylabel` 那类「天然落在 figure 外 0.17in」的**正常布局**全判成"出画布"。

**二三版错在「预测画布」**：试图用「文字 bbox 并集 + 平移」算出 tight 之后的画布。
实测误差 **0.29 / 0.10 / 0.06 in** —— 全错。根因两条：
1. `tight` 的框还含 **patch / 图例框**，文字 bbox 测不到它们（fig1 就被 patch 撑宽）；
2. 我写的平移量是**自指**的（`off_x` 里含 `w_in`，`w_in` 里又含 `x0`）。

**还踩了两个假阳性**：
- `set_axis_off()` 之后**刻度对象依然存在**，不做 `ax.axison` 判断就会把隐藏的
  `0/10/.../100` 当成真文字（fig1 报的 `-0.19~+3.57 in` 就是它）；
- twinx 的刻度是**不同对象、相同位置**，按 `id` 去重抓不到，得按 `(文字, bbox)`。

**结论：不要预测画布，让画布按构造确定。**
`save()` 已改成**不用** `bbox_inches='tight'`、画布 = `figsize`，
再用 `assert_canvas()` 直接量产物 PDF 的 `/MediaBox` 断言。
`MediaBox` 就是 LaTeX 真正拿去缩放的东西 —— 量它没有中间假设，**这是唯一不会自欺的检查**。

## 用法

    python tools/make_figures.py                       # 全部
    python tools/make_figures.py --out <dir>
    python tools/make_figures.py --only fig3 fig6
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
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认输出到**本仓库**的 `figures/`。给论文出图时显式传 `--out`。
DEFAULT_OUT = os.environ.get("LPR_FIGURE_OUT", os.path.join(ROOT, "figures"))

#: IEEEtran 单栏宽度（英寸）。所有图都按这个宽度设计，1:1 呈现。
COL_W = 3.5

# 统一风格。字号就是最终印出的字号 —— 因为按 1:1 呈现。
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "legend.frameon": False,
    "legend.fontsize": 6,
    "legend.handlelength": 1.2,
    "legend.borderpad": 0.2,
    "lines.linewidth": 0.9,
    "lines.markersize": 2.2,
})

C_NPU, C_CPU, C_CANN, C_ACC, C_BAD = "#1f6feb", "#8b949e", "#2da44e", "#d29922", "#cf222e"

LINT_PROBLEMS: list[str] = []


# ---------------------------------------------------------------- linter

def _texts(fig) -> list[tuple[str, object]]:
    """收集所有**真正会被印出来**的 Text。

    两个必须过滤掉的假阳性（第一版 linter 就是被它们骗了）：

    1. **twinx 的刻度标签是同一批对象**。`ax.get_xticklabels()` 在 twin 轴上返回的
       就是主轴那些 Text 对象 —— 不去重就会被判成"自己和自己重叠"。
    2. **视图范围外的候选刻度**。matplotlib 会为 `xlim=(0,23)` 的轴保留 `-5` 这样的
       候选刻度 Text（位置在 x0=-4.3，已经出了图框），它并不会被画出来。
    """
    seen: set[int] = set()
    out: list[tuple[str, object]] = []

    def add(t, ax=None, axis=None) -> None:
        if t is None or id(t) in seen:
            return
        txt = t.get_text().strip()
        if not txt or not t.get_visible():
            return
        if ax is not None and axis is not None:
            # ★ 轴被关掉时（fig1 的 `set_axis_off()`），**刻度对象依然存在**，
            #   但它们一个都不会被画出来。不做这个判断就会把隐藏的
            #   `0/10/.../100` 当成真文字 → 误报「超出画布」。这就是 fig1
            #   报 -0.19~+3.57 in 的真身（其实它是空轴刻度，不是内容）。
            if not (ax.axison and getattr(ax, f"{axis}axis").get_visible()):
                return
            pos = t.get_position()
            if axis == "x":
                lo, hi = sorted(ax.get_xlim())
                if not (lo - 1e-9 <= pos[0] <= hi + 1e-9):
                    return
            else:
                lo, hi = sorted(ax.get_ylim())
                if not (lo - 1e-9 <= pos[1] <= hi + 1e-9):
                    return
        seen.add(id(t))
        out.append((txt.replace("\n", " ")[:34], t))

    for ax in fig.get_axes():
        add(ax.title)
        add(ax.xaxis.label)
        add(ax.yaxis.label)
        for t in ax.texts:
            add(t)
        for t in ax.get_xticklabels():
            add(t, ax, "x")
        for t in ax.get_yticklabels():
            add(t, ax, "y")
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                add(t)
    for t in fig.texts:
        add(t)
    return out


def lint(fig, name: str) -> tuple[list[str], list[tuple[str, object]], float]:
    """检出排版问题，判据是「**印出来是什么样**」。

    ## 坐标系（这里连着栽了三次，务必先读这段）

    `bbox_inches='tight'` 把**所有文字的并集**（+`pad_inches`）重新定义成画布。
    三条**实测**结论（2026-09-22，哨兵实验）：

    | 实验 | 结果 | 含义 |
    |---|---|---|
    | `fig.text(0,…)` + `fig.text(1,…)` | 3.659 in = 3.5+0.16 | 画布外文字**会**被算进并集，框是 `[-0.08, 3.58]` |
    | `subplots_adjust(left=0.05)` 挤压 | 3.572 in，高度还变小 | tight **受边距影响**，ylabel **在**并集里 |

    所以：

        w_saved = (max_x1 - min_x0) + 2 * pad_inches      ← 单位英寸，屏幕坐标/dpi

    前两版的错误：**先假定框是 `[0, 3.5]`**，再用「并集 + 平移」去反推，
    结果是自指的（`off_x` 里含 `w_in`，`w_in` 里含 `x0`），既漏了真问题
    又把正常布局报成「溢出」。**别再用平移模型，只用上面这一个公式。**

    ## 三类判据

    1. **左右文字重叠** —— 用户报的「重叠 / 差点交叉」。
       去重按 `(文字, 取整 bbox)`：twinx 的刻度是**不同对象、相同位置**，按 id 去重抓不到。
    2. **宽度不是单栏宽** —— 保存后画布宽 ≠ `COL_W` ⇒ LaTeX 会缩放图片 ⇒
       字要么被放大（发虚、和正文不搭），要么被缩小（回到原来那个 4.5pt 的问题）。
       容差 ±2%。
    3. **文字越过单栏宽** —— 注意：**这跟「出图框」不是一回事**。`ylabel` 落在
       名义框外是 matplotlib 的**正常布局**，tight 会把它包进来，不算问题；
       有问题的是**内容比单栏还宽** —— 那才会被 LaTeX 压小。

    返回 `(问题列表, 文字项, 保存后画布宽)`。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items: list[tuple[str, object]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for label, t in _texts(fig):
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:                                    # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        key = (label, round(bb.x0), round(bb.y0), round(bb.x1), round(bb.y1))
        if key in seen:                                      # twinx 同位置重复
            continue
        seen.add(key)
        items.append((label, bb))

    probs: list[str] = []
    if not items:
        return probs, items, 0.0

    PAD = 0.0                                                # save() 不再用 tight ⇒ 无 pad
    x0 = min(bb.x0 for _, bb in items) / fig.dpi
    x1 = max(bb.x1 for _, bb in items) / fig.dpi
    w_saved = (x1 - x0) + 2 * PAD

    # 2) 内容必须落在单栏宽内。
    #    save() 已改成 `bbox_inches=None`，画布**按构造** = figsize = COL_W，
    #    所以这里判的是「文字有没有比画布还宽」—— 一旦超了就真的会被裁掉。
    if x1 - x0 > COL_W + 0.02 * COL_W or x0 < -0.02 * COL_W or x1 > COL_W * 1.02:
        widest = max(items, key=lambda it: it[1].x1 - it[1].x0)
        probs.append(f"{name}: 内容横向 {x0:+.2f}~{x1:+.2f}in 超出画布 "
                     f"0~{COL_W}in（最宽文字 {widest[0]!r}）")

    # 1) 文字互相重叠（内缩容差：相邻刻度天然贴得近，不算问题）
    TOL = 1.5
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0) - TOL
            oy = min(a.y1, b.y1) - max(a.y0, b.y0) - TOL
            if ox > 0 and oy > 0:
                probs.append(f"{name}: 重叠 → {items[i][0]!r} × {items[j][0]!r}")
    return probs, items, w_saved


def probe(fig, name: str) -> None:
    """调试用：打印保存后的真实画布尺寸 + 内容并集宽度。"""
    probs, items, w_saved = lint(fig, name)
    x0 = min(bb.x0 for _, bb in items) / fig.dpi
    x1 = max(bb.x1 for _, bb in items) / fig.dpi
    print(f"  [probe] {name}  设计 {fig.get_figwidth():.2f}x"
          f"{fig.get_figheight():.2f}in → 内容并集 {x1 - x0:.3f}in"
          f" → 保存后 {w_saved:.3f}in（单栏 {COL_W}）")
    for p in probs:
        print(f"      {p}")


def boxed_text(ax, x, y, s, fontsize=5.0, pad_px=3.0, **kw):
    """先量文字宽度、再画刚好包住它的框。

    这样就不会出现「文字伸出框框」—— 框是跟着文字走的，不是反过来。
    """
    t = ax.text(x, y, s, fontsize=fontsize, family="DejaVu Sans Mono",
                va="center", ha="left", **kw)
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    inv = ax.transData.inverted()
    (x0, y0) = inv.transform((bb.x0 - pad_px, bb.y0 - pad_px))
    (x1, y1) = inv.transform((bb.x1 + pad_px, bb.y1 + pad_px))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=0.7, edgecolor="#444444",
                                facecolor="#f6f8fa", zorder=0))
    t.set_zorder(3)
    return t


def save(fig, out: str, name: str) -> None:
    """保存并 lint。

    ## 为什么**不用** `bbox_inches='tight'`（第三次返工的关键决定）

    tight 会把「所有艺术家的并集」重算成画布，于是**画布宽变成不可预测的**：
    实测同一套 `figsize=(3.5, …)` 出来 2.75~3.29 in，取决于图里有没有 patch、
    ylabel 多宽、图例在哪。结果是 LaTeX 把图放大 107~127%，字**发虚、和正文不搭**。

    而且它**无法在保存前算准**：文字 bbox 的并集 ≠ tight 的框（fig1 里有
    `FancyBboxPatch`，patch 撑出的宽度文字测不到）。我先后试了三种「并集 + 平移」
    模型，误差最大 0.29 in —— 全是错的。**能算准的只有「不用 tight」。**

    改法：各图自己用 `subplots_adjust` 定边距，`bbox_inches=None` 让画布就等于
    `figsize`。这样宽度**按构造**等于 `COL_W`，LaTeX 1:1 呈现、字号即所见。
    """
    if not getattr(fig, "_lpr_margins_set", False):
        # 兜底：没显式调过边距的图，用 tight_layout 排一次内部间距。
        # （twinx / add_patch 的图会有 UserWarning，但边距仍是合法的）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(pad=0.25)
    probs, _, _ = lint(fig, name)
    if os.environ.get("FIG_PROBE"):
        probe(fig, name)
    if probs:
        LINT_PROBLEMS.extend(probs)
        print(f"  {name}: **{len(probs)} 个排版问题**")
        for p in probs[:8]:
            print(f"      {p}")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    # ★ 交付前对**产物本身**断言：不动产物文件才是真证据，别只信内存里的推断。
    probs += assert_canvas(os.path.join(out, f"{name}.pdf"), name)
    if probs:
        LINT_PROBLEMS.extend(probs)
        for p in probs:
            print(f"      {p}")
    if not probs:
        print(f"  {name}.png / .pdf  (lint OK)")


def assert_canvas(pdf: str, name: str) -> list[str]:
    """直接量 PDF 的 `/MediaBox`，断言画布宽 == 单栏宽。

    这是**唯一不会自欺的检查**：前面三版 linter 都在「保存前推断」，
    分别错了 0.29 / 0.10 / 0.06 in，还报过 40 个假阳性。
    MediaBox 是 LaTeX 真正拿去缩放的东西，量它没有中间假设。
    """
    try:
        with open(pdf, "rb") as f:
            raw = f.read()
    except OSError as exc:                                   # pragma: no cover
        return [f"{name}: 读不到 {os.path.basename(pdf)}（{exc}）"]
    m = re.search(rb"/MediaBox\s*\[([\d\.\s\-]+)\]", raw)
    if not m:
        return [f"{name}: PDF 里找不到 /MediaBox"]
    v = [float(x) for x in m.group(1).split()]
    w_pt, h_pt = v[2] - v[0], v[3] - v[1]
    w_in = w_pt / 72
    if abs(w_in - COL_W) > 0.02 * COL_W:
        return [f"{name}: 产物画布 {w_in:.3f}in ≠ 单栏 {COL_W}in ⇒ "
                f"LaTeX 会放大 {COL_W / w_in:.0%}"]
    return []


# ---------------------------------------------------------------- 数据

KV = re.compile(r"(\w+)=([^\s]*)")


def read_lines(rel: str) -> list[str]:
    with io.open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV.finditer(line)}


# ---------------------------------------------------------------- fig1

def fig1(out: str) -> None:
    """四级流水线 + 落点自证记录。单栏 3.5in。"""
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    # axis off 的示意图，tight_layout 推不动边距 ⇒ 显式钉死，保证 xlim 铺满单栏宽
    fig.subplots_adjust(left=0.004, right=0.996, top=0.98, bottom=0.02)
    fig._lpr_margins_set = True

    stages = [("Detection", "CPU", C_CPU, "19.6 ms"),
              ("Rectify", "CPU", C_CPU, "0.5 ms"),
              ("Recognition", "NPU", C_NPU, "4.0 ms"),
              ("Colour", "CPU", C_CPU, "0.1 ms")]
    w, gap = 19.0, 8.0
    for i, (nm, be, col, ms) in enumerate(stages):
        x = 2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 23), w, 16,
                                    boxstyle="round,pad=0.4,rounding_size=1.0",
                                    linewidth=1.0, edgecolor=col,
                                    facecolor=col + "1a"))
        ax.text(x + w / 2, 34.6, nm, ha="center", va="center", fontsize=6.2,
                weight="bold")
        ax.text(x + w / 2, 30.2, be, ha="center", va="center", fontsize=5.8,
                color=col)
        ax.text(x + w / 2, 26.2, ms, ha="center", va="center", fontsize=5.2,
                color="#555555")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 31.0), (x + w + gap, 31.0),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.9, color="#444444"))
    # 标题必须短到单栏放得下：实测 "4-stage Chinese licence-plate pipeline on
    # Kirin 8020" 宽 3.74 in > 3.5 in，横向溢出 -0.18~+3.56 in（真会被裁掉）。
    # 板子型号挪进图注，标题只留结论。
    ax.text(50, 44.5, "4-stage pipeline, one NPU stage",
            ha="center", va="center", fontsize=6.6, weight="bold")

    # 落点自证：两行，框跟着文字走（boxed_text）
    ax.text(2, 17.0, "Landing self-evidencing (logged per inference)",
            fontsize=6.0, weight="bold", va="center")
    boxed_text(ax, 2, 10.0, "req=nnrt  LANDED=NNRT:NPU_ohos.boot.kirin8020_v2_0",
               fontsize=4.4)
    boxed_text(ax, 2, 4.0, "fallback=   fingerprint  l2=4.0489", fontsize=4.4)
    save(fig, out, "fig1-pipeline-and-landing")


# ---------------------------------------------------------------- fig2

def fig2(out: str) -> None:
    """算子覆盖：NNRT vs CANN。单栏，上下两段。"""
    path = os.path.join(ROOT, "models_om_ops/op_collide.csv")
    with io.open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    cann_n = len(rows)
    cann_pass = sum(1 for r in rows if r["run_rc"] == "0" and r["build_rc"] == "0")
    nnrt_total, nnrt_ok = 36, 20

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 2.4),
                                   gridspec_kw={"height_ratios": [1, 1.3],
                                                "hspace": 0.8})
    ax1.barh(["CANN", "NNRT"],
             [cann_pass / cann_n * 100, nnrt_ok / nnrt_total * 100],
             color=[C_CANN, C_CPU], height=0.55)
    for y, (ok, tot) in enumerate([(cann_pass, cann_n), (nnrt_ok, nnrt_total)]):
        ax1.text(ok / tot * 100 + 3, y, f"{ok}/{tot}", va="center",
                 fontsize=6.2, weight="bold")
    ax1.set_xlim(0, 124)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("Admitted to NPU (%)", labelpad=1)
    ax1.set_title("Same operator family, two toolchains", pad=3)
    ax1.grid(axis="y", visible=False)

    ax2.set_axis_off()
    ax2.set_title("Rejected by NNRT, accepted by CANN", pad=3)
    rejected = ["relu", "sigmoid", "softmax", "maxpool", "pad",
                "cast_f16", "transpose", "resize", "tanh"]
    for i, nm in enumerate(rejected):
        ax2.text(0.02 + (i % 3) * 0.33, 0.78 - (i // 3) * 0.24, f"· {nm}",
                 fontsize=5.8, family="DejaVu Sans Mono",
                 transform=ax2.transAxes, va="center")
    ax2.text(0.02, 0.02, "ConvTranspose: not even convertible on NNRT.",
             fontsize=5.6, color=C_ACC, weight="bold",
             transform=ax2.transAxes, va="bottom")
    save(fig, out, "fig2-operator-coverage")


# ---------------------------------------------------------------- fig3

def fig3(out: str) -> None:
    """落点敏感性：上下两段（单栏放不下并排两个子图）。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.0),
                                   gridspec_kw={"height_ratios": [1, 1.35],
                                                "hspace": 0.9})
    labels = ["rec backend", "det backend"]
    div = [0.0, 0.1]
    bars = ax1.bar(labels, div, color=C_NPU, width=0.42)
    for b, v in zip(bars, div):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.007,
                 f"{v}% ({int(v * 10)}/1000)", ha="center", fontsize=5.8,
                 weight="bold")
    ax1.set_ylim(0, 0.19)
    ax1.set_yticks([0, 0.05, 0.10, 0.15])
    ax1.set_ylabel("Changed (%)", labelpad=1)
    ax1.set_title("Real vehicle scenes (n=1000)", pad=3)
    ax1.grid(axis="x", visible=False)

    # 标签尽量短；p 值与对照条件写进图注（正文栏宽放不下）
    # ⚠️ 左边界必须手动让位：默认 auto margin 按最宽刻度算不出这 3 个长标签，
    #    实测左溢 0.16~0.19 in。`subplots_adjust` 在 tight bbox 之前生效，
    #    能真正把轴区推右，而不是靠 tight bbox 加宽画布。
    items = [("rec NPU/CPU", +3.2, C_NPU),
             ("device/host", -0.7, C_CPU),
             ("8- vs 7-char", -6.6, C_ACC)]
    vals = [i[1] for i in items]
    ax2.barh(range(len(items)), vals, color=[i[2] for i in items], height=0.5)
    ax2.axvline(0, color="#24292f", linewidth=0.7)
    for i, v in enumerate(vals):
        ax2.text(v + (0.4 if v > 0 else -0.4), i, f"{v:+.1f} pp",
                 va="center", ha="left" if v > 0 else "right", fontsize=5.8,
                 weight="bold")
    ax2.set_yticks(range(len(items)))
    ax2.set_yticklabels([i[0] for i in items], fontsize=5.6)
    ax2.set_xlim(-10.5, 7)
    ax2.set_xticks([-8, -4, 0, 4])
    ax2.set_xlabel("Accuracy delta (pp)", labelpad=1)
    ax2.set_title("What actually moves accuracy", pad=3)
    ax2.grid(axis="y", visible=False)
    ax2.invert_yaxis()
    # 给上面那组 3 个长刻度标签留出左侧空间（实测需 ≥0.22 in）
    fig.subplots_adjust(left=0.27, right=0.98, top=0.90, bottom=0.13)
    fig._lpr_margins_set = True
    save(fig, out, "fig3-backend-sensitivity")


# ---------------------------------------------------------------- fig4

def fig4(out: str) -> None:
    """逐位替换错误：上下两段。"""
    def per_position(rel: str, prefix: str, code_key: str) -> list[int]:
        sub = [0] * 8
        for line in read_lines(rel):
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

    a = per_position("evidence/crop_bare.log", "BARE", "code_np")
    b = per_position("evidence/scene_green_rec.log", "SCENE", "code_np")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.1),
                                   gridspec_kw={"height_ratios": [1, 1],
                                                "hspace": 0.85})
    x = range(8)
    ax1.bar(x, b, color=[C_ACC] + [C_CPU] * 7, width=0.6)
    for i, v in enumerate(b):
        if v:
            ax1.text(i, v + 0.9, str(v), ha="center", fontsize=5.2, weight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(["prov", "1", "2", "3", "4", "5", "6", "7"], fontsize=5.4)
    ax1.set_ylim(0, max(b) * 1.32)
    ax1.set_xlabel("Character position", labelpad=1)
    ax1.set_ylabel("Substitution\nerrors", labelpad=1)
    tot = sum(b)
    ax1.set_title(f"8-char new-energy plates, n=971\n"
                  f"province = {b[0]}/{tot} = {100 * b[0] / tot:.1f}%", pad=3)
    ax1.grid(axis="x", visible=False)

    w = 0.38
    ax2.bar([i - w / 2 for i in x], [v / sum(a) * 100 for v in a], w,
            label="T11 cropped, bare-fed", color=C_CPU)
    ax2.bar([i + w / 2 for i in x], [v / sum(b) * 100 for v in b], w,
            label="T13 real scene, 8-char", color=C_ACC)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(["prov", "1", "2", "3", "4", "5", "6", "7"], fontsize=5.4)
    ax2.set_xlabel("Character position", labelpad=1)
    ax2.set_ylabel("Share (%)", labelpad=1)
    ax2.set_title("Province position dominates in both regimes", pad=3)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax2.grid(axis="x", visible=False)
    save(fig, out, "fig4-error-position")


# ---------------------------------------------------------------- fig5

def fig5(out: str, rel: str) -> None:
    """RQ4 持续负载。单栏，两行。"""
    with io.open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t_s"]) / 60 for r in rows]
    ms = [float(r["totalMs"]) for r in rows]
    batt = [float(r["battC"]) for r in rows]
    thermal = [int(r["thermal"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1],
                                                "hspace": 0.4})
    gaps = [(t[i], t[i + 1] - t[i]) for i in range(len(t) - 1)
            if t[i + 1] - t[i] > 1.0]
    for x, g in gaps:
        ax1.axvspan(x, x + g, color=C_BAD, alpha=0.12, zorder=0)

    ax1.plot(t, ms, "-o", color=C_NPU)
    p50 = statistics.median(ms)
    ax1.axhline(p50, color=C_ACC, linestyle="--", linewidth=0.8,
                label=f"p50 = {p50:.1f} ms")
    ax1.set_ylabel("End-to-end\nlatency (ms)", labelpad=1)
    ax1.set_ylim(0, max(ms) * 1.30)
    ax1.set_title(f"RQ4 sustained load: {len(rows)} rounds, {t[-1]:.0f} min, "
                  f"thermal level {sorted(set(thermal))[0]}", pad=3)
    ax1.legend(loc="upper left", ncols=2)
    if gaps:
        gx, gw = gaps[0]
        ax1.text(gx + gw / 2, max(ms) * 1.20, f"stall {gw * 60:.0f} s",
                 fontsize=5.0, color=C_BAD, ha="center", va="top")

    ax2.plot(t, batt, "-o", color=C_CPU)
    ax2.set_ylabel("Battery (°C)", color=C_CPU, labelpad=1)
    ax2.set_xlabel("Elapsed (min)", labelpad=1)
    ax2.set_ylim(min(batt) - 1.2, max(batt) + 1.2)
    ax2b = ax2.twinx()
    ax2b.step(t, thermal, where="post", color=C_BAD, linewidth=0.9)
    ax2b.set_ylabel("Thermal", color=C_BAD, labelpad=1)
    ax2b.set_ylim(0, 6)
    ax2b.grid(False)
    # "End-to-end latency (ms)" 折成两行后仍比默认左边距宽（实测左溢 0.06 in）
    fig.subplots_adjust(left=0.19, right=0.90, top=0.88, bottom=0.15)
    fig._lpr_margins_set = True
    save(fig, out, "fig5-rq4-sustained")


# ---------------------------------------------------------------- fig6

def fig6(out: str) -> None:
    """C8：隔离基准 ≠ 流水线内成本。单栏，上下两段。"""
    pat = re.compile(r"LANDED=CPU:t(\d) gapMs=([\d.]+) polluteKB=(\d+) "
                     r"spinMs=([\d.]+) p50=([\d.]+)")
    rows: list[tuple[int, float, int, float, float]] = []
    for line in read_lines("evidence/camera_gap_sweep.log"):
        if "y5fu_320x_head_fp32.ms" not in line:
            continue
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5))))

    def pick(threads: int, gap: float, spin: float = 0.0, pollute: int = 0) -> float:
        for th, g, p, s, p50 in rows:
            if th == threads and abs(g - gap) < 0.01 and abs(s - spin) < 0.01 \
                    and p == pollute:
                return p50
        raise KeyError((threads, gap, spin, pollute))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.0),
                                   gridspec_kw={"height_ratios": [1, 1.1],
                                                "hspace": 0.9})
    gaps = [0, 8, 33]
    ax1.plot(gaps, [pick(4, g) for g in gaps], "-o", color=C_NPU, label="4 threads")
    ax1.plot(gaps, [pick(1, g) for g in gaps], "-o", color=C_CPU, label="1 thread")
    ax1.set_xlabel("Idle gap between calls (ms)", labelpad=1)
    ax1.set_ylabel("p50 (ms)", labelpad=1)
    ax1.set_title("Sparse calls are slower", pad=3)
    ax1.set_xticks(gaps)
    ax1.legend(loc="upper left")
    ax1.axhspan(15.26, 31.56, color=C_ACC, alpha=0.12, zorder=0)
    ax1.text(33, 29.6, "in-pipeline", fontsize=5.0, color=C_ACC, ha="right")

    labels = ["gap 0", "gap 33", "spin 33", "pollute"]
    t4v = [pick(4, 0), pick(4, 33), pick(4, 0, spin=33), pick(4, 0, pollute=1200)]
    t1v = [pick(1, 0), pick(1, 33), pick(1, 0, spin=33), pick(1, 0, pollute=1200)]
    x = range(len(labels))
    w = 0.36
    ax2.bar([i - w / 2 for i in x], t4v, w, color=C_NPU, label="4 threads")
    ax2.bar([i + w / 2 for i in x], t1v, w, color=C_CPU, label="1 thread")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=5.4)
    ax2.set_ylabel("p50 (ms)", labelpad=1)
    ax2.set_title("Busy-wait fixes it only at 1 thread", pad=3)
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0, max(t4v) * 1.35)
    save(fig, out, "fig6-c8-dose-response")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rq4", default="evidence/rq4_thermal_80r.csv")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    print(f"输出目录: {args.out}   （按单栏 {COL_W} in 设计）")
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
        except Exception as exc:                             # noqa: BLE001
            print(f"  {name} 失败: {type(exc).__name__}: {exc}")
            LINT_PROBLEMS.append(f"{name}: 生成失败 {exc}")

    print()
    if LINT_PROBLEMS:
        print(f"**{len(LINT_PROBLEMS)} 个排版问题** —— 见上。不许这样交付。")
        return 1
    print("全部图生成完毕，lint 无问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
