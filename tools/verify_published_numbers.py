#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发布数字守卫：论文 / 简历 / 网页里出现的每个数字，都从原始证据**重算**一遍。

## 为什么需要它

项目要同时往三个出口发布数字：论文、简历、个人网页。三者一旦不一致，
或者证据文件被重新生成后数字漂移，**没有任何机制会报警**。

更具体的教训（本仓库已经发生过）：
- `docs/notes/t10` 引用的 `evidence/crop_ab_rec.log` **并不存在**，实际文件是
  `evidence/crop_ab.log`；
- `docs/notes/t8` 引用的 `op_collide.json` **并不存在**，且 `op_collide.csv`
  在 `models_om_ops/` 而不是 `evidence/`；
- `docs/notes/t7` 的展示表一度漏掉了 max 那一轮（数字对、表错）。

所以这个脚本做两件事：
1. **路径守卫**：声明的证据文件必须存在，否则直接失败；
2. **数值守卫**：声明的数字必须能从证据重算出来（带容差），否则失败。

## 用法

    python tools/verify_published_numbers.py                    # 检查全部
    python tools/verify_published_numbers.py --rq4 evidence/rq4_thermal.csv
    python tools/verify_published_numbers.py --list

退出码 0 = 全部通过；1 = 有 FAIL（**此时不得发布任何数字**）。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- 证据解析

KV = re.compile(r"(\w+)=([^\s]*)")


def read_lines(rel: str) -> list[str]:
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def kv_of(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV.finditer(line)}


def parse_ab(rel: str, prefix: str) -> list[dict[str, str]]:
    """解析 `PREFIX file=… gt=… code_cpu=… code_np=… agree=… ok_cpu=… ok_np=…` 形态。"""
    out: list[dict[str, str]] = []
    for line in read_lines(rel):
        if not line.startswith(prefix + " ") or "file=" not in line:
            continue
        out.append(kv_of(line))
    return out


def len_err(rows: list[dict[str, str]], key: str) -> int:
    """长度错误 = 输出字符数与真值不等（比"是否 7/8 位"更严格，不预设牌长）。"""
    return sum(1 for r in rows if len(r.get(key, "")) != len(r.get("gt", "")))


# ---------------------------------------------------------------- 守卫

class Guard:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def check(self, name: str, computed: float, expected: float, tol: float,
              src: str, unit: str = "") -> None:
        ok = abs(computed - expected) <= tol
        fmt = (lambda v: f"{v:.2f}") if isinstance(computed, float) else (lambda v: str(v))
        self.rows.append((name, f"{fmt(computed)}{unit}", f"{fmt(expected)}{unit}", ok))

    def exists(self, name: str, rel: str) -> None:
        path = os.path.join(ROOT, rel)
        ok = os.path.exists(path)
        size = f"{os.path.getsize(path) / 1024:.0f} KB" if ok else "缺失"
        self.rows.append((name, size, "存在", ok))

    def report(self) -> int:
        bad = [r for r in self.rows if not r[3]]
        w1 = max(len(r[0]) for r in self.rows)
        w2 = max(len(r[1]) for r in self.rows)
        print(f"\n{'检查项':<{w1}}  {'重算值':>{w2}}  {'声明值':>{w2}}  结果")
        print("-" * (w1 + w2 * 2 + 12))
        for name, got, want, ok in self.rows:
            print(f"{name:<{w1}}  {got:>{w2}}  {want:>{w2}}  {'OK' if ok else '**FAIL**'}")
        print("-" * (w1 + w2 * 2 + 12))
        if bad:
            print(f"\n{len(bad)} 项不符 —— **不得发布任何数字**，先查证据或改声明。")
            return 1
        print(f"\n全部 {len(self.rows)} 项通过。数字可发布。")
        return 0


# ---------------------------------------------------------------- 各轮结论

def t11_bare_head(g: Guard) -> None:
    g.exists("T11 证据存在", "evidence/crop_bare.log")
    rows = parse_ab("evidence/crop_bare.log", "BARE")
    n = len(rows)
    g.check("T11 n", n, 1000, 0, "evidence/crop_bare.log")
    ok_np = sum(1 for r in rows if r["ok_np"] == "1")
    ok_cpu = sum(1 for r in rows if r["ok_cpu"] == "1")
    g.check("T11 端侧 NPU 命中", ok_np, 899, 0, "evidence/crop_bare.log")
    g.check("T11 端侧 CPU 命中", ok_cpu, 867, 0, "evidence/crop_bare.log")
    g.check("T11 NPU 准确率", 100 * ok_np / n, 89.9, 0.05, "evidence/crop_bare.log", "%")
    g.check("T11 CPU 准确率", 100 * ok_cpu / n, 86.7, 0.05, "evidence/crop_bare.log", "%")
    g.check("T11 两端不一致", sum(1 for r in rows if r["agree"] == "0"), 49, 0,
            "evidence/crop_bare.log")
    only_np = sum(1 for r in rows if r["ok_np"] == "1" and r["ok_cpu"] == "0")
    only_cpu = sum(1 for r in rows if r["ok_cpu"] == "1" and r["ok_np"] == "0")
    g.check("T11 仅 NPU 对", only_np, 38, 0, "evidence/crop_bare.log")
    g.check("T11 仅 CPU 对", only_cpu, 6, 0, "evidence/crop_bare.log")
    g.check("T11 NPU 长度错误", len_err(rows, "code_np"), 17, 0, "evidence/crop_bare.log")
    g.check("T11 CPU 长度错误", len_err(rows, "code_cpu"), 50, 0, "evidence/crop_bare.log")
    # 识别耗时。
    #
    # ⚠️ 口径必须写死，否则数字会漂：
    #   - `ms_cpu` 只在 CPU 端**出结果**时才有值 → 961/1000 条非空
    #     （39 条 CPU 完全没出串，对应 `ok_cpu=0` 且 `code_cpu=` 为空）
    #   - `ms_np` 同理 → 993/1000 条非空
    #   所以「mean」是对**非空子集**取的均值，不是对 1000 条取的。
    #
    # 2026-09-21 更正：`t11-bare-head-accuracy.md` 原先报 CPU mean **16.12 ms**、
    # 加速比 **4.32×**，**任何口径都重算不出来**（全量 16.78 / 去首行 16.78 /
    # 仅 ok=1 16.81 / 配对 16.73 / 中位数 16.03 / 各种截尾 16.58–16.71）。
    # 已按本脚本重算值更正笔记。这里是**重算值**，不是抄来的值。
    ms_np = [float(r["ms_np"]) for r in rows if r.get("ms_np")]
    ms_cpu = [float(r["ms_cpu"]) for r in rows if r.get("ms_cpu")]
    g.check("T11 NPU 耗时样本数", len(ms_np), 993, 0, "evidence/crop_bare.log")
    g.check("T11 CPU 耗时样本数", len(ms_cpu), 961, 0, "evidence/crop_bare.log")
    g.check("T11 NPU 识别耗时 mean", statistics.mean(ms_np), 3.76, 0.02,
            "evidence/crop_bare.log", " ms")
    g.check("T11 CPU 识别耗时 mean", statistics.mean(ms_cpu), 16.78, 0.02,
            "evidence/crop_bare.log", " ms")
    g.check("T11 加速比（mean 口径）", statistics.mean(ms_cpu) / statistics.mean(ms_np),
            4.46, 0.02, "evidence/crop_bare.log", "×")
    g.check("T11 加速比（median 口径）", statistics.median(ms_cpu) / statistics.median(ms_np),
            4.42, 0.02, "evidence/crop_bare.log", "×")


def t12_scene(g: Guard) -> None:
    """真实整车场景（CCPD1000）det / rec 落点 A/B。"""
    for name, rel, want_agree in (
        ("T12 det", "evidence/ccpd_scene.log", 999),
        ("T12 rec", "evidence/ccpd_scene_rec.log", 1000),
    ):
        g.exists(f"{name} 证据存在", rel)
        rows = parse_ab(rel, "SCENE")
        n = len(rows)
        g.check(f"{name} n", n, 1000, 0, rel)
        agree = sum(1 for r in rows if r["agree"] == "1")
        g.check(f"{name} 逐字一致", agree, want_agree, 0, rel)
        g.check(f"{name} 分歧率", 100 * (n - agree) / n,
                round(100 * (n - want_agree) / n, 2), 0.02, rel, "%")
        g.check(f"{name} ok_cpu", sum(1 for r in rows if r["ok_cpu"] == "1"), 991, 0, rel)
        g.check(f"{name} ok_np", sum(1 for r in rows if r["ok_np"] == "1"), 991, 0, rel)
        g.check(f"{name} 未检出", sum(1 for r in rows if r["none_cpu"] == "1"), 0, 0, rel)


def t13_green(g: Guard) -> None:
    rel = "evidence/scene_green_rec.log"
    g.exists("T13 证据存在", rel)
    rows = parse_ab(rel, "SCENE")
    n = len(rows)
    g.check("T13 n", n, 1000, 0, rel)
    g.check("T13 逐字一致", sum(1 for r in rows if r["agree"] == "1"), 999, 0, rel)
    g.check("T13 ok_np", sum(1 for r in rows if r["ok_np"] == "1"), 925, 0, rel)
    g.check("T13 准确率", 100 * sum(1 for r in rows if r["ok_np"] == "1") / n,
            92.5, 0.05, rel, "%")
    g.check("T13 长度错误", len_err(rows, "code_np"), 29, 0, rel)
    g.check("T13 未检出", sum(1 for r in rows if r["none_np"] == "1"), 1, 0, rel)
    # 省份位占替换错误的比：等长行里逐位比较
    sub = [0] * 8
    total = 0
    for r in rows:
        gt, code = r.get("gt", ""), r.get("code_np", "")
        if len(gt) != len(code):
            continue
        for i, (a, b) in enumerate(zip(gt, code)):
            if a != b:
                sub[i] += 1
                total += 1
    # ⚠️ 2026-09-21 更正：`t13-green-plate-8char.md` 的表格列的是
    # [33, 2, 7, 11, 5, 1, 4, 4]，**它自己合计就是 67**；但同一节的正文写
    # 「占全部替换错误（66）的 50%」—— 分子分母都对不上自己的表。
    # 重算确认表是对的、正文是错的：33/67 = 49.25%。
    g.check("T13 等长行 n", sum(1 for r in rows if len(r.get("gt", "")) == len(r.get("code_np", ""))),
            971, 0, rel)
    g.check("T13 省份位替换错误数", sub[0], 33, 0, rel)
    g.check("T13 替换错误总数", total, 67, 0, rel)
    g.check("T13 省份位占比", 100 * sub[0] / total, 49.25, 0.05, rel, "%")


def t8_operators(g: Guard) -> None:
    rel = "models_om_ops/op_collide.csv"
    g.exists("T8 证据存在", rel)
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    g.check("T8 算子探针数", n, 51, 0, rel)
    g.check("T8 compat=0", sum(1 for r in rows if r["compat"] == "0"), 51, 0, rel)
    g.check("T8 construct=ok", sum(1 for r in rows if r["construct"] == "ok"), 51, 0, rel)
    g.check("T8 build_rc=0", sum(1 for r in rows if r["build_rc"] == "0"), 51, 0, rel)
    g.check("T8 run_rc=0", sum(1 for r in rows if r["run_rc"] == "0"), 51, 0, rel)
    # 陪跑证据：NNRT 侧被拒、CANN 侧通过的那批必须真在这 51 个里
    rejected = {"relu", "sigmoid", "softmax", "maxpool", "pad",
                "cast_f16", "transpose", "resize", "tanh"}
    names = {r["op"] for r in rows}
    g.check("T8 含 NNRT 侧被拒的 9 个算子", len(rejected & names), 9, 0, rel)
    g.check("T8 含 convtranspose", 1 if "convtranspose" in names else 0, 1, 0, rel)


# ⚠ 命名歧义提醒：`t8_operators` 是**算子覆盖对撞**（.om vs .ms，51 个探针）；
# 下面的 `t8_vehicle_npu` 是**车辆流水线二期**（换原版 YOLOv5 + 上 NPU）。
# 两者都叫 T8 但是两件不同的事，别合并、也别互相引用数字。

DRIFT = re.compile(
    r"DET DRIFT mk=(?P<mk>\S+) (?P<be>\S+) landed=(?P<landed>\S*)"
    r" req=(?P<req>\S*) fallback=(?P<fb>\S*)"
    r" L2=(?P<l2>\S+) maxAbs=(?P<maxabs>\S+)"
    r" p50=(?P<p50>\S+) mean=(?P<mean>\S+) dtype=(?P<dtype>\S+) elems=(?P<elems>\S+)")


def t8_vehicle_npu(g: Guard) -> None:
    """车辆流水线 T8：换原版 anchor-based YOLOv5 后的落点与延迟。

    判据只认 `landed=` 这个**实际落点**字段 —— 请求后端与落点是两件事。
    落点自证两条：(a) landed 逐字回 NPU 设备名且 fallback 为空；
    (b) 同模型 nnrt 与 cpu 的 checksum **不同**（若静默回落 CPU 必须逐位相同）。
    """
    rel = "_veh/devlog_T8V7.txt"
    g.exists("T8v 真机证据存在", rel)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in read_lines(rel):
        m = DRIFT.search(line)
        if m:
            rows[(m.group("mk"), m.group("be"))] = m.groupdict()
    g.check("T8v DET DRIFT 条目数", len(rows), 6, 0, rel)

    def one(mk: str, be: str, key: str) -> float:
        return float(rows[(mk, be)][key])

    fp32 = "yolov5s_v7_320_npu_fp32.ms"
    fp16 = "yolov5s_v7_320_npu_fp16.ms"

    # (a) 落点自证之一：landed 是 NPU 设备名，且没有 fallback
    for mk in (fp32, fp16):
        r = rows[(mk, "nnrt")]
        g.check(f"T8v {mk} 落点含 NPU 设备名",
                1 if r["landed"].startswith("NNRT:NPU_ohos") else 0, 1, 0, rel)
        g.check(f"T8v {mk} nnrt 无 fallback", 1 if r["fb"] == "" else 0, 1, 0, rel)
        g.check(f"T8v {mk} req=nnrt", 1 if r["req"] == "nnrt" else 0, 1, 0, rel)

    # (b) 落点自证之二：nnrt 与 cpu 的 checksum 必须不同
    for mk in (fp32, fp16):
        a, b = rows[(mk, "nnrt")]["l2"], rows[(mk, "cpu")]["l2"]
        g.check(f"T8v {mk} nnrt/cpu checksum 不同",
                1 if a != b else 0, 1, 0, rel)

    # 延迟：同引擎同模型，唯一变量是后端
    g.check("T8v fp32 NPU p50", one(fp32, "nnrt", "p50"), 5.392, 0.01, rel, " ms")
    g.check("T8v fp32 CPU p50", one(fp32, "cpu", "p50"), 42.016, 0.01, rel, " ms")
    g.check("T8v fp32 加速比",
            one(fp32, "cpu", "p50") / one(fp32, "nnrt", "p50"), 7.79, 0.01, rel, "×")
    g.check("T8v fp16 NPU p50", one(fp16, "nnrt", "p50"), 5.537, 0.01, rel, " ms")
    g.check("T8v fp16 CPU p50", one(fp16, "cpu", "p50"), 41.646, 0.01, rel, " ms")
    g.check("T8v fp16 加速比",
            one(fp16, "cpu", "p50") / one(fp16, "nnrt", "p50"), 7.52, 0.01, rel, "×")
    # 首输出 (1,255,40,40) 的元素数，用来确认跑的是三个裸头而不是被裁坏的图
    g.check("T8v 首输出元素数", one(fp32, "nnrt", "elems"), 408000, 0, rel)

    # converter 对比：v5-u 失败、裁切后成功 —— 这是「模型×工具链」判否的现场
    fail_rel = "_veh/yolov5su_320_fp32.convert.log"
    g.exists("T8v v5-u 转换日志存在", fail_rel)
    fail_txt = "\n".join(read_lines(fail_rel))
    g.check("T8v v5-u 死于 dfl/conv 形状推断",
            1 if "InferShapeByNNACL for op: /model.24/dfl/conv/Conv failed" in fail_txt
            else 0, 1, 0, fail_rel)
    g.check("T8v v5-u 转换失败",
            1 if "Convert model failed" in fail_txt else 0, 1, 0, fail_rel)
    for rel2 in ("_veh/yolov5s_v7_320_npu_fp32.convert.log",
                 "_veh/yolov5s_v7_320_npu_fp16.convert.log"):
        g.exists("T8v 转换日志存在 " + os.path.basename(rel2), rel2)
        txt = "\n".join(read_lines(rel2))
        g.check("T8v 转换成功 " + ("fp32" if "fp32" in rel2 else "fp16"),
                1 if "CONVERT RESULT SUCCESS:0" in txt else 0, 1, 0, rel2)


def t7_rq4(g: Guard, rel: str) -> None:
    g.exists("T7 证据存在", rel)
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    ms = sorted(float(r["totalMs"]) for r in rows)
    p50 = statistics.median(ms)
    mean = statistics.mean(ms)
    cv = 100 * statistics.stdev(ms) / mean
    print(f"\n[T7/RQ4 数据源: {rel}  n={n}]")
    print(f"  thermal 取值 = {sorted({r['thermal'] for r in rows})}")
    print(f"  battC 取值   = {sorted({r['battC'] for r in rows})}")
    print(f"  charging 取值= {sorted({r.get('charging','') for r in rows})}")
    print(f"  load 取值    = {sorted({r['load'] for r in rows})}")
    print(f"  p50={p50:.1f} ms  mean={mean:.1f} ms  max={max(ms):.1f} ms  CV={cv:.1f}%")
    print(f"  分段: " + ", ".join(
        f"{k}={statistics.mean(float(r[k]) for r in rows):.2f}"
        for k in ("letterbox_ms", "rectify_ms", "recog_ms", "infer_ms")
        if k in rows[0]))
    # 三件套稳定性：指纹与识别串必须逐轮恒定，否则说明有未初始化状态
    for col, label in (("det_l2", "det.l2"), ("rec_l2", "rec.l2"), ("code", "code")):
        if col in rows[0]:
            uniq = sorted({r[col] for r in rows})
            g.check(f"T7 {label} 逐轮恒定", len(uniq), 1, 0, rel)


def rq4_drift(g: Guard, rel: str) -> None:
    """80 轮补跑的**持续负载退化**结论。

    这是 RQ4 唯一的正面结论，也是最容易被误引的一条：
    它必须**排除**那次系统冻结（r=54→55 间隔 213 s），否则退化量会被污染。
    """
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t_s"]) for r in rows]
    # 找出异常停顿（>60 s 的轮间隔）
    freeze_at = None
    for i in range(len(t) - 1):
        if t[i + 1] - t[i] > 60:
            freeze_at = i + 1
            break
    g.check("RQ4 存在一次 >60 s 的停顿（须在论文里标出）",
            1 if freeze_at else 0, 1, 0, rel)
    g.check("RQ4 thermal 全程恒定", len({r["thermal"] for r in rows}), 1, 0, rel)
    # 退化比较用**固定的轮次区间**，而不是"冻结之前的全部"——
    # 后者的样本量会随冻结位置浮动，数字就不稳定了。
    g.check("RQ4 比较窗口未被冻结污染", 1 if (freeze_at is None or freeze_at > 53) else 0,
            1, 0, rel)

    def seg(sub, keys):
        return statistics.mean(sum(float(r[k]) for k in keys) for r in sub)

    cpu = ["detect_ms", "rectify_ms", "cls_ms"]
    npu = ["recog_ms"]
    a, b = rows[0:18], rows[36:54]      # r=0–17 vs r=36–53
    g.check("RQ4 端到端退化", 100 * (seg(b, ["totalMs"]) - seg(a, ["totalMs"]))
            / seg(a, ["totalMs"]), 26.9, 0.2, rel, "%")
    g.check("RQ4 CPU 侧退化", 100 * (seg(b, cpu) - seg(a, cpu)) / seg(a, cpu),
            29.3, 0.2, rel, "%")
    g.check("RQ4 NPU 侧退化", 100 * (seg(b, npu) - seg(a, npu)) / seg(a, npu),
            19.2, 0.2, rel, "%")


def c8_dose_response(g: Guard) -> None:
    """C8：隔离基准 ≠ 流水线内成本（论文 Fig. 6 的四个数字）。

    论文的 Fig. 6 图注里发布了 7.26 / 31.56 / +20.8 / +0.29 四个数，
    所以它们必须可机检 —— 这是本项目的纪律：**发布了的数字就要能被重算**。
    """
    rel = "evidence/camera_gap_sweep.log"
    g.exists("C8 证据存在", rel)
    pat = re.compile(r"LANDED=CPU:t(\d) gapMs=([\d.]+) polluteKB=(\d+) spinMs=([\d.]+) p50=([\d.]+)")
    rows: list[tuple[int, float, int, float, float]] = []
    for line in read_lines(rel):
        if "y5fu_320x_head_fp32.ms" not in line:
            continue
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)),
                         float(m.group(4)), float(m.group(5))))

    def pick(threads: int, gap: float, spin: float = 0.0, pollute: int = 0) -> float:
        for th, gp, po, sp, p50 in rows:
            if th == threads and abs(gp - gap) < 0.01 and abs(sp - spin) < 0.01 and po == pollute:
                return p50
        raise KeyError((threads, gap, spin, pollute))

    g.check("C8 紧循环基线 (t4)", pick(4, 0), 7.26, 0.02, rel, " ms")
    g.check("C8 gap=8 (t4)", pick(4, 8), 16.14, 0.02, rel, " ms")
    g.check("C8 gap=33 (t4)", pick(4, 33), 31.56, 0.02, rel, " ms")
    # 忙等对照：t4 上几乎没救回来，t1 上几乎完全消除 —— 这是分离两个机理的关键
    g.check("C8 忙等残差 (t4)", pick(4, 0, spin=33) - pick(4, 0), 20.80, 0.02, rel, " ms")
    g.check("C8 忙等残差 (t1)", pick(1, 0, spin=33) - pick(1, 0), 0.29, 0.02, rel, " ms")
    g.check("C8 缓存污染无影响 (t4)", pick(4, 0, pollute=1200), 7.50, 0.02, rel, " ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq4", default="evidence/rq4_thermal_80r.csv",
                    help="RQ4 数据文件（相对仓库根）")
    ap.add_argument("--list", action="store_true", help="只列证据文件是否存在")
    args = ap.parse_args()

    g = Guard()
    if args.list:
        for rel in ("evidence/crop_bare.log", "evidence/ccpd_scene.log",
                    "evidence/ccpd_scene_rec.log", "evidence/scene_green_rec.log",
                    "models_om_ops/op_collide.csv", args.rq4,
                    "evidence/crop_ab.log", "evidence/crop_ab_det.log",
                    "evidence/camera_windows.csv", "evidence/camera_gap_sweep.log",
                    "_veh/devlog_T8V7.txt",
                    "_veh/yolov5su_320_fp32.convert.log",
                    "_veh/yolov5s_v7_320_npu_fp32.convert.log",
                    "_veh/yolov5s_v7_320_npu_fp16.convert.log"):
            p = os.path.join(ROOT, rel)
            ok = os.path.exists(p)
            print(f"{'OK  ' if ok else 'MISS'} {rel}"
                  + (f"  ({os.path.getsize(p) / 1024:.0f} KB)" if ok else ""))
        return 0

    t11_bare_head(g)
    t12_scene(g)
    t13_green(g)
    t8_operators(g)
    t8_vehicle_npu(g)
    t7_rq4(g, args.rq4)
    rq4_drift(g, args.rq4)
    c8_dose_response(g)
    return g.report()


if __name__ == "__main__":
    sys.exit(main())
