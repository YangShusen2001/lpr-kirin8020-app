"""落点自证与分段计时的自洽性断言（宿主机可跑，无需设备）。

对应用户票据 T2 未达成项「无头测试覆盖」：当时的证据全部来自真机 hilog，
没有可重跑的自动化断言。这里把 T2 与 T7 里**只靠自洽性才能发现**的那几条
不变量落成断言，跑在已提交的证据文件上。

为什么值得单独测：T2 修掉的两个口径缺陷（`tDetectMs` 与 `tLetterboxMs` 是同一个
表达式、`tRectifyMs` 横跨整个检测段）都不是崩溃或错值，而是**字段名承诺的语义
与实际不符**。它们只会在「分段之和 vs 端到端」这类自洽性校验里露出来。
把那几条校验固化成断言，才能防止再次悄悄退化。

用法：
    python tools/test_landing_invariants.py
退出码 0 = 全部通过。
"""
import csv
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RQ4_CSV = os.path.join(ROOT, "evidence", "rq4_thermal.csv")

# p0 第 10 字段的分段顺序，与 napi_init.cpp:776 严格对应
STAGES = ["detect", "letterbox", "encInfer", "decNms", "rectify",
          "recog", "cls", "pack", "infer"]
# 顶层分解：detect 是【整段】，已含 letterbox/pack/infer/decNms，不可重复相加
TOP = ["detect", "rectify", "recog", "cls"]

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)
    return cond


def load_rows():
    if not os.path.exists(RQ4_CSV):
        print(f"!! 找不到证据文件: {RQ4_CSV}")
        return []
    # 上游是 utf-8（无 BOM），但为稳妥两种都试
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            with open(RQ4_CSV, encoding=enc, newline="") as fh:
                return list(csv.DictReader(fh))
        except (UnicodeDecodeError, UnicodeError):
            continue
    print("!! 证据文件编码无法识别")
    return []


def num(row, key):
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    rows = load_rows()
    print(f"证据: evidence/rq4_thermal.csv   行数={len(rows)}")
    if not rows:
        return 1

    good = [r for r in rows if num(r, "totalMs") is not None and num(r, "detect_ms") is not None]
    print(f"可用于分段断言的轮次: {len(good)}")
    if not good:
        # 证据文件可能是旧版（只有 totalMs/指纹，没有分段列）。那时本脚本的分段
        # 断言无从谈起 —— 要明确说清楚，而不是抛异常。
        print()
        print("!! 该证据文件没有分段列（detect_ms 等），无法做分段断言。")
        print("   这是旧版 CSV 的形态；请用 tools/parse_rq4.py 重新生成。")
        return 1
    print()

    # ---------------------------------------------------------------- 1. 语义不变量
    print("=== 1. 分段语义（T2 修掉的两个口径缺陷）===")

    # tDetectMs 必须【大于或等于】tLetterboxMs —— 检测是整段，letterbox 只是其中一步。
    # 缺陷形态：两者是同一个表达式 => 恒等。恒等即回归。
    eq = [r["round"] for r in good
          if num(r, "detect_ms") == num(r, "letterbox_ms")]
    check("tDetectMs 不等于 tLetterboxMs（检测是整段，不是 letterbox 的别名）",
          not eq, f"恒等的轮次={eq}" if eq else "")

    # tDetectMs 必须涵盖 pack+infer+decodeNms+letterbox（容差给浮点）
    bad_cov = []
    for r in good:
        lb = num(r, "letterbox_ms") or 0
        pk = num(r, "pack_ms") or 0
        inf = num(r, "infer_ms") or 0
        nms = num(r, "decNms_ms") or 0
        det = num(r, "detect_ms") or 0
        if det + 1e-6 < lb + pk + inf + nms:
            bad_cov.append(r["round"])
    check("tDetectMs >= letterbox+pack+infer+decNms（涵盖检测段全部子步骤）",
          not bad_cov, f"不满足的轮次={bad_cov}" if bad_cov else "")

    # tRectifyMs 不得把检测段算进去 —— 缺陷形态：它横跨整个检测段，于是
    # rectify >= detect 会成立。
    bad_rect = [r["round"] for r in good
                if (num(r, "rectify_ms") or 0) >= (num(r, "detect_ms") or 0)]
    check("tRectifyMs < tDetectMs（矫正段不含检测段）",
          not bad_rect, f"可疑轮次={bad_rect}" if bad_rect else "")

    # ---------------------------------------------------------------- 2. 闭合性
    print()
    print("=== 2. 完整性：顶层四段之和 vs 端到端 ===")
    rels = []
    for r in good:
        top = sum(num(r, f"{s}_ms") or 0 for s in TOP)
        tot = num(r, "totalMs")
        if tot:
            rels.append((tot - top) / tot)
    if not rels:
        check("至少有一轮可算闭合残差", False, "rels 为空")
        mean_rel = worst = float("nan")
    else:
        mean_rel = statistics.mean(rels) * 100
        worst = max(abs(x) for x in rels) * 100
        # 残差应为正的小量（流水线外的计时收尾）。负值意味着重复计入某个分段。
        check("分层之和不超过端到端（无重复计入）",
              all(x >= -0.005 for x in rels),
              f"最小残差={min(rels)*100:.2f}%")
        check("残差 < 3%（分列未漏掉大块时间）",
              worst < 3.0, f"mean={mean_rel:.2f}%  worst={worst:.2f}%")

    # ---------------------------------------------------------------- 3. 自报字段自洽
    print()
    print("=== 3. 落点与指纹自洽 ===")
    # cpu_side_ms / npu_side_ms 必须与分段一致（它们是导出列，可能漂移）
    bad_side = []
    for r in good:
        c = num(r, "cpu_side_ms")
        n = num(r, "npu_side_ms")
        if c is None or n is None:
            continue
        exp_c = sum(num(r, f"{s}_ms") or 0 for s in ("detect", "rectify", "cls"))
        exp_n = num(r, "recog_ms") or 0
        if abs(c - exp_c) > 1e-3 or abs(n - exp_n) > 1e-3:
            bad_side.append(r["round"])
    check("cpu_side_ms / npu_side_ms 与分段一致（导出列未漂移）",
          not bad_side, f"不一致轮次={bad_side}" if bad_side else "")

    # 指纹在本轮内应恒定 —— T7 的结论②依赖这一点
    det_l2 = {r.get("det_l2") for r in rows if r.get("det_l2")}
    rec_l2 = {r.get("rec_l2") for r in rows if r.get("rec_l2")}
    check("det.l2 在本轮内恒定（T7 结论②）", len(det_l2) <= 1, f"取到 {det_l2}")
    check("rec.l2 在本轮内恒定（T7 结论②）", len(rec_l2) <= 1, f"取到 {rec_l2}")

    # 落点必须与生产档一致：det=CPU / rec=NNRT...kirin8020 / cls=CPU
    bad_land = []
    for r in rows:
        d, c = r.get("det_landed", ""), r.get("rec_landed", "")
        if d and d != "CPU":
            bad_land.append((r["round"], "det", d))
        if c and not c.startswith("NNRT:NPU_ohos"):
            bad_land.append((r["round"], "rec", c))
    check("落点符合生产档 det=CPU / rec=NNRT:NPU(kirin8020)",
          not bad_land, f"异常={bad_land[:3]}" if bad_land else "")

    # ---------------------------------------------------------------- 4. 热态如实
    print()
    print("=== 4. 热态与破例轮如实记录 ===")
    thermals = sorted({r.get("thermal") for r in rows if r.get("thermal")})
    check("thermal 被逐轮记录（非空）", bool(thermals), f"取到 {thermals}")
    # 破例轮必须保留空 totalMs，不得被剔除 —— 票面明确要求如实保留
    frac = sum(1 for r in rows if not (r.get("totalMs") or "").strip())
    print(f"  [info] 破例轮 {frac}/{len(rows)}（空 totalMs 的行被保留，未剔除）")

    print()
    if FAILURES:
        print(f"=> **{len(FAILURES)} 项失败**: {FAILURES}")
        return 1
    print("=> 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
