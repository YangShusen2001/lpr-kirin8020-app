"""临时审计：从原始日志重算 conv / 各分段分布，核对 camera_summary.md 的 p50。"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "..", "evidence")

# stages 字段顺序（napi_init.cpp L806-808）
NAMES = ["tDetectMs", "tLetterboxMs", "tEncodeInferMs", "tDecodeNmsMs",
         "tRectifyMs", "tRecogMs", "tClsMs", "tPackMs", "tInferMs"]


def p50(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round((len(s) - 1) * p)))
    return s[i]


def audit(path):
    conv_hit, conv_miss = [], []
    seg_hit = {n: [] for n in NAMES}
    seg_miss = {n: [] for n in NAMES}
    n_hit = n_miss = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("STAGE "):
                continue
            m = re.search(r"conv=([\d.]+) infer=([\d.]+) count=(\d+)", line)
            if not m:
                continue
            conv = float(m.group(1))
            count = int(m.group(3))
            sm = re.search(r"stages=([\d.|-]+)", line)
            hit = count > 0
            if hit:
                n_hit += 1
                conv_hit.append(conv)
            else:
                n_miss += 1
                conv_miss.append(conv)
            if sm and sm.group(1) != "-":
                vals = sm.group(1).split("|")
                if len(vals) == len(NAMES):
                    tgt = seg_hit if hit else seg_miss
                    for n, v in zip(NAMES, vals):
                        try:
                            tgt[n].append(float(v))
                        except ValueError:
                            pass

    print(f"### {os.path.basename(path)}")
    print(f"  STAGE 行：检出 {n_hit} / 未检出 {n_miss}")
    for label, cv in (("检出", conv_hit), ("未检出", conv_miss)):
        if cv:
            print(f"  conv {label}: n={len(cv)} p50={p50(cv):.2f} "
                  f"p10={pct(cv,0.10):.2f} p90={pct(cv,0.90):.2f} "
                  f"min={min(cv):.2f} max={max(cv):.2f}")
    print("  分段 p50（检出桶 / 未检出桶）:")
    for n in NAMES:
        a, b = seg_hit[n], seg_miss[n]
        sa = f"{p50(a):.3f}" if a else "  -  "
        sb = f"{p50(b):.3f}" if b else "  -  "
        print(f"    {n:<15} {sa:>9}  {sb:>9}")
    if seg_hit["tDetectMs"]:
        comp = (p50(seg_hit["tLetterboxMs"]) + p50(seg_hit["tEncodeInferMs"]) +
                p50(seg_hit["tDecodeNmsMs"]))
        print(f"  det 闭合核对（检出 p50）：tDetect={p50(seg_hit['tDetectMs']):.3f} "
              f"vs lb+encinf+nms={comp:.3f}  缺口={p50(seg_hit['tDetectMs'])-comp:.3f}")
    print()


for f in ("camera_sweep1.log", "camera_sweep2.log"):
    p = os.path.join(EV, f)
    if os.path.exists(p):
        audit(p)
    else:
        print(f"[skip] {f} 不存在")
