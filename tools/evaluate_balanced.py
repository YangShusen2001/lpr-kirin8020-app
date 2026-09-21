"""T6: 在省份均衡集上评识别精度，与既有 1000 张集对照。

裁决的问题：**既有声明的 90.6% 是否被省份偏斜抬高？**

方法（严格遵循 ADR-015 的教训）：
  - 真值 = 文件名（人工标注），**绝不用被测系统自己的输出**
  - 同一份参考实现（hlpr_reference，即 native C++ 的 Python 对照）评两个集
  - 逐省报告，因为偏斜的后果集中体现在省份位

输出：balanced_accuracy.json
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PRIOR_SHOWCASE, SCRATCH  # noqa: E402

BASE = PRIOR_SHOWCASE
sys.path.insert(0, os.path.join(BASE, "tools"))
sys.stdout.reconfigure(encoding="utf-8")

import hlpr_reference as H

rec = H.sess("rpv3_mdict_160_r3.onnx")

SETS = {
    "legacy_1000_anhui_skewed": os.path.join(BASE, "_dataset", "real", "crops"),
    "balanced_326_26_provinces": os.path.join(SCRATCH, "balanced", "crops"),
}
OUT = os.path.join(SCRATCH, "balanced", "balanced_accuracy.json")


def evaluate(crops_dir):
    files = sorted(Path(crops_dir).glob("*.jpg"))
    total = 0
    correct = 0
    by_prov_total = Counter()
    by_prov_correct = Counter()
    len_total = Counter()
    len_correct = Counter()
    # 错误发生在哪一位
    pos_errors = Counter()
    first_char_errors = 0
    failures = []

    for f in files:
        truth = f.stem
        # 卫兵：车牌真值只能是 7 或 8 字符。出现别的长度说明文件名被判读错
        # （首轮就踩过：抽取时给重名文件加了 `_8171` 后缀，评测端把它并进真值，
        #  凭空造出 12 字符"真值"，把 29 条本应正确的样本记成失败）。
        if len(truth) not in (7, 8):
            raise ValueError(
                f"真值长度非法：{f.name!r} -> {truth!r}（{len(truth)} 字符）。"
                f"文件名必须就是车牌串本身。")
        img = H.imread_u(f)
        try:
            code, conf = H.recognize(rec, img)
        except Exception as e:
            failures.append({"file": f.name, "reason": str(e)})
            continue
        total += 1
        prov = truth[0] if truth else "?"
        by_prov_total[prov] += 1
        len_total[len(truth)] += 1
        if code == truth:
            correct += 1
            by_prov_correct[prov] += 1
            len_correct[len(truth)] += 1
        else:
            failures.append({"file": f.name, "truth": truth, "pred": code,
                             "conf": round(conf, 4)})
            if code and truth and code[0] != truth[0]:
                first_char_errors += 1
            # 逐位比对（取到较短的长度）
            for i in range(min(len(code), len(truth))):
                if code[i] != truth[i]:
                    pos_errors[i] += 1

    return {
        "dir": crops_dir,
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "first_char_errors": first_char_errors,
        "position_error_histogram": dict(sorted(pos_errors.items())),
        "province_total": dict(by_prov_total),
        "province_correct": dict(by_prov_correct),
        "province_accuracy": {
            p: round(by_prov_correct[p] / by_prov_total[p], 3)
            for p in by_prov_total
        },
        "length_total": dict(len_total),
        "length_accuracy": {
            L: round(len_correct[L] / len_total[L], 3) for L in len_total
        },
        "n_failures_logged": len(failures),
        "failures_sample": failures[:40],
    }


results = {}
for name, d in SETS.items():
    if not os.path.isdir(d):
        print(f"skip {name}: {d} not found")
        continue
    print(f"=== {name} ===")
    r = evaluate(d)
    results[name] = r
    print(f"  n={r['n']}  correct={r['correct']}  accuracy={r['accuracy']:.4f}")
    print(f"  first-char errors = {r['first_char_errors']}")
    print(f"  provinces = {len(r['province_total'])}")
    print(f"  length accuracy = {r['length_accuracy']}")
    print()

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

print("=== 对照 ===")
for name, r in results.items():
    print(f"  {name:<30} acc={r['accuracy']:.4f}  n={r['n']:<5} "
          f"provinces={len(r['province_total'])}  first-char-err={r['first_char_errors']}")
print(f"\n-> {OUT}")
