"""T6: 从 val.mdb 稀疏抽取省份均衡的评测样本。

背景：既有 1000 张真实集 **956/1000 是皖**，84 个同长度替换错误里 78 个发生在省份位。
所以「省份泛化能力」从未被测过 —— 这个脚本补的就是这个缺口。

约束：
  - 只下载 val/data.mdb（17.8 MiB），不碰 681.8 MiB 的整包
  - 按省份抽 N 张，输出到 crops/ 并以**文件名即真值**的形式命名（与既有 1000 张集同约定）
  - 不用被测系统自己的输出当真值（ADR-015 的教训）

输出：
  crops/<plate>.jpg          抽出的样本，文件名 = 真值
  manifest.json              每张的来源、省份、原 LMDB 索引
  province_report.json       省份分布与抽样统计
"""
import json
import os
import sys
from collections import Counter, defaultdict

import lmdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCRATCH  # noqa: E402

MDb = os.path.join(SCRATCH, "balanced", "train.mdb")
OUT = os.path.join(SCRATCH, "balanced")
CROPS = os.path.join(OUT, "crops")
PER_PROV = 20
# 皖是这份数据集的绝对多数（唯一车牌 3150 vs 其余 25 省合计 240），
# 不限量会把均衡度重新拉偏。
ANHUI = "\u7696"

os.makedirs(CROPS, exist_ok=True)

env = lmdb.open(MDb, subdir=False, readonly=True, lock=False, max_readers=1)
with env.begin() as txn:
    total = int(txn.get(b"num-samples") or 0)
    print(f"LMDB num-samples = {total}")

    by_prov = defaultdict(list)
    for i in range(1, total + 1):
        v = txn.get(b"label-%09d" % i)
        if not v:
            continue
        s = v.decode("utf-8")
        if not s:
            continue
        by_prov[s[0]].append((i, s))

    print(f"provinces present = {len(by_prov)}")
    full = {p: len(v) for p, v in by_prov.items()}
    print("counts:", dict(sorted(full.items(), key=lambda kv: -kv[1])))

    manifest = []
    written = 0
    for prov in sorted(by_prov):
        # 皖（本数据集的绝对多数）只取 PER_PROV 张，避免它把均衡度重新拉偏；
        # 其余省**全取唯一车牌** —— 这份数据里 25 个非皖省合计只有 240 张唯一样本，
        # 「每省 20 张」的均匀抽样会在稀有省上取不满（有的省只有 1 张），
        # 所以上限就是"全取"。
        cap = PER_PROV if prov == ANHUI else 10_000
        seen = set()
        picked = []
        for idx, label in by_prov[prov]:
            if label in seen:
                continue
            seen.add(label)
            picked.append((idx, label))
            if len(picked) >= cap:
                break
        for idx, label in picked:
            img = txn.get(b"image-%09d" % idx)
            if not img:
                continue
            # 文件名即真值 —— 与既有 _dataset/real/crops 同约定。
            #
            # ⚠️ 重名时**不能**在文件名里加后缀：评测端用 `Path.stem` 当真值，
            # `皖H89C70_8171.jpg` 会被读成 12 字符的串，凭空制造长度错误
            # （首轮实测因此把 29 条本应正确的样本记成了"模型读错"）。
            # 上面的 `seen` 已按车牌串去重，所以这里不会重名。
            safe = label.replace("/", "_").replace("\\", "_")
            path = os.path.join(CROPS, f"{safe}.jpg")
            if os.path.exists(path):
                continue
            with open(path, "wb") as f:
                f.write(img)
            written += 1
            manifest.append({
                "file": os.path.basename(path),
                "label": label,
                "province": prov,
                "lmdb_index": idx,
            })

env.close()

prov_hist = Counter(m["province"] for m in manifest)
length_hist = Counter(len(m["label"]) for m in manifest)

with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

report = {
    "source": "richjjj/chinese_license_plate_rec :: val/data.mdb (via hf-mirror.com)",
    "mdb_bytes": os.path.getsize(MDb),
    "lmdb_total": total,
    "per_province_requested": PER_PROV,
    "written": written,
    "provinces_sampled": len(prov_hist),
    "province_histogram": dict(prov_hist),
    "label_length_histogram": dict(length_hist),
    "provinces_available_full_counts": dict(sorted(full.items(), key=lambda kv: -kv[1])),
}
with open(os.path.join(OUT, "province_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print()
print(f"written = {written}")
print(f"provinces sampled = {len(prov_hist)}")
print(f"length histogram = {dict(length_hist)}")
print(f"-> {CROPS}")
