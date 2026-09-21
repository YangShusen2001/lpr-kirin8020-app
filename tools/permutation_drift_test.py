"""#14 的漂移检验：对单次运行内的「前窗 vs 后窗」做**置换检验**。

## 为什么不能只看两个均值相减

论文原口径是「r0–17 mean 64.44 → r36–53 mean 81.76，+26.9%」——
两个**点估计**相减。本项目的 RQ4 单轮 CV 就有 ~20–27%，
在这种噪声下「两个 18 轮窗口的均值差」本身有很宽的分布：
后窗只要恰好抽到几个高值，就能造出 +20% 的「漂移」。

**所以「漂移」这个量必须带一个「纯噪声能否造出同样大小的差」的对照。**

## 检验设计

零假设 H0：**前后两窗同分布**（即没有真实漂移，观测到的差来自轮间抖动）。

检验统计量：|mean(后窗) − mean(前窗)| / mean(全段)。
做法：把整段 40 轮的 `totalMs` **在同一段内随机置换**（打乱顺序、
数值集合不变），重算同一个统计量，重复 N 次，得到 H0 下的分布；
再看真实值在分布中的位置 → 经验 p 值。

置换而非重抽样（bootstrap）的理由：40 个点是**同一段连续负载**的时间序列，
重抽样会破坏它的自相关结构，置换同样破坏但更保守地保留了「同一批数值」。

## 用法

    python tools/permutation_drift_test.py <run.json> [--iters 20000] [--frac 0.3]

退出码：0 = 漂移显著（p<0.05）；1 = 不显著。
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys


def stat(vals: list[float], frac: float) -> float:
    n = len(vals)
    w = max(2, int(round(n * frac)))
    early, late = vals[:w], vals[-w:]
    m = statistics.mean(vals)
    return (statistics.mean(late) - statistics.mean(early)) / m


def test(path: str, frac: float = 0.3, iters: int = 20000,
         seed: int = 20260922) -> dict:
    rows = [r for r in json.load(open(path, encoding="utf-8"))
            if r.get("totalMs") is not None]
    vals = [r["totalMs"] for r in rows]
    obs = stat(vals, frac)

    rng = random.Random(seed)
    null = []
    buf = list(vals)
    for _ in range(iters):
        rng.shuffle(buf)
        null.append(stat(buf, frac))

    # 双尾经验 p：H0 下 |统计量| ≥ |观测| 的比例
    p = (sum(1 for x in null if abs(x) >= abs(obs)) + 1) / (iters + 1)
    n = len(vals)
    w = max(2, int(round(n * frac)))
    return {
        "source": os.path.basename(path),
        "n_rounds": n,
        "window_rounds": w,
        "observed_delta_pct": round(obs * 100, 1),
        "null_mean_pct": round(statistics.mean(null) * 100, 2),
        "null_stdev_pct": round(statistics.stdev(null) * 100, 2),
        "null_p95_abs_pct": round(
            sorted(abs(x) for x in null)[int(iters * 0.95)] * 100, 1),
        "p_value": round(p, 4),
        "iters": iters,
        "significant_at_0.05": p < 0.05,
        "verdict": ("漂移显著于轮间噪声" if p < 0.05
                    else "**漂移不显著 —— 与轮间噪声不可区分**"),
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = 20000
    frac = 0.3
    if "--iters" in sys.argv:
        iters = int(sys.argv[sys.argv.index("--iters") + 1])
    if "--frac" in sys.argv:
        frac = float(sys.argv[sys.argv.index("--frac") + 1])
    if not args:
        raise SystemExit(__doc__)

    rc = 0
    for a in args:
        r = test(a, frac, iters)
        print(f"=== {r['source']} ===")
        print(f"  n={r['n_rounds']} 轮，前后窗各 {r['window_rounds']} 轮")
        print(f"  观测漂移   : {r['observed_delta_pct']:+.1f}%")
        print(f"  H0 零分布  : mean={r['null_mean_pct']:+.2f}%  "
              f"stdev={r['null_stdev_pct']:.2f}%  "
              f"|95% 分位|={r['null_p95_abs_pct']:.1f}%")
        print(f"  经验 p 值  : {r['p_value']}（{r['iters']} 次置换）")
        print(f"  ⇒ {r['verdict']}")
        print()
        if not r["significant_at_0.05"]:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
