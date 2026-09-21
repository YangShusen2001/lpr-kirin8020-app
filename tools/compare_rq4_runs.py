"""#14：RQ4 **两次独立运行**的漂移对照。

读两份 `parse_rq4.py` 产出的 JSON（各自 40 轮、两次之间完全重启 App），
按**同一个窗口口径**（论文用的是 r0–17 → r36–53）算漂移，并给出：

  - 两次运行各自的漂移量（端到端 / CPU 侧 / NPU 侧）
  - 是否复现（判据见下）
  - 窗口内的热档与电池温度**是否恒定**（漂移若伴随热档上升，归因就不同）
  - 张量指纹是否逐轮恒定（证明后端计算路径未退化）

## 窗口口径为什么是 r0–17 / r36–53

那是论文 80 轮版用的窗口（`evidence/rq4_thermal_80r.csv`）：
前后各 18 轮，跳过中间，避免首帧 warm-up 与单点抖动。
**对本脚本输入的 40 轮运行，这个窗口同样可用**（40 > 53 不成立 —— 见下）。

⚠️ 40 轮跑不满 r36–53 窗口。故本脚本对 40 轮输入**改用 r0–11 / r28–39**
（前后各 12 轮、等宽、跳过中间 16 轮），并把**实际用的窗口写进输出**，
避免「用了另一个窗口却声称是同一个」这种不可比的引用。

## 判据：什么叫「复现」

    repeat = (两次的漂移**同号**) 且 (两次漂移的**量级差 < 10 pp**)

只用同号不够 —— +26.9% 与 +0.5% 同号但显然不是同一个现象。
10 pp 是经验阈值，写在输出里，便于别人不同意时改。

## 用法

    python tools/compare_rq4_runs.py <runA.json> <runB.json> [--out result.json]
"""
from __future__ import annotations

import json
import os
import statistics
import sys


def _slice(rows: list[dict], frac: float = 0.3) -> tuple[int, int, int, int]:
    """按比例给出前后两个等宽窗口（含端点）。

    40 轮 + frac=0.3 → 前 12 轮 r0–11、后 12 轮 r28–39。
    80 轮 + frac=0.225 → 前 18 轮 r0–17、后 18 轮 r36–53（与论文一致）。
    """
    n = len(rows)
    w = max(2, int(round(n * frac)))
    return 0, w - 1, n - w, n - 1


def _ms(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        if key == "totalMs":
            out.append(r["totalMs"])
        else:
            sm = r.get("stage_ms")
            if not sm:
                continue
            if key == "cpu":
                out.append(sm["detect"] + sm["rectify"] + sm["cls"])
            elif key == "npu":
                out.append(sm["recog"])
    return out


def _drift(vals: list[float], a0: int, a1: int, b0: int, b1: int) -> dict:
    early = vals[a0:a1 + 1]
    late = vals[b0:b1 + 1]
    if not early or not late:
        return {}
    me, ml = statistics.mean(early), statistics.mean(late)
    return {
        "early_mean": round(me, 2),
        "late_mean": round(ml, 2),
        "delta_ms": round(ml - me, 2),
        "delta_pct": round((ml - me) / me * 100, 1),
        "early_n": len(early),
        "late_n": len(late),
    }


def analyse(path: str, label: str) -> dict:
    rows = json.load(open(path, encoding="utf-8"))
    rows = [r for r in rows if r.get("totalMs") is not None]
    a0, a1, b0, b1 = _slice(rows)
    res = {
        "label": label,
        "source": os.path.basename(path),
        "n_rounds": len(rows),
        "window": {"early": f"r{a0}–r{a1}", "late": f"r{b0}–r{b1}"},
        "drift": {},
        "thermal": sorted({r["thermal"] for r in rows}),
        "battC_range": [min(r["battC"] for r in rows),
                        max(r["battC"] for r in rows)],
        "soc_range": [min(r["soc"] for r in rows), max(r["soc"] for r in rows)],
        "charging": sorted({r["charging"] for r in rows}),
        "load_range": sorted({r["load"] for r in rows}),
        # 张量指纹：各角色的取值集合大小。1 = 逐轮恒定。
        "fingerprint": {
            role: len({r["backends"][role]["l2"] for r in rows if role in r["backends"]})
            for role in ("det", "rec")
        },
        "code_values": len({r.get("code") for r in rows}),
    }
    for key in ("totalMs", "cpu", "npu"):
        res["drift"][key] = _drift(_ms(rows, key), a0, a1, b0, b1)
    return res


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outp = None
    if "--out" in sys.argv:
        outp = sys.argv[sys.argv.index("--out") + 1]
    if len(args) < 2:
        raise SystemExit(__doc__)

    a = analyse(args[0], "runA")
    b = analyse(args[1], "runB")

    print(f"{'':16s} {'runA':>12s} {'runB':>12s}")
    for key, name in (("totalMs", "端到端"), ("cpu", "CPU 侧"), ("npu", "NPU 侧")):
        da, db = a["drift"].get(key, {}), b["drift"].get(key, {})
        fa = f"{da.get('delta_pct', float('nan')):+.1f}%" if da else "-"
        fb = f"{db.get('delta_pct', float('nan')):+.1f}%" if db else "-"
        print(f"  {name:12s} {fa:>12s} {fb:>12s}")
    print()
    print(f"  runA: n={a['n_rounds']} 窗口 {a['window']['early']} → {a['window']['late']}"
          f"  thermal={a['thermal']} 电池 {a['battC_range']}℃ chg={a['charging']}"
          f" 指纹 det/rec={a['fingerprint']['det']}/{a['fingerprint']['rec']}"
          f" code集合={a['code_values']}")
    print(f"  runB: n={b['n_rounds']} 窗口 {b['window']['early']} → {b['window']['late']}"
          f"  thermal={b['thermal']} 电池 {b['battC_range']}℃ chg={b['charging']}"
          f" 指纹 det/rec={b['fingerprint']['det']}/{b['fingerprint']['rec']}"
          f" code集合={b['code_values']}")

    # 复现判据
    pa = a["drift"].get("totalMs", {}).get("delta_pct")
    pb = b["drift"].get("totalMs", {}).get("delta_pct")
    verdict = {
        "same_sign": (pa is not None and pb is not None and pa * pb > 0),
        "magnitude_gap_pp": round(abs(pa - pb), 1) if pa is not None and pb is not None else None,
        "threshold_pp": 10.0,
    }
    verdict["repeated"] = bool(verdict["same_sign"]
                               and verdict["magnitude_gap_pp"] is not None
                               and verdict["magnitude_gap_pp"] < verdict["threshold_pp"])
    verdict["statement"] = (
        "复现" if verdict["repeated"] else "未复现"
    )
    print()
    print(f"  runA 漂移 {pa:+.1f}%  runB 漂移 {pb:+.1f}%")
    print(f"  同号={verdict['same_sign']}  量级差={verdict['magnitude_gap_pp']} pp"
          f" (阈值 {verdict['threshold_pp']} pp)")
    print(f"  ⇒ 判定：**{verdict['statement']}**")

    payload = {"runA": a, "runB": b, "verdict": verdict}
    if outp:
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"-> {outp}")
    return 0 if verdict["repeated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
