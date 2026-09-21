"""T8: 解析算子对撞的原始 hilog，产出结构化结果表。

输入：从设备抓下来的 `op_collide_raw.txt`（hilog -x 全量）
输出：op_collide.csv / op_collide.json

注意：hilog 的历史缓冲会让同一条记录出现多次，**必须按 (op) 去重**，
取最新一条（后者覆盖前者），否则会把一个算子数成两个。
"""
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCRATCH  # noqa: E402

RAW = os.path.join(SCRATCH, "op_collide.log")
OUT_DIR = SCRATCH

# 从日志行里抠出 `OPCOLLIDE op=<name> bytes=<n> <kv...>` 的 kv 部分
RE_OP = re.compile(r"OPCOLLIDE op=(\S+) bytes=(\d+)\s*(.*)$")


def parse_kv(s):
    kv = {}
    for part in s.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv


records = OrderedDict()
with open(RAW, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = RE_OP.search(line)
        if not m:
            continue
        name, nbytes, rest = m.group(1), int(m.group(2)), m.group(3)
        kv = parse_kv(rest)
        # 后出现的覆盖先出现的（同一算子的最后一次运行才是当前状态）
        records[name] = {"op": name, "bytes": nbytes, **kv}

rows = list(records.values())
print(f"去重后算子数: {len(rows)}")


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


out = []
for r in rows:
    # run_ms_each 形如 "3.147|0.135|0.105"（三次或多次）
    ms_each = [to_float(x) for x in r.get("run_ms_each", "").split("|") if x]
    ms_each = [x for x in ms_each if x is not None]
    out.append({
        "op": r["op"],
        "bytes": r["bytes"],
        "compat": to_int(r.get("hiai_compat_code")),
        "construct": r.get("construct"),
        "build_rc": to_int(r.get("build_rc")),
        "run_rc": to_int(r.get("run_rc")),
        "p50_ms": ms_each[0] if ms_each else None,
        "min_ms": min(ms_each) if ms_each else None,
        "all_ms": ms_each,
        "in_shape": r.get("in0", ""),
        "out_shape": r.get("out0", ""),
        "o0_sum": to_float(r.get("o0_sum")),
        "out_dtype": (re.search(r"dtype=(\w+)", r.get("out0", "")) or [None, None])[1]
        if r.get("out0") else None,
    })

out.sort(key=lambda x: (x["build_rc"] != 0, x["op"]))

# 汇总
ok = [x for x in out if x["build_rc"] == 0]
bad = [x for x in out if x["build_rc"] != 0]
print(f"build 成功: {len(ok)}  失败: {len(bad)}")
print(f"run  成功: {len([x for x in out if x['run_rc'] == 0])}")
print(f"compat=0 : {len([x for x in out if x['compat'] == 0])}")
print()

with open(os.path.join(OUT_DIR, "op_collide.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

with open(os.path.join(OUT_DIR, "op_collide.csv"), "w", encoding="utf-8") as f:
    f.write("op,bytes,compat,construct,build_rc,run_rc,p50_ms,min_ms,in_shape,out_shape\n")
    for x in out:
        f.write(f"{x['op']},{x['bytes']},{x['compat']},{x['construct']},{x['build_rc']},"
                f"{x['run_rc']},{x['p50_ms']},{x['min_ms']},"
                f"\"{x['in_shape']}\",\"{x['out_shape']}\"\n")

print("=== build 失败的算子 ===")
for x in bad:
    print(f"  {x['op']:<24} compat={x['compat']} construct={x['construct']} build_rc={x['build_rc']}")
if not bad:
    print("  （无）")

print()
print("=== 延迟最低的 8 个（p50_ms）===")
for x in sorted([y for y in out if y["p50_ms"] is not None], key=lambda y: y["p50_ms"])[:8]:
    print(f"  {x['op']:<24} {x['p50_ms']:.3f} ms   {x['out_shape']}")

print()
print("=== 延迟最高的 8 个 ===")
for x in sorted([y for y in out if y["p50_ms"] is not None], key=lambda y: -y["p50_ms"])[:8]:
    print(f"  {x['op']:<24} {x['p50_ms']:.3f} ms   {x['out_shape']}")
