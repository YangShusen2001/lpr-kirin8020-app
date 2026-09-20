"""T7: 解析 RQ4 持续负载热特性日志。

输入：设备端 rq4_thermal.log（每轮一行 RQ4 + 一行 RQ4P 完整 p0）
输出：rq4_thermal.csv / rq4_thermal.json

要点：
  - 热档 / 电池温度 / SOC / 充电状态 / 系统负载档 逐轮记录
  - 端到端延迟 + 落点证据（det/rec 的 l2 指纹）
  - 轮间方差**按 CPU / NPU 分开报**，不合并成一个数字
  - 破例轮（ok!=1）如实保留，不剔除

关于「按 CPU / NPU 分开报」：
  p0 的第 10 个字段是 9 个分段时延，顺序固定（见 napi_init.cpp:776）：
    tDetectMs | tLetterboxMs | tEncodeInferMs | tDecodeNmsMs
    tRectifyMs | tRecogMs | tClsMs | tPackMs | tInferMs
  生产档 `det=cpu / rec=nnrt / cls=cpu`，故按**后端归属**分两侧：
    CPU 侧 = tDetectMs + tRectifyMs + tClsMs   （检测整段含 CPU 推理 + 矫正 + 判色）
    NPU 侧 = tRecogMs                           （识别在 NNRT/NPU 上）
  注意这是**阶段归属的近似**，不是纯设备内核时间：tRecogMs 里仍含 NNRT 的
  host 侧前后处理。引用时必须带上这一条。
"""
import json
import os
import re
import statistics

RAW = r"C:\Users\26671\lpr-data\rq4_final.log"
OUT = r"C:\Users\26671\lpr-data"

RE_R = re.compile(
    r"^RQ4 r=(\d+) t=(\d+) thermal=(-?\d+) battC=([\d.]+) soc=(\d+) chg=(\d+) "
    r"load=(\S+) totalMs=([\d.]*) code=(\S*) backends=(.*)$")
RE_P = re.compile(r"^RQ4P r=(\d+) p0=(.*)$")

# p0 第 10 字段（index 9）的分段顺序，与 napi_init.cpp:776 严格对应
STAGES = ["detect", "letterbox", "encInfer", "decNms", "rectify",
          "recog", "cls", "pack", "infer"]
# 生产档后端归属：det=cpu / rec=nnrt / cls=cpu
CPU_STAGES = ["detect", "rectify", "cls"]
NPU_STAGES = ["recog"]


def parse_backends(b):
    """从 backends= 串里取出各角色的 landed 与 l2。"""
    out = {}
    for seg in b.split("|"):
        f = seg.split(",")
        if len(f) < 7:
            continue
        out[f[0]] = {"req": f[1], "landed": f[2], "fallback": f[3],
                     "l2": f[4], "l2AsFp16": f[5], "used": f[6]}
    return out


rows = {}
p0s = {}
with open(RAW, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = RE_R.match(line.strip())
        if m:
            r = int(m.group(1))
            rows[r] = {
                "round": r,
                "t_s": int(m.group(2)),
                "thermal": int(m.group(3)),
                "battC": float(m.group(4)),
                "soc": int(m.group(5)),
                "charging": int(m.group(6)),
                "load": m.group(7),
                "totalMs": float(m.group(8)) if m.group(8) else None,
                "code": m.group(9),
                "backends": parse_backends(m.group(10)),
            }
            continue
        m = RE_P.match(line.strip())
        if m:
            p0s[int(m.group(1))] = m.group(2)

# 把 p0 明细并入
for r, p0 in p0s.items():
    if r in rows and p0 and p0 != "-":
        f = p0.split(",")
        if len(f) >= 11:
            rows[r].update({
                "chars": f[7],
                "cropSum": f[10],
                "cropHW": f[5],
                "rect": f[4],
                "colour": f[6],
                "stages": f[9],
            })
            try:
                sv = [float(x) for x in f[9].split("|")]
            except ValueError:
                sv = []
            if len(sv) == len(STAGES):
                rows[r]["stage_ms"] = dict(zip(STAGES, sv))

seq = [rows[k] for k in sorted(rows)]
print(f"轮数: {len(seq)}")

ok = [r for r in seq if r["totalMs"] is not None]
print(f"有效轮: {len(ok)}  破例轮: {len(seq) - len(ok)}")

lat = [r["totalMs"] for r in ok]
print()
print("=== 端到端延迟 ===")
print(f"  n={len(lat)}  min={min(lat):.1f}  p50={statistics.median(lat):.1f}  "
      f"max={max(lat):.1f}  mean={statistics.mean(lat):.1f}")
if len(lat) > 1:
    print(f"  stdev={statistics.stdev(lat):.1f}  "
          f"变异系数={statistics.stdev(lat)/statistics.mean(lat)*100:.1f}%")

print()
print("=== 热态时间线 ===")
for r in seq:
    tm = f"{r['totalMs']:.1f}" if r["totalMs"] is not None else "FAIL"
    print(f"  t={r['t_s']:>4}s  thermal={r['thermal']}  {r['battC']:.1f}℃  "
          f"soc={r['soc']}  load={r['load']}  total={tm:>8} ms  {r.get('code','-')}")


def _desc(v):
    if len(v) < 2:
        return f"n={len(v)}"
    return (f"n={len(v)}  mean={statistics.mean(v):.2f}  stdev={statistics.stdev(v):.2f}  "
            f"CV={statistics.stdev(v)/statistics.mean(v)*100:.1f}%  "
            f"min={min(v):.2f}  max={max(v):.2f}")


# 轮间方差：按后端归属分开报（票面验收项）
seg = [r for r in seq if "stage_ms" in r]
print()
print(f"=== 轮间方差：按 CPU / NPU 分开报（n={len(seg)} 轮有分段）===")
if seg:
    cpu = [sum(r["stage_ms"][s] for s in CPU_STAGES) for r in seg]
    npu = [sum(r["stage_ms"][s] for s in NPU_STAGES) for r in seg]
    print(f"  CPU 侧 ({'+'.join(CPU_STAGES)}):")
    print(f"    {_desc(cpu)}")
    print(f"  NPU 侧 ({'+'.join(NPU_STAGES)}):")
    print(f"    {_desc(npu)}")
    print("  对照 —— 端到端（合并，会掩盖两侧差异）：")
    print(f"    {_desc([r['totalMs'] for r in seg])}")
    print()
    print("  逐分段 CV：")
    for s in STAGES:
        v = [r["stage_ms"][s] for r in seg]
        if statistics.mean(v) > 0:
            print(f"    {s:<10} mean={statistics.mean(v):7.2f}  "
                  f"CV={statistics.stdev(v)/statistics.mean(v)*100:5.1f}%")
else:
    print("  (日志无 p0 分段，无法分列 —— 检查探针是否记录完整 p0)")

# 落点 l2 的稳定性
det_l2 = [r["backends"]["det"]["l2"] for r in seq if "det" in r["backends"]]
rec_l2 = [r["backends"]["rec"]["l2"] for r in seq if "rec" in r["backends"]]
print()
print("=== 张量指纹（落点自证）稳定性 ===")
print(f"  det.l2 : {len(set(det_l2))} 个不同值  -> {sorted(set(det_l2))[:4]}")
print(f"  rec.l2 : {len(set(rec_l2))} 个不同值  -> {sorted(set(rec_l2))[:4]}")

# 热档分段
print()
print("=== 按热档分段的延迟 ===")
by_thermal = {}
for r in ok:
    by_thermal.setdefault(r["thermal"], []).append(r["totalMs"])
for t in sorted(by_thermal):
    v = by_thermal[t]
    print(f"  thermal={t}: n={len(v)}  p50={statistics.median(v):.1f}  "
          f"min={min(v):.1f}  max={max(v):.1f}")

with open(os.path.join(OUT, "rq4_thermal.json"), "w", encoding="utf-8") as fh:
    json.dump(seq, fh, ensure_ascii=False, indent=1)

with open(os.path.join(OUT, "rq4_thermal.csv"), "w", encoding="utf-8") as fh:
    fh.write("round,t_s,thermal,battC,soc,charging,load,totalMs,code,cropSum,"
             "det_landed,det_l2,rec_landed,rec_l2,"
             + ",".join(f"{s}_ms" for s in STAGES)
             + ",cpu_side_ms,npu_side_ms\n")
    for r in seq:
        d = r["backends"].get("det", {})
        c = r["backends"].get("rec", {})
        sm = r.get("stage_ms")
        if sm:
            stage_cols = ",".join(f"{sm[s]:.4f}" for s in STAGES)
            cpu = sum(sm[s] for s in CPU_STAGES)
            npu = sum(sm[s] for s in NPU_STAGES)
            side_cols = f"{cpu:.4f},{npu:.4f}"
        else:
            stage_cols = "," * (len(STAGES) - 1)
            side_cols = ","
        fh.write(f"{r['round']},{r['t_s']},{r['thermal']},{r['battC']},{r['soc']},"
                 f"{r['charging']},{r['load']},{r['totalMs'] or ''},{r.get('code','')},"
                 f"{r.get('cropSum','')},{d.get('landed','')},{d.get('l2','')},"
                 f"{c.get('landed','')},{c.get('l2','')},{stage_cols},{side_cols}\n")
print()
print(f"-> {OUT}\\rq4_thermal.csv / .json")
