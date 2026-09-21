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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCRATCH  # noqa: E402

# 用法：`parse_rq4.py <log> [outdir] [--out-prefix NAME]`
#
# 2026-09-21 改：原先用 `sys.argv[2]` 直接当 outdir，一旦带 `--out-prefix`
# 就会把选项名当成目录名。改为先摘掉带值的选项，剩下的按位置取。
_args = list(sys.argv[1:])
_out_prefix = None
if "--out-prefix" in _args:
    _i = _args.index("--out-prefix")
    _out_prefix = _args[_i + 1] if _i + 1 < len(_args) else None
    del _args[_i:_i + 2]

RAW = _args[0] if len(_args) > 0 else os.path.join(SCRATCH, "rq4_final.log")
OUT = _args[1] if len(_args) > 1 else SCRATCH


def open_log(path):
    """hdc 重定向输出是 UTF-16LE（带 BOM）。"""
    with open(path, "rb") as fh:
        head = fh.read(2)
    enc = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return open(path, encoding=enc, errors="replace")


# hilog 行形如 `09-21 02:51:10.521 65016 65016 I A0D001/pkg/Tag: RQ4 r=...`，
# 所以锚点不能是行首，匹配目标串即可。
RE_R = re.compile(
    r"RQ4 r=(\d+) t=(\d+) thermal=(-?\d+) battC=([\d.]+) soc=(\d+) chg=(\d+) "
    r"load=(\S+) totalMs=([\d.]*) code=(\S*) backends=(.*)$")
RE_P = re.compile(r"RQ4P r=(\d+) p0=(.*)$")
# 探针的行是 RQ4（不带 p0），完整 p0 在同一次推理的 pipeline: 行里。
# 两行时间戳只差几十毫秒，按"出现在该 RQ4 行之后的第一条 pipeline: 行"配对。
RE_PIPE_P0 = re.compile(r"pipeline: (.*)$")

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
# 完整 p0 的来源有两处：
#   1) 同一次推理的 `LprNative: pipeline: ...p0=...` 行 —— 它出现在 RQ4 行**之前**
#      （实测 r=0: pipeline 在 .442，RQ4 在 .521）
#   2) 探针自己另起的 `RQ4P r=N p0=...` 行 —— 出现在 RQ4 行之后
# 所以两个方向都要兜：缓存最近一条 pipeline 的 p0 给随后的 RQ4 用，
# 同时 RQ4P 出现时以它为准（同名覆盖）。
last_pipe_p0 = None
with open_log(RAW) as fh:
    for line in fh:
        s = line.strip()
        m = RE_R.search(s)
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
            if last_pipe_p0 is not None:
                p0s.setdefault(r, last_pipe_p0)
            continue
        m = RE_P.search(s)
        if m:
            p0s[int(m.group(1))] = m.group(2)   # 显式 RQ4P 优先
            continue
        if "pipeline:" in s and "p0=" in s:
            i = s.find("p0=")
            last_pipe_p0 = s[i + 3:].split(";")[0]

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
if not ok:
    print()
    print(f"!! 没有解析到任何 RQ4 轮。检查日志路径与格式：{RAW}")
    raise SystemExit(1)

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

    # 完整性校验：顶层四段是否闭合到端到端。
    # detect 是【整段】（已含 letterbox/pack/infer/decNms），故顶层分解只能是
    #   detect + rectify + recog + cls
    # 若它俩不闭合，说明有未计入的时间 —— CPU/NPU 分列就会漏掉那段。
    print()
    print("=== 完整性校验：顶层四段之和 vs 端到端 ===")
    print("  （detect 已含 letterbox/pack/infer/decNms，不可再重复相加）")
    gaps = []
    for r in seg:
        sm = r["stage_ms"]
        top = sm["detect"] + sm["rectify"] + sm["recog"] + sm["cls"]
        gaps.append(r["totalMs"] - top)
    print(f"  未计入的残差: n={len(gaps)}  mean={statistics.mean(gaps):.2f} ms  "
          f"min={min(gaps):.2f}  max={max(gaps):.2f}")
    rel = statistics.mean(gaps) / statistics.mean([r["totalMs"] for r in seg]) * 100
    print(f"  占端到端: {rel:.2f}%  -> "
          + ("闭合良好，分列未漏时间" if abs(rel) < 3 else "**存在未解释时间，分列有漏项**"))
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

# 输出文件名的**前缀**。
#
# ⚠️ 2026-09-21 修：原先无条件写死 `rq4_thermal.csv` / `.json`，于是
# **重新解析任何一份日志都会静默覆盖既有数据**。实际发生过一次：
# 解析 80 轮新日志时把 T7 的 25 轮数据覆盖了（靠 `rq4_thermal_release.csv`
# 那份副本才恢复出来）。
#
# 现在的规则：默认前缀由**输入日志的文件名**推导（去掉扩展名），
# 于是 `--rq4_thermal_device_80r.log` → `rq4_thermal_device_80r.csv`，
# 不同日志不会互相踩。要写进既有名字必须显式 `--out-prefix`。
if _out_prefix:
    PREFIX = _out_prefix
else:
    PREFIX = os.path.splitext(os.path.basename(RAW))[0]

with open(os.path.join(OUT, PREFIX + ".json"), "w", encoding="utf-8") as fh:
    json.dump(seq, fh, ensure_ascii=False, indent=1)

with open(os.path.join(OUT, PREFIX + ".csv"), "w", encoding="utf-8") as fh:
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
print(f"-> {OUT}\\{PREFIX}.csv / .json")
