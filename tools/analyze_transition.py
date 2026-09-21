"""分析 pipeline 日志里 rec 落点指纹与解码结果的关系。

疑点：rec.l2 在同一次会话内出现多个值（4.0489 / 4.0956 / 3.0965），
且 `used` 字段与 l2 同步翻转。先前登记为「输出逐位稳定」，需核实。
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCRATCH  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

LOG = os.path.join(SCRATCH, "live_transition.log")


def open_log(path):
    """hdc 重定向输出是 UTF-16LE（带 BOM），与设备端 UTF-8 不同。"""
    with open(path, "rb") as fh:
        head = fh.read(2)
    enc = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    return open(path, encoding=enc, errors="replace")


RE = re.compile(
    r"^(\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*?pipeline: (.*)$")

# p0 字段：code,detScore,recConf,layer,rect,cropHW,colour,chars,charProbs,stages,cropSum
RE_REC = re.compile(
    r"rec,([^,]*),([^,]*),([^,]*),([\d.]+|nan),([\d.]+|nan),(\d)")


def main():
    rows = []
    for line in open_log(LOG):
        m = RE.match(line.rstrip("\n"))
        if not m:
            continue
        ts, kv = m.group(1), m.group(2)
        r = RE_REC.search(kv)
        if not r:
            continue
        req, landed, fb, l2, l2f, used = r.groups()
        p0 = ""
        i = kv.find("p0=")
        if i >= 0:
            p0 = kv[i + 3:].split(";")[0]
        f = p0.split(",")
        code = f[0] if f else ""
        chars = f[7] if len(f) > 7 else ""
        cropSum = f[10] if len(f) > 10 else ""
        cropHW = f[5] if len(f) > 5 else ""
        rect = f[4] if len(f) > 4 else ""
        rows.append(dict(ts=ts, req=req, landed=landed, l2=l2, l2f=l2f,
                         used=used, code=code, chars=chars, cropSum=cropSum,
                         cropHW=cropHW, rect=rect))

    print(f"解析到 {len(rows)} 条带 rec 的 pipeline 记录")
    print(f"时间范围 {rows[0]['ts']} .. {rows[-1]['ts']}")
    print()

    print("=== (code, used, l2, l2AsFp16) 组合计数 ===")
    c = Counter((r["code"], r["used"], r["l2"], r["l2f"]) for r in rows)
    for (code, used, l2, l2f), n in c.most_common():
        print(f"  {n:>4}x  code={code:<10} used={used}  l2={l2:<9} l2AsFp16={l2f}")

    print()
    print("=== 输入是否变化：(cropSum, cropHW, rect) 组合 ===")
    c2 = Counter((r["cropSum"], r["cropHW"], r["rect"]) for r in rows)
    for k, n in c2.most_common():
        print(f"  {n:>4}x  cropSum={k[0]}  cropHW={k[1]}  rect={k[2]}")

    print()
    print("=== 同一输入下的输出分歧 ===")
    by_input = {}
    for r in rows:
        by_input.setdefault((r["cropSum"], r["rect"]), []).append(r)
    for k, v in by_input.items():
        codes = Counter(x["code"] for x in v)
        l2s = Counter(x["l2"] for x in v)
        print(f"  cropSum={k[0]} rect={k[1]}  n={len(v)}")
        print(f"    code  : {dict(codes)}")
        print(f"    rec.l2: {dict(l2s)}")

    print()
    print("=== l2 与 used 的交叉表 ===")
    c3 = Counter((r["l2"], r["used"]) for r in rows)
    print(f"  {'l2':<10} {'used=1':>8} {'used=0':>8}")
    for l2 in sorted({r["l2"] for r in rows}):
        a = c3.get((l2, "1"), 0)
        b = c3.get((l2, "0"), 0)
        print(f"  {l2:<10} {a:>8} {b:>8}")

    print()
    print("=== 时间线（只打印状态变化点）===")
    prev = None
    for r in rows:
        cur = (r["code"], r["l2"], r["used"])
        if cur != prev:
            print(f"  {r['ts']}  code={r['code']:<10} l2={r['l2']:<9} "
                  f"used={r['used']}  l2AsFp16={r['l2f']}")
            prev = cur


if __name__ == "__main__":
    main()
