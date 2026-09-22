# -*- coding: utf-8 -*-
"""真机驱动小工具（UTF-8 取回 hdc 输出）。

为什么不用 PowerShell：PowerShell 管道会按遗留代码页重解码中文
（`赣AD38888` -> 乱码），且其 stdout 不一定回传。这里统一走 Python subprocess。

用法：
  python dev.py state                 唤醒 + 取布局树，报告锁屏状态 / 前台 bundle
  python dev.py layout                只取布局树并打印所有带文字的控件
  python dev.py all                   取布局树并打印**所有**节点（含无文字容器）
  python dev.py struct                打印带 id / name 的节点（判断窗口归属）
  python dev.py start                 启动 com.shusen.lprdemo / EntryAbility
  python dev.py tap <x> <y>           注入点击
  python dev.py longpress <x> <y>     注入长按
  python dev.py swipe <x1> <y1> <x2> <y2> [velocity]
  python dev.py log <TAG>             读 hilog 全量并本地过滤（-x 抓全量再 grep）
  python dev.py logclear              清 hilog 缓冲
  python dev.py ps                    看 lprdemo 进程是否在跑
"""
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

HDC = r"D:\Applications\CamStreamReceiver\connection\hdc.exe"
BUNDLE = "com.shusen.lprdemo"
ABILITY = "EntryAbility"
WORK = os.path.dirname(os.path.abspath(__file__))
REMOTE_LAYOUT = "/data/local/tmp/devlayout.json"


def run(args, timeout=120, cwd=None):
    p = subprocess.run(args, capture_output=True, timeout=timeout, cwd=cwd)
    return (p.returncode,
            p.stdout.decode("utf-8", "replace"),
            p.stderr.decode("utf-8", "replace"))


def sh(cmd, timeout=120):
    return run([HDC, "shell", cmd], timeout=timeout)


def wake():
    # 唤醒与手势必须压在同一条 shell 里（否则命令间隙屏幕自动休眠）
    cmd = ("power-shell wakeup; sleep 1; "
           "uitest uiInput swipe 612 2700 612 300 4000; sleep 1; "
           "uitest uiInput swipe 612 2700 612 300 4000")
    return sh(cmd)


def dump_layout(local_name="devlayout.json"):
    rc, o, e = sh("uitest dumpLayout -p " + REMOTE_LAYOUT)
    local = os.path.join(WORK, local_name)
    try:
        if os.path.exists(local):
            os.remove(local)
    except OSError:
        pass
    # hdc 会把绝对路径拼上 cwd —— 必须 cd 到目录 + 相对文件名
    rc2, o2, e2 = run([HDC, "file", "recv", REMOTE_LAYOUT, local_name], cwd=WORK)
    if not os.path.exists(local):
        return None, "dumpLayout rc=%s out=%s err=%s | recv rc=%s out=%s err=%s" % (
            rc, o.strip(), e.strip(), rc2, o2.strip(), e2.strip())
    with open(local, encoding="utf-8") as f:
        return json.load(f), ""


def iter_nodes(n):
    yield n
    for c in (n.get("children") or []):
        for x in iter_nodes(c):
            yield x


def attr(n, key, default=""):
    t = n.get("attributes", n)
    v = t.get(key)
    if v is None:
        v = n.get(key)
    return default if v is None else v


def rows(d):
    out = []
    for n in iter_nodes(d):
        txt = str(attr(n, "text")).strip()
        if txt:
            out.append((txt, attr(n, "bounds"), attr(n, "type"), attr(n, "bundleName")))
    return out


def bundles(d):
    s = []
    for n in iter_nodes(d):
        b = attr(n, "bundleName")
        if b and b not in s:
            s.append(b)
    return s


# 锁屏判据：必须用「专有词」，不能用泛化的 lock ——
# ClockStatusView / clock_home_row / TextClock 都含 "lock"，会假阳性。
LOCK_MARKERS = ("Digital_PSD_Input", "PSD_Input_Tip", "ScreenLock", "screen_lock",
                "上滑解锁", "未识别成功", "图案解锁", "输入密码")


def is_locked(d):
    hits = []
    for n in iter_nodes(d):
        t = n.get("attributes", n)
        blob = " ".join(str(t.get(k, "")) for k in ("id", "text", "type", "name"))
        for m in LOCK_MARKERS:
            if m in blob:
                hits.append(m)
                break
    return hits


def cmd_state():
    rc, o, e = wake()
    print("[wake] rc=%s %s %s" % (rc, o.strip(), e.strip()))
    time.sleep(1)
    d, err = dump_layout()
    if d is None:
        print("[layout] FAIL " + err)
        return 1
    bs = bundles(d)
    hits = is_locked(d)
    print("[bundles] " + ", ".join(bs))
    print("[lock-marker] " + (", ".join(sorted(set(hits))) if hits else "(none)"))
    print("[verdict] " + ("LOCKED" if "com.ohos.sceneboard" in bs and hits else
                          ("SCENEBOARD-NO-MARKER" if "com.ohos.sceneboard" in bs else "UNLOCKED?")))
    for r in rows(d)[:60]:
        print("   ", r)
    return 0


def cmd_layout():
    d, err = dump_layout()
    if d is None:
        print("[layout] FAIL " + err)
        return 1
    print("[bundles] " + ", ".join(bundles(d)))
    for r in rows(d):
        print(r)
    return 0


def cmd_struct():
    """打印所有带 id / name 的节点 —— 判断窗口归属（锁屏 vs 桌面 vs 自己的 App）。"""
    d, err = dump_layout()
    if d is None:
        print("[struct] FAIL " + err)
        return 1
    n_btn = 0
    for n in iter_nodes(d):
        a = n.get("attributes", n)
        t = str(a.get("type") or "")
        if t.lower() == "button":
            n_btn += 1
        i = str(a.get("id") or "")
        nm = str(a.get("name") or "")
        if i or nm:
            print("%-14s id=%-34s name=%-28s %s"
                  % (t, i[:34], nm[:28], a.get("bounds", "")))
    print("[struct] buttons=%d" % n_btn)
    return 0


def cmd_start():
    rc, o, e = sh("aa start -a %s -b %s" % (ABILITY, BUNDLE))
    print("[start] rc=%s out=%s err=%s" % (rc, o.strip(), e.strip()))
    return 0 if "start ability successfully" in (o + e) else 1


def cmd_tap(x, y):
    rc, o, e = sh("uitest uiInput click %d %d" % (x, y))
    print("[tap %d,%d] rc=%s out=%s err=%s" % (x, y, rc, o.strip(), e.strip()))
    return 0


def cmd_swipe(x1, y1, x2, y2, v=4000):
    rc, o, e = sh("uitest uiInput swipe %d %d %d %d %d" % (x1, y1, x2, y2, v))
    print("[swipe] rc=%s out=%s err=%s" % (rc, o.strip(), e.strip()))
    return 0


def cmd_log(tag, clear=False):
    if clear:
        sh("hilog -r")
    rc, o, e = sh("hilog -x", timeout=180)
    lines = [l for l in o.splitlines() if tag in l]
    out = os.path.join(WORK, "devlog_%s.txt" % re.sub(r"\W+", "_", tag))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print("[log] tag=%s lines=%d -> %s" % (tag, len(lines), out))
    for l in lines[:80]:
        print("   ", l)
    return 0


def cmd_logclear():
    rc, o, e = sh("hilog -r")
    print("[logclear] rc=%s %s" % (rc, o.strip()))
    return 0


def cmd_ps():
    rc, o, e = sh("ps -ef")
    ls = [l for l in o.splitlines() if "lprdemo" in l]
    print("[ps] lprdemo lines=%d" % len(ls))
    for l in ls:
        print("   ", l)
    return 0


def cmd_all():
    """打印**所有**节点（含无文字容器）—— 叠加层的车牌框是无文字的 Row，
    只有这个视图能看见它；`layout` 只吐带文字的控件，会漏掉它。"""
    d, err = dump_layout()
    if d is None:
        print("[all] FAIL " + err)
        return 1
    print("[bundles] " + ", ".join(bundles(d)))
    for n in iter_nodes(d):
        t = str(attr(n, "type") or "")
        b = str(attr(n, "bounds") or "")
        if not b:
            continue
        txt = str(attr(n, "text")).strip()
        bn = str(attr(n, "bundleName") or "")
        print("%-12s %-24s %-28s %s" % (t, b, bn[:28], txt[:26]))
    return 0


def cmd_longpress(x, y):
    rc, o, e = sh("uitest uiInput longClick %d %d" % (x, y))
    print("[longpress %d,%d] rc=%s out=%s err=%s" % (x, y, rc, o.strip(), e.strip()))
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    c = sys.argv[1]
    a = sys.argv[2:]
    if c == "state":
        return cmd_state()
    if c == "layout":
        return cmd_layout()
    if c == "all":
        return cmd_all()
    if c == "struct":
        return cmd_struct()
    if c == "start":
        return cmd_start()
    if c == "tap":
        return cmd_tap(int(a[0]), int(a[1]))
    if c == "longpress":
        return cmd_longpress(int(a[0]), int(a[1]))
    if c == "swipe":
        return cmd_swipe(int(a[0]), int(a[1]), int(a[2]), int(a[3]),
                         int(a[4]) if len(a) > 4 else 4000)
    if c == "log":
        return cmd_log(a[0], clear=("--clear" in a))
    if c == "logclear":
        return cmd_logclear()
    if c == "ps":
        return cmd_ps()
    print("unknown command: " + c)
    return 2


if __name__ == "__main__":
    sys.exit(main())
