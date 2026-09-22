#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest_camera_ui_check.py —— 判据脚本的元验证（不依赖设备）。

一个只会"通过"的判据脚本等于没有判据。这里用合成日志验两件事：
  ① 正确输入 → 通过；
  ② 故意把 draw 写偏 5 px / 把 scale 写错 → **必须**被抓到。

合成值取自真机实测：相机页预览区 1224x1300，源图 320x320 ⇒
scale = min(1224/320, 1300/320) = 3.825，offX = 0，offY = 38。
"""
import io
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CHECK = os.path.join(HERE, "camera_ui_check.py")

SRCW, SRCH = 320.0, 320.0
VIEWW, VIEWH = 1224.0, 1300.0
SC = min(VIEWW / SRCW, VIEWH / SRCH)
OFFX = (VIEWW - SRCW * SC) / 2
OFFY = (VIEWH - SRCH * SC) / 2
# 真机实测：density=2.8625，预览区屏幕原点 (0, 450) px（标题区之下）
DENSITY = 2.8625
VIEWPOS_PX = (0.0, 450.0)


def draw_of(src):
    return [OFFX + src[0] * SC, OFFY + src[1] * SC, OFFX + src[2] * SC, OFFY + src[3] * SC]


def drawpx_of(src):
    return [v * DENSITY for v in draw_of(src)]


# 源框取自 PC 参考（车框 b0）与 T4 直检（车牌框）
VEH_SRC = [19.54, 65.32, 245.04, 184.39]
PLATE_SRC = [107.0, 113.0, 171.0, 131.0]


def make_log(path, *, scale=SC, veh_draw=None, plate_draw=None, plate="浙AG557A",
             colour="green", owner=-1, hit="1", seg="-"):
    veh_draw = veh_draw or draw_of(VEH_SRC)
    plate_draw = plate_draw or draw_of(PLATE_SRC)
    veh_dpx = [v * DENSITY for v in veh_draw]
    plate_dpx = [v * DENSITY for v in plate_draw]
    lines = [
        f"FEED ok=1 count=1 hit={hit} veh=1 useRoi=0 src=320x320 convMs=0.00 inferMs=41.20",
        f"FEED GEOM src=320x320 view={VIEWW:.2f}x{VIEWH:.2f} scale={scale:.6f} "
        f"vehDraw=1 plateDraw=1 density={DENSITY:.4f} "
        f"viewPos={VIEWPOS_PX[0]/DENSITY:.2f},{VIEWPOS_PX[1]/DENSITY:.2f} "
        f"viewPosPx={VIEWPOS_PX[0]:.1f},{VIEWPOS_PX[1]:.1f}",
        f"FEED BOX veh i=0 src={VEH_SRC[0]:.2f},{VEH_SRC[1]:.2f},{VEH_SRC[2]:.2f},{VEH_SRC[3]:.2f} "
        f"draw={veh_draw[0]:.2f},{veh_draw[1]:.2f},{veh_draw[2]:.2f},{veh_draw[3]:.2f} "
        f"drawPx={veh_dpx[0]:.1f},{veh_dpx[1]:.1f},{veh_dpx[2]:.1f},{veh_dpx[3]:.1f}",
        f"FEED BOX plate i=0 owner={owner} "
        f"src={PLATE_SRC[0]:.2f},{PLATE_SRC[1]:.2f},{PLATE_SRC[2]:.2f},{PLATE_SRC[3]:.2f} "
        f"draw={plate_draw[0]:.2f},{plate_draw[1]:.2f},{plate_draw[2]:.2f},{plate_draw[3]:.2f} "
        f"drawPx={plate_dpx[0]:.1f},{plate_dpx[1]:.1f},{plate_dpx[2]:.1f},{plate_dpx[3]:.1f}",
        f"FEED RESULT plate={plate} colour={colour} owner={owner} hit={hit} seg={seg}",
    ]
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def make_layout(path, *, badge_dx=0.0, badge_dy=0.0, n_badge=1):
    """造一份 `dev.py all` 风格的布局树文本：车辆框的 `#N` 徽标（屏幕 px）。"""
    lines = ["[bundles] com.shusen.lprdemo, com.ohos.sceneboard"]
    for i in range(n_badge):
        dpx = drawpx_of(VEH_SRC)
        # 徽标在 Column 左上，带 1.5vp 边框 ⇒ 约 4.3 px 内缩
        x1 = VIEWPOS_PX[0] + dpx[0] + 4.3 + badge_dx
        y1 = VIEWPOS_PX[1] + dpx[1] + 4.3 + badge_dy
        lines.append(f"Text         [{x1:.0f},{y1:.0f}][{x1+30:.0f},{y1+20:.0f}]"
                     f"             com.shusen.lprdemo           #{i}")
    lines.append("Text         [45,270][526,364]              com.shusen.lprdemo"
                 "           相机实时识别")
    lines.append("Text         [256,1749][420,1789]            com.shusen.lprdemo"
                 "           余量 7.3 ms")
    lines.append("Text         [437,1752][968,1786]            com.shusen.lprdemo"
                 "           = 33.3 − frameMs p50（服务时间，非 fps）")
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def run(log, expect_roi, layout=None):
    ets = os.path.join(HERE, "..", "LprDemo", "entry", "src", "main", "ets", "pages", "CameraPage.ets")
    cmd = [PY, CHECK, "--log", log, "--ets", ets, "--expect-roi", str(expect_roi)]
    if layout:
        cmd += ["--layout", layout]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout


def main():
    tmp = os.path.join(HERE, "_selftest_logs")
    os.makedirs(tmp, exist_ok=True)
    fails = []

    # ① 正确输入（直检路径 owner=-1）
    good = os.path.join(tmp, "good.txt")
    make_log(good)
    rc, out = run(good, 0)
    print("=== ① 正确输入（期望 rc=0）===")
    print(out.strip()[-400:])
    if rc != 0:
        fails.append("正确输入被判失败")

    # ② draw 故意偏 5 px —— 必须被抓到
    bad = os.path.join(tmp, "bad_draw.txt")
    d = draw_of(VEH_SRC)
    d = [d[0] + 5, d[1], d[2], d[3]]
    make_log(bad, veh_draw=d)
    rc, out = run(bad, 0)
    print("\n=== ② draw 偏 5 px（期望 rc!=0）===")
    print(out.strip()[-400:])
    if rc == 0:
        fails.append("draw 偏 5 px 未被抓到")

    # ③ scale 写错 —— 必须被抓到
    bad2 = os.path.join(tmp, "bad_scale.txt")
    make_log(bad2, scale=SC * 1.01)
    rc, out = run(bad2, 0)
    print("\n=== ③ scale 错 1%（期望 rc!=0）===")
    print(out.strip()[-400:])
    if rc == 0:
        fails.append("scale 错 1% 未被抓到")

    # ④ ROI 路径下 owner=-1 —— 必须被抓到
    bad3 = os.path.join(tmp, "bad_owner.txt")
    make_log(bad3, owner=-1, seg="veh 12.0 + roi 30.0 ms · 车框 5 · 去重丢 0")
    rc, out = run(bad3, 1)
    print("\n=== ④ ROI 路径却 owner=-1（期望 rc!=0）===")
    print(out.strip()[-400:])
    if rc == 0:
        fails.append("ROI 路径 owner=-1 未被抓到")

    # ⑤ 车牌串读错 —— 必须被抓到
    bad4 = os.path.join(tmp, "bad_code.txt")
    make_log(bad4, plate="苏E05172")
    rc, out = run(bad4, 0)
    print("\n=== ⑤ 车牌串读错（期望 rc!=0）===")
    print(out.strip()[-300:])
    if rc == 0:
        fails.append("车牌串读错未被抓到")

    # ⑥ 布局树与预计算一致 —— 应通过（覆盖层2 + 层4）
    lay_ok = os.path.join(tmp, "layout_ok.txt")
    make_layout(lay_ok)
    rc, out = run(good, 0, layout=lay_ok)
    print("\n=== ⑥ 布局树一致（期望 rc=0）===")
    print(out.strip()[-420:])
    if rc != 0:
        fails.append("布局树一致时被判失败")

    # ⑦ 徽标偏 20 px —— 必须被抓到
    lay_bad = os.path.join(tmp, "layout_bad.txt")
    make_layout(lay_bad, badge_dx=20.0)
    rc, out = run(good, 0, layout=lay_bad)
    print("\n=== ⑦ 徽标偏 20 px（期望 rc!=0）===")
    print(out.strip()[-420:])
    if rc == 0:
        fails.append("徽标偏 20 px 未被抓到")

    print()
    if fails:
        print("[meta] ✗ 元验证失败：")
        for f in fails:
            print("   - " + f)
        return 1
    print("[meta] ✓ 元验证通过：正确输入放行，6 类错误输入全部被拦")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
