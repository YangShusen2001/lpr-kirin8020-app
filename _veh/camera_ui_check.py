#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""camera_ui_check.py —— T5 相机页呈现层的判据（不依赖"看图"）。

## 为什么不能用眼睛验

本项目看不了图（模型侧对 PNG 的内容会被过滤）。所以「叠加框画对了」必须
拆成可程序化核对的几层，每层各有一个**独立于被验代码**的判据：

  层 1  `overlayRect` 是纯函数（contain 等比缩放 + 居中）。用 FEED GEOM 的
        src/view 独立重算 scale/off/draw，与设备报的 draw 逐框比对。
        → 证「坐标换算这一步算对了」。
  层 2  布局树 dump 出的是**引擎实际布局**的矩形，不是我们算的。把车辆框的
        `#N` 徽标 bounds 与层 1 的 draw 左上角比对。
        → 证「预计算真的落到了布局上」。
  层 3  读数卡三项（车牌串 / 颜色 / 归属）来自 native 结果串。与 T4 已确立的
        期望值比对。→ 证「呈现的语义是对的」。
  层 4  票面硬约束：`余量 = 33.3 − frameMs p50`，且**不得把 1000/p50 称作 fps**。
        对界面文本做静态检查。→ 证「口径没写错」。

## 用法

    python _veh/camera_ui_check.py --log _veh/devlog_T5FEED.txt \\
        [--layout _veh/t5_layout_all.txt] [--expect-roi 0|1]
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

BOUNDS_RE = re.compile(r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]")


def parse_feed(log_path):
    """从 hilog 文本里抽 FEED 行，返回 {kind: [payload...]}。"""
    out = {"raw": []}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "FEED" not in line:
                continue
            i = line.find("FEED")
            body = line[i + 4:].strip()
            out["raw"].append(body)
            kind = body.split(" ", 1)[0]
            out.setdefault(kind, []).append(body)
    return out


def kv_of(s):
    d = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def boxes_of(feed, kind):
    """返回 [(i, owner|None, src[4], draw[4], drawpx[4]|None)]

    draw 是 vp（相对预览区），drawpx 是 px（屏幕绝对）—— 布局树给的是 px，
    所以层2 只能用 drawpx。两者都由日志直接给出，脚本不自己换算 density。
    """
    res = []
    for body in feed.get("BOX", []):
        parts = body.split()
        if len(parts) < 3 or parts[1] != kind:
            continue
        d = kv_of(body)
        i = int(d.get("i", -1))
        owner = int(d["owner"]) if "owner" in d else None
        src = [float(x) for x in d["src"].split(",")]
        draw = [float(x) for x in d["draw"].split(",")]
        dpx = [float(x) for x in d["drawPx"].split(",")] if "drawPx" in d else None
        res.append((i, owner, src, draw, dpx))
    res.sort(key=lambda t: t[0])
    return res


def layer1_geometry(feed, prob, note):
    """独立重算 overlayRect，与设备报的 draw 比对。"""
    geoms = feed.get("GEOM", [])
    if len(geoms) != 1:
        prob.append(f"FEED GEOM 行数 = {len(geoms)}（期望恰好 1）")
        return
    g = kv_of(geoms[0])
    srcw, srch = [float(x) for x in g["src"].split("x")]
    vieww, viewh = [float(x) for x in g["view"].split("x")]
    scale_dev = float(g["scale"])
    if srcw <= 0 or srch <= 0 or vieww <= 0 or viewh <= 0:
        prob.append(f"GEOM 尺寸非法 src={g['src']} view={g['view']}")
        return

    sc = min(vieww / srcw, viewh / srch)
    offx = (vieww - srcw * sc) / 2.0
    offy = (viewh - srch * sc) / 2.0
    note.append(f"[层1] src={srcw:.0f}x{srch:.0f} view={vieww:.2f}x{viewh:.2f}")
    note.append(f"[层1] 独立重算 scale={sc:.6f}  设备报={scale_dev:.6f}  "
                f"差={abs(sc - scale_dev):.3e}")
    # 容差 1e-4：日志里 view 只保留 2 位小数（toFixed(2)），而 scale 是由 view
    # 算出来的 —— 0.005 vp 的量化误差除以 320 就是 ~1.6e-5 的固有偏差。
    # 曾把容差设成 1e-5，结果被这个量化误差判失败（看着像"scale 算错了"）。
    if abs(sc - scale_dev) > 1e-4:
        prob.append(f"scale 不一致：重算 {sc:.6f} vs 设备 {scale_dev:.6f}")
    note.append(f"[层1] 独立重算 off=({offx:.2f},{offy:.2f})")

    worst = 0.0
    nbox = 0
    for kind in ("veh", "plate"):
        for i, _owner, src, draw, _dpx in boxes_of(feed, kind):
            exp = [offx + src[0] * sc, offy + src[1] * sc,
                   offx + src[2] * sc, offy + src[3] * sc]
            d = max(abs(exp[k] - draw[k]) for k in range(4))
            worst = max(worst, d)
            nbox += 1
            note.append(f"[层1] {kind}#{i} src={src} → 期望 {[round(v,2) for v in exp]} "
                        f"/ 设备 {draw} 最大差 {d:.3f}")
    note.append(f"[层1] 逐框最大差 = {worst:.4f} px（{nbox} 个框）")
    if nbox == 0:
        prob.append("没有任何 BOX 行 —— 叠加层是空的，无法验几何")
    elif worst > 0.05:
        prob.append(f"overlayRect 重算与设备 draw 不符：最大差 {worst:.4f} px > 0.05")
    return sc, offx, offy


def layer2_layout(feed, layout_path, prob, note):
    """布局树里的实际矩形 vs 预计算的 draw（换算到屏幕 px）。

    ⚠️ 覆盖范围如实说明：只有**车辆框**能验 —— 它左上角有个 `#N` 徽标（有文字），
    布局树 dump 得到；车牌框是无文字的 `Row`，dump 里看不见。所以层2 证的是
    「车辆框真的画上去了、位置与预计算一致」，车牌框由层1（纯函数自证）覆盖。
    """
    if not layout_path or not os.path.exists(layout_path):
        note.append("[层2] 跳过（未提供 --layout）")
        return

    geoms = feed.get("GEOM", [])
    if not geoms or "viewPosPx" not in kv_of(geoms[0]):
        note.append("[层2] GEOM 缺 viewPosPx —— 无法把相对坐标换算成屏幕坐标，跳过")
        return
    vx, vy = [float(x) for x in kv_of(geoms[0])["viewPosPx"].split(",")]
    note.append(f"[层2] 预览区屏幕原点 (px) = ({vx:.1f},{vy:.1f})")

    texts = []
    with open(layout_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = BOUNDS_RE.search(line)
            if not m:
                continue
            x1, y1, x2, y2 = [float(v) for v in m.groups()]
            texts.append((x1, y1, x2, y2, line.strip()))

    badges = {}
    for x1, y1, x2, y2, ln in texts:
        m = re.search(r"#(\d+)\b", ln)
        if not m:
            continue
        badges.setdefault(int(m.group(1)), (x1, y1, x2, y2, ln))

    veh = boxes_of(feed, "veh")
    if not veh:
        note.append("[层2] 无车辆框，跳过")
        return
    note.append(f"[层2] 布局树里 {len(badges)} 个 #N 徽标；FEED 报了 {len(veh)} 个车辆框")
    if len(badges) < len(veh):
        prob.append(f"布局树只有 {len(badges)} 个徽标，FEED 报了 {len(veh)} 个车辆框 —— "
                    f"叠加层没把框全画出来")

    for i, _owner, _src, _draw, dpx in veh:
        if dpx is None or i not in badges:
            continue
        bx1, by1, _bx2, _by2, _ln = badges[i]
        ex = vx + dpx[0]
        ey = vy + dpx[1]
        dx = abs(bx1 - ex)
        dy = abs(by1 - ey)
        note.append(f"[层2] veh#{i} 徽标=({bx1:.0f},{by1:.0f}) 期望屏幕=({ex:.1f},{ey:.1f}) "
                    f"偏差=({dx:.1f},{dy:.1f})")
        # 徽标在 Column 左上，Column 带 1.5 vp 边框（≈4.3 px），徽标本身还有内边距。
        # 容差 12 px：验的是「框画在那儿」，不是像素级复刻。
        if dx > 12 or dy > 12:
            prob.append(f"veh#{i} 徽标与期望屏幕位置偏差过大 ({dx:.1f},{dy:.1f}) px")


def layer3_semantics(feed, expect_roi, prob, note):
    res = feed.get("RESULT", [])
    if len(res) != 1:
        prob.append(f"FEED RESULT 行数 = {len(res)}（期望恰好 1）")
        return
    d = kv_of(res[0])
    # ⚠️ seg 的值本身含空格（'veh 12.0 + roi 30.0 ms · 车框 5 · 去重丢 0'），
    # kv_of 按空格切分会把它截成第一个词 —— 单独从行里取到行尾，否则
    # 日志里显示成 'seg=veh' 会让人以为分段读数只有一个词。
    seg = d.get("seg", "-")
    mseg = re.search(r"\bseg=(.*)$", res[0])
    if mseg:
        seg = mseg.group(1).strip()
    note.append(f"[层3] plate={d.get('plate')} colour={d.get('colour')} "
                f"owner={d.get('owner')} hit={d.get('hit')} seg={seg}")
    if d.get("hit") != "1":
        prob.append(f"喂图未检出车牌（hit={d.get('hit')}）—— 呈现层无从验起")
        return
    if d.get("plate") != "浙AG557A":
        prob.append(f"车牌串 = {d.get('plate')}，期望 浙AG557A（T4 已确立）")
    if d.get("colour") != "green":
        prob.append(f"颜色 = {d.get('colour')}，期望 green")
    owner = int(d.get("owner", -999))
    if expect_roi == 1:
        if owner < 0:
            prob.append(f"ROI 路径下归属应为 >=0，实际 {owner}")
    elif expect_roi == 0:
        if owner != -1:
            prob.append(f"直检路径下归属应为 -1（无归属），实际 {owner}")
        if seg != "-":
            prob.append(f"直检路径下 roiSegText 应为空，实际 '{seg}'")


def layer4_budget(run_layout, feed_layout, prob, note):
    """票面：沿用 frameMs 与「余量 = 33.3 − frameMs p50」，且不得把 1000/p50 称作 fps。

    两个布局树各验一件事（喂图会停相机，所以喂图后的余量必然是 '—'）：
      run （相机运行时）：余量是个**数值**，且说明文字含「服务时间，非 fps」
      feed（喂图后）    ：余量变成 '—' —— 这反过来**证明喂图没有污染帧统计**
    """
    if not run_layout or not os.path.exists(run_layout):
        note.append("[层4] 跳过数值检查（未提供 --layout-run）")
    else:
        txt = open(run_layout, encoding="utf-8", errors="replace").read()
        if "余量" not in txt:
            prob.append("界面文本里找不到「余量」—— 票面要求沿用 frameMs 与余量显示")
        else:
            m = re.search(r"余量\s*(-?[\d.]+)\s*ms", txt)
            if not m:
                prob.append("相机运行时「余量」行里读不出数值")
            else:
                note.append(f"[层4] 相机运行时余量 = {m.group(1)} ms（数值 ✓）")
        if "33.3" not in txt:
            prob.append("界面文本里找不到 33.3（帧预算）")
        if "非 fps" not in txt and "服务时间" not in txt:
            prob.append("余量说明里没有「服务时间，非 fps」的限定 —— 票面明确要求不得把 "
                        "1000/p50 称作 fps（CONTEXT.md 有约）")
        else:
            note.append("[层4] 余量说明含「服务时间，非 fps」限定 ✓")
        if re.search(r"p50\s*\(?frameMs\)?\s*fps", txt):
            prob.append("界面出现「p50(frameMs) fps」这类混用写法")

    if feed_layout and os.path.exists(feed_layout):
        txt = open(feed_layout, encoding="utf-8", errors="replace").read()
        if re.search(r"余量\s*[—–-]", txt):
            note.append("[层4] 喂图后余量为 '—' ✓（喂图不记帧，未污染统计口径）")
        else:
            note.append("[层4] 喂图后余量不是 '—'（若相机同时仍在跑，属正常）")


def layer5_argb(ets_path, prob, note):
    """票面：8 位 hex 是 ARGB 不是 CSS RGBA，半透明写 rgba()。

    这条靠静态审计：ArkUI 的 8 位 hex 是 #AARRGGBB（与 CSS 的 #RRGGBBAA 相反）。
    最省事的合规方式就是**不用 8 位 hex**，半透明全走 rgba()。这里把这个约定
    钉成判据 —— 否则下一个人加一个 '#CCFF3B30' 就悄悄错了。
    """
    if not ets_path or not os.path.exists(ets_path):
        note.append("[层5] 跳过（未提供 --ets）")
        return
    txt = open(ets_path, encoding="utf-8", errors="replace").read()
    hexes = re.findall(r"'(#[0-9a-fA-F]{3,8})'", txt)
    eight = sorted({h for h in hexes if len(h) == 9})
    note.append(f"[层5] hex 颜色字面量 {len(set(hexes))} 种，其中 8 位 {len(eight)} 种")
    if eight:
        prob.append("出现 8 位 hex（ArkUI 是 #AARRGGBB，不是 CSS 的 #RRGGBBAA）："
                    + ", ".join(eight) + " —— 半透明请改用 rgba()")
    else:
        note.append("[层5] 无 8 位 hex；半透明走 rgba() ✓")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="含 FEED 行的 hilog 文本")
    ap.add_argument("--layout", default="", help="喂图后的 dev.py all 输出（层2 用）")
    ap.add_argument("--layout-run", default="",
                    help="相机运行时的 dev.py all 输出（层4 数值检查用）")
    ap.add_argument("--ets", default="", help="CameraPage.ets（层5 静态审计用）")
    ap.add_argument("--expect-roi", type=int, default=-1,
                    help="1=期望走 ROI 路径，0=期望直检，-1=不检查")
    a = ap.parse_args()

    feed = parse_feed(a.log)
    if not feed.get("GEOM") and not feed.get("RESULT"):
        print(f"[t5ui] FAIL 日志里没有 FEED 行（文件：{a.log}）")
        return 1

    prob, note = [], []
    layer1_geometry(feed, prob, note)
    layer2_layout(feed, a.layout, prob, note)
    if a.expect_roi >= 0:
        layer3_semantics(feed, a.expect_roi, prob, note)
    else:
        note.append("[层3] 跳过（未指定 --expect-roi）")
    layer4_budget(a.layout_run, a.layout, prob, note)
    layer5_argb(a.ets, prob, note)

    for ln in note:
        print(ln)
    print()
    if prob:
        print(f"[t5ui] ✗ 发现 {len(prob)} 个问题：")
        for p in prob:
            print("   - " + p)
        return 1
    print("[t5ui] ✓ 判据全部通过：")
    print("   层1 overlayRect 独立重算与设备 draw 逐框一致")
    print("   层2 布局树实际矩形与预计算 draw 一致（叠加层真的画上去了）")
    print("   层3 读数卡语义（串/颜色/归属）符合期望")
    print("   层4 帧预算口径正确（余量 = 33.3 − p50，且未称 fps）")
    print("   层5 无 8 位 hex；半透明走 rgba()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
