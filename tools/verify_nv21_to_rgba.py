"""回归守卫：NV21→RGBA 的旋转分块优化必须与现状**逐位相等**。

## 背景

`lpr_pipeline.cpp` 的 `LprNv21ToRgba` 每帧做 NV21→RGBA（640x480 = 307200 像素），
相机档实测 `conv` p50 = **7.15 ms**（`evidence/camera_sweep1.log`，n=81）。
按 307200 像素折算 = **23 ns/像素** —— 对一个只有 6 次整数乘法的定点 BT.601
循环来说快得离谱地慢。

## 已定位的原因：转置型旋转的**散射写出**

`rot=90`（相机档的实际取值）时目标坐标是

    dx = height - 1 - y,  dy = x

而内层循环走的是 `x`。于是 `dy = x` 每步 +1，写出地址每次跨
`out.width * 4 = 480*4 = 1920` 字节 —— **每写一个 4 字节像素就碰一条新的缓存行，
且每条缓存行只用掉 4/64 字节**。307200 次写出 = 307200 条缓存行 = 19.6 MB 的
无效流量（真实数据只有 1.2 MB）。

`rot=270` 同理（`dy = width - 1 - x`）。`rot=0/180` 的内层写出是行内连续的，
不受影响。

## 修法：按**源空间**分块（blocked transpose）

把源切成 16x16 的块，块内先做颜色换算、再转置写出：

- 读：块内 16 行，每行 16 字节 —— 每行 1 条缓存行装 16 个有用字节（放大 4x）
- 写：`rot=90` 时块映射到 16 条输出行，每行 16 像素 = **64 字节，正好一条缓存行**

两侧都接近顺序访问，缓存行利用率从 4/64 升到 64/64。

## 本守卫做什么

1. **数值等价性** —— 用 Python 重实现「现状」与「分块版」，在多种
   尺寸/stride/旋转角上断言输出**逐字节相等**。
2. **坐标映射的正确性** —— 单独用手算可验证的标记点断言四个旋转角的
   落点，防止「现状」与「分块版」以同一种方式一起写错。
3. **敏感性** —— 确认本守卫真的能发现 1 个字节的差异（不是空守卫）。
4. **源码守卫** —— 确认 C++ 里分块循环确实存在。

第 4 项在改 `lpr_pipeline.cpp` **之前会红**，这是预期的 TDD 红灯。

用法：
    python tools/verify_nv21_to_rgba.py
退出码 0 = 通过。不需要设备。
"""
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "..", "LprDemo", "entry", "src", "main", "cpp",
                   "lpr_pipeline.cpp")

# 分块尺寸必须与 C++ 侧的 kNv21Tile 一致。
TILE = 16


# ---------------------------------------------------------------- 公共部分

def _yuv(y: int, u: int, v: int):
    """整数 BT.601 有限范围 —— 与 C++ 逐位一致（含算术右移对负数的行为）。

    C++ 的 `>>` 对负 int 是实现定义，但 ARM64/x86 都是算术右移；Python 的
    `>>` 对负整数同样是向下取整的算术右移，因此两者一致。
    """
    c = y - 16
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    r = 0 if r < 0 else (255 if r > 255 else r)
    g = 0 if g < 0 else (255 if g > 255 else g)
    b = 0 if b < 0 else (255 if b > 255 else b)
    return r, g, b


def _plan(width, height, stride, rotation):
    if stride <= 0:
        stride = width
    rot = ((rotation % 360) + 360) % 360
    swap = rot in (90, 270)
    out_w = height if swap else width
    out_h = width if swap else height
    return rot, stride, out_w, out_h


def _fit(nv21, width, height, stride):
    """复刻 C++ 的长度校验：ySize + stride*uvRows。"""
    y_size = stride * height
    uv_rows = (height + 1) // 2
    return len(nv21) >= y_size + stride * uv_rows


# ---------------------------------------------------------------- 现状（参考实现）

def nv21_to_rgba_ref(nv21, width, height, stride, rotation):
    """逐字复刻当前 C++：内层循环走 x，写出按旋转角散射。"""
    rot, stride, out_w, out_h = _plan(width, height, stride, rotation)
    if not _fit(nv21, width, height, stride):
        return None
    y_size = stride * height
    uv = y_size
    out = bytearray(out_w * out_h * 4)
    for i in range(3, len(out), 4):
        out[i] = 255

    for y in range(height):
        y_row = y * stride
        uv_row = uv + (y // 2) * stride
        for x in range(width):
            Y = nv21[y_row + x]
            uv_idx = (x // 2) * 2
            V = nv21[uv_row + uv_idx]
            U = nv21[uv_row + uv_idx + 1]
            r, g, b = _yuv(Y, U, V)
            if rot == 90:
                dx, dy = height - 1 - y, x
            elif rot == 180:
                dx, dy = width - 1 - x, height - 1 - y
            elif rot == 270:
                dx, dy = y, width - 1 - x
            else:
                dx, dy = x, y
            o = (dy * out_w + dx) * 4
            out[o] = r
            out[o + 1] = g
            out[o + 2] = b
            out[o + 3] = 255
    return out_w, out_h, bytes(out)


# ---------------------------------------------------------------- 候选（分块转置）

def nv21_to_rgba_tiled(nv21, width, height, stride, rotation):
    """候选实现：rot=90/270 走源空间分块；rot=0/180 与现状同构。

    与 C++ 版一一对应，改动这里必须同步改 C++。
    """
    rot, stride, out_w, out_h = _plan(width, height, stride, rotation)
    if not _fit(nv21, width, height, stride):
        return None
    y_size = stride * height
    uv = y_size
    out = bytearray(out_w * out_h * 4)
    for i in range(3, len(out), 4):
        out[i] = 255

    if rot in (90, 270):
        for y0 in range(0, height, TILE):
            y1 = min(y0 + TILE, height)
            for x0 in range(0, width, TILE):
                x1 = min(x0 + TILE, width)
                # x 在外：rot=90 时 dy=x，rot=270 时 dy=width-1-x ——
                # 固定 x 就固定了输出行，内层 y 只让 dx 在 64 字节内移动。
                for x in range(x0, x1):
                    dy = x if rot == 90 else width - 1 - x
                    uv_idx = (x // 2) * 2
                    o_row = dy * out_w * 4
                    for y in range(y0, y1):
                        Y = nv21[y * stride + x]
                        uv_row = uv + (y // 2) * stride
                        V = nv21[uv_row + uv_idx]
                        U = nv21[uv_row + uv_idx + 1]
                        r, g, b = _yuv(Y, U, V)
                        dx = height - 1 - y if rot == 90 else y
                        o = o_row + dx * 4
                        out[o] = r
                        out[o + 1] = g
                        out[o + 2] = b
                        out[o + 3] = 255
    else:
        for y in range(height):
            y_row = y * stride
            uv_row = uv + (y // 2) * stride
            for x in range(width):
                Y = nv21[y_row + x]
                uv_idx = (x // 2) * 2
                V = nv21[uv_row + uv_idx]
                U = nv21[uv_row + uv_idx + 1]
                r, g, b = _yuv(Y, U, V)
                if rot == 180:
                    dx, dy = width - 1 - x, height - 1 - y
                else:
                    dx, dy = x, y
                o = (dy * out_w + dx) * 4
                out[o] = r
                out[o + 1] = g
                out[o + 2] = b
                out[o + 3] = 255
    return out_w, out_h, bytes(out)


# ---------------------------------------------------------------- 构造样本

# 缓冲区尾部留白。**不是为了让守卫好写，而是因为现状真的会越界读 1 字节**：
#
# `LprNv21ToRgba` 校验的是 `ySize + stride * uvRows`，但内层循环取色度用的是
# `uvRow[(x/2)*2 + 1]`。当 `stride` 为奇数时，最后一行色度的最后一个像素会
# 访问到 `uv_row[stride]` —— 正好越过被校验的长度 1 字节。
#
# 相机实际给的 stride 是 640（偶），所以线上不会触发；但这是真实存在的
# 潜伏越界（读，非写）。这里留白让守卫能继续工作，同时用 check_oob() 把
# 这个事实显式记录下来，而不是靠"测试碰巧没崩"掩盖。
PAD = 64


def make_nv21(width, height, stride, seed):
    """造一张合法的 NV21：Y 平面 stride*height，VU 交错 stride*uvRows（+尾部留白）。"""
    rng = random.Random(seed)
    uv_rows = (height + 1) // 2
    n = stride * height + stride * uv_rows + PAD
    return bytes(rng.randrange(256) for _ in range(n))


def make_marker_nv21(width, height, stride, mx, my):
    """Y 全 16（黑）、UV 全 128（中性），只在 (mx,my) 放一个亮点。"""
    uv_rows = (height + 1) // 2
    n = stride * height + stride * uv_rows + PAD
    buf = bytearray(n)
    for i in range(stride * height):
        buf[i] = 16
    for i in range(stride * height, n):
        buf[i] = 128
    buf[my * stride + mx] = 235
    return bytes(buf)


# ---------------------------------------------------------------- 检查项

def check_equivalence():
    print("=== 1. 数值等价性（现状 vs 分块版，逐字节）===")
    cases = [
        # (w, h, stride, rot) —— 覆盖相机实际形态与边界
        (640, 480, 640, 90),    # 相机档实际形态
        (640, 480, 640, 270),
        (640, 480, 640, 0),
        (640, 480, 640, 180),
        (640, 480, 768, 90),    # stride > width（相机常给对齐 stride）
        (640, 480, 768, 270),
        (16, 16, 16, 90),       # 正好一个块
        (17, 16, 17, 90),       # 非块对齐
        (16, 17, 16, 270),
        (15, 15, 15, 90),       # 奇数尺寸
        (33, 7, 40, 270),       # 奇数 + stride 填充
        (1, 1, 1, 90),
        (2, 3, 4, 270),
        (31, 32, 32, 90),
        (32, 31, 32, 270),
    ]
    ok = True
    for i, (w, h, st, rot) in enumerate(cases):
        nv = make_nv21(w, h, st, seed=100 + i)
        a = nv21_to_rgba_ref(nv, w, h, st, rot)
        b = nv21_to_rgba_tiled(nv, w, h, st, rot)
        if a is None or b is None:
            print(f"  [FAIL] {w}x{h} st={st} rot={rot} 参考实现返回 None")
            ok = False
            continue
        same = (a == b)
        if not same:
            ok = False
            diff = next(j for j in range(len(a[2])) if a[2][j] != b[2][j])
            print(f"  [FAIL] {w}x{h} st={st} rot={rot} 首个差异 @byte {diff}: "
                  f"{a[2][diff]} vs {b[2][diff]}")
        else:
            print(f"  [OK]   {w}x{h:<5} st={st:<4} rot={rot:<3} "
                  f"out={a[0]}x{a[1]} n={len(a[2]):>7} 逐字节相等")
    return ok


def check_sanity():
    """手算可验证的坐标映射 —— 防止参考实现与候选一起写错。"""
    print()
    print("=== 2. 坐标映射（标记点必须落在手算位置上）===")
    w, h, st = 8, 6, 8
    mx, my = 1, 2
    # 期望落点（由 dx/dy 公式手算，与两个实现无关）
    expect = {
        0:   (mx, my),
        90:  (h - 1 - my, mx),
        180: (w - 1 - mx, h - 1 - my),
        270: (my, w - 1 - mx),
    }
    ok = True
    for rot, (ex, ey) in expect.items():
        nv = make_marker_nv21(w, h, st, mx, my)
        res = nv21_to_rgba_ref(nv, w, h, st, rot)
        out_w, out_h, data = res
        # 收集所有亮像素
        lit = []
        for py in range(out_h):
            for px in range(out_w):
                o = (py * out_w + px) * 4
                if data[o] > 200 and data[o + 1] > 200 and data[o + 2] > 200:
                    lit.append((px, py))
        good = (lit == [(ex, ey)])
        if not good:
            ok = False
        print(f"  {'[OK]  ' if good else '[FAIL]'} rot={rot:<3} 期望亮点 ({ex},{ey})  "
              f"实测 {lit}")
        # 候选版必须给出同一结果
        c = nv21_to_rgba_tiled(nv, w, h, st, rot)
        if c != res:
            ok = False
            print(f"         [FAIL] 分块版与现状不等（rot={rot}）")
    return ok


def check_oob():
    """记录一个**潜伏的越界读**：奇数 stride 时色度会多读 1 字节。

    C++ 校验的长度是 `ySize + stride * uvRows`，但取色度用
    `uvRow[(x/2)*2 + 1]`，x = width-1 且 width 为奇数时落到 `uv_row[stride]`。
    相机给的是 stride=640（偶），线上不触发；这里把它显式记下来。

    本函数**不判失败** —— 它记录的是现状事实，不是本次优化的回归。
    若将来有人收紧了长度校验（+1 字节），这里会提示需要同步更新。
    """
    print()
    print("=== 2b. 潜伏越界读（记录，不判失败）===")
    for w, h, st in ((15, 15, 15), (33, 7, 33), (640, 480, 640)):
        uv_rows = (h + 1) // 2
        checked = st * h + st * uv_rows
        # 内层最大访问下标（不含 +1 的 U）
        max_uv_idx = ((w - 1) // 2) * 2
        max_access = st * h + (h // 2) * st + max_uv_idx + 1
        over = max_access + 1 - checked
        print(f"  {w}x{h} st={st}: 校验长度={checked} 最大访问={max_access + 1} "
              f"-> {'越界 ' + str(over) + ' 字节' if over > 0 else '无越界'}")
    return True


def check_sensitivity():
    """守卫必须能发现差异，否则是空守卫。"""
    print()
    print("=== 3. 敏感性（守卫能发现 1 像素差异）===")
    w, h, st = 64, 48, 64
    nv = make_nv21(w, h, st, seed=777)
    base = nv21_to_rgba_ref(nv, w, h, st, 90)[2]
    mutated = bytearray(nv)
    mutated[5 * st + 7] ^= 0xFF          # 翻一个 Y 字节
    other = nv21_to_rgba_ref(bytes(mutated), w, h, st, 90)[2]
    detected = (base != other)
    print(f"  翻转 1 个 Y 字节 -> 输出{'不同（守卫有效）' if detected else '相同（守卫无效）'}")
    # 再确认「分块版自己也对差异敏感」
    t = nv21_to_rgba_tiled(bytes(mutated), w, h, st, 90)[2]
    detected2 = (t != nv21_to_rgba_tiled(nv, w, h, st, 90)[2])
    print(f"  分块版对同一改动{'敏感' if detected2 else '不敏感（FAIL）'}")
    return detected and detected2


def check_source():
    """源码守卫：C++ 里确实有分块循环（改代码前为红灯，属预期）。"""
    print()
    print("=== 4. 源码守卫（lpr_pipeline.cpp）===")
    if not os.path.exists(CPP):
        print(f"  [FAIL] 找不到 {CPP}")
        return False
    with open(CPP, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    start = text.find("bool LprNv21ToRgba(")
    if start < 0:
        print("  [FAIL] 找不到 LprNv21ToRgba 定义")
        return False
    body = text[start:]
    ok = True
    if "kNv21Tile" in body:
        print("  [OK] 存在 kNv21Tile 分块常量")
    else:
        print("  [FAIL] 未见 kNv21Tile —— 分块转置尚未落地")
        ok = False
    if "rot == 90 || rot == 270" in body:
        print("  [OK] 存在转置型旋转的分支")
    else:
        print("  [FAIL] 未见转置型旋转分支")
        ok = False
    return ok


def main() -> int:
    ok = check_equivalence()
    ok = check_sanity() and ok
    check_oob()
    ok = check_sensitivity() and ok
    src = check_source()
    print()
    if ok and src:
        print("=> 全部通过：分块版与现状逐位相等，坐标映射正确，守卫敏感")
        return 0
    if ok and not src:
        print("=> 数值部分通过；源码守卫未通过（若尚未落地分块优化，这是预期的红灯）")
        return 1
    print("=> 未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())
