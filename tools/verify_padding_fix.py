"""验证 padding 清零修复（宿主机可跑，无需设备）。

两部分：

**一、数值验证** —— 对照 Python 参照实现检查编码张量的 padding 区。
  修复前：out.resize() 不清零，padding 列 [resizedW, targetW) 保留上次调用的值。
  修复后：out.assign(3*plane, 0) —— padding 列恒为 0，与 hlpr_reference.py:178
  的 `padding_im = np.zeros(...)` 一致。

**二、源码回归守卫** —— 直接读 lpr_pipeline.cpp，断言 `LprEncodePlateInto`
  仍然零填充输出缓冲。

  为什么需要这一条：数值验证里的 C++ 逻辑是**手抄**的，会与真实代码漂移 ——
  有人把 assign 改回 resize，数值验证仍会通过。所以要直接检查源文件。
  这一条只断言「写缓冲前先清零」，不检查算术，因此不会误报重构。

用法：
    python tools/verify_padding_fix.py
退出码 0 = 两部分都通过。
"""
import math
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, "..", "LprDemo", "entry", "src", "main", "cpp",
                   "lpr_pipeline.cpp")

# ---------------------------------------------------------------- 参照实现
def encode_reference(image, imgH, imgW, limited_max_width=160, limited_min_width=48):
    """逐行照抄 hlpr_reference.py :: encode_images（唯一差别是不做 resize，
    直接用给定尺寸的图像当输入）。"""
    imgC = 3
    max_wh_ratio = max(image.shape[1] / image.shape[0], imgW / imgH)
    imgW = int(imgH * max_wh_ratio)
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = image.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)

    # cv2.resize 的替身：这里只关心尺寸，值本身用确定性填充
    resized_image = np.full((imgH, resized_w, imgC), 0.5, dtype="float32")
    resized_image = (resized_image.transpose((2, 0, 1)) - 127.5) / 127.5

    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)   # ← 关键
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im, resized_w, imgW


# ---------------------------------------------------------------- 旧 C++ 语义
def encode_cpp_old(scratch, image, imgH, imgW, limited_max_width=160, limited_min_width=48):
    """复刻修复前的 LprEncodePlateInto：out.resize() 复用同一块缓冲。"""
    imgC = 3
    max_wh_ratio = max(image.shape[1] / image.shape[0], imgW / imgH)
    imgW = int(imgH * max_wh_ratio)
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = image.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)

    plane = imgH * imgW
    # std::vector::resize —— 仅在元素数变大时才值初始化
    if scratch["n"] < 3 * plane:
        scratch["buf"] = np.zeros(3 * plane, dtype=np.float32)
        scratch["n"] = 3 * plane
    buf = scratch["buf"]
    for y in range(imgH):
        for x in range(resized_w):
            i = y * imgW + x
            buf[i] = -0.5      # b
            buf[plane + i] = -0.5
            buf[2 * plane + i] = -0.5
    return buf.reshape(imgC, imgH, imgW)


def encode_cpp_new(scratch, image, imgH, imgW, limited_max_width=160, limited_min_width=48):
    """复刻修复后：assign(3*plane, 0) 显式清零。"""
    imgC = 3
    max_wh_ratio = max(image.shape[1] / image.shape[0], imgW / imgH)
    imgW = int(imgH * max_wh_ratio)
    imgW = max(min(imgW, limited_max_width), limited_min_width)
    h, w = image.shape[:2]
    ratio = w / float(h)
    ratio_imgH = math.ceil(imgH * ratio)
    ratio_imgH = max(ratio_imgH, limited_min_width)
    resized_w = imgW if ratio_imgH > imgW else int(ratio_imgH)

    plane = imgH * imgW
    buf = np.zeros(3 * plane, dtype=np.float32)   # ← assign 语义
    for y in range(imgH):
        for x in range(resized_w):
            i = y * imgW + x
            buf[i] = -0.5
            buf[plane + i] = -0.5
            buf[2 * plane + i] = -0.5
    scratch["buf"] = buf
    scratch["n"] = 3 * plane
    return buf.reshape(imgC, imgH, imgW)


def check_source_guard():
    """源码守卫：LprEncodePlateInto 必须在写 padding 之前把输出缓冲清零。

    这是回归守卫，不是等价性检查 —— 它只保证「不会退回 resize()」，从而不会
    再让另一条布局的残值留在 padding 里。
    """
    print()
    print("=== 源码回归守卫 ===")
    if not os.path.exists(CPP):
        print(f"!! 找不到源文件: {os.path.normpath(CPP)}")
        return False
    src = open(CPP, encoding="utf-8", errors="replace").read()

    m = re.search(r"static void LprEncodePlateInto\([^)]*\)\s*\{", src, re.S)
    if not m:
        print("!! 没找到 LprEncodePlateInto 的定义")
        return False
    # 取函数体：从定义起大括号配对
    i = m.end() - 1
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    body = src[i:j + 1]
    print(f"  函数体 {len(body)} 字符")

    ok = True

    # 1) 必须有清零（assign 带初值，或显式 fill/clear+resize）
    zeroing = re.search(r"out\.assign\s*\(\s*[^,]+,\s*0(?:\.0f?)?\s*\)", body)
    if zeroing:
        print(f"  [OK] 找到零填充: {zeroing.group(0).strip()}")
    else:
        print("  [FAIL] 未找到 out.assign(n, 0) 形式的零填充")
        ok = False

    # 2) 不得退回裸 resize（那是这次缺陷的形态）
    bare = re.search(r"out\.resize\s*\(", body)
    if bare:
        print(f"  [FAIL] 发现裸 out.resize() —— 正是导致 padding 留残值的写法: "
              f"{bare.group(0).strip()}")
        ok = False
    else:
        print("  [OK] 没有裸 out.resize()")

    # 3) 引用了参照实现（可追溯性）
    if "np.zeros" in body or "padding" in body.lower():
        print("  [OK] 注释里保留了与参照 np.zeros 的对应关系")
    else:
        print("  [warn] 未提及参照实现，建议在注释里写明 padding 应恒为 0")

    return ok


def main():
    imgH, imgW = 48, 160
    # 先用一个"宽"的 crop 把 padding 区写脏，再用真正的 crop
    wide = np.zeros((48, 160, 3), dtype=np.uint8)    # resizedW 会 = 160
    real = np.zeros((78, 123, 3), dtype=np.uint8)    # 真实 crop: resizedW 很小

    ref_real, rw_real, tw_real = encode_reference(real, imgH, imgW)
    ref_wide, rw_wide, tw_wide = encode_reference(wide, imgH, imgW)
    print("=== 数值验证（对照 Python 参照）===")
    print(f"crop 123x78 -> resizedW={rw_real}, targetW={tw_real}  "
          f"padding 列 = [{rw_real}, {tw_real})")
    print(f"crop 160x48 -> resizedW={rw_wide}, targetW={tw_wide}  (无 padding)")
    print()

    ok = True
    for name, fn, expect_zero in (("修复前 resize()", encode_cpp_old, False),
                                  ("修复后 assign()", encode_cpp_new, True)):
        scratch = {"buf": None, "n": 0}
        _ = fn(scratch, wide, imgH, imgW)      # 先污染
        got = fn(scratch, real, imgH, imgW)    # 再编真实 crop
        pad = got[:, :, rw_real:]
        allzero = bool(np.all(pad == 0))
        print(f"{name}: padding 区 max|v|={np.abs(pad).max():.4f}  全为 0? {allzero}")
        # 数值侧只断言「修复后的写法确实全为 0」
        if expect_zero and not allzero:
            print("  [FAIL] 修复后的写法本应给出全 0 padding")
            ok = False
        if not expect_zero and allzero:
            print("  [warn] 修复前的写法这次也全 0 —— 污染模型可能失效")

    pad_ref = ref_real[:, :, rw_real:]
    ref_zero = bool(np.all(pad_ref == 0))
    print()
    print("参照实现的对应量:")
    print(f"  padding 区 max|v|={np.abs(pad_ref).max():.4f}  全为 0? {ref_zero}")
    if not ref_zero:
        print("  [FAIL] 参照实现的 padding 不是 0，测试前提有误")
        ok = False

    ok = check_source_guard() and ok

    print()
    print("=> " + ("全部通过" if ok else "**有失败项**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
