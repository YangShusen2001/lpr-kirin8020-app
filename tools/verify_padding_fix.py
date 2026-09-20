"""验证 padding 清零修复：对照 Python 参照实现检查编码张量的 padding 区。

修复前：out.resize() 不清零，padding 列 [resizedW, targetW) 保留上次调用的值。
修复后：out.assign(3*plane, 0) —— padding 列恒为 0，与 hlpr_reference.py:178
的 `padding_im = np.zeros(...)` 一致。

这里直接复刻两份编码逻辑，用同一个 crop 连续编两次（模拟 scratch 复用），
看第二次的 padding 区是什么。
"""
import math
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

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


def main():
    imgH, imgW = 48, 160
    # 先用一个"宽"的 crop 把 padding 区写脏，再用真正的 crop
    wide = np.zeros((48, 160, 3), dtype=np.uint8)    # resizedW 会 = 160
    real = np.zeros((78, 123, 3), dtype=np.uint8)    # 真实 crop: resizedW 很小

    ref_real, rw_real, tw_real = encode_reference(real, imgH, imgW)
    ref_wide, rw_wide, tw_wide = encode_reference(wide, imgH, imgW)
    print(f"crop 123x78 -> resizedW={rw_real}, targetW={tw_real}  "
          f"padding 列 = [{rw_real}, {tw_real})")
    print(f"crop 160x48 -> resizedW={rw_wide}, targetW={tw_wide}  (无 padding)")
    print()

    for name, fn in (("修复前 resize()", encode_cpp_old), ("修复后 assign()", encode_cpp_new)):
        scratch = {"buf": None, "n": 0}
        _ = fn(scratch, wide, imgH, imgW)      # 先污染
        got = fn(scratch, real, imgH, imgW)    # 再编真实 crop
        pad = got[:, :, rw_real:]
        print(f"{name}: padding 区 max|v|={np.abs(pad).max():.4f}  "
              f"全为 0? {bool(np.all(pad == 0))}")
        print(f"  与参照逐位相同? {bool(np.array_equal(got, ref_real))}")

    print()
    print("参照实现的对应量:")
    pad_ref = ref_real[:, :, rw_real:]
    print(f"  padding 区 max|v|={np.abs(pad_ref).max():.4f}  全为 0? {bool(np.all(pad_ref == 0))}")


if __name__ == "__main__":
    main()
