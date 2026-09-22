#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_yolov5_v7.py —— 用**原版 anchor-based YOLOv5（v7.0 源码）**导出静态 ONNX（T8）。

## 为什么必须绕这么大一圈

目标：把车辆检测模型换成**原版 YOLOv5**（anchor-based，无 DFL），以便过 NPU 硬门。
踩到三层坑，逐层记录（都是实测，不是推测）：

**坑 1 —— ultralytics 8.4 拒绝加载原版 checkpoint，且会静默换权重**

    TypeError: ... NOT forwards compatible with YOLOv8

更坑的是传 `--weights yolov5s.pt` 时它**静默改用 `yolov5su.pt`（v5-u，含 DFL）**，
产物却**命名成 `yolov5s_320.onnx`** —— 名字对、内容错，光看文件名发现不了。
`tools/scan_onnx.py` 一扫就露馅：

    /model.24/dfl/Transpose perm=[0, 3, 1, 2]   <-- NOT SUPPORTED

**坑 2 —— PyPI 轮子 `yolov5==7.0.14` 不可用（两处）**

    (a) 它 from huggingface_hub.utils._errors import RepositoryNotFoundError，
        而本机 huggingface_hub 1.24.0 已删除该模块（1.x 移除了 utils/_errors.py）；
        且 attempt_download() 在 PyPI 版里是**先查 hub、后查本地**，本地权重在也照样炸。
    (b) attempt_load() 函数体内是**裸 `from models.yolo import Detect, Model`**，
        而 PyPI 轮子把 models/ 放在 yolov5/ 包**内** —— 顶层 `models` 解析不到。

**坑 3 —— yolov5 上游只支持「仓库根目录在 sys.path 上」这一种用法**

v7.0 tag 的 utils/downloads.py 是 `if not file.exists():` 守卫（本地命中即返回），
且 import 写成裸 `from utils.downloads import ...`。
所以正解就是**浅克隆 v7.0 源码，把仓库根目录插到 sys.path[0] 并 chdir 过去**。

    git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git

顺带厘清许可：**yolov5 仓库是 GPL-3.0**，与 ultralytics 8.x 的 **AGPL-3.0** 不同。
（README 的 Per-asset licences 表里已按此分开记。）

## 判据（导出后必须跑）

    python tools/scan_onnx.py <产物>.onnx

两条硬门都过才算拿到合格产物（rank ≤ 4；Transpose perm 只能是 [0,1,3,2]）。
本项目已踩过一次「文件名说 A、内容是 B」，所以**一律以扫描结果为准**。

## 用法

    python _veh/export_yolov5_v7.py --weights _veh/yolov5s.pt --imgsz 320
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.join(_HERE, "third_party", "yolov5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(_HERE, "yolov5s.pt"))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="")
    ap.add_argument("--opset", type=int, default=12)
    a = ap.parse_args()

    # 权重路径必须在 chdir 之前转成绝对路径
    weights = os.path.abspath(a.weights)
    if not os.path.isfile(weights):
        print(f"[v7] 找不到权重 {weights}", file=sys.stderr)
        return 2

    repo = os.path.abspath(a.repo)
    if not os.path.isfile(os.path.join(repo, "models", "experimental.py")):
        print(f"[v7] 找不到 yolov5 v7.0 源码仓库：{repo}\n"
              f"     先跑：git clone --depth 1 --branch v7.0 "
              f"https://github.com/ultralytics/yolov5.git {repo}", file=sys.stderr)
        return 2

    # 上游设计要求：仓库根目录在 sys.path 上（裸 `models.` / `utils.`），cwd 也在仓库根
    sys.path.insert(0, repo)
    os.chdir(repo)

    # yolov5 7.0 的 utils/general.py:34 仍 `import pkg_resources as pkg`，
    # 而 setuptools >= 81 起不再随包发布 pkg_resources。全局 setuptools 已升到 83
    # 且被其他工具链占用 → 用 _veh/pylibs（旧版 setuptools，--target --no-deps 装的）
    # 追加到 sys.path 末尾补上，不降级全局环境。
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        _pylibs = os.path.join(_HERE, "pylibs")
        if os.path.isdir(os.path.join(_pylibs, "pkg_resources")):
            sys.path.append(_pylibs)
            print(f"[v7] 已追加本地库目录以补 pkg_resources：{_pylibs}")
        else:
            print(f"[v7] 缺 pkg_resources，且 {_pylibs} 里也没有。\n"
                  f"     先跑：python -m pip install \"setuptools<81\" "
                  f"--target {_pylibs} --no-deps", file=sys.stderr)
            return 2

    import torch
    from models.experimental import attempt_load

    # ---- 坑 4：PyTorch >= 2.6 把 torch.load 的 weights_only 默认值从 False 改成 True，
    # 而 yolov5 7.0（2022 年发布）在 attempt_load 里没传这个参数 → 直接
    #    _pickle.UnpicklingError: Unsupported global: GLOBAL models.yolo.Model
    # 本机 torch 2.13，必然踩。官方权重的来源可信，所以**只在本进程内**把默认值改回
    # False，并把 checkpoint 的 sha256 打出来留证（可信度靠哈希，不靠"我觉得"）。
    import hashlib
    with open(weights, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f"[v7] checkpoint sha256 = {digest}")
    print(f"[v7] checkpoint size   = {os.path.getsize(weights)} bytes")

    _orig_load = torch.load

    def _load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_trusted

    # 与上游 export.py:528 完全一致的调用方式
    model = attempt_load(weights, device="cpu", inplace=True, fuse=True)
    model.eval()

    names = getattr(model, "names", None)
    veh = []
    if isinstance(names, dict):
        veh = [(i, n) for i, n in names.items()
               if n in ("car", "bus", "truck", "motorcycle")]
        print(f"[v7] 类别数 = {len(names)}（COCO 80 类）")
    else:
        print(f"[v7] 警告：拿不到类别名表（names={type(names).__name__}），无法核对车辆类索引")

    # 锚框是 anchor-based 的判据：模型里应当有 Detect 且 anchors 非空、无 dfl
    n_anchor = 0
    has_dfl = False
    for m in model.modules():
        if hasattr(m, "anchors") and getattr(m, "anchors", None) is not None:
            try:
                n_anchor += int(m.anchors.shape[0])
            except Exception:
                pass
        if hasattr(m, "dfl"):
            has_dfl = True
    print(f"[v7] anchor 层数 = {n_anchor}  |  含 DFL = {has_dfl}"
          f"  （原版 v5s 应为 3 层 anchor、无 DFL；含 DFL 说明又被换成 v5-u）")

    out = a.out or os.path.join(_HERE, f"yolov5s_v7_{a.imgsz}.onnx")
    out = os.path.abspath(out)
    dummy = torch.zeros(1, 3, a.imgsz, a.imgsz)
    with torch.no_grad():
        raw = model(dummy)
    shape = tuple(raw.shape) if hasattr(raw, "shape") else tuple(raw[0].shape)
    print(f"[v7] 前向输出 shape = {shape}  （320 输入原版 v5s 应为 (1, 6300, 85)）")

    torch.onnx.export(
        model, dummy, out,
        opset_version=a.opset,
        input_names=["images"], output_names=["output0"],
        dynamic_axes=None,          # 静态形状：NPU 转换要求固定输入
        # ---- 坑 5：torch 2.13 的 torch.onnx.export 默认走 dynamo=True **新导出器**，
        # 它依赖 onnxscript（本机没装）。而 yolov5 7.0 的正式导出路径是
        # **旧版 TorchScript exporter**（export.py:150 那个调用）。
        # 所有 ONNX→NPU 工具链都是在旧导出器的产物上验证的，
        # 所以这里显式关掉 dynamo，不引入未经验证的新导出器。
        dynamo=False,
    )
    size = os.path.getsize(out)
    print(f"[v7] 权重 = {weights}")
    print(f"[v7] 车辆类索引 = {veh}（COCO 里属于车辆的 4 类）")
    print(f"[v7] ONNX = {out}  大小 = {size}")
    print("[v7] ⇒ 下一步必须跑 tools/scan_onnx.py 验两条硬门，别只看文件名")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
