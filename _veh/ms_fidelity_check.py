#!/usr/bin/env python3
"""ms_fidelity_check.py —— 在 PC 上验证 `.ms` 是否忠实于 `.onnx`。

为什么需要
==========
DFL 改写（`patch_dfl.py`）证明的是「**ONNX** 改后图与原图数值等价」。
但真正上设备的是 `converter_lite` 产出的 `.ms` —— 中间还隔着一层转换器。
本轮新引入了 `ReduceMax / Sub / Exp / Mul / ReduceSum / Div` 六个算子，
转换器有没有把它们编对，是**另一件事**，`patch_dfl.py` 管不到。

做法
====
用 MindSpore Lite 自带的 `benchmark.exe`（Windows CPU 后端）跑 `.ms`，
用 `--benchmarkDataFile` 喂黄金输出，让它算逐元素相对误差与余弦距离。

⚠️ 布局陷阱：MS Lite 的 CPU 后端把从 NCHW ONNX 转来的输入张量**报成 NHWC**
（见 `ms_engine.h` 的 `inputFormat` 注释），benchmark 是直接 memcpy 进张量缓冲区的，
所以喂给 benchmark 的 `.bin` 必须是 **NHWC** 排布；而 onnxruntime 那边仍是 NCHW。

用法
====
  python _veh/ms_fidelity_check.py --onnx _veh/yolov5su_320_ms.onnx \
                                   --ms   _veh/yolov5su_320_veh.ms \
                                   --outdir _veh/mscheck

产出
====
  <outdir>/input_nhwc.bin   喂给 benchmark --inDataFile
  <outdir>/golden_nchw.bin  喂给 benchmark --benchmarkDataFile
  <outdir>/ref.json         输入指纹 + 黄金输出统计 + 建议的 benchmark 命令

⚠️⚠️ 实测结论（2026-09-22）：**这条路走不通，别重复尝试。**
================================================================
`benchmark.exe` 确实能加载并跑这个 `.ms`（`PrepareTime ≈ 66~76 ms`，输入张量与
输入值都正常打出），但 accuracy 对照**必然失败**：

    MarkAccuracy
    ERROR benchmark_base.cc:193 ReadTensorData] get data type failed.
    ERROR benchmark_base.cc:160 ReadCalibData] Read tensor data failed, tensor name:
    ERROR model_impl.cc:789 GetOutputByTensorName] Model does not contains tensor  .

`ReadCalibData` 是**按输出张量名**去取每个输出的数据类型与形状的，而本模型在
unified benchmark API 下 `GetOutputTensorNamesChar()` 返回的是**空名**，于是它拿空串
去查张量、必然落空。换 `--modelType=MindIR_Lite` 也一样。

顺带记下两个 flag 名的坑（官方文档与二进制里的用法串不一致）：
  - 黄金数据是 `--benchmarkDataFile` + `--benchmarkDataType`，**不是** `calibDataFile`
    （传 `calibDataFile` 会被直接拒为 "not a valid flag"）。
  - `.ms` 走默认 `--modelType=MindIR` 即可，不必写 `MindIR_Lite`。

于是 `.ms` 的逐元素保真度改由**真机侧**验证（那本来就是最终要过的关）：
`veh_ref.py --compare` 拿真机 hilog 与 PC 参考对照。
`benchmark.exe` 另外还缺 `libssp-0.dll`（在 converter 目录里），要把它那目录加进 PATH。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    if len(shape) != 4 or shape[1] != 3:
        sys.exit(f"[mscheck] 只处理 NCHW 4-D 图像输入，实际 {shape}")
    n, c, h, w = shape

    # 与 patch_dfl 自证同量纲：0..255 的浮点图（不是 0..1），这样 Exp 的溢出风险也被覆盖
    rng = np.random.default_rng(args.seed)
    x_nchw = (rng.random((n, c, h, w), dtype=np.float32) * 255.0).astype(np.float32)
    x_nhwc = np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1))

    os.makedirs(args.outdir, exist_ok=True)
    p_nhwc = os.path.join(args.outdir, "input_nhwc.bin")
    p_gold = os.path.join(args.outdir, "golden_nchw.bin")
    p_ref = os.path.join(args.outdir, "ref.json")

    x_nhwc.tofile(p_nhwc)

    y = sess.run(None, {inp.name: x_nchw})
    # MS Lite benchmark 的 calib 数据是**单输出平铺**；本项目模型恰好只有一个输出
    if len(y) != 1:
        sys.exit(f"[mscheck] 模型有 {len(y)} 个输出，benchmark 的 calib 对照只支持单输出")
    golden = np.ascontiguousarray(y[0].astype(np.float32))
    golden.tofile(p_gold)

    def sha(path: str) -> str:
        hh = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                hh.update(blk)
        return hh.hexdigest()[:16]

    ms_dir = os.path.dirname(os.path.abspath(args.ms))
    cmd = (
        f'benchmark.exe --modelFile={os.path.abspath(args.ms)} '
        f'--inDataFile={os.path.abspath(p_nhwc)} '
        f'--benchmarkDataFile={os.path.abspath(p_gold)} '
        f'--calibDataType=FLOAT --device=CPU '
        f'--accuracyThreshold=0.001 --cosineDistanceThreshold=1e-6 '
        f'--warmUpLoopCount=1 --loopCount=1'
    )
    ref = {
        "onnx": os.path.abspath(args.onnx),
        "ms": os.path.abspath(args.ms),
        "ms_bytes": os.path.getsize(args.ms),
        "ms_dir": ms_dir,
        "seed": args.seed,
        "input": {
            "onnx_layout": shape,
            "onnx_name": inp.name,
            "benchmark_layout": [n, h, w, c],
            "dtype": "float32",
            "range": [float(x_nchw.min()), float(x_nchw.max())],
        },
        "golden": {
            "shape": list(golden.shape),
            "elems": int(golden.size),
            "min": float(golden.min()),
            "max": float(golden.max()),
            "l2": float(np.sqrt((golden.astype(np.float64) ** 2).sum())),
        },
        "sha256_16": {
            "input_nhwc.bin": sha(p_nhwc),
            "golden_nchw.bin": sha(p_gold),
        },
        "benchmark_cmd": cmd,
        "note": "input_nhwc.bin 是 NHWC；golden_nchw.bin 是 .onnx 的单输出平铺（NCHW 语义，但布局无歧义）",
    }
    with open(p_ref, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)

    print(json.dumps(ref, ensure_ascii=False, indent=2))
    print()
    print("[mscheck] 在 benchmark.exe 所在目录执行：")
    print("  " + cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
