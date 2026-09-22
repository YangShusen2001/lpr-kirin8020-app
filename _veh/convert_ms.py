#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_ms.py —— 调 MindSpore Lite converter 把 ONNX 转成 .ms（T8 第三步）。

## 本机已验证的调用方式（来自 D:\\Tools\\mindspore-lite\\run_conv.bat，不自己发明）

    CONVDIR = D:\\Tools\\mindspore-lite\\mindspore-lite-2.6.0-win-x64\\tools\\converter
    PATH    = %CONVDIR%\\lib;%CONVDIR%\\converter;%PATH%
    cwd     = 转换器目录（技能第四节：否则找不到依赖库）

## 历史日志给的先验（同一台机器留下的 yolo_conv*.txt）

| 日志 | 模型 | 结果 |
|---|---|---|
| yolo_conv.txt / yolo_conv2.txt | v5-u（含 DFL） | **失败**：`InferShapeByNNACL for op: /model.22/dfl/conv/Conv failed` |
| yolo_conv3.txt | （裁切后） | `CONVERT RESULT SUCCESS:0` |

所以 DFL 的 v5-u 根本过不了转换器，必须换原版 anchor-based v5 并裁掉解码段。

## 用法

    python _veh/convert_ms.py --onnx _veh/yolov5s_v7_320_npu.onnx --tag yolov5s_v7_320_npu
    python _veh/convert_ms.py --onnx ... --fp16
"""
import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

MS_ROOT = r"D:\Tools\mindspore-lite\mindspore-lite-2.6.0-win-x64"
CONV_DIR = os.path.join(MS_ROOT, "tools", "converter")
CONV_EXE = os.path.join(CONV_DIR, "converter", "converter_lite.exe")

_HERE = os.path.dirname(os.path.abspath(__file__))


def convert(onnx: str, out_stem: str, fp16: bool, input_shape: str, log_path: str) -> int:
    if not os.path.isfile(CONV_EXE):
        print(f"[ms] 找不到转换器：{CONV_EXE}", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        os.path.join(CONV_DIR, "lib"),
        os.path.join(CONV_DIR, "converter"),
        env.get("PATH", ""),
    ])

    cmd = [
        CONV_EXE,
        "--fmk=ONNX",
        f"--modelFile={onnx}",
        f"--outputFile={out_stem}",
    ]
    if input_shape:
        cmd.append(f"--inputShape={input_shape}")
    if fp16:
        cmd.append("--fp16=on")
    else:
        cmd.append("--fp16=off")

    print(f"[ms] cwd  = {CONV_DIR}")
    print(f"[ms] cmd  = {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=CONV_DIR, env=env, capture_output=True)
    out = p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace")
    # Node 侧落盘 UTF-8 再读，避免 PowerShell 管道把中文/UTF-8 弄坏
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)

    ok = "CONVERT RESULT SUCCESS" in out
    print(f"[ms] 输出末尾：")
    for line in out.strip().splitlines()[-6:]:
        print("      " + line)
    print(f"[ms] 日志 = {log_path}")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--input-shape", default="")
    ap.add_argument("--fp16", action="store_true", help="fp16 量化（默认 off = fp32）")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    onnx = os.path.abspath(a.onnx)
    if not os.path.isfile(onnx):
        print(f"[ms] 找不到 {onnx}", file=sys.stderr)
        return 2

    tag = a.tag or os.path.splitext(os.path.basename(onnx))[0]
    tag = tag + ("_fp16" if a.fp16 else "_fp32")
    out_dir = os.path.abspath(a.out_dir or os.path.join(_HERE, "mscheck"))
    os.makedirs(out_dir, exist_ok=True)
    out_stem = os.path.join(out_dir, tag)
    log_path = os.path.join(out_dir, tag + ".convert.log")

    # 图里输入是 images:1,3,320,320，显式给全，避免「图丢失 shape 信息」时报错
    input_shape = a.input_shape or "images:1,3,320,320"
    return convert(onnx, out_stem, a.fp16, input_shape, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
