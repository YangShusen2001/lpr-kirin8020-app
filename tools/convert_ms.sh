#!/usr/bin/env bash
# 从三个 ONNX 重新产出 .ms，并与 App 内置的 .ms 逐字节比对。
#
# 三个必须处理的坑（复刻 lpr-showcase/tools/convert_to_ms.py 的 job table）：
#   1) 固定 batch = 1 —— 动态 batch 维度会在 OH_AI_ModelBuildFromFile 阶段直接失败
#   2) --fp16=on —— 但注意 ADR-0003 记录：fp16 输出张量可能被声明为 FP32 而实际是
#      FP16 位流（误差 7.78e18%）。本项目三个模型已在图尾插入 Cast 规避。
#   3) 输入名与形状必须与 ONNX 图一致（y5fu=input / rpv3=data / cls=data）
set -u
# 路径从环境解析，不写死本机位置（与 tools/paths.py 同一套变量名）。
# 2026-09-21 改：原先硬编码本机路径，脱敏时被替换成占位符导致脚本失效。
MSLITE_DIR="${MSLITE_DIR:-D:/Tools/mindspore-lite}"
PRIOR_WORK="${LPR_PRIOR_WORK:-$HOME/Desktop/Test}"
APP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS="$MSLITE_DIR/mindspore-lite-2.6.0-win-x64/tools/converter/converter"
CONV="$MS/converter_lite.exe"
SRC="$PRIOR_WORK/lpr-harmony/models_ms"
OUT="$APP_REPO/models_ms"
mkdir -p "$OUT"

export PATH="$MS/lib:$MS:$PATH"

run() {
  local name="$1" inname="$2" shape="$3"
  echo "=== $name ($inname:$shape) ==="
  "$CONV" --fmk=ONNX \
          --modelFile="$SRC/$name.onnx" \
          --outputFile="$OUT/$name" \
          --fp16=on \
          --inputShape="${inname}:${shape}" 2>&1 | tail -4
  echo "[rc=${PIPESTATUS[0]}]"
  if [ -f "$OUT/$name.ms" ]; then
    printf '  -> %s.ms  %s bytes  magic=' "$name" "$(stat -c%s "$OUT/$name.ms")"
    head -c 4 "$OUT/$name.ms" | od -An -tx1
  else
    echo "  -> NOT PRODUCED"
  fi
  echo
}

run y5fu_320x_sim      input 1,3,320,320
run rpv3_mdict_160_r3  data  1,3,48,160
run litemodel_cls_96x_r1 data 1,3,96,96

echo "=== 与 App 内置 .ms 逐字节比对 ==="
APPMS="$APP_REPO/LprDemo/entry/src/main/resources/rawfile/models"
for n in y5fu_320x_sim rpv3_mdict_160_r3 litemodel_cls_96x_r1; do
  if [ -f "$OUT/$n.ms" ] && [ -f "$APPMS/$n.ms" ]; then
    a=$(sha256sum "$OUT/$n.ms" | cut -d' ' -f1)
    b=$(sha256sum "$APPMS/$n.ms" | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then echo "  $n: IDENTICAL"; else
      echo "  $n: DIFFERS"
      echo "     repro $a"
      echo "     app   $b"
    fi
  else
    echo "  $n: 缺文件"
  fi
done
