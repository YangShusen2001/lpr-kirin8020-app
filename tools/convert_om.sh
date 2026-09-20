#!/usr/bin/env bash
# CANN 侧：ONNX -> .om
#
# 【必须在 WSL Ubuntu 里跑，不能在 Git Bash / Windows 下跑】
# omg 是 Linux ELF64 二进制，Git Bash 下报 "Exec format error"。
#   用法: wsl -d Ubuntu -- bash /mnt/c/Users/26671/lpr-kirin8020-app/tools/convert_om.sh
#
# 路径统一用 /mnt/c 形式（WSL 认识；Git Bash 不适用）。
#
# 四个实测踩到的陷阱，缺一不可：
#
#   1) 入口必须是【包装脚本】tools_omg/omg，不是底层 master/omg 二进制。
#      包装脚本自己设 LD_LIBRARY_PATH / CCE_LIB_DIR / loader 软链；
#      直接跑二进制会报 libomg.so not found 或 unknown flag。
#
#   2) 输出路径不能含非 ASCII 字符。OMG 会拒：
#        failed validation of new value ... for flag 'output'
#        CheckOutputPathValid ... false
#      与 hvigor 拒绝非 ASCII 工程路径（00306003）是同一类缺陷。
#
#   3) 不要传 --target=omc。那会产出 .omc 中间产物而非 .om，
#      而且 rc=0 静默成功（实测产出了 3 个 .omc 却无 .om）。
#
#   4) 版本参数是 --hiai_version=<master|IR|Vxxx>，不是 --omg_version。
#      传后者会 "unknown command line flag" 然后静默不出模型。
#
# 另：以下加载失败是【噪声】，不影响产出，勿误判为故障：
#   - te_fusion / librl_search.so / libai_npucore_generated.so
#   - "kernel binary initialize failed, this store can use JIT only"
# 09-18 的成功日志里同样有这些行。
set -u

DDK="${DDK:-/mnt/c/Users/26671/lpr-harmony/omg_conv/ddk}"
SRC="${SRC:-/mnt/c/Users/26671/lpr-harmony/omg_conv}"
OUT="${1:-/mnt/c/Users/26671/lpr-kirin8020-app/models_om}"
OMG="$DDK/tools/tools_omg/omg"          # <-- 包装脚本，不是 master/omg

[[ -x "$OMG" ]] || { echo "FAIL: 找不到包装脚本 $OMG" >&2; exit 3; }
mkdir -p "$OUT"

convert() {
  local name="$1" inname="$2" shape="$3"
  echo "=== $name ($inname:$shape) ==="
  "$OMG" --model "$SRC/$name.onnx" --framework 5 \
         --output "$OUT/om_$name" \
         --input_shape "${inname}:${shape}" \
         --platform=kirin9020 > "$OUT/log_om_$name.txt" 2>&1
  echo "rc=$?"
  if [ -f "$OUT/om_$name.om" ]; then
    printf '  -> om_%s.om  %s bytes  magic=' "$name" "$(stat -c%s "$OUT/om_$name.om")"
    head -c 4 "$OUT/om_$name.om" | od -An -tx1
  else
    echo "  -> NOT PRODUCED"
    tail -5 "$OUT/log_om_$name.txt"
  fi
  echo
}

convert cls      data  1,3,96,96
convert dethead  input 1,3,320,320
convert rec      data  1,3,48,160

echo "=== 与 2026-09-18 原件逐字节比对 ==="
for pair in "cls:om_cls" "dethead:om_dethead" "rec:om_rec"; do
  n="${pair%%:*}"; o="${pair##*:}"
  orig="$SRC/out/$o.om"
  if [ -f "$OUT/om_$n.om" ] && [ -f "$orig" ]; then
    a=$(stat -c%s "$OUT/om_$n.om"); b=$(stat -c%s "$orig")
    if [ "$a" = "$b" ]; then
      echo "  $n: 字节数一致 ($a)"
    else
      echo "  $n: 不一致  repro=$a orig=$b"
    fi
  else
    echo "  $n: 缺文件（repro=$([ -f "$OUT/om_$n.om" ] && echo y || echo n) orig=$([ -f "$orig" ] && echo y || echo n)）"
  fi
done
