#!/usr/bin/env bash
# T8: 把 ShusenPaper 的算子探针模型从 ONNX 转成 .om（CANN 侧）
#
# 目的：与 .ms（NNRT 侧）做**逐算子对撞**，回答「NPU 支持性是 模型 × 工具链 的联合属性」。
#
# 必须在 WSL 里跑（omg 是 Linux ELF64；Git Bash 下报 Exec format error）。
#   用法: wsl -d Ubuntu -- bash /mnt/c/<app-repo>/tools/convert_om_ops.sh
#
# 四个 OMG 陷阱见 convert_om.sh 的注释（包装脚本入口 / 非 ASCII 输出路径 /
# 不要 --target=omc / --hiai_version 而非 --omg_version）。
set -u

# ---- 路径从环境解析（见 convert_om.sh 的说明：WSL 的 $HOME 不是 Windows 家目录）----
if [ -z "${LPR_WIN_HOME:-}" ]; then
  _up="$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r\n')"
  if [ -n "$_up" ]; then
    LPR_WIN_HOME="$(wslpath "$_up" 2>/dev/null || true)"
  fi
fi
: "${LPR_WIN_HOME:?无法确定 Windows 用户主目录，请设置 LPR_WIN_HOME（如 /mnt/c/Users/<你的用户名>）}"

DDK="$LPR_WIN_HOME/Desktop/Test/lpr-harmony/omg_conv/ddk"
OMG="$DDK/tools/tools_omg/omg"
SRC="$LPR_WIN_HOME/Desktop/Test/ShusenPaper/models/onnx/opbench"
OUT="$LPR_WIN_HOME/lpr-kirin8020-app/models_om_ops"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

[[ -x "$OMG" ]] || { echo "FAIL: 找不到 $OMG"; exit 3; }
[[ -d "$SRC" ]] || { echo "FAIL: 找不到 $SRC"; exit 3; }

ok=0; fail=0
: > "$OUT/convert_results.tsv"
printf 'name\tstatus\tsize\tlog\n' >> "$OUT/convert_results.tsv"

for f in "$SRC"/*.onnx; do
  name="$(basename "$f" .onnx)"
  log="$LOGS/$name.txt"
  "$OMG" --model "$f" --framework 5 \
         --output "$OUT/$name" \
         --platform=kirin9020 > "$log" 2>&1
  rc=$?
  if [ -f "$OUT/$name.om" ]; then
    sz=$(stat -c%s "$OUT/$name.om")
    printf '%s\tOK\t%s\t%s\n' "$name" "$sz" "$(basename "$log")" >> "$OUT/convert_results.tsv"
    ok=$((ok+1))
  else
    # 记录失败原因的首条 error 行，便于分类（算子不支持 vs 版本问题）
    reason=$(grep -m1 -iE 'error|not support|unsupported|fail' "$log" 2>/dev/null | tr '\t' ' ' | cut -c1-200)
    [ -z "$reason" ] && reason="rc=$rc"
    printf '%s\tFAIL\t0\t%s\n' "$name" "$reason" >> "$OUT/convert_results.tsv"
    fail=$((fail+1))
  fi
done

echo "OK=$ok FAIL=$fail"
echo "-> $OUT/convert_results.tsv"
