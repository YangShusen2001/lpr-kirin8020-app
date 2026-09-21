#!/usr/bin/env bash
# lpr-kirin8020-app 构建包装（复刻 lpr-harmony/build.sh 的环境变量与两个陷阱规避）
#
# 三个必须的环境变量：
#   DEVECO_HOME        DevEco Studio 安装根
#   DEVECO_SDK_HOME    必须指向 sdk 根（hvigor 的本地 SDK 扫描器只遍历子目录找
#                      <child>/sdk-pkg.json，从不检查根目录自身；本机布局是
#                      sdk/default/sdk-pkg.json，所以必须指到 sdk）
#   NODE_HOME          DevEco 自带 node
# 两个必须规避的陷阱：
#   1) unset NODE_OPTIONS —— 宿主会注入 node-language-shim.cjs（内含 safe-delete
#      保护），它会拦截 hvigor 退出时清理 .hvigor/report/*.json 的 unlinkSync，
#      累计删除数触阈值就抛 SAFE_DELETE_BULK_CONFIRM_REQUIRED，导致构建在
#      「打包已完成、签名收尾前」崩掉（症状：unsigned.hap 是新的、signed.hap 是旧的）
#   2) JAVA_HOME 必须用 DevEco 自带的 jbr(JDK21) —— 系统 PATH 上是 JDK 1.8，
#      读不了 JDK21 生成的 PKCS12 密钥库，会报 11014003 Init keystore failed
set -e
unset NODE_OPTIONS

# ---- 路径从环境解析（不要写死本机路径）----
#
# 2026-09-21 改：原先硬编码 `D:/IDE/DevEco_Studio` 等本机路径。为开源做脱敏时
# 那些路径被替换成占位符，**脚本直接失效**。现在改为「环境变量优先，否则用
# DevEco 的默认安装位置」，且工程目录由脚本自身位置推导 —— 换机器只需设环境变量。
# 与 `tools/paths.py` 使用同一套变量名。
DEVECO_HOME="${DEVECO_HOME:-D:/IDE/DevEco_Studio}"
export DEVECO_HOME
export DEVECO_SDK_HOME="${DEVECO_SDK_HOME:-$DEVECO_HOME/sdk}"
export LPR_NODE_HOME="${LPR_NODE_HOME:-$DEVECO_HOME/tools/node}"
export LPR_JAVA_HOME="${LPR_JAVA_HOME:-$DEVECO_HOME/jbr}"
export JAVA_HOME="$LPR_JAVA_HOME"
export PATH="$LPR_JAVA_HOME/bin:$PATH"

# 工程目录 = 本脚本所在目录 + /LprDemo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$SCRIPT_DIR/LprDemo"

# 构建模式默认 **release**。
#
# 2026-09-21 改：本脚本原先硬编码 `buildMode=debug`，而 hvigor 的 debug 构建会让
# CMake 用 `CMAKE_BUILD_TYPE=Debug`，把自己的 `-O0 -g` 追加在 `build-profile.json5`
# 的 `-O3` 之后 —— 编译器取最后一个，`-O3` 被静默废掉，**无任何报错**。
# 真机实测差 1.7–4×（conv 25 ms → 1.42 ms；相机 7 fps → 20 fps）。
#
# 所有性能/热特性数据都必须在 release 下采集，所以默认值必须是 release，
# 而不是「记得手动加 -p buildMode=release」—— 那正是 T7 首版数据被 -O0 污染的成因。
# 需要 debug 时：BUILD_MODE=debug bash build.sh assembleHap ...
BUILD_MODE="${BUILD_MODE:-release}"

cd "$PROJ"
echo "PWD=$(pwd)"
echo "BUILD_MODE=$BUILD_MODE"
echo "DEVECO_HOME=$DEVECO_HOME"
echo "=== assembleHap ==="
"$LPR_NODE_HOME/node.exe" \
  "$DEVECO_HOME/tools/hvigor/bin/hvigorw.js" \
  assembleHap --mode module -p product=default -p buildMode="$BUILD_MODE" --no-daemon "$@" 2>&1
rc=$?
echo "[hvigor rc=$rc]"
echo "=== 产物 ==="
ls -la "$PROJ/entry/build/default/outputs/default/" 2>&1 || echo "(无 outputs 目录)"
exit $rc
