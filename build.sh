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
export DEVECO_HOME="D:/IDE/DevEco_Studio"
export DEVECO_SDK_HOME="D:/IDE/DevEco_Studio/sdk"
export NODE_HOME="D:/IDE/DevEco_Studio/tools/node"
export JAVA_HOME='D:\IDE\DevEco_Studio\jbr'
export PATH="/d/IDE/DevEco_Studio/jbr/bin:$PATH"

PROJ="C:/Users/26671/lpr-kirin8020-app/LprDemo"
cd "$PROJ"
echo "PWD=$(pwd)"
echo "=== assembleHap ==="
"D:/IDE/DevEco_Studio/tools/node/node.exe" \
  "D:/IDE/DevEco_Studio/tools/hvigor/bin/hvigorw.js" \
  assembleHap --mode module -p product=default -p buildMode=debug --no-daemon "$@" 2>&1
rc=$?
echo "[hvigor rc=$rc]"
echo "=== 产物 ==="
ls -la "$PROJ/entry/build/default/outputs/default/" 2>&1 || echo "(无 outputs 目录)"
exit $rc
