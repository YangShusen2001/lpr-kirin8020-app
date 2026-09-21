#!/usr/bin/env bash
# sign_hap.sh —— 用 .signing/ 的材料签署 entry 模块的 unsigned HAP
#
# 为什么不经 hvigor 的 SignHap：
#   hvigor 的 decipher-util.js 会**无条件**把 build-profile.json5 里的口令当 AES-GCM
#   密文解密（还要求 keystore 同目录有 material/{ac,ce,fd}）。明文口令走不通。
#   本脚本直接调 hap-sign-tool.jar sign-app，支持明文口令。
#
# 前置：先跑 bash tools/make_signing_material.sh
# 产物：entry/build/default/outputs/default/entry-default-signed.hap
#
# 可选环境变量：
#   LPR_HAP_OUT   签名后 HAP 的输出路径，默认就地输出 signed.hap
#                 （把桌面当交付目录时设它，例如 C:/Users/<you>/Desktop/lpr-demo-signed.hap）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 同 make_signing_material.sh：把 MSYS 路径转成 Windows 可读的混合风格
if command -v cygpath >/dev/null 2>&1; then
  REPO_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  REPO_ROOT="$(printf '%s' "$REPO_ROOT" | sed -E 's#^/([a-zA-Z])/#\U\1:/#')"
fi
SIGN_DIR="$REPO_ROOT/.signing"
OUT_DIR="$REPO_ROOT/LprDemo/entry/build/default/outputs/default"
UNSIGNED="$OUT_DIR/entry-default-unsigned.hap"

DEVECO_HOME="${DEVECO_HOME:-D:/IDE/DevEco_Studio}"
JAVA="$DEVECO_HOME/jbr/bin/java.exe"
SIGN_TOOL="$DEVECO_HOME/sdk/default/openharmony/toolchains/lib/hap-sign-tool.jar"

# ---- 前置检查 -------------------------------------------------------------
[ -f "$SIGN_DIR/pw.txt" ] || { echo "[sign_hap] 未找到 .signing/pw.txt，先跑 tools/make_signing_material.sh" >&2; exit 1; }
[ -f "$SIGN_DIR/lpr-work.p12" ] || { echo "[sign_hap] 未找到 .signing/lpr-work.p12" >&2; exit 1; }
[ -f "$UNSIGNED" ] || { echo "[sign_hap] 未找到 $UNSIGNED，先跑 bash build.sh" >&2; exit 1; }

PW="$(cat "$SIGN_DIR/pw.txt")"

# compatibleVersion 必须与 build-profile.json5 的 compatibleSdkVersion 一致（API 24）
COMPAT_VER="$(python -c "
import re
s=open(r'$REPO_ROOT/LprDemo/build-profile.json5',encoding='utf-8').read()
m=re.search(r'compatibleSdkVersion\"?\s*:\s*\"?[0-9.]+\((\d+)\)',s)
print(m.group(1) if m else 24)
")"

OUT_HAP="${LPR_HAP_OUT:-$OUT_DIR/entry-default-signed.hap}"
mkdir -p "$(dirname "$OUT_HAP")"
rm -f "$OUT_HAP"   # 避免签失败后留下上一次的旧产物（这是 build.sh 记录过的陷阱症状）

echo "[sign_hap] 输入   = $UNSIGNED"
echo "[sign_hap] 输出   = $OUT_HAP"
echo "[sign_hap] API    = $COMPAT_VER"

# ---- 签名 -----------------------------------------------------------------
"$JAVA" -jar "$SIGN_TOOL" sign-app -mode localSign \
  -keyAlias "lpr-app-key" -keyPwd "$PW" \
  -appCertFile "$SIGN_DIR/lpr-app.cer" \
  -profileFile "$SIGN_DIR/lpr-debug-profile.p7b" \
  -inFile  "$UNSIGNED" \
  -signAlg SHA256withECDSA \
  -keystoreFile "$SIGN_DIR/lpr-work.p12" -keystorePwd "$PW" \
  -outFile "$OUT_HAP" \
  -compatibleVersion "$COMPAT_VER" 2>&1 | grep -E "success|ERROR|Error" || true

[ -s "$OUT_HAP" ] || { echo "[sign_hap] 签名未产出文件。" >&2; exit 1; }

# ---- 验证（不通过就当失败） ------------------------------------------------
echo "[sign_hap] 校验签名…"
VERIFY_LOG="$("$JAVA" -jar "$SIGN_TOOL" verify-app -inFile "$OUT_HAP" \
  -outCertChain "$SIGN_DIR/verify-chain.cer" \
  -outProfile  "$SIGN_DIR/verify-profile.p7b" 2>&1 || true)"
if ! echo "$VERIFY_LOG" | grep -q "verify-app success"; then
  echo "$VERIFY_LOG" | tail -10 >&2
  echo "[sign_hap] 签名校验失败。" >&2
  exit 1
fi
echo "$VERIFY_LOG" | grep -E "Digest verify result|verify-app success"

# ---- 包名一致性（真机安装的硬性前提） -------------------------------------
BUNDLE_IN_PROFILE="$(python -c "
import re
s=open(r'$SIGN_DIR/verify-profile.p7b','rb').read()
m=re.search(rb'\"bundle-name\"\s*:\s*\"([^\"]+)\"',s)
print(m.group(1).decode() if m else '')
")"
BUNDLE_IN_APP="$(python -c "
import re
s=open(r'$REPO_ROOT/LprDemo/AppScope/app.json5',encoding='utf-8').read()
print(re.search(r'\"bundleName\"\s*:\s*\"([^\"]+)\"',s).group(1))
")"
echo "[sign_hap] profile 包名 = $BUNDLE_IN_PROFILE / AppScope 包名 = $BUNDLE_IN_APP"
if [ "$BUNDLE_IN_PROFILE" != "$BUNDLE_IN_APP" ]; then
  echo "[sign_hap] 包名不一致，真机会拒装。" >&2
  exit 1
fi

echo "[sign_hap] 完成：$OUT_HAP"
