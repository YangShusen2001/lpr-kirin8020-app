#!/usr/bin/env bash
# make_signing_material.sh —— 生成 CLI 可用的 OpenHarmony 签名材料（幂等，可重复执行）
#
# 为什么需要这个脚本
# ------------------
# DevEco Studio「自动签名」写进 build-profile.json5 的口令是 **DevEco 加密串**
# （形如 0000001A3866D7...，84 字符），它只能用 DevEco 自己的密钥链解开。在纯 CLI
# 环境里 hvigor 会尝试解密，然后报：
#     ERROR: 11014003 Init keystore failed
#     parseAlgParameters failed: ObjectIdentifier() -- data isn't an object ID (tag = 48)
# 且 hvigor 的 decryptPwd() 是**无条件调用**的，它还要求 keystore 同目录下存在
# material/{ac,ce,fd} 三个材料目录（见 hvigor-ohos-plugin/src/utils/decipher-util.js）：
#     ENOENT: no such file or directory, stat '<keystore 目录>\material'
# 也就是说 **明文口令在 hvigor 的 assembleHap 链路上根本无法使用**。
#
# 因此本脚本走另一条路：绕开 hvigor 的 SignHap，直接用 hap-sign-tool.jar 签 HAP。
# 它接受明文口令，不经过 DecipherUtil。
#
# 用法
# ----
#   bash tools/make_signing_material.sh          # 生成材料到 .signing/
#   bash tools/sign_hap.sh                       # 用材料签 entry 的 unsigned HAP
#
# 产物（全部落在 <repo>/.signing/）
# ---------------------------------
#   lpr-work.p12            工作密钥库（含 OpenHarmony CA 私钥 + 本应用私钥）
#   lpr-app.cer             三级应用证书链（leaf=CN=LPR Demo ← Application CA ← Root CA）
#   lpr-debug-profile.p7b   已签名的 debug provision profile，bundle-name 与 AppScope 一致
#   pw.txt                  密钥库口令（32 位十六进制；由本脚本生成，勿手改）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Bash 下 pwd 给的是 MSYS 风格 /c/...，而 Windows 版 python / java 只认 C:/...
# 统一转成混合风格（cygpath -m），缺失时用 sed 兜底。
if command -v cygpath >/dev/null 2>&1; then
  REPO_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  REPO_ROOT="$(printf '%s' "$REPO_ROOT" | sed -E 's#^/([a-zA-Z])/#\U\1:/#')"
fi
SIGN_DIR="$REPO_ROOT/.signing"
mkdir -p "$SIGN_DIR"

# ---- 定位 DevEco 工具链 ---------------------------------------------------
DEVECO_HOME="${DEVECO_HOME:-D:/IDE/DevEco_Studio}"
JAVA="$DEVECO_HOME/jbr/bin/java.exe"
KEYTOOL="$DEVECO_HOME/jbr/bin/keytool.exe"
TOOL_LIB="$DEVECO_HOME/sdk/default/openharmony/toolchains/lib"
SIGN_TOOL="$TOOL_LIB/hap-sign-tool.jar"
CA_STORE="$TOOL_LIB/OpenHarmony.p12"   # OpenHarmony 官方调试根材料，口令固定 123456
CA_PEM="$TOOL_LIB/OpenHarmonyProfileRelease.pem"

for f in "$JAVA" "$KEYTOOL" "$SIGN_TOOL" "$CA_STORE" "$CA_PEM"; do
  if [ ! -f "$f" ]; then
    echo "[signing] 缺少必要文件：$f" >&2
    echo "          检查 DEVECO_HOME 是否指向 DevEco Studio 安装根。" >&2
    exit 1
  fi
done

# ---- bundleName 必须与 profile 一致 --------------------------------------
BUNDLE_NAME="$(python -c "
import re
s=open(r'$REPO_ROOT/LprDemo/AppScope/app.json5',encoding='utf-8').read()
print(re.search(r'\"bundleName\"\s*:\s*\"([^\"]+)\"',s).group(1))
")"
echo "[signing] bundleName = $BUNDLE_NAME"

# ---- 目标真机 UDID（debug profile 的装机白名单） --------------------------
# 不写进 profile 就会在设备侧报 "fail to verify pkcs7 file"（见第 2 步注释）。
# 自动从已连接的 hdc 设备取；取不到就留空（只影响真机装机，不影响签名本身）。
if [ -z "${LPR_DEVICE_UDID:-}" ]; then
  HDC_BIN="$(command -v hdc || echo 'D:/Applications/CamStreamReceiver/connection/hdc.exe')"
  if [ -x "$HDC_BIN" ] || command -v hdc >/dev/null 2>&1; then
    LPR_DEVICE_UDID="$("$HDC_BIN" shell bm get -u 2>/dev/null \
      | sed -n 's/.*udid of current device is *: *//p' | tr -d '\r\n ' || true)"
    export LPR_DEVICE_UDID
    if [ -n "$LPR_DEVICE_UDID" ]; then
      echo "[signing] 真机 UDID = $LPR_DEVICE_UDID"
    else
      echo "[signing] 未取到 hdc 设备 UDID —— 若之后要在真机装，用 LPR_DEVICE_UDID=<64位hex> 重跑"
    fi
  fi
fi

# ---- 口令：hvigor 强制 ≥32 位且偶数长度；此处生成一次并复用 ----------------
PW_FILE="$SIGN_DIR/pw.txt"
if [ ! -f "$PW_FILE" ]; then
  python -c "
import secrets
# 纯小写十六进制，32 位（hvigor 会把口令当 hex 密文，长度/奇偶校验都基于此）
open(r'$PW_FILE','w',encoding='utf-8').write(secrets.token_hex(16))
"
fi
PW="$(cat "$PW_FILE")"
if [ "${#PW}" -ne 32 ]; then
  echo "[signing] pw.txt 里的口令长度是 ${#PW}，要求 32。删除它重跑本脚本。" >&2
  exit 1
fi

# ---- 1. 拆出 CA 链（root / subCA） ---------------------------------------
python - "$CA_PEM" "$SIGN_DIR" <<'PY'
import re, sys, os
src, out = sys.argv[1], sys.argv[2]
pem = open(src, encoding='utf-8').read()
certs = re.findall(r'-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----', pem, re.S)
# 顺序固定为 [Root CA, Application CA, Profile Release CA]
for i, c in enumerate(certs):
    open(os.path.join(out, f'ca-chain-{i}.cer'), 'w', encoding='utf-8').write(c + '\n')
print(f'[signing] CA 链拆出 {len(certs)} 张证书')
PY

# ---- 2. 生成未签名的 debug profile（替换 bundle-name） --------------------
#
# device-ids 必须包含目标真机的 UDID，否则设备侧报：
#     failed to install bundle. code:9568257 error: fail to verify pkcs7 file.
# 注意这个报错**名字会骗人** —— 它看着像证书链/签名格式问题（本地 verify-app
# 也确实报 true），实际是 debug profile 的 debug-info.device-ids 里没有这台设备。
# 签名工具本地校验只验摘要，不验设备授权，所以本地永远是成功。
#
# 取真机 UDID：  hdc shell bm get -u    →  输出 "udid of current device is : <64位hex>"
# 模板里预置的是云手机 UDID，真机装机必须补上当前这台。
# 可多个，用逗号分隔；用 LPR_EXTRA_UDIDS 追加。
python - "$TOOL_LIB" "$SIGN_DIR" "$BUNDLE_NAME" <<'PY'
import json, sys, os
tool_lib, out, bundle = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(os.path.join(tool_lib, 'UnsgnedDebugProfileTemplate.json'), encoding='utf-8'))
d['bundle-info']['bundle-name'] = bundle
d['bundle-info']['developer-id'] = 'OpenHarmony'
# 放宽有效期：模板里的是 2021–2024，早已过期
d['validity'] = {'not-before': 1757000000, 'not-after': 2062000000}
d['uuid'] = '9f2a7c31-4b8e-4d2a-b6f1-3c9d5e7a1b20'
d.pop('app-privilege-capabilities', None)

# 合并 UDID 白名单：模板自带的 + 环境变量追加的 + 现场从 hdc 抓的
ids = list(d.get('debug-info', {}).get('device-ids', []))
if ids and ids[0] == 'all':
    ids = []
extra = [u.strip() for u in os.environ.get('LPR_EXTRA_UDIDS', '').split(',') if u.strip()]
device = os.environ.get('LPR_DEVICE_UDID', '').strip()
for u in extra + ([device] if device else []):
    if u not in ids:
        ids.append(u)
        print(f'[signing] + UDID {u}')
d.setdefault('debug-info', {})
d['debug-info']['device-ids'] = ids
d['debug-info']['device-id-type'] = 'udid'

json.dump(d, open(os.path.join(out, 'lpr-debug-profile.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=4)
print(f'[signing] profile 模板已改写 bundle-name={bundle}，UDID 白名单 n={len(ids)}')
PY

# ---- 3. 签名 profile ------------------------------------------------------
"$JAVA" -jar "$SIGN_TOOL" sign-profile -mode localSign \
  -keyAlias "openharmony application profile release" -keyPwd 123456 \
  -profileCertFile "$CA_PEM" \
  -inFile "$SIGN_DIR/lpr-debug-profile.json" \
  -signAlg SHA256withECDSA \
  -keystoreFile "$CA_STORE" -keystorePwd 123456 \
  -outFile "$SIGN_DIR/lpr-debug-profile.p7b" 2>&1 | grep -E "success|ERROR" || true

# ---- 4. 工作密钥库：CA 材料 + 本应用密钥对（同一库，供 generate-app-cert 用）--
cp -f "$CA_STORE" "$SIGN_DIR/lpr-work.p12"
"$KEYTOOL" -storepasswd -keystore "$SIGN_DIR/lpr-work.p12" -storetype PKCS12 \
  -storepass 123456 -new "$PW" 2>&1 | grep -vE "^\s*$" || true

# 生成应用密钥对（先落在临时库，再并入工作库）
if [ ! -f "$SIGN_DIR/lpr-app.p12" ]; then
  "$JAVA" -jar "$SIGN_TOOL" generate-keypair -keyAlias "lpr-app-key" -keyPwd "$PW" \
    -keyAlg ECC -keySize NIST-P-256 \
    -keystoreFile "$SIGN_DIR/lpr-app.p12" -keystorePwd "$PW" 2>&1 | grep -E "success|ERROR" || true
fi
"$KEYTOOL" -importkeystore \
  -srckeystore "$SIGN_DIR/lpr-app.p12" -srcstoretype PKCS12 -srcstorepass "$PW" \
  -destkeystore "$SIGN_DIR/lpr-work.p12" -deststoretype PKCS12 -deststorepass "$PW" \
  -srcalias "lpr-app-key" -destalias "lpr-app-key" -destkeypass "$PW" -noprompt 2>&1 | grep -vE "^\s*$" || true

# ---- 5. 签发应用证书链 ----------------------------------------------------
"$JAVA" -jar "$SIGN_TOOL" generate-app-cert \
  -keyAlias "lpr-app-key" -keyPwd "$PW" \
  -issuer "C=CN,O=OpenHarmony,OU=OpenHarmony Team,CN=OpenHarmony Application CA" \
  -issuerKeyAlias "openharmony application ca" -issuerKeyPwd "$PW" \
  -subject "C=CN,O=OpenHarmony,OU=OpenHarmony Team,CN=LPR Demo" \
  -signAlg SHA256withECDSA \
  -keystoreFile "$SIGN_DIR/lpr-work.p12" -keystorePwd "$PW" \
  -outForm certChain \
  -rootCaCertFile "$SIGN_DIR/ca-chain-0.cer" \
  -subCaCertFile  "$SIGN_DIR/ca-chain-1.cer" \
  -outFile "$SIGN_DIR/lpr-app.cer" 2>&1 | grep -E "success|ERROR" || true

# ---- 6. 自检 --------------------------------------------------------------
FAIL=0
for f in lpr-work.p12 lpr-app.cer lpr-debug-profile.p7b; do
  if [ -s "$SIGN_DIR/$f" ]; then
    echo "[signing] ✓ $f ($(stat -c%s "$SIGN_DIR/$f") 字节)"
  else
    echo "[signing] ✗ $f 缺失或为空" >&2
    FAIL=1
  fi
done
CERTS="$(grep -c 'BEGIN CERTIFICATE' "$SIGN_DIR/lpr-app.cer" 2>/dev/null || echo 0)"
echo "[signing] 应用证书链张数 = $CERTS （期望 3）"
[ "$CERTS" -eq 3 ] || FAIL=1

if [ "$FAIL" -ne 0 ]; then
  echo "[signing] 材料生成未通过自检。" >&2
  exit 1
fi
echo "[signing] 完成。接下来跑：bash tools/sign_hap.sh"
