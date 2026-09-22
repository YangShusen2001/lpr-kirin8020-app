#!/usr/bin/env bash
# sign_hap_huawei.sh —— 用**华为**调试材料签 HAP，供华为商用真机安装。
#
# 为什么需要它（这是本项目装机链路上最关键的一条结论）
# ====================================================
# OpenHarmony 官方调试材料（sdk/.../toolchains/lib/OpenHarmony.p12）的信任根是
#     CN=OpenHarmony Application Root CA
# 而华为商用机（例：MIA-AL00 / HarmonyOS 6.1.0.135）信任的是
#     CN=Huawei CBG Root CA G2
# 两者是**互不相认的两个信任域**。用 OH 材料签出来的包，本地 verify-app 会报成功
# （它只验摘要），但设备侧必然拒装，且报错**极具误导性**：
#
#     failed to install bundle. code:9568257 error: fail to verify pkcs7 file.
#
# 设备 hilog 里的真实原因（C011FE/foundation/HapVerify）：
#     GetCertsChain:322  it do not come from trusted root,
#                        issuer: C=CN, O=OpenHarmony, ... CN=OpenHarmony Application Root CA
#     VerifyCertChain:123  get cert chain for signInfo failed
#
# 注意这个报错的措辞会让人往「证书过期 / 签名算法 / 包名不匹配 / UDID 白名单」上查，
# 全都不是。**唯一判据是信任根对不对**。诊断命令：
#     hdc shell hilog -x | grep -a HapVerify
#
# 正确做法：用 DevEco Studio「自动签名」为**本项目**生成的那套材料
# （落在 ~/.ohos/config/，信任根是 Huawei CBG Root CA G2），
# 其 profile 的 device-ids 也已包含目标真机 UDID。
#
# 前置
# ----
#   ~/.ohos/config/default_<项目名>_<hash>=.{p12,cer,p7b}   ← DevEco 自动签名产出
#   ~/.ohos/config/material/{ac,ce,fd}/                     ← 口令解密材料
#   python tools/deveco_pwd.py --verify                     ← 应先能解出口令并通过校验
#
# 用法
# ----
#   bash tools/sign_hap_huawei.sh
#   LPR_HAP_OUT=/path/out.hap bash tools/sign_hap_huawei.sh
#
# 与 sign_hap.sh 的关系
# --------------------
#   sign_hap.sh          → OH 官方材料，仅用于**云手机 / OH 开发板**（信任 OH 根）
#   sign_hap_huawei.sh   → 华为材料，用于**华为商用真机**（信任华为根）
#   两者产出同名 signed.hap，按目标设备选一个跑即可。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v cygpath >/dev/null 2>&1; then
  REPO_ROOT="$(cygpath -m "$REPO_ROOT")"
else
  REPO_ROOT="$(printf '%s' "$REPO_ROOT" | sed -E 's#^/([a-zA-Z])/#\U\1:/#')"
fi

DEVECO_HOME="${DEVECO_HOME:-D:/IDE/DevEco_Studio}"
JAVA="$DEVECO_HOME/jbr/bin/java.exe"
SIGN_TOOL="$DEVECO_HOME/sdk/default/openharmony/toolchains/lib/hap-sign-tool.jar"

OHOS_DIR="${OHOS_DIR:-$HOME/.ohos/config}"
# Bash 的 $HOME 是 MSYS 风格（/c/Users/...），而 Windows 版 python 只认 C:/Users/...
if command -v cygpath >/dev/null 2>&1; then
  OHOS_DIR="$(cygpath -m "$OHOS_DIR")"
else
  OHOS_DIR="$(printf '%s' "$OHOS_DIR" | sed -E 's#^/([a-zA-Z])/#\U\1:/#')"
fi
OUT_DIR="$REPO_ROOT/LprDemo/entry/build/default/outputs/default"
UNSIGNED="$OUT_DIR/entry-default-unsigned.hap"

PY="${LPR_PYTHON:-python}"

# ---- 前置检查 -------------------------------------------------------------
[ -f "$JAVA" ] || { echo "[sign_huawei] 找不到 $JAVA（用 DEVECO_HOME 指定）" >&2; exit 1; }
[ -f "$SIGN_TOOL" ] || { echo "[sign_huawei] 找不到 hap-sign-tool.jar" >&2; exit 1; }
[ -f "$UNSIGNED" ] || { echo "[sign_huawei] 未找到 $UNSIGNED，先跑 bash build.sh" >&2; exit 1; }
[ -d "$OHOS_DIR/material" ] || { echo "[sign_huawei] 找不到 $OHOS_DIR/material" >&2; exit 1; }

# ---- 定位本项目对应的那套材料（按 profile 里的 bundle-name 选，不靠猜文件名）----
BUNDLE_NAME="$("$PY" -c "
import re
s=open(r'$REPO_ROOT/LprDemo/AppScope/app.json5',encoding='utf-8').read()
print(re.search(r'\"bundleName\"\s*:\s*\"([^\"]+)\"',s).group(1))
")"
echo "[sign_huawei] bundleName = $BUNDLE_NAME"

MATERIAL="$("$PY" - "$OHOS_DIR" "$BUNDLE_NAME" <<'PY'
import glob, os, re, sys
ohos_dir, bundle = sys.argv[1], sys.argv[2]
hits = []
for p7b in glob.glob(os.path.join(ohos_dir, "*.p7b")):
    raw = open(p7b, "rb").read()
    m = re.search(rb'"bundle-name"\s*:\s*"([^"]+)"', raw)
    if m and m.group(1).decode() == bundle:
        base = p7b[:-4]
        if os.path.isfile(base + ".p12") and os.path.isfile(base + ".cer"):
            hits.append(base)
hits.sort()
print(hits[0] if hits else "")
PY
)"

if [ -z "$MATERIAL" ]; then
  echo "[sign_huawei] $OHOS_DIR 下没有 bundle-name=$BUNDLE_NAME 的材料。" >&2
  echo "              用 DevEco Studio 打开本工程 → File > Project Structure >" >&2
  echo "              Signing Configs → 勾选 Automatically generate signature。" >&2
  exit 1
fi
echo "[sign_huawei] 材料     = $(basename "$MATERIAL")"

# ---- 解出口令（DevEco 密文 → 明文，算法用 DevEco 原版，见 tools/deveco_pwd.py）----
STORE_PW="$("$PY" "$REPO_ROOT/tools/deveco_pwd.py" --material-root "$OHOS_DIR" --print-store)"
KEY_PW="$("$PY" "$REPO_ROOT/tools/deveco_pwd.py" --material-root "$OHOS_DIR" --print-key)"
[ -n "$STORE_PW" ] || { echo "[sign_huawei] 没解出 storePassword" >&2; exit 1; }
[ -n "$KEY_PW" ] || KEY_PW="$STORE_PW"

# ---- 从 p12 里读真实 keyAlias（DevEco 写的是 debugKey，大小写别猜）--------
KEY_ALIAS="$("$PY" - "$JAVA" "$MATERIAL.p12" "$STORE_PW" <<'PY'
import re, subprocess, sys
java, p12, pw = sys.argv[1], sys.argv[2], sys.argv[3]
r = subprocess.run([java, "-list", "-keystore", p12, "-storetype", "PKCS12",
                    "-storepass", pw], capture_output=True, text=True, errors="replace")
m = re.search(r"Alias name:\s*(\S+)", r.stdout or "")
if not m:
    m = re.search(r"别名[:：]\s*(\S+)", r.stdout or "")
print(m.group(1) if m else "debugKey")
PY
)"
echo "[sign_huawei] keyAlias = $KEY_ALIAS"

# ---- compatibleVersion 必须与 build-profile.json5 的 compatibleSdkVersion 一致 ----
COMPAT_VER="$("$PY" -c "
import re
s=open(r'$REPO_ROOT/LprDemo/build-profile.json5',encoding='utf-8').read()
m=re.search(r'compatibleSdkVersion\"?\s*:\s*\"?[0-9.]+\((\d+)\)',s)
print(m.group(1) if m else 24)
")"

OUT_HAP="${LPR_HAP_OUT:-$OUT_DIR/entry-default-signed.hap}"
mkdir -p "$(dirname "$OUT_HAP")"
rm -f "$OUT_HAP"

echo "[sign_huawei] 输入   = $UNSIGNED"
echo "[sign_huawei] 输出   = $OUT_HAP"
echo "[sign_huawei] API    = $COMPAT_VER"

# ---- 签名 -----------------------------------------------------------------
"$JAVA" -jar "$SIGN_TOOL" sign-app -mode localSign \
  -keyAlias "$KEY_ALIAS" -keyPwd "$KEY_PW" \
  -appCertFile "$MATERIAL.cer" \
  -profileFile "$MATERIAL.p7b" \
  -inFile  "$UNSIGNED" \
  -signAlg SHA256withECDSA \
  -keystoreFile "$MATERIAL.p12" -keystorePwd "$STORE_PW" \
  -outFile "$OUT_HAP" \
  -compatibleVersion "$COMPAT_VER" 2>&1 | grep -E "success|ERROR|Error" || true

[ -s "$OUT_HAP" ] || { echo "[sign_huawei] 签名未产出文件。" >&2; exit 1; }

# ---- 本地校验（只验摘要，**不能**证明设备会接受 —— 信任根是设备侧才查的）----
echo "[sign_huawei] 本地校验签名（注意：这不代表真机会接受）…"
VERIFY_LOG="$("$JAVA" -jar "$SIGN_TOOL" verify-app -inFile "$OUT_HAP" \
  -outCertChain "$OHOS_DIR/verify-chain.cer" \
  -outProfile  "$OHOS_DIR/verify-profile.p7b" 2>&1 || true)"
echo "$VERIFY_LOG" | grep -E "Digest verify result|verify-app success" || true

# ---- 打印信任根，这是装机成败的真正判据 ----
#
# ⚠️ 2026-09-22 修（这个坑很隐蔽，记全）
#
# 原写法是 `ROOT_SUBJ="$("$PY" ... 2>/dev/null || true)"`，把失败**吞成空串**，
# 于是脚本打「（未解析出）」—— 读数上像「解析了但没得出结论」，
# 实际是**解析器根本没跑起来**。实测本机根因：
#   `PY` 默认是裸 `python` → PATH 上是 3.13.12，**没装 cryptography**
#   （cert_root_cn.py 依赖它做 X.509 解析）→ ModuleNotFoundError 被 2>/dev/null 吞掉。
# 而这条检查正是 ADR-0009 那个「装机最高频踩坑（信任根）」的**唯一护栏**，
# 静默降级等于把护栏拆了还挂个绿灯。所以改成：
#   ① 主动找一个装了 cryptography 的 python；
#   ② 找不到 / 解析失败 → **显式报错**，并说清「这不是没结论，是没跑」。
_find_crypto_py() {
  local c
  # 注意 `${LPR_PYTHON:-}`：脚本开头是 `set -u`，而 LPR_PYTHON 只在给 PY 赋值时
  # 以 `:-` 默认形式出现过、**从未被赋值或导出**，直接引用会让函数在 set -u 下中止
  # （症状：明明有可用的 python 却报「找不到」）。
  for c in "${LPR_PYTHON:-}" python3 python; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1 \
       && "$c" -c "import cryptography" >/dev/null 2>&1; then
      printf '%s' "$c"; return 0
    fi
  done
  # Windows 上 per-user / 全机安装的常见位置
  for c in "$HOME"/AppData/Local/Programs/Python/*/python.exe \
           "$LOCALAPPDATA"/Programs/Python/*/python.exe \
           /c/Python*/python.exe; do
    [ -f "$c" ] || continue
    if "$c" -c "import cryptography" >/dev/null 2>&1; then printf '%s' "$c"; return 0; fi
  done
  return 1
}

ROOT_OUT=""; ROOT_RC=0
if PY_CRYPTO="$(_find_crypto_py)"; then
  ROOT_OUT="$("$PY_CRYPTO" "$REPO_ROOT/tools/cert_root_cn.py" "$MATERIAL.cer" 2>&1)" || ROOT_RC=$?
else
  ROOT_RC=127
  ROOT_OUT="找不到装了 cryptography 的 python（cert_root_cn.py 依赖它）；可用 LPR_PYTHON 指定"
fi

# cert_root_cn.py 的退出码约定：0=Huawei 根、2=OpenHarmony 根（**2 也是有效识别**）、3=无法判定。
# 注意别把 2 误当失败 —— 认出 OpenHarmony 根恰恰是最该报出来的结论。
if [ -n "$ROOT_OUT" ] && { [ "$ROOT_RC" -eq 0 ] || [ "$ROOT_RC" -eq 2 ]; }; then
  echo "[sign_huawei] 信任根 CN = $ROOT_OUT"
  case "$ROOT_OUT" in
    *Huawei*)      echo "[sign_huawei] ✓ 华为根 —— 华为商用真机可装" ;;
    *OpenHarmony*) echo "[sign_huawei] ✗ OpenHarmony 根 —— 华为真机必拒装（9568257）" >&2 ;;
    *)             echo "[sign_huawei] ? 认不出这个根，装机失败时先查 hdc hilog 里的 HapVerify" ;;
  esac
else
  echo "[sign_huawei] ⚠️ 信任根检查【未执行】—— 这不是「解析了但没结论」，是护栏没跑。" >&2
  echo "[sign_huawei]    原因：$ROOT_OUT" >&2
  echo "[sign_huawei]    装机失败时优先修这里，再查 hdc hilog 里的 HapVerify。" >&2
fi

echo "[sign_huawei] 完成：$OUT_HAP"
echo "[sign_huawei] 装机：hdc install -r \"$OUT_HAP\""
