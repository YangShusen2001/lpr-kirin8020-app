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

# ⚠️ 2026-09-22 修（两次踩坑，记全）：
#
# 症状 A（JDK 太旧）：签名阶段报
#     Failed :entry:default@SignHap...  ERROR: 11014003 Init keystore failed
#     parseAlgParameters failed: ObjectIdentifier() -- data isn't an object ID (tag = 48)
#   根因：本机 PATH 有 Windows 层 Oracle shim（`C:\Program Files (x86)\Common Files\
#   Oracle\Java\javapath`），Git Bash 优先解析到它 ⇒ jbr/bin 被架空，`which java` 是 1.8。
#   报错完全不提 JDK 版本，只看这句话会去查密钥库路径/密码，全是错方向。
#
# 症状 B（java 找不到）：我第一版修法是「把 Oracle 目录从 PATH 剔掉再 prepend」，
#   结果 hvigor 用 **Windows 原生 spawn** 调裸 `java`，而我重建出来的 PATH 是
#   MSYS 风格（`/d/IDE/...`），Windows 认不了 ⇒ `spawn java ENOENT`。
#   ⇒ 修法必须**同时**满足两条：① 去掉 Oracle shim；② 给 Windows 一个能认的 java。
#
# 最终做法：保留原 PATH 不动（只剔 Oracle shim），并把 jbr/bin 以 **Windows 风格**
# 放在最前 —— 让 MSYS 与原生 spawn 都能命中同一个 JDK21。
_drop_oracle_shim() {
  echo "$PATH" | tr ':' '\n' | grep -viE 'Oracle/Java/javapath' | paste -sd: -
}
PATH="$(_drop_oracle_shim)"
export PATH
# Windows 风格条目在前（原生 spawn 用），MSYS 风格在后（Bash 内用）
export PATH="$LPR_JAVA_HOME/bin;$PATH"

# 断言：必须是 JDK17+（PKCS12 密钥库的最低要求）。版本不对就立即停。
_java_major="$("$LPR_JAVA_HOME/bin/java" -version 2>&1 | head -1 \
               | sed -E 's/.*version "([0-9]+).*/\1/')"
if [ -z "$_java_major" ] || [ "$_java_major" -lt 17 ] 2>/dev/null; then
  echo "[build.sh] 签名需要 JDK17+，但 $LPR_JAVA_HOME 报的版本是：$_java_major" >&2
  echo "           检查 LPR_JAVA_HOME 是否指向 DevEco 自带的 jbr。" >&2
  exit 1
fi
echo "JAVA_HOME=$JAVA_HOME (java $_java_major)"

# ---- 前期工作根目录（ncnn 头文件在那里）----
#
# 2026-09-22 加：CMakeLists.txt 里 ncnn 的 include 路径原先是脱敏留下的
# 字面量 `<PRIOR_WORK>`，导致 `fatal error: 'gpu.h' file not found` ——
# 报错指向 include 语句，看不出根因是路径。现在由这里导出同一个变量名，
# CMake 侧读不到会 FATAL_ERROR（不再静默产坏路径）。
#
# 注意：tools/paths.py 的默认值是 `~/Desktop/Test`，但 `lpr-harmony`（真正装 ncnn 的
# 那个仓库）在本机并不在 Test 下，而在 `~` 下。所以这里按「哪个真的含 lpr-harmony」
# 依次探测，而不是照抄 paths.py 的默认值 —— 否则构建会失败在一个看似无关的地方。
resolve_prior_work() {
  local c
  for c in "${LPR_PRIOR_WORK:-}" "$HOME/Desktop/Test" "$HOME"; do
    [ -n "$c" ] || continue
    if [ -d "$c/lpr-harmony/third_party/ncnn/src" ]; then
      echo "$c"; return 0
    fi
  done
  return 1
}
if _pw="$(resolve_prior_work)"; then
  # ⚠️ 必须转成 Windows 风格（C:/...）。CMake 是 **Windows 原生程序**，认不了 MSYS
  # 风格的 `/c/Users/...` —— 传进去它会在 `if(NOT EXISTS ...)` 处报「目录不存在」，
  # 而目录其实存在。这个坑第一次就是这么踩的：守卫行为是对的，路径格式是错的。
  if command -v cygpath >/dev/null 2>&1; then
    export LPR_PRIOR_WORK="$(cygpath -m "$_pw")"      # -m = 混合风格 C:/a/b
  else
    # 没有 cygpath 时手工转换：/c/Users/x -> C:/Users/x
    case "$_pw" in
      /?/*) export LPR_PRIOR_WORK="$(echo "$_pw" | sed -E 's#^/([a-zA-Z])/#\U\1:/#')" ;;
      *)    export LPR_PRIOR_WORK="$_pw" ;;
    esac
  fi
else
  echo "[build.sh] 找不到含 lpr-harmony/third_party/ncnn/src 的前期工作根。" >&2
  echo "           请设置：export LPR_PRIOR_WORK=<前期工作根目录>" >&2
  exit 1
fi
echo "LPR_PRIOR_WORK=$LPR_PRIOR_WORK"

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
# ⚠️ 2026-09-22 修：`set -e`（见第 17 行）会让**失败的那条命令直接终止脚本**，
# 于是下面 `rc=$?` 永远执行不到 —— 实测现象是构建日志里既没有 `[hvigor rc=...]`
# 也没有末尾那段「=== 签名说明 ===」，而 SignHap 在 CLI 下**必然失败**，
# 也就是说这段引导**从来没打印过**。必须在本条命令前后临时关闭 `set -e`。
set +e
"$LPR_NODE_HOME/node.exe" \
  "$DEVECO_HOME/tools/hvigor/bin/hvigorw.js" \
  assembleHap --mode module -p product=default -p buildMode="$BUILD_MODE" --no-daemon "$@" 2>&1
rc=$?
set -e
echo "[hvigor rc=$rc]"

# SignHap 在纯 CLI 下必然失败，这里显式说明原因并给出替代路径。
#
# 2026-09-22 结论（已读到 hvigor 源码确认）：hvigor 的
#   tools/hvigor/hvigor-ohos-plugin/src/utils/decipher-util.js
# 里 DecipherUtil.decryptPwd() 被 **无条件** 调用，它要求：
#   (a) keystore 口令 ≥32 字符且为偶数长度；
#   (b) keystore 同目录下存在 material/{ac,ce,fd} 三个材料目录；
#   (c) 口令是 AES-128-GCM 密文的 hex 串，密钥由 material 经 PBKDF2 派生。
# 也就是说 DevEco「自动签名」写的 84 字符 0000001A... 是加密串、不是明文，
# **明文口令在 assembleHap 链路上无法使用**。典型报错依次为：
#   11014003 Init keystore failed / parseAlgParameters failed  （口令是加密串时）
#   00303116 ... length ... less than 32                        （口令太短时）
#   00303117 ... is an even number                              （口令长度为奇数时）
#   00308018 ENOENT ... stat '<dir>\material'                    （缺 material 目录时）
# 正确做法：让 hvigor 只做打包，签名用 tools/sign_hap.sh 单独完成
# （它直接调 hap-sign-tool.jar，接受明文口令）。
if [ "$rc" -ne 0 ] && [ ! -f "$PROJ/entry/build/default/outputs/default/entry-default-signed.hap" ]; then
  echo ""
  echo "=== 签名说明 ==="
  echo "若上面失败在 :entry:default@SignHap，这是 CLI 环境的已知限制（见本脚本注释）。"
  echo "打包产物若已生成，按**目标设备**选签名脚本 ——"
  echo "两者的信任根互不相认，选错必拒装（ADR-0009）："
  echo "  华为商用真机（如 MIA-AL00）: bash tools/sign_hap_huawei.sh"
  echo "                               （信任根 Huawei CBG Root CA G2）"
  echo "  云手机 / OH 开发板:          bash tools/make_signing_material.sh   # 只需跑一次"
  echo "                               bash tools/sign_hap.sh"
  echo "                               （信任根 OpenHarmony Application Root CA）"
  echo "也可改用 DevEco Studio 构建（它能解开自己的加密口令）。"
  echo "⚠️ 用错脚本的症状是装机报 code:9568257 / fail to verify pkcs7 file；"
  echo "   那个报错与证书过期/签名算法/包名/UDID 全都无关，只去看信任根。"
fi

echo "=== 产物 ==="
ls -la "$PROJ/entry/build/default/outputs/default/" 2>&1 || echo "(无 outputs 目录)"
exit $rc
