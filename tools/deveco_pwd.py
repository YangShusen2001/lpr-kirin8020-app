#!/usr/bin/env python3
"""deveco_pwd.py —— 解出 DevEco Studio「自动签名」写进 build-profile.json5 的明文口令。

为什么需要这个
==============
DevEco 把签名口令加密后写进 build-profile.json5，形如 ::

    "storePassword": "0000001A3866D7...FD9"     # 84 字符十六进制

这份密文只能由 DevEco 自己的材料解开。纯 CLI 环境里，正确做法是
**直接 require DevEco 自带的 decipher-util.js**（它就在 DevEco 安装目录里），
而不是重新实现它的算法 —— 原因见下。

为什么不自己重新实现（两次踩坑记录）
====================================
`DecipherUtil.getRootKey` 的实现是：

    const i = t.concat(this.component);           // t = 3 个 16 字节的 fd 片段
    const s = this.xorComponents(i, e);            // 逐字节异或，得 Int8Array
    const _ = crypto.pbkdf2Sync(s.toString(), r, 1e4, 16, "sha256");
    //                          ^^^^^^^^^^^^ 陷阱在这
    //  r 是 salt，由 fs.readFileSync 得来，在 readDirBytes 里被包成 Int8Array

两个陷阱，各自都会让复现结果偏离：

1. **`s.toString()` 不是 hex、不是 latin1。**
   `Int8Array` 没有覆写 `toString()`，于是退化成 `Array.prototype.toString()`：
   每个元素按 **十进制** 转文本、用 `,` 连接，且元素是**有符号** int8。
   本机这份 material 的 32 字节 x 会变成 120 个 ASCII 字符，形如
       "48,-102,-18,-77,67,112,...,-64,119"
   （`0x30` → `"48"` 是数字 48，不是字符 `'0'`；`0x9A` → `"-102"`。）
   网上把它当 hex / latin1 的解读**都是错的**。

2. **salt 侧走的是 `Buffer.from(Int8Array).toString('latin1')` 的膨胀语义。**
   Node 的 Buffer `'latin1'` **解码**对 >0x7F 的字节会走 utf8 lone-surrogate，
   产出 U+DC80..U+DCFF，编码回 bytes 时该字节占 3 个字节。
   即：原本 16 字节的 `ac` 会膨胀成 30+ 字节。

这两处只要有一处还原不准，PBKDF2 出来的 rootKey 就是另一个值，
后续 GCM 会直接 `InvalidTag` —— **而且不会告诉你错在哪**。
因此本脚本的设计原则是：**能用原版就用原版**。

做法
====
1. 用 node `require()` DevEco 自带的 `decipher-util.js`，注入一个最小 stub 顶掉
   它依赖的 `@ohos/hvigor` 日志模块（我们只需要算法，不需要它的日志）。
2. 调 `DecipherUtil.decryptPwd(materialDir, cipherHex, label)` 拿明文。
3. 把结果打印出来，交给 tools/sign_hap.sh。

沿用 `DecipherUtil` 的**另一个好处**：DevEco 升级后算法若有变化，
本脚本自动跟随，不会因为版本漂移而悄悄解错。

用法
====
  python tools/deveco_pwd.py                 # 自动找 ~/.ohos/config 与仓库里的配置备份
  python tools/deveco_pwd.py --print-store   # 只打印 storePassword
  python tools/deveco_pwd.py --print-key     # 只打印 keyPassword
  python tools/deveco_pwd.py --verify        # 额外用 keytool 校验 .p12 能否打开

环境变量
========
  DEVECO_DECIPHER_UTIL   decipher-util.js 的路径（默认按 DEVECO_HOME 推）
  DEVECO_HOME            DevEco Studio 安装根，默认 D:/IDE/DevEco_Studio
  LPR_NODE_BIN           用来跑 DevEco 脚本的 node，默认取本机 node
  LPR_DEVECO_PROFILE     显式指定含密文的 build-profile.json5

安全
====
输出的是**明文口令**。只在本地终端用，不要写进任何会提交的文件、不要贴进聊天。
本脚本不内置任何密钥 —— 密文与材料都从本机 ~/.ohos/config 读，
算法由本机 DevEco 提供。签名材料是「本机专用」的调试材料，不构成发布凭据。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_DEVECO_HOME = os.environ.get("DEVECO_HOME", "D:/IDE/DevEco_Studio")
DEFAULT_MATERIAL_ROOT = os.path.join(os.path.expanduser("~"), ".ohos", "config")

# 用 DevEco 原版算法解密。注意两个点：
#   - 通过 Module._load 拦截，把 ohos-logger 换成 stub（它 require @ohos/hvigor）
#   - 走 process.stdout.write 而不是 console.log，避免被主机日志污染
_JS_TEMPLATE = r"""
const path = require('path');
const Module = require('module');

const UTILS = process.argv[2];
const LOGGER = path.join(path.dirname(UTILS), 'log', 'ohos-logger.js').replace(/\\/g, '/');

const origLoad = Module._load;
Module._load = function () {
  let resolved = '';
  try { resolved = Module._resolveFilename(arguments[0], arguments[1], arguments[2]); } catch (e) { }
  if (resolved.replace(/\\/g, '/') === LOGGER) {
    return {
      OhosLogger: {
        getLogger: () => ({
          printErrorExit: (code, args) => {
            throw new Error('DecipherUtil.exit ' + code + ' :: ' + JSON.stringify(args));
          },
          printError: () => { }, printWarn: () => { },
          printInfo: () => { }, printDebug: () => { },
        }),
      },
    };
  }
  return origLoad.apply(this, arguments);
};

const D = require(UTILS).DecipherUtil;
const materialDir = process.argv[3];
const cipherHex = process.argv[4];
process.stdout.write(D.decryptPwd(materialDir, cipherHex, 'deveco_pwd.py'));
"""


def _find_node() -> str:
    """优先用环境变量，其次 PATH，最后扫本机常见 managed 位置。"""
    explicit = os.environ.get("LPR_NODE_BIN", "")
    if explicit and os.path.isfile(explicit):
        return explicit
    for name in ("node.exe", "node"):
        found = shutil.which(name)
        if found:
            return found
    import glob

    pats = [
        os.path.join(os.path.expanduser("~"), ".workbuddy", "binaries", "node",
                     "versions", "*", "node.exe"),
        r"D:\Tools\NodeJS\node.exe",
        r"C:\Program Files\nodejs\node.exe",
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    raise SystemExit("[deveco_pwd] 找不到 node。用 LPR_NODE_BIN 指定。")


def _find_decipher_util() -> str:
    explicit = os.environ.get("DEVECO_DECIPHER_UTIL", "")
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"[deveco_pwd] DEVECO_DECIPHER_UTIL 指向的文件不存在：{explicit}")
        return explicit.replace("\\", "/")
    cand = os.path.join(
        DEFAULT_DEVECO_HOME, "tools", "hvigor", "hvigor-ohos-plugin",
        "src", "utils", "decipher-util.js",
    )
    if os.path.isfile(cand):
        return cand.replace("\\", "/")
    raise SystemExit(
        "[deveco_pwd] 找不到 decipher-util.js。\n"
        f"           试过：{cand}\n"
        "           用 DEVECO_HOME 或 DEVECO_DECIPHER_UTIL 显式指定。"
    )


def decrypt_with_deveco(cipher_hex: str, material_root: str) -> str:
    """调 DevEco 自带的 DecipherUtil 解密，返回明文。"""
    js_path = _find_decipher_util()
    node = _find_node()
    utils = js_path

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(_JS_TEMPLATE)
        tmp_js = fh.name
    try:
        r = subprocess.run(
            [node, tmp_js, utils, material_root.replace("\\", "/"), cipher_hex],
            capture_output=True, text=True, errors="replace",
        )
        if r.returncode != 0:
            raise SystemExit(
                "[deveco_pwd] DevEco 解密器报错：\n" + (r.stderr or r.stdout).strip()[:1200]
            )
        return r.stdout
    finally:
        os.unlink(tmp_js)


def find_config(material_root: str) -> tuple:
    """找一份含 storePassword/keyPassword 的 build-profile.json5（JSON5，不能直接 loads）。"""
    import re

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    explicit = os.environ.get("LPR_DEVECO_PROFILE", "")
    if explicit:
        candidates.append(explicit)
    # 仓库里 DevEco 写入的原版配置备份（sign_hap / make_signing_material 的参考来源）
    candidates.append(os.path.join(here, ".signing", "build-profile.json5.deveco-bak"))
    candidates.append(os.path.join(material_root, "build-profile.json5"))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        raw = open(path, encoding="utf-8").read()
        store = re.search(r'"storePassword"\s*:\s*"([0-9A-Fa-f]+)"', raw)
        keyp = re.search(r'"keyPassword"\s*:\s*"([0-9A-Fa-f]+)"', raw)
        if store or keyp:
            return path, (store.group(1) if store else "", keyp.group(1) if keyp else "")
    raise SystemExit("[deveco_pwd] 没找到含 storePassword/keyPassword 的 build-profile.json5")


def _verify_keystore(material_root: str, store_pw: str, config_path: str) -> str:
    """用 keytool 打开配置里 `storeFile` 指向的那个 .p12，确认口令真的可用。

    ★ 必须按配置里的 storeFile 定位，不能拿目录下第一个 .p12 ——
      ~/.ohos/config/ 下同时躺着好几个项目的签名材料，
      随便抓一个来试会拿到 `rc=1`，然后误判成"口令不对"（这个坑踩过一次）。
    """
    import glob
    import re

    keytool = os.environ.get(
        "LPR_KEYTOOL", os.path.join(DEFAULT_DEVECO_HOME, "jbr", "bin", "keytool.exe")
    )
    if not os.path.isfile(keytool):
        return "（跳过：找不到 keytool）"

    raw = open(config_path, encoding="utf-8").read()
    m = re.search(r'"storeFile"\s*:\s*"([^"]+)"', raw)
    if m:
        p12 = m.group(1).replace("\\\\", "\\")
    else:
        hits = glob.glob(os.path.join(material_root, "*.p12"))
        if not hits:
            return "（跳过：配置里没有 storeFile，目录下也没有 .p12）"
        p12 = hits[0]
    if not os.path.isfile(p12):
        return f"（跳过：storeFile 不存在 {p12}）"

    r = subprocess.run(
        [keytool, "-list", "-keystore", p12, "-storetype", "PKCS12", "-storepass", store_pw],
        capture_output=True, text=True, errors="replace",
    )
    name = os.path.basename(p12)
    return f"通过（{name} 打开成功）" if r.returncode == 0 else f"失败 rc={r.returncode}（{name}）"


def main() -> int:
    ap = argparse.ArgumentParser(description="解出 DevEco 自动签名的明文口令")
    ap.add_argument("--material-root", default=DEFAULT_MATERIAL_ROOT,
                    help=f"含 material/ 的目录，默认 {DEFAULT_MATERIAL_ROOT}")
    ap.add_argument("--from", dest="from_path", default="",
                    help="显式指定 build-profile.json5（或环境变量 LPR_DEVECO_PROFILE）")
    ap.add_argument("--print-store", action="store_true", help="只打印 storePassword")
    ap.add_argument("--print-key", action="store_true", help="只打印 keyPassword")
    ap.add_argument("--verify", action="store_true", help="额外用 keytool 校验 .p12 能否打开")
    args = ap.parse_args()

    if args.from_path:
        os.environ["LPR_DEVECO_PROFILE"] = args.from_path

    material_root = args.material_root
    if not os.path.isdir(os.path.join(material_root, "material")):
        raise SystemExit(f"[deveco_pwd] 找不到 {os.path.join(material_root, 'material')}")

    path, (store_c, key_c) = find_config(material_root)

    store_pw = decrypt_with_deveco(store_c, material_root) if store_c else ""
    key_pw = decrypt_with_deveco(key_c, material_root) if key_c else ""

    if args.print_store:
        print(store_pw)
        return 0
    if args.print_key:
        print(key_pw)
        return 0

    print(f"[deveco_pwd] 材料   = {os.path.join(material_root, 'material')}")
    print(f"[deveco_pwd] 配置   = {path}")
    print(f"[deveco_pwd] 算法   = {_find_decipher_util()}（DevEco 原版）")
    print(f"[deveco_pwd] storePassword = {store_pw}")
    if key_c:
        print(f"[deveco_pwd] keyPassword   = {key_pw}")
    else:
        print("[deveco_pwd] keyPassword   = （配置里没有，通常与 storePassword 同值）")
    if args.verify and store_pw:
        print(f"[deveco_pwd] .p12 校验 = {_verify_keystore(material_root, store_pw, path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
