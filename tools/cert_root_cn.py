#!/usr/bin/env python3
"""cert_root_cn.py —— 读出证书链最靠根的 Issuer CN，判定该包能被哪类设备接受。

用途
====
装机链路上，真正决定「这台设备会不会接受这个包」的是**证书链的信任根**，
而不是签名算法、有效期、包名或 UDID 名单。本脚本把这个判据打出来：

    Huawei CBG ...          → 华为商用真机可装
    OpenHarmony ...         → 华为商用真机必拒装（code 9568257 / fail to verify pkcs7 file）

实测对照（本项目 2026-09-22）
============================
    .signing/lpr-app.cer                              → OpenHarmony Application Root CA
    ~/.ohos/config/default_LprDemo_*.cer              → Huawei CBG Root CA G2

为什么不用 `keytool -printcert` 的文本输出
==========================================
1. 它的输出是**本地化编码**（本机 GBK），按 utf-8 读会乱码，
   连 "Owner:"/"所有者:" 这个标签都可能匹配不上。
2. 更根本的是：DN 在 DER 里是 **SEQUENCE of SET of SEQUENCE{OID, UTF8String}**，
   `CN` 的 ASCII 与 `=` 之间隔着 OID 与长度字节，**不是明文相邻的**。
   所以 `grep 'CN='` 在 DER 上必然抓空（这个坑踩过）。
   正确做法是用 X.509 解析器读 Subject/Issuer。

用法
====
  python tools/cert_root_cn.py <cert.cer>
  python tools/cert_root_cn.py <cert.cer> --all      # 打印链上每一张的 subject / issuer

退出码：认出 Huawei 根 → 0；认出 OpenHarmony 根 → 2；无法判定 → 3。
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import warnings

DEFAULT_DEVECO_HOME = os.environ.get("DEVECO_HOME", "D:/IDE/DevEco_Studio")

# Java 签出来的证书在 signatureAlgorithm 参数里带 NULL，cryptography 会告警。
# 这只影响解析、不影响判定，压掉以免污染脚本输出（引用 issue 见告警原文）。
warnings.filterwarnings("ignore", category=Warning)

_PEM_RE = re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S)


def certs_from_cer(cer_path: str) -> list:
    """用 keytool -printcert -rfc 把 .cer 转成 DER 列表（顺序：leaf → … → root）。"""
    from cryptography import x509

    keytool = os.environ.get(
        "LPR_KEYTOOL", os.path.join(DEFAULT_DEVECO_HOME, "jbr", "bin", "keytool.exe")
    )
    if not os.path.isfile(keytool):
        raise SystemExit(f"[cert_root_cn] 找不到 keytool：{keytool}（用 LPR_KEYTOOL 指定）")
    if not os.path.isfile(cer_path):
        raise SystemExit(f"[cert_root_cn] 找不到证书：{cer_path}")

    r = subprocess.run(
        [keytool, "-printcert", "-rfc", "-file", cer_path],
        capture_output=True, text=True, errors="replace",
    )
    if r.returncode != 0:
        raise SystemExit(f"[cert_root_cn] keytool 失败：{(r.stderr or r.stdout)[:400]}")

    out = []
    for pem in _PEM_RE.findall(r.stdout or ""):
        body = "".join(pem.strip().splitlines()[1:-1])
        out.append(x509.load_der_x509_certificate(base64.b64decode(body)))
    return out


def describe(cert) -> tuple:
    """返回 (subject_cn, issuer_cn)。用 RFC4514 串再取 CN，避免手写 DN 解析。"""
    def cn(dn) -> str:
        m = re.search(r"(?:^|,)CN=([^,]+)", dn.rfc4514_string())
        return m.group(1) if m else dn.rfc4514_string()

    return cn(cert.subject), cn(cert.issuer)


def main() -> int:
    ap = argparse.ArgumentParser(description="读证书链最靠根的 Issuer CN")
    ap.add_argument("cer", help=".cer 文件路径")
    ap.add_argument("--all", action="store_true", help="打印链上每一张")
    args = ap.parse_args()

    certs = certs_from_cer(args.cer)
    if not certs:
        print("[cert_root_cn] 没解析出任何证书", file=sys.stderr)
        return 3

    descs = [(s, i) for s, i in (describe(c) for c in certs)]
    if args.all:
        for idx, (subj, iss) in enumerate(descs):
            print(f"[{idx}] subject = {subj}")
            print(f"     issuer  = {iss}")
        return 0

    # 最靠根的那张 = 最后一张的 issuer（自签根则 issuer==subject）
    last_subj, last_iss = descs[-1]
    root = last_iss if last_iss else last_subj
    print(root)
    if "Huawei" in root:
        return 0
    if "OpenHarmony" in root:
        return 2
    return 3


if __name__ == "__main__":
    sys.exit(main())
