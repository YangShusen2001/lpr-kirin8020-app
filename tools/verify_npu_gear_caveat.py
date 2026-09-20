"""回归守卫：引用「全 NPU 档 30 fps」之前，必须先看见这条警示。

## 背景

2026-09-21 测到相机页**「全 NPU」档**（det 与 rec 都走 NNRT）能跑到
**30.00 fps（±0.3，0 丢帧）**，而生产档（det=CPU rec=NPU）只有 ~28 fps。
30 fps 正是相机的硬上限，所以这看起来是「把 NPU 用好 = 达到传感器上限」，
是个很适合写进论文的结论。

**但它不能直接用。** 同一批数据的 E2E 矩阵显示，当**检测器**走 NNRT 时，
参考图的识别结果会读错：

    det req=cpu        -> 苏ED5172   match=1   (7 字符，合法)
    det req=nnrt       -> 苏E05172   match=0   (8 字符，非法)
    det req=nnrt_fp32  -> 苏E05172   match=0
    det req=nnrt_fp16  -> 苏E05172   match=0

三个 NNRT 变体**全部**读错，且错法一致（在省位后插入一个 0）。
而相机页的「全 NPU」档正是 det 走 NNRT ——
**即 30 fps 是在「读错」的前提下达到的。**

## 本守卫做什么

检查 `evidence/` 与文档里是否已记录这条警示，防止有人只看到 30 fps
就把它当成可用的性能结论。它不是阻止优化，而是强制把代价写下来。

具体断言：
  1. docs/notes/camera-fps-ceiling.md 中存在「读错」「match=0」等警示
  2. 该文档明确写出 30 fps 与准确率之间的取舍（不能只报帧率）
  3. evidence/ 里留有 E2E 矩阵的证据文件

用法：
    python tools/verify_npu_gear_caveat.py
退出码 0 = 警示已就位。
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

def find_ceiling_doc() -> Path:
    """定位 docs 仓库里的 camera-fps-ceiling.md。

    两个仓库的实际位置不固定（本仓库在 ~/lpr-kirin8020-app，
    docs 仓库在 ~/Desktop/车牌识别），所以按候选路径依次找，
    找不到再退回全局搜索。硬编码单一路径会在换机器时误报。
    """
    here = Path(__file__).resolve().parent
    candidates = [
        Path.home() / "Desktop" / "车牌识别" / "docs" / "notes"
                   / "camera-fps-ceiling.md",
        here.parent.parent / "车牌识别" / "docs" / "notes"
                   / "camera-fps-ceiling.md",
        here.parent / "docs" / "notes" / "camera-fps-ceiling.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 最后手段：向上两级目录内搜
    for base in (Path.home() / "Desktop", here.parent.parent):
        if not base.is_dir():
            continue
        try:
            hits = list(base.rglob("docs/notes/camera-fps-ceiling.md"))
        except OSError:
            continue
        if hits:
            return hits[0]
    return candidates[0]


ROOT = Path(__file__).resolve().parent.parent
CEILING = find_ceiling_doc()
EVIDENCE = ROOT / "evidence"

REQUIRED_MENTIONS = [
    ("苏E05172 这个错误结果", ["苏E05172"]),
    ("match=0 的证据", ["match=0"]),
    ("8 字符非法的说明", ["8 字符", "非法"]),
]

print("全 NPU 档 30 fps 的准确率警示检查")
print("=" * 62)

ok = True

if not CEILING.exists():
    print(f"  [FAIL] 找不到文档 {CEILING}")
    print("         30 fps 的结论必须有文档承载其准确率代价，不能只在提交信息里")
    ok = False
else:
    text = CEILING.read_text(encoding="utf-8", errors="replace")
    print(f"  [OK] 找到文档 {CEILING.name}")
    for label, keys in REQUIRED_MENTIONS:
        hit = [k for k in keys if k in text]
        if hit:
            print(f"  [OK] 已记录：{label}（命中 {hit}）")
        else:
            print(f"  [FAIL] 未记录：{label}（需含 {keys} 之一）")
            ok = False

    # 关键：不能只报帧率，必须同时给出"读错"的取舍
    has_fps = "30.00 fps" in text or "30 fps" in text
    has_warn = "苏E05172" in text
    if has_fps and has_warn:
        print("  [OK] 帧率与准确率代价并列呈现（不是只报 30 fps）")
    else:
        print("  [FAIL] 30 fps 与读错警示未并列 —— 单独引用帧率会产生误导")
        ok = False

# E2E 矩阵证据
e2e = sorted(EVIDENCE.glob("e2e_matrix*")) if EVIDENCE.is_dir() else []
if e2e:
    print(f"  [OK] evidence 中有 E2E 矩阵证据: {e2e[0].name}")
else:
    print("  [warn] evidence 中没有 E2E 矩阵证据文件 —— 建议存档一份，"
          "以便他人复核 det 走 NNRT 时确实读错")

print("=" * 62)
if ok:
    print()
    print("=> 警示已就位：引用全 NPU 档 30 fps 时，读者会同时看到准确率代价")
    print("   仍需注意：该结论只在参考图 + 相机实时场景验证过，")
    print("   未做多图统计，正式引用前需补准确率对照。")
    sys.exit(0)

print()
print("=> 未通过：30 fps 的准确率代价尚未被完整记录")
sys.exit(1)
