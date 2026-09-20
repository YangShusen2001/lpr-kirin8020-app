"""回归守卫：核对落点时，必须同时核 `cropSum`，不能只看 `landed=`。

## 背景

2026-09-21 发现：把**检测器**从 CPU 移到 NPU，参考图的识别结果从
`苏ED5172` 变成 `苏E05172`（8 字符，非法）。

排查时发现一个陷阱：**det 的 box 取整坐标完全相同**
（`1751|747|1879|855`），如果只看框，会误判为"检测器输出一致"。
实际差异藏在两个地方：

    detConf  0.7273 (cpu)  vs  0.7230 (nnrt)   <- 亚像素坐标不同
    cropSum  2773473 (cpu) vs  2763123 (nnrt)  <- 送进识别器的像素不同

即：**框的取整坐标相同 ≠ 送进下游的像素相同**。
亚像素差异经透视矫正被放大，最终把 `D` 读成了 `0`。

而 `landed=` 只告诉你"跑在哪个后端上"，**不告诉你算得对不对** ——
这次 det 确实落在 NNRT 上（landed 正确），结果却是错的。

## 本守卫做什么

1. 确认 `docs/notes/det-backend-alters-result.md` 存在，且记录了
   `cropSum` 的两个具体数值（差异的证据）。
2. 确认该文档明确写了"landed 正确 != 结果正确"这一条。
3. 确认 `evidence/e2e_matrix_det_backend.log` 在（可供复核）。

它不是阻止任何改动，而是防止"只看落点就下结论"这个错误重复发生。

用法：
    python tools/verify_cropsum_crosscheck.py
退出码 0 = 通过。不需要设备。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"


def find_note() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.home() / "Desktop" / "车牌识别" / "docs" / "notes"
                   / "det-backend-alters-result.md",
        here.parent.parent / "车牌识别" / "docs" / "notes"
                   / "det-backend-alters-result.md",
        here.parent / "docs" / "notes" / "det-backend-alters-result.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    for base in (Path.home() / "Desktop", here.parent.parent):
        if not base.is_dir():
            continue
        try:
            hits = list(base.rglob("docs/notes/det-backend-alters-result.md"))
        except OSError:
            continue
        if hits:
            return hits[0]
    return candidates[0]


NOTE = find_note()

print("cropSum 交叉核对守卫")
print("=" * 62)
ok = True

if not NOTE.exists():
    print(f"  [FAIL] 找不到 {NOTE}")
    print("         「det 换后端会改变识别结果」这条必须有文档承载。")
    ok = False
else:
    text = NOTE.read_text(encoding="utf-8", errors="replace")
    print(f"  [OK] 找到文档 {NOTE.name}")

    # cropSum 的两个具体数值 —— 这是差异的硬证据
    for label, token in (("CPU 的 cropSum 2773473", "2773473"),
                         ("NPU 的 cropSum 2763123", "2763123")):
        if token in text:
            print(f"  [OK] 记录了 {label}")
        else:
            print(f"  [FAIL] 未记录 {label} —— 没有数值就无法复核")
            ok = False

    # 关键教训
    if "landed" in text and ("结果正确" in text or "算对" in text):
        print("  [OK] 明确了「landed 正确 != 结果正确」")
    else:
        print("  [FAIL] 未写明「landed 正确 != 结果正确」这一教训")
        ok = False

e2e = sorted(EVIDENCE.glob("e2e_matrix*")) if EVIDENCE.is_dir() else []
if e2e:
    print(f"  [OK] evidence 有 E2E 矩阵证据: {e2e[0].name}")
else:
    print("  [warn] evidence 缺 E2E 矩阵证据，建议存档以便复核")
    ok = False

print("=" * 62)
print()
if ok:
    print("=> 通过：cropSum 交叉核对的要求已记录在案")
    print("   提醒：换任何角色的落点后，都要同时比对 cropSum 与 code，")
    print('   只比 box 的取整坐标会被「看起来一样」骗过去。')
    sys.exit(0)
print("=> 未通过")
sys.exit(1)
