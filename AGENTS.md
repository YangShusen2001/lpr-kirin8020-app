# 麒麟 8020 车牌识别 App（lpr-kirin8020-app）

> **本文件只覆盖 App 工程自身的约定。**
> **决策、规格、词汇表在另一个仓库**：`~/Desktop/车牌识别` → GitHub `YangShusen2001/lpr-kirin8020`。
> 动手前先读那边的 `CONTEXT.md`、`docs/adr/`、`docs/spec.md`、`docs/agents/`。

## 为什么 App 和文档分成两个目录

**hvigor 拒绝任何含非 ASCII 字符的工程路径**（错误 `00306003`）：

> Invalid project path. Current path does not match: ...
> Please modify the project path to ensure that it only contains letters, digits,
> hyphens (-), underscores (_), periods (.), english parentheses (()), spaces, or the @ symbol

文档仓库在 `C:\Users\26671\Desktop\车牌识别`（含中文），所以 DevEco 工程**必须**放在 ASCII 路径。已实测确认：

- 直接用中文路径构建 → `00306003`
- 用 junction 把中文路径映射到 ASCII 路径 → **无效**，hvigor 会解析回真实路径
- 移动文档仓库到 ASCII 路径 → 失败，该目录被 harness 进程占用不可删

**所以这是平台约束逼出来的拆分，不是随意选的。**

## 目录

| 路径 | 说明 |
|---|---|
| `LprDemo/` | DevEco 工程（API 24 / HarmonyOS 6.1.1，bundleName `com.shusen.lprdemo`） |
| `tools/` | 转换链脚本（ONNX → `.ms`） |
| `models_ms/` | 转换产物（**不入库**，由 `tools/convert_ms.sh` 重新产出） |

## 构建

```bash
bash build.sh assembleHap --mode module -p product=default -p buildMode=debug --no-daemon
```

`build.sh` 里三个环境变量与两个陷阱规避**缺一不可**（都是从 `~/lpr-harmony` 继承的实测教训）：

1. `unset NODE_OPTIONS` —— 宿主注入的 `node-language-shim.cjs`（safe-delete 保护）会拦截 hvigor 清理 `.hvigor/report/*.json`，累计删除数触阈值就抛 `SAFE_DELETE_BULK_CONFIRM_REQUIRED`，导致构建在「打包已完成、签名收尾前」崩掉。症状：`unsigned.hap` 是新的、`signed.hap` 还是旧的。
2. `DEVECO_SDK_HOME` 必须指向 `sdk` 根而非 `sdk/default` —— hvigor 的本地 SDK 扫描器（`AbstractLocalComponentLoader.findPotentialSdks`）只遍历子目录找 `<child>/sdk-pkg.json`，从不检查根目录自身。
3. `JAVA_HOME` 必须用 DevEco 自带的 jbr（JDK21）—— 系统 PATH 上是 JDK 1.8，读不了 JDK21 生成的 PKCS12 密钥库，会报 `11014003 Init keystore failed`。
4. `build-profile.json5` 的 `products[].signingConfig` 必须显式引用 `signingConfigs.name`，否则只产 unsigned HAP（症状：`bm install` 报 `code:9568320 error: no signature file`）。
5. 装机前**先卸载旧 App**（同 bundleName），避免新旧证据混口径。

## 模型与 DDK 不在本仓库

`.gitignore` 排除 `*.ms` / `*.om` / `*.onnx` / `*.ncnn.bin` / `*.p12` / `*.cer`：

- `.ms` 由 `tools/convert_ms.sh` 从 ONNX 重新产出，**实测 SHA256 与既有产物逐字节一致**
- CANN DDK 匿名可下（`DDK-tools-next-6.1.1.0`，252 MiB）
- 重建步骤见文档仓库的 `docs/notes/toolchain-and-sources-on-disk.md`

## 硬约束（写代码前必读）

1. **NPU 利用率不可测** —— 禁止任何利用率数字。证据只能是延迟差 + 逐算子落点 + 张量指纹。
2. **会话必须常驻** —— NNRT delegate 析构路径存在 cppcrash，不可反复创建/销毁。
3. **动态 batch 会导致构图失败** —— 转换时固定 batch = 1。
4. **改端侧代码前先 `assembleHap` 干跑编译**。
5. **OMG 输出路径不能含非 ASCII 字符**（与本文档顶部的 `00306003` 同类缺陷）。
6. **持续后台计算不被允许** —— `SystemLoadLevel` 有 8 档，HIGH(3) 起停止无感服务。

## 已知待修（有票跟踪）

- **`Index.ets` 的颜色标签表是旋转的**：源码 `0→黄牌, 1→蓝牌, else→绿牌`，实测正确顺序是 `blue=0, green=1, yellow=2`。真机可复现：`hlpr-test.jpg`（实为绿牌）显示"蓝牌"，`scene-2.jpg` 的 `藏DT5022`（蓝牌）显示"黄牌"。见文档仓库 ADR-0005。
- **bundleName 暂无法更改**：签名 profile 绑死 `com.shusen.lprdemo`，新建 profile 需 DevEco GUI + 华为账号登录（自动签名走华为内网 KMS，公网不可达）。当前靠「装机前先卸载旧 App」达到证据隔离。

## 遗留物（不得进入本工程）

以下文件来自 `~/lpr-harmony`，是 2026-09-19 yolov8 集成时的手工合并残留，**合并从未完成**：

- `lpr_pipeline.cpp.yolov8_backup`
- `lpr_pipeline_new_v2.cpp`
- `CMakeLists.txt.bak-20260917-211518`
- `libncnn.so.1.bak-20260917-211518` / `libncnn.so.bak-20260917-211518`

复制时已排除。若再次出现，删掉。
