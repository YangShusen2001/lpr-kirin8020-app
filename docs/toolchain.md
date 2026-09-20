# CANN DDK 与工具链

## 当前状态：DDK 已在磁盘上

`~/lpr-harmony/omg_conv/ddk/`（**822 MB**，326 个 `.so`）：

| 内容 | 说明 |
|---|---|
| `tools/tools_omg/omg` | 包装脚本（6661 B） |
| `tools/tools_omg/master/omg` | 真正的 ELF64 二进制（178 KB） |
| `tools/tools_dopt/{dopt_onnx_py3,dopt_pytorch_py3,dopt_tf_py3}/dopt` | INT8 PTQ（校准） |
| `tools/tools_ascendc/` | AscendC（2349 个条目，大头是 bisheng/lld，单文件 >100MB） |
| `tools/platform/kirin9020/` | 平台插件 |

`ddk_platform_plugin_info` → `DDK_PLATFORM_PLUGIN_100.600.020.010`。
`config/kirin9020.ini`：`SoC_version=kirin9020`、`AIC_version=AIC-L-310`。

**本项目不需要重新获取。** 若需重新下载，见下。

## 重新下载（匿名，无需账号）

| 项 | 值 |
|---|---|
| 文档页 | <https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations>（「开发准备」→ Tools下载 表） |
| 包 | `DDK-tools-next-6.1.1.0.zip` |
| 大小 | **252.12 MiB** zipped / 796.1 MiB unpacked / 2937 条目 |
| SHA256 | `87d7e3f186ad5c527a9385cea555559ea53c63b87dc483820523bcf7bf6f87e5` |
| 准入 | **无需登录、无需实名、无需企业认证**（实测匿名 GET 成功） |

⚠️ 两点保留：
- 「匿名可下」是**程序化验证**的（绕过了浏览器 UI），不能排除华为界面上有登录墙。
- CDN 对 **HEAD 请求返回 403，对 GET/Range 返回 200** —— 用 HEAD 探测会误判为"被墙"。

## 环境要求

- **64 位 Linux**（OMG 是 ELF64）。Windows 的 OMG 在 `100.600.010.010` 已被华为删除。
- 捆绑 glibc **2.35** ⇒ Ubuntu 22.04（WSL2-Ubuntu-22.04 应可满足；本机 WSL 是 Ubuntu 26.04 / glibc 2.43，**实测可跑**）。
- `tools_dopt` 需 Python 3.10 + onnx 1.14 / onnxruntime 1.15。

## 平台插件与版本对齐

DDK 附带的平台插件：**`kirin9020` / `kirinx90` / `kirin9030`**。

⚠️ **没有 `kirin8020` 插件**。不要混淆 `kirin9020`（DDK 插件名）与 `Kirin 8020`（nova 14 Pro 的中端 SoC）—— 是不同的零件号。本项目的 `.om` 用 `--platform=kirin9020` 转换，**已在 Kirin 8020 真机上跑通**（A17：`compat=0/build_rc=0/run_rc=0`），所以该插件覆盖 8020。

版本对齐：

- 设备侧读取：`hdc shell param get const.hiai.vendor.hiaiversion`
  - ⚠️ 实测在本机（USB 连接的非 2in1 设备）返回 **`fail! errNum is:1002`**（参数不存在）。
    官方文档称该命令可用，但在非 2in1 上取不到。**只能走应用内 API `HMS_HiAI_GetVersion`。**
- 华为在 5.0.1.0 **删除了**「CANN Version 与芯片匹配表」⇒ **不存在官方版本↔芯片映射表**。
- OMG↔设备版本不匹配的行为**无官方文档**。首次转换须以 `compat=` / `build_rc=` / `run_rc=` 三关实测为准。

## 转换命令

```bash
wsl -d Ubuntu -- bash /mnt/c/Users/26671/lpr-kirin8020-app/tools/convert_om.sh
```

脚本内已记录四个实测陷阱（包装脚本入口 / 非 ASCII 输出路径 / 不要 `--target=omc` /
`--hiai_version` 而非 `--omg_version`）。

## 已知噪声（不是故障）

以下加载失败在 09-18 的成功日志里同样出现，**不影响产出**：

- `te_fusion` / `librl_search.so` / `libai_npucore_generated.so` 加载失败
- `kernel binary initialize failed, this store can use JIT only`

## 已验证的转换结果

| 模型 | 输入 | 字节数 | magic | 与 09-18 原件 |
|---|---|---|---|---|
| `om_cls` | `data:1,3,96,96` | 841,728 | `IMOD` | 一致 |
| `om_dethead` | `input:1,3,320,320` | 1,010,950 | `IMOD` | 一致 |
| `om_rec` | `data:1,3,48,160` | 5,041,253 | `IMOD` | 一致 |

`.ms` 侧（`converter_lite`，PC 端，**无需华为账号**）三个模型 SHA256 与既有产物**逐字节一致**。
