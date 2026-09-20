# T8 · 算子覆盖对撞（`.om` vs `.ms`）

- **日期**：2026-09-21
- **票**：#9
- **对应 ADR**：ADR-0002（论文主张）
- **数据**：`op_collide.csv` / `op_collide.json`

## 方法

把 `ShusenPaper` 的算子探针模型（51 个）从 ONNX 转成 **CANN `.om`**，在真机上逐个跑三关：

| 关 | 字段 | 含义 |
|---|---|---|
| 格式准入 | `hiai_compat_code` | 0 = COMPATIBLE，1 = INCOMPATIBLE |
| 图构建 | `construct` / `build_rc` | 能否构造 compilation 并 build |
| 真跑 | `run_rc` / `run_ms_each` | 能否执行并给出延迟 |

`.ms` 侧的数据来自 `ShusenPaper` 的既有 L1 算子矩阵（36 算子三信号判定）。

## 结果：CANN 侧 **51/51 全部通过**

```
compat=0 : 51
construct=ok : 51
build_rc=0 : 51
run_rc=0 : 51
```

**零失败、零回落。** 包括 `.ms`/NNRT 侧被拒的那一批：
`relu` · `sigmoid` · `softmax` · `maxpool` · `pad` · `cast_f16` · `transpose` · `resize` · `tanh`。

> **这是「NPU 支持性是 模型 × 工具链 的联合属性」最直接的一条证据。**
> 同一个算子，在 NNRT 通路上被切出子图回落 CPU，在 CANN 通路上完整跑通 ——
> 差别不在硬件，在工具链。

### 一个额外的样本

`convtranspose` 在 **`.ms` 侧根本转不出来**（`ShusenPaper` 的 manifest notes 记着
`converter_lite 2.6.0` 形状推断失败），却在 CANN 侧**转换成功并跑通**（`build_rc=0`、`run_rc=0`）。
同一个算子、同一份 ONNX，一侧连模型都产不出来，另一侧能跑 —— 这是联合属性最尖锐的一例。

## 延迟数据

真机 `run_ms_each` 首值（毫秒）：

**最快的 8 个**

| 算子 | p50 (ms) | 输出 |
|---|---|---|
| `cast_f16` | 0.763 | `[1x32x112x112]` FLOAT16 |
| `matmul_256` | 0.910 | `[1x256x1x1]` |
| `rl_matmul_512` | 0.910 | `[1x512x1x1]` |
| `strided_slice` | 0.910 | `[1x32x55x112]` |
| `rl_matmul_1024` | 0.959 | `[1x1024x1x1]` |
| `slice_channel` | 0.984 | `[1x16x112x112]` |
| `conv3x3_s2` | 1.028 | `[1x64x56x56]` |
| `maxpool` | 1.032 | `[1x32x56x56]` |

**最慢的 8 个**

| 算子 | p50 (ms) | 输出 |
|---|---|---|
| `rl_conv_k3_c64` | **34.827** | `[1x64x112x112]` |
| `repro_mislabel` | **31.119** | `[1x16x112x112]` |
| `concat` | 6.662 | `[1x64x112x112]` |
| `rl_conv_k5_c512` | 6.659 | `[1x512x112x112]` |
| `rl_conv_k3_c512` | 5.360 | `[1x512x112x112]` |
| `rl_conv_k1_c512` | 4.964 | `[1x512x112x112]` |
| `erf` | 4.650 | `[1x32x112x112]` |
| `mul_const` | 4.185 | `[1x32x112x112]` |

⚠️ **这批延迟不可直接当基准用**，原因是每条只跑了 3 次且**没有热身**（`run_ms_each` 的三个值
离散度很大，如 `cast_f16` 的 `3.147|0.135|0.105`）。它们只用于**排序**，不作为绝对数字引用。
要做基准须走 T2 的协议（warm-up 10 + 100 次取 p50 + ≥5 轮 + 热态标注）。

## 本轮修掉的一个真实工程缺陷

首轮 51 个算子里有 **13 个在 ArkTS 侧收到空串**，而 native 日志里只看到 46 次执行。

根因在 `InferenceRunner`（专用推理线程）——它是**单任务槽 + 单一 `has_` 标志**：

```cpp
Submit: task_ = fn; has_ = true; notify; doneCv_.wait([&]{ return !has_; });
Loop  : 取走 task_ 执行; has_ = false; doneCv_.notify_all();
```

当 A 正在执行、B 提交时，**B 覆盖 `task_`**；A 结束后 Loop 置 `has_=false` 并 `notify_all`，
**A 和 B 都被唤醒，但 B 的 `fn` 从未执行** —— B 的 `job->kv` 保持空串，调用方只看到
一个"成功 resolve 但内容为空"的 Promise。

**修法**：改成**真正的队列**，每个任务带自己的 `finished` 标志与条件变量，
提交者只等自己那一个，任务不可能被覆盖。任务本身仍串行执行（NNRT 委托不允许并发）。

修后干净基线下：**51/51 全部 `ok=1` 完整返回，0 个空返回。**

另加两道卫兵：
1. native 侧空 KV 一律改写成 `ok=0;error=empty-return`，**绝不让空串到达 JS**
   （否则日志分不清"没执行"与"执行了但结果为空"）
2. ArkTS 侧每轮 `await this.tick()` 让出事件循环，降低并发压力

## 附带：WSL 内存配额

T8 一开始被 WSL 启动失败挡住：`Wsl/Service/CreateInstance/CreateVm/HCS/0x800705aa`
（资源不足）。默认配额取宿主内存 50% ≈ 7.7 GB，但实际起不来。

修法：写 `~/.wslconfig` 显式压低到 `memory=3GB / processors=4 / swap=2GB`。
OMG 是单线程 CLI，不需要大配额。

## 限制（引用时必须一起说）

- `.ms` 侧的数据是 **`ShusenPaper` 的既有矩阵**（36 算子），不是本轮重测。
  两侧的**协议不同**，所以延迟数字**不可跨栈比较**；可比的只有"通过/被拒"这一列。
- `.om` 侧每条只跑 3 次、无热身 —— 延迟只用于排序。
- 探针输入是**合成确定性填充**，因此 `o0_argmax` 不可解释（日志里已标
  `synthetic-input->argmax NOT meaningful`）。本轮结论只覆盖**机制与准入**，不涉及精度。
- CANN 平台插件只有 `kirin9020`/`kirinx90`/`kirin9030`，**没有 `kirin8020`**；
  本轮用 `--platform=kirin9020` 转换并在 8020 真机跑通，与该插件覆盖 8020 的既有结论一致。
- 「CANN 全通过」**不等于**「CANN 更快」。已有的生产档实测是 MS Lite 在识别器上更快
  （rec 3.99 vs 5.19 ms）。本轮的结论只关于**准入边界**。
