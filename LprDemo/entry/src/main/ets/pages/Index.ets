/**
 * 原生 LPR 应用 —— **纯原生，无 ArkWeb**（2026-09-18 重写）
 *
 * 为什么拆掉 ArkWeb：网页演示只能吃 CPU（onnxruntime-web 既没有 WebGL EP，
 * 也拿不到 WebGPU adapter，见 ADR-004 §6.1），而本项目的核心恰恰是
 * 「三个模型各自落到 CPU / NPU / GPU 的哪一张卡上」。留一个跑不了加速器的
 * WebView 只能分散注意力，还要为它多打包 13 MB 的 ORT wasm 和三个网页。
 * 现在手机端只有一个原生界面：喂图 → 三后端推理 → 结果 + 分段计时。
 *
 * 两个标签页：
 *   原生演示   → 加载模型 / 枚举镜头 / 跑样本 / 看车牌与耗时
 *   探针控制台 → 加速器矩阵、GPU 专项、NPU 漂移、ncnn 自证、Vulkan 探针 + 日志
 *
 * ⚠️ 线程契约（与 libentry 一致，别绕过）：
 *   `lpr.*` 的**同步**版本会阻塞 UI 线程，连续占用 >3 s / >6 s 会被系统 watchdog
 *   判 THREAD_BLOCK 并 SIGKILL（实测 uvLoopTask 8887 ms，见 _evidence/A6-*.md）。
 *   本文件一律使用 `loadModelAsync` / `pipelineAsync` / `benchAsync` / `ncnnLoadAsync`；
 *   会话名传**模型文件路径**（原生侧按 (文件,后端) 去重，传角色标签会去重失效）。
 *
 * 注意：ArkTS 禁用 any/unknown，JSON.parse 过不了 linter，
 * 所以原生侧一律返回 `key=value;` 扁平串（见 napi_init.cpp 的 KvSanitize）。
 */
// 【2026-09-22 相机恢复】上一轮按 ADR-0008 把相机路径整体砍掉，连带删了三个 import。
// 同日用户要求「恢复相机实时」（docs/spec-vehicle-pipeline.md），这里只把仍需要的
// `router` 加回来：
//   @kit.ArkUI 的 router —— 跳 `pages/CameraPage`（入口见本文件 `pageHeader()`）
// `@kit.CameraKit` 与 `abilityAccessCtrl / Permissions` **仍然不在这里用**：
// 枚举物理镜头、申请 CAMERA 权限都在 CameraPage 自己那一页做，本页只负责跳转。
// 往回加用不到的 import 会让 linter 报 unused，也会让读者以为本页碰相机。
import { common } from '@kit.AbilityKit';
import { image } from '@kit.ImageKit';
import { router } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import lpr from 'libentry.so';
// T7 · RQ4 热特性：native sysroot 里**没有热态头文件**（已查 native/sysroot/usr/include，
// 无 thermal*），所以热档与电池温度只能在 ArkTS 侧取。这三个都在
// `openharmony/ets/api/` 下，不在 `hms/ets/api/`。
import { thermal } from '@kit.BasicServicesKit';
import { batteryInfo } from '@kit.BasicServicesKit';
import { systemLoad } from '@kit.BasicServicesKit';
// T12 读 gt.tsv 需要 TextDecoder（ArkTS 无浏览器那个全局 TextDecoder）。
import { util } from '@kit.ArkTS';
// 导出（长按/点「复制」）走系统剪贴板。用户定的导出形态是「终端界面上直接看 + 复制」，
// 不是写文件 —— App 私有目录在真机/云手机上 hdc 都拉不出来，复制成文本更可用。
import { pasteboard } from '@kit.BasicServicesKit';

const TAG = 'LprMatrix';
const NATIVE_TAG = 'LprNativeUI';
const DOMAIN = 0xD001;

// ------------------------------------------------- 深色仪器风设计变量（ADR-0008）
//
// 【为什么整体换掉 iOS 浅色】
// 旧控制台是 26 个探针按钮平铺在浅色长页上，结果是「看不了、认不出、跑不动」。
// 重设计选深色仪器风（跑分台/监控台那一类）：深底让状态色更跳、等宽数字对齐后
// 一列 p50 能竖着扫，比浅色卡片更适合高密度读数。
//
// 【⚠️ 颜色写法仍受 ArkUI 约束】
// ArkUI 的 8 位 hex 是 **ARGB**（#AARRGGBB），不是 CSS 的 RGBA。半透明一律写
// `rgba(r,g,b,a)`，绝不用 8 位 hex 抄语义色 —— 历史上照抄 iOS 的 `#7676801f`
// 被解析成 A=0x76 的一层橄榄绿。
//
// 【命名】INS_ = instrument（仪器）。

/** 页面底色：接近纯黑的冷灰，不是 #000（纯黑下发色容易显脏）。 */
const INS_BG: string = '#0E1116';
/** 卡片/面板底：比页面亮一档，用来切分区域。 */
const INS_PANEL: string = '#171B22';
/** 卡内次级区域（表头、行底纹）。 */
const INS_PANEL2: string = '#1E232B';
/** 主文本：不用纯白，避免长时间盯屏刺眼。 */
const INS_LABEL: string = '#E6EAF2';
/** 次级文本（标签、单位）。 */
const INS_LABEL2: string = 'rgba(230,234,242,0.62)';
/** 三级文本（禁用、占位）。 */
const INS_LABEL3: string = 'rgba(230,234,242,0.34)';
/** 描边与分隔线。 */
const INS_LINE: string = 'rgba(230,234,242,0.10)';

/** 主操作色（深底上的青蓝，比 #007AFF 更亮才压得住）。 */
const INS_ACCENT: string = '#4C9AFF';
/** 主操作色浅底。 */
const INS_ACCENT_SOFT: string = 'rgba(76,154,255,0.16)';

/** 状态色 —— 深底上取高明度版本，保证对比度。 */
/** 通过 / 已落点。 */
const INS_OK: string = '#3DD68C';
const INS_OK_SOFT: string = 'rgba(61,214,140,0.16)';
/** 警告 / 回退（如请求 gpu 但落到 cpu）。 */
const INS_WARN: string = '#F5A623';
const INS_WARN_SOFT: string = 'rgba(245,166,35,0.16)';
/** 失败 / 未复现。 */
const INS_BAD: string = '#FF6B6B';
const INS_BAD_SOFT: string = 'rgba(255,107,107,0.16)';
/** 中性（未跑、跳过）。 */
const INS_IDLE: string = '#8B93A3';

/** 数据字体栈：数字列必须等宽，否则 p50 一列对不齐。 */
const INS_MONO: string = 'HarmonyOS Sans, SF Mono, Menlo, Consolas, monospace';
/** 界面字体栈。 */
const INS_FONT: string = 'HarmonyOS Sans, SF Pro Text, Helvetica Neue, sans-serif';

// ---------------------------------------------------------------- 原生演示配置

/**
 * 三个模型各自的请求后端（见文件头注释的实测依据）。
 *
 * det 用**裸 head** 但落 CPU，是实测后的取舍（ADR-009）：
 *   裸 head 上 NPU 是通的，但同输入同布局下 NPU 与 CPU 仍有 L2 1043.47 vs 1042.66
 *   （0.08%）的差；关键点 = logit×anchor（anchor 最大 433），0.023 的 logit 差经放大
 *   就是约 10 px 偏移 —— 实测把 苏ED5172 认成 苏E05172。矫正对 1 像素都敏感，
 *   所以 det 不吃 NPU；识别模型走 NPU 没问题（argmax+CTC 对数值噪声鲁棒，且更快）。
 * cls 用 fp16 变体时 NPU 构不出 kernel（`Create full model kernel failed`）→ 落 CPU。
 *
 * 实验 E1 结论（2026-09-20，A18 §8）：生产档 det 曾试过 `cpu_t1` 单线程，
 * 流水线内 det 推理 21.6 ms ≈ 四线程 20.4 ms —— **in-pipeline 时间与线程数无关**，
 * "线程池段间睡着付唤醒费"的假设同样被证伪。已回 'cpu'。
 */
const BACKEND_DET: string = 'cpu';
const BACKEND_REC: string = 'nnrt';
const BACKEND_CLS: string = 'cpu';

const MODEL_DET: string = 'models/y5fu_320x_head_fp32.ms';
const MODEL_REC: string = 'models/rpv3_mdict_160_r3.ms';
const MODEL_CLS: string = 'models/litemodel_cls_96x_r1.ms';
/**
 * T2 车辆检测（YOLOv5u/ultralytics，320，80 类 COCO）。
 *
 * 它是**DFL 改写后**才转出来的 .ms：原版 yolov5u 的 DFL 头里有两个
 * `perm != [0,1,3,2]` 的 Transpose，MindSpore Lite 转换器在
 * `/model.24/dfl/conv/Conv` 上直接失败（`Transform meta graph failed! ret=-500`），
 * 连 CPU 的 .ms 都产不出来。改写脚本 `_veh/patch_dfl.py` 把
 * 「转置→softmax→转置→1x1Conv」等价替换成「减最大值→Exp→加权→归约→相除」，
 * 并用 onnxruntime 做了逐元素自证；`tools/scan_onnx.py` 两条 NPU 硬门也随之全过。
 *
 * 后端固定 'cpu'：一期不上 NPU（T8 才做原版 anchor-based 的 NPU 转换）。
 */
const MODEL_VEH: string = 'models/yolov5su_320_veh_fp32.ms';
/**
 * T2 自检图。**必须恰好 320x320**，理由见 `vehicleSelfTest`。
 * 由 `_veh/veh_ref.py --make-image` 从 CCPD 真实道路场景图生成。
 */
const VEH_SAMPLE: string = 'assets/veh320/veh320.png';
const VEH_SIZE: number = 320;

/**
 * 一份**厂商转换器产出的真 `.om`**（来自华为官方 HiAI Foundation codelab，Caffe/SqueezeNet
 * 图像分类，ImageNet 1000 类；`IMOD` 魔数）。它存在的意义：验证「`.ms` 被拒」到底是
 * *格式*问题而不是*通路*问题 —— 如果这份 `.om` 能编译，就证明 CANN 链路的终点可达。
 */
const MODEL_VENDOR_OM: string = 'models/probe_hiai_imagenet.om';

/**
 * **我们自己用 OMG 转出来的 `.om`**（DDK-tools-next-6.0.1.0 + kirin9020 插件）。
 *
 * 与 `MODEL_VENDOR_OM`（厂商现成的 ImageNet 分类模型）的区别：这三份是**本项目自己的模型**，
 * 一旦能编译并跑对，CANN 就从"机制证明"升级为"可用的第四个后端"。
 * 注意 det 是风险最高的那份（anchor-grid decode 含 rank-5 常量，当年正是它让
 * MindSpore Lite 侧的 NPU 建图失败，见 ADR-004 §4.1）—— 但 OMG 是另一套编译器。
 */
const MODEL_OM_DET: string = 'models/probe_om_det.om';
const MODEL_OM_DETHEAD: string = 'models/probe_om_dethead.om';
const MODEL_OM_REC: string = 'models/probe_om_rec.om';
const MODEL_OM_CLS: string = 'models/probe_om_cls.om';

/**
 * **CANN 友好结构候选：LPRNet**（OPMG 自转，`input [1,3,24,94]` → `output [1,68,18]`）。
 *
 * 为什么单独加这一份：按官方《NPU 性能友好计算结构》与算子性能表逐条对照，
 * 我们现有三个生产模型的卷积通道对齐率是 det 70.6% / rec 55.6% / cls **19.2%**，
 * 而 LPRNet 是 **81.2%**（16 个卷积里 13 个 Cin&Cout 均为 16 的倍数）、零 ☆ 差评算子。
 * 官方规则：Cin、Cout 都是 16 的倍数时达 NPU 最大算力；都小于 16 时按 (Cin×Cout)/256 折算。
 * 所以 cls 在 NPU 上比 CPU 慢 26% 是**结构决定的**，不是实现缺陷 —— 这份模型就是用来
 * 验证"换一个 NPU 友好的识别器能不能真的吃到加速"。
 */
const MODEL_OM_LPRNET: string = 'models/probe_om_lprnet.om';
const MODEL_OM_LPRNET_NPUFIX: string = 'models/probe_om_lprnet_npufix.om';

/**
 * T8 · 算子覆盖对撞用的探针名单（51 个）。
 *
 * 与 ShusenPaper 的 `models/onnx/opbench/opbench_manifest.json`（50 个）同源，
 * 外加一个 `convtranspose` —— 那个算子在 `.ms` 侧**根本转不出来**
 * （manifest 的 notes 记着 converter_lite 2.6.0 形状推断失败），却在 CANN 侧
 * 转换成功。这本身就是"支持性是模型 × 工具链的联合属性"的一条直接证据。
 *
 * 三类：
 *   matrix   —— 单算子矩阵（relu / softmax / matmul / reducemean …）
 *   rl_*     —— roofline 通道阶梯（conv k1/k3/k5 × 通道 64/128/256/512、matmul 512/1024/2048）
 *   repro_*  —— 缺陷复现（repro_mislabel = yolov8n 尾部的 Conv→Mul+Sigmoid→Concat 模式）
 *
 * rawfile 里的文件名是 `models/op_<name>.om`。
 */
/**
 * T8 · 算子覆盖对撞的探针名单。
 *
 * 【2026-09-22 改为动态枚举】原先是 51 个名字**硬编码**在这里，理由是
 * `resourceManager` 没有目录枚举 API。这个理由一直不成立：
 * `assetManager.listDir` 就能列 rawfile 目录，实测可用（e2eMatrix 已用
 * `getRawFileList` 列过 crop 目录）。硬编码的代价很实际 ——
 * 往 opbench 加一个算子就得回来改 ArkTS，忘了改就是**静默漏测**：
 * 面板显示"51 个全过"，而磁盘上有 52 个。
 *
 * 现在由 `resolveOpProbes()` 在运行时列 `models/` 下所有 `op_*.om`。
 * 下面这份名单**只作为兜底**：枚举失败（权限/路径/平台差异）时用它，
 * 保证对撞实验永远跑得起来，而不是静默跑 0 个。
 */
const OP_PROBE_NAMES_FALLBACK: string[] = [
  'add_const', 'avgpool', 'batchnorm', 'cast_f16',
  'concat', 'conv1x1', 'conv3x3_s1', 'conv3x3_s2',
  'conv5x5', 'convtranspose', 'div_const', 'dwconv3x3',
  'erf', 'hardswish_explicit', 'leakyrelu', 'matmul_1024',
  'matmul_256', 'max_const', 'maxpool', 'min_const',
  'mul_const', 'pad', 'reducemean', 'reducesum',
  'relu', 'relu6_clip', 'repro_mislabel', 'reshape',
  'resize', 'rl_conv_k1_c128', 'rl_conv_k1_c256', 'rl_conv_k1_c512',
  'rl_conv_k1_c64', 'rl_conv_k3_c128', 'rl_conv_k3_c256', 'rl_conv_k3_c512',
  'rl_conv_k3_c64', 'rl_conv_k5_c256', 'rl_conv_k5_c512', 'rl_matmul_1024',
  'rl_matmul_2048', 'rl_matmul_512', 'sigmoid', 'silu_explicit',
  'slice_channel', 'softmax', 'softplus', 'strided_slice',
  'tanh', 'transpose', 'unsqueeze'
];

/**
 * 运行时名单（枚举成功则由 `resolveOpProbes()` 填入，长度可能 ≠ 51）。
 */
let OP_PROBE_NAMES: string[] = OP_PROBE_NAMES_FALLBACK;

/**
 * 列 `models/` 下所有 `op_*.om`，解析出算子名。
 *
 * 返回 `op_<name>.om` 里的 `<name>`，排序后返回（排序是为了让对撞输出
 * 每次顺序一致 —— 否则两次跑的日志无法逐行 diff）。
 * 失败返回空数组，调用方自行决定兜底。
 */
function scanOpProbeNames(ctx: common.UIAbilityContext): string[] {
  const out: string[] = [];
  try {
    const names: string[] = ctx.resourceManager.getRawFileListSync('models');
    for (let i = 0; i < names.length; i++) {
      const n: string = names[i];
      if (n.indexOf('op_') !== 0 || n.length <= 6) {
        continue;
      }
      if (n.lastIndexOf('.om') !== n.length - 3) {
        continue;
      }
      out.push(n.substring(3, n.length - 3));
    }
  } catch (e) {
    return [];
  }
  out.sort();
  return out;
}

/**
 * ncnn 版本的识别 / 分类模型（GPU 唯一通路；pnnx 从同一份 ONNX 转出）。
 *
 * GPU（Vulkan）在麒麟 8020 上只有 ncnn 一条通路（ADR-008），而 MS Lite 的 GPU 档是
 * 编译期判否 —— 只让检测器吃 GPU 等于「三模型 × 三后端」永远缺一格。
 * 这两份由 pnnx 从**同一份 ONNX** 转出（`tools/` 侧转换，见 A9 证据）。
 */
const NCNN_REC_PARAM: string = 'models/rpv3_mdict_160_r3.ncnn.param';
const NCNN_REC_BIN: string = 'models/rpv3_mdict_160_r3.ncnn.bin';
const NCNN_CLS_PARAM: string = 'models/litemodel_cls_96x_r1.ncnn.param';
const NCNN_CLS_BIN: string = 'models/litemodel_cls_96x_r1.ncnn.bin';
/** ncnn 槽位号：0 保留给检测旁路，1 = 识别，2 = 分类。 */
const SLOT_REC: number = 1;
const SLOT_CLS: number = 2;

/** 可识别的样本（rawfile 路径）。前两张是整车图，后几张是切好的车牌。 */
const NATIVE_SAMPLES: string[] = [
  'assets/samples/hlpr-test.jpg',
  'assets/samples/scene-2.jpg',
  'assets/samples/crop-0-津B6H920.jpg'
];

interface Row {
  model: string;
  backend: string;
}

/**
 * 正式基准矩阵（论文口径）：**三个生产模型** × 四档后端，每项 warmup 5 + 正式 30 次 p50。
 *
 * 与 `E2E_ROLES` 的区别：那个是「功能验证（3 次 + 1 预热）」，本组才是**能给论文用的计时口径**。
 * 模型选型说明：
 *   - 检测用 `y5fu_320x_head_fp32`（裸 head，decode 已移出图外，见 ADR-006）
 *   - 识别用 `rpv3_mdict_160_r3`（NPU 上唯一真正吃到收益的模型）
 *   - 分类用 **fp32** 变体 —— fp16 变体在 NPU 上构不出 kernel（A6）
 * 另附一个 fp16 分类模型只跑 nnrt，用来复现"同结构不同量化 → NPU 是否可建图"的对照。
 */
const FULL_BACKENDS: string[] = ['nnrt', 'gpu', 'kirin', 'cpu'];
const MODELS_FULL: string[] = [
  'models/y5fu_320x_head_fp32.ms',
  'models/rpv3_mdict_160_r3.ms',
  'models/litemodel_cls_96x_r1_fp32.ms'
];
const MODELS_NNRT_GPU: string[] = [
  'models/litemodel_cls_96x_r1.ms'
];
/**
 * CPU 线程扫描档（2026-09-20，threadSweep 用）。与 ms_engine.cpp 的
 * `cpu_t{N}` 分支一一对应：`OH_AI_ContextSetThreadNum` 此前恒传 4 且从未扫描，
 * 论文里所有 NPU/CPU 比值的 CPU 端都是未调优基线（ADR-015 未完成动作）。
 */
const THREAD_BACKENDS: string[] = ['cpu_t1', 'cpu_t2', 'cpu_t4', 'cpu_t6', 'cpu_t8'];

/**
 * CPU **亲和**扫描档（2026-09-21，affinitySweep 用）。对应 ms_engine.cpp 的
 * `cpu_a{N}` 分支（1=大核优先 / 2=小核优先）。
 *
 * 为什么加这个：跑完线程扫描后发现**同一份 det 模型、同一后端、同样 4 线程**，
 * 隔离基准 7.4 ms 而相机流水线内 15.3–19.5 ms（**慢 2 倍**），且**更热的
 * 那一轮反而更快** —— 所以主因不是热降频，而可能是 DVFS/核位调度。
 * `OH_AI_ContextSetThreadAffinityMode` 此前**从未被调用过**，这是验证该假设
 * 最便宜的一刀：不写任何算子代码，只钉核位。
 * 详见 docs/notes/camera-npu-headroom.md §3.2c。
 *
 * 线程数固定 4 —— 已由 THREAD_BACKENDS 的扫描证明 t4 最优（7.42 ms，
 * t6/t8 反而更差 9.9/9.3 ms），所以亲和实验只需在 t4 上做。
 */
const AFFINITY_BACKENDS: string[] = ['cpu_t4', 'cpu_a1', 'cpu_a2'];

/**
 * 【C8 定位实验】间隔/污染/忙等档：`[gapMs, polluteKB, spinMs]`。
 *
 * 要回答的问题：同一模型、同一后端、同样 4 线程，
 * **隔离基准 7.4 ms，相机流水线内 15.3–19.5 ms（慢 2 倍）**。
 *
 * 已排除：热降频、线程数（t4 最优）、亲和绑定（无绑定最快）、
 * 输入打包（0.49 ms）、输出拷贝（p50Io 已含）。
 *
 * 实测（2026-09-21，y5fu_320x，cpu_t4）：
 *   | 档 | p50 |
 *   |---|---|
 *   | 紧循环 | 7.41 ms |
 *   | gap 4 ms | 9.16 ms |
 *   | gap 8 ms | 16.33 ms |
 *   | gap 16 ms | 24.16 ms |
 *   | gap 33 ms | 30.74 ms |
 *   | 污染 1.2 MB（无 gap） | 7.52 ms |
 * **单调剂量-反应**，且污染档**毫无影响** ⇒ 缓存/带宽假设被否掉。
 *
 * 但「gap 越大越慢」仍有两种解释：
 *   (a) **DVFS**：CPU 空闲 → 降频 → 推理跑在低频；
 *   (b) 「两次调用间隔多久」这件事本身有开销。
 * **`spinMs` 就是用来分开它们的**：忙等与睡眠经过时间相同，
 * 但忙等让 CPU 保持忙碌、频率顶住。若自旋档回到 ~7.4 ms ⇒ 是 (a)。
 */
const GAP_TIERS: number[][] = [
  [0, 0, 0],      // 基线：紧循环
  [8, 0, 0],      // 稀疏（中）
  [33, 0, 0],     // 稀疏 = 流水线帧间隔
  [0, 0, 33],     // ★ 对照组：忙等 33 ms（时间相同，但 CPU 不空闲）
  [0, 1200, 0],   // 缓存污染（1.2 MB ≈ conv 的 RGBA 写）
];

/**
 * gap 实验要在哪些后端上跑 —— 这是**定位机理**的关键一刀。
 *
 * 已确证：`gapMs` 单调地把 p50 从 7.3 推到 31.9 ms，而**忙等（spinMs=33）
 * 同样慢（30.4 ms）** ⇒ **DVFS 被否掉**（CPU 忙着，频率没掉，却依然慢）。
 * 污染档无影响 ⇒ 缓存/带宽也被否掉。
 *
 * 剩下的解释：**worker 线程池的唤醒延迟**。MS Lite 的 CPU 后端按
 * `thread_num` 起一个线程池；主线程 sleep 时**池里的工作线程也闲置并 park**，
 * 下次 predict 要先唤醒它们，这段开销落在计时区内。
 * 注意**忙等只占住调用线程本身**，另外 3 个池线程照样 park —— 这正好
 * 解释了「忙等也慢」这个反直觉现象。
 *
 * 判据：若 `cpu_t1`（**单线程，没有池可 park**）上 gap 效应消失或大幅缩小，
 * 则机理是池唤醒延迟；若 t1 上也一样慢，则另有来源。
 */
const GAP_BACKENDS: string[] = ['cpu_t4', 'cpu_t1'];

/**
 * 端到端后端矩阵：逐模型 × 逐后端都跑一遍完整流水线（补后端覆盖）。
 * 一次只替换一个模型，其余保持生产配置 → 任一行差异只能来自被替换者。
 * 判据：`LANDED=`（实际落点，不看请求值；MS 行另带 `req=`/`fallback=`，ncnn 行带
 * `vk=`/`vkLayers=`）+ `match=`（**与 CPU 基线逐字符一致**，不是「读对了」—— 见
 * `REFERENCE_CODE_RPV3` 的注释与 ADR-015）。
 */
const E2E_BACKENDS: string[] = ['cpu', 'nnrt', 'nnrt_fp32', 'nnrt_fp16', 'gpu', 'kirin'];
interface E2eRole {
  role: string;
  file: string;
}
const E2E_ROLES_ALL: E2eRole[] = [
  { role: 'det', file: 'models/y5fu_320x_head_fp32.ms' },
  { role: 'rec', file: 'models/rpv3_mdict_160_r3.ms' },
  { role: 'cls', file: 'models/litemodel_cls_96x_r1.ms' },
  { role: 'cls-fp32', file: 'models/litemodel_cls_96x_r1_fp32.ms' }
];
const E2E_ROLES_CLS: E2eRole[] = [
  { role: 'cls', file: 'models/litemodel_cls_96x_r1.ms' },
  { role: 'cls-fp32', file: 'models/litemodel_cls_96x_r1_fp32.ms' }
];
/** 当前生效的角色集。全量 16 组合在异步化之后已能安全跑完。 */
const E2E_ROLES: E2eRole[] = E2E_ROLES_ALL;

/**
 * 【2026-09-22 删除两个编译期自检开关】
 *
 * 这里原本是 `AUTO_SELFTEST` 与 `AUTO_RQ4` 两个 boolean 常量，用来在装机后
 * **自动**跑一串实验、并把结果写进 hilog。它们存在的原因是当时原生 ArkUI
 * 的按钮点不了（CDP 只能驱动 Web 内容；uitest dumpLayout 又会把 App 推到后台
 * 随即被系统冻结，T13 7 实测 `FROZEN_AFTER_ENTER_BG`），所以只能靠编译期
 * 开关充当代替手指的角色。
 *
 * 现在界面重设计（ADR-0008）给了三个真实可点的入口：「一键全跑」「跑当前组」
 * 「复制」，这两个开关的**唯一理由消失了**。留着它们反而有害：
 *   - 打开 App 就自己跑 20-30 分钟，对一台要交给别人验机的设备是负担；
 *   - 两个互斥开关的正确组合只存在于我脑子里，复现的人一定会配错，
 *     而配错的后果是**静默的**（把热态带进基线、把线程池带进 RQ4 前提）。
 *
 * 需要无人值守长跑时，改用「一键全跑」—— 顺序由 GROUP_ORDER 保证，
 * 不再依赖任何一个编译期常量。
 */
/** 自检用的样本（整车图，能同时验证检测 + 矫正 + 识别 + 分类）。 */
const SELFTEST_SAMPLE: string = 'assets/samples/hlpr-test.jpg';
/**
 * 保真参照串：rpv3 在**默认内部裁剪几何**下对该样本的输出。
 *
 * ⚠️ 它**不是人工标注的真值**。A16 §4.6 已判定该样本的正确真值是 8 字符的
 * `苏ED51712`（车牌被偏航压缩约 2×，标准比例下 rpv3 自己就读 8 字符）—— 旧版本把这个
 * 串叫 `EXPECTED_CODE`/「期望串」，是拿被测系统的输出去验收被测系统，属循环论证。
 *
 * 因此 `match=` 的语义固定为：**与该角色的 CPU 基线逐字符一致**，用于抓「换后端引入的
 * 输出变化」；它**不构成任何正确性/精度证据**。要声称读对了，只能走带人工真值的
 * 第三方真实裁剪图集（`_dataset/real/crops`，n=1000，rpv3 基线 90.6%）。见 ADR-015。
 */
const REFERENCE_CODE_RPV3: string = '苏ED5172';

// ---------------------------------------------------------------- 工具函数

function parseKv(s: string): Map<string, string> {
  const m: Map<string, string> = new Map<string, string>();
  const parts: string[] = s.split(';');
  for (let i = 0; i < parts.length; i++) {
    const p: string = parts[i];
    const eq: number = p.indexOf('=');
    if (eq > 0) {
      m.set(p.substring(0, eq), p.substring(eq + 1));
    }
  }
  return m;
}

function num(m: Map<string, string>, k: string): number {
  const v: string | undefined = m.get(k);
  if (v === undefined || v.length === 0) {
    return 0;
  }
  const f: number = Number.parseFloat(v);
  return Number.isNaN(f) ? 0 : f;
}

function str(m: Map<string, string>, k: string): string {
  return m.get(k) ?? '';
}

/** CameraType 枚举值 -> 人类可读的镜头类型（Camera Kit 的常量顺序）。 */
function lensName(t: number): string {
  if (t === 1) { return '广角'; }
  if (t === 2) { return '超广角'; }
  if (t === 3) { return '长焦'; }
  if (t === 4) { return 'TOF'; }
  if (t === 5) { return '真深度'; }
  return '默认';
}

/** CameraPosition: 0=未指定 1=后置 2=前置 */
function posName(p: number): string {
  if (p === 1) { return '后置'; }
  if (p === 2) { return '前置'; }
  return '未知';
}

/**
 * 牌色标签：native 侧现在是**像素测量**的结论（ADR-0005，2026-09-21）。
 *
 * 这里原本按分类器三个 logits 取最大下标再查表，而那张表是**旋转的**：
 * 源码写 `0→黄牌, 1→蓝牌, else→绿牌`，实测正确顺序是 `blue=0, green=1, yellow=2`。
 * 真机可复现的后果 —— hlpr-test（实为绿牌）显示"蓝牌"、scene-2 的 藏DT5022
 * （蓝牌）显示"黄牌"。现在 native 直接给结论字符串，ArkTS 不做任何查表。
 */
function colourLabel(colour: string): string {
  if (colour === 'blue') { return '蓝牌'; }
  if (colour === 'green') { return '绿牌'; }
  if (colour === 'yellow') { return '黄牌'; }
  return '未知';
}

/** 解码后的 RGBA 帧：pipeline 只吃这个。 */
interface RgbaFrame {
  data: ArrayBuffer;
  width: number;
  height: number;
}

// ------------------------------------------------------------ 结果类型

/** 一条读数。tone: ok / warn / bad / ''（中性）/ sub（次级）/ label（灰标签）。 */
interface ResultRow {
  label: string;
  value: string;
  tone: string;
}

/** 一块实验结果。key 用于 ForEach 去重，tone 作用于分组状态灯。 */
interface ResultGroup {
  key: string;
  title: string;
  tag: string;
  tone: string;
  rows: ResultRow[];
}

/** 原始输出区只保留末尾 N 行 —— 20 分钟热循环有 2000+ 行，全渲染会卡。 */
const RESULT_RAW_TAIL: number = 300;

/** 分组标识。 */
const GRP_RQ3: string = 'rq3';
const GRP_RQ4: string = 'rq4';
const GRP_ACC: string = 'acc';
const GRP_T8: string = 't8';

/**
 * 分组顺序。即 `runAll` 的执行顺序（RQ4 最后）—— 见 ADR-0008「热循环必须收尾」。
 */
const GROUP_ORDER: string[] = [GRP_RQ3, GRP_ACC, GRP_T8, GRP_RQ4];

/**
 * RQ4 热循环时长（分钟）。
 *
 * 20 分钟 / 80 轮，来自 ADR-0007 的 RQ4 口径：原 T7 只有 25 轮 / 6 分钟、thermal
 * 恒为 1，全程没跨过 2→3 的台阶，所以「持续负载下 NPU 收益能否维持」这个问题
 * 实际上从未被测到。热特性要穿过台阶才叫热特性，所以这个值不能改小。
 *
 * 为什么不再用 `AUTO_RQ4` 那个编译期开关：那个开关存在的原因是「原生 ArkUI 按钮
 * 点不了，只能靠 uitest 驱动，而 uitest dumpLayout 会把 App 冻掉」。现在界面
 * 重设计后有了「一键全跑」，「跑 RQ4 组」是面板上的一个点击 —— 开关可以退休了。
 */
const RQ4_MINUTES: number = 20;

/**
 * 从 `key=value;` 两段式载荷里取值。
 *
 * 用**最后一个** `key=` 而不是第一个：`trail=` 这类字段内部可能再次出现
 * 同名 token，取最后一个更接近「外层字段」。
 */
function payloadField(payload: string, key: string): string {
  const needle: string = key + '=';
  const i: number = payload.lastIndexOf(needle);
  if (i < 0) {
    return '';
  }
  const start: number = i + needle.length;
  let j: number = start;
  while (j < payload.length) {
    const c: string = payload.charAt(j);
    if (c === ' ' || c === ';') {
      break;
    }
    j++;
  }
  return payload.substring(start, j);
}

@Entry
@Component
struct Index {
  @State lines: string[] = [];
  @State progress: string = '';
  @State busy: boolean = false;

  // ---- 重设计（ADR-0008）新增状态
  /**
   * 结构化结果。面板的正片；`lines` 降级成"原始输出"副表。
   *
   * 【为什么必须结构化】旧界面只有一坨 `lines`：26 个实验的顺序输出混在一起，
   * 没有分组、没有单位、找不到哪行属于哪个实验 —— 用户的原话是"很多标签看都看不了"。
   * 现在每个探针跑完，`rebuildResults()` 从 lines 里解析出**带标签的读数**。
   * 解析器只搬运不换算，所以面板和原始输出永远对得上。
   */
  @State results: ResultGroup[] = [];
  /** 顶部状态条：''（待机）/ run / ok / warn / bad。 */
  @State runState: string = '';
  /** 当前正在进行的步骤（如「RQ3 · 延迟分解」）。 */
  @State currentExp: string = '';
  /** 已用时，mm:ss / hh:mm:ss。 */
  @State elapsed: string = '';
  /** 当前选中的分组：GRP_RQ3 / GRP_RQ4 / GRP_ACC / GRP_T8。 */
  @State group: string = GRP_RQ3;

  // ---- 生产档落点（原「原生演示页」面板的字段，UI 砍掉后仅作编排用）
  /**
   * 三个生产会话的**实际落点**串（`det=NNRT:... rec=NNRT:...`）。
   * 原先只给"原生演示"页显示，现在没有那个页了，但它仍是 E2E 矩阵的基线声明
   * （backendProbe 把它写进日志头），所以保留。
   */
  @State nBackends: string = '';

  // ---- T7 · RQ4 持续负载热特性
  /** 热态/功耗/负载读数 + 每轮延迟，供离线聚合 */
  @State rq4Rows: string[] = [];
  @State rq4Status: string = '';
  /** 当前热档（0=COOL … 越大越热）。native 侧读不到，只能 ArkTS 取。 */
  @State thermalLevel: number = -1;
  @State batteryTempC: number = 0;

  private ctx: common.UIAbilityContext | undefined = undefined;

  /** 原生会话 id（loadModelAsync 返回），-1 表示尚未加载。 */
  private detId: number = -1;
  private recId: number = -1;
  private clsId: number = -1;
  /**
   * T2 车辆检测会话 id（yolov5su）—— 与上面三个**不是同一批模型**。
   *
   * 车牌检测是 y5fu_320x（3 个 head 输出），车辆检测是 yolov5su（单输出 1x84x2100）。
   * 把 `detId` 当车辆模型用会得到 `yolov5u 期望单输出，实际 3` —— 设备上实测踩过。
   */
  private vehId: number = -1;

  // ---------------------------------------------------------- 页面跳转

  /**
   * 打开相机实时页（T1：把 2026-09-22 砍掉的入口加回**深色仪器风**版本上）。
   *
   * 【为什么用 router 而不是 Navigation】
   * `CameraPage` 是 `@Entry` 页（2140 行，含双路预览 / 权限 / 生命周期全套），
   * 与它在 2026-09-18 建立时的方式一致，也已注册在 `main_pages.json`。
   * 换成 `Navigation` 就得动那页的页面装饰器 —— 超出 T1「先跑通原功能、不改造」
   * 的范围，而风险不小（那页是本项目相机坑位的唯一载体）。
   *
   * 【为什么 catch 只打日志】
   * 相机可用性取决于**设备权限与硬件**（云手机就没有相机）。
   * 跳转失败不该把探针台一起带崩，留一条 hilog 足够定位。
   */
  openCamera(): void {
    router.pushUrl({ url: 'pages/CameraPage' })
      .catch((e: BusinessError) => {
        hilog.error(DOMAIN, TAG, 'pushUrl(pages/CameraPage) 失败: %{public}s',
          JSON.stringify(e));
      });
  }

  private append(s: string): void {
    this.lines = this.lines.concat([s]);
    hilog.info(DOMAIN, TAG, '%{public}s', s);
  }

  private appendNative(s: string): void {
    this.lines = this.lines.concat([s]);
    hilog.info(DOMAIN, NATIVE_TAG, '%{public}s', s);
  }

  private async tick(): Promise<void> {
    await new Promise<void>((resolve: () => void) => {
      setTimeout(() => resolve(), 0);
    });
  }

  private sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve: () => void) => {
      setTimeout(() => resolve(), ms);
    });
  }

  private toArrayBuffer(u: Uint8Array): ArrayBuffer {
    return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
  }

  /** 样本文件名（去掉目录与后缀），列在按钮下方 —— 按钮只标序号，避免长名被截断。 */
  private sampleNames(): string {
    const out: string[] = [];
    for (let i = 0; i < NATIVE_SAMPLES.length; i++) {
      out.push((i + 1).toString() + ' = ' +
        NATIVE_SAMPLES[i].replace('assets/samples/', '').replace('.jpg', ''));
    }
    return out.join('   ');
  }

  // ------------------------------------------------------------ 原生演示

  /**
   * 加载三个模型，逐个指定后端。
   *
   * 关键点：loadModelAsync 的第三个参数决定这个会话落到哪个后端，三个会话互相独立 ——
   * 所以「检测 CPU + 识别 NPU + 分类 CPU」这种混合配置是表达得出来的。
   * 返回串里的 `backend=` 是**实际落点**（可能是回退后的 CPU），不是请求值；
   * 想确认是否真的上了 NPU，只能看这个字段。
   */
  private async loadNative(): Promise<void> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    try {
      const t0: number = Date.now();

      // 会话名一律用**模型文件路径**，不用角色标签 —— 原生侧按 (name|backend) 去重，
      // 用标签会为同一份模型重复建会话，矩阵跑一轮就撞上会话上限。
      const detBytes = c.resourceManager.getRawFileContentSync(MODEL_DET);
      const d: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_DET, this.toArrayBuffer(detBytes), BACKEND_DET));

      const recBytes = c.resourceManager.getRawFileContentSync(MODEL_REC);
      const r: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_REC, this.toArrayBuffer(recBytes), BACKEND_REC));

      const clsBytes = c.resourceManager.getRawFileContentSync(MODEL_CLS);
      const k: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_CLS, this.toArrayBuffer(clsBytes), BACKEND_CLS));

      if (d.get('ok') !== '1' || r.get('ok') !== '1' || k.get('ok') !== '1') {
        this.appendNative('NATIVE LOAD FAIL det=' + str(d, 'error') +
          ' rec=' + str(r, 'error') + ' cls=' + str(k, 'error'));
        return;
      }

      this.detId = num(d, 'id');
      this.recId = num(r, 'id');
      this.clsId = num(k, 'id');

      this.nBackends = 'det=' + str(d, 'backend') + '  rec=' + str(r, 'backend') +
        '  cls=' + str(k, 'backend');
      this.appendNative('NATIVE LOAD ok ' + this.nBackends);
      this.appendNative('NATIVE TRAIL det[' + str(d, 'attemptLog') + '] rec[' +
        str(r, 'attemptLog') + '] cls[' + str(k, 'attemptLog') + ']');
      // 会话重建后落点可能变，立刻反映到面板（否则面板会显示旧的落点）。
      this.rebuildResults();
    } catch (e) {
      const err = e as Error;
      this.appendNative('NATIVE LOAD EXCEPTION ' + err.message);
    }
  }

  /** 把 rawfile 里的 JPEG 解成 RGBA —— pipeline 只吃 RGBA 缓冲。 */
  private async decodeRgba(file: string): Promise<RgbaFrame | null> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return null;
    }
    const bytes: Uint8Array = c.resourceManager.getRawFileContentSync(file);
    const src: image.ImageSource = image.createImageSource(this.toArrayBuffer(bytes));
    const pm: image.PixelMap = await src.createPixelMap();
    const info: image.ImageInfo = await pm.getImageInfo();
    const w: number = info.size.width;
    const h: number = info.size.height;
    const rgba: ArrayBuffer = new ArrayBuffer(w * h * 4);
    await pm.readPixelsToBuffer(rgba);
    await pm.release();
    await src.release();
    this.appendNative('NATIVE DECODE ' + file + ' ' + w + 'x' + h);
    return { data: rgba, width: w, height: h };
  }

  /**
   * T2 车辆检测自检：把 rawfile 里那张测试图喂给车辆检测器，把结果留在屏上。
   *
   * ## 为什么测试图**必须**恰好 320x320
   *
   * `LprVehicleDetect` 会先把源图 letterbox 到 320x320。源图若本身就是 320x320，
   * 则 `r = min(320/320, 320/320) = 1`、`left = top = 0`，letterbox 退化成
   * **逐字节拷贝** —— 于是 PC 侧参考实现（`_veh/veh_ref.py`，跑 onnxruntime）
   * 与真机看到的模型输入是同一串字节。
   *
   * 这一条不是洁癖：T2 的验收是「PC 侧与 native 侧在同一张图上结果一致」。
   * 若源图尺寸不等于 320，两侧就要各自做一次插值缩放，而两边的插值实现
   * （native 的手写双线性 vs PIL/Pillow）本来就不会逐位相同，于是"结果不一致"
   * 会变成一个分不清是解码写错、还是插值差异的糊涂账。把 resize 从等式里消掉，
   * 剩下的差异就只可能来自解码。
   *
   * 所以这里**硬校验尺寸并拒绝**，而不是"差不多就行"。
   */
  private async vehicleSelfTest(): Promise<void> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined || this.busy) {
      return;
    }
    this.busy = true;
    this.progress = '车辆检测自检';
    try {
      // 1) 载入模型。会话按 (rawfile|后端) 缓存，重复点不会重建；
      //    这里显式传 'cpu'，一期不上 NPU。
      const bytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_VEH);
      const loadRaw: string = await lpr.loadModelAsync(MODEL_VEH, this.toArrayBuffer(bytes), 'cpu');
      const lk: Map<string, string> = parseKv(loadRaw);
      if (lk.get('ok') !== '1') {
        this.appendNative('VEH LOAD FAIL ' + loadRaw);
        return;
      }
      const id: number = num(lk, 'id');
      this.vehId = id;  // 记住会话 id：ROI 探针要用同一个车辆检测器
      this.appendNative('VEH LOAD ok id=' + id.toString() +
        ' LANDED=' + str(lk, 'backend') + ' req=' + str(lk, 'requested') +
        ' fallback=' + str(lk, 'fallbackFrom') + ' trail=[' + str(lk, 'attemptLog') + ']');

      // 2) 测试图 -> RGBA
      const frame: RgbaFrame | null = await this.decodeRgba(VEH_SAMPLE);
      if (frame === null) {
        this.appendNative('VEH FAIL 解不出测试图 ' + VEH_SAMPLE);
        return;
      }
      if (frame.width !== VEH_SIZE || frame.height !== VEH_SIZE) {
        this.appendNative('VEH FAIL 测试图必须是 ' + VEH_SIZE.toString() + 'x' +
          VEH_SIZE.toString() + '，实际 ' + frame.width.toString() + 'x' +
          frame.height.toString() + ' —— letterbox 会真的插值，与 PC 参考不再是同一份输入');
        return;
      }

      // 3) 预热一次再读结果：首跑含缓冲分配与首次触碰页错误，
      //    计进去会把冷启动误当成后端/实现差异（与 e2eMatrix 同一口径）。
      await lpr.vehicleDetectAsync(id, frame.data, frame.width, frame.height);
      const raw: string = await lpr.vehicleDetectAsync(id, frame.data, frame.width, frame.height);
      const kv: Map<string, string> = parseKv(raw);
      if (kv.get('ok') !== '1') {
        this.appendNative('VEH FAIL ' + raw);
        return;
      }
      const n: number = num(kv, 'count');
      this.appendNative('VEH RESULT ' + raw);
      this.appendNative('T2VEH count=' + n.toString() +
        ' truncated=' + str(kv, 'truncated') +
        ' conf=' + str(kv, 'conf') + ' iou=' + str(kv, 'iou') +
        ' vehicleOnly=' + str(kv, 'vehicleOnly') +
        ' size=' + str(kv, 'size') + ' nhwc=' + str(kv, 'nhwc') +
        ' inferMs=' + str(kv, 'inferMs') + ' totalMs=' + str(kv, 'totalMs') +
        ' LANDED=' + str(kv, 'backend'));
      // 逐框单独一行，便于在屏上核对与复制
      for (let i = 0; i < n; i++) {
        const f: string[] = str(kv, 'b' + i.toString()).split(',');
        if (f.length < 4) {
          continue;
        }
        this.appendNative('T2BOX idx=' + i.toString() + ' cls=' + f[0] + ' score=' + f[1] +
          ' rect=' + f[2] + ' name=' + f[3]);
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('VEH EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
      this.progress = '';
    }
  }

  /**
   * T3 自证：ROI 裁剪 + 坐标映射。**不需要模型**，测的是纯几何与逐字节裁剪。
   *
   * 复用 T2 那张 320x320 测试图：它是真实素材、尺寸已知，于是报告里的 `rgbSum`
   * 可以在 PC 侧拿**同一张 PNG** 独立重算 —— 构成「设备裁出来的那块区域
   * == PC 算出来的那块区域」的跨实现证据（不是同一条代码自说自话）。
   *
   * 报告是多行文本，逐行回显到屏上（也是逐行写 hilog）。
   */
  private async roiSelfTest(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.progress = 'ROI 自证';
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(VEH_SAMPLE);
      if (frame === null) {
        this.appendNative('T3ROI FAIL 解不出测试图 ' + VEH_SAMPLE);
        return;
      }
      const raw: string = await lpr.roiSelfTestAsync(frame.data, frame.width, frame.height);
      const lines: string[] = raw.split('\n');
      let verdict: string = 'total=?;failed=?';
      for (let i = 0; i < lines.length; i++) {
        const ln: string = lines[i].trim();
        if (ln.length === 0) {
          continue;
        }
        this.appendNative('T3ROI ' + ln);
        if (ln.indexOf('failed=') >= 0) {
          verdict = ln;
        }
      }
      this.appendNative('T3ROI VERDICT ' + verdict);
    } catch (e) {
      const err = e as Error;
      this.appendNative('T3ROI EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
      this.progress = '';
    }
  }

  /**
   * T3 探针：车框 → 裁 ROI → 车牌检测 → **映射回原图坐标**。
   *
   * 只做分数最高的那一个车框（遍历所有框 + 合并去重是 T4）。要证的是一件事：
   * ROI 路径映射回来的车牌框，必须与整图直检出来的框落在**同一处**。
   * 判据是 IoU，逐框打印 —— 不靠人眼看图。
   */
  private async roiPlateProbe(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.progress = 'ROI 探针';
    try {
      // 车辆检测器与车牌三检测器是**两批模型**，ensureAllSessions 负责都备齐。
      const ready: boolean = await this.ensureAllSessions();
      if (!ready) {
        return;
      }
      const frame: RgbaFrame | null = await this.decodeRgba(VEH_SAMPLE);
      if (frame === null) {
        this.appendNative('T3PROBE FAIL 解不出测试图 ' + VEH_SAMPLE);
        return;
      }
      // 探两个框：0 = 最高分（这一张图上它贴着左边缘，ROI 被 clamp 成 x0=0），
      // 4 = 一个 x0>0 的框。**只探 0 的话，x 方向的映射等于没被验证** ——
      // "忘了加 x0"与"映射正确"在 x0=0 时结果完全一样。
      const n0: number = await this.roiProbeOne(frame, 0, true);
      if (n0 > 4) {
        await this.roiProbeOne(frame, 4, false);
      } else {
        this.appendNative('T3PROBE 车辆框只有 ' + n0.toString() +
          ' 个，这一张图里没有 x0>0 的框可探（x 方向映射未被区分）');
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('T3PROBE EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
      this.progress = '';
    }
  }

  /**
   * ROI 探针的单次执行（探第 `boxIdx` 个车辆框）。返回车辆框总数，失败返回 -1。
   *
   * `logDirect` 为假时不重复打印直检框 —— 直检与选哪个车框无关，
   * 每次探针都一模一样，重复打印只是噪声。
   */
  private async roiProbeOne(frame: RgbaFrame, boxIdx: number,
    logDirect: boolean): Promise<number> {
    const raw: string = await lpr.roiPlateProbeAsync(this.vehId, this.detId, this.recId,
      this.clsId, frame.data, frame.width, frame.height, boxIdx);
    const kv: Map<string, string> = parseKv(raw);
    if (kv.get('ok') !== '1') {
      this.appendNative('T3PROBE FAIL boxIdx=' + boxIdx.toString() + ' ' + raw);
      return -1;
    }
    this.appendNative('T3PROBE boxIdx=' + str(kv, 'boxIdx') +
      ' direct=' + str(kv, 'directCount') + ' veh=' + str(kv, 'vehCount') +
      ' roi=' + str(kv, 'roiX0') + ',' + str(kv, 'roiY0') + ',' + str(kv, 'roiW') + ',' +
      str(kv, 'roiH') + ' clamped=' + str(kv, 'roiClamped') +
      ' coversBox=' + str(kv, 'roiCoversBox') + ' expand=' + str(kv, 'expand') +
      ' roiPlates=' + str(kv, 'roiCount') + ' directMs=' + str(kv, 'directMs') +
      ' roiMs=' + str(kv, 'roiMs'));
    const nr: number = num(kv, 'roiCount');
    for (let i = 0; i < nr; i++) {
      const f: string[] = str(kv, 'r' + i.toString()).split(',');
      if (f.length < 5) {
        continue;
      }
      this.appendNative('T3ROIBOX boxIdx=' + str(kv, 'boxIdx') + ' idx=' + i.toString() +
        ' rect=' + f[0] + ' score=' + f[1] + ' directJ=' + f[2] + ' iou=' + f[3] +
        ' code=' + f[4]);
    }
    if (logDirect) {
      const nd: number = num(kv, 'directCount');
      for (let j = 0; j < nd; j++) {
        const f: string[] = str(kv, 'd' + j.toString()).split(',');
        if (f.length < 3) {
          continue;
        }
        this.appendNative('T3DIRECTBOX idx=' + j.toString() + ' rect=' + f[0] +
          ' score=' + f[1] + ' code=' + f[2]);
      }
    }
    return num(kv, 'vehCount');
  }

  /**
   * 确保四个会话都在：车牌三会话（det / rec / cls）+ **车辆检测会话（veh）**。
   *
   * 分开写成一个方法，是因为 T3 探针与 T4 路径都要这一步；两处各写一遍必然漂移。
   * 返回 false 表示有会话不可用（原因已经 append 到屏上）。
   */
  private async ensureAllSessions(): Promise<boolean> {
    if (this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      this.appendNative('会话未加载，先加载车牌三模型…');
      await this.loadNative();
    }
    if (this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      this.appendNative('FAIL 车牌会话不可用 det=' + this.detId.toString() +
        ' rec=' + this.recId.toString() + ' cls=' + this.clsId.toString());
      return false;
    }
    if (this.vehId < 0) {
      this.appendNative('车辆检测会话未加载，先加载 ' + MODEL_VEH + ' …');
      const c: common.UIAbilityContext | undefined = this.ctx;
      if (c === undefined) {
        this.appendNative('FAIL 无 UIAbilityContext');
        return false;
      }
      const bytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_VEH);
      const loadRaw: string = await lpr.loadModelAsync(MODEL_VEH, this.toArrayBuffer(bytes),
        'cpu');
      const lk: Map<string, string> = parseKv(loadRaw);
      if (lk.get('ok') !== '1') {
        this.appendNative('FAIL 车辆模型加载失败 ' + loadRaw);
        return false;
      }
      this.vehId = num(lk, 'id');
    }
    return true;
  }

  /**
   * T4：ROI 路径端到端 —— 车辆检测 → **逐框**裁 ROI → 逐框车牌检测 → 映射回原图
   * → 合并去重，输出车牌框 + 车牌串 + 颜色 + 归属的车辆。
   *
   * 与「ROI 探针」的区别：探针只做**最高分那一个**车框、且不去重；这里遍历所有车框
   * 并合并去重 —— 也就是 spec D1 + D3 要求的完整路径。
   */
  private async roiPipelineRun(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.progress = 'ROI 路径';
    try {
      const ready: boolean = await this.ensureAllSessions();
      if (!ready) {
        return;
      }
      const frame: RgbaFrame | null = await this.decodeRgba(VEH_SAMPLE);
      if (frame === null) {
        this.appendNative('T4PIPE FAIL 解不出测试图 ' + VEH_SAMPLE);
        return;
      }
      const raw: string = await lpr.roiPipelineAsync(this.vehId, this.detId, this.recId,
        this.clsId, frame.data, frame.width, frame.height);
      const kv: Map<string, string> = parseKv(raw);
      if (kv.get('ok') !== '1') {
        this.appendNative('T4PIPE FAIL ' + raw);
        return;
      }
      this.appendNative('T4PIPE veh=' + str(kv, 'vehCount') +
        ' truncated=' + str(kv, 'vehTruncated') +
        ' roiTried=' + str(kv, 'roiTried') + ' roiSkipped=' + str(kv, 'roiSkipped') +
        ' rawHits=' + str(kv, 'rawHits') + ' dropped=' + str(kv, 'dedupeDropped') +
        ' count=' + str(kv, 'count') +
        ' vehInferMs=' + str(kv, 'vehInferMs') + ' roiDetectMs=' + str(kv, 'roiDetectMs') +
        ' totalMs=' + str(kv, 'totalMs'));
      const nv: number = num(kv, 'vehCount');
      for (let i = 0; i < nv; i++) {
        const f: string[] = str(kv, 'v' + i.toString()).split(',');
        if (f.length < 4) {
          continue;
        }
        this.appendNative('T4VEH idx=' + i.toString() + ' cls=' + f[0] + ' score=' + f[1] +
          ' rect=' + f[2] + ' name=' + f[3]);
      }
      const np: number = num(kv, 'count');
      for (let i = 0; i < np; i++) {
        const f: string[] = str(kv, 'p' + i.toString()).split(',');
        if (f.length < 6) {
          continue;
        }
        this.appendNative('T4PLATE idx=' + i.toString() + ' rect=' + f[0] + ' score=' + f[1] +
          ' owner=' + f[2] + ' colour=' + f[3] + ' code=' + f[4] + ' recConf=' + f[5]);
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('T4PIPE EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
      this.progress = '';
    }
  }

  /**
   * T4：合并去重的单元自证（纯数据，不需要模型）。
   *
   * 构造完全重合 / IoU 0.667 / IoU 0.333 / 链式重叠 / 边界相接 / 空输入等已知输入，
   * 断言保留条数与**保留的是哪一条**。这是「构造重叠车框验证去重生效」的落地形式：
   * 重叠车框的后果就是同一块牌被检出两次，这里直接把那种输入喂给去重函数。
   */
  private async roiDedupeSelfTest(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.progress = '去重自证';
    try {
      // 1) 纯数据单元用例
      const raw: string = await lpr.roiDedupeSelfTestAsync();
      const lines: string[] = raw.split('\n');
      let verdict: string = 'total=?;failed=?';
      for (let i = 0; i < lines.length; i++) {
        const ln: string = lines[i].trim();
        if (ln.length === 0) {
          continue;
        }
        this.appendNative('T4DEDUPE ' + ln);
        if (ln.indexOf('failed=') >= 0) {
          verdict = ln;
        }
      }
      this.appendNative('T4DEDUPE VERDICT ' + verdict);

      // 2) 构造重叠车框的集成验证（票面原文要的就是这一条，单元用例只是它的近亲）
      const ready: boolean = await this.ensureAllSessions();
      if (!ready) {
        this.appendNative('T4OVERLAP SKIP 会话不可用，构造重叠车框验证未跑');
        return;
      }
      const frame: RgbaFrame | null = await this.decodeRgba(VEH_SAMPLE);
      if (frame === null) {
        this.appendNative('T4OVERLAP SKIP 解不出测试图 ' + VEH_SAMPLE);
        return;
      }
      const oRaw: string = await lpr.roiOverlapSelfTestAsync(this.vehId, this.detId,
        this.recId, this.clsId, frame.data, frame.width, frame.height);
      const oLines: string[] = oRaw.split('\n');
      for (let i = 0; i < oLines.length; i++) {
        const ln: string = oLines[i].trim();
        if (ln.length === 0) {
          continue;
        }
        this.appendNative('T4OVERLAP ' + ln);
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('T4DEDUPE EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
      this.progress = '';
    }
  }

  /**
   * 逐模型 × 逐后端的端到端矩阵（补后端覆盖）。
   *
   * 一次只替换一个模型的后端，其余两个保持生产配置 → 任一行的差异只能来自被替换者。
   * 请求值不算数，只看 `LANDED=`（实际落点）与 `match=`（与 CPU 基线逐字符一致）。
   * 每跑完一个组合就 appendLine 落盘：hilog 是环形缓冲，被杀之后结果只能靠运气留存。
   */
  private async e2eMatrix(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined || this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      this.appendNative('E2EMATRIX FAIL missing sessions');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/matrix.log';
    this.appendNative('E2EMATRIX BEGIN reference=' + REFERENCE_CODE_RPV3 +
      ' prod=det:cpu rec:nnrt cls:cpu pairs=' +
      (E2E_ROLES.length * E2E_BACKENDS.length).toString() + ' log=' + logPath);
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(SELFTEST_SAMPLE);
      if (frame === null) {
        this.appendNative('E2EMATRIX FAIL no frame');
        return;
      }
      for (let r = 0; r < E2E_ROLES.length; r++) {
        const role: string = E2E_ROLES[r].role;
        const file: string = E2E_ROLES[r].file;
        let bytes: Uint8Array;
        try {
          bytes = ctx.resourceManager.getRawFileContentSync(file);
        } catch (e) {
          this.appendNative('E2EMATRIX ' + role + ' MISSING-RAWFILE ' + file);
          continue;
        }
        for (let b = 0; b < E2E_BACKENDS.length; b++) {
          const be: string = E2E_BACKENDS[b];
          const loadRaw: string =
            await lpr.loadModelAsync(file, this.toArrayBuffer(bytes), be);
          const li: Map<string, string> = parseKv(loadRaw);
          lpr.appendLine(logPath, 'E2EMATRIX LOAD role=' + role + ' backend=' + be + ' ' + loadRaw);
          if (li.get('ok') !== '1') {
            this.appendNative('E2EMATRIX ' + role + ' ' + be + ' BUILD-FAIL err=' +
              str(li, 'error') + ' trail=' + str(li, 'attemptLog'));
            continue;
          }
          const id: number = num(li, 'id');
          const landed: string = str(li, 'backend');
          const trail: string = str(li, 'attemptLog');
          let detId: number = this.detId;
          let recId: number = this.recId;
          let clsId: number = this.clsId;
          // cls-fp32 是「同一角色的另一份模型文件」，槽位仍是 cls
          const slot: string = role.indexOf('cls') === 0 ? 'cls' : role;
          if (slot === 'det') {
            detId = id;
          } else if (slot === 'rec') {
            recId = id;
          } else {
            clsId = id;
          }
          // 预热一次再计时：首跑含缓冲分配/首访页错误，计进去会把冷启动当成后端差异
          await lpr.pipelineAsync(frame.data, frame.width, frame.height, detId, recId, clsId, 0);
          for (let rep = 0; rep < 3; rep++) {
            const pr: string = await lpr.pipelineAsync(
              frame.data, frame.width, frame.height, detId, recId, clsId, 0);
            const kv: Map<string, string> = parseKv(pr);
            const f: string[] = str(kv, 'p0').split(',');
            const code: string = f[0];
            const stages: string = f.length > 9 ? f[9] : '-';
            const line: string = 'E2EMATRIX RESULT role=' + role + ' backend=' + be +
              ' LANDED=' + landed + ' engine=ms req=' + str(li, 'requested') +
              ' fallback=' + str(li, 'fallbackFrom') + ' rep=' + rep +
              ' match=' + (kv.get('ok') === '1' && code === REFERENCE_CODE_RPV3 ? '1' : '0') +
              ' code=' + code + ' total=' + str(kv, 'totalMs') +
              ' stages=' + stages + ' trail=' + trail;
            this.appendNative(line);
            lpr.appendLine(logPath, line);
          }
        }
      }
      this.appendNative('E2EMATRIX END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('E2EMATRIX EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  // ------------------------------------------------------------ 识别器后端 A/B（批量）

  /**
   * 【T10】真实数据集 × 识别器落点 A/B。
   *
   * 要回答的问题：`det-backend-alters-result.md` 证明了「det 落点会改变识别输出」，
   * 但那是 **n=1**（只有参考图一张），笔记自己把「多图统计」列为待办 1。
   * 本探针把样本量提到整个真实数据集，测的是**同一问题的识别器版本**：
   * rec 从 NPU 换到 CPU，输出改变的比例是多少、各后端准确率是多少。
   *
   * 为什么用裁剪图：`_dataset/real/crops` 是 1000 张 94x24 的**已裁剪**车牌小图
   * （`t6-balanced-evaluation.md` 附带限制已声明「不含检测环节」），
   * 正好是识别器的输入形态。det/cls 固定为生产配置，**唯一变量是 rec 的落点**。
   *
   * 为什么读 rawfile 而不是 filesDir：先试过 filesDir，**走不通** ——
   * App 私有目录既不能 `hdc shell` 写也不能 `hdc file send` 写（两者都是 shell uid，
   * 实测均 permission denied），数据集送不进去。1000 张小图打进 rawfile 的代价是
   * HAP +2.5 MB、构建略慢，但换来「零权限、随包装走、可复现」——
   * 对一份要写进论文的数据集，后者更重要。
   * （T9 把「离线路径只读 rawfile」记为缺口；本探针证明这个缺口的解法就是打包。）
   *
   * 真值口径：**文件名即真值**（CONTEXT.md：只有 `_dataset/real/crops/` 的文件名
   * 是真值）。长度非 7/8 的直接跳过并计数 —— t6 踩过「文件名被工具改过、
   * 凭空制造 29 条长度错误」的坑，这里必须先验真值本身干不干净。
   *
   * 判据：逐张记 `code_cpu` / `code_np` / `agree` / 两个后端各自的 `ok_cpu` `ok_np`，
   * 全部落盘到 `filesDir/crop_ab_<role>.log`，最后由主机侧脚本汇总。
   *
   * @param role 变哪个角色的落点：`'rec'`（识别器）或 `'det'`（检测器）。
   *   其余两个角色固定为生产配置。T10 已跑 rec（0.3% 不一致）；
   *   `det` 是同一问题的检测器版本，用来与 `det-backend-alters-result.md` 的
   *   n=1 结论对上——**注意样本量仍不可比**（那边是 n=1，这边是 n=1000，
   *   且输入是裁剪图而非整车图，判det的工况不同）。
   */
  private async cropBackendAB(role: string): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('CROPAB FAIL no context');
      return;
    }
    if (this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      await this.loadNative();
      if (this.detId < 0) {
        this.appendNative('CROPAB FAIL missing sessions');
        return;
      }
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/crop_ab_' + role + '.log';
    // rawfile 里的目录（相对 rawfile 根，不带前导斜杠）。
    const cropDir: string = 'assets/crops';
    this.appendNative('CROPAB BEGIN role=' + role + ' dir=' + cropDir + ' log=' + logPath);
    try {
      // 列出数据集。目录不存在要说清楚，而不是静默跑 0 张。
      // ⚠️ getRawFileList 在本 API 版本同样返回 **Promise<string[]>**（不是同步）——
      // 与 fileIo.listFile 一致。直接当数组用在编译期报「missing length/pop/push」。
      let names: string[] = [];
      try {
        names = await ctx.resourceManager.getRawFileList(cropDir);
      } catch (e) {
        const err = e as BusinessError;
        this.appendNative('CROPAB FAIL listdir code=' + err.code + ' msg=' + err.message +
          ' —— rawfile 里没有 ' + cropDir + '（打包漏了？）');
        this.busy = false;
        return;
      }
      names = names.filter((n: string) => n.toLowerCase().endsWith('.jpg')).sort();
      if (names.length === 0) {
        this.appendNative('CROPAB FAIL 目录里没有 .jpg');
        this.busy = false;
        return;
      }
      this.appendNative('CROPAB FILES ' + names.length.toString());

      // 载入**可变角色**的两个落点；其余两个角色用生产会话（不变），
      // 保证唯一变量就是 role 的落点。
      const roleFile: string = role === 'det' ? MODEL_DET : MODEL_REC;
      const roleBytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(roleFile);
      const loadCpu: string =
        await lpr.loadModelAsync(roleFile, this.toArrayBuffer(roleBytes), 'cpu');
      const loadNp: string =
        await lpr.loadModelAsync(roleFile, this.toArrayBuffer(roleBytes), 'nnrt');
      const ci: Map<string, string> = parseKv(loadCpu);
      const ni: Map<string, string> = parseKv(loadNp);
      lpr.appendLine(logPath, 'CROPAB LOAD ' + role + ' cpu ' + loadCpu);
      lpr.appendLine(logPath, 'CROPAB LOAD ' + role + ' nnrt ' + loadNp);
      if (ci.get('ok') !== '1' || ni.get('ok') !== '1') {
        this.appendNative('CROPAB FAIL ' + role + ' load cpu_ok=' + str(ci, 'ok') +
          ' np_ok=' + str(ni, 'ok'));
        this.busy = false;
        return;
      }
      const cpuId: number = num(ci, 'id');
      const npId: number = num(ni, 'id');
      this.appendNative('CROPAB SESSIONS role=' + role +
        ' det=' + this.detId.toString() + ' rec=' + this.recId.toString() +
        ' cls=' + this.clsId.toString() + ' cpu_id=' + cpuId.toString() +
        ' np_id=' + npId.toString());

      let n: number = 0;
      let agree: number = 0;
      let badGt: number = 0;
      let okCpu: number = 0;
      let okNp: number = 0;
      for (let i = 0; i < names.length; i++) {
        const name: string = names[i];
        // 真值 = 文件名去扩展名；带 `_1234` 下标后缀的（t6 首轮抽取产物）也剥掉。
        const gt: string = name.replace(/\.jpg$/i, '').split('_')[0];
        if (gt.length !== 7 && gt.length !== 8) {
          // 真值本身不干净 —— 跳过并计数，绝不静默计错。
          badGt++;
          lpr.appendLine(logPath, 'CROPAB SKIP file=' + name + ' gt_len=' + gt.length.toString());
          continue;
        }
        const path: string = cropDir + '/' + name;
        let frame: RgbaFrame | null = null;
        try {
          // rawfile 路径直接走既有的 decodeRgba（getRawFileContentSync）。
          frame = await this.decodeRgba(path);
        } catch (e) {
          const err = e as BusinessError;
          lpr.appendLine(logPath, 'CROPAB DECODEFAIL file=' + name + ' code=' + err.code);
          continue;
        }
        if (frame === null) {
          continue;
        }
        // 同一张图、同样的其余两个角色，只换 role 的落点。
        // 用局部变量而不是三元塞进实参列表：塞进去可读性差，且容易传错顺序。
        const dCpu: number = role === 'det' ? cpuId : this.detId;
        const dNp: number = role === 'det' ? npId : this.detId;
        const rCpu: number = role === 'rec' ? cpuId : this.recId;
        const rNp: number = role === 'rec' ? npId : this.recId;
        const rawCpu: string = await lpr.pipelineAsync(
          frame.data, frame.width, frame.height, dCpu, rCpu, this.clsId, 0);
        const rawNp: string = await lpr.pipelineAsync(
          frame.data, frame.width, frame.height, dNp, rNp, this.clsId, 0);
        const kc: Map<string, string> = parseKv(rawCpu);
        const kn: Map<string, string> = parseKv(rawNp);
        const cc: string = str(kc, 'p0').split(',')[0];
        const cn: string = str(kn, 'p0').split(',')[0];
        const same: boolean = cc === cn;
        n++;
        if (same) {
          agree++;
        }
        if (cc === gt) {
          okCpu++;
        }
        if (cn === gt) {
          okNp++;
        }
        lpr.appendLine(logPath, 'CROPAB role=' + role + ' file=' + name + ' gt=' + gt +
          ' code_cpu=' + cc + ' code_np=' + cn +
          ' agree=' + (same ? '1' : '0') +
          ' ok_cpu=' + (cc === gt ? '1' : '0') +
          ' ok_np=' + (cn === gt ? '1' : '0'));
        // 每 100 张报一次进度：长跑没反馈看不出是卡了还是在算。
        if (n % 100 === 0) {
          this.appendNative('CROPAB PROGRESS n=' + n.toString() +
            ' agree=' + agree.toString() + ' ok_cpu=' + okCpu.toString() +
            ' ok_np=' + okNp.toString());
        }
      }
      this.appendNative('CROPAB END role=' + role + ' n=' + n.toString() +
        ' agree=' + agree.toString() +
        ' ok_cpu=' + okCpu.toString() + ' ok_np=' + okNp.toString() +
        ' bad_gt=' + badGt.toString() + ' log=' + logPath);
    } finally {
      this.busy = false;
    }
  }

  // ------------------------------------------------------------ 识别器裸测准确率（T11）

  /**
   * 【T11】识别器**真实**准确率：跳过 det/rectify，把裁剪图直喂识别器。
   *
   * 为什么需要它：此前所有准确率数字都有口径问题 ——
   *   - t6 / A16 的 90.6%：**主机** onnxruntime 测的，从没在端侧验过；
   *   - T10 的 60.8%：端侧，但走**完整流水线**，det 要在一张 94x24 的
   *     已裁剪小图上重新找牌，输入被劣化（长度错误率 25.8% vs A16 实测
   *     rpv3 的 1.0%），且把检测器误差和识别器误差混在一起。
   *
   * 本探针用新增的 napi `recognise`（跳过 det/rectify 直喂 rec），
   * 与 t6/A16 的工况**一致**，因此数字可直接对比。
   *
   * 同时跑两个落点（cpu / nnrt）：既给各自准确率，也给一致率。
   * 判据：逐张记 code_cpu / code_np / agree / ok_*，落盘 crop_bare.log。
   */
  private async bareHeadAccuracy(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('BARE FAIL no context');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/crop_bare.log';
    const cropDir: string = 'assets/crops';
    this.appendNative('BARE BEGIN dir=' + cropDir + ' log=' + logPath);
    try {
      let names: string[] = [];
      try {
        names = await ctx.resourceManager.getRawFileList(cropDir);
      } catch (e) {
        const err = e as BusinessError;
        this.appendNative('BARE FAIL listdir code=' + err.code + ' msg=' + err.message);
        this.busy = false;
        return;
      }
      names = names.filter((n: string) => n.toLowerCase().endsWith('.jpg')).sort();
      if (names.length === 0) {
        this.appendNative('BARE FAIL 目录里没有 .jpg');
        this.busy = false;
        return;
      }
      this.appendNative('BARE FILES ' + names.length.toString());

      // 两个 rec 会话：cpu / nnrt。det/cls 完全不参与。
      const recBytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(MODEL_REC);
      const loadCpu: string =
        await lpr.loadModelAsync(MODEL_REC, this.toArrayBuffer(recBytes), 'cpu');
      const loadNp: string =
        await lpr.loadModelAsync(MODEL_REC, this.toArrayBuffer(recBytes), 'nnrt');
      const ci: Map<string, string> = parseKv(loadCpu);
      const ni: Map<string, string> = parseKv(loadNp);
      lpr.appendLine(logPath, 'BARE LOAD cpu ' + loadCpu);
      lpr.appendLine(logPath, 'BARE LOAD nnrt ' + loadNp);
      if (ci.get('ok') !== '1' || ni.get('ok') !== '1') {
        this.appendNative('BARE FAIL load cpu_ok=' + str(ci, 'ok') + ' np_ok=' + str(ni, 'ok'));
        this.busy = false;
        return;
      }
      const recCpuId: number = num(ci, 'id');
      const recNpId: number = num(ni, 'id');
      this.appendNative('BARE SESSIONS rec_cpu=' + recCpuId.toString() +
        ' rec_np=' + recNpId.toString());

      let n: number = 0;
      let agree: number = 0;
      let badGt: number = 0;
      let okCpu: number = 0;
      let okNp: number = 0;
      let lenCpu: number = 0;
      let lenNp: number = 0;
      for (let i = 0; i < names.length; i++) {
        const name: string = names[i];
        const gt: string = name.replace(/\.jpg$/i, '').split('_')[0];
        if (gt.length !== 7 && gt.length !== 8) {
          badGt++;
          lpr.appendLine(logPath, 'BARE SKIP file=' + name + ' gt_len=' + gt.length.toString());
          continue;
        }
        const path: string = cropDir + '/' + name;
        let frame: RgbaFrame | null = null;
        try {
          frame = await this.decodeRgba(path);
        } catch (e) {
          const err = e as BusinessError;
          lpr.appendLine(logPath, 'BARE DECODEFAIL file=' + name + ' code=' + err.code);
          continue;
        }
        if (frame === null) {
          continue;
        }
        // 同一张裁剪图，只换 rec 落点；**不走 pipeline，直喂识别器**。
        const rawCpu: string = lpr.recognise(frame.data, frame.width, frame.height, recCpuId, -1);
        const rawNp: string = lpr.recognise(frame.data, frame.width, frame.height, recNpId, -1);
        const kc: Map<string, string> = parseKv(rawCpu);
        const kn: Map<string, string> = parseKv(rawNp);
        const cc: string = str(kc, 'code');
        const cn: string = str(kn, 'code');
        const same: boolean = cc === cn;
        n++;
        if (same) {
          agree++;
        }
        if (cc === gt) {
          okCpu++;
        }
        if (cn === gt) {
          okNp++;
        }
        // 长度错误单独计：A16 用这个指标对比过 LPRNet 与 rpv3，是识别器
        // 退化的最敏感信号（比总准确率更早动）。
        if (cc.length !== gt.length) {
          lenCpu++;
        }
        if (cn.length !== gt.length) {
          lenNp++;
        }
        lpr.appendLine(logPath, 'BARE file=' + name + ' gt=' + gt +
          ' code_cpu=' + cc + ' code_np=' + cn +
          ' agree=' + (same ? '1' : '0') +
          ' ok_cpu=' + (cc === gt ? '1' : '0') +
          ' ok_np=' + (cn === gt ? '1' : '0') +
          ' ms_cpu=' + str(kc, 'recMs') + ' ms_np=' + str(kn, 'recMs'));
        if (n % 100 === 0) {
          this.appendNative('BARE PROGRESS n=' + n.toString() +
            ' agree=' + agree.toString() + ' ok_cpu=' + okCpu.toString() +
            ' ok_np=' + okNp.toString() + ' lenerr_cpu=' + lenCpu.toString() +
            ' lenerr_np=' + lenNp.toString());
        }
      }
      this.appendNative('BARE END n=' + n.toString() + ' agree=' + agree.toString() +
        ' ok_cpu=' + okCpu.toString() + ' ok_np=' + okNp.toString() +
        ' lenerr_cpu=' + lenCpu.toString() + ' lenerr_np=' + lenNp.toString() +
        ' bad_gt=' + badGt.toString() + ' log=' + logPath);
    } finally {
      this.busy = false;
    }
  }

  // ------------------------------------------------------------ 整车场景 det A/B（T12）

  /**
   * 【T12】真实整车场景 × 检测器落点 A/B。
   *
   * 为什么必须做：T10 用 1000 张 **94×24 裁剪图**测出「det 落点分歧率 6.8%」，
   * 但那是在「det 面对的是一张已经是车牌的小图」这个**非真实工况**下得到的。
   * 真实场景里 det 面对的是 720×1160 的整车画面 —— 候选框数量、尺度分布、
   * NMS 行为全都不同，6.8% 不能直接外推。
   *
   * 数据集：CCPD2019-balance 分层抽 1000 张（30 省，每省上限 100，皖不再占 54%）。
   * 真值来自 CCPD 文件名编码（`tools/ccpd_gt.py` 解出，与本探针同源），
   * 随包的 `gt.tsv` 是唯一真值源，ArkTS 侧不重解。
   *
   * 设计同 T10/T12：唯一变量是被测角色的落点，其余两个角色固定生产配置。
   *
   * @param role 变哪个角色：`'det'`（检测器）或 `'rec'`（识别器）。
   *   **必须两个都跑**：T12 已证明 det 在真实场景下分歧率只有 0.1%，
   *   而 rec 的 4.9% 是在**裁剪图裸喂**下测的（T11）—— 跨工况不可比。
   *   只有同输入、同工况的 det vs rec 才能回答「真实场景下哪个角色更敏感」。
   * @param set 数据集：`'ccpd'`（7 字符常规牌，1000 张，30 省分层）或
   *   `'green'`（**8 字符新能源牌**，1000 张，13 省）。绿牌集省份极偏斜
   *   （原分布皖 11373/11776 = 96.6%，抽样后 60%），引用时须一起说。
   *   两个集的图都是 CCPD 整车图，同一套探针与真值口径。
   */
  private async ccpdSceneAB(role: string, set: string): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('SCENE FAIL no context');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/scene_' + set + '_' + role + '.log';
    const sceneDir: string = 'assets/' + set;
    this.appendNative('SCENE BEGIN set=' + set + ' role=' + role +
      ' dir=' + sceneDir + ' log=' + logPath);
    try {
      // 真值表：从随包的 gt.tsv 读，**不重解文件名** —— 两处解译就会有两条真值源。
      let gtMap: Map<string, string> = new Map();
      try {
        const gtBytes: Uint8Array =
          ctx.resourceManager.getRawFileContentSync(sceneDir + '/gt.tsv');
        // ArkTS 没有浏览器那个全局 TextDecoder，用 util.TextDecoder。
        // 注意它的 decode 收 **Uint8Array**（不是 ArrayBuffer）—— 传 buffer 会在
        // 编译期报 "not assignable to parameter of type 'Uint8Array'"。
        const gtText: string = new util.TextDecoder('utf-8').decode(gtBytes);
        const lines: string[] = gtText.split('\n');
        for (let i = 0; i < lines.length; i++) {
          const ln: string = lines[i].trim();
          if (ln.length === 0) {
            continue;
          }
          const tab: number = ln.indexOf('\t');
          if (tab <= 0) {
            continue;
          }
          gtMap.set(ln.substring(0, tab), ln.substring(tab + 1));
        }
      } catch (e) {
        const err = e as BusinessError;
        this.appendNative('SCENE FAIL gt.tsv code=' + err.code + ' msg=' + err.message);
        this.busy = false;
        return;
      }
      this.appendNative('SCENE GT ' + gtMap.size.toString());
      if (gtMap.size === 0) {
        this.appendNative('SCENE FAIL gt.tsv 是空的');
        this.busy = false;
        return;
      }

      let names: string[] = [];
      try {
        names = await ctx.resourceManager.getRawFileList(sceneDir);
      } catch (e) {
        const err = e as BusinessError;
        this.appendNative('SCENE FAIL listdir code=' + err.code + ' msg=' + err.message);
        this.busy = false;
        return;
      }
      names = names.filter((n: string) => n.toLowerCase().endsWith('.jpg')).sort();
      if (names.length === 0) {
        this.appendNative('SCENE FAIL 目录里没有 .jpg');
        this.busy = false;
        return;
      }
      this.appendNative('SCENE FILES ' + names.length.toString());

      // 载入**可变角色**的两个落点；其余两个角色用生产会话（不变）。
      const roleFile: string = role === 'det' ? MODEL_DET : MODEL_REC;
      const roleBytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(roleFile);
      const loadCpu: string =
        await lpr.loadModelAsync(roleFile, this.toArrayBuffer(roleBytes), 'cpu');
      const loadNp: string =
        await lpr.loadModelAsync(roleFile, this.toArrayBuffer(roleBytes), 'nnrt');
      const ci: Map<string, string> = parseKv(loadCpu);
      const ni: Map<string, string> = parseKv(loadNp);
      lpr.appendLine(logPath, 'SCENE LOAD ' + role + ' cpu ' + loadCpu);
      lpr.appendLine(logPath, 'SCENE LOAD ' + role + ' nnrt ' + loadNp);
      if (ci.get('ok') !== '1' || ni.get('ok') !== '1') {
        this.appendNative('SCENE FAIL ' + role + ' load cpu_ok=' + str(ci, 'ok') +
          ' np_ok=' + str(ni, 'ok'));
        this.busy = false;
        return;
      }
      const cpuId: number = num(ci, 'id');
      const npId: number = num(ni, 'id');
      this.appendNative('SCENE SESSIONS set=' + set + ' role=' + role +
        ' det=' + this.detId.toString() + ' rec=' + this.recId.toString() +
        ' cls=' + this.clsId.toString() + ' cpu_id=' + cpuId.toString() +
        ' np_id=' + npId.toString());

      let n: number = 0;
      let agree: number = 0;
      let okCpu: number = 0;
      let okNp: number = 0;
      let noPlateCpu: number = 0;
      let noPlateNp: number = 0;
      for (let i = 0; i < names.length; i++) {
        const name: string = names[i];
        const gt: string | undefined = gtMap.get(name);
        if (gt === undefined || (gt.length !== 7 && gt.length !== 8)) {
          lpr.appendLine(logPath, 'SCENE SKIP file=' + name + ' reason=no-gt');
          continue;
        }
        let frame: RgbaFrame | null = null;
        try {
          frame = await this.decodeRgba(sceneDir + '/' + name);
        } catch (e) {
          const err = e as BusinessError;
          lpr.appendLine(logPath, 'SCENE DECODEFAIL file=' + name + ' code=' + err.code);
          continue;
        }
        if (frame === null) {
          continue;
        }
        // 同一张整车图、同样的其余两个角色，只换 role 的落点。
        // 局部变量而不是三元塞实参：可读性差且容易传错顺序。
        const dCpu: number = role === 'det' ? cpuId : this.detId;
        const dNp: number = role === 'det' ? npId : this.detId;
        const rCpu: number = role === 'rec' ? cpuId : this.recId;
        const rNp: number = role === 'rec' ? npId : this.recId;
        const rawCpu: string = await lpr.pipelineAsync(
          frame.data, frame.width, frame.height, dCpu, rCpu, this.clsId, 0);
        const rawNp: string = await lpr.pipelineAsync(
          frame.data, frame.width, frame.height, dNp, rNp, this.clsId, 0);
        const kc: Map<string, string> = parseKv(rawCpu);
        const kn: Map<string, string> = parseKv(rawNp);
        const cc: string = str(kc, 'p0').split(',')[0];
        const cn: string = str(kn, 'p0').split(',')[0];
        const same: boolean = cc === cn;
        n++;
        if (same) {
          agree++;
        }
        if (cc === gt) {
          okCpu++;
        }
        if (cn === gt) {
          okNp++;
        }
        // 检不出牌要单独计：det 换后端可能直接把牌弄丢了，那是比读错更严重的失效。
        if (cc.length === 0) {
          noPlateCpu++;
        }
        if (cn.length === 0) {
          noPlateNp++;
        }
        lpr.appendLine(logPath, 'SCENE set=' + set + ' role=' + role + ' file=' + name + ' gt=' + gt +
          ' code_cpu=' + cc + ' code_np=' + cn +
          ' agree=' + (same ? '1' : '0') +
          ' ok_cpu=' + (cc === gt ? '1' : '0') +
          ' ok_np=' + (cn === gt ? '1' : '0') +
          ' none_cpu=' + (cc.length === 0 ? '1' : '0') +
          ' none_np=' + (cn.length === 0 ? '1' : '0'));
        if (n % 50 === 0) {
          this.appendNative('SCENE PROGRESS n=' + n.toString() +
            ' agree=' + agree.toString() + ' ok_cpu=' + okCpu.toString() +
            ' ok_np=' + okNp.toString());
        }
      }
      this.appendNative('SCENE END set=' + set + ' role=' + role + ' n=' + n.toString() +
        ' agree=' + agree.toString() +
        ' ok_cpu=' + okCpu.toString() + ' ok_np=' + okNp.toString() +
        ' none_cpu=' + noPlateCpu.toString() + ' none_np=' + noPlateNp.toString() +
        ' log=' + logPath);
    } finally {
      this.busy = false;
    }
  }

  // ------------------------------------------------------------ ncnn 槽位探针

  /**
   * 让**识别与分类**也走 ncnn（CPU 与 Vulkan 各一遍）—— 补上「三模型 × 三后端」
   * 里 GPU 那一格。
   *
   * 模型由 pnnx 从同一份 ONNX 转出，槽位号 1 = 识别、2 = 分类；预处理在 C++ 里复用
   * 同一条 LprEncodePlate / LprEncodeClassify，只有内存布局不同（MS 是 NHWC、ncnn 是
   * NCHW）—— 所以两边的差异只可能来自推理实现本身。
   *
   * 判据与其它矩阵一致：逐字符比对车牌串 + 颜色分类；同时打印每个字符的概率
   * 与三分类分数 —— 换了后端这些数值若仍逐位一致，说明数值保真；若只有车牌串一致而
   * 分数漂移，那就与 NPU 漂移同性质，必须写清楚。
   */
  private async ncnnSlotProbe(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined || this.detId < 0) {
      this.appendNative('NCNNSLOT FAIL missing sessions');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/matrix.log';
    this.appendNative('NCNNSLOT BEGIN reference=' + REFERENCE_CODE_RPV3);
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(SELFTEST_SAMPLE);
      if (frame === null) {
        this.appendNative('NCNNSLOT FAIL no frame');
        return;
      }
      const roles: string[] = ['rec', 'cls'];
      const slots: number[] = [SLOT_REC, SLOT_CLS];
      const params: string[] = [NCNN_REC_PARAM, NCNN_CLS_PARAM];
      const bins: string[] = [NCNN_REC_BIN, NCNN_CLS_BIN];
      const modes: string[] = ['cpu', 'vulkan'];
      for (let r = 0; r < roles.length; r++) {
        let paramBytes: Uint8Array;
        let binBytes: Uint8Array;
        try {
          paramBytes = ctx.resourceManager.getRawFileContentSync(params[r]);
          binBytes = ctx.resourceManager.getRawFileContentSync(bins[r]);
        } catch (e) {
          this.appendNative('NCNNSLOT ' + roles[r] + ' MISSING-RAWFILE');
          continue;
        }
        for (let m = 0; m < modes.length; m++) {
          const useVulkan: boolean = m === 1;
          const loadRaw: string = await lpr.ncnnLoadSlotAsync(
            slots[r], this.toArrayBuffer(paramBytes), this.toArrayBuffer(binBytes), useVulkan);
          const li: Map<string, string> = parseKv(loadRaw);
          const loadLine: string = 'NCNNSLOT LOAD role=' + roles[r] + ' mode=' + modes[m] +
            ' ' + loadRaw;
          this.appendNative(loadLine);
          lpr.appendLine(logPath, loadLine);
          if (li.get('ok') !== '1') {
            continue;
          }
          const recSlot: number = roles[r] === 'rec' ? slots[r] : -1;
          const clsSlot: number = roles[r] === 'cls' ? slots[r] : -1;
          // ncnn 行的落点必须自己声明：`mode=` 只是请求值。effectiveVulkan=0 时
          // use_vulkan_compute 被静默忽略，那一行跑的其实是 CPU，不能算 GPU 数据点。
          const vkReq: string = str(li, 'requestedVulkan');
          const vkReal: string = str(li, 'effectiveVulkan');
          // vkLayers=X/Y：只有 X 层带 Vulkan 实现。X<Y 时那一格是「GPU + 部分 CPU 回落」，
          // 不能当纯 GPU 数据点引用。
          const vkCover: string = str(li, 'vkLayers');
          const landedNcnn: string = (useVulkan && vkReal === '1') ? 'GPU' : 'CPU';
          // 预热一次：Vulkan 首跑含 shader 编译与缓冲分配，计进去会把冷启动当差异
          await lpr.pipelineAsync(frame.data, frame.width, frame.height,
            this.detId, this.recId, this.clsId, 0, recSlot, clsSlot);
          for (let rep = 0; rep < 3; rep++) {
            const pr: string = await lpr.pipelineAsync(frame.data, frame.width, frame.height,
              this.detId, this.recId, this.clsId, 0, recSlot, clsSlot);
            const kv: Map<string, string> = parseKv(pr);
            const f: string[] = str(kv, 'p0').split(',');
            const code: string = f[0];
            const colour: string = f.length > 6 ? colourLabel(f[6].split('|')[0]) : '-';
            const line: string = 'NCNNSLOT RESULT role=' + roles[r] + ' mode=' + modes[m] +
              ' LANDED=' + landedNcnn + ' engine=ncnn vkReq=' + vkReq + ' vk=' + vkReal +
              ' vkLayers=' + vkCover + ' rep=' + rep +
              ' match=' + (kv.get('ok') === '1' && code === REFERENCE_CODE_RPV3 ? '1' : '0') +
              ' code=' + code + ' colour=' + colour +
              ' total=' + str(kv, 'totalMs') +
              ' stages=' + (f.length > 9 ? f[9] : '-') +
              ' chars=' + (f.length > 7 ? f[7] : '-') +
              ' clsScores=' + (f.length > 6 ? f[6] : '-');
            this.appendNative(line);
            lpr.appendLine(logPath, line);
          }
        }
      }
      this.appendNative('NCNNSLOT END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('NCNNSLOT EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  // ------------------------------------------------------------ CANN / NNRt 探针

  /**
   * 问设备两件事（不改任何生产路径）：
   *   1) NNRt 与 hiAI 的库能不能加载、CANN 什么版本、NNRt 上有几张什么类型的设备；
   *   2) 我们手上的 `.ms` 模型，NNRt 的 offline model 入口收不收、HiAI 认不认。
   *
   * 背景：官方文档说 offline model 是「设备厂商提供的转换器」编出来的，多半只吃 `.om`；
   * 但 `.ms` 是否也被同一入口接受，只有真机试一次才知道。
   */
  private async cannProbe(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('CANN FAIL no context');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/matrix.log';
    this.appendNative('CANN BEGIN');
    try {
      const enumRaw: string = await lpr.nnrtProbeAsync();
      const enumLine: string = 'CANN ENUM ' + enumRaw;
      this.appendNative(enumLine);
      lpr.appendLine(logPath, enumLine);

      const models: string[] = [MODEL_VENDOR_OM, MODEL_OM_LPRNET, MODEL_OM_LPRNET_NPUFIX, MODEL_OM_DET,
        MODEL_OM_DETHEAD, MODEL_OM_REC, MODEL_OM_CLS, MODEL_DET, MODEL_REC, MODEL_CLS];
      for (let i = 0; i < models.length; i++) {
        let bytes: Uint8Array;
        try {
          bytes = ctx.resourceManager.getRawFileContentSync(models[i]);
        } catch (e) {
          this.appendNative('CANN TRY model=' + models[i] + ' MISSING-RAWFILE');
          continue;
        }
        const raw: string = await lpr.nnrtTryModelAsync(this.toArrayBuffer(bytes), 0);
        const line: string = 'CANN TRY model=' + models[i] + ' size=' + bytes.length + ' ' + raw;
        this.appendNative(line);
        lpr.appendLine(logPath, line);
      }
      this.appendNative('CANN END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('CANN EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  /**
   * T8 · 算子覆盖对撞：把 50 个算子探针模型逐个喂给 CANN 侧，跑三关
   * （`hiai_compat_code` / `construct` / `build_rc`），并与 `.ms` 侧的既有矩阵对照。
   *
   * 回答的问题：**「NPU 支持性」不是模型的属性，而是「模型 × 工具链」的联合属性。**
   * CANN 侧有 166 个 IR 算子，MS Lite 侧的官方算子表只有 58/198 且全 FP16-only ——
   * 同一个算子在一侧被拒、在另一侧通过，正是这条主张的直接证据。
   *
   * 名单硬编码而非枚举 rawfile：`resourceManager` 没有目录枚举 API，而这份名单
   * 同时就是实验记录（与 ShusenPaper 的 opbench_manifest.json 同源）。
   * rawfile 里的文件名带 `op_` 前缀，避免与生产模型的 `probe_om_*.om` 混淆。
   */
  private async opCollideProbe(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('OPCOLLIDE FAIL no context');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/op_collide.log';
    this.appendNative('OPCOLLIDE BEGIN n=' + OP_PROBE_NAMES.length);
    lpr.appendLine(logPath, 'OPCOLLIDE BEGIN n=' + OP_PROBE_NAMES.length +
      ' ts=' + Date.now().toString());
    try {
      for (let i = 0; i < OP_PROBE_NAMES.length; i++) {
        const name: string = OP_PROBE_NAMES[i];
        const file: string = 'models/op_' + name + '.om';
        let bytes: Uint8Array;
        try {
          bytes = ctx.resourceManager.getRawFileContentSync(file);
        } catch (e) {
          const miss: string = 'OPCOLLIDE op=' + name + ' MISSING-RAWFILE';
          this.appendNative(miss);
          lpr.appendLine(logPath, miss);
          continue;
        }
        const raw: string = await lpr.nnrtTryModelAsync(this.toArrayBuffer(bytes), 0);
        // 一行一条，字段名与 T2 的落点证据同风格，便于离线聚合。
        // 空返回标成显式失败：首轮 51 个里有 13 个收到空串，而日志分不清
        // "没执行" 与 "执行了但结果为空"。
        const payload: string = raw.length > 0 ? raw : 'ok=0;error=empty-return';
        const line: string = 'OPCOLLIDE op=' + name + ' bytes=' + bytes.length + ' ' + payload;
        this.appendNative(line);
        lpr.appendLine(logPath, line);
        // 每轮让出事件循环。
        //
        // 【2026-09-21 更正】这里原本写的理由是「native 侧 InferenceRunner 是单任务槽，
        // 多个池线程同时 Submit 会被覆盖」—— 那个描述现在**不成立也不该再出现**：
        // InferenceRunner 已是真正的队列（每任务自带 finished 与自己的条件变量，
        // 见 napi_init.cpp 的类注释），并发提交不会再丢任务。
        //
        // 保留 `await this.tick()` 的理由变成两条，都与「别丢任务」无关：
        //   1. 51 个模型连续 await 会把 ArkTS 事件循环饿死 —— UI 无响应、日志刷不出来；
        //   2. 每轮之间让出一次，面板能实时看到进度。
        // 即：它现在是**响应性**措施，不是并发正确性措施。
        await this.tick();
      }
      this.appendNative('OPCOLLIDE END');
      lpr.appendLine(logPath, 'OPCOLLIDE END ts=' + Date.now().toString());
    } catch (e) {
      const err = e as Error;
      this.appendNative('OPCOLLIDE EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  /**
   * T7 · RQ4：持续负载热特性。
   *
   * 回答的问题：**「榨干」的收益能不能维持？**
   *
   * 背景：`ShusenPaper` 的既有实测发现同构建同协议两轮之间 **CPU 端方差 30–50%、
   * NPU 端 <8%**，主因疑为热态累积。本探针把那条线索变成一条可引用的曲线。
   *
   * 系统会主动杀负载：`SystemLoadLevel` 有 8 档，官方要求 HIGH(3) 起「停止所有无感服务」。
   * 所以本探针**不把被降级当作失败** —— 降级时刻本身就是数据点，如实记录。
   *
   * 热态数据来源：native sysroot **没有**热态头文件，只能在 ArkTS 侧取。
   *   thermal.getThermalLevel()      热档 0=COOL … 5=EMERGENCY
   *   batteryInfo.batteryTemperature 电池温度，单位 **0.1 ℃**（要除以 10）
   *   batteryInfo.chargingStatus / batterySOC
   *   systemLoad.getLevel()          系统负载档（异步）
   */
  private async rq4SustainedProbe(minutes: number): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined) {
      this.appendNative('RQ4 FAIL no context');
      return;
    }
    this.busy = true;
    const logPath: string = ctx.filesDir + '/rq4_thermal.log';
    const frameFile: string = NATIVE_SAMPLES[0];
    const intervalMs: number = 15000;                 // 每 15 s 采一轮
    const rounds: number = Math.floor((minutes * 60000) / intervalMs);
    const t0: number = Date.now();

    this.rq4Rows = [];
    this.rq4Status = '预热中…';
    this.appendNative('RQ4 BEGIN minutes=' + minutes + ' rounds=' + rounds +
      ' intervalMs=' + intervalMs + ' frame=' + frameFile);
    lpr.appendLine(logPath, 'RQ4 BEGIN minutes=' + minutes + ' rounds=' + rounds +
      ' intervalMs=' + intervalMs + ' frame=' + frameFile + ' ts=' + t0.toString());

    let frame: RgbaFrame | null = null;
    try {
      frame = await this.decodeRgba(frameFile);
    } catch (e) {
      this.appendNative('RQ4 FAIL decodeRgba: ' + (e as Error).message);
      this.busy = false;
      return;
    }
    if (frame === null) {
      this.appendNative('RQ4 FAIL frame null');
      this.busy = false;
      return;
    }
    // 局部常量：TS 在闭包内不会保留对 `frame` 的非空收窄，取出来更稳。
    const f: RgbaFrame = frame;

    try {
      for (let r = 0; r < rounds; r++) {
        const rStart: number = Date.now();
        // ---- 一次完整流水线（与生产档同路径）
        const raw: string = await lpr.pipelineAsync(
          f.data, f.width, f.height,
          this.detId, this.recId, this.clsId, 0);
        const kv: Map<string, string> = parseKv(raw);
        const ok: string = str(kv, 'ok');
        const totalMs: string = str(kv, 'totalMs');
        const backends: string = str(kv, 'backends');
        const code: string = ok === '1' ? str(kv, 'p0').split(',')[0] : '-';
        // 记完整 p0（含 cropSum 与分段明细）。首版只记 code，导致发现
        // 「苏ED5172 → 苏ED512」时无法自查 —— cropSum 正是为这种跨会话对照设计的。
        const p0: string = ok === '1' ? str(kv, 'p0') : '-';

        // ---- 热态/功耗读数
        const tl: number = thermal.getThermalLevel();
        const bt: number = batteryInfo.batteryTemperature;   // 0.1 ℃
        const soc: number = batteryInfo.batterySOC;
        const charging: number = batteryInfo.chargingStatus;
        let load: string = '-';
        try {
          load = (await systemLoad.getLevel()).toString();
        } catch (e) {
          load = 'na';
        }
        this.thermalLevel = tl;
        this.batteryTempC = bt / 10;

        // 一行一轮：时间 / 热档 / 温度 / SOC / 充电 / 负载档 / 端到端 / 车牌 / 落点
        const line: string = 'RQ4 r=' + r + ' t=' + Math.round((rStart - t0) / 1000) +
          ' thermal=' + tl + ' battC=' + (bt / 10).toFixed(1) +
          ' soc=' + soc + ' chg=' + charging + ' load=' + load +
          ' totalMs=' + totalMs + ' code=' + code + ' backends=' + backends;
        // p0 单独一行：它很长（含 chars/probs/分段），与上面那行拼在一起会被
        // hilog 的单行上限截断。单独成行才能完整落盘。
        const p0line: string = 'RQ4P r=' + r + ' p0=' + p0;
        this.rq4Rows = this.rq4Rows.concat([line]);
        this.appendNative(line);
        lpr.appendLine(logPath, line);
        lpr.appendLine(logPath, p0line);

        this.rq4Status = '第 ' + (r + 1) + '/' + rounds + ' 轮 · 热档 ' + tl +
          ' · ' + (bt / 10).toFixed(1) + '℃ · 端到端 ' + totalMs + ' ms';

        // ---- 等下一轮（保持连续负载，不空转 JS）
        const elapsed: number = Date.now() - rStart;
        const wait: number = intervalMs - elapsed;
        if (wait > 0) {
          await this.sleep(wait);
        }
      }
      const totalMin: number = (Date.now() - t0) / 60000;
      this.rq4Status = '完成 · ' + rounds + ' 轮 · ' + totalMin.toFixed(1) + ' 分钟';
      this.appendNative('RQ4 END rounds=' + rounds + ' minutes=' + totalMin.toFixed(2));
      lpr.appendLine(logPath, 'RQ4 END rounds=' + rounds +
        ' minutes=' + totalMin.toFixed(2) + ' ts=' + Date.now().toString());
    } catch (e) {
      const err = e as Error;
      this.appendNative('RQ4 EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

// ------------------------------------------------------------ 探针

  private async probeOne(ctx: common.UIAbilityContext, file: string, backend: string): Promise<void> {
    const short: string = file.replace('models/', '');
    let bytes: Uint8Array;
    try {
      bytes = ctx.resourceManager.getRawFileContentSync(file);
    } catch (e) {
      this.append('MATRIX ' + short + ' ' + backend + ' MISSING-RAWFILE');
      return;
    }

    const loadRaw: string = await lpr.loadModelAsync(file, this.toArrayBuffer(bytes), backend);
    const li: Map<string, string> = parseKv(loadRaw);
    if (li.get('ok') !== '1') {
      this.append('MATRIX ' + short + ' ' + backend + ' BUILD-FAIL err=' +
        str(li, 'error') + ' trail=' + str(li, 'attemptLog'));
      return;
    }

    const id: number = num(li, 'id');
    const landed: string = str(li, 'backend');
    // 请求值与实际落点必须同时出现：`backend=gpu LANDED=CPU fallback=GPU:fp16` 这一串
    // 才是「GPU 没被调用」的自证证据，单看 LANDED=CPU 会被读成「我们选了 CPU」。
    const reqBe: string = str(li, 'requested');
    const fellBack: string = str(li, 'fallbackFrom');
    const b: Map<string, string> = parseKv(await lpr.benchAsync(id, 5, 30));
    if (b.get('ok') !== '1') {
      this.append('MATRIX ' + short + ' ' + backend + ' LANDED=' + landed +
        ' engine=ms req=' + reqBe + ' fallback=' + fellBack +
        ' BENCH-FAIL err=' + str(b, 'error'));
      return;
    }

    const cks: number = num(b, 'checksum');
    const ck16: number = num(b, 'checksumAsFp16');
    // 已知缺陷：fp16 位流被声明成 fp32 → 按声明读会得到 1e6 量级的垃圾值
    let verdict: string = 'PASS';
    if (landed.indexOf('CPU') < 0 && cks > 1e6) {
      verdict = 'SUSPECT-FP16-MISLABEL';
    }

    this.append('MATRIX ' + short + ' ' + backend + ' LANDED=' + landed +
      ' engine=ms req=' + reqBe + ' fallback=' + fellBack +
      ' in=' + str(li, 'inputShape') + '/' + str(li, 'inputFormat') +
      ' p50=' + str(b, 'p50Ms') + 'ms mean=' + str(b, 'meanMs') + 'ms' +
      ' p50Io=' + str(b, 'p50IoMs') + 'ms' +
      ' L2=' + cks.toFixed(4) + ' L2asFp16=' + ck16.toFixed(4) +
      ' maxAbs=' + str(b, 'maxAbs') + ' dtype=' + str(b, 'outputDtype') +
      ' verdict=' + verdict);
  }

  /**
   * CPU 线程扫描（2026-09-20）：MODELS_FULL × cpu_t{1,2,4,6,8}，每项 warmup 5 + 30 次 p50。
   *
   * 为什么需要：`ms_engine.cpp` 的 `OH_AI_ContextSetThreadNum` 此前恒传 4 且从未扫描，
   * 论文里所有 NPU/CPU 比值的 CPU 端都是未调优基线（ADR-015 未完成动作）。
   * 本探针给出每模型的最优线程数与对应 p50，让比值的 CPU 端从此有据可依。
   * 行格式与 probeAll 一致（`MATRIX <model> cpu_tN ...`），只是 backend 换成线程档。
   */
  private async threadSweep(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('THREAD SWEEP BEGIN models=' + MODELS_FULL.length.toString() +
      ' tiers=' + THREAD_BACKENDS.join(','));
    try {
      for (let i = 0; i < MODELS_FULL.length; i++) {
        for (let j = 0; j < THREAD_BACKENDS.length; j++) {
          await this.probeOne(ctx, MODELS_FULL[i], THREAD_BACKENDS[j]);
          await this.tick();
        }
      }
      this.appendNative('THREAD SWEEP END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('THREAD SWEEP EXCEPTION ' + err.message);
    }
  }

  /**
   * CPU 亲和扫描（2026-09-21）：MODELS_FULL × {cpu_t4, cpu_a1, cpu_a2}。
   *
   * 回答的问题：线程扫描发现「隔离 7.4 ms vs 流水线内 15.3–19.5 ms」的 2 倍差距，
   * 且更热的一轮反而更快 ⇒ 主因**不是**热降频，怀疑是 DVFS/核位调度。
   * 本扫描用 `OH_AI_ContextSetThreadAffinityMode` 钉核位，看能否改变 p50。
   *
   * 判据：与 cpu_t4（无绑定基线）比 p50。若 a1 明显更快 ⇒ 支持调度假设；
   * 若三者相同 ⇒ 调度假设被削弱，应转向「流水线内另有开销」的解释。
   * `L2` 三者必须逐位相同（同一模型、同一确定性输入），否则说明实验有串扰。
   */
  private async affinitySweep(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('AFFINITY SWEEP BEGIN models=' + MODELS_FULL.length.toString() +
      ' tiers=' + AFFINITY_BACKENDS.join(','));
    try {
      for (let i = 0; i < MODELS_FULL.length; i++) {
        for (let j = 0; j < AFFINITY_BACKENDS.length; j++) {
          await this.probeOne(ctx, MODELS_FULL[i], AFFINITY_BACKENDS[j]);
          await this.tick();
        }
      }
      this.appendNative('AFFINITY SWEEP END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('AFFINITY SWEEP EXCEPTION ' + err.message);
    }
  }

  /**
   * 【C8】间隔/污染/忙等扫描（2026-09-21）：定位「隔离 7.3 ms vs 流水线 19.5 ms」。
   *
   * 已确证的结论（见 GAP_TIERS / GAP_BACKENDS 的注释）：
   *   - `gapMs` 单调把 p50 从 7.3 推到 31.9 ms（剂量-反应）
   *   - **忙等（spinMs=33）同样慢（30.4 ms）⇒ DVFS 被否掉**
   *   - 污染 1.2 MB 无影响（7.33 ms）⇒ 缓存/带宽被否掉
   *   - 同一次调用里的 `p50Io`（独立紧循环）恒定 ~8 ms ⇒ 会话本身没退化
   * 现测**机理**：在 `GAP_BACKENDS` 的每个后端上重跑同一组档，
   * 看 `cpu_t1`（单线程、无线程池可 park）是否消除 gap 效应。
   *
   * 每个模型 × 每个后端只加载一次，再对该后端的各档重复 bench ——
   * 保证档与档之间唯一的差异就是 gapMs/polluteKB/spinMs，而不是重新加载。
   * 每档的 `L2` 必须逐位相同，否则说明干扰影响了数值（不应发生）。
   */
  private async gapSweep(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('GAP SWEEP BEGIN models=' + MODELS_FULL.length.toString() +
      ' backends=' + GAP_BACKENDS.join(',') +
      ' tiers=' + GAP_TIERS.length.toString());
    try {
      for (let bi = 0; bi < GAP_BACKENDS.length; bi++) {
        const backend: string = GAP_BACKENDS[bi];
        for (let i = 0; i < MODELS_FULL.length; i++) {
          const file: string = MODELS_FULL[i];
          const short: string = file.replace('models/', '');
          let bytes: Uint8Array;
          try {
            bytes = ctx.resourceManager.getRawFileContentSync(file);
          } catch (e) {
            this.append('MATRIX ' + short + ' MISSING-RAWFILE');
            continue;
          }
          const loadRaw: string = await lpr.loadModelAsync(
            file, this.toArrayBuffer(bytes), backend);
          const li: Map<string, string> = parseKv(loadRaw);
          if (li.get('ok') !== '1') {
            this.append('MATRIX ' + short + ' ' + backend +
              ' BUILD-FAIL err=' + str(li, 'error'));
            continue;
          }
          const id: number = num(li, 'id');
          const landed: string = str(li, 'backend');
          for (let j = 0; j < GAP_TIERS.length; j++) {
            const gap: number = GAP_TIERS[j][0];
            const pol: number = GAP_TIERS[j][1];
            const spn: number = GAP_TIERS[j][2];
            const b: Map<string, string> = parseKv(
              await lpr.benchAsync(id, 5, 30, gap, pol, spn));
            if (b.get('ok') !== '1') {
              this.append('MATRIX ' + short + ' ' + backend +
                ' gap=' + gap.toString() + ' pol=' + pol.toString() +
                ' spin=' + spn.toString() + ' BENCH-FAIL err=' + str(b, 'error'));
              continue;
            }
            const cks: number = num(b, 'checksum');
            // 行里同时回显 gapMs/polluteKB/spinMs（来自 native，不是我们传的值）——
            // 这样「档真的生效了」是自证的，不靠 ArkTS 侧的记忆。
            this.append('MATRIX ' + short + ' LANDED=' + landed +
              ' gapMs=' + str(b, 'gapMs') + ' polluteKB=' + str(b, 'polluteKB') +
              ' spinMs=' + str(b, 'spinMs') +
              ' p50=' + str(b, 'p50Ms') + 'ms mean=' + str(b, 'meanMs') + 'ms' +
              ' p50Io=' + str(b, 'p50IoMs') + 'ms' +
              ' min=' + str(b, 'minMs') + 'ms max=' + str(b, 'maxMs') + 'ms' +
              ' L2=' + cks.toFixed(4));
            await this.tick();
          }
        }
      }
      this.appendNative('GAP SWEEP END');
    } catch (e) {
      const err = e as Error;
      this.appendNative('GAP SWEEP EXCEPTION ' + err.message);
    }
  }

  /** 加速器矩阵：逐模型 × 逐后端，每项 warmup 5 + 正式 30 次 p50。 */
  private async probeAll(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.lines = [];

    try {
      const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
      this.append('MATRIX NNRT-DEVICES ' + lpr.listNnrtDevices());

      const plan: Row[] = [];
      for (let i = 0; i < MODELS_FULL.length; i++) {
        for (let j = 0; j < FULL_BACKENDS.length; j++) {
          plan.push({ model: MODELS_FULL[i], backend: FULL_BACKENDS[j] });
        }
      }
      for (let i = 0; i < MODELS_NNRT_GPU.length; i++) {
        plan.push({ model: MODELS_NNRT_GPU[i], backend: 'nnrt' });
        plan.push({ model: MODELS_NNRT_GPU[i], backend: 'gpu' });
      }

      for (let i = 0; i < plan.length; i++) {
        this.progress = (i + 1).toString() + '/' + plan.length.toString();
        await this.probeOne(ctx, plan[i].model, plan[i].backend);
        await this.tick();
      }
      this.progress = 'done';
      this.append('MATRIX END');
    } catch (e) {
      const err = e as Error;
      this.append('MATRIX EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  /**
   * GPU 专项：三个正式模型逐个请求 `gpu` 后端。
   *
   * 只看 `backend=` 这个**实际落点**字段，不看请求值 —— MS Lite 的后端拒绝是
   * 编译期判否（`IsValid# GPU is not supported`）后静默回退 CPU，
   * 所以「请求了 gpu」和「跑在 gpu 上」是两件事。attemptLog 保留完整尝试链。
   * GPU 的真正通路是 ncnn-Vulkan，见 ncnnProbe。
   */
  private async gpuProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.appendNative('NATIVE GPU PROBE BEGIN');
    const models: string[] = [MODEL_DET, MODEL_REC, MODEL_CLS];
    for (let i = 0; i < models.length; i++) {
      await this.probeOne(ctx, models[i], 'gpu');
      await this.tick();
    }
    this.appendNative('NATIVE GPU PROBE END');
  }

  /**
   * 检测器数值漂移对照：同一份裸 head 分别落 nnrt / cpu，比 L2 与 maxAbs。
   *
   * 回答「nnrt 档是 NPU 真在算，还是静默回落 CPU」：若静默回落，两次输出必须
   * **逐位相同**（同一份 CPU 代码路径）；checksum 不同则 NPU 确实参与计算。
   */
  private async detDriftProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const file: string = 'models/y5fu_320x_head_fp32.ms';
    const backends: string[] = ['nnrt', 'cpu'];
    for (let i = 0; i < backends.length; i++) {
      const be: string = backends[i];
      try {
        const bytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(file);
        const loadRaw: string = await lpr.loadModelAsync(file, this.toArrayBuffer(bytes), be);
        const li: Map<string, string> = parseKv(loadRaw);
        if (li.get('ok') !== '1') {
          this.appendNative('DET DRIFT ' + be + ' LOAD-FAIL ' + str(li, 'error'));
          continue;
        }
        const id: number = num(li, 'id');
        const b: Map<string, string> = parseKv(await lpr.benchAsync(id, 3, 10));
        this.appendNative('DET DRIFT ' + be + ' landed=' + str(li, 'backend') +
          ' L2=' + str(b, 'checksum') + ' maxAbs=' + str(b, 'maxAbs') +
          ' p50=' + str(b, 'p50Ms') + ' mean=' + str(b, 'meanMs') +
          ' dtype=' + str(b, 'outputDtype') + ' elems=' + str(b, 'outputElems'));
      } catch (e) {
        const err = e as Error;
        this.appendNative('DET DRIFT ' + be + ' EXCEPTION ' + err.message);
      }
    }
  }

  /**
   * ncnn 路径自证：加载 ncnn 裸 head → 同一张样本图跑 → 打印三输出 L2/maxAbs 与耗时。
   * 这些值与 PC 侧 ONNX 同图参考对照，差值即 ncnn 移植保真度。
   * 预处理复用 lpr_pipeline 的 letterbox/NCHW，保证与 PC 参考同源可比。
   * GPU（Vulkan）在麒麟 8020 上的唯一通路就是这条（ADR-008）。
   */
  private async ncnnProbe(): Promise<void> {
    const ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    try {
      const param: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.param');
      const bin: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.bin');
      const l: Map<string, string> = parseKv(await lpr.ncnnLoadAsync(
        this.toArrayBuffer(param), this.toArrayBuffer(bin), true));
      if (l.get('ok') !== '1') {
        this.appendNative('NCNN LOAD FAIL ' + str(l, 'error'));
        return;
      }
      this.appendNative('NCNN LOAD ok layers=' + str(l, 'layers') +
        ' in=' + str(l, 'inputs') + ' out=' + str(l, 'outputs') +
        ' vulkan=' + str(l, 'effectiveVulkan') + ' gpu=' + str(l, 'gpuProbe'));
      const frame: RgbaFrame | null = await this.decodeRgba(SELFTEST_SAMPLE);
      if (frame === null) {
        this.appendNative('NCNN RUN SKIP (no frame)');
        return;
      }
      this.appendNative('NCNN RUN ' + lpr.ncnnRun(frame.data, frame.width, frame.height, 5));
    } catch (e) {
      const err = e as Error;
      this.appendNative('NCNN EXCEPTION ' + err.message);
    }
  }

  /**
   * GPU 终审探针（ADR-008）：调原生 vulkanProbe，把 JSON 切块打进 hilog。
   * hilog 单条会截断，所以按 ~400 字符切块。
   * 前置探针（hdc shell）已确认 const.SystemCapability.Graphic.Vulkan=true、
   * /vendor/etc/vulkan/ 存在 —— 这一步回答「App 到底能不能 dlopen loader、
   * 枚不枚得出带 COMPUTE 队列的 GPU」。
   */
  private vulkanProbeRun(): void {
    try {
      const raw: string = lpr.vulkanProbe();
      this.appendNative('VULKAN PROBE len=' + raw.length.toString());
      const chunk: number = 400;
      for (let i = 0; i < raw.length; i += chunk) {
        const end: number = Math.min(i + chunk, raw.length);
        this.appendNative('VULKAN PROBE [' + Math.floor(i / chunk).toString() + '] ' +
          raw.substring(i, end));
      }
    } catch (e) {
      const err = e as Error;
      this.appendNative('VULKAN PROBE EXCEPTION ' + err.message);
    }
  }
  /**
   * 启动：只做两件事 —— 拿 context、解析算子探针名单。
   *
   * 【为什么现在什么实验都不自动跑】重设计之前，原生 ArkUI 按钮点不了
   * （CDP 管不到原生组件，uitest dumpLayout 又会把 App 冻掉），所以只能靠
   * `AUTO_SELFTEST` 编译期开关兜底。现在「一键全跑」是面板上真实可点的按钮，
   * 开关就该退休 —— 留着一个"打开 App 就自己跑 30 分钟"的隐藏行为，
   * 对一台要交给别人验机的设备是负担，不是功能。
   *
   * 相机权限也不再申请：ADR-0008 把相机相关功能整体砍掉（云手机没有摄像头，
   * 用户当前这台真机上相机路径也不是论文要测的东西），留着弹窗只会挡住界面。
   */
  aboutToAppear(): void {
    this.ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    // 算子对撞名单：优先运行时枚举 rawfile，失败才回落到硬编码名单。
    // 这里同步做，因为 opCollideProbe 要用它拼文件路径，而枚举是一次目录读。
    const scanned: string[] = scanOpProbeNames(this.ctx);
    if (scanned.length > 0) {
      OP_PROBE_NAMES = scanned;
      this.appendNative('OP SCAN n=' + scanned.length.toString() +
        ' (runtime, from models/op_*.om)');
    } else {
      OP_PROBE_NAMES = OP_PROBE_NAMES_FALLBACK;
      this.appendNative('OP SCAN FAILED -> fallback n=' +
        OP_PROBE_NAMES_FALLBACK.length.toString() +
        ' (硬编码名单；可能与磁盘不一致，见 OP_PROBE_NAMES_FALLBACK 注释)');
    }
    this.runState = '';
  }

  // ============================================================ 重设计新增：状态 / 类型 / 编排
  //
  // 这一节是 ADR-0008 落地的**新增部分**，逻辑层（上面的探针）一行未动。
  // 分三类：
  //   1. 结构化结果类型（ResultRow / ResultGroup）—— 替换「一坨等宽 lines」
  //   2. 探针结果 → 结构化读数的**解析器**（只读 lines，不重算任何数字）
  //   3. 编排（runAll / runGroup / copyAll / clearAll）
  //
  // 【一条硬规矩】下面的解析器**只做纯文本→结构化的搬运**：数值一律来自探针
  // 自己写进 lines 的字符串，解析器不换算单位、不做统计、不臆测缺失值。
  // 面板显示的每个数字都必须能在「原始输出」里逐字找到 —— 否则面板就成了
  // 第二个真值源，而两份真值源一定会漂移（这个项目已经吃过一次亏：
  // 牌色查表 vs 像素测量、文件名真值 vs gt.tsv）。

  // ------------------------------------------------------------ 小工具

  /** 取数组末 n 项（ArkTS 的 slice 负下标行为不稳，显式算起点）。 */
  private tailOf(a: string[], n: string | number): string[] {
    const k: number = typeof n === 'string' ? Number.parseInt(n, 10) : n;
    if (a.length <= k) {
      return a;
    }
    return a.slice(a.length - k);
  }

  /** 从 `key=value` 串里取一个字段。**不用正则**，纯字符串操作。 */
  private kvField(s: string, key: string): string {
    const needle: string = key + '=';
    let i: number = s.indexOf(needle);
    if (i < 0) {
      return '';
    }
    i += needle.length;
    let j: number = i;
    while (j < s.length) {
      const c: string = s.charAt(j);
      if (c === ' ' || c === ',') {
        break;
      }
      j++;
    }
    return s.substring(i, j);
  }

  /**
   * 找最后一条以 `prefix` 开头的行。
   *
   * 为什么取**最后**一条：这些探针都是「LOAD 每条一次 + RESULT 每次上报一次」，
   * 我们要的是最终状态（最后一次 load 的实际落点、最后一个 rep 的输出）。
   */
  private lastLineWith(prefix: string): string {
    for (let i = this.lines.length - 1; i >= 0; i--) {
      if (this.lines[i].indexOf(prefix) === 0) {
        return this.lines[i];
      }
    }
    return '';
  }

  /** 所有以 `prefix` 开头的行。 */
  private linesWith(prefix: string): string[] {
    const out: string[] = [];
    for (let i = 0; i < this.lines.length; i++) {
      if (this.lines[i].indexOf(prefix) === 0) {
        out.push(this.lines[i]);
      }
    }
    return out;
  }

  /**
   * 数字裁剪（用于把 `7.410000000000001` 显示成 `7.41`）。
   * 只做**显示**裁剪，不改任何计算值；原值始终在「原始输出」里。
   */
  private fmtNum(s: string, unit: string): string {
    const f: number = Number.parseFloat(s);
    if (Number.isNaN(f)) {
      return s.length > 0 ? s : '—';
    }
    return f.toFixed(2) + unit;
  }

  // ------------------------------------------------------------ 模型名

  /**
   * 从文件名取个短标签。
   *
   * 为什么要短：`y5fu_320x_head_fp32.ms` 这样的名字在 12px 等宽下占满一行，
   * 挤掉同一行右侧的 p50 —— 而 p50 才是要竖着扫的那一列。文件名进小字副标题。
   */
  private modelShort(file: string): string {
    const base: string = file.replace('models/', '').replace('.ms', '');
    if (base.indexOf('y5fu') === 0) {
      return 'det y5fu';
    }
    if (base.indexOf('rpv3') === 0) {
      return 'rec rpv3';
    }
    if (base.indexOf('litemodel_cls') === 0) {
      return base.indexOf('fp32') >= 0 ? 'cls fp32' : 'cls fp16';
    }
    if (base.indexOf('lprnet') >= 0) {
      return 'lprnet';
    }
    return base;
  }

  /** 后端档位 → 是否落到 NPU（用于着色）。 */
  private isNpuLanding(landed: string): boolean {
    return landed.indexOf('NPU') >= 0 || landed.indexOf('NNRT') >= 0 ||
      landed.indexOf('nnrt') >= 0;
  }

  /**
   * 落点 tone。
   *
   * 语义按**论文关心的那件事**定：请求 NPU 却落到 CPU = warn（不是错误，
   * 是"没吃到加速器"这一条证据）；真落到 NPU = ok。
   * **静默回落必须看得见** —— native 侧一直如实上报 `backend=` 与
   * `fallbackFrom=`，面板不能把它压成一个中性的 CPU。
   */
  private landingTone(req: string, landed: string): string {
    const npu: boolean = this.isNpuLanding(landed);
    if (npu) {
      return 'ok';
    }
    const wantsNpu: boolean = req.indexOf('nnrt') >= 0 || req.indexOf('gpu') >= 0 ||
      req.indexOf('kirin') >= 0;
    return wantsNpu ? 'warn' : '';
  }

  // ------------------------------------------------------------ 结果组装

  private resetResults(): void {
    this.results = [];
  }

  private pushGroup(key: string, title: string, tag: string, tone: string,
                    rows: ResultRow[]): void {
    this.results = this.results.concat([{ key: key, title: title, tag: tag, tone: tone, rows: rows }]);
  }

  private row(label: string, value: string, tone: string): ResultRow {
    return { label: label, value: value, tone: tone };
  }

  private rawTail(): string[] {
    return this.tailOf(this.lines, RESULT_RAW_TAIL);
  }

  private groupLabel(g: string): string {
    if (g === GRP_RQ3) {
      return 'RQ3 · 延迟分解（端到端矩阵 + 逐模型矩阵 + 线程/亲和/gap 扫描）';
    }
    if (g === GRP_RQ4) {
      return 'RQ4 · 持续负载热特性（20 分钟循环）';
    }
    if (g === GRP_ACC) {
      return '精度 · A/B 对照（裁剪集合 / 整车场景 / ncnn 槽位）';
    }
    return 'T8 · 工具链联合属性（算子覆盖对撞 + CANN/NNRt + GPU/Vulkan）';
  }

  // ------------------------------------------------------------ 解析：延迟矩阵

  /**
   * 逐模型矩阵（probeAll / gpuProbe / threadSweep / affinitySweep / gapSweep 共用一行格式）。
   *
   * 行格式（probeOne 写的）：
   *   `MATRIX <file-short> <backend> LANDED=<..> engine=ms req=<..> fallback=<..>
   *    in=<shape>/<fmt> p50=<..>ms mean=<..>ms p50Io=<..>ms L2=<..> L2asFp16=<..>
   *    maxAbs=<..> dtype=<..> verdict=<..>`
   *
   * 全部字段都从字符串里**读出来**，唯一"加工"是把 p50 裁到两位小数显示。
   */
  private parseMatrixRows(): ResultRow[] {
    const rows: ResultRow[] = [];
    const hits: string[] = this.linesWith('MATRIX ');
    for (let i = 0; i < hits.length; i++) {
      const s: string = hits[i];
      if (s.indexOf('NNRT-DEVICES') >= 0 || s.indexOf('MATRIX END') === 0) {
        continue;
      }
      const parts: string[] = s.split(' ');
      if (parts.length < 3) {
        continue;
      }
      const model: string = this.modelShort(parts[1]);
      const backend: string = parts[2];
      const landed: string = this.kvField(s, 'LANDED');
      const req: string = this.kvField(s, 'req');
      const p50: string = this.kvField(s, 'p50');
      const mean: string = this.kvField(s, 'mean');
      const p50io: string = this.kvField(s, 'p50Io');
      const verdict: string = this.kvField(s, 'verdict');
      const tone: string = verdict.indexOf('SUSPECT') >= 0 ? 'bad'
        : this.landingTone(req, landed);
      const fellBack: string = this.kvField(s, 'fallback');
      let value: string = this.fmtNum(p50, ' ms');
      if (mean.length > 0) {
        value += '  · 均值 ' + this.fmtNum(mean, '');
      }
      if (p50io.length > 0) {
        value += '  · IO ' + this.fmtNum(p50io, '');
      }
      let label: string = model + ' @ ' + backend;
      if (fellBack.length > 0) {
        label += '  ← 回落自 ' + fellBack;
      }
      rows.push(this.row(label, value, tone));
      // 落点单独一行：这是 ADR-0003 的落点自证，不能被塞进上面那行挤掉。
      rows.push(this.row('    落点', landed.length > 0 ? landed : '—',
        this.landingTone(req, landed)));
      if (verdict.length > 0 && verdict !== 'PASS') {
        rows.push(this.row('    判定', verdict, 'bad'));
      }
    }
    return rows;
  }

  /**
   * 端到端矩阵（e2eMatrix）。
   *
   * 一行一个 (role, backend, rep)。总行数 = 4 角色 × 6 后端 × 3 次 = 72 —— 全展开
   * 会把面板冲垮，所以**按 (role, backend) 聚合成一行，显示最后一次 rep**，
   * 并在值里标出 `match` 是否在全 rep 上一致。聚合只做"取末次 + 数一致性"，
   * 不改数值。
   */
  private parseE2eRows(): ResultRow[] {
    const hits: string[] = this.linesWith('E2EMATRIX RESULT ');
    if (hits.length === 0) {
      return [];
    }
    const order: string[] = [];
    const last: Map<string, string> = new Map<string, string>();
    const total: Map<string, string> = new Map<string, string>();
    const matchN: Map<string, string> = new Map<string, string>();
    for (let i = 0; i < hits.length; i++) {
      const s: string = hits[i];
      const role: string = this.kvField(s, 'role');
      const be: string = this.kvField(s, 'backend');
      const k: string = role + '@' + be;
      if (!last.has(k)) {
        order.push(k);
        total.set(k, '0');
        matchN.set(k, '0');
      }
      last.set(k, s);
      total.set(k, (Number.parseInt(total.get(k) ?? '0', 10) + 1).toString());
      if (this.kvField(s, 'match') === '1') {
        matchN.set(k, (Number.parseInt(matchN.get(k) ?? '0', 10) + 1).toString());
      }
    }
    const rows: ResultRow[] = [];
    for (let i = 0; i < order.length; i++) {
      const k: string = order[i];
      const s: string = last.get(k) ?? '';
      const landed: string = this.kvField(s, 'LANDED');
      const req: string = this.kvField(s, 'req');
      const code: string = this.kvField(s, 'code');
      const totalMs: string = this.kvField(s, 'total');
      const rep: string = this.kvField(s, 'rep');
      const allMatch: boolean = (matchN.get(k) ?? '0') === (total.get(k) ?? '1');
      let value: string = code + '  ·  ' + this.fmtNum(totalMs, ' ms');
      if (!allMatch) {
        value = code + '  ·  ' + this.fmtNum(totalMs, ' ms') + '  ✗ 不一致';
      }
      const tone: string = allMatch ? this.landingTone(req, landed) : 'warn';
      rows.push(this.row(k, value, tone));
      rows.push(this.row('    落点 · rep=' + rep + '/' + (total.get(k) ?? '—'),
        landed.length > 0 ? landed : '—', this.landingTone(req, landed)));
    }
    return rows;
  }

  // ------------------------------------------------------------ 解析：精度

  /**
   * 精度 A/B（cropBackendAB / ccpdSceneAB / bareHeadAccuracy 共用后缀口径）。
   *
   * 三者的 END 行字段名不完全一致（`ok_cpu` vs `ok_cpu` 一致，但场景多两个
   * `none_*`），所以统一按「有的就念、没有的不显示」处理，**绝不补 0**。
   * 补 0 会凭空造出一个 0% 准确率，这比不显示危险得多。
   */
  private accRowsFromEnd(s: string, valueLabel: string): ResultRow[] {
    const rows: ResultRow[] = [];
    const n: string = this.kvField(s, 'n');
    const agree: string = this.kvField(s, 'agree');
    const okCpu: string = this.kvField(s, 'ok_cpu');
    const okNp: string = this.kvField(s, 'ok_np');
    const noneCpu: string = this.kvField(s, 'none_cpu');
    const noneNp: string = this.kvField(s, 'none_np');
    const badGt: string = this.kvField(s, 'bad_gt');
    const cnt: number = Number.parseInt(n, 10);
    if (!Number.isNaN(cnt) && cnt > 0) {
      rows.push(this.row('样本数 n', n, ''));
      rows.push(this.row('两后端输出一致', agree + ' / ' + n + '  (' +
        (Number.parseInt(agree, 10) * 100 / cnt).toFixed(1) + '%)', ''));
    } else {
      rows.push(this.row('样本数 n', n.length > 0 ? n : '—', 'warn'));
    }
    if (okCpu.length > 0 && !Number.isNaN(cnt) && cnt > 0) {
      rows.push(this.row(valueLabel + ' · CPU 命中', okCpu + ' / ' + n + '  (' +
        (Number.parseInt(okCpu, 10) * 100 / cnt).toFixed(1) + '%)', ''));
    }
    if (okNp.length > 0 && !Number.isNaN(cnt) && cnt > 0) {
      rows.push(this.row(valueLabel + ' · NPU 命中', okNp + ' / ' + n + '  (' +
        (Number.parseInt(okNp, 10) * 100 / cnt).toFixed(1) + '%)', 'ok'));
    }
    if (noneCpu.length > 0 || noneNp.length > 0) {
      rows.push(this.row('未检出车牌', 'CPU ' + noneCpu + ' · NPU ' + noneNp, 'warn'));
    }
    if (badGt.length > 0 && badGt !== '0') {
      rows.push(this.row('真值表异常条数', badGt, 'bad'));
    }
    return rows;
  }

  /**
   * 精度组：裁剪集合（T10/T11）+ 整车场景（T12）+ ncnn 槽位。
   *
   * 为什么读 END 行而不扫逐条：逐条在 filesDir 的 .log 里（1000 行），
   * lines 里只有 PROGRESS。END 行**本身就是探针算好的汇总**，重算一遍是
   * 制造第二个真值源。这里只搬运。
   */
  private parseAccGroups(): void {
    // ---- 裁剪集合：T10（det/rec 落点 A/B，走完整流水线）
    const cropEnd: string = this.lastLineWith('CROPAB END ');
    if (cropEnd.length > 0) {
      const role: string = this.kvField(cropEnd, 'role');
      const rows: ResultRow[] = this.accRowsFromEnd(cropEnd, '逐字符命中');
      rows.push(this.row('唯一变量', 'rec/det 落点（其余角色固定生产配置）', 'label'));
      rows.push(this.row('日志', this.kvField(cropEnd, 'log'), 'sub'));
      this.pushGroup('g-acc-crop', 'T10/T11 · 裁剪图集落点 A/B · role=' + role,
        'n=1000', this.toneFromAcc(rows), rows);
    }

    // ---- 裸头准确率（T11，跳过 det/rectify 直喂 rec）
    const bareEnd: string = this.lastLineWith('BARE END ');
    if (bareEnd.length > 0) {
      const rows: ResultRow[] = this.accRowsFromEnd(bareEnd, '逐字符命中');
      rows.push(this.row('工况', '跳过 det/rectify，裁剪图直喂识别器', 'label'));
      rows.push(this.row('日志', this.kvField(bareEnd, 'log'), 'sub'));
      this.pushGroup('g-acc-bare', 'T11 · 识别器裸头准确率（cpu vs nnrt）',
        'n=1000', this.toneFromAcc(rows), rows);
    }

    // ---- 整车场景：ccpd（7 字符）+ green（8 字符新能源）
    const sets: string[] = ['ccpd', 'green'];
    const roles: string[] = ['det', 'rec'];
    for (let si = 0; si < sets.length; si++) {
      for (let ri = 0; ri < roles.length; ri++) {
        const prefix: string = 'SCENE END set=' + sets[si] + ' role=' + roles[ri] + ' ';
        const s: string = this.lastLineWith(prefix);
        if (s.length === 0) {
          continue;
        }
        const rows: ResultRow[] = this.accRowsFromEnd(s, '整条车牌全对');
        rows.push(this.row('唯一变量', roles[ri] + ' 的落点（其余固定生产配置）', 'label'));
        rows.push(this.row('真值来源', '随包 gt.tsv（ArkTS 侧不重解文件名）', 'label'));
        rows.push(this.row('日志', this.kvField(s, 'log'), 'sub'));
        this.pushGroup('g-acc-scene-' + sets[si] + '-' + roles[ri],
          'T12 · 整车场景 · ' + sets[si] + ' · role=' + roles[ri],
          sets[si] === 'ccpd' ? 'n=1000 · 7 字符' : 'n=1000 · 8 字符绿牌',
          this.toneFromAcc(rows), rows);
      }
    }

    // ---- ncnn 槽位（GPU 唯一通路的自证）
    const slotHits: string[] = this.linesWith('NCNNSLOT RESULT ');
    if (slotHits.length > 0) {
      const seen: string[] = [];
      const last: Map<string, string> = new Map<string, string>();
      for (let i = 0; i < slotHits.length; i++) {
        const s: string = slotHits[i];
        const k: string = this.kvField(s, 'role') + '@' + this.kvField(s, 'mode');
        if (!last.has(k)) {
          seen.push(k);
        }
        last.set(k, s);
      }
      const rows: ResultRow[] = [];
      for (let i = 0; i < seen.length; i++) {
        const s: string = last.get(seen[i]) ?? '';
        const landed: string = this.kvField(s, 'LANDED');
        const vk: string = this.kvField(s, 'vk');
        const vkLayers: string = this.kvField(s, 'vkLayers');
        const code: string = this.kvField(s, 'code');
        const match: string = this.kvField(s, 'match');
        const totalMs: string = this.kvField(s, 'total');
        rows.push(this.row(seen[i], code + '  ·  ' + this.fmtNum(totalMs, ' ms'),
          match === '1' ? '' : 'warn'));
        rows.push(this.row('    落点 / vk / 层覆盖',
          landed + '  ·  vk=' + vk + '  ·  ' + vkLayers, this.landingTone('gpu', landed)));
      }
      this.pushGroup('g-acc-ncnn', 'NCNN 槽位 · 三模型 × (CPU / Vulkan)',
        'GPU 通路自证', 'ok', rows);
    }
  }

  /** 分组状态灯：有任何 bad → bad；有 warn → warn；有读数 → ok；否则 idle。 */
  private toneFromAcc(rows: ResultRow[]): string {
    let warn: boolean = false;
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].tone === 'bad') {
        return 'bad';
      }
      if (rows[i].tone === 'warn') {
        warn = true;
      }
    }
    if (warn) {
      return 'warn';
    }
    return rows.length > 0 ? 'ok' : 'idle';
  }

  // ------------------------------------------------------------ 解析：RQ4 热

  /**
   * RQ4 热特性：把逐轮 `RQ4 r=.. t=.. thermal=.. battC=.. soc=.. chg=.. load=..
   * totalMs=.. code=.. backends=..` 聚合成可读的几行。
   *
   * 【唯一允许的统计】稳态段斜率：取**后 2/3** 轮的 totalMs 做一次最小二乘，
   * 报 ms/轮。理由 —— RQ4 要回答的是「NPU 收益能否维持」，而"维持得住吗"
   * 的判据就是延迟有没有单调爬升；报首末两点的差会被单轮噪声主导，
   * 报均值则完全看不出漂移。**用后 2/3 是为了避开预热段**（前 1/3 还在
   * 从冷跑到热，斜率会被那段带偏）。这一条的取舍在 ADR-0008 里记着。
   *
   * 除此之外一切数值都是**末轮实测值**或**计数**，不做外推。
   */
  private parseRq4Group(): void {
    const hits: string[] = this.linesWith('RQ4 r=');
    if (hits.length === 0) {
      return;
    }
    const rounds: number = hits.length;
    const first: string = hits[0];
    const last: string = hits[rounds - 1];
    const lastTotal: string = this.kvField(last, 'totalMs');
    const lastThermal: string = this.kvField(last, 'thermal');
    const lastBatt: string = this.kvField(last, 'battC');
    const lastSoc: string = this.kvField(last, 'soc');
    const lastChg: string = this.kvField(last, 'chg');
    const lastLoad: string = this.kvField(last, 'load');
    const lastCode: string = this.kvField(last, 'code');
    const lastBackends: string = this.kvField(last, 'backends');
    const firstTotal: string = this.kvField(first, 'totalMs');
    const firstThermal: string = this.kvField(first, 'thermal');

    // 稳态段 = 后 2/3
    const startIdx: number = Math.floor(rounds / 3);
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = startIdx; i < rounds; i++) {
      const t: number = Number.parseFloat(this.kvField(hits[i], 'totalMs'));
      if (!Number.isNaN(t)) {
        xs.push(i);
        ys.push(t);
      }
    }
    const rows: ResultRow[] = [];
    rows.push(this.row('轮数', rounds.toString() + ' 轮 · 每 15 s 一轮', ''));
    rows.push(this.row('首轮 / 末轮端到端',
      this.fmtNum(firstTotal, ' ms') + '  →  ' + this.fmtNum(lastTotal, ' ms'), ''));
    if (xs.length >= 4) {
      let sx: number = 0;
      let sy: number = 0;
      let sxy: number = 0;
      let sxx: number = 0;
      for (let i = 0; i < xs.length; i++) {
        sx += xs[i];
        sy += ys[i];
        sxy += xs[i] * ys[i];
        sxx += xs[i] * xs[i];
      }
      const n: number = xs.length;
      const den: number = n * sxx - sx * sx;
      const slope: number = den === 0 ? 0 : (n * sxy - sx * sy) / den;
      rows.push(this.row('稳态斜率（后 2/3 轮最小二乘）',
        (slope >= 0 ? '+' : '') + slope.toFixed(3) + ' ms/轮', ''));
    }
    rows.push(this.row('热档', firstThermal + '  →  ' + lastThermal, ''));
    rows.push(this.row('电池温度', lastBatt + ' ℃  ·  SOC ' + lastSoc + '%  ·  充电 ' + lastChg, ''));
    rows.push(this.row('系统负载档', lastLoad, ''));
    rows.push(this.row('末轮车牌', lastCode, ''));
    rows.push(this.row('末轮落点', lastBackends, lastBackends.indexOf('NPU') >= 0 ||
      lastBackends.indexOf('NNRT') >= 0 ? 'ok' : 'warn'));
    rows.push(this.row('状态', this.rq4Status.length > 0 ? this.rq4Status : '—', 'sub'));
    // 热档没跨过 2→3 台阶是老数据的已知缺陷（t7-rq4-thermal.md 自己记着），
    // 面板直接把它标出来，免得又一次拿恒为 1 的 thermal 去谈"热特性"。
    if (lastThermal === firstThermal) {
      rows.push(this.row('⚠ 热档未变化',
        'thermal 全程恒为 ' + lastThermal + ' —— 未跨过台阶，不足以支撑"热特性"结论', 'bad'));
    }
    this.pushGroup('g-rq4', 'RQ4 · 持续负载热特性',
      rounds.toString() + ' 轮', firstThermal === lastThermal ? 'warn' : 'ok', rows);
  }

  // ------------------------------------------------------------ 解析：T8 工具链

  /**
   * T8 工具链联合属性：算子对撞 + CANN/NNRt + GPU/Vulkan。
   *
   * 这一组的论点是**「支持性是模型 × 工具链的联合属性」** —— 所以面板必须
   * 同时显示「多少算子建图成功」和「同一批算子在 .ms 侧建不出来的那几个」，
   * 只报一个数字就把论点丢了。
   */
  private parseT8Group(): void {
    const rows: ResultRow[] = [];

    // ---- 算子对撞：逐条 `OPCOLLIDE op=<name> bytes=<n> <payload>`
    const opHits: string[] = this.linesWith('OPCOLLIDE op=');
    if (opHits.length > 0) {
      let ok: number = 0;
      let fail: number = 0;
      const failNames: string[] = [];
      for (let i = 0; i < opHits.length; i++) {
        const s: string = opHits[i];
        if (s.indexOf('ok=1') >= 0) {
          ok++;
        } else {
          fail++;
          const opName: string = this.kvField(s, 'op');
          if (failNames.length < 6) {
            failNames.push(opName);
          }
        }
      }
      rows.push(this.row('算子对撞', opHits.length.toString() + ' 个探针 · ok=' + ok +
        ' · fail=' + fail, fail > 0 ? 'warn' : 'ok'));
      if (failNames.length > 0) {
        rows.push(this.row('建图失败（前 6）', failNames.join(', '), 'bad'));
      }
      rows.push(this.row('清单总数（硬编码）', OP_PROBE_NAMES.length.toString() +
        ' 个 · 见 OP_PROBE_NAMES', 'label'));
    }

    // ---- CANN / NNRt 设备枚举
    const enumLine: string = this.lastLineWith('CANN ENUM ');
    if (enumLine.length > 0) {
      rows.push(this.row('NNRt 设备', enumLine.replace('CANN ENUM ', ''), 'label'));
    }
    const cannTry: string[] = this.linesWith('CANN TRY ');
    if (cannTry.length > 0) {
      let accepted: number = 0;
      const acceptedNames: string[] = [];
      for (let i = 0; i < cannTry.length; i++) {
        const s: string = cannTry[i];
        if (s.indexOf('ok=1') >= 0) {
          accepted++;
          acceptedNames.push(this.kvField(s, 'model'));
        }
      }
      rows.push(this.row('offline model 入口收不收',
        cannTry.length.toString() + ' 份试投 · 接受 ' + accepted + ' 份',
        accepted > 0 ? 'ok' : 'warn'));
      if (acceptedNames.length > 0) {
        rows.push(this.row('被接受的模型', acceptedNames.join(', '), 'ok'));
      }
    }

    // ---- GPU 专项：MS Lite 的 gpu 档（编译期判否 → 静默回落）
    const gpuHits: string[] = this.linesWith('MATRIX ');
    let gpuReq: number = 0;
    let gpuLanded: number = 0;
    for (let i = 0; i < gpuHits.length; i++) {
      const s: string = gpuHits[i];
      const parts: string[] = s.split(' ');
      if (parts.length >= 3 && parts[2] === 'gpu') {
        gpuReq++;
        if (this.isNpuLanding(this.kvField(s, 'LANDED'))) {
          gpuLanded++;
        }
      }
    }
    if (gpuHits.length > 0) {
      rows.push(this.row('MS Lite gpu 档', '请求 ' + gpuReq + ' 次 · 真落到加速器 ' +
        gpuLanded + ' 次', gpuLanded === 0 && gpuReq > 0 ? 'warn' : ''));
      rows.push(this.row('说明',
        'MS Lite 的 GPU 是编译期判否后静默回落 CPU；GPU 真通路是 ncnn-Vulkan', 'label'));
    }

    // ---- Vulkan 探针（原生 dlopen loader 的结果，切块打进 hilog）
    const vkHits: string[] = this.linesWith('VULKAN PROBE ');
    if (vkHits.length > 0) {
      const joined: string = vkHits.join('');
      rows.push(this.row('Vulkan loader', joined.indexOf('"ok":1') >= 0 ||
        joined.indexOf('ok=1') >= 0 ? 'dlopen 成功' : '见原始输出', 'label'));
      rows.push(this.row('探针块数', vkHits.length.toString() + ' 块（hilog 单行上限切块）', 'sub'));
    }

    // ---- ncnn 裸 head 自证
    const ncnnLoad: string = this.lastLineWith('NCNN LOAD ');
    if (ncnnLoad.length > 0) {
      rows.push(this.row('ncnn head 载入', ncnnLoad.replace('NCNN LOAD ', ''), 'label'));
    }
    const ncnnRun: string = this.lastLineWith('NCNN RUN ');
    if (ncnnRun.length > 0) {
      rows.push(this.row('ncnn 裸跑', ncnnRun.replace('NCNN RUN ', ''), 'sub'));
    }

    // ---- 检测器数值漂移（同一份裸 head，nnrt vs cpu）
    const drift: string[] = this.linesWith('DET DRIFT ');
    if (drift.length > 0) {
      for (let i = 0; i < drift.length; i++) {
        rows.push(this.row('det 漂移', drift[i].replace('DET DRIFT ', ''), 'sub'));
      }
    }

    if (rows.length > 0) {
      this.pushGroup('g-t8', 'T8 · 工具链联合属性',
        '算子 × 工具链', this.toneFromAcc(rows), rows);
    }
  }

  /** 从 lines 重建全部结构化结果。每次探针跑完调一次。 */
  private rebuildResults(): void {
    const keep: ResultGroup[] = this.results;
    this.results = [];
    // 顺序即论文顺序：RQ3 → 精度 → T8 → RQ4。
    this.buildRq3Group();
    this.parseAccGroups();
    this.parseT8Group();
    this.parseRq4Group();
    // 若一组都没建出来（比如只跑了 Vulkan 探针），保留原样。
    if (this.results.length === 0) {
      this.results = keep;
    }
  }

  /** RQ3 延迟分解组：端到端矩阵 + 逐模型矩阵 + 三张扫描表。 */
  private buildRq3Group(): void {
    const rows: ResultRow[] = [];
    const e2e: ResultRow[] = this.parseE2eRows();
    if (e2e.length > 0) {
      rows.push(this.row('端到端矩阵（逐角色换落点）', '', 'label'));
      for (let i = 0; i < e2e.length; i++) {
        rows.push(e2e[i]);
      }
    }
    const matrix: ResultRow[] = this.parseMatrixRows();
    if (matrix.length > 0) {
      rows.push(this.row('逐模型矩阵（逐模型换后端）', '', 'label'));
      for (let i = 0; i < matrix.length; i++) {
        rows.push(matrix[i]);
      }
    }
    const devices: string = this.lastLineWith('MATRIX NNRT-DEVICES ');
    if (devices.length > 0) {
      rows.push(this.row('NNRt 可用设备', devices.replace('MATRIX NNRT-DEVICES ', ''), 'label'));
    }
    if (rows.length === 0) {
      return;
    }
    // 端到端矩阵是 RQ3 的主证据；它的 match 不一致直接决定分组灯色。
    let anyWarn: boolean = false;
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].tone === 'bad') {
        this.pushGroup('g-rq3', 'RQ3 · 端到端延迟分解',
          rows.length.toString() + ' 项', 'bad', rows);
        return;
      }
      if (rows[i].tone === 'warn') {
        anyWarn = true;
      }
    }
    this.pushGroup('g-rq3', 'RQ3 · 端到端延迟分解',
      rows.length.toString() + ' 项', anyWarn ? 'warn' : 'ok', rows);
  }

  // ------------------------------------------------------------ 编排

  /**
   * 跑一个分组。
   *
   * 【为什么分组是这样切的】ADR-0008 定的三条准则：
   *   论文要引用其数据的 → 留；纯调试用 → 砍；依赖摄像头的 → 砍（云手机没摄像头，
   *   而用户现在也只有这一台真机）。留下来的正好按 RQ 分成四组。
   *
   * 【顺序】组内顺序**不能随便调**：
   *   - rq3 内部 load 必须最先（其它全依赖会话）；
   *   - acc 的三个探针各自 load 自己的会话，彼此独立，但要在 rq3 之后，
   *     因为 acc 的 crop 探针会重建 rec 会话（换落点），洗掉 rq3 的会话状态；
   *   - t8 的对撞探针每轮 load 51 个模型，会把会话表灌满，所以它必须在
   *     所有"依赖会话"的实验之后；
   *   - rq4 永远最后（热循环要给后续实验留冷基线，自己却不在乎前面积的热）。
   */
  private async runGroup(g: string): Promise<void> {
    if (this.busy) {
      return;
    }
    this.group = g;
    this.runState = 'run';
    if (g === GRP_RQ3) {
      await this.probeRq3();
    } else if (g === GRP_ACC) {
      await this.probeAccuracy();
    } else if (g === GRP_T8) {
      await this.probeT8();
    } else {
      await this.probeRq4();
    }
    this.rebuildResults();
    this.runState = 'ok';
  }

  /**
   * 一键全跑。
   *
   * 【顺序是 ADR-0008 里定的，不要调】load → RQ3 → 精度 → T8 → RQ4。
   * 前四步都要会话且彼此抢会话表，RQ4 只关心热、必须收尾。这条链上
   * 任何一步失败都不中断后续 —— 一次装机后的长跑机会很贵，跑完能跑的
   * 比中途停在第一步有用。
   */
  private async runAll(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.runState = 'run';
    this.currentExp = '加载模型';
    this.elapsed = '00:00';
    this.progress = '';
    this.lines = [];
    this.results = [];
    const t0: number = Date.now();
    this.tickClock(t0);
    this.appendNative('RUNALL BEGIN ' + new Date().toISOString());

    this.currentExp = 'RQ3 · 延迟分解';
    await this.probeRq3();
    this.rebuildResults();

    this.currentExp = '精度 · A/B 对照';
    await this.probeAccuracy();
    this.rebuildResults();

    this.currentExp = 'T8 · 工具链联合属性';
    await this.probeT8();
    this.rebuildResults();

    this.currentExp = 'RQ4 · 持续负载热特性';
    await this.probeRq4();
    this.rebuildResults();

    const total: number = Date.now() - t0;
    this.currentExp = '全部完成';
    this.elapsed = this.hhmmss(total);
    this.appendNative('RUNALL END totalMs=' + total.toString());
    this.runState = 'ok';
    this.busy = false;
  }

  /** ① RQ3 组：加载生产会话 → 端到端矩阵 → 逐模型矩阵 → 三张扫描表。 */
  private async probeRq3(): Promise<void> {
    if (this.ctx === undefined) {
      this.appendNative('RQ3 FAIL no context');
      return;
    }
    this.progress = '加载模型';
    await this.loadNative();
    this.progress = '端到端矩阵';
    await this.e2eMatrix();
    this.progress = '逐模型矩阵';
    await this.probeAll();
    this.progress = '线程扫描';
    await this.threadSweep();
    this.progress = '亲和扫描';
    await this.affinitySweep();
    this.progress = '间隔扫描';
    await this.gapSweep();
    this.progress = '';
  }

  /**
   * ② 精度组。
   *
   * 【为什么这四个都跑】它们回答的是四个**不同**的问题，不能互相替代：
   *   - crop rec/det：换落点会不会改变输出（同工况下的分歧率）
   *   - bare rec：跳过 det/rectify 后识别器本身的准确率（与 PC 侧 A16 可比）
   *   - scene ccpd/green：真实整车工况下谁更敏感（det vs rec 必须都跑，
   *     因为跨工况不可比 —— T12 已证明这一点）
   *
   * 【时长】这个组是全流程最贵的：crop 1000 张 × 2 落点、scene 1000×2×2、
   * bare 1000。全跑约 20–30 分钟量级。
   */
  private async probeAccuracy(): Promise<void> {
    this.progress = '裁剪图集 rec A/B';
    await this.cropBackendAB('rec');
    this.progress = '裁剪图集 det A/B';
    await this.cropBackendAB('det');
    this.progress = '识别器裸头准确率';
    await this.bareHeadAccuracy();
    this.progress = '整车场景 ccpd';
    await this.ccpdSceneAB('det', 'ccpd');
    await this.ccpdSceneAB('rec', 'ccpd');
    this.progress = '整车场景 green';
    await this.ccpdSceneAB('det', 'green');
    await this.ccpdSceneAB('rec', 'green');
    this.progress = 'ncnn 槽位';
    await this.ncnnSlotProbe();
    this.progress = '';
  }

  /** ③ T8 组：算子对撞（51 个）→ CANN/NNRt → GPU/Vulkan → ncnn 自证。 */
  private async probeT8(): Promise<void> {
    this.progress = '算子覆盖对撞';
    await this.opCollideProbe();
    this.progress = 'CANN / NNRt';
    await this.cannProbe();
    this.progress = 'GPU 专项';
    await this.gpuProbe();
    this.progress = '检测器漂移';
    await this.detDriftProbe();
    this.progress = 'ncnn 自证';
    await this.ncnnProbe();
    this.progress = 'Vulkan 探针';
    this.vulkanProbeRun();
    this.progress = '';
    this.lines = this.lines.concat(['T8 GROUP END ts=' + Date.now().toString()]);
  }

  /**
   * ④ RQ4 组：持续负载热循环。
   *
   * 【必须自己加载模型】不能指望前面留下的会话：热循环要跑 20 分钟，
   * 期间如果会话被别的探针换过（比如 acc 的 crop 探针把 rec 换到了 cpu），
   * 整个基线就悄悄变了 —— 而 RQ4 的全部意义就是基线可比。
   */
  private async probeRq4(): Promise<void> {
    if (this.ctx === undefined) {
      this.appendNative('RQ4 GROUP FAIL no context');
      return;
    }
    this.progress = '加载生产会话';
    await this.loadNative();
    this.progress = '热循环 ' + RQ4_MINUTES.toString() + ' 分钟';
    await this.rq4SustainedProbe(RQ4_MINUTES);
    this.progress = '';
  }

  /** 跑一个分组（对外入口，含 busy 保护与结果刷新）。 */
  private async runGroupSafe(g: string): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    this.runState = 'run';
    const t0: number = Date.now();
    this.tickClock(t0);
    try {
      await this.runGroup(g);
    } catch (e) {
      const err = e as Error;
      this.appendNative('GROUP EXCEPTION ' + g + ' ' + err.message);
      this.runState = 'bad';
    }
    this.elapsed = this.hhmmss(Date.now() - t0);
    if (this.runState === 'ok') {
      this.runState = 'ok';
    }
    this.busy = false;
  }

  /** 时钟：每 500 ms 刷一次 elapsed，让长跑看得出还在动。 */
  private tickClock(t0: number): void {
    const step: () => void = () => {
      if (!this.busy) {
        return;
      }
      this.elapsed = this.hhmmss(Date.now() - t0);
      setTimeout(step, 500);
    };
    setTimeout(step, 500);
  }

  private hhmmss(ms: number): string {
    const s: number = Math.floor(ms / 1000);
    const h: number = Math.floor(s / 3600);
    const m: number = Math.floor((s % 3600) / 60);
    const sec: number = s % 60;
    const two = (v: number): string => v < 10 ? '0' + v.toString() : v.toString();
    return h > 0 ? (two(h) + ':' + two(m) + ':' + two(sec)) : (two(m) + ':' + two(sec));
  }

  // ------------------------------------------------------------ 导出

  /**
   * 复制全部结果到剪贴板 —— **这就是"一键导出"**。
   *
   * 为什么不写文件：用户明确要「终端界面上直接看 + 复制」；而且云手机/真机的
   * App 私有目录 hdc 拉不出来（既不能 shell 写也不能 shell 读）。复制成文本后
   * 粘到哪都行，比一个拉不出来的 .log 有用。
   *
   * 输出结构固定三段，方便直接贴进论文笔记：
   *   1) 头部：时间 / 设备 / 结论一行
   *   2) 结构化结果：分组标题 + 每条读数（与面板显示逐字一致）
   *   3) 原始输出：完整 lines（不截断 —— 面板才截 300 行，复制要给全的）
   */
  private copyAll(): void {
    const now: Date = new Date();
    const parts: string[] = [];
    parts.push('=== LPR Kirin8020 探针结果 ===');
    parts.push('时间: ' + now.toLocaleString());
    parts.push('分组: ' + this.groupLabel(this.group));
    parts.push('状态: ' + (this.runState === '' ? 'idle' : this.runState) +
      '   用时: ' + (this.elapsed.length > 0 ? this.elapsed : '-'));
    parts.push('结果组数: ' + this.results.length.toString() +
      '   原始输出: ' + this.lines.length.toString() + ' 行');
    parts.push('');

    if (this.results.length > 0) {
      parts.push('---------- 结构化结果 ----------');
      for (let i = 0; i < this.results.length; i++) {
        const g: ResultGroup = this.results[i];
        parts.push('');
        parts.push('## ' + g.title + '  [' + g.tag + ']');
        for (let j = 0; j < g.rows.length; j++) {
          const r: ResultRow = g.rows[j];
          if (r.value.length === 0) {
            parts.push(r.label);
          } else {
            parts.push(r.label + ' : ' + r.value);
          }
        }
      }
      parts.push('');
    }

    if (this.lines.length > 0) {
      parts.push('---------- 原始输出（完整 ' +
        this.lines.length.toString() + ' 行） ----------');
      for (let i = 0; i < this.lines.length; i++) {
        parts.push(this.lines[i]);
      }
    }

    const text: string = parts.join('\n');
    try {
      pasteboard.createPlainTextData(text);
      const board: pasteboard.SystemPasteboard = pasteboard.getSystemPasteboard();
      board.setData(pasteboard.createPlainTextData(text)).then(() => {
        this.progress = '已复制 ' + text.length.toString() + ' 字符';
        this.appendNative('COPY OK chars=' + text.length.toString() +
          ' groups=' + this.results.length.toString() +
          ' lines=' + this.lines.length.toString());
      }).catch((e: Error) => {
        this.appendNative('COPY FAIL ' + e.message);
        this.progress = '复制失败';
      });
    } catch (e) {
      const err = e as Error;
      this.appendNative('COPY EXCEPTION ' + err.message);
      this.progress = '复制失败';
    }
  }

  /** 清空结果与原始输出。不清 busy —— busy 时按钮本来就不可点。 */
  private clearAll(): void {
    this.lines = [];
    this.results = [];
    this.progress = '';
    this.currentExp = '';
    this.elapsed = '';
    this.runState = '';
    this.rq4Rows = [];
    this.rq4Status = '';
    this.appendNative('CLEAR ts=' + Date.now().toString());
  }

  // ============================================================ 界面层
  //
  // 重设计见 docs/adr/0008-probe-console-redesign.md。
  // 旧界面是 26 个按钮平铺 + 两个 tab，问题是「看不了 / 认不出 / 跑不动」。
  // 本层只做三件事：**分组、呈现、编排** —— 不改任何测量逻辑（上面 2100 行）。
  //
  // 【三区结构】
  //   ① 顶部状态条  —— 跑没跑 / 跑到哪 / 跑了多久（旧界面这些信息沉在 lines 末尾）
  //   ② 中部结果区  —— 结构化分组结果 + 原始输出；**长按整区复制 = 导出**
  //   ③ 底部操作区  —— 4 个分组 chip + 一键全跑 / 跑当前组 / 清空
  //
  // 【为什么没有「导出文件」】用户要的是「终端界面上直接看 + 复制」。云手机场景下
  // hdc 拉 filesDir 本来也不通（App 私有目录既不能 shell 写也不能 shell 读），
  // 所以导出就是**长按复制到剪贴板**，粘贴出来是可直接贴进文档的纯文本块。

  // ---------------------------------------------------------- ⓪ 页头 / 相机入口

  /**
   * 页头：App 身份 + **相机实时页入口**。
   *
   * 【为什么加在这里】上一轮按 ADR-0008 把相机页整页砍掉时，本文件也顺手删了
   * `router` import 与所有跳转。用户随后要求「恢复相机实时」（spec 见
   * `docs/spec-vehicle-pipeline.md`）。入口加在**深色仪器风版本**上，
   * 而不是回退旧的浅色版 —— 这是 T1 票的硬要求（不得回退 UI 重设计）。
   *
   * 【为什么单独一行，而不是塞进 statusBar】
   * `statusBar` 那一行全是**读数**（状态灯 / 已用时）。把动作混进读数行，
   * 「相机」会被读成又一个指标。单独一行既给了页面身份，也让入口一眼可寻 ——
   * 旧版把 26 个操作平铺在一页里，找什么都难，这次不重复那个错。
   */
  @Builder
  pageHeader() {
    Row({ space: 10 }) {
      Text('车牌识别 · 探针台')
        .fontSize(13)
        .fontWeight(FontWeight.Medium)
        .fontColor(INS_LABEL2)
        .fontFamily(INS_FONT)
        .layoutWeight(1)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })

      Text('相机 ›')
        .fontSize(12)
        .fontWeight(FontWeight.Medium)
        .fontColor(INS_ACCENT)
        .fontFamily(INS_FONT)
        .textAlign(TextAlign.Center)
        .height(30)
        .padding({ left: 14, right: 14 })
        .backgroundColor(INS_ACCENT_SOFT)
        .borderRadius(8)
        .onClick(() => {
          this.openCamera();
        })
    }
    .width('100%')
    .padding({ left: 2, right: 2 })
  }

  // ---------------------------------------------------------- ① 顶部状态条

  /** 状态灯：idle / run / ok / warn / bad。 */
  @Builder
  statusDot(state: string) {
    Row()
      .width(7)
      .height(7)
      .borderRadius(4)
      .backgroundColor(state === 'run' ? INS_ACCENT
        : state === 'ok' ? INS_OK
          : state === 'warn' ? INS_WARN
            : state === 'bad' ? INS_BAD
              : INS_IDLE)
  }

  /**
   * 分组 chip。label 是人读的短名，tag 是它在论文里对应什么。
   * 双行是有意的：单行 chip 在 360vp 宽里放不下「RQ3 延迟分解」这种有信息量的名字。
   */
  @Builder
  groupChip(label: string, tag: string, active: boolean, onTap: () => void) {
    Column({ space: 2 }) {
      Text(label)
        .fontSize(13)
        .fontWeight(active ? FontWeight.Medium : FontWeight.Normal)
        .fontColor(active ? INS_BG : INS_LABEL)
        .fontFamily(INS_FONT)
      Text(tag)
        .fontSize(9)
        .fontColor(active ? INS_BG : INS_LABEL3)
        .fontFamily(INS_MONO)
    }
    .padding({ left: 11, right: 11, top: 7, bottom: 7 })
    .backgroundColor(active ? INS_ACCENT : INS_PANEL2)
    .borderRadius(8)
    .onClick(onTap)
  }

  /** 顶部状态条：一眼看出「跑没跑 / 跑到哪 / 跑了多久」。 */
  @Builder
  statusBar() {
    Column({ space: 8 }) {
      Row({ space: 8 }) {
        this.statusDot(this.busy ? 'run' : this.runState)
        Text(this.busy ? '运行中'
          : this.runState === 'ok' ? '已完成'
            : this.runState === 'bad' ? '出错'
              : this.runState === 'warn' ? '有异常'
                : '待机')
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(this.busy ? INS_ACCENT : INS_LABEL)
          .fontFamily(INS_FONT)
        Blank()
        Text(this.elapsed.length > 0 ? this.elapsed : '—')
          .fontSize(12)
          .fontColor(INS_LABEL2)
          .fontFamily(INS_MONO)
      }
      .width('100%')

      Row({ space: 8 }) {
        Text(this.currentExp.length > 0 ? this.currentExp : '未开始')
          .fontSize(12)
          .fontColor(INS_LABEL2)
          .fontFamily(INS_FONT)
          .layoutWeight(1)
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
        Text(this.progress.length > 0 ? this.progress : '')
          .fontSize(12)
          .fontColor(INS_LABEL3)
          .fontFamily(INS_MONO)
      }
      .width('100%')
    }
    .width('100%')
    .padding({ left: 14, right: 14, top: 11, bottom: 11 })
    .backgroundColor(INS_PANEL)
    .borderRadius(10)
  }

  // ---------------------------------------------------------- ② 结果区

  /**
   * 单条结果行。左标签右值，值等宽且带状态色。
   *
   * 等宽的意义在**竖着扫**：一列 p50 右对齐后小数点能对齐，比浅色卡片里
   * 的散排数字快得多。这是深色仪器风相对旧界面的主要收益。
   */
  @Builder
  resultRow(label: string, value: string, tone: string) {
    Row({ space: 10 }) {
      Text(label)
        .fontSize(12)
        .fontColor(tone === 'sub' ? INS_LABEL3 : INS_LABEL2)
        .fontFamily(tone === 'sub' ? INS_MONO : INS_FONT)
        .layoutWeight(1)
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .padding({ left: tone === 'sub' ? 8 : 0 })
      Text(value)
        .fontSize(12)
        .fontColor(tone === 'ok' ? INS_OK : tone === 'warn' ? INS_WARN
          : tone === 'bad' ? INS_BAD
            : tone === 'sub' ? INS_LABEL3
              : tone === 'label' ? INS_LABEL2 : INS_LABEL)
        .fontFamily(INS_MONO)
    }
    .width('100%')
    .padding({ left: 12, right: 12, top: 5, bottom: 5 })
  }

  /** 结果分组卡：一个实验一块。 */
  @Builder
  resultGroupCard(g: ResultGroup) {
    Column({ space: 0 }) {
      Row({ space: 8 }) {
        this.statusDot(g.tone === '' ? 'idle' : g.tone)
        Text(g.title)
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(INS_LABEL)
          .fontFamily(INS_FONT)
          .layoutWeight(1)
        Text(g.tag)
          .fontSize(10)
          .fontColor(INS_LABEL3)
          .fontFamily(INS_MONO)
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 9, bottom: 7 })

      ForEach(g.rows, (r: ResultRow) => {
        this.resultRow(r.label, r.value, r.tone)
      }, (r: ResultRow, i: number) => g.key + '#' + i.toString())
    }
    .width('100%')
    .margin({ bottom: 8 })
    .backgroundColor(INS_PANEL2)
    .borderRadius(8)
  }

  /**
   * 结果面板：**结构化结果在前，原始输出在后**。
   *
   * 旧界面只有一坨等宽 `lines`，26 个实验的顺序输出混在一起 —— 这就是
   * 「标签看不了」的根源：找不到哪一行属于哪个实验，也没有单位。
   * 现在每个探针在跑的过程中往 `results` 里填**带标签的读数**，
   * 原始 `lines` 保留在下半段以备核对（它仍是落盘日志的同源副本）。
   *
   * **长按整区 = 复制全部**（本 App 的导出方式）。
   */
  @Builder
  resultPanel() {
    Column({ space: 0 }) {
      Row({ space: 8 }) {
        Text('结果')
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(INS_LABEL)
          .fontFamily(INS_FONT)
        if (this.results.length > 0) {
          Text(this.results.length.toString() + ' 组')
            .fontSize(10)
            .fontColor(INS_LABEL3)
            .fontFamily(INS_MONO)
        }
        Blank()
        Text(this.busy ? '运行中…' : '长按复制全部')
          .fontSize(10)
          .fontColor(INS_LABEL3)
          .fontFamily(INS_FONT)
      }
      .width('100%')
      .padding({ left: 12, right: 12, top: 10, bottom: 9 })

      Divider().color(INS_LINE).strokeWidth(1)

      if (this.results.length === 0 && this.lines.length === 0) {
        Column({ space: 8 }) {
          Text('尚无结果')
            .fontSize(13)
            .fontColor(INS_LABEL3)
            .fontFamily(INS_FONT)
          Text('点下方「一键全跑」按论文顺序跑完全部实验，')
            .fontSize(11)
            .fontColor(INS_LABEL3)
            .fontFamily(INS_FONT)
          Text('或先选一个分组，点「跑当前组」单独复跑。')
            .fontSize(11)
            .fontColor(INS_LABEL3)
            .fontFamily(INS_FONT)
        }
        .width('100%')
        .padding({ top: 40, bottom: 40 })
      } else {
        Scroll() {
          Column({ space: 0 }) {
            ForEach(this.results, (g: ResultGroup) => {
              this.resultGroupCard(g)
            }, (g: ResultGroup) => g.key)

            if (this.lines.length > 0) {
              Column({ space: 0 }) {
                Row({ space: 8 }) {
                  Text('原始输出')
                    .fontSize(11)
                    .fontWeight(FontWeight.Medium)
                    .fontColor(INS_LABEL3)
                    .fontFamily(INS_FONT)
                  Blank()
                  Text('末 ' + Math.min(this.lines.length, RESULT_RAW_TAIL).toString() +
                    ' / ' + this.lines.length.toString() + ' 行')
                    .fontSize(10)
                    .fontColor(INS_LABEL3)
                    .fontFamily(INS_MONO)
                }
                .width('100%')
                .padding({ left: 12, right: 12, top: 10, bottom: 6 })

                ForEach(this.rawTail(), (s: string) => {
                  Text(s)
                    .fontSize(10)
                    .fontColor(INS_LABEL2)
                    .fontFamily(INS_MONO)
                    .width('100%')
                    .padding({ left: 12, right: 12, top: 2, bottom: 2 })
                }, (s: string, i: number) => 'raw#' + i.toString())
              }
              .width('100%')
              .margin({ top: 4 })
            }
          }
          .width('100%')
          .padding({ top: 8, bottom: 8 })
        }
        .layoutWeight(1)
        .width('100%')
        .align(Alignment.Top)
        .scrollBar(BarState.Auto)
      }
    }
    .layoutWeight(1)
    .width('100%')
    .backgroundColor(INS_PANEL)
    .borderRadius(10)
    .clip(true)
    .gesture(
      LongPressGesture()
        .onAction(() => {
          this.copyAll();
        })
    )
  }

  // ---------------------------------------------------------- ③ 操作区

  /** 主按钮。 */
  @Builder
  primaryButton(label: string, enabled: boolean, onTap: () => void) {
    Text(label)
      .fontSize(14)
      .fontWeight(FontWeight.Medium)
      .fontColor(enabled ? '#0B0E12' : INS_LABEL3)
      .fontFamily(INS_FONT)
      .textAlign(TextAlign.Center)
      .width('100%')
      .height(46)
      .backgroundColor(enabled ? INS_ACCENT : INS_PANEL2)
      .borderRadius(10)
      .onClick(() => {
        if (enabled) {
          onTap();
        }
      })
  }

  /**
   * 底部操作区（固定，不随结果滚动跑掉）。
   *
   * 旧界面的操作按钮和结果混在同一个 Scroll 里，长跑跑到一半想清空要滑很久；
   * 且 26 个按钮没有主次，找「跑全部」比找任意一个探针都难。
   */
  @Builder
  actionBar() {
    Column({ space: 8 }) {
      Row({ space: 6 }) {
        this.groupChip('RQ3', '延迟分解', this.group === 'rq3', () => {
          this.group = 'rq3';
        })
        this.groupChip('RQ4', '持续负载', this.group === 'rq4', () => {
          this.group = 'rq4';
        })
        this.groupChip('精度', 'A/B 对照', this.group === 'acc', () => {
          this.group = 'acc';
        })
        this.groupChip('T8', '工具链', this.group === 't8', () => {
          this.group = 't8';
        })
      }
      .width('100%')

      Row({ space: 8 }) {
        Column() {
          this.primaryButton(
            this.busy ? ('运行中 · ' + this.progress) : '一键全跑',
            !this.busy,
            () => {
              this.runAll();
            })
        }
        .layoutWeight(1)

        Text('跑当前组')
          .fontSize(12)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(46)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(10)
          .onClick(() => {
            if (!this.busy) {
              this.runGroup(this.group);
            }
          })

        Text('复制')
          .fontSize(12)
          .fontColor(INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(46)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(10)
          .onClick(() => {
            this.copyAll();
          })
      }
      .width('100%')

      // 提示单独占一行：按钮从 4 个涨到 6 个后，它跟按钮抢同一行会被压成一条缝。
      Text('RQ4 热循环必须最后跑（先跑会把热态带上基线）。当前组：' +
        this.groupLabel(this.group))
        .fontSize(9)
        .fontColor(INS_LABEL3)
        .fontFamily(INS_FONT)
        .width('100%')
        .maxLines(2)

      Row({ space: 6 }) {
        Text('车辆检测')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.vehicleSelfTest();
            }
          })

        Text('ROI 自证')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.roiSelfTest();
            }
          })

        Text('ROI 探针')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.roiPlateProbe();
            }
          })

        Text('ROI 路径')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.roiPipelineRun();
            }
          })

        Text('去重自证')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_LABEL)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.roiDedupeSelfTest();
            }
          })

        Text('清空')
          .fontSize(11)
          .fontColor(this.busy ? INS_LABEL3 : INS_BAD)
          .fontFamily(INS_FONT)
          .textAlign(TextAlign.Center)
          .height(28)
          .padding({ left: 12, right: 12 })
          .backgroundColor(INS_PANEL2)
          .borderRadius(8)
          .onClick(() => {
            if (!this.busy) {
              this.clearAll();
            }
          })
      }
      .width('100%')
    }
    .width('100%')
    .padding({ left: 2, right: 2, top: 2 })
  }

  // ---------------------------------------------------------- 根布局

  build() {
    Column({ space: 10 }) {
      this.pageHeader()
      this.statusBar()
      this.resultPanel()
      this.actionBar()
    }
    .width('100%')
    .height('100%')
    .padding({ left: 12, right: 12, top: 12, bottom: 12 })
    .backgroundColor(INS_BG)
  }
}
