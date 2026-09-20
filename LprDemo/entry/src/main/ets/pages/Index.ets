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
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { camera } from '@kit.CameraKit';
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

const TAG = 'LprMatrix';
const NATIVE_TAG = 'LprNativeUI';
const DOMAIN = 0xD001;

// ---------------------------------------------------------------- iOS 设计变量
// 与 CameraPage.ets 同一套（Apple HIG 系统色 + SF Pro 字体栈），
// 两个页面必须视觉一致，否则来回跳转像两个 App。

/** 系统分组背景（iOS 浅色模式：F2F2F7）。 */
const IOS_BG: string = '#F2F2F7';
/** 卡片/分组内容背景。 */
const IOS_CARD: string = '#FFFFFF';
/** 主文本（label）。 */
const IOS_LABEL: string = '#000000';
/**
 * 次级文本（secondaryLabel）。
 * ⚠️ 必须用 `rgba(...)` 而不是 8 位 hex：ArkUI 的 8 位 hex 是 **ARGB**（#AARRGGBB），
 * 而 iOS 文档给的是 **RGBA**（#RRGGBBAA）。照抄 iOS 的 8 位 hex 会被解析成
 * A=RR、R=GG、G=BB、B=AA —— 实测把 `#7676801f` 渲染成了一层橄榄绿。
 */
const IOS_LABEL2: string = 'rgba(60,60,67,0.60)';
/** 三级文本（tertiaryLabel）。 */
const IOS_LABEL3: string = 'rgba(60,60,67,0.30)';
/** 分隔线（separator）。 */
const IOS_SEP: string = 'rgba(60,60,67,0.12)';
/** 系统蓝（systemBlue）—— 主操作色。 */
const IOS_BLUE: string = '#007AFF';
/** 系统绿（systemGreen）。 */
const IOS_GREEN: string = '#34C759';
/** 系统橙（systemOrange）。 */
const IOS_ORANGE: string = '#FF9500';
/** 系统红（systemRed）。 */
const IOS_RED: string = '#FF3B30';
/** 系统灰（systemGray）—— 未选中分段控件底。 */
const IOS_GRAY: string = '#8E8E93';
/** 分段控件轨道色（fill/tertiarySystemFill）。 */
const IOS_TRACK: string = 'rgba(118,118,128,0.12)';
/** 主按钮的浅色底（systemBlue 12%）。 */
const IOS_BLUE_SOFT: string = 'rgba(0,122,255,0.12)';
/** 危险按钮的浅色底（systemRed 12%）。 */
const IOS_RED_SOFT: string = 'rgba(255,59,48,0.12)';

/** SF Pro 字体栈：鸿蒙无 SF Pro 时回落到系统字体，视觉上仍贴近 iOS。 */
const IOS_FONT: string = 'HarmonyOS Sans, SF Pro Text, Helvetica Neue, sans-serif';

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
const OP_PROBE_NAMES: string[] = [
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
 * 启动自检：装机后自动「Vulkan 探针 → 枚举镜头 → 加载模型 → 跑样本 → 四后端对照
 * → 全量矩阵」，全部写 hilog。
 *
 * 存在的理由：原生 ArkUI 的按钮点不了（CDP 只能驱动 Web 内容，管不到原生组件），
 * 命令行验证需要一条不需要人手指的路径。演示前把它改成 false。
 *
 * 2026-09-18：默认改为 **false** —— 相机实时页（pages/CameraPage）已能通过
 * `hdc shell uitest uiInput click` 驱动，不需要靠自检兜底；而且自检要跑几分钟，
 * 开着会让「打开 App → 点相机」这一步变成漫长的等待。要复跑矩阵时手动改回 true。
 */
const AUTO_SELFTEST: boolean = true;
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

/**
 * 解析 native 的 `backends=` 字段为落点面板的显示行（ADR-0003）。
 *
 * 输入形如（三个角色用 `|` 分隔，字段用 `,` 分隔）：
 *   det,cpu,CPU,,1076.7692,nan,1|rec,nnrt,NNRT:NPU_...kirin8020_v2_0,,4.0489,3692.1536,1|...
 * 字段顺序：role, req, landed, fallback, l2, l2AsFp16, used
 *
 * 显示要点（这些是"榨干"这个项目的全部证据，不能糊）：
 *   - req 与 landed 不同时，明说"回落"而不是只显示 landed —— 静默回落正是要抓的东西
 *   - 落点是 NPU 时给个标记，方便一眼看出谁真的上了加速器
 *   - used=0 表示本次没调这个会话（如 det 走 ncnn 时），此时 l2 是陈旧值，不显示
 *   - l2AsFp16 不显示：它只在"声明 fp16 却按 fp32 读"的错标场景里才有意义，
 *     三个模型输出都声明 FP32 时它是垃圾（实测 nan / 10424）
 */
function evidenceRows(backends: string): string[] {
  const rows: string[] = [];
  if (backends.length === 0) {
    return rows;
  }
  for (const seg of backends.split('|')) {
    const f: string[] = seg.split(',');
    if (f.length < 7) {
      continue;
    }
    const role: string = f[0];
    // cls 角色已无模型：牌色自 2026-09-21 起走像素测量（ADR-0005），native 仍会在
    // backends= 里带一行 cls 以保持字段数稳定，但它恒为 l2=0 且 used=1（没有会话可调）。
    // 展示它会让读者以为"cls 也算了"，所以面板直接跳过。
    if (role === 'cls') {
      continue;
    }
    const req: string = f[1];
    const landed: string = f[2];
    const fallback: string = f[3];
    const l2: string = f[4];
    const used: string = f[6];
    const isNpu: boolean = landed.indexOf('NPU') >= 0 || landed.indexOf('NNRT') >= 0;
    let line: string = role + '  ' + (isNpu ? '⚡' : '  ') + 'LANDED=' + landed;
    if (req.length > 0 && req !== landed) {
      line += '  (req=' + req + (fallback.length > 0 ? ', 回落自 ' + fallback : '') + ')';
    }
    if (used === '1') {
      line += '  l2=' + l2;
    } else {
      line += '  (本次未调用)';
    }
    rows.push(line);
  }
  return rows;
}

/** 解码后的 RGBA 帧：pipeline 只吃这个。 */
interface RgbaFrame {
  data: ArrayBuffer;
  width: number;
  height: number;
}

@Entry
@Component
struct Index {
  @State lines: string[] = [];
  @State progress: string = '';
  @State busy: boolean = false;
  /** 0 = 原生演示，1 = 探针控制台。 */
  @State tab: number = 0;

  // ---- 原生演示页状态
  @State nStatus: string = '未加载模型';
  @State nBackends: string = '';
  @State nPlate: string = '';
  @State nDetail: string = '';
  @State nTiming: string = '';
  /**
   * 落点自证面板（ADR-0003）。内容直接来自 native 的 `backends=` 字段，**不重算**。
   * 每行形如：`det  req=cpu  LANDED=cpu  l2=1076.77`，回落时额外标出 `←from`。
   */
  @State nEvidence: string[] = [];
  @State nLenses: string[] = [];

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
    this.nStatus = '加载中…';
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
        this.nStatus = '加载失败';
        this.appendNative('NATIVE LOAD FAIL det=' + str(d, 'error') +
          ' rec=' + str(r, 'error') + ' cls=' + str(k, 'error'));
        return;
      }

      this.detId = num(d, 'id');
      this.recId = num(r, 'id');
      this.clsId = num(k, 'id');

      this.nBackends = 'det=' + str(d, 'backend') + '  rec=' + str(r, 'backend') +
        '  cls=' + str(k, 'backend');
      this.nStatus = '就绪（' + ((Date.now() - t0) / 1000).toFixed(1) + ' s）';
      this.appendNative('NATIVE LOAD ok ' + this.nBackends);
      this.appendNative('NATIVE TRAIL det[' + str(d, 'attemptLog') + '] rec[' +
        str(r, 'attemptLog') + '] cls[' + str(k, 'attemptLog') + ']');
    } catch (e) {
      const err = e as Error;
      this.nStatus = '异常: ' + err.message;
      this.appendNative('NATIVE LOAD EXCEPTION ' + err.message);
    }
  }

  /**
   * 枚举这台机器上所有物理镜头。
   *
   * 这是 Web 路径做不到的事：ArkWeb 的 getUserMedia 只能拿到系统给的那一路默认流，
   * 无法按镜头类型挑选；原生 Camera Kit 才能把广角 / 超广角 / 长焦逐个列出来并按
   * cameraId 单独打开。
   */
  private listLenses(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    try {
      const mgr = camera.getCameraManager(c);
      const cams: camera.CameraDevice[] = mgr.getSupportedCameras();
      const out: string[] = [];
      for (let i = 0; i < cams.length; i++) {
        const cam: camera.CameraDevice = cams[i];
        const line: string = '#' + cam.cameraId + ' ' + posName(cam.cameraPosition) +
          ' · ' + lensName(cam.cameraType);
        out.push(line);
        this.appendNative('NATIVE LENS id=' + cam.cameraId +
          ' pos=' + cam.cameraPosition + ' type=' + cam.cameraType + ' -> ' + line);
      }
      this.nLenses = out;
      this.appendNative('NATIVE LENS count=' + cams.length);
    } catch (e) {
      const err = e as Error;
      this.appendNative('NATIVE LENS EXCEPTION ' + err.message);
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
   * 跑一次原生完整流水线。
   *
   * 全链路都在 C++：letterbox → 检测 → NMS → 单应矫正（双三次）→ 识别 → CTC → 分类。
   * JS 侧只递一张 RGBA 图和三个会话 id，拿回几十字节的结果串。
   */
  private async runNative(file: string): Promise<void> {
    if (this.busy) {
      return;
    }
    if (this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      await this.loadNative();
      if (this.detId < 0) {
        return;
      }
    }
    this.busy = true;
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(file);
      if (frame === null) {
        this.busy = false;
        return;
      }
      const raw: string = await lpr.pipelineAsync(frame.data, frame.width, frame.height,
        this.detId, this.recId, this.clsId, 0);
      const kv: Map<string, string> = parseKv(raw);

      if (kv.get('ok') !== '1') {
        this.nStatus = '推理失败';
        this.nPlate = '—';
        this.appendNative('NATIVE PIPE FAIL ' + str(kv, 'error'));
        this.busy = false;
        return;
      }

      const count: number = num(kv, 'count');
      this.nStatus = count > 0 ? '识别成功' : '未检测到车牌';
      if (count === 0) {
        this.nPlate = '—';
        this.nDetail = '';
        this.nTiming = 'total ' + str(kv, 'totalMs') + ' ms';
        this.appendNative('NATIVE PIPE ok=0-plates total=' + str(kv, 'totalMs'));
        this.busy = false;
        return;
      }

      const p0: string = str(kv, 'p0');
      const f: string[] = p0.split(',');
      // 0=code 1=det 2=rec 3=layer 4=rect 5=crop 6=colour 7=chars 8=probs 9=times 10=sum
      // f[6] 现在是 `colour|confidence`（像素测量），不再是三个分类器 logits
      const clsArr: string[] = f[6].split('|');
      const colour: string = colourLabel(clsArr[0]);
      this.nPlate = f[0] + '  ' + colour;
      this.nDetail = '检测 ' + f[1] + ' · 识别 ' + f[2] + ' · 层数 ' +
        (f[3] === '1' ? '双层' : '单层') + ' · 裁剪 ' + f[5].replace('|', 'x') +
        ' · 字符 ' + f[7];
      this.nTiming = 'total ' + str(kv, 'totalMs') + ' ms   [' + f[9].replace(/\\|/g, ' / ') + ']';
      // 落点面板：直接取 native 的 backends= 字段渲染，ArkTS 不重算（ADR-0003）
      this.nEvidence = evidenceRows(str(kv, 'backends'));
      this.appendNative('NATIVE PIPE code=' + f[0] + ' colour=' + colour +
        ' count=' + count + ' total=' + str(kv, 'totalMs') + ' stages=' + f[9] +
        ' cropSum=' + f[10] +
        ' backends=' + str(kv, 'backends'));
    } catch (e) {
      const err = e as Error;
      this.nStatus = '异常: ' + err.message;
      this.appendNative('NATIVE PIPE EXCEPTION ' + err.message);
    }
    this.busy = false;
  }

  // ------------------------------------------------------------ 端到端对照

  /**
   * 四后端检测对照：同一张解码图 × 4 条检测路径（MS CPU / MS NNRT / ncnn CPU /
   * ncnn Vulkan）× 3 次重复，识别与分类固定为生产配置 → 差异全部来自检测输出。
   * match 由这里逐字符比对（原生侧只回结果串）。
   */
  private async backendProbe(): Promise<void> {
    const ctx = this.ctx;
    if (ctx === undefined || this.detId < 0 || this.recId < 0 || this.clsId < 0) {
      this.appendNative('BACKEND E2E FAIL missing sessions');
      return;
    }
    this.busy = true;
    this.appendNative('BACKEND E2E BEGIN v1 reference=' + REFERENCE_CODE_RPV3 + ' ' + this.nBackends);
    try {
      const frame: RgbaFrame | null = await this.decodeRgba(SELFTEST_SAMPLE);
      if (frame === null) {
        this.appendNative('BACKEND E2E FAIL no frame');
        return;
      }
      const detBytes: Uint8Array = ctx.resourceManager.getRawFileContentSync(MODEL_DET);
      const npuRaw: string =
        await lpr.loadModelAsync(MODEL_DET, this.toArrayBuffer(detBytes), 'nnrt');
      this.appendNative('BACKEND E2E NPU LOAD ' + npuRaw);
      const ni: Map<string, string> = parseKv(npuRaw);
      const ids: number[] = [this.detId, ni.get('ok') === '1' ? num(ni, 'id') : -1,
        this.detId, this.detId];
      const modes: number[] = [0, 0, 1, 2];
      const labels: string[] = ['ms-cpu', 'ms-nnrt', 'ncnn-cpu', 'ncnn-vulkan'];
      const param: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.param');
      const bin: Uint8Array =
        ctx.resourceManager.getRawFileContentSync('models/y5fu_320x_head.ncnn.bin');
      for (let k = 0; k < labels.length; k++) {
        if (ids[k] < 0) {
          this.appendNative('BACKEND E2E SKIP ' + labels[k] + ' load failed');
          continue;
        }
        if (modes[k] !== 0) {
          const lr: string = await lpr.ncnnLoadAsync(
            this.toArrayBuffer(param), this.toArrayBuffer(bin), modes[k] === 2);
          this.appendNative('BACKEND E2E LOAD ' + labels[k] + ' ' + lr);
          if (parseKv(lr).get('ok') !== '1') {
            continue;
          }
        }
        for (let rep = 0; rep < 3; rep++) {
          const raw: string = await lpr.pipelineAsync(frame.data, frame.width, frame.height,
            ids[k], this.recId, this.clsId, modes[k]);
          const kv: Map<string, string> = parseKv(raw);
          const code: string = str(kv, 'p0').split(',')[0];
          this.appendNative('BACKEND E2E RESULT mode=' + labels[k] + ' rep=' + rep +
            ' match=' + (kv.get('ok') === '1' && code === REFERENCE_CODE_RPV3 ? '1' : '0') + ' ' + raw);
        }
      }
      this.appendNative('BACKEND E2E END v1');
    } catch (e) {
      const err = e as Error;
      this.appendNative('BACKEND E2E EXCEPTION ' + err.message);
    } finally {
      this.busy = false;
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
        // 节流：native 侧的 InferenceRunner 是**单任务槽**（`task_` + `has_` 会被后来者
        // 覆盖），而 ExecuteOnPool 在 libuv 池线程上阻塞等待 `doneCv_`。多个池线程同时
        // Submit 时，被覆盖的任务可能永远等不到唤醒 —— 表现就是 Promise 拿到空串或挂起。
        // 让出事件循环，保证同一时刻只有一个任务在飞。
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

  private requestCamera(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    const perms: Permissions[] = ['ohos.permission.CAMERA'];
    try {
      const atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
      atManager.requestPermissionsFromUser(c, perms).then((res) => {
        this.appendNative('CAMERA PERM auth=' + res.authResults.join(','));
      }).catch((e: Error) => {
        this.appendNative('CAMERA PERM FAIL ' + e.message);
      });
    } catch (e) {
      const err = e as Error;
      this.appendNative('CAMERA PERM EXCEPTION ' + err.message);
    }
  }

  aboutToAppear(): void {
    this.ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.requestCamera();
    // 自检：等 UI 起来（权限弹窗不阻塞）再跑，全部结果进 hilog。
    if (AUTO_SELFTEST) {
      this.sleep(1500).then(() => {
        this.vulkanProbeRun();
        this.listLenses();
        this.loadNative().then(() => {
          this.runNative(SELFTEST_SAMPLE).then(() => {
          this.backendProbe().then(() => {
            this.e2eMatrix().then(() => {
              this.ncnnSlotProbe().then(() => {
                this.cannProbe().then(() => {
                  this.probeAll().then(() => {
                    this.threadSweep();
                  });
                });
              });
            });
          });
          });
        });
      });
    }
  }

  // ------------------------------------------------------------ 界面

  /** iOS 分段控件的一段：选中为白底卡片 + 阴影，未选中透明。 */
  @Builder
  tabButton(label: string, idx: number) {
    Text(label)
      .fontSize(13)
      .fontWeight(this.tab === idx ? FontWeight.Medium : FontWeight.Normal)
      .fontColor(this.tab === idx ? IOS_LABEL : IOS_LABEL2)
      .textAlign(TextAlign.Center)
      .layoutWeight(1)
      .padding({ top: 7, bottom: 7 })
      .backgroundColor(this.tab === idx ? IOS_CARD : Color.Transparent)
      .borderRadius(7)
      .shadow(this.tab === idx
        ? { radius: 3, color: 'rgba(0,0,0,0.12)', offsetX: 0, offsetY: 1 }
        : { radius: 0, color: Color.Transparent, offsetX: 0, offsetY: 0 })
      .onClick(() => {
        this.tab = idx;
      })
  }

  @Builder
  nativeHeader() {
    Column({ space: 14 }) {
      // ── iOS 大标题（largeTitle）
      Column({ space: 2 }) {
        Text('车牌识别')
          .fontSize(28)
          .fontWeight(FontWeight.Bold)
          .fontColor(IOS_LABEL)
          .width('100%')
        Text('原生端到端 · 麒麟 8020 · 无 ArkWeb')
          .fontSize(12)
          .fontColor(IOS_LABEL2)
          .width('100%')
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)

      // ── 分组卡片 1：主操作
      Column({ space: 10 }) {
        Button('相机实时识别')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor('#ffffff')
          .backgroundColor(IOS_BLUE)
          .type(ButtonType.Normal)
          .borderRadius(10)
          .width('100%')
          .height(44)
          .onClick(() => {
            router.pushUrl({ url: 'pages/CameraPage' });
          })
        Row({ space: 10 }) {
          Button('加载模型')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.loadNative();
            })
          Button('枚举镜头')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.listLenses();
            })
        }
        .width('100%')
      }
      .width('100%')
      .padding(14)
      .backgroundColor(IOS_CARD)
      .borderRadius(12)

      // ── 分组卡片 2：后端与状态
      Column({ space: 6 }) {
        Row() {
          Text('后端')
            .fontSize(12)
            .fontColor(IOS_LABEL2)
          Blank()
          // 不能用 layoutWeight(1)：Blank() 已经把值推到右端，再加权重会让
          // 这个 Text 撑满剩余宽度（实测 bounds [154,916]→[1139,956]），
          // 一个「—」占位符被拉成整行宽 —— 看起来像渲染故障。
          Text(this.nBackends.length > 0 ? this.nBackends : '—')
            .fontSize(12)
            .fontColor(IOS_LABEL)
            .maxLines(2)
            .textAlign(TextAlign.End)
            .constraintSize({ maxWidth: '70%' })
        }
        .width('100%')
        Text(this.nStatus)
          .fontSize(12)
          .fontColor(IOS_LABEL2)
          .width('100%')
      }
      .width('100%')
      .padding(14)
      .backgroundColor(IOS_CARD)
      .borderRadius(12)
      .alignItems(HorizontalAlign.Start)

      // ── 分组卡片 3：样本
      Column({ space: 8 }) {
        Text('样本图')
          .fontSize(12)
          .fontColor(IOS_LABEL2)
          .width('100%')
        Row({ space: 8 }) {
          ForEach(NATIVE_SAMPLES, (s: string, idx: number) => {
            // 按钮标签用**序号**而不是文件名：`crop-0-津B6H920` 这类长名在三个等宽
            // 按钮里必然被截成 `crop-0-津…`（实测 labelStyle 自适应字号也不生效）。
            // 文件名不丢 —— 跑完显示在读数卡的「样本」行里。
            Button('样本 ' + (idx + 1).toString())
              .fontSize(12)
              .fontColor(IOS_BLUE)
              .backgroundColor(IOS_BLUE_SOFT)
              .type(ButtonType.Normal)
              .borderRadius(8)
              .layoutWeight(1)
              .height(34)
              .onClick(() => {
                this.runNative(s);
              })
          }, (s: string) => s)
        }
        .width('100%')
        // 文件名单独一行列出，三个按钮就不必挤长标签。
        Text(this.sampleNames())
          .fontSize(10)
          .fontColor(IOS_LABEL3)
          .width('100%')
          .maxLines(2)
      }
      .width('100%')
      .padding(14)
      .backgroundColor(IOS_CARD)
      .borderRadius(12)
      .alignItems(HorizontalAlign.Start)

      // ── 识别结果卡（iOS 读数卡：大号车牌 + 明细）
      if (this.nPlate.length > 0) {
        Column({ space: 4 }) {
          Text(this.nPlate)
            .fontSize(26)
            .fontWeight(FontWeight.Bold)
            .fontColor(IOS_LABEL)
            .width('100%')
          Text(this.nDetail)
            .fontSize(11)
            .fontColor(IOS_LABEL2)
            .width('100%')
          Text(this.nTiming)
            .fontSize(11)
            .fontColor(IOS_LABEL3)
            .width('100%')
          // ── 落点自证（ADR-0003）
          // NPU 利用率在这平台上读不到，所以"谁真的算了"只能这样显示：
          // 模块名前缀的 ⚡ = 该角色落在 NPU 上，否则是 CPU。
          // req 与 landed 不同时明写"回落"，因为静默回落正是要抓的东西。
          if (this.nEvidence.length > 0) {
            Divider()
              .color(IOS_SEP)
              .margin({ top: 6, bottom: 6 })
            Text('落点自证 · LANDED')
              .fontSize(10)
              .fontColor(IOS_LABEL3)
              .width('100%')
            ForEach(this.nEvidence, (row: string) => {
              Text(row)
                .fontSize(10)
                .fontFamily('monospace')
                .fontColor(IOS_LABEL2)
                .width('100%')
            }, (row: string) => row)
          }
        }
        .width('100%')
        .padding(14)
        .backgroundColor(IOS_CARD)
        .borderRadius(12)
        .alignItems(HorizontalAlign.Start)
      }

      // ── 镜头列表卡
      if (this.nLenses.length > 0) {
        Column({ space: 4 }) {
          Text('物理镜头 ' + this.nLenses.length.toString())
            .fontSize(12)
            .fontColor(IOS_LABEL2)
            .width('100%')
          ForEach(this.nLenses, (l: string) => {
            Text(l)
              .fontSize(11)
              .fontColor(IOS_LABEL)
              .width('100%')
          }, (l: string) => l)
        }
        .width('100%')
        .padding(14)
        .backgroundColor(IOS_CARD)
        .borderRadius(12)
        .alignItems(HorizontalAlign.Start)
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  probeHeader() {
    Column({ space: 14 }) {
      Column({ space: 2 }) {
        Text('探针控制台')
          .fontSize(28)
          .fontWeight(FontWeight.Bold)
          .fontColor(IOS_LABEL)
          .width('100%')
        Text('MindSpore Lite NDK · NNRT / GPU / KIRIN_NPU / CPU 逐项实测 + ncnn-Vulkan')
          .fontSize(12)
          .fontColor(IOS_LABEL2)
          .width('100%')
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)

      Column({ space: 10 }) {
        Row({ space: 10 }) {
          Button(this.busy ? ('探测中 ' + this.progress) : '加速器矩阵')
            .fontSize(13)
            .fontColor('#ffffff')
            .backgroundColor(IOS_BLUE)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.probeAll();
            })
          Button('GPU 专项')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.gpuProbe();
            })
          Button('清空')
            .fontSize(13)
            .fontColor(IOS_RED)
            .backgroundColor(IOS_RED_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.lines = [];
            })
        }
        .width('100%')

        Row({ space: 10 }) {
          Button('NPU 漂移')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.detDriftProbe();
            })
          Button('ncnn/Vulkan')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.ncnnProbe();
            })
          Button('三后端槽位')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.ncnnSlotProbe();
            })
        }
        .width('100%')

        Row({ space: 10 }) {
          Button('CANN 探针')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.cannProbe();
            })
          Button('算子对撞')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.opCollideProbe();
            })
          Button('RQ4 热特性')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              // 20 分钟：够跨过 thermal 2→3 的台阶，又不至于把电量耗尽
              this.rq4SustainedProbe(20);
            })
          Button('正式基准 30×')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.probeAll();
            })
          Button('线程扫描')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.threadSweep();
            })
          Button('Vulkan 探针')
            .fontSize(13)
            .fontColor(IOS_BLUE)
            .backgroundColor(IOS_BLUE_SOFT)
            .type(ButtonType.Normal)
            .borderRadius(10)
            .layoutWeight(1)
            .height(38)
            .onClick(() => {
              this.vulkanProbeRun();
            })
        }
        .width('100%')
      }
      .width('100%')
      .padding(14)
      .backgroundColor(IOS_CARD)
      .borderRadius(12)
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
  }

  @Builder
  probeLog() {
    Scroll() {
      Column({ space: 3 }) {
        // 空态：iOS 风格给个占位说明，别让一大块黑面板空着。
        if (this.lines.length === 0) {
          Text('运行上方任一探针后，日志会出现在这里')
            .fontSize(11)
            .fontFamily('monospace')
            .fontColor('rgba(255,255,255,0.4)')
            .width('100%')
        }
        ForEach(this.lines, (item: string) => {
          Text(item)
            .fontSize(10)
            .fontFamily('monospace')
            .fontColor('#d7dce3')
            .width('100%')
        }, (item: string, idx: number) => idx.toString() + '_' + item)

        // ── T7 · RQ4 热特性卡
        // 热档与电池温度**只有 ArkTS 侧读得到**（native sysroot 无热态头文件）。
        // 放在探针页而不是原生演示页：它属于测量，不属于演示。
        // 注意：Scroll 只能有一个子组件，所以这张卡必须在这个 Column 内。
        if (this.rq4Status.length > 0) {
          Column({ space: 6 }) {
            Text('RQ4 · 持续负载热特性')
              .fontSize(13)
              .fontWeight(FontWeight.Medium)
              .fontColor(IOS_LABEL)
              .width('100%')
            Text(this.rq4Status)
              .fontSize(11)
              .fontColor(IOS_LABEL2)
              .width('100%')
            if (this.rq4Rows.length > 0) {
              Text('热档 ' + this.thermalLevel.toString() + ' · 电池 ' +
                this.batteryTempC.toFixed(1) + '℃')
                .fontSize(11)
                .fontColor(IOS_LABEL3)
                .width('100%')
              ForEach(this.rq4Rows, (row: string) => {
                Text(row)
                  .fontSize(9)
                  .fontFamily('monospace')
                  .fontColor(IOS_LABEL2)
                  .width('100%')
              }, (row: string, idx: number) => 'rq4_' + idx.toString())
            }
          }
          .width('100%')
          .padding(12)
          .backgroundColor(IOS_CARD)
          .borderRadius(12)
          .alignItems(HorizontalAlign.Start)
          .margin({ top: 8 })
        }
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)
    }
    .layoutWeight(1)
    .width('100%')
    .padding(12)
    .backgroundColor('#1c1c1e')
    .borderRadius(12)
    // 必须显式顶对齐：Scroll 默认把内容居中，实测那行日志飘在面板中间。
    .align(Alignment.TopStart)
  }

  build() {
    Column({ space: 0 }) {
      // ── iOS 分段控件（segmented control）：灰轨道 + 白滑块
      // ⚠️ 这里不能写 .width('100%') —— 它取父容器 100% 后再叠加 margin，
      // 会溢出屏幕（实测 [5,185]→[1224,279]，宽 1219 > 1224 视口），
      // 而内容卡是 x=45，左边缘对不齐。靠 margin 撑开即可。
      Row({ space: 0 }) {
        this.tabButton('原生演示', 0)
        // iOS 两段式分段控件在未选中段之间有一条发丝分隔线。
        // 注意：Divider 在 Row 里若不显式给 width 会渲染不出来（实测不可见）。
        Divider()
          .vertical(true)
          .width(1)
          .height(16)
          .color('rgba(60,60,67,0.29)')
        this.tabButton('探针控制台', 1)
      }
      .padding(2)
      .backgroundColor(IOS_TRACK)
      .borderRadius(9)
      .margin({ left: 16, right: 16, top: 12, bottom: 12 })

      if (this.tab === 0) {
        Scroll() {
          Column() {
            this.nativeHeader()
          }
          .width('100%')
          .padding({ left: 16, right: 16, bottom: 16 })
        }
        .layoutWeight(1)
        .width('100%')
        .align(Alignment.Top)
      }

      if (this.tab === 1) {
        Column({ space: 10 }) {
          this.probeHeader()
          this.probeLog()
        }
        .layoutWeight(1)
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 16 })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor(IOS_BG)
  }
}