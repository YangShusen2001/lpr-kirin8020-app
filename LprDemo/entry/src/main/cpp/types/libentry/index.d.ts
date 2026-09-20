/**
 * 原生 LPR 引擎（libentry.so）的 NAPI 接口声明。
 * 实现在 entry/src/main/cpp/{napi_init,ms_engine,lpr_pipeline,ncnn_engine}.cpp。
 *
 * ⚠️ 线程契约（2026-09-18 重写）：所有 MindSpore Lite / ncnn 调用都在**一根专用推理
 * 线程**上串行执行；会话按 (name|backend) 去重并封顶 12 份。
 * ArkTS 侧一律用 `*Async` —— 同步版本仍会阻塞 UI 线程，连续占用 >3 s / >6 s 会被
 * 系统 watchdog 判 THREAD_BLOCK 并 SIGKILL（实测 uvLoopTask 8887 ms）。
 */
/** backend: "auto" | "nnrt" | "gpu" | "kirin" | "cpu" */
export const loadModel: (name: string, bytes: ArrayBuffer, backend: string) => string;
export const run: (id: number, input: Float32Array) => Float32Array;
/** 稳态计时 + 输出校验和（CPU-diff 协议）。返回扁平 kv 串。 */
export const bench: (id: number, warmup: number, repeat: number) => string;
export const listNnrtDevices: () => string;

/** ─ 异步版本（推荐）：推理在专用线程上跑，Promise 在 JS 线程 resolve ── */

/** 会话按 (name|backend) 去重并封顶 12；重复加载同一组合时回 `cached=1` 并复用 id。 */
export const loadModelAsync: (name: string, bytes: ArrayBuffer, backend: string) => Promise<string>;

/**
 * NNRt / HiAI（CANN Kit）探针。
 *
 * nnrtProbeAsync：枚举库是否可加载、CANN 版本号、NNRt 上的设备（名字 + 类型）。
 * nnrtTryModelAsync：把模型字节喂给 offline model 入口，并问 HiAI 认不认这段字节。
 *   返回 kv 里 build_rc=0 表示设备端接受了这段字节的格式并能完成编译。
 */
export const nnrtProbeAsync: () => Promise<string>;
export const nnrtTryModelAsync: (modelBytes: ArrayBuffer, deviceIndex?: number) => Promise<string>;

export const pipelineAsync: (
  rgba: ArrayBuffer, width: number, height: number,
  detId: number, recId: number, clsId: number, detMode?: number,
  recSlot?: number, clsSlot?: number) => Promise<string>;

/**
 * 相机帧一步到位：NV21 原始 buffer →（native 转 RGBA + 旋转）→ 完整流水线。
 *
 * 为什么单独一条入口：相机预览档是 YUV_420_SP(NV21)，而 pipeline 只吃 RGBA。
 * 若在 ArkTS 侧转（createPixelMap → rotate → readPixelsToBuffer），实测 21-38 ms，
 * 比整个推理还贵（NPU 档 10 ms）。这里一趟 C++ 循环做完，并省掉两次跨语言拷贝。
 *
 * 返回 kv 串，在 pipelineAsync 的字段外多两个：
 *   convMs  = NV21→RGBA(+旋转) 的耗时
 *   inferMs = 流水线本身的耗时
 *   w/h     = 转换后的图像尺寸（旋转 90/270 时与原帧互换）
 */
export const cameraFrameAsync: (
  nv21: ArrayBuffer, width: number, height: number, stride: number, rotation: number,
  detId: number, recId: number, clsId: number, detMode?: number,
  recSlot?: number, clsSlot?: number) => Promise<string>;

/**
 * ncnn 通用模型槽位（识别=1 / 分类=2，0 保留给检测旁路）。
 *
 * GPU（Vulkan）在麒麟 8020 上只有 ncnn 一条通路，而 MS Lite 的 GPU 档编译期判否 ——
 * 要让识别/分类也吃 GPU，就得把它们的 param/bin 加载成槽位，再在 pipelineAsync 的
 * 第 8/9 参里指定 `recSlot` / `clsSlot`。
 */
export const ncnnLoadSlotAsync: (
  slot: number, param: ArrayBuffer, bin: ArrayBuffer, useVulkan?: boolean) => Promise<string>;

export const benchAsync: (id: number, warmup: number, repeat: number) => Promise<string>;

/** 追加一行到文件（矩阵落盘续跑用，被杀也能知道跑到哪）。 */
export const appendLine: (path: string, line: string) => string;

/**
 * 原生完整流水线（检测 → NMS → 单应矫正 → 识别 → CTC → 分类），全在 C++ 里跑。
 *
 * 三个会话可以落在不同后端 —— 这正是必须在原生侧重建流水线的原因：
 * 检测模型上不了 NPU（ADR-004 §4.1），识别模型能上（2.2–3.0×），
 * 而 onnxruntime-web 根本够不到 NPU（ADR-004 §6.1）。
 *
 * 返回扁平 kv 串：
 *   ok=1;count=N;totalMs=...;error=
 *   p0=<code>,<detScore>,<recConf>,<layer>,<x1|x2|x3|x4>,<cropH|cropW>,
 *      <cls0|cls1|cls2>,<char|...>,<prob|...>,<tDet|tRect|tRec|tCls>,<cropSum>;
 */
export const pipeline: (
  rgba: ArrayBuffer, width: number, height: number,
  detId: number, recId: number, clsId: number, detMode?: number) => string;

/**
 * GPU 终审探针（vulkan_probe.cpp）：dlopen Vulkan loader → vkCreateInstance →
 * vkEnumeratePhysicalDevices → queue families。返回 JSON，verdict 字段四选一。
 * 只打日志用，ArkTS 侧不解析（JSON 里有引号/逗号，KvSanitize 语法装不下）。
 */
export const vulkanProbe: () => string;

/**
 * ncnn 引擎（A 阶段第二步）：param/bin 由 ArkTS 从 rawfile 读入后传字节。
 * ncnnRun 用真实 RGBA 图跑裸 head，返回三输出 L2/maxAbs 与耗时（kv 串）。
 */
export const ncnnLoad: (param: ArrayBuffer, bin: ArrayBuffer, useVulkan?: boolean) => string;
/** Vulkan 首次加载含 shader 编译，异步版避免压在 JS 线程上。 */
export const ncnnLoadAsync: (param: ArrayBuffer, bin: ArrayBuffer, useVulkan?: boolean) => Promise<string>;
export const ncnnRun: (rgba: ArrayBuffer, width: number, height: number, repeat: number) => string;
export const ncnnRelease: () => string;
