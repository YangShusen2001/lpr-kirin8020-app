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
 * 车辆检测（T2）。**只跑车辆检测器，不进车牌流水线。**
 *
 * `detId` 是会话 id（用 `loadModelAsync('models/yolov5su_320_veh_fp32.ms', bytes, 'cpu')`
 * 拿到）；`rgba` 是 **RGBA8888** 缓冲，大小必须等于 w*h*4。
 *
 * 返回扁平 kv 串（ArkTS 禁用 `any`，所以不用 JSON）：
 *
 *   ok=1;count=3;truncated=0;conf=0.0500;iou=0.5000;vehicleOnly=1;size=320;nhwc=1;
 *   inferMs=..;totalMs=..;backend=CPU;requested=cpu;fallbackFrom=;error=;
 *   b0=2,0.8123,12.0000|34.0000|56.0000|78.0000,car;
 *   ...
 *
 * 每个 `b<i>` = `类号,分数,x1|x2|x2|x2,类名`，坐标是**源图坐标**（已做 letterbox 反变换）。
 *
 * `truncated=1` 表示检出数被上限（100）截断过 —— 调用方必须如实显示，不能当成"全部"。
 *
 * 阈值缺省：confThresh=0.05（比车牌检测的 0.25 低得多，为 T4 不漏车框）、
 * iouThresh=0.5、vehicleOnly=true（只留 car/motorcycle/bus/truck）。
 * 显式传入的阈值必须是 (0,1) 内的有限数，否则**报错**而不是静默退回默认值。
 */
export const vehicleDetectAsync: (
  detId: number, rgba: ArrayBuffer, width: number, height: number,
  confThresh?: number, iouThresh?: number, vehicleOnly?: boolean) => Promise<string>;

/**
 * T3：ROI 裁剪 + 坐标映射的单元自证。
 *
 * 只要一张 RGBA 图，**不需要**模型会话 —— 它测的是纯几何与逐字节裁剪：
 * 外扩取整方向、图边界 clamp、退化输入不崩、ROI→源图坐标映射、
 * 以及"裁出来的像素与源图对应区域逐字节相同"。
 *
 * 返回**多行报告**（不是 kv 串），每行 `case=<名字>;ok=0/1;<细节>`，
 * 末行 `total=N;failed=M`。传入无效图时原生侧会合成确定性图案并写明
 * `note=src-synth`，不会假装用的是真实素材。
 */
export const roiSelfTestAsync: (
  rgba: ArrayBuffer, width: number, height: number) => Promise<string>;

/**
 * T3：车框 → 裁 ROI → 车牌检测 → **映射回原图坐标**（只做分数最高的一个车框）。
 *
 * 刻意不做"遍历所有车框 + 合并去重"——那是 T4。本入口要证的是
 * 「ROI 裁得对、映射不偏」，判据是 ROI 路径映射回来的车牌框与整图直检框的 IoU，
 * 逐框写在日志与返回串里（`r0=...` 是 ROI 路径、`d0=...` 是直检对照）。
 *
 * **四个会话 id 不是同一批模型**：`vehId` 是车辆检测器（yolov5su，单输出），
 * `detId`/`recId`/`clsId` 是车牌流水线（y5fu_320x 检测 + 识别 + 分类）。
 * 把车牌检测器当车辆模型传会得到 `yolov5u 期望单输出，实际 3`。
 *
 * 车辆检测固定用 conf=0.05 / iou=0.5 / 只留车辆类（与 T2 同口径）。
 *
 * `boxIdx` 选探第几个车辆框（按分数降序，缺省 0）。**必须探一个 `x0 > 0` 的框**：
 * 最高分框常贴着左边缘，ROI 会被 clamp 成 `x0 = 0`，这时"忘了加 x0"与"映射正确"
 * 结果完全一样 —— x 方向的映射等于没验证。
 *
 * `expand` 缺省 0.15（`kRoiExpandDefault`），显式传入必须落在 [0,1) 否则报错。
 */
export const roiPlateProbeAsync: (
  vehId: number, detId: number, recId: number, clsId: number,
  rgba: ArrayBuffer, width: number, height: number,
  boxIdx?: number, expand?: number) => Promise<string>;

/**
 * T4：ROI 路径端到端 —— 车辆检测 → **逐框**裁 ROI → 逐框车牌检测 → 映射回原图
 * → 合并去重。返回车牌框 + 车牌串 + 颜色 + **归属的车辆下标**。
 *
 * 默认参数即 spec 定案值：`vehConf=0.05`（D1 极低阈值）、`roiExpand=0.15`、
 * `dedupeIou=0.5`（D3）。三者都可显式传，但必须在合法区间内，否则**报错**而非静默退回。
 *
 * 返回 kv 串，字段：
 * - 汇总：`vehCount` / `vehTruncated` / `roiTried` / `roiSkipped` / `rawHits` /
 *   `dedupeDropped` / `count` / `vehInferMs` / `roiDetectMs` / `totalMs`
 * - 车辆框：`v<i>=<cls>,<score>,<x1>|<y1>|<x2>|<y2>,<name>`
 * - 车牌：`p<i>=<x1>|<y1>|<x2>|<y2>,<detScore>,<ownerVeh>,<colour>,<code>,<recConf>`
 *
 * `rawHits - count == dedupeDropped`；`dedupeDropped > 0` 就是去重真的生效了的证据。
 *
 * ⚠️ `vehId`（yolov5su 车辆检测器）与 `detId`（y5fu_320x 车牌检测器）不是同一个模型。
 */
export const roiPipelineAsync: (
  vehId: number, detId: number, recId: number, clsId: number,
  rgba: ArrayBuffer, width: number, height: number,
  vehConf?: number, roiExpand?: number, dedupeIou?: number) => Promise<string>;

/**
 * T4：合并去重的单元自证（纯数据，无入参、不需要模型会话）。
 *
 * 构造完全重合 / IoU 0.667 / IoU 0.333 / 链式重叠 / 边界相接 / 空输入等已知输入，
 * 断言保留条数与**保留的是哪一条**（高分者）。返回多行报告，末行 `total=N;failed=M`。
 */
export const roiDedupeSelfTestAsync: () => Promise<string>;

/**
 * T4：「构造重叠车框」的集成验证（需要**真的跑模型**，与上面的纯数据自证互补）。
 *
 * 取图上分数最高的真实车框 A，人为构造一个向四周外扩 6 px 的车框 B（与 A 必然重叠）；
 * 两个重叠 ROI 各自跑车牌检测 ⇒ 同一块牌被检出两次 ⇒ 去重后必须只剩 1 条。
 * 报告里给出 `raw` / `kept` / `iouBefore` / `keptCode`。
 *
 * `raw != 2` 时判失败并在细节里写明「去重未被触发（用例无效）」—— 不把用例无效
 * 混成"去重实现错了"。
 */
export const roiOverlapSelfTestAsync: (
  vehId: number, detId: number, recId: number, clsId: number,
  rgba: ArrayBuffer, width: number, height: number) => Promise<string>;

/**
 * ncnn 通用模型槽位（识别=1 / 分类=2，0 保留给检测旁路）。
 *
 * GPU（Vulkan）在麒麟 8020 上只有 ncnn 一条通路，而 MS Lite 的 GPU 档编译期判否 ——
 * 要让识别/分类也吃 GPU，就得把它们的 param/bin 加载成槽位，再在 pipelineAsync 的
 * 第 8/9 参里指定 `recSlot` / `clsSlot`。
 */
export const ncnnLoadSlotAsync: (
  slot: number, param: ArrayBuffer, bin: ArrayBuffer, useVulkan?: boolean) => Promise<string>;

/**
 * 隔离基准：`warmup` 次预热后，对同一 session 计时 `repeat` 次推理。
 *
 * `gapMs` / `polluteKB`（2026-09-21 新增，可选）在**计时区之外**制造干扰，
 * 用于定位「隔离 7.4 ms vs 相机流水线内 19.5 ms」的 2 倍差距：
 *   - `gapMs`：迭代间 sleep，检验 DVFS（调用变稀疏是否掉频）。
 *   - `polluteKB`：迭代间搬运该大小的缓冲，检验缓存/内存带宽污染
 *     （流水线每帧还要写 1.2 MB RGBA + 读 1.2 MB NV21 + letterbox，
 *     而 sleep 完全模拟不到这一层）。
 * 不传即历史行为（两者为 0）。返回值里带 `gapMs=` / `polluteKB=` 回显，便于自证。
 */
export const benchAsync: (
  id: number, warmup: number, repeat: number,
  gapMs?: number, polluteKB?: number, spinMs?: number) => Promise<string>;

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
 * T11 裸识别（bare-head）：**跳过 det/rectify**，把裁剪图直喂识别器。
 *
 * 为什么单独一条入口：此前所有准确率数字都有口径问题 —— t6/A16 的 90.6% 是
 * **主机** onnxruntime 测的；T10 的 60.8% 是端侧但走完整流水线，det 要在一张
 * 已裁剪小图上重新找牌，输入被劣化且混入检测器误差。本入口与 t6/A16 工况一致，
 * 数字可直接对比。
 *
 * Args: (rgba, width, height, recId, [recSlot])   recSlot >= 0 走 ncnn 槽位。
 * Returns: ok=1;code=...;conf=...;recMs=...;backend=...  或  ok=0;error=...
 */
export const recognise: (
  rgba: ArrayBuffer, width: number, height: number,
  recId: number, recSlot?: number) => string;

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
