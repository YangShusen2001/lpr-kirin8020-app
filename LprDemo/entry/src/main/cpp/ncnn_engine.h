#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct LetterBoxed;

/**
 * ncnn 引擎（A 阶段第二步：GPU 路径的载体）。
 *
 * 为什么要有它：ADR-008 证明 Vulkan compute 在麒麟 8020 上可达（Maleoon 920C，
 * 2 条 G+C 队列）。onnxruntime / MindSpore Lite 都够不到这条路径，
 * 而 ncnn 是唯一成熟的、能在 OHOS 上跑 Vulkan 后端的推理框架。
 *
 * 本文件先做 CPU 后端（交叉编译已验证），Vulkan 后端在下一步
 * （-DNCNN_VULKAN=ON 重编 + use_vulkan_compute=true）打开。
 *
 * 保真纪律：输入预处理**复用 lpr_pipeline 的 LprLetterBox / LprToNchw**，
 * 不另写一份 —— 这样设备侧 ncnn 的输入与 PC 侧 ONNX 参考逐字节同源，
 * 两侧输出可直接对比。
 */

/** 从内存加载 ncnn 模型（param 为文本，bin 为权重；均由 ArkTS 从 rawfile 读入）。
 *  useVulkan=true 时走 Vulkan 后端（需 NCNN_VULKAN=ON 构建的 libncnn.so），
 *  失败（无 libvulkan/无设备）时返回 false 并在 err 里带原因。 */
bool NcnnLoad(const std::vector<char>& param, const std::vector<char>& bin,
              bool useVulkan, std::string& err);

/** 释放当前网络（可重新加载）。 */
void NcnnRelease();

/** 是否已有可用网络。 */
bool NcnnLoaded();

/** GPU 实例状态：ok=1;count=..;name=..;driver=.. 或 ok=0;error=..
 *  create_gpu_instance 成功后可调用，用于自证落点（Maleoon 920C 而非回落 CPU）。 */
std::string NcnnGpuProbe();

/**
 * 用真实 RGBA 图像跑裸 head（可多轮计时）。
 * 返回扁平 kv：ok=1;o0l2=..;o0max=..;o0n=..;o1l2=..;o1max=..;o2l2=..;o2max=..;
 *              warmup=..;repeat=..;p50Ms=..;meanMs=..;error=
 * 三个输出（out0/out1/out2，各 [1,45,H,W]）的 L2/maxAbs 用于与 PC 参考对照。
 */
std::string NcnnRunRgba(const uint8_t* rgba, int w, int h, int warmup, int repeat);

/** 网络结构摘要（层数 / 输入输出名），加载后调用。 */
std::string NcnnInfo();

/**
 * 切换 ncnn 检测后端模式（CPU / Vulkan），必要时用缓存的 param/bin 重载网络。
 * 同一份字节、两种后端 —— 模式不变时是幂等的（不重复加载）。
 */
bool NcnnReload(bool useVulkan, std::string& err);

/**
 * 检测旁路（完整流水线用）：对已 letterbox 的 320×320 图跑 ncnn 裸 head，
 * 返回三个 [45,H,H] NCHW 浮点输出（NcnnRunRgba 同源预处理，直接喂
 * LprDecodeBareHead）。只允许 elempack==1 的 fp32 输出，布局异常即报错。
 */
bool NcnnDetect(const LetterBoxed& lb, std::vector<std::vector<float>>& outs,
                std::string& err);

// ---------------------------------------------------------------- 通用模型槽位
//
// 用途：识别（rpv3）与分类（litemodel）也要吃到 GPU，而它们既不是检测旁路的多输出形状，
// 也不是「letterbox 320×320」那套预处理 —— 塞进同一条路只会长出一堆特例判断。
//
// 为什么另起一套而不是重构上面那条检测路径：检测旁路（NcnnLoad/NcnnDetect）已由
// ADR-011 / A4 在真机上验证过（218 层裸 head、三输出、与 PC ONNX 相对差 ≤0.05%），
// 它的结论散落在日志与证据文件里。为了给识别/分类腾位置去重构它，等于把已验证的路径
// 放到回归风险下 —— 所以这里新增一套通用槽位注册表，检测那条路一个字节都不动。
//
// 槽位号由调用方（ArkTS）分配：1 = 识别，2 = 分类（0 保留给检测旁路）。

/**
 * 把一份 param/bin 加载到指定槽位。同一槽位重复加载会先释放再建。
 * `useVulkan=true` 走 Vulkan 后端（需 NCNN_VULKAN=ON 的 libncnn.so + 可用设备）。
 * 每槽位独立持有 param/bin 副本 —— ncnn 的内存版 load_model 只引用不拷贝。
 */
bool NcnnLoadSlot(int slot, const std::vector<char>& param, const std::vector<char>& bin,
                  bool useVulkan, std::string& err);

/** 释放指定槽位。 */
void NcnnReleaseSlot(int slot);

/** 槽位摘要：ok=1;slot=..;layers=..;inputs=..;outputs=..;requestedVulkan=..;effectiveVulkan=..;gpuProbe=.. */
std::string NcnnInfoSlot(int slot);

/**
 * 通用推理：输入 NCHW（c,h,w）float（调用方负责布局，别让引擎猜），
 * 输出按 ncnn Mat 的内存序拍平到 `out`，形状写进 `shapeKv`
 * （`dims=..;w=..;h=..;c=..;n=..`）—— 形状不写死：识别模型 [1,20,78] 与分类模型 [1,3]
 * 维度完全不同，调用方要知道实际形状才能做 argmax / CTC。
 *
 * 只接受 elempack==1 的 fp32 输出（Vulkan 后端默认 fp16 存储，故加载时显式关闭
 * fp16 packed/storage/arithmetic，见 NcnnLoadSlot）。
 */
bool NcnnRunSlot(int slot, const float* nchw, int c, int h, int w,
                 std::vector<float>& out, std::string& shapeKv, std::string& err);

/** 时间基准（warmup + repeat 次），返回 p50/mean 与输出校验和，供 CPU-diff 协议使用。 */
std::string NcnnBenchSlot(int slot, const float* nchw, int c, int h, int w,
                          int warmup, int repeat);
