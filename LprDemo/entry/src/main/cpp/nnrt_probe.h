#pragma once
#include <cstdint>
#include <string>
#include <vector>

/**
 * NNRt / HiAI Foundation（CANN Kit）探针。
 *
 * 为什么要有它：项目的 NPU 路径一直只有「MindSpore Lite + NNRT delegate」一条，落到哪张卡
 * 只能靠解析日志推断。而设备上其实还躺着两条**标准 NDK** 通路：
 *
 *   /system/lib64/ndk/libneural_network_runtime.so   (NNRt，OpenHarmony 标准 NN 运行时)
 *   /system/lib64/ndk/libhiai_foundation.so          (HiAI Foundation = CANN Kit)
 *
 * 两者的 syscap 在设备上都是 true（`SystemCapability.AI.NeuralNetworkRuntime` /
 * `SystemCapability.AI.HiAIFoundation`），即不需要合作伙伴白名单。
 *
 * 本探针只做两件事：**枚举设备**与**试探模型格式**，不改变任何生产路径。
 */

/** 枚举：候选库能否 dlopen、HiAI 版本号、NNRt 上到底有几张设备（名字 + 类型）。 */
std::string NnrtProbeReport();

/**
 * 把一段模型字节喂给 NNRt 的 offline model 入口，看设备收不收。
 *
 * offline model 在官方文档里的定义是「由**设备厂商提供的模型转换器**离线编译」的产物，
 * 因此它大概率只吃 `.om`。但 `.ms`（MindSpore Lite 的离线格式）是否也被同一入口接受，
 * 以及 HiAI 的兼容性检查怎么判，只有真机跑一次才算数 —— 这就是本函数的用途。
 *
 * @param model        模型文件字节（ArkTS 从 rawfile 读入）
 * @param deviceIndex  用枚举结果里的第几张设备（越界则用第一张）
 */
std::string NnrtTryModelReport(const std::vector<uint8_t>& model, int deviceIndex);
