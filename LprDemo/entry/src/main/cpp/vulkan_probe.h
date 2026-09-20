#pragma once
#include <string>

/**
 * 应用级 Vulkan 终审探针（ADR-008 的证据入口）。
 *
 * 回答一个问题：这台麒麟 8020 上，第三方原生 App 到底能不能触达 GPU 计算？
 * 路径：dlopen Vulkan loader → vkCreateInstance → vkEnumeratePhysicalDevices →
 *       读 deviceName / vendorID / queue families（找 VK_QUEUE_COMPUTE_BIT）。
 *
 * 不 include <vulkan/vulkan.h>（SDK 是否随附 Vulkan 头不受我们控制），
 * 所需的 ABI 按官方 spec 手写最小子集，Vulkan ABI 自 1.0 起冻结，安全。
 *
 * 任何一步失败都不抛异常、不崩溃：结果是一个 JSON 字符串，
 * verdict ∈ NO_VK_LOADER | VK_INSTANCE_FAIL | NO_DEVICE | NO_COMPUTE_QUEUE
 *         | GPU_COMPUTE_FEASIBLE。
 */
std::string VulkanProbeJson();
