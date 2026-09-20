#include "vulkan_probe.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- Vulkan ABI（手写最小子集）
// 依据 Vulkan 1.3 官方 spec；句柄是不透明指针，结构体只取实际要读的字段。
// Vulkan ABI 冻结承诺（spec §Fundamentals）保证这些偏移不会变。

typedef int32_t VkResult;   // VK_SUCCESS = 0, VK_INCOMPLETE = 5
typedef void*   VkInstance;
typedef void*   VkPhysicalDevice;

#define VK_SUCCESS          0
#define VK_INCOMPLETE       5
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_QUEUE_GRAPHICS_BIT   0x00000001
#define VK_QUEUE_COMPUTE_BIT    0x00000002

struct VkInstanceCreateInfoRaw {
  uint32_t sType;
  const void* pNext;
  uint32_t flags;
  const void* pApplicationInfo;
  uint32_t enabledLayerCount;
  const char* const* ppEnabledLayerNames;
  uint32_t enabledExtensionCount;
  const char* const* ppEnabledExtensionNames;
};

/**
 * VkPhysicalDeviceProperties 的头部镜像（ 前 292 字节与官方结构逐字段一致；
 * 结构体尾部是几百字节的 limits —— 用 1024B 的零初始化缓冲接住整个输出，
 * 只按已知偏移解析，不越界不误读。
 */
struct VkPhysicalDevicePropertiesHead {
  uint32_t apiVersion;
  uint32_t driverVersion;
  uint32_t vendorID;
  uint32_t deviceID;
  uint32_t deviceType;
  char     deviceName[256];
  uint8_t  pipelineCacheUUID[16];
};
static_assert(sizeof(VkPhysicalDevicePropertiesHead) == 292, "abi drift");

/** VkQueueFamilyProperties（24B，无对齐坑）：queueFlags/queueCount/timestampValidBits/粒度。 */
struct VkQueueFamilyPropertiesRaw {
  uint32_t queueFlags;
  uint32_t queueCount;
  uint32_t timestampValidBits;
  uint32_t granularity[3];
};

/** VkExtensionProperties：extensionName[256] + specVersion（260B，对齐 4）。 */
struct VkExtensionPropertiesRaw {
  char     extensionName[256];
  uint32_t specVersion;
};

typedef VkResult (*PFN_vkEnumerateInstanceExtensionProperties)(
    const char* pLayerName, uint32_t* pPropertyCount, void* pProperties);
typedef VkResult (*PFN_vkCreateInstance)(
    const void* pCreateInfo, const void* pAllocator, void** pInstance);
typedef void (*PFN_vkDestroyInstance)(void* instance, const void* pAllocator);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(
    void* instance, uint32_t* pPhysicalDeviceCount, void** pPhysicalDevices);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(
    void* physicalDevice, void* pProperties);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(
    void* physicalDevice, uint32_t* pQueueFamilyPropertyCount, void* pQueueFamilyProperties);
typedef VkResult (*PFN_vkEnumerateDeviceExtensionProperties)(
    void* physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, void* pProperties);

// ---------------------------------------------------------------- 小工具

std::string J(const std::string& in) {          // JSON 字符串转义
  std::string out;
  for (char c : in) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    if ((unsigned char)c >= 0x20) {
      out += c;
    }
  }
  return out;
}

std::string VkVer(uint32_t v) {
  char buf[48];
  snprintf(buf, sizeof(buf), "%u.%u.%u", v >> 22, (v >> 12) & 0x3FF, v & 0xFFF);
  return buf;
}

std::string Hex(uint32_t v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0x%04x", v);
  return buf;
}

const char* DeviceType(uint32_t t) {
  switch (t) {
    case 1: return "integrated";
    case 2: return "discrete";
    case 3: return "virtual";
    case 4: return "cpu";
    default: return "other";
  }
}

}  // namespace

std::string VulkanProbeJson() {
  std::string json = "{\"probe\":\"vulkan-app-level\",\"date\":\"2026-09-17\",";
  std::string verdict;
  bool hasCompute = false;
  VkInstance inst = nullptr;
  void* lib = nullptr;

  // ---- 1) loader：dlopen 候选链。失败会把 dlerror() 原样带出去 ——
  //         OHOS 的 so namespace 白名单错误信息非常有辨识度。
  const char* candidates[] = {"libvulkan.so", "libvulkan.so.1", "libvulkan_hos.so"};
  const char* lastErr = nullptr;
  for (int i = 0; i < 3; i++) {
    lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    if (lib != nullptr) {
      json += std::string("\"loader\":{\"ok\":true,\"lib\":\"") + candidates[i] + "\"},";
      break;
    }
    lastErr = dlerror();
  }
  if (lib == nullptr) {
    json += std::string("\"loader\":{\"ok\":false,\"tried\":\"libvulkan.so,libvulkan.so.1,"
                        "libvulkan_hos.so\",\"error\":\"") +
            J(lastErr ? lastErr : "unknown") + "\"},";
    json += "\"verdict\":\"NO_VK_LOADER\"}";
    return json;
  }

  // ---- 2) 实例扩展（layerName=NULL）
  PFN_vkEnumerateInstanceExtensionProperties enumInstExt =
      (PFN_vkEnumerateInstanceExtensionProperties)dlsym(lib, "vkEnumerateInstanceExtensionProperties");
  if (enumInstExt == nullptr) {
    json += std::string("\"dlsym\":{\"error\":\"vkEnumerateInstanceExtensionProperties missing\"},"
                        "\"verdict\":\"NO_VK_LOADER\"}");
    dlclose(lib);
    return json;
  }
  uint32_t extCount = 0;
  json += "\"instanceExtensions\":[";
  if (enumInstExt(nullptr, &extCount, nullptr) == VK_SUCCESS && extCount > 0) {
    std::vector<VkExtensionPropertiesRaw> exts(extCount);
    VkResult r = enumInstExt(nullptr, &extCount, exts.data());
    for (uint32_t i = 0; i < exts.size() && i < extCount && i < 32; i++) {
      if (i > 0) {
        json += ",";
      }
      json += "\"" + J(exts[i].extensionName) + "\"";
    }
  }
  json += "],";

  // ---- 3) vkCreateInstance（无扩展无层；枚举物理设备不需要 surface）
  PFN_vkCreateInstance createInst = (PFN_vkCreateInstance)dlsym(lib, "vkCreateInstance");
  PFN_vkDestroyInstance destroyInst = (PFN_vkDestroyInstance)dlsym(lib, "vkDestroyInstance");
  if (createInst == nullptr) {
    json += "\"dlsym\":{\"error\":\"vkCreateInstance missing\"},\"verdict\":\"NO_VK_LOADER\"}";
    dlclose(lib);
    return json;
  }
  VkInstanceCreateInfoRaw ci = {};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  VkResult r = createInst(&ci, nullptr, &inst);
  if (r != VK_SUCCESS || inst == nullptr) {
    json += "\"createInstance\":{\"result\":" + std::to_string(r) +
            "},\"verdict\":\"VK_INSTANCE_FAIL\"}";
    dlclose(lib);
    return json;
  }
  json += "\"createInstance\":{\"result\":0},";

  // ---- 4) 物理设备
  PFN_vkEnumeratePhysicalDevices enumDevs =
      (PFN_vkEnumeratePhysicalDevices)dlsym(lib, "vkEnumeratePhysicalDevices");
  PFN_vkGetPhysicalDeviceProperties getProps =
      (PFN_vkGetPhysicalDeviceProperties)dlsym(lib, "vkGetPhysicalDeviceProperties");
  PFN_vkGetPhysicalDeviceQueueFamilyProperties getQueues =
      (PFN_vkGetPhysicalDeviceQueueFamilyProperties)dlsym(lib, "vkGetPhysicalDeviceQueueFamilyProperties");
  PFN_vkEnumerateDeviceExtensionProperties enumDevExt =
      (PFN_vkEnumerateDeviceExtensionProperties)dlsym(lib, "vkEnumerateDeviceExtensionProperties");
  if (enumDevs == nullptr || getProps == nullptr || getQueues == nullptr) {
    json += "\"dlsym\":{\"error\":\"device-level symbols missing\"},\"verdict\":\"NO_VK_LOADER\"}";
    if (destroyInst) {
      destroyInst(inst, nullptr);
    }
    dlclose(lib);
    return json;
  }

  uint32_t devCount = 0;
  r = enumDevs(inst, &devCount, nullptr);
  if (r != VK_SUCCESS || devCount == 0) {
    json += "\"devices\":{\"result\":" + std::to_string(r) + ",\"count\":0},\"verdict\":\"NO_DEVICE\"}";
    if (destroyInst) {
      destroyInst(inst, nullptr);
    }
    dlclose(lib);
    return json;
  }
  std::vector<VkPhysicalDevice> devs(devCount);
  r = enumDevs(inst, &devCount, devs.data());
  json += "\"devices\":{\"result\":" + std::to_string(r) + ",\"count\":" + std::to_string(devCount) + "},";
  json += "\"deviceList\":[";

  for (uint32_t d = 0; d < devs.size() && d < devCount; d++) {
    if (d > 0) {
      json += ",";
    }
    unsigned char props[1024] = {0};
    getProps(devs[d], props);
    VkPhysicalDevicePropertiesHead* p = (VkPhysicalDevicePropertiesHead*)props;

    uint32_t qfCount = 0;
    getQueues(devs[d], &qfCount, nullptr);
    std::vector<VkQueueFamilyPropertiesRaw> qf(qfCount);
    if (qfCount > 0) {
      getQueues(devs[d], &qfCount, qf.data());
    }
    uint32_t gq = 0, cq = 0, gcq = 0;
    for (uint32_t q = 0; q < qf.size() && q < qfCount; q++) {
      uint32_t n = qf[q].queueCount;
      bool g = (qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
      bool c = (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
      if (g && !c) gq += n;
      if (c && !g) cq += n;
      if (g && c) gcq += n;
      if (c) hasCompute = true;
    }

    json += "{\"name\":\"" + J(std::string(p->deviceName, strnlen(p->deviceName, 256))) + "\"" +
            ",\"vendorID\":" + Hex(p->vendorID) +
            ",\"deviceID\":" + Hex(p->deviceID) +
            ",\"deviceType\":\"" + DeviceType(p->deviceType) + "\"" +
            ",\"apiVersion\":\"" + VkVer(p->apiVersion) + "\"" +
            ",\"driverVersion\":\"" + VkVer(p->driverVersion) + "\"" +
            ",\"queueFamilies\":" + std::to_string(qfCount) +
            ",\"graphicsOnlyQueues\":" + std::to_string(gq) +
            ",\"computeOnlyQueues\":" + std::to_string(cq) +
            ",\"graphicsComputeQueues\":" + std::to_string(gcq);

    if (enumDevExt != nullptr) {
      uint32_t deCount = 0;
      if (enumDevExt(devs[d], nullptr, &deCount, nullptr) == VK_SUCCESS && deCount > 0) {
        std::vector<VkExtensionPropertiesRaw> dex(deCount);
        VkResult rde = enumDevExt(devs[d], nullptr, &deCount, dex.data());
        json += ",\"deviceExtensions\":[";
        uint32_t shown = 0;
        for (uint32_t i = 0; i < dex.size() && i < deCount && shown < 32; i++) {
          std::string n = J(dex[i].extensionName);
          if (n.empty()) continue;
          if (shown > 0) json += ",";
          json += "\"" + n + "\"";
          shown++;
        }
        json += "],\"deviceExtensionsCount\":" + std::to_string(deCount) +
                ",\"deviceExtensionsResult\":" + std::to_string(rde);
      }
    }
    json += "}";
  }
  json += "]";

  verdict = hasCompute ? "GPU_COMPUTE_FEASIBLE" : "NO_COMPUTE_QUEUE";
  if (destroyInst) {
    destroyInst(inst, nullptr);
  }
  dlclose(lib);
  json += ",\"verdict\":\"" + verdict + "\"}";
  return json;
}
