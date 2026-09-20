#include "nnrt_probe.h"

#include <dlfcn.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace {

// ---------------------------------------------------------------------------
// 最小类型/签名声明（逐字对齐官方头）
//   neural_network_runtime/neural_network_core.h   —— OpenHarmony 标准 SDK
//   CANNKit/hiai_helper.h                          —— HMS SDK
// 这里**不 include 官方头**：探针要能在不依赖 SDK 头路径的情况下编译，
// 也避免把 availability 属性带进来。签名必须与头文件逐字一致。
// ---------------------------------------------------------------------------

enum OhNnDeviceType {
  OH_NN_OTHERS = 0,
  OH_NN_CPU = 1,
  OH_NN_GPU = 2,
  OH_NN_ACCELERATOR = 3,
};

struct OhNnCompilation;  // 不透明句柄

using FnGetAllDevicesID = int32_t (*)(const size_t** allDevicesID, uint32_t* deviceCount);
using FnGetName = int32_t (*)(size_t deviceID, const char** name);
using FnGetType = int32_t (*)(size_t deviceID, OhNnDeviceType* deviceType);
using FnConstructFromOfflineBuffer = OhNnCompilation* (*)(const void* modelBuffer, size_t modelSize);
using FnConstructFromOfflineFile = OhNnCompilation* (*)(const char* modelPath);
using FnSetDevice = int32_t (*)(OhNnCompilation* compilation, size_t deviceID);
using FnBuild = int32_t (*)(OhNnCompilation* compilation);
using FnDestroyCompilation = void (*)(OhNnCompilation** compilation);
using FnSetPerformanceMode = int32_t (*)(OhNnCompilation* compilation, int32_t performanceMode);
using FnEnableFloat16 = int32_t (*)(OhNnCompilation* compilation, bool enableFloat16);
// 执行器与张量描述符（编译成功后读 IO 规格用）
struct OhNnExecutor;
struct NnTensorDesc;
using FnExecutorConstruct = OhNnExecutor* (*)(OhNnCompilation* compilation);
using FnExecutorDestroy = void (*)(OhNnExecutor** executor);
using FnGetInputCount = int32_t (*)(const OhNnExecutor* executor, size_t* inputCount);
using FnGetOutputCount = int32_t (*)(const OhNnExecutor* executor, size_t* outputCount);
using FnCreateInputDesc = NnTensorDesc* (*)(const OhNnExecutor* executor, size_t index);
using FnCreateOutputDesc = NnTensorDesc* (*)(const OhNnExecutor* executor, size_t index);
using FnDescGetName = int32_t (*)(const NnTensorDesc* desc, const char** name);
using FnDescGetDataType = int32_t (*)(const NnTensorDesc* desc, int32_t* dataType);
using FnDescGetShape = int32_t (*)(const NnTensorDesc* desc, int32_t** shape, size_t* shapeLength);
using FnDescDestroy = void (*)(NnTensorDesc** desc);
// 张量与推理执行
struct OhTensor;
using FnTensorCreate = OhTensor* (*)(size_t deviceID, NnTensorDesc* desc);
using FnTensorDestroy = void (*)(OhTensor** tensor);
using FnTensorGetDataBuffer = void* (*)(const OhTensor* tensor);
using FnTensorGetSize = int32_t (*)(const OhTensor* tensor, size_t* size);
using FnRunSync = int32_t (*)(OhNnExecutor* executor, OhTensor* inputTensor[],
                              size_t inputCount, OhTensor* outputTensor[], size_t outputCount);
// CANN Kit（libhiai_foundation.so）
using FnHiaiGetVersion = const char* (*)(void);
using FnHiaiCompatCheckFromBuffer = int32_t (*)(const void* data, size_t size);

/** OH_NN_DataType（取自 neural_network_runtime_type.h，逐值核对过）。 */
const char* DataTypeName(int t) {
  switch (t) {
    case 0: return "UNKNOWN";
    case 1: return "BOOL";
    case 2: return "INT8";
    case 3: return "INT16";
    case 4: return "INT32";
    case 5: return "INT64";
    case 6: return "UINT8";
    case 7: return "UINT16";
    case 8: return "UINT32";
    case 9: return "UINT64";
    case 10: return "FLOAT16";
    case 11: return "FLOAT32";
    case 12: return "FLOAT64";
    default: return "?";
  }
}

const char* DeviceTypeName(int t) {
  switch (t) {
    case OH_NN_OTHERS: return "OTHERS";
    case OH_NN_CPU: return "CPU";
    case OH_NN_GPU: return "GPU";
    case OH_NN_ACCELERATOR: return "ACCELERATOR";
    default: return "UNKNOWN";
  }
}

/** 官方 NDK 库在设备上的固定位置（/system/lib64 不可列目录，但单文件可 stat）。 */
struct LibCandidate {
  const char* soname;   // dlopen 用 soname（走 linker 默认搜索路径）
  const char* absPath;  // soname 失败时的绝对路径兜底
};

const LibCandidate kLibs[] = {
    {"libneural_network_runtime.so", "/system/lib64/ndk/libneural_network_runtime.so"},
    {"libneural_network_core.so", "/system/lib64/ndk/libneural_network_core.so"},
    {"libhiai_foundation.so", "/system/lib64/ndk/libhiai_foundation.so"},
};

struct Loaded {
  void* handle = nullptr;
  std::string how;  // "soname" / "abs" / "fail:<dlerror>"
};

Loaded TryLoad(const LibCandidate& c) {
  Loaded r;
  dlerror();
  void* h = dlopen(c.soname, RTLD_NOW | RTLD_LOCAL);
  if (h != nullptr) {
    r.handle = h;
    r.how = "soname";
    return r;
  }
  const char* e1 = dlerror();
  std::string err1 = e1 != nullptr ? e1 : "(no dlerror)";
  dlerror();
  h = dlopen(c.absPath, RTLD_NOW | RTLD_LOCAL);
  if (h != nullptr) {
    r.handle = h;
    r.how = "abs";
    return r;
  }
  const char* e2 = dlerror();
  r.how = std::string("fail:") + err1 + " | abs:" + (e2 != nullptr ? e2 : "(no dlerror)");
  return r;
}

std::string Kv(const char* k, int v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%s=%d;", k, v);
  return buf;
}

/** size_t 版本：NNRt 的 deviceID 是 64 位，按 int 打印会被截断（实测踩过）。 */
std::string KvU64(const char* k, size_t v) {
  char buf[80];
  snprintf(buf, sizeof(buf), "%s=%llu;", k, static_cast<unsigned long long>(v));
  return buf;
}

std::string Kv(const char* k, const std::string& v) {
  return std::string(k) + "=" + v + ";";
}

}  // namespace

std::string NnrtProbeReport() {
  std::string out = "ok=1;";
  out += Kv("probe", "nnrt+hiai");

  // ---- 1) 库能不能加载
  Loaded runtime = TryLoad(kLibs[0]);
  Loaded core = TryLoad(kLibs[1]);
  Loaded hiai = TryLoad(kLibs[2]);
  out += Kv("nnrt_runtime", runtime.how);
  out += Kv("nnrt_core", core.how);
  out += Kv("hiai_foundation", hiai.how);

  // ---- 2) CANN Kit 版本（最直接的自证：这条通路到底活着没有）
  if (hiai.handle != nullptr) {
    auto version = reinterpret_cast<FnHiaiGetVersion>(dlsym(hiai.handle, "HMS_HiAI_GetVersion"));
    if (version != nullptr) {
      const char* v = version();
      out += Kv("cann_version", v != nullptr ? std::string(v) : std::string("(null)"));
    } else {
      out += Kv("cann_version", "symbol-missing");
    }
  }

  // ---- 3) 枚举 NNRt 设备
  void* nh = runtime.handle != nullptr ? runtime.handle : core.handle;
  if (nh == nullptr) {
    out += Kv("devices", "no-lib");
    return out;
  }
  auto getAll = reinterpret_cast<FnGetAllDevicesID>(dlsym(nh, "OH_NNDevice_GetAllDevicesID"));
  auto getName = reinterpret_cast<FnGetName>(dlsym(nh, "OH_NNDevice_GetName"));
  auto getType = reinterpret_cast<FnGetType>(dlsym(nh, "OH_NNDevice_GetType"));
  if (getAll == nullptr || getName == nullptr || getType == nullptr) {
    out += Kv("devices", "sym-missing");
    return out;
  }

  const size_t* ids = nullptr;
  uint32_t count = 0;
  int32_t rc = getAll(&ids, &count);
  out += Kv("getAllDevices_rc", static_cast<int>(rc));
  out += Kv("device_count", static_cast<int>(count));
  if (rc != 0 || ids == nullptr) {
    return out;
  }
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = nullptr;
    int32_t nrc = getName(ids[i], &name);
    OhNnDeviceType type = OH_NN_OTHERS;
    int32_t trc = getType(ids[i], &type);
    char buf[256];
    snprintf(buf, sizeof(buf), "id=%zu,name=%s,type=%s,nameRc=%d,typeRc=%d",
             ids[i], (nrc == 0 && name != nullptr) ? name : "?", DeviceTypeName(type),
             static_cast<int>(nrc), static_cast<int>(trc));
    char key[32];
    snprintf(key, sizeof(key), "d%u", i);
    out += Kv(key, std::string(buf));
  }
  return out;
}

std::string NnrtTryModelReport(const std::vector<uint8_t>& model, int deviceIndex) {
  std::string out = "ok=1;";
  out += Kv("bytes", static_cast<int>(model.size()));

  Loaded runtime = TryLoad(kLibs[0]);
  Loaded core = TryLoad(kLibs[1]);
  Loaded hiai = TryLoad(kLibs[2]);
  void* nh = runtime.handle != nullptr ? runtime.handle : core.handle;

  // ---- A) CANN Kit 的兼容性判定（0=COMPATIBLE，1=INCOMPATIBLE）
  if (hiai.handle != nullptr && !model.empty()) {
    auto compat = reinterpret_cast<FnHiaiCompatCheckFromBuffer>(
        dlsym(hiai.handle, "HMS_HiAICompatibility_CheckFromBuffer"));
    if (compat != nullptr) {
      out += Kv("hiai_compat_code", static_cast<int>(compat(model.data(), model.size())));
      out += Kv("hiai_compat_note", "0=compatible,1=incompatible");
    } else {
      out += Kv("hiai_compat_code", "sym-missing");
    }
  } else {
    out += Kv("hiai_compat_code", hiai.handle == nullptr ? "no-lib" : "no-data");
  }

  if (nh == nullptr) {
    out += Kv("construct", "no-nnrt-lib");
    return out;
  }

  // ---- B) 用 offline model 入口吃这段字节
  auto constructBuf = reinterpret_cast<FnConstructFromOfflineBuffer>(
      dlsym(nh, "OH_NNCompilation_ConstructWithOfflineModelBuffer"));
  if (constructBuf == nullptr) {
    out += Kv("construct", "sym-missing");
    return out;
  }
  OhNnCompilation* comp = constructBuf(model.data(), model.size());
  if (comp == nullptr) {
    out += Kv("construct", "null(rejected)");
    return out;
  }
  out += Kv("construct", "ok");

  // ---- C) 选设备 → 编译
  auto getAll = reinterpret_cast<FnGetAllDevicesID>(dlsym(nh, "OH_NNDevice_GetAllDevicesID"));
  auto setDevice = reinterpret_cast<FnSetDevice>(dlsym(nh, "OH_NNCompilation_SetDevice"));
  auto build = reinterpret_cast<FnBuild>(dlsym(nh, "OH_NNCompilation_Build"));
  auto destroy = reinterpret_cast<FnDestroyCompilation>(
      dlsym(nh, "OH_NNCompilation_Destroy"));

  size_t picked = 0;
  const size_t* ids = nullptr;
  uint32_t count = 0;
  if (getAll != nullptr && getAll(&ids, &count) == 0 && ids != nullptr && count > 0) {
    picked = ids[(deviceIndex >= 0 && static_cast<uint32_t>(deviceIndex) < count) ? deviceIndex : 0];
    out += KvU64("picked_device", picked);
    out += KvU64("picked_device_all0", ids[0]);
  } else {
    out += Kv("picked_device", "none");
  }

  if (setDevice != nullptr) {
    out += Kv("set_device_rc", static_cast<int>(setDevice(comp, picked)));
  }
  int32_t brc = -1;
  if (build != nullptr) {
    // 编译这一步是真正的判定：能过说明设备端接受了这段字节的格式
    brc = build(comp);
    out += Kv("build_rc", static_cast<int>(brc));
    out += Kv("build_note", "0=SUCCESS, else see OH_NN_ReturnCode");
  }

  // ---- D) 编译成功 → 读 IO 规格（这一步把"能编译"变成"能用"）
  if (brc == 0) {
    auto execConstruct = reinterpret_cast<FnExecutorConstruct>(
        dlsym(nh, "OH_NNExecutor_Construct"));
    auto execDestroy = reinterpret_cast<FnExecutorDestroy>(dlsym(nh, "OH_NNExecutor_Destroy"));
    auto getIn = reinterpret_cast<FnGetInputCount>(dlsym(nh, "OH_NNExecutor_GetInputCount"));
    auto getOut = reinterpret_cast<FnGetOutputCount>(dlsym(nh, "OH_NNExecutor_GetOutputCount"));
    auto mkIn = reinterpret_cast<FnCreateInputDesc>(
        dlsym(nh, "OH_NNExecutor_CreateInputTensorDesc"));
    auto mkOut = reinterpret_cast<FnCreateOutputDesc>(
        dlsym(nh, "OH_NNExecutor_CreateOutputTensorDesc"));
    auto dName = reinterpret_cast<FnDescGetName>(dlsym(nh, "OH_NNTensorDesc_GetName"));
    auto dType = reinterpret_cast<FnDescGetDataType>(dlsym(nh, "OH_NNTensorDesc_GetDataType"));
    auto dShape = reinterpret_cast<FnDescGetShape>(dlsym(nh, "OH_NNTensorDesc_GetShape"));
    auto dDestroy = reinterpret_cast<FnDescDestroy>(dlsym(nh, "OH_NNTensorDesc_Destroy"));

    OhNnExecutor* exec = execConstruct != nullptr ? execConstruct(comp) : nullptr;
    if (exec == nullptr) {
      out += Kv("executor", "null");
    } else {
      out += Kv("executor", "ok");
      auto dumpDesc = [&](NnTensorDesc* d, const char* key) {
        if (d == nullptr) {
          out += Kv(key, "null");
          return;
        }
        std::string s;
        const char* nm = nullptr;
        if (dName != nullptr && dName(d, &nm) == 0 && nm != nullptr) {
          s += std::string("name=") + nm + ",";
        }
        int dt = -1;
        if (dType != nullptr) {
          int32_t v = -1;
          if (dType(d, &v) == 0) dt = v;
        }
        s += std::string("dtype=") + DataTypeName(dt) + ",";
        int32_t* shape = nullptr;
        size_t slen = 0;
        if (dShape != nullptr && dShape(d, &shape, &slen) == 0 && shape != nullptr) {
          s += "shape=[";
          for (size_t i = 0; i < slen; ++i) {
            s += std::to_string(shape[i]);
            if (i + 1 < slen) s += "x";
          }
          s += "]";
        } else {
          s += "shape=?";
        }
        out += Kv(key, s);
        if (dDestroy != nullptr) dDestroy(&d);
      };

      size_t inN = 0;
      size_t outN = 0;
      if (getIn != nullptr && getIn(exec, &inN) == 0) out += Kv("in_count", static_cast<int>(inN));
      if (getOut != nullptr && getOut(exec, &outN) == 0) {
        out += Kv("out_count", static_cast<int>(outN));
      }
      for (size_t i = 0; i < inN && i < 4; ++i) {
        char k[24];
        snprintf(k, sizeof(k), "in%zu", i);
        dumpDesc(mkIn != nullptr ? mkIn(exec, i) : nullptr, k);
      }
      for (size_t i = 0; i < outN && i < 4; ++i) {
        char k[24];
        snprintf(k, sizeof(k), "out%zu", i);
        dumpDesc(mkOut != nullptr ? mkOut(exec, i) : nullptr, k);
      }
      // ---- E) 在设备上真跑一次推理：确定性输入 → RunSync ×3 → 输出统计 + 计时
      auto tCreate = reinterpret_cast<FnTensorCreate>(dlsym(nh, "OH_NNTensor_Create"));
      auto tDestroy = reinterpret_cast<FnTensorDestroy>(dlsym(nh, "OH_NNTensor_Destroy"));
      auto tBuf = reinterpret_cast<FnTensorGetDataBuffer>(dlsym(nh, "OH_NNTensor_GetDataBuffer"));
      auto tSize = reinterpret_cast<FnTensorGetSize>(dlsym(nh, "OH_NNTensor_GetSize"));
      auto runSync = reinterpret_cast<FnRunSync>(dlsym(nh, "OH_NNExecutor_RunSync"));
      if (tCreate != nullptr && tBuf != nullptr && runSync != nullptr && inN >= 1 && outN >= 1) {
        // 多输入/多输出：**必须建齐并全部传给 RunSync**。
        // 踩过的坑：裸 head 检测模型声明 3 个输出，只喂 1 个 → run_rc=1（OH_NN_FAILED），
        // 表面上像"执行失败"，实际是调用方张量数量不匹配 —— 会误判成模型不可用。
        std::vector<NnTensorDesc*> inDescs;
        std::vector<NnTensorDesc*> outDescs;
        std::vector<OhTensor*> inTensors;
        std::vector<OhTensor*> outTensors;
        bool tensorsOk = true;
        for (size_t i = 0; i < inN; ++i) {
          NnTensorDesc* d = mkIn != nullptr ? mkIn(exec, i) : nullptr;
          OhTensor* t = d != nullptr ? tCreate(picked, d) : nullptr;
          if (t == nullptr) tensorsOk = false;
          inDescs.push_back(d);
          inTensors.push_back(t);
        }
        for (size_t i = 0; i < outN; ++i) {
          NnTensorDesc* d = mkOut != nullptr ? mkOut(exec, i) : nullptr;
          OhTensor* t = d != nullptr ? tCreate(picked, d) : nullptr;
          if (t == nullptr) tensorsOk = false;
          outDescs.push_back(d);
          outTensors.push_back(t);
        }

        if (!tensorsOk) {
          out += Kv("tensors", "create-failed");
        } else {
          size_t inBytes = 0;
          size_t outBytes = 0;
          // 确定性填充，沿用 ADR-004 §3.4 的同一式：x[j] = ((j*2654435761) mod 1000)/1000
          for (size_t i = 0; i < inTensors.size(); ++i) {
            size_t sz = 0;
            if (tSize != nullptr) tSize(inTensors[i], &sz);
            inBytes += sz;
            float* f = static_cast<float*>(tBuf(inTensors[i]));
            const size_t n = sz / sizeof(float);
            if (f != nullptr) {
              for (size_t j = 0; j < n; ++j) {
                f[j] = static_cast<float>((j * 2654435761ULL) % 1000ULL) / 1000.0f;
              }
            }
          }
          for (size_t i = 0; i < outTensors.size(); ++i) {
            size_t sz = 0;
            if (tSize != nullptr) tSize(outTensors[i], &sz);
            outBytes += sz;
          }
          out += Kv("in_bytes", static_cast<int>(inBytes));
          out += Kv("out_bytes", static_cast<int>(outBytes));
          out += Kv("run_tensors", std::to_string(inTensors.size()) + "in/" +
                                       std::to_string(outTensors.size()) + "out");

          std::string msList;
          int lastRc = -1;
          for (int rep = 0; rep < 3; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            lastRc = runSync(exec, inTensors.data(), inTensors.size(),
                             outTensors.data(), outTensors.size());
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            char b[32];
            snprintf(b, sizeof(b), "%.3f", ms);
            if (!msList.empty()) msList += "|";
            msList += b;
            if (lastRc != 0) break;
          }
          out += Kv("run_rc", lastRc);
          out += Kv("run_ms_each", msList);

          // 逐输出统计（和 / argmax / 最大值）。多输出全列，便于判断"哪个头没出数"。
          for (size_t i = 0; i < outTensors.size() && i < 4; ++i) {
            size_t sz = 0;
            if (tSize != nullptr) tSize(outTensors[i], &sz);
            const float* f = static_cast<const float*>(tBuf(outTensors[i]));
            const size_t on = sz / sizeof(float);
            char key[32];
            if (f == nullptr || on == 0) {
              snprintf(key, sizeof(key), "o%zu_stats", i);
              out += Kv(key, "empty");
              continue;
            }
            double sum = 0.0;
            int am = -1;
            float amv = -1e30f;
            for (size_t j = 0; j < on; ++j) {
              sum += static_cast<double>(f[j]);
              if (f[j] > amv) {
                amv = f[j];
                am = static_cast<int>(j);
              }
            }
            char b[48];
            snprintf(key, sizeof(key), "o%zu_sum", i);
            snprintf(b, sizeof(b), "%.4f", sum);
            out += Kv(key, std::string(b));
            snprintf(key, sizeof(key), "o%zu_argmax", i);
            out += Kv(key, am);
            snprintf(key, sizeof(key), "o%zu_max", i);
            snprintf(b, sizeof(b), "%.4f", static_cast<double>(amv));
            out += Kv(key, std::string(b));
          }
          out += Kv("out_note", "synthetic-input->argmax NOT meaningful");
        }

        for (size_t i = 0; i < inTensors.size(); ++i) {
          OhTensor* t = inTensors[i];
          if (t != nullptr && tDestroy != nullptr) tDestroy(&t);
        }
        for (size_t i = 0; i < outTensors.size(); ++i) {
          OhTensor* t = outTensors[i];
          if (t != nullptr && tDestroy != nullptr) tDestroy(&t);
        }
        if (dDestroy != nullptr) {
          for (size_t i = 0; i < inDescs.size(); ++i) {
            NnTensorDesc* d = inDescs[i];
            if (d != nullptr) dDestroy(&d);
          }
          for (size_t i = 0; i < outDescs.size(); ++i) {
            NnTensorDesc* d = outDescs[i];
            if (d != nullptr) dDestroy(&d);
          }
        }
      } else {
        out += Kv("run", "api-missing");
      }

      if (execDestroy != nullptr) {
        execDestroy(&exec);
        out += Kv("executor_destroyed", 1);
      }
    }
  }

  if (destroy != nullptr) {
    destroy(&comp);
    out += Kv("destroyed", 1);
  }
  return out;
}