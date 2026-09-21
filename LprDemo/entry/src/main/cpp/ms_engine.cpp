#include "ms_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include <hilog/log.h>

#define LPR_TAG "LprNative"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, LPR_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, 0xD001, LPR_TAG, __VA_ARGS__)

// ---------------------------------------------------------------- fp16 helpers
static float HalfToFloat(uint16_t h);  // defined below; Fingerprint needs it

/**
 * Tensor fingerprint (ADR-0003): L2 norm of an output buffer read two ways.
 *
 * `declared` reads the buffer according to the dtype MindSpore Lite DECLARED;
 * `asFp16` re-reads the same bytes as fp16. When the two disagree wildly the
 * buffer is an fp16 bitstream mislabelled as fp32 — the known NNRT defect.
 * Returns {declared, asFp16}; `asFp16` equals `declared` when the declared
 * dtype already is fp16.
 */
static void Fingerprint(const void* buf, size_t elems, int dtype,
                        double& declared, double& asFp16) {
  declared = 0.0;
  asFp16 = 0.0;
  if (buf == nullptr || elems == 0) {
    return;
  }
  if (dtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(buf);
    for (size_t i = 0; i < elems; i++) {
      const double v = HalfToFloat(h[i]);
      declared += v * v;
      asFp16 += v * v;
    }
  } else {
    const float* f = reinterpret_cast<const float*>(buf);
    for (size_t i = 0; i < elems; i++) {
      const double v = static_cast<double>(f[i]);
      declared += v * v;
    }
    const uint16_t* h = reinterpret_cast<const uint16_t*>(buf);
    for (size_t i = 0; i < elems; i++) {
      const double v = HalfToFloat(h[i]);
      asFp16 += v * v;
    }
  }
  declared = std::sqrt(declared);
  asFp16 = std::sqrt(asFp16);
}

static float HalfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) & 1u;
  uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
  uint32_t man = (uint32_t)h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign << 31;
    } else {
      uint32_t e = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        e--;
      }
      man &= 0x3FFu;
      f = (sign << 31) | (e << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000u | (man << 13);
  } else {
    f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

static uint16_t FloatToHalf(float v) {
  uint32_t f;
  std::memcpy(&f, &v, 4);
  uint32_t sign = (f >> 31) & 1u;
  int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = f & 0x7FFFFFu;
  if (exp <= 0) {
    return (uint16_t)(sign << 15);  // flush subnormals to zero: our inputs are in [-1,1]
  }
  if (exp >= 31) {
    return (uint16_t)((sign << 15) | 0x7C00u);
  }
  return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (man >> 13));
}

// ---------------------------------------------------------------- NNRT device selection
/**
 * Priority: the hardware-named device first, then HIAI, then anything else.
 *
 * Kirin 8020 (HarmonyOS 6.1.1 / API 24) reports TWO NNRT devices:
 *   [0] NPU_ohos.boot.hardware.kirin8020_v2_0
 *   [1] HIAI_F
 *
 * Measured on device (2026-09-17):
 *   - `HIAI_F` is routed to BuildOfflineModel and always dies at
 *     `nnrt_delegate.cc:168 BuildOfflineModel# not third party model`.
 *   - `NPU_ohos.boot.hardware.kirin8020_v2_0` is routed to BuildKirinNPUModel,
 *     which is the only path that ever succeeds.
 * So "HIAI first" (an earlier note taken from a different device) is WRONG here.
 * Note the earlier note also claimed CheckNPUPrefix *rejects* the kirin device:
 * in fact CheckNPUPrefix logs a strncmp miss for BOTH names and then routes them
 * down different builders, so its log line alone is not a verdict.
 */
static int DeviceRank(const std::string& n) {
  std::string low = n;
  std::transform(low.begin(), low.end(), low.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  if (low.find("kirin") != std::string::npos || low.find("ohos.boot.hardware") != std::string::npos) {
    return 0;
  }
  if (n.rfind("HIAI", 0) == 0) {
    return 1;
  }
  if (low.find("npu") != std::string::npos) {
    return 2;
  }
  return 3;
}

std::vector<std::string> MsNnrtCandidates() {
  std::vector<std::string> names;
  size_t num = 0;
  NNRTDeviceDesc* descs = OH_AI_GetAllNNRTDeviceDescs(&num);
  if (descs != nullptr) {
    for (size_t i = 0; i < num; i++) {
      NNRTDeviceDesc* d = OH_AI_GetElementOfNNRTDeviceDescs(descs, i);
      if (d == nullptr) {
        continue;
      }
      const char* nm = OH_AI_GetNameFromNNRTDeviceDesc(d);
      if (nm != nullptr) {
        names.emplace_back(nm);
      }
    }
    OH_AI_DestroyAllNNRTDeviceDescs(&descs);
  }
  std::stable_sort(names.begin(), names.end(),
                   [](const std::string& a, const std::string& b) {
                     return DeviceRank(a) < DeviceRank(b);
                   });
  return names;
}

// ---------------------------------------------------------------- attempt plan
struct Attempt {
  OH_AI_DeviceType type;
  std::string label;     // what the UI/log shows
  std::string nnrtName;  // only for OH_AI_DEVICETYPE_NNRT
  bool fp16;             // set OH_AI_DeviceInfoSetEnableFP16
  int threads = 0;       // CPU only; <=0 = leave the engine default (4)
  /**
   * CPU only：线程亲和。0=不设（引擎默认，无绑定）/ 1=大核优先 / 2=小核优先。
   *
   * 【2026-09-21 新增】动机：同一份 det 模型、同一后端、同样 4 线程，
   * **隔离基准 7.4 ms，但相机流水线内 15.3–19.5 ms（慢 2 倍）**，且更热的
   * 那一轮反而更快 —— 所以主因不是热降频，而**可能是 DVFS/核位调度**
   * （详见 docs/notes/camera-npu-headroom.md §3.2c）。
   * `OH_AI_ContextSetThreadAffinityMode` 此前**从未被调用过**。
   * 这是验证该假设最便宜的一刀：不写任何算子代码，只钉核位。
   */
  int affinity = 0;
};

static std::vector<Attempt> PlanFor(const std::string& backend) {
  std::vector<Attempt> plan;
  auto addNnrt = [&plan](bool fp16) {
    for (const std::string& n : MsNnrtCandidates()) {
      // The dtype suffix belongs in the label: `backend` is what the log prints as
      // LANDED=, and an fp16/fp32 NPU build must never look like the same landing.
      plan.push_back({OH_AI_DEVICETYPE_NNRT, "NNRT:" + n + (fp16 ? "" : "#fp32"), n, fp16});
    }
  };
  if (backend == "nnrt") {
    // 2026-09-18: 默认改为 fp16 —— fp16/fp32 下 det 都会翻字符，但 rec/cls 在 NPU 上有显著收益
    // 所以生产档用 fp16 以最大化 rec/cls 性能。如果用户想测 fp32，显式用 nnrt_fp32 档。
    addNnrt(true);
  } else if (backend == "nnrt_fp32") {
    addNnrt(false);
  } else if (backend == "nnrt_fp16") {
    // 显式 fp16 档，用于对比实验
    addNnrt(true);
  } else if (backend.rfind("cpu_t", 0) == 0) {
    // 2026-09-20: CPU 线程扫描档 cpu_t1 / cpu_t2 / cpu_t4 / cpu_t6 / cpu_t8 ——
    // 回答「CPU 到底几线程最快」：此前线程数硬编码 4 且从未扫描，
    // 所有 NPU/CPU 比值的 CPU 端都不是调优基线。
    // 只接受纯数字后缀（≤2 位）：畸形档名退回普通 CPU 档，绝不让 stoi 的异常
    // 穿透 NAPI 边界把进程带走。
    const std::string num = backend.substr(5);
    const bool okNum = !num.empty() && num.size() <= 2 &&
                       num.find_first_not_of("0123456789") == std::string::npos;
    if (okNum) {
      const int n = std::stoi(num);
      plan.push_back({OH_AI_DEVICETYPE_CPU, "CPU:t" + std::to_string(n), "", false, n});
    } else {
      plan.push_back({OH_AI_DEVICETYPE_CPU, "CPU", "", false, 0});
    }
  } else if (backend.rfind("cpu_a", 0) == 0) {
    // 2026-09-21: CPU 亲和扫描档 cpu_a1（大核优先）/ cpu_a2（小核优先）。
    // 动机见 Attempt::affinity 的注释 —— 用于验证「流水线内 det 慢 2 倍」
    // 是否来自 DVFS/核位调度。线程数沿用 4（已由 cpu_t 扫描证明最优）。
    const std::string num = backend.substr(5);
    const bool okNum = !num.empty() && num.size() <= 2 &&
                       num.find_first_not_of("0123456789") == std::string::npos;
    if (okNum) {
      const int n = std::stoi(num);
      plan.push_back({OH_AI_DEVICETYPE_CPU, "CPU:t4a" + std::to_string(n), "", false, 4, n});
    } else {
      plan.push_back({OH_AI_DEVICETYPE_CPU, "CPU", "", false, 0});
    }
  } else if (backend == "gpu") {
    plan.push_back({OH_AI_DEVICETYPE_GPU, "GPU:fp16", "", true});
    plan.push_back({OH_AI_DEVICETYPE_GPU, "GPU:fp32", "", false});
  } else if (backend == "kirin") {
    plan.push_back({OH_AI_DEVICETYPE_KIRIN_NPU, "KIRIN_NPU", "", true});
  } else if (backend != "cpu") {
    // "auto" and anything unrecognised: widest net, cheapest-first is not used on
    // purpose — the accelerator order matters more than the build time.
    addNnrt(true);
    plan.push_back({OH_AI_DEVICETYPE_GPU, "GPU:fp16", "", true});
    plan.push_back({OH_AI_DEVICETYPE_GPU, "GPU:fp32", "", false});
    plan.push_back({OH_AI_DEVICETYPE_KIRIN_NPU, "KIRIN_NPU", "", true});
  }
  plan.push_back({OH_AI_DEVICETYPE_CPU, "CPU", "", false});
  return plan;
}

// ---------------------------------------------------------------- build
static bool TryBuild(MsSession* s, const Attempt& a, std::string& err) {
  OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
  if (ctx == nullptr) {
    err = "ContextCreate returned null";
    return false;
  }
  OH_AI_ModelHandle model = OH_AI_ModelCreate();
  if (model == nullptr) {
    err = "ModelCreate returned null";
    s->retiredCtx.push_back(ctx);
    return false;
  }

  OH_AI_DeviceInfoHandle dev = nullptr;
  if (a.type == OH_AI_DEVICETYPE_CPU) {
    // 2026-09-18: 榨干 CPU 性能 —— 设为高性能模式 + 线程数扫描
    OH_AI_ContextSetThreadNum(ctx, a.threads > 0 ? a.threads : 4);  // cpu_t{N} 档覆写，默认 4
    // 2026-09-21: 线程亲和（cpu_a{N} 档）。0 表示不设，保持历史行为不变。
    if (a.affinity > 0) {
      OH_AI_ContextSetThreadAffinityMode(ctx, a.affinity);
    }
    dev = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
    // 关键：CPU 也要设高性能模式！之前漏了，导致 CPU 跑在节能档
    OH_AI_DeviceInfoSetPerformanceMode(dev, OH_AI_PERFORMANCE_HIGH);
  } else if (a.type == OH_AI_DEVICETYPE_NNRT) {
    dev = OH_AI_CreateNNRTDeviceInfoByName(a.nnrtName.c_str());
  } else {
    dev = OH_AI_DeviceInfoCreate(a.type);
  }
  if (dev == nullptr) {
    err = "DeviceInfoCreate failed for " + a.label;
    s->retiredCtx.push_back(ctx);
    s->retiredModel.push_back(model);
    return false;
  }
  if (a.type != OH_AI_DEVICETYPE_CPU) {
    OH_AI_DeviceInfoSetEnableFP16(dev, a.fp16);
    OH_AI_DeviceInfoSetPerformanceMode(dev, OH_AI_PERFORMANCE_HIGH);
  }
  OH_AI_ContextAddDeviceInfo(ctx, dev);

  OH_AI_Status st = OH_AI_ModelBuild(model, s->modelBytes.data(), s->modelBytes.size(),
                                     OH_AI_MODELTYPE_MINDIR, ctx);
  if (st != OH_AI_STATUS_SUCCESS) {
    err = a.label + " ModelBuild status=" + std::to_string((int)st);
    // Retain instead of destroy: a failed build may still have touched the NNRT
    // delegate path, and destroying it is the documented cppcrash trigger.
    s->retiredCtx.push_back(ctx);
    s->retiredModel.push_back(model);
    return false;
  }

  s->ctx = ctx;
  s->model = model;
  s->backend = a.label;
  return true;
}

static void FillIo(MsSession* s) {
  OH_AI_TensorHandleArray ins = OH_AI_ModelGetInputs(s->model);
  OH_AI_TensorHandleArray outs = OH_AI_ModelGetOutputs(s->model);
  // Cache the handle arrays on the session: the per-inference path (MsRun /
  // MsRunMulti) then stops paying for GetInputs/GetOutputs on every frame
  // (A18 section 8). Handles stay valid for the model's lifetime.
  s->ins = ins;
  s->outs = outs;

  // Dynamic-shape guard: a dim <= 0 builds fine but predicts with -1. Our three
  // models are fully static, but keep the fallback so swapping models stays safe.
  bool dyn = false;
  for (size_t i = 0; i < ins.handle_num; i++) {
    size_t n = 0;
    const int64_t* shp = OH_AI_TensorGetShape(ins.handle_list[i], &n);
    for (size_t k = 0; k < n; k++) {
      if (shp[k] <= 0) {
        dyn = true;
      }
    }
  }
  if (dyn) {
    std::vector<OH_AI_ShapeInfo> infos(ins.handle_num);
    for (size_t i = 0; i < ins.handle_num; i++) {
      size_t n = 0;
      const int64_t* shp = OH_AI_TensorGetShape(ins.handle_list[i], &n);
      infos[i].shape_num = n;
      for (size_t k = 0; k < n && k < OH_AI_MAX_SHAPE_NUM; k++) {
        infos[i].shape[k] = (shp[k] <= 0) ? 1 : shp[k];
      }
    }
    OH_AI_Status rs = OH_AI_ModelResize(s->model, ins, infos.data(), ins.handle_num);
    LOGI("dynamic shape detected, ModelResize status=%{public}d", (int)rs);
    ins = OH_AI_ModelGetInputs(s->model);
    outs = OH_AI_ModelGetOutputs(s->model);
    s->ins = ins;  // re-cache: resize may have replaced the tensors
    s->outs = outs;
  }

  if (ins.handle_num >= 1) {
    OH_AI_TensorHandle it = ins.handle_list[0];
    const char* inm = OH_AI_TensorGetName(it);
    s->inputName = (inm != nullptr) ? inm : "input";
    size_t n = 0;
    const int64_t* shp = OH_AI_TensorGetShape(it, &n);
    s->inputShape.assign(shp, shp + n);
    s->inputFormat = OH_AI_TensorGetFormat(it);
    s->inputDtype = OH_AI_TensorGetDataType(it);
    s->inputElems = (size_t)OH_AI_TensorGetElementNum(it);
  }
  if (outs.handle_num >= 1) {
    OH_AI_TensorHandle ot = outs.handle_list[0];
    const char* onm = OH_AI_TensorGetName(ot);
    s->outputName = (onm != nullptr) ? onm : "output";
    size_t n = 0;
    const int64_t* shp = OH_AI_TensorGetShape(ot, &n);
    s->outputShape.assign(shp, shp + n);
    s->outputFormat = OH_AI_TensorGetFormat(ot);
    s->outputDtype = OH_AI_TensorGetDataType(ot);
    s->outputElems = (size_t)OH_AI_TensorGetElementNum(ot);
  }
}

MsSession* MsLoad(const std::vector<char>& bytes, const std::string& backend, std::string& err) {
  if (bytes.empty()) {
    err = "empty model buffer";
    return nullptr;
  }
  MsSession* s = new MsSession();
  s->modelBytes = bytes;

  std::vector<Attempt> plan = PlanFor(backend);
  LOGI("backend plan for '%{public}s': %{public}zu attempt(s)", backend.c_str(), plan.size());

  std::string trail;
  bool ok = false;
  for (const Attempt& a : plan) {
    std::string e;
    if (TryBuild(s, a, e)) {
      LOGI("model built on %{public}s", a.label.c_str());
      trail += a.label + "(ok)";
      ok = true;
      break;
    }
    LOGE("build attempt failed: %{public}s", e.c_str());
    trail += a.label + "(fail) ";
  }
  s->attemptLog = trail;

  if (!ok) {
    err = "all backends failed; last=" + trail;
    // Do NOT delete s: the failed attempts' contexts/models are retained inside it
    // precisely because destroying them crashes the process.
    return nullptr;
  }

  s->requested = backend;
  // A plan always ends at CPU, so landing anywhere but the first attempt means the
  // requested accelerator was NOT used. Say so explicitly instead of letting the UI
  // print a bare LANDED=CPU that reads like a choice.
  if (!plan.empty() && plan[0].label != s->backend) {
    s->fallbackFrom = plan[0].label;
    LOGE("requested '%{public}s' but landed on '%{public}s' (fell back from '%{public}s')",
         backend.c_str(), s->backend.c_str(), s->fallbackFrom.c_str());
  }

  FillIo(s);
  if (s->inputElems == 0 || s->outputElems == 0) {
    err = "model reports zero-sized input/output";
    return nullptr;
  }

  std::string inShapeStr;
  for (size_t i = 0; i < s->inputShape.size(); i++) {
    if (i > 0) {
      inShapeStr += ",";
    }
    inShapeStr += std::to_string((long long)s->inputShape[i]);
  }
  LOGI("loaded backend=%{public}s trail=[%{public}s] in=%{public}s shape=[%{public}s] fmt=%{public}d "
       "elems=%{public}zu dtype=%{public}d out=%{public}s elems=%{public}zu dtype=%{public}d",
       s->backend.c_str(), s->attemptLog.c_str(), s->inputName.c_str(), inShapeStr.c_str(),
       (int)s->inputFormat, s->inputElems, (int)s->inputDtype, s->outputName.c_str(),
       s->outputElems, (int)s->outputDtype);
  return s;
}

// ---------------------------------------------------------------- run
bool MsRun(MsSession* s, const float* in, std::vector<float>& out, std::string& err) {
  if (s == nullptr || s->model == nullptr) {
    err = "session is null";
    return false;
  }
  if (in == nullptr) {
    err = "input is null";
    return false;
  }

  // Cached at load (A18 section 8): the per-call GetInputs/GetOutputs pair is
  // what the pipeline used to pay on every frame.
  OH_AI_TensorHandleArray ins = s->ins.handle_num > 0 ? s->ins
                                                      : OH_AI_ModelGetInputs(s->model);
  OH_AI_TensorHandleArray outs = s->outs.handle_num > 0 ? s->outs
                                                        : OH_AI_ModelGetOutputs(s->model);
  if (ins.handle_num < 1 || outs.handle_num < 1) {
    err = "model has no tensor";
    return false;
  }

  OH_AI_TensorHandle it = ins.handle_list[0];
  void* dst = OH_AI_TensorGetMutableData(it);
  if (dst == nullptr) {
    err = "input tensor data is null";
    return false;
  }
  size_t n = s->inputElems;
  if (s->inputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
    std::memcpy(dst, in, n * sizeof(float));
  } else if (s->inputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    uint16_t* h = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = 0; i < n; i++) {
      h[i] = FloatToHalf(in[i]);
    }
  } else {
    err = "unsupported input dtype=" + std::to_string((int)s->inputDtype);
    return false;
  }

  OH_AI_Status st = OH_AI_ModelPredict(s->model, ins, &outs, nullptr, nullptr);
  if (st != OH_AI_STATUS_SUCCESS) {
    err = "ModelPredict failed status=" + std::to_string((int)st);
    return false;
  }

  OH_AI_TensorHandle ot = outs.handle_list[0];
  size_t on = (size_t)OH_AI_TensorGetElementNum(ot);
  const void* od = OH_AI_TensorGetData(ot);
  if (od == nullptr) {
    err = "output tensor data is null";
    return false;
  }
  out.resize(on);
  if (s->outputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
    std::memcpy(out.data(), od, on * sizeof(float));
  } else if (s->outputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(od);
    for (size_t i = 0; i < on; i++) {
      out[i] = HalfToFloat(h[i]);
    }
  } else {
    err = "unsupported output dtype=" + std::to_string((int)s->outputDtype);
    return false;
  }
  // Landing evidence (ADR-0003): NPU utilisation is unreadable on this platform,
  // so the output buffer's own statistics are what proves the backend ran.
  Fingerprint(od, on, (int)s->outputDtype, s->lastL2, s->lastL2AsFp16);
  return true;
}

// ---------------------------------------------------------------- bench
bool MsRunMulti(MsSession* s, const float* in,
                std::vector<std::vector<float>>& outs, std::string& err) {
  if (s == nullptr || s->model == nullptr) {
    err = "session is null";
    return false;
  }
  if (in == nullptr) {
    err = "input is null";
    return false;
  }

  // Cached at load (A18 section 8) — see MsRun above.
  OH_AI_TensorHandleArray ins = s->ins.handle_num > 0 ? s->ins
                                                      : OH_AI_ModelGetInputs(s->model);
  OH_AI_TensorHandleArray os = s->outs.handle_num > 0 ? s->outs
                                                      : OH_AI_ModelGetOutputs(s->model);
  if (ins.handle_num < 1 || os.handle_num < 1) {
    err = "model has no tensor";
    return false;
  }

  OH_AI_TensorHandle it = ins.handle_list[0];
  void* dst = OH_AI_TensorGetMutableData(it);
  if (dst == nullptr) {
    err = "input tensor data is null";
    return false;
  }
  size_t n = s->inputElems;
  if (s->inputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
    std::memcpy(dst, in, n * sizeof(float));
  } else if (s->inputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    uint16_t* h = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = 0; i < n; i++) {
      h[i] = FloatToHalf(in[i]);
    }
  } else {
    err = "unsupported input dtype=" + std::to_string((int)s->inputDtype);
    return false;
  }

  OH_AI_Status st = OH_AI_ModelPredict(s->model, ins, &os, nullptr, nullptr);
  if (st != OH_AI_STATUS_SUCCESS) {
    err = "ModelPredict failed status=" + std::to_string((int)st);
    return false;
  }

  // No clear(): the caller may pass a reused buffer (pipeline scratch, A18 §1) —
  // resize() fixes the size and the inner vectors keep their capacity.
  outs.resize(os.handle_num);
  for (size_t k = 0; k < os.handle_num; k++) {
    OH_AI_TensorHandle ot = os.handle_list[k];
    size_t on = (size_t)OH_AI_TensorGetElementNum(ot);
    const void* od = OH_AI_TensorGetData(ot);
    if (od == nullptr) {
      err = "output tensor data is null (output " + std::to_string(k) + ")";
      return false;
    }
    outs[k].resize(on);
    if (s->outputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT32) {
      std::memcpy(outs[k].data(), od, on * sizeof(float));
    } else if (s->outputDtype == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
      const uint16_t* hp = reinterpret_cast<const uint16_t*>(od);
      for (size_t i = 0; i < on; i++) {
        outs[k][i] = HalfToFloat(hp[i]);
      }
    } else {
      err = "unsupported output dtype=" + std::to_string((int)s->outputDtype);
      return false;
    }
    if (k == 0) {
      // Landing evidence on the primary output (ADR-0003). The bare-head
      // detector has three [1,45,H,H] outputs; output 0 is the one the
      // pipeline consumes, so it is the one whose statistics prove the run.
      Fingerprint(od, on, (int)s->outputDtype, s->lastL2, s->lastL2AsFp16);
    }
  }
  return true;
}

// ---------------------------------------------------------------- bench
/**
 * Deterministic pseudo-random fill, identical to the PC reference protocol
 * (ShusenPaper `ref_checksum.py`): x[j] = ((j * 2654435761) % 1000) / 1000.
 * Kept byte-identical so an NPU number here can be compared against a PC run.
 */
static void FillDeterministic(void* dst, size_t elems, OH_AI_DataType dt) {
  // A18 section 8, run r7: an experiment filled this with the pipeline's
  // mostly-zeros distribution; bench time did not move (7.54 vs 7.44 ms), so
  // the fill is back to the dense deterministic pattern the CPU-diff protocol
  // specifies.
  if (dt == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    uint16_t* p = reinterpret_cast<uint16_t*>(dst);
    for (size_t j = 0; j < elems; j++) {
      float v = (float)((j * 2654435761u) % 1000) / 1000.0f;
      p[j] = FloatToHalf(v);
    }
  } else {
    float* p = reinterpret_cast<float*>(dst);
    for (size_t j = 0; j < elems; j++) {
      p[j] = (float)((j * 2654435761u) % 1000) / 1000.0f;
    }
  }
}

/** L2 + maxAbs of a buffer read as the declared dtype. */
static void SumL2(const void* buf, size_t n, OH_AI_DataType dt, double& l2, double& maxAbs) {
  double acc = 0.0;
  maxAbs = 0.0;
  if (dt == OH_AI_DATATYPE_NUMBERTYPE_FLOAT16) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(buf);
    for (size_t j = 0; j < n; j++) {
      float v = HalfToFloat(p[j]);
      acc += (double)v * v;
      double a = std::fabs((double)v);
      if (a > maxAbs) maxAbs = a;
    }
  } else {
    const float* p = reinterpret_cast<const float*>(buf);
    for (size_t j = 0; j < n; j++) {
      acc += (double)p[j] * p[j];
      double a = std::fabs((double)p[j]);
      if (a > maxAbs) maxAbs = a;
    }
  }
  l2 = std::sqrt(acc);
}

/** Same bytes, reinterpreted as fp16 — the NNRT mislabelled-dtype fingerprint. */
static double SumL2AsFp16(const void* buf, size_t n) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(buf);
  double acc = 0.0;
  for (size_t j = 0; j < n; j++) {
    float v = HalfToFloat(p[j]);
    acc += (double)v * v;
  }
  return std::sqrt(acc);
}

static double Pctl(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(p * (v.size() - 1) + 0.5);
  if (i >= v.size()) i = v.size() - 1;
  return v[i];
}

MsBench MsBenchRun(MsSession* s, int warmup, int repeat, double gapMs, int polluteKB,
                   double spinMs) {
  MsBench r;
  if (s == nullptr || s->model == nullptr) {
    r.error = "session is null";
    return r;
  }
  if (warmup < 0) warmup = 0;
  if (repeat < 1) repeat = 1;
  r.backend = s->backend;
  r.warmup = warmup;
  r.repeat = repeat;
  r.gapMs = gapMs;
  r.polluteKB = polluteKB;
  r.spinMs = spinMs;

  OH_AI_TensorHandleArray ins = OH_AI_ModelGetInputs(s->model);
  OH_AI_TensorHandleArray outs = OH_AI_ModelGetOutputs(s->model);
  if (ins.handle_num < 1 || outs.handle_num < 1) {
    r.error = "model has no tensor";
    return r;
  }
  for (size_t i = 0; i < ins.handle_num; i++) {
    OH_AI_TensorHandle t = ins.handle_list[i];
    void* buf = OH_AI_TensorGetMutableData(t);
    if (buf == nullptr) {
      r.error = "input tensor data is null";
      return r;
    }
    FillDeterministic(buf, (size_t)OH_AI_TensorGetElementNum(t), OH_AI_TensorGetDataType(t));
  }

  for (int i = 0; i < warmup; i++) {
    OH_AI_Status st = OH_AI_ModelPredict(s->model, ins, &outs, nullptr, nullptr);
    if (st != OH_AI_STATUS_SUCCESS) {
      r.error = "warmup predict failed status=" + std::to_string((int)st);
      return r;
    }
  }

  std::vector<double> times;
  times.reserve(repeat);
  // 干扰缓冲：只在 polluteKB>0 时分配。它模拟流水线里 conv 写的 1.2 MB RGBA
  // 与 letterbox 输出对缓存/内存带宽的占用。
  std::vector<uint8_t> pollute;
  if (polluteKB > 0) {
    pollute.assign(static_cast<size_t>(polluteKB) * 1024, 0xA5);
  }
  for (int i = 0; i < repeat; i++) {
    // ---- 迭代之间的干扰（**不在计时区内**）----
    // 关键：这几件事都发生在 t0 之前，所以被计时的那次 ModelPredict 本身
    // 与历史基线逐位相同。变的只是「这次推理是在什么状态下被调用」。
    //
    // 【2026-09-21 结论】这三档把「隔离 7.3 ms vs 流水线 19.5 ms」定位成了
    // **两个叠加的机理**（详见 docs/notes/camera-npu-headroom.md §3.2c.2）：
    //   1. DVFS：gapMs 单调把 p50 从 7.4 推到 30.7 ms；在 cpu_t1 上
    //      用 spinMs（忙等）可**完全消除**该效应（+26.8 -> +0.3 ms）⇒ 是频率。
    //   2. 线程池唤醒：cpu_t4 上 spinMs **救不回来**（仍 +20.8 ms），
    //      因为忙等只占住调用线程，另外 3 个工作线程照样 park，
    //      下次 predict 要先唤醒它们 —— 这段开销落在计时区内。
    //   polluteKB=1200（≈conv 的 1.2 MB RGBA 写）**无影响** ⇒ 缓存/带宽假设被否。
    //
    // 注意：这里曾有一条旧结论「sleep 3 ms 只把 p50 从 7.54 推到 8.71 ms，
    // 所以负载节奏解释证据薄弱」—— **那是剂量太小**。3 ms 落在剂量-反应曲线
    // 的第一段，斜率尚未起来；8/16/33 ms 下效应非常清楚。
    if (gapMs > 0) {
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(gapMs));
    }
    if (spinMs > 0) {
      // 对照组：**忙等**同样长的时间。与 sleep 的区别只有一个 ——
      // 这段时间里 CPU 是**忙**的，所以频率不会被降下去。
      // 若这一档的 p50 回到紧循环水平，就证明「掉频」是主因，
      // 而不是「两次调用间隔了多久」这件事本身。
      const auto spinEnd = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double, std::milli>(spinMs));
      volatile uint64_t acc = 0;
      while (std::chrono::steady_clock::now() < spinEnd) {
        for (int k = 0; k < 1000; k++) {
          acc = acc * 6364136223846793005ULL + 1442695040888963407ULL;
        }
      }
      (void)acc;
    }
    if (!pollute.empty()) {
      // 逐字节读+写，确保真的过一遍缓存与内存，不被编译器优化掉。
      volatile uint8_t sink = 0;
      for (size_t k = 0; k < pollute.size(); k += 64) {
        pollute[k] = static_cast<uint8_t>(pollute[k] + 1);
        sink = static_cast<uint8_t>(sink + pollute[k]);
      }
      (void)sink;
    }
    auto t0 = std::chrono::steady_clock::now();
    OH_AI_Status st = OH_AI_ModelPredict(s->model, ins, &outs, nullptr, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    if (st != OH_AI_STATUS_SUCCESS) {
      r.error = "predict failed at run " + std::to_string(i) + " status=" + std::to_string((int)st);
      return r;
    }
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  double sum = 0.0;
  r.minMs = times[0];
  r.maxMs = times[0];
  for (double t : times) {
    sum += t;
    if (t < r.minMs) r.minMs = t;
    if (t > r.maxMs) r.maxMs = t;
  }
  r.mean = sum / (double)times.size();
  r.p50 = Pctl(times, 0.50);
  r.p95 = Pctl(times, 0.95);

  // ---- I/O-faithful pass (A18 section 8) ----------------------------------
  // The pipeline's per-frame cost is memcpy-in + predict + copy-out; the pure
  // p50 above times predict alone. Running the same session with the marshaling
  // inside the timed region quantifies how much of the pipeline-vs-bench gap
  // is I/O rather than inference. Byte-level copies match MsRunMulti's fp32
  // path exactly (our models are fp32 in / fp32 out).
  {
    auto elemSize = [](OH_AI_DataType dt) -> size_t {
      switch (dt) {
        case OH_AI_DATATYPE_NUMBERTYPE_FLOAT32: return 4;
        case OH_AI_DATATYPE_NUMBERTYPE_FLOAT16: return 2;
        case OH_AI_DATATYPE_NUMBERTYPE_INT8: return 1;
        default: return 4;
      }
    };
    std::vector<uint8_t> inSnap;
    std::vector<size_t> outBytes(outs.handle_num, 0);
    size_t outTotal = 0;
    void* inBuf = OH_AI_TensorGetMutableData(ins.handle_list[0]);
    if (inBuf != nullptr) {
      const size_t n = (size_t)OH_AI_TensorGetElementNum(ins.handle_list[0]) *
                       elemSize(OH_AI_TensorGetDataType(ins.handle_list[0]));
      const uint8_t* p = reinterpret_cast<const uint8_t*>(inBuf);
      inSnap.assign(p, p + n);
    }
    for (size_t k = 0; k < outs.handle_num; k++) {
      outBytes[k] = (size_t)OH_AI_TensorGetElementNum(outs.handle_list[k]) *
                    elemSize(OH_AI_TensorGetDataType(outs.handle_list[k]));
      outTotal += outBytes[k];
    }
    std::vector<uint8_t> outSnap(outTotal);
    std::vector<double> ioTimes;
    ioTimes.reserve(repeat);
    for (int i = 0; i < repeat; i++) {
      const auto a = std::chrono::steady_clock::now();
      if (!inSnap.empty() && inBuf != nullptr) {
        std::memcpy(inBuf, inSnap.data(), inSnap.size());
      }
      OH_AI_Status st = OH_AI_ModelPredict(s->model, ins, &outs, nullptr, nullptr);
      size_t off = 0;
      for (size_t k = 0; k < outs.handle_num && st == OH_AI_STATUS_SUCCESS; k++) {
        const void* od = OH_AI_TensorGetData(outs.handle_list[k]);
        if (od != nullptr) {
          std::memcpy(outSnap.data() + off, od, outBytes[k]);
        }
        off += outBytes[k];
      }
      const auto b = std::chrono::steady_clock::now();
      if (st != OH_AI_STATUS_SUCCESS) {
        r.error = "io pass predict failed at run " + std::to_string(i);
        return r;
      }
      ioTimes.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    double ioSum = 0.0;
    for (double t : ioTimes) ioSum += t;
    r.meanIo = ioSum / (double)ioTimes.size();
    r.p50Io = Pctl(ioTimes, 0.50);
  }

  OH_AI_TensorHandle ot = outs.handle_list[0];
  r.outputDtype = (int)OH_AI_TensorGetDataType(ot);
  r.outputElems = (size_t)OH_AI_TensorGetElementNum(ot);
  const void* od = OH_AI_TensorGetData(ot);
  if (od != nullptr && r.outputElems > 0) {
    SumL2(od, r.outputElems, OH_AI_TensorGetDataType(ot), r.checksum, r.maxAbs);
    r.checksumAsFp16 = SumL2AsFp16(od, r.outputElems);
  }
  r.ok = true;
  return r;
}
