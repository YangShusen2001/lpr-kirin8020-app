/**
 * NAPI surface for the native LPR engine.
 *
 *   —— 同步（兼容保留，内部同样走专用推理线程）——
 *   loadModel(name, bytes: ArrayBuffer, backend: string) -> string  (flat kv)
 *   run(id: number, input: Float32Array)                 -> Float32Array | null
 *   bench(id: number, warmup, repeat)                    -> string  (flat kv)
 *   pipeline(rgba, width, height, detId, recId, clsId, mode) -> string (flat kv)
 *   ncnnLoad(param, bin, useVulkan) / ncnnRun(rgba,w,h,repeat) / ncnnRelease()
 *   listNnrtDevices() / vulkanProbe() / appendLine(path, line)
 *
 *   —— 异步（**推荐**：ArkTS 侧一律用这组）——
 *   loadModelAsync(name, bytes, backend)  -> Promise<string>
 *   pipelineAsync(rgba,w,h,detId,recId,clsId,mode) -> Promise<string>
 *   benchAsync(id, warmup, repeat)        -> Promise<string>
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 线程契约（2026-09-18 重写，这不是风格问题，是能不能活下去的问题）
 *
 * 旧实现把每个 NAPI 调用都**同步执行在 ArkTS 主线程**上。实测后果：整轮后端矩阵
 * 连续占用 uv loop **8.887 s** → 系统 watchdog 判 THREAD_BLOCK_3S/6S → **SIGKILL**
 * （`XCollie: uvLoopTask Duration Time: 8887 ms`，见 _evidence/A6-*.md）。
 *
 * 现在的三条纪律：
 *   1. **所有 MindSpore Lite / ncnn 调用只跑在一根专用推理线程上**（InferenceRunner）。
 *      libuv 线程池里跑也能"不阻塞 JS"，但 4 根线程轮流碰同一份 NNRT 会话，
 *      线程亲和性没有任何文档保证；串行执行也更贴近硬件现实（只有一块 NPU）。
 *   2. **会话按 (name|backend) 去重并封顶**。旧实现每次 loadModel 都 push_back，
 *      而 MsSession 因 NNRT 析构 crash 而永不释放（见 ms_engine.h 的生命周期说明），
 *      跑一轮矩阵就常驻十几份会话，把内存和设备侧模型槽位一起吃光。
 *   3. **不阻塞 JS 线程**：`*Async` 走 napi_async_work（execute 在池线程上等待推理线程，
 *      完成回调在 JS 线程 resolve Promise），UI 线程全程可响应。
 *
 * `backend` is "auto" | "nnrt" | "gpu" | "kirin" | "cpu"; see ms_engine.h.
 *
 * Deliberately small on the JS side: the whole compute pipeline lives in C++, so JS
 * only ships the image in and a few KB of flat kv out.
 */
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hilog/log.h>
#include <napi/native_api.h>

#include "ms_engine.h"
#include "lpr_pipeline.h"
#include "vulkan_probe.h"
#include "ncnn_engine.h"
#include "nnrt_probe.h"
#include <gpu.h>

#define LPR_TAG "LprNative"
#define LOGI(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, LPR_TAG, __VA_ARGS__)
#define LOGE(...) OH_LOG_Print(LOG_APP, LOG_ERROR, 0xD001, LPR_TAG, __VA_ARGS__)

// ============================================================ 会话登记表与推理线程

/**
 * 会话登记表（纪律 2）。
 *
 * key = "<rawfile 名>|<请求后端>"，同一个 key 只建一次，命中直接复用并回 `cached=1`。
 * id 就是下标，且**永不回收** —— ArkTS 侧可能还攥着旧 id；宁可按上限拒绝，
 * 也不做会静默改变语义的 id 重用。
 */
struct SessionEntry {
  std::string key;
  MsSession* s;
};
static std::vector<SessionEntry> g_sessions;
static std::map<std::string, int> g_sessionByKey;
static std::mutex g_regMutex;
/**
 * 常驻会话上限。到顶就诚实失败（ok=0;error=session cap ...），不继续漏。
 *
 * 2026-09-20 由 20 抬到 64：E2E_BACKENDS 扩到 6 档 × 4 模型 = 24 个唯一键，
 * 再加 threadSweep 的 3 模型 × 5 线程档 = 15 个，20 会静默截断矩阵尾部
 * （实测 cls-fp32 的 {nnrt_fp32, nnrt_fp16, gpu, kirin} 4 组被 cap 掉，
 * 日志有 `err=session cap 20 reached`，A17 §1）。64 的显存代价可忽略
 * （会话键 = (模型文件|后端)，同一键命中缓存；每模型编译产物 MB 级）。
 */
static const size_t kMaxSessions = 64;

/**
 * 专用推理线程（纪律 1）。任务串行执行，答案通过 doneCv_ 交回。
 * 生命周期：模块加载时 start，进程退出时随进程回收（不提供 stop，避免析构期竞态）。
 */
class InferenceRunner {
 public:
  void Start() {
    if (th_.joinable()) {
      return;
    }
    th_ = std::thread([this] { Loop(); });
  }
  /** 提交并在当前线程等待完成（同步 NAPI 用；JS 线程会被阻塞，仅兼容路径用）。 */
  void Submit(const std::function<void()>& fn) {
    std::unique_lock<std::mutex> lk(m_);
    task_ = fn;
    has_ = true;
    cv_.notify_one();
    doneCv_.wait(lk, [this] { return !has_; });
  }

 private:
  void Loop() {
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return has_; });
        fn = task_;
      }
      fn();  // 不持锁执行：推理期间不挡新任务入队
      {
        std::unique_lock<std::mutex> lk(m_);
        has_ = false;
        doneCv_.notify_all();
      }
    }
  }
  std::thread th_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable doneCv_;
  std::function<void()> task_;
  bool has_ = false;
};
static InferenceRunner g_runner;

/** 一次推理任务的全部输入/输出。异步与同步共用同一份定义。 */
enum class JobKind {
  kLoad,
  kRun,
  kPipeline,
  kBench,
  kNcnnLoad,
  kNcnnRun,
  kNcnnSlotLoad,
  kNnrtProbe,
  kNnrtTryModel,
  kCameraFrame
};

struct AsyncJob {
  napi_async_work work = nullptr;
  napi_deferred deferred = nullptr;
  JobKind kind = JobKind::kLoad;
  /** 参数校验就已经失败时预置结果：RunJob 直接返回，不再干活。 */
  bool prefilled = false;

  // kLoad / kNcnnLoad
  std::string name;
  std::string backend;
  std::vector<char> bytes;
  std::vector<char> param;
  std::vector<char> bin;
  bool useVulkan = false;

  // kRun
  std::vector<float> input;

  // kPipeline / kNcnnRun
  std::vector<uint8_t> rgba;
  int w = 0;
  int h = 0;
  int detId = -1;
  int recId = -1;
  int clsId = -1;
  int mode = 0;
  /** 识别 / 分类的 ncnn 槽位（-1 = 用 MS 会话）；只对 kPipeline 有意义。 */
  int recSlot = -1;
  int clsSlot = -1;
  /** kSlotLoad 的槽位号。 */
  int slot = 0;
  /** kNnrtTryModel：模型字节（ArkTS 从 rawfile 读入的 .ms/.om）与目标设备下标。 */
  std::vector<uint8_t> modelBytes;
  int deviceIndex = -1;

  // kBench
  int warmup = 0;
  int repeat = 0;

  // kCameraFrame：NV21 原始帧 → RGBA(+旋转) → 流水线，一次调用做完
  int stride = 0;
  int rotation = 0;
  /** 分段计时（毫秒）：转换 / 推理。 */
  double convMs = 0;
  double inferMs = 0;

  // outputs
  std::string kv;
  std::vector<float> output;
  bool runOk = false;
};

/** 定义在文件后段（依赖 Num/Scrub/KvSanitize 等辅助函数）。 */
static void RunJob(AsyncJob* job);

/** 在专用线程上跑一个任务并返回 kv（同步入口用）。 */
static std::string RunJobSync(AsyncJob& job) {
  g_runner.Submit([&job] { RunJob(&job); });
  return job.kv;
}

// ============================================================ 通用辅助

static std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

static std::string ShapeToJson(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); i++) {
    if (i > 0) {
      out += ",";
    }
    out += std::to_string((long long)s[i]);
  }
  out += "]";
  return out;
}

static const char* FormatName(OH_AI_Format f) {
  switch (f) {
    case OH_AI_FORMAT_NCHW: return "NCHW";
    case OH_AI_FORMAT_NHWC: return "NHWC";
    case OH_AI_FORMAT_NHWC4: return "NHWC4";
    case OH_AI_FORMAT_NC4HW4: return "NC4HW4";
    case OH_AI_FORMAT_NC: return "NC";
    case OH_AI_FORMAT_NC4: return "NC4";
    case OH_AI_FORMAT_HW: return "HW";
    case OH_AI_FORMAT_HW4: return "HW4";
    default: return "OTHER";
  }
}

static napi_value MakeString(napi_env env, const std::string& s) {
  napi_value v = nullptr;
  napi_create_string_utf8(env, s.c_str(), s.size(), &v);
  return v;
}

/**
 * ArkTS forbids `any`/`unknown`, so `JSON.parse()` on the JS side will not compile
 * under the strict ArkTS linter. Every NAPI call therefore returns a flat
 * `key=value;key=value;` string instead — trivially parseable with split().
 * `;` and `=` inside a value are folded to `,` so the grammar stays unambiguous.
 */
static std::string KvSanitize(const std::string& in) {
  std::string out = in;
  for (char& c : out) {
    if (c == ';' || c == '=' || c == '\n' || c == '\r') {
      c = ',';
    }
  }
  return out;
}

static napi_value MakeNull(napi_env env) {
  napi_value v = nullptr;
  napi_get_null(env, &v);
  return v;
}

static std::string Num(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.4f", v);
  return std::string(buf);
}

/** Steady-clock milliseconds; lpr_pipeline.cpp keeps its own copy in an anon ns. */
static double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/**
 * Plate codes and character lists are re-emitted into a flat `;`-separated string,
 * so any delimiter appearing inside a value would corrupt the grammar. The charset
 * cannot produce them, but a wrong-session mix-up could, hence the scrub.
 */
static std::string Scrub(const std::string& in) {
  std::string out = in;
  for (char& c : out) {
    if (c == ';' || c == '=' || c == ',' || c == '|') {
      c = '_';
    }
  }
  return out;
}

static bool ReadStringArg(napi_env env, napi_value v, char* buf, size_t cap, std::string& out) {
  size_t len = 0;
  if (napi_get_value_string_utf8(env, v, buf, cap - 1, &len) != napi_ok) {
    return false;
  }
  out.assign(buf, len);
  return true;
}

static bool ReadArrayBufferArg(napi_env env, napi_value v, std::vector<char>& out) {
  void* data = nullptr;
  size_t len = 0;
  if (napi_get_arraybuffer_info(env, v, &data, &len) != napi_ok || data == nullptr || len == 0) {
    return false;
  }
  out.assign(reinterpret_cast<char*>(data), reinterpret_cast<char*>(data) + len);
  return true;
}

/** 同一份内存的 uint8 视图（RGBA 图）。不做 reinterpret_cast 硬转容器类型。 */
static bool ReadArrayBufferArgU8(napi_env env, napi_value v, std::vector<uint8_t>& out) {
  void* data = nullptr;
  size_t len = 0;
  if (napi_get_arraybuffer_info(env, v, &data, &len) != napi_ok || data == nullptr || len == 0) {
    return false;
  }
  out.assign(reinterpret_cast<uint8_t*>(data), reinterpret_cast<uint8_t*>(data) + len);
  return true;
}

// ============================================================ 同步入口（兼容路径）

/** loadModel：去重 + 封顶，实际构建在推理线程上。 */
static napi_value LoadModel(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 2) {
    return MakeString(env, "ok=0;error=loadModel needs (name, bytes, backend)");
  }
  AsyncJob job;
  job.kind = JobKind::kLoad;
  char nameBuf[128] = {0};
  if (!ReadStringArg(env, args[0], nameBuf, sizeof(nameBuf), job.name)) {
    return MakeString(env, "ok=0;error=name must be a string");
  }
  if (!ReadArrayBufferArg(env, args[1], job.bytes)) {
    return MakeString(env, "ok=0;error=model ArrayBuffer is empty");
  }
  job.backend = "auto";
  if (argc >= 3) {
    char beBuf[64] = {0};
    ReadStringArg(env, args[2], beBuf, sizeof(beBuf), job.backend);
    if (job.backend.empty()) {
      job.backend = "auto";
    }
  }
  const std::string kv = RunJobSync(job);
  LOGI("loadModel -> %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value RunModel(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2] = {nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 2) {
    LOGE("run needs (id, Float32Array)");
    return MakeNull(env);
  }
  int32_t id = -1;
  napi_get_value_int32(env, args[0], &id);

  napi_typedarray_type type;
  size_t length = 0;
  void* data = nullptr;
  napi_value ab = nullptr;
  size_t byteOffset = 0;
  napi_status st = napi_get_typedarray_info(env, args[1], &type, &length, &data, &ab, &byteOffset);
  if (st != napi_ok || data == nullptr || type != napi_float32_array) {
    LOGE("run: input must be a Float32Array (status=%{public}d type=%{public}d)", (int)st, (int)type);
    return MakeNull(env);
  }

  AsyncJob job;
  job.kind = JobKind::kRun;
  job.detId = id;
  job.input.assign(reinterpret_cast<float*>(data), reinterpret_cast<float*>(data) + length);
  RunJobSync(job);
  if (!job.runOk) {
    return MakeNull(env);
  }

  void* outData = nullptr;
  napi_value outAb = nullptr;
  napi_create_arraybuffer(env, job.output.size() * sizeof(float), &outData, &outAb);
  if (outData != nullptr && !job.output.empty()) {
    std::memcpy(outData, job.output.data(), job.output.size() * sizeof(float));
  }
  napi_value outArr = nullptr;
  napi_create_typedarray(env, napi_float32_array, job.output.size(), outAb, 0, &outArr);
  return outArr;
}

static napi_value BenchModel(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob job;
  job.kind = JobKind::kBench;
  job.warmup = 10;
  job.repeat = 50;
  if (argc >= 1) napi_get_value_int32(env, args[0], &job.detId);
  if (argc >= 2) napi_get_value_int32(env, args[1], &job.warmup);
  if (argc >= 3) napi_get_value_int32(env, args[2], &job.repeat);
  const std::string kv = RunJobSync(job);
  LOGI("bench -> %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value ListNnrt(napi_env env, napi_callback_info info) {
  std::vector<std::string> names = MsNnrtCandidates();
  std::string json = "[";
  for (size_t i = 0; i < names.size(); i++) {
    if (i > 0) {
      json += ",";
    }
    json += "\"" + JsonEscape(names[i]) + "\"";
  }
  json += "]";
  return MakeString(env, json);
}

/**
 * pipeline(rgba: ArrayBuffer, width, height, detId, recId, clsId, mode) -> flat kv
 *
 * Runs the whole HyperLPR3 chain in native code: the sessions may sit on different
 * backends (detector on CPU because the NPU rejects it, recogniser on the NPU), and
 * that combination is the entire point of having the pipeline here instead of in
 * the WebView — onnxruntime-web cannot reach the NPU at all.
 *
 * Layout of the returned string (values never contain ';' or '='):
 *   ok=1;count=N;totalMs=...;err=
 *   p0=<code>,<detScore>,<recConf>,<layer>,<x1|x2...>,<cropH|cropW>,
 *      <cls0|cls1|cls2>,<char|char|...>,<prob|prob|...>,
 *      <tDet|tRect|tRec|tCls>,<cropSum>
 *   p1=...
 */
static napi_value PipelineRun(napi_env env, napi_callback_info info) {
  size_t argc = 9;
  napi_value args[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob job;
  job.kind = JobKind::kPipeline;
  if (argc < 6) {
    return MakeString(env, "ok=0;count=0;totalMs=0;error=pipeline needs "
                           "(rgba, width, height, detId, recId, clsId)");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job.rgba)) {
    return MakeString(env, "ok=0;count=0;totalMs=0;error=rgba ArrayBuffer is empty");
  }
  napi_get_value_int32(env, args[1], &job.w);
  napi_get_value_int32(env, args[2], &job.h);
  napi_get_value_int32(env, args[3], &job.detId);
  napi_get_value_int32(env, args[4], &job.recId);
  napi_get_value_int32(env, args[5], &job.clsId);
  if (argc >= 7 && (napi_get_value_int32(env, args[6], &job.mode) != napi_ok ||
                    job.mode < 0 || job.mode > 2)) {
    return MakeString(env, "ok=0;error=invalid detector mode");
  }
  // 可选 8/9 参：识别 / 分类的 ncnn 槽位（-1 = 用 MS 会话）
  if (argc >= 8) napi_get_value_int32(env, args[7], &job.recSlot);
  if (argc >= 9) napi_get_value_int32(env, args[8], &job.clsSlot);
  const std::string kv = RunJobSync(job);
  LOGI("pipeline -> %{public}s", kv.c_str());
  return MakeString(env, kv);
}

/**
 * GPU 终审探针（ADR-008）：dlopen Vulkan loader → 枚举物理设备 → 找 COMPUTE 队列。
 * 返回 JSON；见 vulkan_probe.h。这里不套 KvSanitize —— JSON 是给 PC 侧看的，
 * ArkTS 侧只做切块打日志，不解析。
 */
static napi_value VulkanProbe(napi_env env, napi_callback_info info) {
  return MakeString(env, VulkanProbeJson());
}

static napi_value NcnnLoadFn(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob job;
  job.kind = JobKind::kNcnnLoad;
  if (argc < 2) {
    return MakeString(env, "ok=0;error=ncnnLoad needs (param, bin)");
  }
  if (!ReadArrayBufferArg(env, args[0], job.param) || !ReadArrayBufferArg(env, args[1], job.bin)) {
    return MakeString(env, "ok=0;error=empty param/bin buffer");
  }
  if (argc >= 3 && napi_get_value_bool(env, args[2], &job.useVulkan) != napi_ok) {
    return MakeString(env, "ok=0;error=useVulkan must be boolean");
  }
  const std::string kv = RunJobSync(job);
  LOGI("ncnnLoad -> %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value NcnnRunFn(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4] = {nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob job;
  job.kind = JobKind::kNcnnRun;
  if (argc < 4) {
    return MakeString(env, "ok=0;error=ncnnRun needs (rgba, w, h, repeat)");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job.rgba)) {
    return MakeString(env, "ok=0;error=rgba buffer empty");
  }
  napi_get_value_int32(env, args[1], &job.w);
  napi_get_value_int32(env, args[2], &job.h);
  job.repeat = 5;
  napi_get_value_int32(env, args[3], &job.repeat);
  const std::string kv = RunJobSync(job);
  LOGI("ncnnRun -> %{public}s", kv.c_str());
  return MakeString(env, kv);
}

static napi_value NcnnReleaseFn(napi_env env, napi_callback_info info) {
  g_runner.Submit([] { NcnnRelease(); });
  return MakeString(env, "ok=1");
}

/**
 * 追加一行到文件（矩阵落盘续跑用）。
 *
 * 为什么要有：整轮矩阵一旦被 watchdog 杀掉，hilog 环形缓冲里的结果只能靠运气留存。
 * ArkTS 侧每跑完一个组合就 append 一行，被杀也能知道跑到哪、拿到了什么。
 * 纯文件 IO，不碰推理线程。
 */
static napi_value AppendLine(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2] = {nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 2) {
    return MakeString(env, "ok=0;error=appendLine needs (path, line)");
  }
  char pathBuf[512] = {0};
  char lineBuf[4096] = {0};
  std::string path;
  std::string line;
  if (!ReadStringArg(env, args[0], pathBuf, sizeof(pathBuf), path)) {
    return MakeString(env, "ok=0;error=path must be a string");
  }
  if (!ReadStringArg(env, args[1], lineBuf, sizeof(lineBuf), line)) {
    return MakeString(env, "ok=0;error=line must be a string");
  }
  std::ofstream f(path.c_str(), std::ios::app);
  if (!f) {
    return MakeString(env, "ok=0;error=cannot open " + KvSanitize(path));
  }
  f << line << "\n";
  f.flush();
  return MakeString(env, "ok=1;bytes=" + std::to_string(line.size()));
}

// ============================================================ 任务执行（推理线程）

static std::string KvOfLoad(const MsSession* s, const std::string& name, int id, bool cached) {
  return "ok=1;id=" + std::to_string(id) +
         ";cached=" + std::string(cached ? "1" : "0") +
         ";name=" + KvSanitize(name) +
         ";backend=" + KvSanitize(s->backend) +
         ";requested=" + KvSanitize(s->requested) +
         ";fallbackFrom=" + KvSanitize(s->fallbackFrom) +
         ";attemptLog=" + KvSanitize(s->attemptLog) +
         ";inputName=" + KvSanitize(s->inputName) +
         ";outputName=" + KvSanitize(s->outputName) +
         ";inputShape=" + ShapeToJson(s->inputShape) +
         ";outputShape=" + ShapeToJson(s->outputShape) +
         ";inputFormat=" + std::string(FormatName(s->inputFormat)) +
         ";outputFormat=" + std::string(FormatName(s->outputFormat)) +
         ";inputElems=" + std::to_string((long long)s->inputElems) +
         ";outputElems=" + std::to_string((long long)s->outputElems) +
         ";error=";
}

static void RunJob(AsyncJob* job) {
  if (job->prefilled) {
    return;
  }
  switch (job->kind) {
    // ---------------------------------------------------------------- load
    case JobKind::kLoad: {
      const std::string key = job->name + "|" + job->backend;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        auto it = g_sessionByKey.find(key);
        if (it != g_sessionByKey.end()) {
          job->kv = KvOfLoad(g_sessions[it->second].s, job->name, it->second, true);
          LOGI("loadModel cached: %{public}s", job->kv.c_str());
          return;
        }
        if (g_sessions.size() >= kMaxSessions) {
          job->kv = "ok=0;backend=" + KvSanitize(job->backend) +
                    ";error=session cap " + std::to_string(kMaxSessions) + " reached";
          LOGE("loadModel cap: %{public}s", job->kv.c_str());
          return;
        }
      }
      std::string err;
      MsSession* s = MsLoad(job->bytes, job->backend, err);
      if (s == nullptr) {
        LOGE("loadModel(%{public}s) failed: %{public}s", job->name.c_str(), err.c_str());
        job->kv = "ok=0;backend=" + KvSanitize(job->backend) + ";error=" + KvSanitize(err);
        return;
      }
      std::lock_guard<std::mutex> lk(g_regMutex);
      const int id = (int)g_sessions.size();
      g_sessions.push_back({key, s});
      g_sessionByKey[key] = id;
      job->kv = KvOfLoad(s, job->name, id, false);
      LOGI("loadModel ok: %{public}s", job->kv.c_str());
      return;
    }

    // ---------------------------------------------------------------- run
    case JobKind::kRun: {
      MsSession* s = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        if (job->detId < 0 || job->detId >= (int)g_sessions.size()) {
          LOGE("run: bad session id %{public}d", job->detId);
          return;
        }
        s = g_sessions[job->detId].s;
      }
      if (job->input.size() != s->inputElems) {
        LOGE("run: input length %{public}zu != expected %{public}zu",
             job->input.size(), s->inputElems);
        return;
      }
      std::string err;
      if (!MsRun(s, job->input.data(), job->output, err)) {
        LOGE("run failed: %{public}s", err.c_str());
        return;
      }
      job->runOk = true;
      return;
    }

    // ---------------------------------------------------------------- pipeline
    case JobKind::kPipeline: {
      LprSessions s;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        const int ids[3] = {job->detId, job->recId, job->clsId};
        for (int i = 0; i < 3; i++) {
          if (ids[i] < 0 || ids[i] >= (int)g_sessions.size()) {
            job->kv = "ok=0;count=0;totalMs=0;error=bad session id";
            return;
          }
        }
        s.det = g_sessions[job->detId].s;
        s.rec = g_sessions[job->recId].s;
        s.cls = g_sessions[job->clsId].s;
      }
      s.detNcnn = job->mode != 0;
      s.detVulkan = job->mode == 2;
      s.recSlot = job->recSlot;
      s.clsSlot = job->clsSlot;
      LOGI("PIPE BACKENDS mode=%{public}d det=%{public}s rec=%{public}s cls=%{public}s",
           job->mode, s.detNcnn ? "ncnn" : s.det->backend.c_str(),
           s.recSlot >= 0 ? "ncnn-slot" : s.rec->backend.c_str(),
           s.clsSlot >= 0 ? "ncnn-slot" : s.cls->backend.c_str());

      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = std::move(job->rgba);
      if (!img.Valid()) {
        job->kv = "ok=0;count=0;totalMs=0;error=rgba size != w*h*4";
        return;
      }

      std::vector<PlateResult> plates;
      std::string err;
      const double t0 = NowMs();
      if (!LprRunPipeline(img, s, plates, err)) {
        LOGE("pipeline failed: %{public}s", err.c_str());
        job->kv = "ok=0;count=0;totalMs=0;error=" + KvSanitize(err);
        return;
      }
      const double totalMs = NowMs() - t0;

      // 每个字段都是 `key=value;` —— `error=` 后面那个分号是承重的：少了它第一条车牌
      // 记录会粘进 error 的值（`error=p0=...`），ArkTS 侧就读成 undefined。
      std::string kv = "ok=1;count=" + std::to_string(plates.size()) +
                       ";totalMs=" + Num(totalMs) + ";error=;";
      for (size_t i = 0; i < plates.size(); i++) {
        const PlateResult& p = plates[i];
        std::string v = Scrub(p.code) + "," + Num(p.detScore) + "," + Num(p.recConf) + "," +
                        std::to_string(p.layer) + "," +
                        std::to_string(p.rect[0]) + "|" + std::to_string(p.rect[1]) + "|" +
                        std::to_string(p.rect[2]) + "|" + std::to_string(p.rect[3]) + "," +
                        std::to_string(p.cropH) + "|" + std::to_string(p.cropW) + "," +
                        Num(p.cls[0]) + "|" + Num(p.cls[1]) + "|" + Num(p.cls[2]) + ",";
        for (size_t k = 0; k < p.chars.size(); k++) {
          if (k > 0) {
            v += "|";
          }
          v += Scrub(p.chars[k]);
        }
        v += ",";
        for (size_t k = 0; k < p.charProbs.size(); k++) {
          if (k > 0) {
            v += "|";
          }
          v += Num(p.charProbs[k]);
        }
        v += "," + Num(p.tDetectMs) + "|" + Num(p.tLetterboxMs) + "|" + Num(p.tEncodeInferMs) +
             "|" + Num(p.tDecodeNmsMs) + "|" + Num(p.tRectifyMs) + "|" + Num(p.tRecogMs) + "|" +
             Num(p.tClsMs) + "|" + Num(p.tPackMs) + "|" + Num(p.tInferMs) + "," +
             std::to_string(p.cropSum);
        kv += "p" + std::to_string(i) + "=" + v + ";";
      }
      job->kv = kv;
      LOGI("pipeline: %{public}s", kv.c_str());
      return;
    }

    // ------------------------------------------------- 相机帧：NV21 -> RGBA -> 流水线
    // 一次 native 调用做完「格式转换 + 旋转 + 推理」。
    //
    // 为什么合并：分开做时，NV21→RGBA 要在 ArkTS 侧建 PixelMap、rotate()、
    // readPixelsToBuffer 三步，实测 21-38 ms（NPU 档整个推理才 10 ms）。
    // 合并后这段只是一趟 C++ 循环，且省掉两次跨语言拷贝。
    case JobKind::kCameraFrame: {
      LprSessions s;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        const int ids[3] = {job->detId, job->recId, job->clsId};
        for (int i = 0; i < 3; i++) {
          if (ids[i] < 0 || ids[i] >= (int)g_sessions.size()) {
            job->kv = "ok=0;count=0;totalMs=0;convMs=0;inferMs=0;error=bad session id";
            return;
          }
        }
        s.det = g_sessions[job->detId].s;
        s.rec = g_sessions[job->recId].s;
        s.cls = g_sessions[job->clsId].s;
      }
      s.detNcnn = job->mode != 0;
      s.detVulkan = job->mode == 2;
      s.recSlot = job->recSlot;
      s.clsSlot = job->clsSlot;

      const double tConv0 = NowMs();
      RgbaImage img;
      if (!LprNv21ToRgba(job->rgba.data(), job->rgba.size(), job->w, job->h, job->stride,
                         job->rotation, img)) {
        job->kv = "ok=0;count=0;totalMs=0;convMs=0;inferMs=0;error=nv21->rgba failed";
        return;
      }
      const double convMs = NowMs() - tConv0;
      // RGBA 校验和（仅 RGB 三通道）—— 用来证明 native 的 NV21→RGBA 与系统解码器
      // 等价。没有这个，"换掉了取帧路径"就只是换了、而不是验证过。
      long long rgbaSum = 0;
      for (size_t i = 0; i < img.data.size(); i++) {
        if (i % 4 != 3) {
          rgbaSum += img.data[i];
        }
      }

      std::vector<PlateResult> plates;
      std::string err;
      const double t0 = NowMs();
      if (!LprRunPipeline(img, s, plates, err)) {
        LOGE("cameraFrame pipeline failed: %{public}s", err.c_str());
        job->kv = "ok=0;count=0;totalMs=0;convMs=" + Num(convMs) +
                  ";inferMs=0;error=" + KvSanitize(err);
        return;
      }
      const double totalMs = NowMs() - t0;

      std::string kv = "ok=1;count=" + std::to_string(plates.size()) +
                       ";totalMs=" + Num(totalMs) +
                       ";convMs=" + Num(convMs) +
                       ";inferMs=" + Num(totalMs) +
                       ";w=" + std::to_string(img.width) +
                       ";h=" + std::to_string(img.height) +
                       ";rgbaSum=" + std::to_string(rgbaSum) + ";error=;";
      for (size_t i = 0; i < plates.size(); i++) {
        const PlateResult& p = plates[i];
        std::string v = Scrub(p.code) + "," + Num(p.detScore) + "," + Num(p.recConf) + "," +
                        std::to_string(p.layer) + "," +
                        std::to_string(p.rect[0]) + "|" + std::to_string(p.rect[1]) + "|" +
                        std::to_string(p.rect[2]) + "|" + std::to_string(p.rect[3]) + "," +
                        std::to_string(p.cropH) + "|" + std::to_string(p.cropW) + "," +
                        Num(p.cls[0]) + "|" + Num(p.cls[1]) + "|" + Num(p.cls[2]) + ",";
        for (size_t k = 0; k < p.chars.size(); k++) {
          if (k > 0) {
            v += "|";
          }
          v += Scrub(p.chars[k]);
        }
        v += ",";
        for (size_t k = 0; k < p.charProbs.size(); k++) {
          if (k > 0) {
            v += "|";
          }
          v += Num(p.charProbs[k]);
        }
        v += "," + Num(p.tDetectMs) + "|" + Num(p.tLetterboxMs) + "|" + Num(p.tEncodeInferMs) +
             "|" + Num(p.tDecodeNmsMs) + "|" + Num(p.tRectifyMs) + "|" + Num(p.tRecogMs) + "|" +
             Num(p.tClsMs) + "|" + Num(p.tPackMs) + "|" + Num(p.tInferMs) + "," +
             std::to_string(p.cropSum);
        kv += "p" + std::to_string(i) + "=" + v + ";";
      }
      job->kv = kv;
      job->convMs = convMs;
      job->inferMs = totalMs;
      return;
    }

    // ---------------------------------------------------------------- bench
    case JobKind::kBench: {
      MsSession* s = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        if (job->detId < 0 || job->detId >= (int)g_sessions.size()) {
          job->kv = "ok=0;error=bad session id";
          return;
        }
        s = g_sessions[job->detId].s;
      }
      MsBench b = MsBenchRun(s, job->warmup, job->repeat);
      job->kv = "ok=" + std::string(b.ok ? "1" : "0") +
                ";backend=" + KvSanitize(b.backend) +
                ";warmup=" + std::to_string(b.warmup) +
                ";repeat=" + std::to_string(b.repeat) +
                ";meanMs=" + Num(b.mean) +
                ";p50Ms=" + Num(b.p50) +
                ";p95Ms=" + Num(b.p95) +
                ";minMs=" + Num(b.minMs) +
                ";maxMs=" + Num(b.maxMs) +
                ";p50IoMs=" + Num(b.p50Io) +
                ";meanIoMs=" + Num(b.meanIo) +
                ";checksum=" + Num(b.checksum) +
                ";checksumAsFp16=" + Num(b.checksumAsFp16) +
                ";maxAbs=" + Num(b.maxAbs) +
                ";outputDtype=" + std::to_string(b.outputDtype) +
                ";outputElems=" + std::to_string((long long)b.outputElems) +
                ";error=" + KvSanitize(b.error);
      LOGI("bench: %{public}s", job->kv.c_str());
      return;
    }

    // ---------------------------------------------------------------- ncnn
    case JobKind::kNcnnLoad: {
      std::string err;
      if (!NcnnLoad(job->param, job->bin, job->useVulkan, err)) {
        LOGE("ncnnLoad failed: %{public}s", err.c_str());
        job->kv = "ok=0;error=" + KvSanitize(err);
        return;
      }
      job->kv = NcnnInfo();
      return;
    }
    case JobKind::kNcnnRun: {
      job->kv = NcnnRunRgba(job->rgba.data(), job->w, job->h, 1, job->repeat);
      return;
    }
    case JobKind::kNcnnSlotLoad: {
      std::string err;
      if (!NcnnLoadSlot(job->slot, job->param, job->bin, job->useVulkan, err)) {
        LOGE("ncnnLoadSlot(%{public}d) failed: %{public}s", job->slot, err.c_str());
        job->kv = "ok=0;slot=" + std::to_string(job->slot) + ";error=" + KvSanitize(err);
        return;
      }
      job->kv = NcnnInfoSlot(job->slot);
      LOGI("ncnnLoadSlot: %{public}s", job->kv.c_str());
      return;
    }

    // -------------------------------------------------- NNRt / HiAI（CANN Kit）
    case JobKind::kNnrtProbe: {
      job->kv = NnrtProbeReport();
      LOGI("nnrtProbe: %{public}s", job->kv.c_str());
      return;
    }
    case JobKind::kNnrtTryModel: {
      job->kv = NnrtTryModelReport(job->modelBytes, job->deviceIndex);
      LOGI("nnrtTryModel: %{public}s", job->kv.c_str());
      return;
    }
  }
}

// ============================================================ 异步入口（推荐路径）

/** execute 回调跑在 libuv 池线程上：把活交给专用推理线程并等它做完（不碰 JS 线程）。 */
static void ExecuteOnPool(napi_env env, void* data) {
  AsyncJob* job = static_cast<AsyncJob*>(data);
  g_runner.Submit([job] { RunJob(job); });
}

/** complete 回调回到 JS 线程：resolve Promise 并回收任务。 */
static void CompleteOnJs(napi_env env, napi_status status, void* data) {
  AsyncJob* job = static_cast<AsyncJob*>(data);
  napi_value v = MakeString(env, job->kv);
  napi_resolve_deferred(env, job->deferred, v);
  napi_delete_async_work(env, job->work);
  delete job;
}

static napi_value QueueJob(napi_env env, AsyncJob* job, const char* resourceName) {
  napi_value promise = nullptr;
  napi_create_promise(env, &job->deferred, &promise);
  napi_value rn = nullptr;
  napi_create_string_utf8(env, resourceName, NAPI_AUTO_LENGTH, &rn);
  napi_create_async_work(env, nullptr, rn, ExecuteOnPool, CompleteOnJs, job, &job->work);
  napi_queue_async_work(env, job->work);
  return promise;
}

/** 参数错误时也要 resolve（返回 kv 串），绝不让 Promise 悬着。 */
static napi_value RejectedJob(napi_env env, AsyncJob* job, const std::string& kv,
                              const char* resourceName) {
  job->kv = kv;
  job->prefilled = true;
  return QueueJob(env, job, resourceName);
}

static napi_value LoadModelAsync(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kLoad;
  if (argc < 2) {
    return RejectedJob(env, job, "ok=0;error=loadModelAsync needs (name, bytes, backend)",
                       "lpr.loadModelAsync");
  }
  char nameBuf[128] = {0};
  if (!ReadStringArg(env, args[0], nameBuf, sizeof(nameBuf), job->name)) {
    return RejectedJob(env, job, "ok=0;error=name must be a string", "lpr.loadModelAsync");
  }
  if (!ReadArrayBufferArg(env, args[1], job->bytes)) {
    return RejectedJob(env, job, "ok=0;error=model ArrayBuffer is empty", "lpr.loadModelAsync");
  }
  job->backend = "auto";
  if (argc >= 3) {
    char beBuf[64] = {0};
    ReadStringArg(env, args[2], beBuf, sizeof(beBuf), job->backend);
    if (job->backend.empty()) {
      job->backend = "auto";
    }
  }
  return QueueJob(env, job, "lpr.loadModelAsync");
}

static napi_value PipelineAsync(napi_env env, napi_callback_info info) {
  size_t argc = 9;
  napi_value args[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kPipeline;
  if (argc < 6) {
    return RejectedJob(env, job, "ok=0;count=0;totalMs=0;error=pipelineAsync needs 6 args",
                       "lpr.pipelineAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job->rgba)) {
    return RejectedJob(env, job, "ok=0;count=0;totalMs=0;error=rgba ArrayBuffer is empty",
                       "lpr.pipelineAsync");
  }
  napi_get_value_int32(env, args[1], &job->w);
  napi_get_value_int32(env, args[2], &job->h);
  napi_get_value_int32(env, args[3], &job->detId);
  napi_get_value_int32(env, args[4], &job->recId);
  napi_get_value_int32(env, args[5], &job->clsId);
  if (argc >= 7) {
    if (napi_get_value_int32(env, args[6], &job->mode) != napi_ok ||
        job->mode < 0 || job->mode > 2) {
      return RejectedJob(env, job, "ok=0;error=invalid detector mode", "lpr.pipelineAsync");
    }
  }
  // 可选 8/9 参：识别 / 分类的 ncnn 槽位（-1 = 用 MS 会话）
  if (argc >= 8) napi_get_value_int32(env, args[7], &job->recSlot);
  if (argc >= 9) napi_get_value_int32(env, args[8], &job->clsSlot);
  return QueueJob(env, job, "lpr.pipelineAsync");
}

/**
 * 把一份 ncnn param/bin 加载到槽位（识别=1 / 分类=2，0 保留给检测旁路）。
 * 统一走推理线程：Vulkan 首次加载含 shader 编译，压在 JS 线程上会被 watchdog 杀。
 */
static napi_value NcnnLoadSlotAsync(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4] = {nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kNcnnSlotLoad;
  if (argc < 3) {
    return RejectedJob(env, job, "ok=0;error=ncnnLoadSlotAsync needs (slot, param, bin)",
                       "lpr.ncnnLoadSlotAsync");
  }
  if (napi_get_value_int32(env, args[0], &job->slot) != napi_ok) {
    return RejectedJob(env, job, "ok=0;error=slot must be an integer", "lpr.ncnnLoadSlotAsync");
  }
  if (!ReadArrayBufferArg(env, args[1], job->param) ||
      !ReadArrayBufferArg(env, args[2], job->bin)) {
    return RejectedJob(env, job, "ok=0;error=empty param/bin buffer", "lpr.ncnnLoadSlotAsync");
  }
  if (argc >= 4 && napi_get_value_bool(env, args[3], &job->useVulkan) != napi_ok) {
    return RejectedJob(env, job, "ok=0;error=useVulkan must be boolean",
                       "lpr.ncnnLoadSlotAsync");
  }
  return QueueJob(env, job, "lpr.ncnnLoadSlotAsync");
}

/**
 * 相机帧一步到位：NV21 原始 buffer →（native 里转 RGBA + 旋转）→ 完整流水线。
 *
 * 参数：nv21, width, height, stride, rotation, detId, recId, clsId,
 *       detMode?, recSlot?, clsSlot?
 *
 * 与 pipelineAsync 的区别只有输入格式与转换位置：那条路要调用方先把图转成 RGBA，
 * 而相机预览给的是 NV21 —— 在 ArkTS 侧转要 21-38 ms，搬到这里就是一趟 C++ 循环。
 */
static napi_value CameraFrameAsync(napi_env env, napi_callback_info info) {
  size_t argc = 11;
  napi_value args[11] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kCameraFrame;
  if (argc < 8) {
    return RejectedJob(env, job,
                       "ok=0;count=0;totalMs=0;convMs=0;inferMs=0;error=cameraFrameAsync needs "
                       "(nv21,w,h,stride,rotation,detId,recId,clsId)",
                       "lpr.cameraFrameAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job->rgba)) {
    return RejectedJob(env, job,
                       "ok=0;count=0;totalMs=0;convMs=0;inferMs=0;error=nv21 ArrayBuffer is empty",
                       "lpr.cameraFrameAsync");
  }
  napi_get_value_int32(env, args[1], &job->w);
  napi_get_value_int32(env, args[2], &job->h);
  napi_get_value_int32(env, args[3], &job->stride);
  napi_get_value_int32(env, args[4], &job->rotation);
  napi_get_value_int32(env, args[5], &job->detId);
  napi_get_value_int32(env, args[6], &job->recId);
  napi_get_value_int32(env, args[7], &job->clsId);
  if (argc >= 9) {
    if (napi_get_value_int32(env, args[8], &job->mode) != napi_ok ||
        job->mode < 0 || job->mode > 2) {
      return RejectedJob(env, job, "ok=0;error=invalid detector mode", "lpr.cameraFrameAsync");
    }
  }
  if (argc >= 10) napi_get_value_int32(env, args[9], &job->recSlot);
  if (argc >= 11) napi_get_value_int32(env, args[10], &job->clsSlot);
  return QueueJob(env, job, "lpr.cameraFrameAsync");
}

/**
 * NNRt / HiAI（CANN Kit）探针：问设备「库在不在、CANN 什么版本、NNRt 上有几张什么类型的设备」。
 * 纯枚举，不碰生产路径。
 */
static napi_value NnrtProbeAsync(napi_env env, napi_callback_info info) {
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kNnrtProbe;
  return QueueJob(env, job, "lpr.nnrtProbeAsync");
}

/**
 * 把一段模型字节喂给 NNRt 的 offline model 入口，并顺带问 HiAI 认不认这段字节。
 * 参数：(modelBytes: ArrayBuffer, deviceIndex?: number)
 */
static napi_value NnrtTryModelAsync(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2] = {nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kNnrtTryModel;
  if (argc < 1) {
    return RejectedJob(env, job, "ok=0;error=nnrtTryModelAsync needs (modelBytes)",
                       "lpr.nnrtTryModelAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job->modelBytes)) {
    return RejectedJob(env, job, "ok=0;error=empty model buffer", "lpr.nnrtTryModelAsync");
  }
  if (argc >= 2) {
    napi_get_value_int32(env, args[1], &job->deviceIndex);
  }
  return QueueJob(env, job, "lpr.nnrtTryModelAsync");
}

static napi_value BenchAsync(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kBench;
  job->warmup = 10;
  job->repeat = 50;
  if (argc >= 1) napi_get_value_int32(env, args[0], &job->detId);
  if (argc >= 2) napi_get_value_int32(env, args[1], &job->warmup);
  if (argc >= 3) napi_get_value_int32(env, args[2], &job->repeat);
  return QueueJob(env, job, "lpr.benchAsync");
}

/** ncnn 加载的异步版本：Vulkan 首次加载含 shader 编译，不能压在 JS 线程上。 */
static napi_value NcnnLoadAsync(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kNcnnLoad;
  if (argc < 2) {
    return RejectedJob(env, job, "ok=0;error=ncnnLoadAsync needs (param, bin)",
                       "lpr.ncnnLoadAsync");
  }
  if (!ReadArrayBufferArg(env, args[0], job->param) ||
      !ReadArrayBufferArg(env, args[1], job->bin)) {
    return RejectedJob(env, job, "ok=0;error=empty param/bin buffer", "lpr.ncnnLoadAsync");
  }
  if (argc >= 3 && napi_get_value_bool(env, args[2], &job->useVulkan) != napi_ok) {
    return RejectedJob(env, job, "ok=0;error=useVulkan must be boolean", "lpr.ncnnLoadAsync");
  }
  return QueueJob(env, job, "lpr.ncnnLoadAsync");
}

// ============================================================ module
EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
  g_sessions.reserve(kMaxSessions);
  g_runner.Start();
  const int gpuInit = ncnn::create_gpu_instance();
  LOGI("NCNN INIT create=%{public}d gpu_count=%{public}d", gpuInit, ncnn::get_gpu_count());
  napi_property_descriptor desc[] = {
      // 异步（推荐）
      {"loadModelAsync", nullptr, LoadModelAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"pipelineAsync", nullptr, PipelineAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"cameraFrameAsync", nullptr, CameraFrameAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"benchAsync", nullptr, BenchAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnLoadAsync", nullptr, NcnnLoadAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnLoadSlotAsync", nullptr, NcnnLoadSlotAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"nnrtProbeAsync", nullptr, NnrtProbeAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"nnrtTryModelAsync", nullptr, NnrtTryModelAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      // 同步（兼容；内部同样走推理线程）
      {"loadModel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"run", nullptr, RunModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"bench", nullptr, BenchModel, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"pipeline", nullptr, PipelineRun, nullptr, nullptr, nullptr, napi_default, nullptr},
      // 工具
      {"listNnrtDevices", nullptr, ListNnrt, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"vulkanProbe", nullptr, VulkanProbe, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"appendLine", nullptr, AppendLine, nullptr, nullptr, nullptr, napi_default, nullptr},
      // ncnn（GPU 唯一通路）
      {"ncnnLoad", nullptr, NcnnLoadFn, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnRun", nullptr, NcnnRunFn, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"ncnnRelease", nullptr, NcnnReleaseFn, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
  return exports;
}
EXTERN_C_END

static napi_module lprModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterLprModule(void) {
  napi_module_register(&lprModule);
}
