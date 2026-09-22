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
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
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
/**
 * 专用推理线程。所有 MindSpore Lite / ncnn 调用都排到这里，串行执行。
 *
 * **必须用队列，不能用单个任务槽。** 曾经是 `task_` + `has_` 一个槽：
 * 两个并发 Submit 时，第二个覆盖 `task_` 而 `has_` 已经是 true；第一个跑完把
 * `has_` 置 false 并 notify_all，于是**两个提交者都被唤醒**，但第二个的任务
 * 从未被执行 —— 它的 job->kv 保持空串，Promise 静默 resolve 成 `""`。
 *
 * 真机症状（2026-09-21，相机页）：`loadModelAsync` 三次调用全部返回空串，
 * 面板显示「档位加载失败」，而 native 侧**一条 loadModel 日志都没有**。
 * 触发条件是并发：相机页加载模型的同一次 ensuresSessions 期间，首页的自动
 * 探针仍在后台提交任务。串行 await 本身不会碰撞，**跨页面的并发提交才会**。
 *
 * 修法：`std::deque` + 每个任务自带 `finished` 标志与自己的 condition_variable，
 * 唤醒只针对该任务，不存在「唤醒错人」。任务在锁外执行，推理期间不挡入队。
 */
class InferenceRunner {
 public:
  void Start() {
    if (th_.joinable()) {
      return;
    }
    th_ = std::thread([this] { Loop(); });
  }

  /** 提交并等待完成（同步 NAPI 用；JS 线程会被阻塞，仅兼容路径用）。 */
  void Submit(const std::function<void()>& fn) {
    auto item = std::make_shared<Item>();
    item->fn = fn;
    {
      std::unique_lock<std::mutex> lk(m_);
      q_.push_back(item);
    }
    cv_.notify_one();
    std::unique_lock<std::mutex> lk(item->m);
    item->doneCv.wait(lk, [&item] { return item->finished; });
  }

 private:
  struct Item {
    std::function<void()> fn;
    bool finished = false;
    std::mutex m;
    std::condition_variable doneCv;
  };

  void Loop() {
    for (;;) {
      std::shared_ptr<Item> item;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return !q_.empty(); });
        item = q_.front();
        q_.pop_front();
      }
      item->fn();  // 不持锁执行：推理期间不挡新任务入队
      {
        std::unique_lock<std::mutex> lk(item->m);
        item->finished = true;
      }
      item->doneCv.notify_all();  // 只唤醒等这一个任务的人
    }
  }

  std::thread th_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Item>> q_;
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
  kCameraFrame,
  /** T2：车辆检测（yolov5u，单模型，不进车牌流水线）。 */
  kVehicleDetect,
  /** T3：ROI 裁剪与坐标映射的单元自证（纯图像运算，不需要模型会话）。 */
  kRoiSelfTest,
  /** T3：车框 → 裁 ROI → 车牌检测 → 映射回原图（单框，不循环、不去重 —— 那是 T4）。 */
  kRoiPlateProbe,
  /** T4：ROI 路径端到端（车辆检测 → 逐框 ROI → 逐框车牌检测 → 映射 → 合并去重）。 */
  kRoiPipeline,
  /** T4：去重的单元自证（纯数据，不需要模型会话）。 */
  kRoiDedupeSelfTest,
  /** T4：「构造重叠车框」的集成验证（需要车辆 + 车牌两批会话）。 */
  kRoiOverlapSelfTest
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
  /**
   * kBench 干扰参数（2026-09-21，用于定位「隔离 7.4 ms vs 流水线 19.5 ms」）：
   *   gapMs     —— 迭代间 sleep，检验 DVFS（调用变稀疏是否掉频）
   *   polluteKB —— 迭代间搬运的干扰缓冲，检验缓存/带宽污染
   * 两者都发生在**计时区之外**，不改变被计时的那次推理本身。
   */
  double benchGapMs = 0;
  int benchPolluteKB = 0;
  /** kBench 忙等毫秒（gapMs 的对照组，见 MsBench::spinMs）。 */
  double benchSpinMs = 0;

  // kCameraFrame：NV21 原始帧 → RGBA(+旋转) → 流水线，一次调用做完
  int stride = 0;
  int rotation = 0;
  /** 分段计时（毫秒）：转换 / 推理。 */
  double convMs = 0;
  double inferMs = 0;

  // kVehicleDetect（T2）：车辆检测的阈值与范围。
  // 默认值只写在 NAPI 入口（conf 0.05 / iou 0.5 / vehicleOnly true），
  // 这里保持中性，避免两处默认值各自演化出分歧。
  float confThresh = 0;
  float iouThresh = 0;
  bool vehicleOnly = true;
  /** 检出数被 kMaxVehicleBoxes 截断过 —— 必须回给调用方，不能伪造"这就是全部"。 */
  bool truncated = false;

  // kRoiPlateProbe（T3）：车辆检测会话 id。
  // ⚠️ 车辆检测器与车牌检测器是**两个不同的模型**：车牌检测是 y5fu_320x（3 个 head 输出），
  // 车辆检测是 yolov5su（单输出 1x84x2100）。把 s.det 当车辆模型喂进去会得到
  // `yolov5u 期望单输出，实际 3` —— 这个错在设备上实测踩过一次。
  int vehId = -1;

  /**
   * kRoiPlateProbe（T3）：探第几个车辆框（按分数降序，0 = 最高分）。
   *
   * 为什么需要它：分数最高的那个车框经常贴着图的左/上边缘，`LprRoiFromBox` 会把它
   * clamp 到 `x0 = 0` —— 这时"映射忘了加 x0"与"映射正确"结果完全一样，
   * x 方向的映射等于**没被验证**。必须再探一个 `x0 > 0` 的框，才能把两个方向都盖住。
   */
  int boxIdx = 0;

  // kRoiPlateProbe（T3）：ROI 外扩比例。默认取 kRoiExpandDefault（0.15），
  // 这里给同一个常量而不是另写一个字面量，避免两处默认值各自演化出分歧。
  float roiExpand = kRoiExpandDefault;

  // kRoiPipeline（T4）：去重阈值（spec D3，0.5）。同上，默认值只写在 NAPI 入口。
  float dedupeIou = 0;

  /**
   * kCameraFrame（T5）：走 ROI 路径还是全图直检。
   *
   * 两条路都在相机帧上跑同一份 NV21→RGBA，差别只在后段：直检是「整图 → 车牌检测」，
   * ROI 是「整图 → 车辆检测 → 逐框 ROI → 逐框车牌检测 → 合并去重」。
   * 界面上的切换开关直接改这个位。
   */
  bool useRoi = false;

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

/** 两个整数框的 IoU（T3 探针用：判断 ROI 路径映射回来的框与直检框是否重合）。 */
static float RoiIou(const int a[4], const int b[4]) {
  const float x1 = static_cast<float>(std::max(a[0], b[0]));
  const float y1 = static_cast<float>(std::max(a[1], b[1]));
  const float x2 = static_cast<float>(std::min(a[2], b[2]));
  const float y2 = static_cast<float>(std::min(a[3], b[3]));
  const float iw = x2 - x1;
  const float ih = y2 - y1;
  if (!(iw > 0) || !(ih > 0)) {
    return 0;
  }
  const float inter = iw * ih;
  const float aa = static_cast<float>(std::max(0, a[2] - a[0]) * std::max(0, a[3] - a[1]));
  const float bb = static_cast<float>(std::max(0, b[2] - b[0]) * std::max(0, b[3] - b[1]));
  const float uni = aa + bb - inter;
  return uni > 0 ? inter / uni : 0;
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

      // ---------------------------------------------------------- 落点自证（ADR-0003）
      // NPU 利用率在 HarmonyOS 上读不到，所以「这个后端真的算了」只能靠输出张量
      // 自己的统计量来证。三个角色各一段，字段：
      //   role,req,landed,fallback,l2,l2AsFp16,used
      // - req      = 请求的后端（verbatim，见 MsSession::requested）
      // - landed   = 实际落点；与 req 不同即发生回落
      // - fallback = 非空表示第一次尝试没建成，加速器被静默跳过
      // - l2AsFp16 非零（而 CPU 行为 0）是「这个后端真算过」的指纹
      // - used     = 本次是否真调用了该 MS 会话。det 走 ncnn 时为 0，此时 l2
      //              是上一次的陈旧值，不得引用
      {
        auto rec = [](const char* role, MsSession* sess, bool used) -> std::string {
          if (sess == nullptr) {
            return std::string(role) + ",,,,0,0,0";
          }
          return std::string(role) + "," + KvSanitize(sess->requested) + "," +
                 KvSanitize(sess->backend) + "," + KvSanitize(sess->fallbackFrom) + "," +
                 Num(sess->lastL2) + "," + Num(sess->lastL2AsFp16) + "," +
                 (used ? "1" : "0");
        };
        kv += "backends=";
        kv += rec("det", s.det, !s.detNcnn);
        kv += "|";
        kv += rec("rec", s.rec, s.recSlot < 0);
        kv += "|";
        // cls 恒定 used=0：牌色自 2026-09-21 起走像素测量（ADR-0005），
        // 分类模型不再进入流水线，没有任何会话被调用。保留这一行是为了让
        // 字段数稳定（下游按位置解析），但 used=0 会让面板跳过它。
        kv += rec("cls", s.cls, false);
        kv += ";";
      }

      for (size_t i = 0; i < plates.size(); i++) {
        const PlateResult& p = plates[i];
        std::string v = Scrub(p.code) + "," + Num(p.detScore) + "," + Num(p.recConf) + "," +
                        std::to_string(p.layer) + "," +
                        std::to_string(p.rect[0]) + "|" + std::to_string(p.rect[1]) + "|" +
                        std::to_string(p.rect[2]) + "|" + std::to_string(p.rect[3]) + "," +
                        std::to_string(p.cropH) + "|" + std::to_string(p.cropW) + "," +
                        // 牌色：像素测量结果（ADR-0005）。原为三个分类器 logits
                        // `cls0|cls1|cls2`，那张标签表是旋转的；现在直接给结论 + 置信度。
                        Scrub(p.colour) + "|" + Num(p.colourConfidence) + ",";
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
      //
      // 【2026-09-21 两处修正】
      //
      // 1) 它**原先完全没有被计时**：夹在 convMs 结束与 t0 开始之间，于是既不在
      //    convMs 也不在 totalMs 里，日志上永远看不见。实测 native 单帧 25.2 ms
      //    （理论 39.7 fps）却只跑到 ~28 fps（35.7 ms/帧），差额约 10.5 ms/帧 ——
      //    这段是其中一块"隐形"开销。现在把它计入 convMs，让它可见。
      //
      // 2) 原写法是逐字节 + `i % 4 != 3` 分支。每字节一次取模与一次分支，
      //    480x640 的 RGBA 是 1.2 MB，全走一遍。主机微基准
      //    （bench_rgba_sum.cpp）实测：0.6903 ms -> 0.1642 ms，**快 4.2x**，
      //    且**与现状逐位相等**（已验证），因此是零风险的纯收益。
      //    改成按像素步进 4、直接累加 R/G/B，去掉取模与分支。
      //
      //    注意：不要改成"抽稀采样"（每 16 像素取 1）。它快 46x，但实测
      //    **对单字节改动不敏感** —— 而校验和的用途正是等价性判定，
      //    漏检会让这个证据失去意义。
      const double tSum0 = NowMs();
      long long rgbaSum = 0;
      {
        const uint8_t* p = img.data.data();
        const size_t n = img.data.size();
        for (size_t i = 0; i + 2 < n; i += 4) {
          rgbaSum += static_cast<long long>(p[i]) + p[i + 1] + p[i + 2];
        }
      }
      // 把校验和并入 convMs：它与 NV21→RGBA 同属"取帧后的数据准备"，
      // 单独列一个字段反而会让"分段之和 vs 端到端"的闭合校验再次出现缺口。
      const double convWithSumMs = NowMs() - tConv0;
      (void)tSum0;
      (void)convMs;

      std::vector<PlateResult> plates;
      std::vector<VehicleBox> vehBoxes;
      RoiPipelineStats roiStats;
      bool vehTruncated = false;
      std::string err;
      const double t0 = NowMs();
      if (job->useRoi) {
        // T5：ROI 路径 —— 车辆检测 → 逐框 ROI → 逐框车牌检测 → 映射回原图 → 合并去重。
        MsSession* veh = nullptr;
        {
          std::lock_guard<std::mutex> lk(g_regMutex);
          if (job->vehId < 0 || job->vehId >= (int)g_sessions.size()) {
            job->kv = "ok=0;count=0;totalMs=0;convMs=" + Num(convWithSumMs) +
                      ";inferMs=0;error=bad vehicle session id";
            return;
          }
          veh = g_sessions[job->vehId].s;
        }
        RoiPipelineOptions opt;
        opt.vehConf = job->confThresh > 0 ? job->confThresh : 0.05f;
        opt.roiExpand = job->roiExpand > 0 ? job->roiExpand : kRoiExpandDefault;
        opt.dedupeIou = job->dedupeIou > 0 ? job->dedupeIou : 0.5f;
        if (!LprRunRoiPipeline(img, s, veh, opt, plates, vehBoxes, vehTruncated, roiStats, err)) {
          LOGE("cameraFrame roiPipeline failed: %{public}s", err.c_str());
          job->kv = "ok=0;count=0;totalMs=0;convMs=" + Num(convWithSumMs) +
                    ";inferMs=0;error=" + KvSanitize(err);
          return;
        }
      } else if (!LprRunPipeline(img, s, plates, err)) {
        LOGE("cameraFrame pipeline failed: %{public}s", err.c_str());
        job->kv = "ok=0;count=0;totalMs=0;convMs=" + Num(convWithSumMs) +
                  ";inferMs=0;error=" + KvSanitize(err);
        return;
      }
      const double totalMs = NowMs() - t0;

      std::string kv = "ok=1;count=" + std::to_string(plates.size()) +
                       ";totalMs=" + Num(totalMs) +
                       ";convMs=" + Num(convWithSumMs) +
                       ";inferMs=" + Num(totalMs) +
                       ";w=" + std::to_string(img.width) +
                       ";h=" + std::to_string(img.height) +
                       ";rgbaSum=" + std::to_string(rgbaSum) +
                       ";useRoi=" + (job->useRoi ? "1" : "0") +
                       ";vehCount=" + std::to_string(roiStats.vehCount) +
                       // 跨类别去重丢掉的框数（同一目标被标成多个类别，IoU 可达 0.99）。
                       // 与 vehCount 分开报，界面才能给出完整账目：
                       // "检出 4 → 实际跑 3（跨类去重 1）"。
                       ";vehDeduped=" + std::to_string(roiStats.vehDeduped) +
                       ";vehTruncated=" + (vehTruncated ? "1" : "0") +
                       ";roiTried=" + std::to_string(roiStats.roiTried) +
                       ";roiSkipped=" + std::to_string(roiStats.roiSkipped) +
                       ";rawHits=" + std::to_string(roiStats.rawHits) +
                       ";dedupeDropped=" + std::to_string(roiStats.dedupeDropped) +
                       ";vehInferMs=" + Num(roiStats.vehInferMs) +
                       ";roiDetectMs=" + Num(roiStats.roiDetectMs) +
                       ";error=;";

      // 相机档同样带落点自证（ADR-0003）—— 相机页的落点面板靠它。
      // 相机是逐帧调用，所以这里的值就是本帧的观测值，不存在"陈旧"问题；
      // 但仍保留 used 字段，以便 det 走 ncnn 旁路时标明该会话未被调用。
      {
        auto rec = [](const char* role, MsSession* sess, bool used) -> std::string {
          if (sess == nullptr) {
            return std::string(role) + ",,,,0,0,0";
          }
          return std::string(role) + "," + KvSanitize(sess->requested) + "," +
                 KvSanitize(sess->backend) + "," + KvSanitize(sess->fallbackFrom) + "," +
                 Num(sess->lastL2) + "," + Num(sess->lastL2AsFp16) + "," +
                 (used ? "1" : "0");
        };
        kv += "backends=";
        kv += rec("det", s.det, !s.detNcnn);
        kv += "|";
        kv += rec("rec", s.rec, s.recSlot < 0);
        kv += "|";
        kv += rec("cls", s.cls, false);  // 牌色走像素测量，无会话调用（ADR-0005）
        kv += ";";
      }

      for (size_t i = 0; i < plates.size(); i++) {
        const PlateResult& p = plates[i];
        std::string v = Scrub(p.code) + "," + Num(p.detScore) + "," + Num(p.recConf) + "," +
                        std::to_string(p.layer) + "," +
                        std::to_string(p.rect[0]) + "|" + std::to_string(p.rect[1]) + "|" +
                        std::to_string(p.rect[2]) + "|" + std::to_string(p.rect[3]) + "," +
                        std::to_string(p.cropH) + "|" + std::to_string(p.cropW) + "," +
                        // 牌色：像素测量结果（ADR-0005）。原为三个分类器 logits
                        // `cls0|cls1|cls2`，那张标签表是旋转的；现在直接给结论 + 置信度。
                        Scrub(p.colour) + "|" + Num(p.colourConfidence) + ",";
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
             std::to_string(p.cropSum) +
             // T5：车牌归属的车辆框下标（-1 = 无归属）。**追加在末尾**，
             // 这样 p0[9]（stages）等既有下标的含义不变 —— 相机页在读它。
             "," + std::to_string(p.ownerVeh);
        kv += "p" + std::to_string(i) + "=" + v + ";";
      }
      // T5：车辆框（供界面叠加与归属连线）。直检模式下为空 —— 界面据此隐藏车框层。
      //
      // ⚠️ 段数与 T4 的 `roiPlateProbeAsync` **必须一致**（4 段：
      // classId,score,rect,cname）。这两处曾不一致 —— 这里 3 段、那里 4 段，
      // 而界面按 4 段解析 ⇒ 车框被**静默丢光**：native 报 `vehCount=4`，
      // 界面 `vehDraw=0`，一个框都画不出来，且不报任何错。
      // 教训：跨入口共用的字段格式要么写成常量，要么在两侧注释里互相点名。
      const std::vector<std::string>& vnames = LprCocoNames();
      for (size_t i = 0; i < vehBoxes.size(); i++) {
        const VehicleBox& b = vehBoxes[i];
        const char* cname =
            (b.classId >= 0 && b.classId < (int)vnames.size()) ? vnames[b.classId].c_str() : "?";
        kv += "v" + std::to_string(i) + "=" + std::to_string(b.classId) + "," + Num(b.score) +
              "," + Num(b.rect[0]) + "|" + Num(b.rect[1]) + "|" + Num(b.rect[2]) + "|" +
              Num(b.rect[3]) + "," + Scrub(cname) + ";";
      }
      job->kv = kv;
      job->convMs = convMs;
      job->inferMs = totalMs;
      return;
    }

    // ---------------------------------------------------------------- vehicle detect (T2)
    case JobKind::kVehicleDetect: {
      MsSession* det = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        if (job->detId < 0 || job->detId >= (int)g_sessions.size()) {
          job->kv = "ok=0;count=0;error=bad session id";
          return;
        }
        det = g_sessions[job->detId].s;
      }
      if (det == nullptr) {
        job->kv = "ok=0;count=0;error=null det session";
        return;
      }

      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = std::move(job->rgba);
      if (!img.Valid()) {
        job->kv = "ok=0;count=0;error=rgba size != w*h*4";
        return;
      }

      std::vector<VehicleBox> boxes;
      bool truncated = false;
      float inferMs = 0;
      std::string err;
      const double t0 = NowMs();
      if (!LprVehicleDetect(img, det, job->confThresh, job->iouThresh, job->vehicleOnly, boxes,
                            truncated, inferMs, err)) {
        LOGE("vehicleDetect failed: %{public}s", err.c_str());
        job->kv = "ok=0;count=0;error=" + KvSanitize(err);
        return;
      }
      const double totalMs = NowMs() - t0;

      int size = 0;
      bool nhwc = false;
      std::string geoErr;
      if (!LprDetectGeometryOf(det, size, nhwc, geoErr)) {
        size = -1;
        nhwc = false;
      }

      // 设备侧证据通道：一行汇总 + 每框一行。走 hilog 而不是往 App 私有目录写文件 ——
      // 私有目录 hdc 拉不出来（T1 踩过），而 `hdc shell hilog -x | grep VEH` 一直可用。
      // 每框单独一行也是为了绕开单条 hilog 的长度上限（框多时 kv 串能到几 KB）。
      LOGI("VEH summary count=%{public}zu truncated=%{public}d conf=%{public}f iou=%{public}f "
           "vehicleOnly=%{public}d size=%{public}d nhwc=%{public}d inferMs=%{public}f "
           "totalMs=%{public}f backend=%{public}s",
           boxes.size(), truncated ? 1 : 0, job->confThresh, job->iouThresh,
           job->vehicleOnly ? 1 : 0, size, nhwc ? 1 : 0, inferMs, totalMs,
           det->backend.c_str());

      std::string kv = "ok=1;count=" + std::to_string(boxes.size()) +
                       ";truncated=" + (truncated ? "1" : "0") +
                       ";conf=" + Num(job->confThresh) + ";iou=" + Num(job->iouThresh) +
                       ";vehicleOnly=" + (job->vehicleOnly ? "1" : "0") +
                       ";size=" + std::to_string(size) +
                       ";nhwc=" + (nhwc ? "1" : "0") +
                       ";inferMs=" + Num(inferMs) +
                       ";totalMs=" + Num(totalMs) +
                       ";backend=" + KvSanitize(det->backend) +
                       ";requested=" + KvSanitize(det->requested) +
                       ";fallbackFrom=" + KvSanitize(det->fallbackFrom) + ";error=;";

      const std::vector<std::string>& names = LprCocoNames();
      for (size_t i = 0; i < boxes.size(); i++) {
        const VehicleBox& b = boxes[i];
        const char* cname =
            (b.classId >= 0 && b.classId < (int)names.size()) ? names[b.classId].c_str() : "?";
        LOGI("VEH box idx=%{public}zu cls=%{public}d name=%{public}s score=%{public}f "
             "rect=%{public}f,%{public}f,%{public}f,%{public}f",
             i, b.classId, cname, b.score, b.rect[0], b.rect[1], b.rect[2], b.rect[3]);
        // 与 p0=... 同一套写法：逗号分段，框内四个数用 | 连。
        kv += "b" + std::to_string(i) + "=" + std::to_string(b.classId) + "," + Num(b.score) +
              "," + Num(b.rect[0]) + "|" + Num(b.rect[1]) + "|" + Num(b.rect[2]) + "|" +
              Num(b.rect[3]) + "," + Scrub(cname) + ";";
      }
      job->kv = kv;
      job->inferMs = inferMs;
      job->truncated = truncated;
      return;
    }

    // ---------------------------------------------------------------- roi self test (T3)
    case JobKind::kRoiSelfTest: {
      // 纯图像运算，不碰任何会话。RGBA 已经在入口拷进 job->rgba。
      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = job->rgba;
      if (!img.Valid()) {
        job->kv = "ok=0;error=roiSelfTest: invalid rgba (" + std::to_string(job->w) + "x" +
                  std::to_string(job->h) + " vs " + std::to_string(job->rgba.size()) + " bytes)";
        return;
      }
      // 逐行报告写 hilog：单条 hilog 有长度上限，而报告有二十来行，
      // 攒成一条会被截断。ArkTS 侧也会逐行回显，两条路径互为印证。
      const std::string report = LprRoiSelfTest(img);
      std::string line;
      for (size_t i = 0; i <= report.size(); i++) {
        if (i == report.size() || report[i] == '\n') {
          if (!line.empty()) {
            LOGI("T3ROI %{public}s", line.c_str());
          }
          line.clear();
        } else {
          line += report[i];
        }
      }
      // 整段原样返回（**不做** KvSanitize）：它是多行报告，不是 kv 串。
      // 走 KvSanitize 会把 '\n' 压成 ','，逐行结构就没了，ArkTS 也就无法逐行回显。
      job->kv = report;
      return;
    }

    // ------------------------------------------------- roi plate probe (T3)
    // 车框 → 裁 ROI → 车牌检测 → **映射回原图**。只做分数最高的那一个框：
    // 循环遍历 + 合并去重是 T4 的事，这一票要证的是"裁得对、映射不偏"。
    case JobKind::kRoiPlateProbe: {
      LprSessions s;
      MsSession* veh = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        const int ids[3] = {job->detId, job->recId, job->clsId};
        for (int i = 0; i < 3; i++) {
          if (ids[i] < 0 || ids[i] >= (int)g_sessions.size()) {
            job->kv = "ok=0;count=0;error=bad session id";
            return;
          }
        }
        if (job->vehId < 0 || job->vehId >= (int)g_sessions.size()) {
          job->kv = "ok=0;count=0;error=bad vehicle session id";
          return;
        }
        veh = g_sessions[job->vehId].s;
        s.det = g_sessions[job->detId].s;
        s.rec = g_sessions[job->recId].s;
        s.cls = g_sessions[job->clsId].s;
      }

      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = std::move(job->rgba);
      if (!img.Valid()) {
        job->kv = "ok=0;count=0;error=rgba size != w*h*4";
        return;
      }

      std::string err;

      // 1) 对照基线：整图直接跑一遍车牌流水线
      std::vector<PlateResult> direct;
      const double t0 = NowMs();
      if (!LprRunPipeline(img, s, direct, err)) {
        job->kv = "ok=0;count=0;error=direct=" + KvSanitize(err);
        return;
      }
      const double directMs = NowMs() - t0;

      // 2) 车辆检测（低阈值 + 遍历所有框，T2 的 D1 决定）。
      //    ⚠️ 用 veh（yolov5su），**不是** s.det（y5fu_320x 车牌检测器）。
      std::vector<VehicleBox> vehBoxes;
      bool trunc = false;
      float vehInferMs = 0;
      if (!LprVehicleDetect(img, veh, job->confThresh, job->iouThresh, /*vehicleOnly=*/true,
                            vehBoxes, trunc, vehInferMs, err)) {
        job->kv = "ok=0;count=0;error=veh=" + KvSanitize(err);
        return;
      }

      std::string kv = "ok=1;directCount=" + std::to_string(direct.size()) +
                       ";vehCount=" + std::to_string(vehBoxes.size()) +
                       ";truncated=" + (trunc ? "1" : "0") +
                       ";conf=" + Num(job->confThresh) + ";iou=" + Num(job->iouThresh) +
                       ";expand=" + Num(job->roiExpand) +
                       ";directMs=" + Num(directMs) + ";error=;";

      if (vehBoxes.empty()) {
        LOGI("T3PROBE no vehicle box; direct=%{public}zu", direct.size());
        job->kv = kv + "roiValid=0;roiCount=0;";
        return;
      }

      if (job->boxIdx < 0 || job->boxIdx >= (int)vehBoxes.size()) {
        job->kv = kv + "roiValid=0;roiCount=0;error=boxIdx out of range";
        return;
      }
      const VehicleBox& b0 = vehBoxes[job->boxIdx];
      const RoiRect roi = LprRoiFromBox(b0.rect, img.width, img.height, job->roiExpand);
      kv += "boxIdx=" + std::to_string(job->boxIdx) +
            ";roiValid=" + std::string(roi.valid ? "1" : "0") +
            ";roiX0=" + std::to_string(roi.x0) + ";roiY0=" + std::to_string(roi.y0) +
            ";roiW=" + std::to_string(roi.w) + ";roiH=" + std::to_string(roi.h) +
            ";roiClamped=" + (roi.clamped ? "1" : "0") +
            ";roiCoversBox=" + (roi.ContainsBox(b0.rect) ? "1" : "0") +
            ";boxCls=" + std::to_string(b0.classId) + ";boxScore=" + Num(b0.score) + ";";
      if (!roi.valid) {
        job->kv = kv + "roiCount=0;";
        return;
      }

      // 3) 裁 ROI → 车牌流水线 → 映射回原图坐标
      RgbaImage crop;
      if (!LprCropRoi(img, roi, crop, err)) {
        job->kv = kv + "roiCount=0;error=" + KvSanitize(err) + ";";
        return;
      }
      std::vector<PlateResult> roiPlates;
      const double t1 = NowMs();
      if (!LprRunPipeline(crop, s, roiPlates, err)) {
        job->kv = kv + "roiCount=0;error=roi=" + KvSanitize(err) + ";";
        return;
      }
      const double roiMs = NowMs() - t1;
      // 少了这一步，框会整体偏移 (roi.x0, roi.y0) —— spec §六.3 说的就是这个坑。
      for (PlateResult& p : roiPlates) {
        LprRoiMapRect(p.rect, roi);
      }
      kv += "roiCount=" + std::to_string(roiPlates.size()) + ";roiMs=" + Num(roiMs) + ";";

      LOGI("T3PROBE boxIdx=%{public}d direct=%{public}zu veh=%{public}zu "
           "roi=%{public}d,%{public}d,%{public}d,%{public}d crop=%{public}dx%{public}d "
           "roiPlates=%{public}zu",
           job->boxIdx, direct.size(), vehBoxes.size(), roi.x0, roi.y0, roi.w, roi.h, crop.width,
           crop.height, roiPlates.size());

      // 逐框证据：ROI 路径映射回来的框 vs 直检框的 IoU。
      // 判据是"两者高度重合"—— 这是程序化判据，不靠人眼看图。
      for (size_t i = 0; i < roiPlates.size(); i++) {
        const PlateResult& p = roiPlates[i];
        float bestIou = 0;
        int bestJ = -1;
        for (size_t j = 0; j < direct.size(); j++) {
          const float v = RoiIou(p.rect, direct[j].rect);
          if (v > bestIou) {
            bestIou = v;
            bestJ = static_cast<int>(j);
          }
        }
        LOGI("T3ROIBOX idx=%{public}zu rect=%{public}d,%{public}d,%{public}d,%{public}d "
             "score=%{public}f code=%{public}s directJ=%{public}d iou=%{public}f",
             i, p.rect[0], p.rect[1], p.rect[2], p.rect[3], p.detScore, p.code.c_str(), bestJ,
             bestIou);
        kv += "r" + std::to_string(i) + "=" + std::to_string(p.rect[0]) + "|" +
              std::to_string(p.rect[1]) + "|" + std::to_string(p.rect[2]) + "|" +
              std::to_string(p.rect[3]) + "," + Num(p.detScore) + "," +
              std::to_string(bestJ) + "," + Num(bestIou) + "," + Scrub(p.code) + ";";
      }
      for (size_t j = 0; j < direct.size(); j++) {
        const PlateResult& p = direct[j];
        LOGI("T3DIRECTBOX idx=%{public}zu rect=%{public}d,%{public}d,%{public}d,%{public}d "
             "score=%{public}f code=%{public}s",
             j, p.rect[0], p.rect[1], p.rect[2], p.rect[3], p.detScore, p.code.c_str());
        kv += "d" + std::to_string(j) + "=" + std::to_string(p.rect[0]) + "|" +
              std::to_string(p.rect[1]) + "|" + std::to_string(p.rect[2]) + "|" +
              std::to_string(p.rect[3]) + "," + Num(p.detScore) + "," + Scrub(p.code) + ";";
      }
      job->kv = kv;
      return;
    }

    // ------------------------------------------------- roi pipeline (T4)
    case JobKind::kRoiPipeline: {
      LprSessions s;
      MsSession* veh = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        const int ids[3] = {job->detId, job->recId, job->clsId};
        for (int i = 0; i < 3; i++) {
          if (ids[i] < 0 || ids[i] >= (int)g_sessions.size()) {
            job->kv = "ok=0;count=0;error=bad session id";
            return;
          }
        }
        if (job->vehId < 0 || job->vehId >= (int)g_sessions.size()) {
          job->kv = "ok=0;count=0;error=bad vehicle session id";
          return;
        }
        veh = g_sessions[job->vehId].s;
        s.det = g_sessions[job->detId].s;
        s.rec = g_sessions[job->recId].s;
        s.cls = g_sessions[job->clsId].s;
      }

      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = std::move(job->rgba);
      if (!img.Valid()) {
        job->kv = "ok=0;count=0;error=rgba size != w*h*4";
        return;
      }

      RoiPipelineOptions opt;
      opt.vehConf = job->confThresh;
      opt.vehIou = job->iouThresh;
      opt.roiExpand = job->roiExpand;
      opt.dedupeIou = job->dedupeIou;

      std::vector<PlateResult> plates;
      std::vector<VehicleBox> vehs;
      bool trunc = false;
      RoiPipelineStats st;
      std::string err;
      if (!LprRunRoiPipeline(img, s, veh, opt, plates, vehs, trunc, st, err)) {
        LOGE("roiPipeline failed: %{public}s", err.c_str());
        job->kv = "ok=0;count=0;error=" + KvSanitize(err);
        return;
      }

      LOGI("T4PIPE veh=%{public}d truncated=%{public}d roiTried=%{public}d roiSkipped=%{public}d "
           "rawHits=%{public}d dropped=%{public}d count=%{public}zu vehInferMs=%{public}f "
           "roiDetectMs=%{public}f totalMs=%{public}f",
           st.vehCount, trunc ? 1 : 0, st.roiTried, st.roiSkipped, st.rawHits, st.dedupeDropped,
           plates.size(), st.vehInferMs, st.roiDetectMs, st.totalMs);

      std::string kv = "ok=1;vehCount=" + std::to_string(st.vehCount) +
                       ";vehTruncated=" + (trunc ? "1" : "0") +
                       ";roiTried=" + std::to_string(st.roiTried) +
                       ";roiSkipped=" + std::to_string(st.roiSkipped) +
                       ";rawHits=" + std::to_string(st.rawHits) +
                       ";dedupeDropped=" + std::to_string(st.dedupeDropped) +
                       ";count=" + std::to_string(plates.size()) +
                       ";conf=" + Num(opt.vehConf) + ";iou=" + Num(opt.vehIou) +
                       ";expand=" + Num(opt.roiExpand) + ";dedupeIou=" + Num(opt.dedupeIou) +
                       ";vehInferMs=" + Num(st.vehInferMs) +
                       ";roiDetectMs=" + Num(st.roiDetectMs) +
                       ";totalMs=" + Num(st.totalMs) + ";error=;";

      const std::vector<std::string>& names = LprCocoNames();
      for (size_t i = 0; i < vehs.size(); i++) {
        const VehicleBox& b = vehs[i];
        const char* cname =
            (b.classId >= 0 && b.classId < (int)names.size()) ? names[b.classId].c_str() : "?";
        LOGI("T4VEH idx=%{public}zu cls=%{public}d name=%{public}s score=%{public}f "
             "rect=%{public}f,%{public}f,%{public}f,%{public}f",
             i, b.classId, cname, b.score, b.rect[0], b.rect[1], b.rect[2], b.rect[3]);
        kv += "v" + std::to_string(i) + "=" + std::to_string(b.classId) + "," + Num(b.score) +
              "," + Num(b.rect[0]) + "|" + Num(b.rect[1]) + "|" + Num(b.rect[2]) + "|" +
              Num(b.rect[3]) + "," + Scrub(cname) + ";";
      }
      for (size_t i = 0; i < plates.size(); i++) {
        const PlateResult& p = plates[i];
        LOGI("T4PLATE idx=%{public}zu rect=%{public}d,%{public}d,%{public}d,%{public}d "
             "score=%{public}f owner=%{public}d colour=%{public}s code=%{public}s "
             "recConf=%{public}f",
             i, p.rect[0], p.rect[1], p.rect[2], p.rect[3], p.detScore, p.ownerVeh,
             p.colour.c_str(), p.code.c_str(), p.recConf);
        kv += "p" + std::to_string(i) + "=" + std::to_string(p.rect[0]) + "|" +
              std::to_string(p.rect[1]) + "|" + std::to_string(p.rect[2]) + "|" +
              std::to_string(p.rect[3]) + "," + Num(p.detScore) + "," +
              std::to_string(p.ownerVeh) + "," + Scrub(p.colour) + "," + Scrub(p.code) + "," +
              Num(p.recConf) + ";";
      }
      job->kv = kv;
      return;
    }

    // ------------------------------------------------- roi dedupe self test (T4)
    case JobKind::kRoiDedupeSelfTest: {
      const std::string report = LprDedupeSelfTest();
      std::string line;
      for (size_t i = 0; i <= report.size(); i++) {
        if (i == report.size() || report[i] == '\n') {
          if (!line.empty()) {
            LOGI("T4DEDUPE %{public}s", line.c_str());
          }
          line.clear();
        } else {
          line += report[i];
        }
      }
      job->kv = report;
      return;
    }

    // ---------------------------------------- roi overlap dedupe probe (T4)
    // 构造重叠车框，验证去重真的把"同一块牌检出两次"收敛成一条。
    case JobKind::kRoiOverlapSelfTest: {
      LprSessions s;
      MsSession* veh = nullptr;
      {
        std::lock_guard<std::mutex> lk(g_regMutex);
        const int ids[3] = {job->detId, job->recId, job->clsId};
        for (int i = 0; i < 3; i++) {
          if (ids[i] < 0 || ids[i] >= (int)g_sessions.size()) {
            job->kv = "case=overlap-detect;ok=0;err=bad session id\ntotal=1;failed=1\n";
            return;
          }
        }
        if (job->vehId < 0 || job->vehId >= (int)g_sessions.size()) {
          job->kv = "case=overlap-detect;ok=0;err=bad vehicle session id\ntotal=1;failed=1\n";
          return;
        }
        veh = g_sessions[job->vehId].s;
        s.det = g_sessions[job->detId].s;
        s.rec = g_sessions[job->recId].s;
        s.cls = g_sessions[job->clsId].s;
      }
      RgbaImage img;
      img.width = job->w;
      img.height = job->h;
      img.data = std::move(job->rgba);
      if (!img.Valid()) {
        job->kv = "case=overlap-detect;ok=0;err=rgba size != w*h*4\ntotal=1;failed=1\n";
        return;
      }
      const std::string report = LprOverlapDedupeProbe(img, s, veh, 6);
      std::string line;
      for (size_t i = 0; i <= report.size(); i++) {
        if (i == report.size() || report[i] == '\n') {
          if (!line.empty()) {
            LOGI("T4OVERLAP %{public}s", line.c_str());
          }
          line.clear();
        } else {
          line += report[i];
        }
      }
      job->kv = report;
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
      MsBench b = MsBenchRun(s, job->warmup, job->repeat, job->benchGapMs,
                             job->benchPolluteKB, job->benchSpinMs);
      job->kv = "ok=" + std::string(b.ok ? "1" : "0") +
                ";backend=" + KvSanitize(b.backend) +
                ";warmup=" + std::to_string(b.warmup) +
                ";repeat=" + std::to_string(b.repeat) +
                ";gapMs=" + Num(b.gapMs) +
                ";polluteKB=" + std::to_string(b.polluteKB) +
                ";spinMs=" + Num(b.spinMs) +
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
 * 车辆检测（T2）。**只跑车辆检测器，不进车牌流水线。**
 *
 * 参数：detId, rgba, width, height, confThresh?, iouThresh?, vehicleOnly?
 *
 * 与 pipelineAsync 的关系：两者都用同一个会话，但 pipelineAsync 输出的是车牌，
 * 这里输出的是车辆框（源图坐标）+ COCO 类号 + 分数，供 T4 拿来逐框找车牌。
 *
 * 默认值（写在**这一层**，`AsyncJob` 里保持中性，避免两处默认值各自漂移）：
 *   confThresh  = 0.05 —— 比车牌检测的 0.25 低得多。车辆是大目标，低阈值图的是
 *                         T4 不因为漏框而整段丢检；代价是候选变多，由按类 NMS 收。
 *   iouThresh   = 0.5
 *   vehicleOnly = true —— 只留 COCO 的 car(2) / motorcycle(3) / bus(5) / truck(7)。
 *
 * 显式给的阈值必须是 (0,1) 内的有限数；不合法就**诚实报错**，不静默退回默认值
 * （静默退回会让"我明明设了 0.3"变成一个查不出来的假象）。
 */
static napi_value VehicleDetectAsync(napi_env env, napi_callback_info info) {
  size_t argc = 7;
  napi_value args[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kVehicleDetect;
  if (argc < 4) {
    return RejectedJob(env, job, "ok=0;count=0;error=vehicleDetectAsync needs (detId, rgba, w, h)",
                       "lpr.vehicleDetectAsync");
  }
  if (napi_get_value_int32(env, args[0], &job->detId) != napi_ok) {
    return RejectedJob(env, job, "ok=0;count=0;error=detId must be an integer",
                       "lpr.vehicleDetectAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[1], job->rgba)) {
    return RejectedJob(env, job, "ok=0;count=0;error=rgba ArrayBuffer is empty",
                       "lpr.vehicleDetectAsync");
  }
  napi_get_value_int32(env, args[2], &job->w);
  napi_get_value_int32(env, args[3], &job->h);

  job->confThresh = 0.05f;
  job->iouThresh = 0.5f;
  job->vehicleOnly = true;

  // JS 的 number 一律是 double，整数参数走同一个取值函数即可。
  double v = 0;
  if (argc >= 5 && napi_get_value_double(env, args[4], &v) == napi_ok) {
    if (!(v > 0.0) || !(v < 1.0)) {
      return RejectedJob(env, job, "ok=0;count=0;error=confThresh must be in (0,1)",
                         "lpr.vehicleDetectAsync");
    }
    job->confThresh = static_cast<float>(v);
  }
  if (argc >= 6 && napi_get_value_double(env, args[5], &v) == napi_ok) {
    if (!(v > 0.0) || !(v < 1.0)) {
      return RejectedJob(env, job, "ok=0;count=0;error=iouThresh must be in (0,1)",
                         "lpr.vehicleDetectAsync");
    }
    job->iouThresh = static_cast<float>(v);
  }
  if (argc >= 7) {
    bool b = true;
    if (napi_get_value_bool(env, args[6], &b) != napi_ok) {
      return RejectedJob(env, job, "ok=0;count=0;error=vehicleOnly must be a boolean",
                         "lpr.vehicleDetectAsync");
    }
    job->vehicleOnly = b;
  }
  return QueueJob(env, job, "lpr.vehicleDetectAsync");
}

/**
 * T3：ROI 裁剪 + 坐标映射的单元自证。
 *
 * 入参只要一张 RGBA 图 —— 它测的是**纯几何与逐字节裁剪**，与模型无关，
 * 所以不需要会话 id。传无效图也能跑完：原生侧会自动合成确定性图案，
 * 并在报告里写明 `note=src-synth`，不假装用的是真实素材。
 *
 * 返回多行报告（**不是** kv 串），每行形如 `case=<名字>;ok=0/1;<细节>`。
 */
static napi_value RoiSelfTestAsync(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3] = {nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kRoiSelfTest;
  if (argc < 3) {
    return RejectedJob(env, job, "roiSelfTestAsync needs (rgba, width, height)",
                       "lpr.roiSelfTestAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[0], job->rgba)) {
    return RejectedJob(env, job, "roiSelfTestAsync: rgba ArrayBuffer is empty",
                       "lpr.roiSelfTestAsync");
  }
  if (napi_get_value_int32(env, args[1], &job->w) != napi_ok ||
      napi_get_value_int32(env, args[2], &job->h) != napi_ok) {
    return RejectedJob(env, job, "roiSelfTestAsync: width/height must be integers",
                       "lpr.roiSelfTestAsync");
  }
  return QueueJob(env, job, "lpr.roiSelfTestAsync");
}

/**
 * T3：车框 → 裁 ROI → 车牌检测 → **映射回原图**（只做分数最高的那一个框）。
 *
 * 循环遍历所有车框 + 合并去重属于 T4，本入口刻意不做 —— 这一票要证的是
 * "ROI 裁得对、映射不偏"。判据是 ROI 路径映射回来的框与直检框的 IoU，
 * 全部在日志里逐框给出，不靠人眼看图。
 *
 * 入参有**四个**会话 id：`vehId` 是车辆检测器（yolov5su），`detId`/`recId`/`clsId`
 * 是车牌流水线的三个模型。它们不是同一批模型 —— 混用会得到
 * `yolov5u 期望单输出，实际 3`（设备上实测踩过）。
 *
 * `expand` 缺省取 `kRoiExpandDefault`（0.15）；显式传入必须落在 [0,1)，否则报错。
 */
static napi_value RoiPlateProbeAsync(napi_env env, napi_callback_info info) {
  size_t argc = 9;
  napi_value args[9] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kRoiPlateProbe;
  // 与 T2 同一口径：车辆检测用低阈值 + 遍历所有车框（spec D1）。
  job->confThresh = 0.05f;
  job->iouThresh = 0.5f;
  if (argc < 7) {
    return RejectedJob(env, job,
                       "roiPlateProbeAsync needs (vehId, detId, recId, clsId, rgba, w, h, "
                       "boxIdx?, expand?)",
                       "lpr.roiPlateProbeAsync");
  }
  if (napi_get_value_int32(env, args[0], &job->vehId) != napi_ok ||
      napi_get_value_int32(env, args[1], &job->detId) != napi_ok ||
      napi_get_value_int32(env, args[2], &job->recId) != napi_ok ||
      napi_get_value_int32(env, args[3], &job->clsId) != napi_ok) {
    return RejectedJob(env, job, "roiPlateProbeAsync: ids must be integers",
                       "lpr.roiPlateProbeAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[4], job->rgba)) {
    return RejectedJob(env, job, "roiPlateProbeAsync: rgba ArrayBuffer is empty",
                       "lpr.roiPlateProbeAsync");
  }
  if (napi_get_value_int32(env, args[5], &job->w) != napi_ok ||
      napi_get_value_int32(env, args[6], &job->h) != napi_ok) {
    return RejectedJob(env, job, "roiPlateProbeAsync: width/height must be integers",
                       "lpr.roiPlateProbeAsync");
  }
  if (argc >= 8) {
    if (napi_get_value_int32(env, args[7], &job->boxIdx) != napi_ok) {
      return RejectedJob(env, job, "roiPlateProbeAsync: boxIdx must be an integer",
                         "lpr.roiPlateProbeAsync");
    }
    if (job->boxIdx < 0) {
      return RejectedJob(env, job, "roiPlateProbeAsync: boxIdx must be >= 0",
                         "lpr.roiPlateProbeAsync");
    }
  }
  if (argc >= 9) {
    double v = 0;
    if (napi_get_value_double(env, args[8], &v) == napi_ok) {
      if (!(v >= 0.0) || !(v < 1.0)) {
        return RejectedJob(env, job, "roiPlateProbeAsync: expand must be in [0,1)",
                           "lpr.roiPlateProbeAsync");
      }
      job->roiExpand = static_cast<float>(v);
    }
  }
  return QueueJob(env, job, "lpr.roiPlateProbeAsync");
}

/**
 * T4：ROI 路径端到端 —— 车辆检测 → **逐框**裁 ROI → 逐框车牌检测 → 映射回原图
 * → 合并去重。输出车牌框 + 车牌串 + 颜色 + 归属的车辆。
 *
 * 默认参数即 spec 的定案值：vehConf=0.05（D1）、roiExpand=0.15、dedupeIou=0.5（D3）。
 * 三个阈值都可显式传，但必须在合法区间内，否则**报错**而不是静默退回默认值。
 *
 * `vehId` 是车辆检测器（yolov5su），与 `detId`（车牌检测器 y5fu_320x）不是同一个模型。
 */
static napi_value RoiPipelineAsync(napi_env env, napi_callback_info info) {
  size_t argc = 10;
  napi_value args[10] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kRoiPipeline;
  job->confThresh = 0.05f;             // D1：极低阈值
  job->iouThresh = 0.5f;
  job->roiExpand = kRoiExpandDefault;  // 0.15
  job->dedupeIou = 0.5f;               // D3
  if (argc < 7) {
    return RejectedJob(env, job,
                       "roiPipelineAsync needs (vehId, detId, recId, clsId, rgba, w, h, "
                       "vehConf?, roiExpand?, dedupeIou?)",
                       "lpr.roiPipelineAsync");
  }
  if (napi_get_value_int32(env, args[0], &job->vehId) != napi_ok ||
      napi_get_value_int32(env, args[1], &job->detId) != napi_ok ||
      napi_get_value_int32(env, args[2], &job->recId) != napi_ok ||
      napi_get_value_int32(env, args[3], &job->clsId) != napi_ok) {
    return RejectedJob(env, job, "roiPipelineAsync: ids must be integers",
                       "lpr.roiPipelineAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[4], job->rgba)) {
    return RejectedJob(env, job, "roiPipelineAsync: rgba ArrayBuffer is empty",
                       "lpr.roiPipelineAsync");
  }
  if (napi_get_value_int32(env, args[5], &job->w) != napi_ok ||
      napi_get_value_int32(env, args[6], &job->h) != napi_ok) {
    return RejectedJob(env, job, "roiPipelineAsync: width/height must be integers",
                       "lpr.roiPipelineAsync");
  }
  double v = 0;
  if (argc >= 8 && napi_get_value_double(env, args[7], &v) == napi_ok) {
    if (!(v > 0.0) || !(v < 1.0)) {
      return RejectedJob(env, job, "roiPipelineAsync: vehConf must be in (0,1)",
                         "lpr.roiPipelineAsync");
    }
    job->confThresh = static_cast<float>(v);
  }
  if (argc >= 9 && napi_get_value_double(env, args[8], &v) == napi_ok) {
    if (!(v >= 0.0) || !(v < 1.0)) {
      return RejectedJob(env, job, "roiPipelineAsync: roiExpand must be in [0,1)",
                         "lpr.roiPipelineAsync");
    }
    job->roiExpand = static_cast<float>(v);
  }
  if (argc >= 10 && napi_get_value_double(env, args[9], &v) == napi_ok) {
    if (!(v > 0.0) || !(v <= 1.0)) {
      return RejectedJob(env, job, "roiPipelineAsync: dedupeIou must be in (0,1]",
                         "lpr.roiPipelineAsync");
    }
    job->dedupeIou = static_cast<float>(v);
  }
  return QueueJob(env, job, "lpr.roiPipelineAsync");
}

/**
 * T4：合并去重的单元自证（纯数据，不需要模型会话，因此也没有入参）。
 *
 * 为什么单独一条入口：票面要求「构造重叠车框验证去重生效」。重叠车框的**后果**就是
 * 同一块牌被检出两次 —— 这里直接把那种输入喂给去重函数，断言保留条数与保留的是哪一条。
 * 返回多行报告（不是 kv 串），末行 `total=N;failed=M`。
 */
static napi_value RoiDedupeSelfTestAsync(napi_env env, napi_callback_info info) {
  (void)env;
  (void)info;
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kRoiDedupeSelfTest;
  return QueueJob(env, job, "lpr.roiDedupeSelfTestAsync");
}

/**
 * T4：「构造重叠车框」的集成验证。
 *
 * 与纯数据的 `roiDedupeSelfTestAsync` 不同，这条需要**真的跑模型**：取图上分数最高的
 * 真实车框 A，人为构造一个向四周外扩 6 px 的车框 B（与 A 必然重叠），两个重叠 ROI 各自
 * 跑车牌检测 —— 同一块牌会被检出两次，去重后必须只剩 1 条。这才是票面要的
 * 「构造重叠车框验证去重生效」，而不是只喂重叠的**车牌框**。
 *
 * 入参同 `roiPipelineAsync` 的前 7 个。返回多行报告，末行 `total=N;failed=M`。
 */
static napi_value RoiOverlapSelfTestAsync(napi_env env, napi_callback_info info) {
  size_t argc = 7;
  napi_value args[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kRoiOverlapSelfTest;
  if (argc < 7) {
    return RejectedJob(env, job,
                       "roiOverlapSelfTestAsync needs (vehId, detId, recId, clsId, rgba, w, h)",
                       "lpr.roiOverlapSelfTestAsync");
  }
  if (napi_get_value_int32(env, args[0], &job->vehId) != napi_ok ||
      napi_get_value_int32(env, args[1], &job->detId) != napi_ok ||
      napi_get_value_int32(env, args[2], &job->recId) != napi_ok ||
      napi_get_value_int32(env, args[3], &job->clsId) != napi_ok) {
    return RejectedJob(env, job, "roiOverlapSelfTestAsync: ids must be integers",
                       "lpr.roiOverlapSelfTestAsync");
  }
  if (!ReadArrayBufferArgU8(env, args[4], job->rgba)) {
    return RejectedJob(env, job, "roiOverlapSelfTestAsync: rgba ArrayBuffer is empty",
                       "lpr.roiOverlapSelfTestAsync");
  }
  if (napi_get_value_int32(env, args[5], &job->w) != napi_ok ||
      napi_get_value_int32(env, args[6], &job->h) != napi_ok) {
    return RejectedJob(env, job, "roiOverlapSelfTestAsync: width/height must be integers",
                       "lpr.roiOverlapSelfTestAsync");
  }
  return QueueJob(env, job, "lpr.roiOverlapSelfTestAsync");
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
  size_t argc = 13;
  napi_value args[13] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kCameraFrame;
  // T5 的 ROI 路径默认参数：与 spec D1/D3 定案值一致（只在入口写默认值）。
  job->confThresh = 0.05f;
  job->roiExpand = kRoiExpandDefault;
  job->dedupeIou = 0.5f;
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
  // T5：第 12/13 参 —— 车辆检测会话 id 与「是否走 ROI 路径」。
  // 走 ROI 时必须给 vehId，否则 native 侧会以 bad vehicle session id 明确报错
  // （而不是悄悄退回直检 —— 那会让界面上的开关变成假的）。
  if (argc >= 12) napi_get_value_int32(env, args[11], &job->vehId);
  if (argc >= 13) {
    bool b = false;
    if (napi_get_value_bool(env, args[12], &b) != napi_ok) {
      return RejectedJob(env, job, "ok=0;error=useRoi must be a boolean",
                         "lpr.cameraFrameAsync");
    }
    job->useRoi = b;
  }
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
  // 参数：(sessionId, warmup, repeat, gapMs, polluteKB, spinMs)
  // 后三个可选（2026-09-21 新增），不传则与历史行为逐位相同。
  size_t argc = 6;
  napi_value args[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  AsyncJob* job = new AsyncJob();
  job->kind = JobKind::kBench;
  job->warmup = 10;
  job->repeat = 50;
  if (argc >= 1) napi_get_value_int32(env, args[0], &job->detId);
  if (argc >= 2) napi_get_value_int32(env, args[1], &job->warmup);
  if (argc >= 3) napi_get_value_int32(env, args[2], &job->repeat);
  if (argc >= 4) {
    double g = 0;
    if (napi_get_value_double(env, args[3], &g) == napi_ok && g >= 0 && g <= 1000) {
      job->benchGapMs = g;
    }
  }
  if (argc >= 5) {
    int32_t p = 0;
    // 上限 64 MB：够模拟流水线那 1.2 MB 的几倍，又不至于把内存吃爆。
    if (napi_get_value_int32(env, args[4], &p) == napi_ok && p >= 0 && p <= 65536) {
      job->benchPolluteKB = p;
    }
  }
  if (argc >= 6) {
    double s = 0;
    if (napi_get_value_double(env, args[5], &s) == napi_ok && s >= 0 && s <= 1000) {
      job->benchSpinMs = s;
    }
  }
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

/**
 * Bare-head recognition probe (T11): crop in, plate string out.
 *
 * Skips detect + rectify entirely — see LprRecogniseCrop's doc comment for why
 * that matters. Every accuracy number quoted before T11 was either host-side
 * onnxruntime (t6's 90.6%) or on-device-but-through-the-full-pipeline (T10's
 * 60.8%, where det re-finds the plate inside an already-cropped image).
 *
 * Args: (rgba, width, height, recId, [recSlot])
 *   recSlot >= 0 routes through the ncnn slot; otherwise the MS session.
 * Returns: ok=1;code=...;conf=...;recMs=...;backend=...  or  ok=0;error=...
 */
static napi_value RecogniseRun(napi_env env, napi_callback_info info) {
  size_t argc = 5;
  napi_value args[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  if (argc < 4) {
    return MakeString(env, "ok=0;error=recognise needs (rgba, width, height, recId)");
  }
  std::vector<uint8_t> rgba;
  if (!ReadArrayBufferArgU8(env, args[0], rgba)) {
    return MakeString(env, "ok=0;error=rgba ArrayBuffer is empty");
  }
  int w = 0;
  int h = 0;
  int recId = -1;
  int recSlot = -1;
  napi_get_value_int32(env, args[1], &w);
  napi_get_value_int32(env, args[2], &h);
  napi_get_value_int32(env, args[3], &recId);
  if (argc >= 5) napi_get_value_int32(env, args[4], &recSlot);

  MsSession* rec = nullptr;
  std::string backend;
  {
    std::lock_guard<std::mutex> lk(g_regMutex);
    if (recId < 0 || recId >= (int)g_sessions.size()) {
      return MakeString(env, "ok=0;error=bad rec session id");
    }
    rec = g_sessions[recId].s;
    backend = rec->backend;
  }
  if (rec == nullptr) {
    return MakeString(env, "ok=0;error=null rec session");
  }

  RgbaImage crop;
  crop.width = w;
  crop.height = h;
  crop.data = std::move(rgba);
  if (!crop.Valid()) {
    return MakeString(env, "ok=0;error=rgba size != w*h*4");
  }

  std::string code;
  std::string err;
  float conf = 0;
  std::vector<std::string> chars;
  std::vector<float> probs;
  const double t0 = NowMs();
  const bool ok = LprRecogniseCrop(crop, rec, recSlot, code, conf, chars, probs, err);
  const double recMs = NowMs() - t0;
  if (!ok) {
    LOGE("recognise failed: %{public}s", err.c_str());
    return MakeString(env, "ok=0;error=" + KvSanitize(err));
  }
  LOGI("recognise -> code=%{public}s conf=%{public}f ms=%{public}f", code.c_str(), conf, recMs);
  return MakeString(env, "ok=1;code=" + KvSanitize(code) + ";conf=" + std::to_string(conf) +
                             ";recMs=" + std::to_string(recMs) + ";backend=" + KvSanitize(backend));
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
      {"vehicleDetectAsync", nullptr, VehicleDetectAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"roiSelfTestAsync", nullptr, RoiSelfTestAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"roiPlateProbeAsync", nullptr, RoiPlateProbeAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"roiPipelineAsync", nullptr, RoiPipelineAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"roiDedupeSelfTestAsync", nullptr, RoiDedupeSelfTestAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
      {"roiOverlapSelfTestAsync", nullptr, RoiOverlapSelfTestAsync, nullptr, nullptr, nullptr, napi_default, nullptr},
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
      // T11 bare-head：跳过 det/rectify 直喂识别器，用于测识别器**真实**准确率
      // （此前所有准确率要么是主机 onnxruntime，要么是端侧但走完整流水线）。
      {"recognise", nullptr, RecogniseRun, nullptr, nullptr, nullptr, napi_default, nullptr},
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
