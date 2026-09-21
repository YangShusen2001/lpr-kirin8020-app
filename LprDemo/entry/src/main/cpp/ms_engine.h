#ifndef LPR_MS_ENGINE_H
#define LPR_MS_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

#include <mindspore/context.h>
#include <mindspore/data_type.h>
#include <mindspore/model.h>
#include <mindspore/status.h>
#include <mindspore/tensor.h>
#include <mindspore/types.h>

/**
 * One MindSpore Lite model session.
 *
 * Lifetime policy (deliberate, do NOT "fix" it):
 *   On Kirin NNRT the backend installs an NNRTDelegate; destroying a built model or
 *   its context after inference crashes the process (cppcrash, with a
 *   `~NNRTDelegate# Delete NNRTDelegate` ERROR as the herald). We therefore build
 *   once, keep everything resident, and never destroy. `retired` holds contexts and
 *   models of *failed* build attempts for the same reason.
 *   Process exit reclaims the memory.
 */
struct MsSession {
  std::vector<char> modelBytes;  // must outlive the model: MS Lite keeps a pointer

  OH_AI_ContextHandle ctx = nullptr;
  OH_AI_ModelHandle model = nullptr;

  std::string backend;  // "NNRT:HIAI_F" | "NNRT:NPU_..." | "GPU" | "KIRIN_NPU" | "CPU"
  /**
   * The backend string the caller asked for, kept verbatim so the log can state
   * "requested X, landed Y". `fallbackFrom` is non-empty only when the first attempt
   * in the plan did not build — i.e. the accelerator was silently NOT used.
   */
  std::string requested;
  std::string fallbackFrom;
  /**
   * Every backend that was tried, in order, with its verdict. The UI shows this so a
   * fallback is never silent: "GPU (fail) -> CPU (ok)" is the honest report.
   */
  std::string attemptLog;

  std::string inputName;
  std::string outputName;
  std::vector<int64_t> inputShape;
  std::vector<int64_t> outputShape;
  /**
   * MindSpore Lite reports the input shape in its own memory format. For a model
   * converted from an NCHW ONNX this comes back as NHWC [1,H,W,C] on the Lite CPU
   * backend, so the caller MUST pack the image to match instead of assuming NCHW.
   */
  OH_AI_Format inputFormat = OH_AI_FORMAT_NHWC;
  OH_AI_Format outputFormat = OH_AI_FORMAT_NHWC;
  OH_AI_DataType inputDtype = OH_AI_DATATYPE_UNKNOWN;
  OH_AI_DataType outputDtype = OH_AI_DATATYPE_UNKNOWN;
  size_t inputElems = 0;
  size_t outputElems = 0;
  /**
   * Tensor handle arrays, cached at load (A18 section 8). `OH_AI_ModelGetInputs`
   * / `GetOutputs` are called once per inference in the naive path; caching them
   * removes whatever that costs per frame. The handles stay valid for the model's
   * lifetime — the 30x benchmark itself reuses one fetched pair across all
   * iterations, which is the proof.
   */
  OH_AI_TensorHandleArray ins{};
  OH_AI_TensorHandleArray outs{};

  std::vector<OH_AI_ContextHandle> retiredCtx;
  std::vector<OH_AI_ModelHandle> retiredModel;

  /**
   * Tensor fingerprint of the LAST MsRun / MsRunMulti call (ADR-0003).
   *
   * `lastL2` is the L2 norm of the output buffer read according to the DECLARED
   * dtype; `lastL2AsFp16` re-reads the same bytes as fp16. The pair fingerprints
   * the known NNRT defect (an fp16 bitstream declared as fp32): then lastL2
   * explodes while lastL2AsFp16 lands on the CPU value.
   *
   * Why this exists: NPU utilisation is NOT readable on HarmonyOS (ADR-0003), so
   * the only honest evidence that an accelerator really computed something is the
   * output bytes' own statistics. Non-zero lastL2AsFp16 on an NPU row with 0.0000
   * on the CPU row is the landing evidence. It proves the backend RAN — not that
   * it ran CORRECTLY; correctness still needs the ground-truth set.
   */
  double lastL2 = 0;
  double lastL2AsFp16 = 0;
};

/** NNRT device names present on this device, priority-sorted (kirin device first). */
std::vector<std::string> MsNnrtCandidates();

/**
 * Build a session from in-memory .ms bytes.
 *
 * `backend` selects the acceleration strategy:
 *   "auto"       — NNRT(kirin) -> NNRT(HIAI) -> GPU(fp16) -> GPU(fp32) -> KIRIN_NPU -> CPU
 *   "nnrt"       — NNRT candidates (fp16, as MS Lite defaults) -> CPU
 *   "nnrt_fp32"  — NNRT candidates with EnableFP16 turned OFF -> CPU. Added because
 *                  every NPU figure in this project so far was measured with fp16
 *                  forced on; the "NPU drifts 0.08% so det cannot run there" verdict
 *                  has never been tested against an fp32 NPU build.
 *   "gpu"        — GPU(fp16) -> GPU(fp32) -> CPU
 *   "kirin"      — OH_AI_DEVICETYPE_KIRIN_NPU -> CPU
 *   "cpu"        — CPU only
 * Every strategy ends at CPU, so a non-null return always means "runnable"; read
 * `s->backend` to find out what it actually landed on, and `s->attemptLog` for the
 * full trail. Returns nullptr only when even the CPU build fails, and fills `err`.
 */
MsSession* MsLoad(const std::vector<char>& bytes, const std::string& backend, std::string& err);

/** Run inference. `in` must hold session->inputElems floats. */
bool MsRun(MsSession* s, const float* in, std::vector<float>& out, std::string& err);

/**
 * Run inference and return EVERY output as fp32 (the bare-head detector has three
 * [1,45,H,H] outputs; MsRun only ever reads the first). Also works for single-output
 * models (outs.size() == 1). Conversion matches MsRun: declared fp16 buffers are
 * widened per element.
 */
bool MsRunMulti(MsSession* s, const float* in,
                std::vector<std::vector<float>>& outs, std::string& err);

/**
 * One benchmark row. `checksum` is the L2 norm of the output buffer read according
 * to the *declared* dtype; `checksumAsFp16` re-reads the same bytes as fp16. The
 * pair is the fingerprint of the known NNRT defect (fp16 bitstream declared as
 * fp32): in that case checksum explodes and checksumAsFp16 lands on the CPU value.
 */
struct MsBench {
  bool ok = false;
  std::string error;
  std::string backend;   // what actually got built (may be CPU after a fallback)
  int warmup = 0;
  int repeat = 0;
  double mean = 0, p50 = 0, p95 = 0, minMs = 0, maxMs = 0;
  // Faithful-to-pipeline timing: memcpy input in + predict + copy outputs out,
  // all inside the timed loop. The pipeline (MsRunMulti) pays this marshaling
  // every frame while the pure-predict p50 above does not; the gap between the
  // two is exactly the I/O cost a per-model number hides (A18 section 8).
  double meanIo = 0, p50Io = 0;
  double checksum = 0;
  double checksumAsFp16 = 0;
  double maxAbs = 0;
  int outputDtype = 0;
  size_t outputElems = 0;
  // ---- 2026-09-21: 用于定位「隔离 7.4 ms vs 流水线 19.5 ms」的 2 倍差距 ----
  /** 迭代之间插入的睡眠（毫秒）。用于检验 DVFS：调用变稀疏是否掉频。 */
  double gapMs = 0;
  /** 迭代之间搬运的干扰缓冲大小（KB）。用于检验缓存/内存带宽污染。 */
  int polluteKB = 0;
  /**
   * 迭代之间**忙等**（自旋）的毫秒数 —— `gapMs` 的对照组。
   *
   * 为什么必须有这一档：`gapMs` 把 p50 从 7.4 推到 30.7 ms（单调剂量-反应），
   * 但那有两种解释：
   *   (a) **DVFS**：CPU 空闲 → 降频 → 推理跑在低频；
   *   (b) **时间本身**：某种与「两次调用相隔多久」有关的开销。
   * 自旋与睡眠的**经过时间相同**，但自旋是计算密集的，会把频率**顶住**。
   * 若自旋下 p50 回到 ~7.4 ms ⇒ 是频率 (a)，不是时间 (b)。
   * 这是本实验最关键的一刀：它把「相关」变成「因果」。
   */
  double spinMs = 0;
};

/**
 * Time `repeat` steady-state inferences on `s`, filling the input deterministically
 * inside C++ so every backend sees byte-identical data (the CPU-diff protocol).
 * Never throws; failures come back in `r.ok` / `r.error`.
 *
 * `gapMs` / `polluteKB`（2026-09-21 新增）默认 0 = 历史行为逐位不变。
 * 它们只改变**迭代之间**发生什么，不改变被计时的那次推理本身：
 *   - `gapMs > 0`：每次迭代前 sleep，模拟流水线「每帧一次、间隔 ~33 ms」的节奏
 *     ⇒ 若 p50 涨到 ~15–19 ms，则 DVFS/频率假设成立。
 *   - `polluteKB > 0`：每次迭代前 memcpy 一个该大小的干扰缓冲，模拟流水线里
 *     conv（1.2 MB RGBA 写）+ letterbox 对缓存/带宽的占用
 *     ⇒ 若 p50 涨到 ~15–19 ms，则缓存/内存污染假设成立。
 * 两者可同时给，用于观察是否叠加。
 */
MsBench MsBenchRun(MsSession* s, int warmup, int repeat, double gapMs = 0,
                   int polluteKB = 0, double spinMs = 0);

#endif  // LPR_MS_ENGINE_H
