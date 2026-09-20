#ifndef LPR_PIPELINE_H
#define LPR_PIPELINE_H

/**
 * Native HyperLPR3 pipeline — C++ transcription of assets/js/pipeline.js.
 *
 * Why this file exists: the Web demo ran the whole pipeline in JS on top of
 * onnxruntime-web, which on Kirin 8020 can only ever use the CPU (ADR-004 §6.1).
 * The NPU is reachable *only* through MindSpore Lite, i.e. only from native code.
 * To put the recogniser on the NPU while the detector stays on the CPU, the whole
 * pipeline has to live here — the NAPI surface of ms_engine.cpp alone only runs
 * one model at a time.
 *
 * Fidelity contract: this is a *port*, not a reimplementation. Every numeric step
 * mirrors pipeline.js, which in turn mirrors the upstream Python/C++ algorithm:
 *   - resizeLinear   -> cv2.resize INTER_LINEAR, OpenCV's two-pass fixed point
 *                       (horizontal pass kept unshifted, one rounding in vertical)
 *   - warpPerspectiveCubic -> cv2.warpPerspective INTER_CUBIC, a = -0.75, BORDER_REPLICATE
 *   - channel order  -> detector wants RGB, recogniser/classifier want BGR
 *   - Uint8ClampedArray stores round-half-to-even, so ClampU8 uses nearbyint()
 * The acceptance test is: same image in, byte-identical crop checksum and
 * character-identical plate code out, versus the browser path.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "ms_engine.h"

/** RGBA8 image. The only image format the pipeline accepts. */
struct RgbaImage {
  std::vector<uint8_t> data;  // width * height * 4, RGBA
  int width = 0;
  int height = 0;

  bool Valid() const {
    return width > 0 && height > 0 &&
           data.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  }
};

/** One plate: detection box + rectified crop + recognition + classification. */
struct PlateResult {
  int rect[4] = {0, 0, 0, 0};  // x1, y1, x2, y2 in source-image coordinates
  float detScore = 0;
  int layer = 0;  // 0 = single, 1 = double (DOUBLE_LAYER)

  std::string code;
  float recConf = 0;
  std::vector<std::string> chars;
  std::vector<float> charProbs;

  int cropH = 0;
  int cropW = 0;
  /** RGB-only sum of the rectified crop — the cross-implementation fingerprint. */
  long long cropSum = 0;

  /**
   * 牌色：由**像素测量**判定，不再用分类模型（ADR-0005，2026-09-21）。
   *
   * 取值 "blue" / "green" / "yellow" / "unknown"。
   *
   * ⚠️ 这里曾是 `float cls[3]` 且注释写 `// yellow, blue, green` —— **那条注释是错的**。
   * 实测正确顺序是 `blue=0, green=1, yellow=2`（肉眼核实四个样本 + 1000 张真实集
   * index 0 占 982）。旧表是它的一个旋转，导致绿牌被解码成"蓝牌"、蓝牌被解码成"黄牌"，
   * 这正是 lpr-showcase ADR-015 那个"未裁决蓝/绿冲突"的真正成因 —— 不是分类器与像素
   * 测量的分歧，而是一张错表造出来的假冲突。
   *
   * 废弃分类模型的第二个理由：它占 2.69 ms/帧（7.4%），而像素测量约 0.1 ms，
   * 且判色是附带属性、不进入识别主链。
   */
  std::string colour = "unknown";
  /** 像素判色的置信度：占优色带在饱和像素中的占比。低置信度时应输出 "unknown"。 */
  float colourConfidence = 0;

  float tDetectMs = 0;
  float tLetterboxMs = 0;    // letterbox 单独计时
  float tEncodeInferMs = 0;  // encode + infer 合并计时
  float tPackMs = 0;         // encode 单独计时（NHWC/NCHW 打包）—— 2026-09-20 拆分
  float tInferMs = 0;        // infer 单独计时（纯推理）—— 2026-09-20 拆分
  float tDecodeNmsMs = 0;    // decode + NMS 合并计时
  float tRectifyMs = 0;
  float tRecogMs = 0;
  float tClsMs = 0;
};

/** The three sessions, each built on whatever backend it actually landed on. */
struct LprSessions {
  MsSession* det = nullptr;  // y5fu_320x_sim      -> CPU (NPU rejected, ADR-004 §4.1)
  MsSession* rec = nullptr;  // rpv3_mdict_160_r3  -> NPU (2.2-3.0x)
  MsSession* cls = nullptr;  // litemodel_cls_96x  -> CPU (NPU slower on small maps)
  bool detNcnn = false;  // diagnostic path; default MS detector unchanged
  bool detVulkan = false;
  bool useYoloV8 = false;  // Use YOLOv8 instead of YOLOv5-face (default: false)
  /**
   * 识别 / 分类的 ncnn 槽位（-1 = 用上面的 MS 会话）。
   *
   * GPU（Vulkan）在麒麟 8020 上只有 ncnn 一条通路（ADR-008），而 MS Lite 的 GPU 档
   * 是编译期判否（`IsValid# GPU is not supported`）。要让三模型都能吃 GPU，
   * 识别与分类也得有 ncnn 版本 —— 槽位号由调用方分配：1 = 识别，2 = 分类。
   */
  int recSlot = -1;
  int clsSlot = -1;
  int detSize = 320;  // 320 for YOLOv5, 640 for YOLOv8
  float confThresh = 0.25f;
  float iouThresh = 0.5f;
};

/**
 * End-to-end: letterbox -> detect -> NMS -> rectify -> recognise -> classify.
 *
 * Returns false only on a hard failure (bad image, missing session, inference
 * error); "no plate found" is a successful run with an empty `out`.
 * `out` is sorted by detector score, best first.
 */
bool LprRunPipeline(const RgbaImage& img, const LprSessions& s,
                    std::vector<PlateResult>& out, std::string& err);

// ---------------------------------------------------------------- test surface
// Exposed so the port can be diffed against pipeline.js function by function
// instead of only end to end. See tools/harmony/ for the diff harness.

/** Letterboxed square image plus the geometry needed to map boxes back. */
struct LetterBoxed {
  RgbaImage img;
  float r = 1;
  int left = 0;
  int top = 0;
};

LetterBoxed LprLetterBox(const RgbaImage& src, int size);

/** NCHW float tensor; `swapRB` true = detector (BGR->RGB), false = recogniser. */
std::vector<float> LprToNchw(const RgbaImage& img, bool swapRB);

/** NHWC float tensor — MindSpore Lite's CPU backend reports this layout. */
std::vector<float> LprToNhwc(const RgbaImage& img, bool swapRB);

/** 14-column rows [x1,y1,x2,y2,score,kp0x..kp3y,layer], padding undone. */
std::vector<std::vector<float>> LprDecodeDetections(const std::vector<float>& raw, int rows,
                                                    float confThresh, float iouThresh,
                                                    float r, int left, int top);

/**
 * Bare-head decode (ADR-006 §5): the three rank-4 head tensors [1,45,40,40] /
 * [1,45,20,20] / [1,45,10,10] (onnx output order) become the same [6300,15] row
 * blob the ORIGINAL in-graph decode produced. Verified element-wise against the
 * original model by tools/verify_head_decode.py (maxAbsDiff=0.000061, MATCH).
 * Anchors are hardcoded (graph constants 1005/1118/1231); grids are generated.
 */
std::vector<float> LprDecodeBareHead(const std::vector<std::vector<float>>& heads);

/** Rectify the quad to an axis-aligned crop (rotating 90 deg if portrait). */
bool LprRotateCrop(const RgbaImage& src, const int marks[4][2], RgbaImage& out);

/**
 * Recognition input: aspect-preserving resize, [-1,1], zero right-padding.
 *
 * `nhwc` selects the memory layout and it is NOT cosmetic: MindSpore Lite reports
 * [1,48,160,3] (pixel-interleaved) for this model while ONNX declares [1,3,48,160]
 * (planar). Feeding planar bytes to a NHWC tensor interleaves the three channels
 * into nonsense and the CTC head then decodes garbage.
 */
std::vector<float> LprEncodePlate(const RgbaImage& crop, int imgH, int imgW,
                                  int limitedMaxWidth, int limitedMinWidth, int& outW,
                                  bool nhwc);

/**
 * 牌色判定（ADR-0005，2026-09-21）：**像素测量，不用分类模型**。
 *
 * 移植自 lpr-showcase 的 `tools/plate_face_colour.py`。判据是"牌面主导饱和色"：
 *   1. 裁掉边框（上下各 12%、左右各 8%），只留牌面；
 *   2. 丢掉近白（字）与近黑（影）像素，只留被饱和涂装的像素；
 *   3. 统计色相落在三个色带里的占比：
 *        green  35..95   （新能源）
 *        blue  100..135  （普通）
 *        yellow 15..34   （大型车 / 出租）
 *   4. 占比最高者胜出，但**必须 ≥ 0.55**，否则判 "unknown"。
 *
 * `outConfidence` 是胜出色带的占比；低置信度时调用方应展示 "unknown" 而非猜测。
 *
 * 为什么不用分类模型：它占 2.69 ms/帧（7.4%）而本函数约 0.1 ms；且实测其标签表
 * 是旋转的（见 PlateResult::colour 的注释）。判色是附带属性，不进入识别主链。
 */
std::string LprPlateColour(const RgbaImage& crop, float& outConfidence);

/** Classification input: square resize, [0,1], BGR. Same `nhwc` caveat. */
std::vector<float> LprEncodeClassify(const RgbaImage& crop, int size, bool nhwc);

/**
 * NV21 (YUV_420_SP) -> RGBA，可顺带做 90 度整数倍旋转。
 *
 * 存在的理由：相机预览档全是 YUV_420_SP，而 pipeline 只吃 RGBA。原先这条转换
 * 走的是 ArkTS 侧「逐行 copy + createPixelMap(srcPixelFormat=NV21) + rotate() +
 * readPixelsToBuffer」四步，实测在麒麟 8020 上要 21-38 ms —— 比整个推理还贵
 * （NPU 档推理 10 ms）。搬到 C++ 里就是一趟循环，省掉 PixelMap 的建立与旋转。
 *
 * `stride` 是 Y 平面每行的字节数（常大于 width，有 padding）。
 * `rotation` 取 0/90/180/270（顺时针），非 0 时输出宽高互换。
 *
 * NV21 的内存布局：Y 平面 height 行 × stride，随后是交错的 VU 平面 height/2 行 × stride。
 * 色度是 2x2 下采样，所以每 2x2 像素共用一个 (V,U) 对。
 */
bool LprNv21ToRgba(const uint8_t* nv21, size_t nv21Size, int width, int height, int stride,
                   int rotation, RgbaImage& out);

/** Character set, index 0 = CTC blank. 44 entries. */
const std::vector<std::string>& LprToken();

/** CTC greedy decode over one [T] index row with its per-step probabilities. */
void LprCtcGreedy(const std::vector<int>& idx, const std::vector<float>& prob,
                  std::string& code, float& conf,
                  std::vector<std::string>& chars, std::vector<float>& probs);

#endif  // LPR_PIPELINE_H
