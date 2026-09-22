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

/**
 * Bare-head recognition: crop in, plate string out. **No detect, no rectify.**
 *
 * Why this exists: every accuracy number this project has quoted so far
 * (t6's 90.6%, A16's 90.6%) was measured **on the host** with onnxruntime,
 * never on the device. The one on-device number we do have (T10's 60.8%) ran
 * crops through the *full* pipeline, where det has to re-find the plate inside
 * an already-cropped 94x24 image — a工况 that both distorts the input
 * (length-error rate 25.8% vs A16's measured 1.0% for rpv3) and conflates
 * detector error with recogniser error.
 *
 * This entry point skips det/rectify and feeds the crop straight to the
 * recogniser, which is exactly what t6/A16 measured. That makes the on-device
 * number directly comparable to the host number and isolates the recogniser's
 * own accuracy from the detector's.
 *
 * `recSlot >= 0` routes through the ncnn slot (GPU path); otherwise `rec`
 * (MindSpore Lite) is used. Returns false only on hard failure; a crop that
 * decodes to an empty string is a successful run.
 */
bool LprRecogniseCrop(const RgbaImage& crop, MsSession* rec, int recSlot,
                      std::string& code, float& conf,
                      std::vector<std::string>& chars, std::vector<float>& probs,
                      std::string& err);

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

// ---------------------------------------------------------------- 车辆检测（T2）
//
// 【为什么不是 ncnn】
// T2 票面写的是「一期走 ncnn(CPU)」。摸排后发现两件事，路线因此改了：
//   1. 本项目的 CPU 通路本来就是 **MindSpore Lite + .ms**（`LprSessions::det` 走
//      `y5fu_320x_head_fp32.ms`，且它本来就在 CPU）；ncnn 在本项目只服务
//      GPU/Vulkan 档（ADR-008）。为一期新引入 ncnn 会多一条并行通路要维护，
//      而不是"复用现成通路"。
//   2. `.ms` 转换被 DFL 的两个 Transpose 挡死（`Transform meta graph failed! ret=-500`），
//      这才是真正的障碍；它对 ncnn 也同样挡（`tools/scan_onnx.py` 判定
//      `perm != [0,1,3,2]` 不支持）。所以无论走哪条路，DFL 都得先改写。
// 于是实际路线 = **改写 DFL → 转 .ms → 走已有的 MS Lite CPU 通路**，与 `det=CPU`
// 完全一致，且 DFL 里的两个 Transpose 一并消失，T8 的 NPU perm 硬门也顺带清了。
// 改写脚本：`_veh/patch_dfl.py`（含 onnxruntime 数值自证，不过就不落盘）。

/** COCO 80 类名，下标即类号。车辆类见 `LprIsVehicleClass`。 */
const std::vector<std::string>& LprCocoNames();

/** COCO 里的车辆类：2=car, 3=motorcycle, 5=bus, 7=truck。 */
bool LprIsVehicleClass(int classId);

/** 一个车辆框。`rect` 与 `PlateResult::rect` 同样是**源图坐标**。 */
struct VehicleBox {
  float rect[4] = {0, 0, 0, 0};  // x1, y1, x2, y2 in source-image coordinates
  float score = 0;
  int classId = 0;
};

/**
 * 解码 YOLOv5u(ultralytics) 的 `output0 [1,84,2100]`。
 *
 * 这个输出**已经是解码过的**：图里最后一段 Slice/Sub/Add/Div/Concat 已经把
 * DFL 的 16-bin 加权和转成 xywh **像素坐标**（输入尺度），并乘过 stride；
 * `Sigmoid` 也已把 80 个类分数压到 [0,1]。所以这里**不做** anchor/stride 解码，
 * 只做「取框 → 按类 NMS → letterbox 反变换」。
 *
 * 布局是**通道优先**：`raw[c * anchors + a]`，c=0..3 是 cx,cy,w,h，c=4..83 是类分数。
 *
 * `r` / `left` / `top` 来自 `LprLetterBox`，用于把框还原到源图坐标。
 */
std::vector<VehicleBox> LprDecodeYolov5u(const std::vector<float>& raw, float confThresh,
                                         float iouThresh, float r, int left, int top,
                                         bool vehicleOnly, int maxBoxes, bool* outTruncated);

/**
 * 车辆检测端到端：letterbox -> encode -> infer -> decode -> 按类 NMS。
 *
 * `confThresh` 由调用方给（T2 默认 0.05，比车牌检测的 0.25 低得多 —— 车辆是大目标，
 * 低阈值是为了 T4 的「车框→车牌」不因为漏框而整段丢检；代价是框变多，靠 NMS 收）。
 * `vehicleOnly` 为真时只留 COCO 的 4 个车辆类。
 *
 * 检出数按分数降序，最多 `kMaxVehicleBoxes` 个（截断时 `outTruncated` 置真，
 * 调用方必须如实报告，不能假装那就是全部）。
 */
constexpr int kMaxVehicleBoxes = 100;

bool LprVehicleDetect(const RgbaImage& img, MsSession* det, float confThresh, float iouThresh,
                      bool vehicleOnly, std::vector<VehicleBox>& out, bool& outTruncated,
                      float& outInferMs, std::string& err);

/** 从会话的输入形状推出 letterbox 边长与需不需要 NHWC 布局。 */
bool LprDetectGeometryOf(const MsSession* det, int& outSize, bool& outNhwc, std::string& err);

// ---------------------------------------------------------------- ROI 裁剪（T3）
//
// 这一层是「车框 → 裁 ROI → 车牌检测 → 框映射回原图」的第一、三步。
//
// **裁剪放在 native 侧**（票面要求）：RGBA 是大 buffer（1080p 一帧 8.3 MB），
// 跨 ArkTS/native 边界反复拷贝会直接吃掉相机帧预算（33.3 ms）。放在这里还有一个
// 附带好处：`LprCropRoi` 是纯逐行 memcpy，无插值、无格式转换、无重采样，
// 像素与源图**逐字节相同** —— 这一点由 T3 自证里的逐字节比对钉住（不是"看着像"）。

/** ROI 外扩比例默认值。实测 0.15 与 0.40 的召回差异 <1%，取小者省像素。 */
constexpr float kRoiExpandDefault = 0.15f;

/**
 * 一个 ROI 的整数几何。`x0`/`y0` 是 ROI 左上角在**源图**中的像素坐标，
 * `w`/`h` 是 ROI 的像素尺寸（已 clamp 到图内）。
 *
 * `imgW`/`imgH` 一并记下来，是为了让 `ContainsBox` 能正确判断「ROI 是否覆盖车框」——
 * 车框本身可以超出图边界（YOLO 的框经常顶到边上），这时该裁的是**框与图的交集**，
 * 不是框本身。
 */
struct RoiRect {
  int x0 = 0, y0 = 0, w = 0, h = 0;
  int imgW = 0, imgH = 0;
  bool valid = false;    // false = 裁剪区域为空（零面积框 / 框完全在图外 / 图尺寸非法）
  bool clamped = false;  // 外扩被图边界削过 ⇒ ROI 比"理想外扩"小，日志要如实写
  float expand = 0;      // 请求的外扩比例（原样记下，便于从日志复现）

  /** ROI 是否覆盖 (车框 ∩ 图)。框与图无交时返回 true（空集被任何集合覆盖）。 */
  bool ContainsBox(const float box[4]) const;
};

/**
 * 由车框（源图坐标）算 ROI：按框**自身宽高**的 `expand` 倍向外扩，再 clamp 到图内。
 *
 * 取整方向是**向外**（左/上用 `floor`，右/下用 `ceil`）。向内取整会切掉车框边线上的
 * 像素，而车牌恰好经常贴着框的边线 —— 这是"少一个像素就丢一块牌"的地方。
 */
RoiRect LprRoiFromBox(const float box[4], int imgW, int imgH, float expand);

/** 从源图裁出 ROI。纯逐行 memcpy，像素与源图逐字节相同。 */
bool LprCropRoi(const RgbaImage& src, const RoiRect& roi, RgbaImage& out, std::string& err);

/**
 * ROI 局部坐标 → 源图坐标（就地）：`x += roi.x0`、`y += roi.y0`。
 *
 * **必须在去重与可视化之前调用**：`LprRunPipeline` 作用在 ROI 上，它返回的
 * `PlateResult::rect` 是 ROI **局部**坐标；直接拿去画框会整体偏移 `(x0, y0)`，
 * 表现为"检测是对的、位置是错的"（spec §六.3 说的就是这个坑）。
 */
void LprRoiMapRect(int rect[4], const RoiRect& roi);
void LprRoiMapRect(float rect[4], const RoiRect& roi);
void LprRoiMapRects(std::vector<VehicleBox>& boxes, const RoiRect& roi);

/** 源图坐标 → ROI 局部坐标（就地）：`LprRoiMapRect` 的逆。T4 用它把车框搬进 ROI 局部系。 */
void LprRoiUnmapRect(float rect[4], const RoiRect& roi);

/** RGB 之和（跳过 alpha）。与 `PlateResult::cropSum` **同口径**，数值可直接互相比对。 */
long long LprRgbSum(const RgbaImage& img);

/**
 * T3 单元级自证：用**构造函数造的已知车框**断言 ROI 几何、覆盖性、坐标映射与裁剪内容。
 * 返回逐行报告（`\n` 分隔，每行形如 `case=<名字>;ok=0/1;<细节>`），由调用方写 hilog。
 *
 * 为什么要有它：票面要求「有单元级验证：构造已知车辆框，断言映射后坐标正确」。
 * 期望值全部**手算后硬编码**在实现里（不是拿同一套公式再算一遍），所以公式一旦回归
 * 断言就会红。跑在真机上，因此这份报告同时是"设备上的 C++ 真的按这套规则算"的证据。
 *
 * `img` 用于"裁剪内容逐字节一致"与 `rgbSum` 这两条跨实现比对；传无效图时
 * 自动改用确定性图案合成一张 320x320，保证自证在任何情况下都能跑完。
 */
std::string LprRoiSelfTest(const RgbaImage& img);

/** CTC greedy decode over one [T] index row with its per-step probabilities. */
void LprCtcGreedy(const std::vector<int>& idx, const std::vector<float>& prob,
                  std::string& code, float& conf,
                  std::vector<std::string>& chars, std::vector<float>& probs);

#endif  // LPR_PIPELINE_H
