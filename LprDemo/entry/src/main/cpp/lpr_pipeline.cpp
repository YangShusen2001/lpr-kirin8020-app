/**
 * Native HyperLPR3 pipeline — see lpr_pipeline.h for the fidelity contract.
 *
 * This is a port of assets/js/pipeline.js. Function names and control flow are kept
 * parallel on purpose so a reviewer can diff the two side by side.
 */

#include "lpr_pipeline.h"
#include "ncnn_engine.h"
#include "yolov8_detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <hilog/log.h>

#define PIPE_TRACE(...) OH_LOG_Print(LOG_APP, LOG_INFO, 0xD001, "LprStages", __VA_ARGS__)

/**
 * Reusable scratch for ResizeLinearInto / the pipeline (A18 §1).
 *
 * Why this exists: the pipeline used to allocate a fresh 5.5 MB int32
 * intermediate (`tmp`: dstW x srcH x 4) on every frame for the 1920x1080 ->
 * 320x320 letterbox, plus ~1.2 MB for the detector input pack. Those
 * allocations' first-touch page faults measured as ~7 ms of "inter-stage"
 * overhead per frame. Reusing the buffers removes the allocation and the
 * faults; the arithmetic is unchanged.
 *
 * Lives at global scope (not inside the anonymous namespace) because the
 * scratch-filling helpers below are shared by the pipeline inside the
 * namespace and the public wrappers outside it.
 */
struct ResizeScratch {
  std::vector<int> xo;
  std::vector<int> cx;
  std::vector<int> yo;
  std::vector<int> cy;
  std::vector<int32_t> tmp;
  RgbaImage img;  // resize destination
};

// Internal: encode into caller-owned scratch (A18 §1). Declared at global
// scope so both the pipeline (inside the anonymous namespace) and the public
// wrappers further down resolve to these same entities.
static void LprEncodePlateInto(const RgbaImage& crop, int imgH, int imgW,
                               int limitedMaxWidth, int limitedMinWidth, int& outW,
                               bool nhwc, std::vector<float>& out, ResizeScratch& sc);
static void LprEncodeClassifyInto(const RgbaImage& crop, int size, bool nhwc,
                                  std::vector<float>& out, ResizeScratch& sc);

namespace {

using Clock = std::chrono::steady_clock;

inline double NowMs() {
  return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

/**
 * Uint8ClampedArray store semantics: clamp to [0,255] with round-half-to-even.
 * std::nearbyint honours the default FE_TONEAREST mode, which is exactly that.
 * Using lround() instead would differ by one level on exact .5 ties.
 */
inline uint8_t ClampU8(float v) {
  if (!(v > 0.0f)) {
    return 0;  // also catches NaN
  }
  if (v >= 255.0f) {
    return 255;
  }
  const float r = std::nearbyint(v);
  if (r < 0.0f) {
    return 0;
  }
  if (r > 255.0f) {
    return 255;
  }
  return static_cast<uint8_t>(r);
}

/** OpenCV INTER_RESIZE_COEF_BITS — resize coefficients are 11-bit fixed point. */
constexpr int kCoefBits = 11;
constexpr int kCoefScale = 1 << kCoefBits;

/**
 * cv2.resize(..., INTER_LINEAR) bit for bit.
 *
 * The horizontal pass writes an *unshifted* integer intermediate and the vertical
 * pass does the single rounding — see the note in pipeline.js. Rounding each pass
 * separately shifts ~6% of pixels by one level, which is enough to flip a character.
 */
static void ResizeLinearInto(const RgbaImage& src, int dstW, int dstH, RgbaImage& out,
                             ResizeScratch& sc) {
  out.width = dstW;
  out.height = dstH;
  if (dstW == src.width && dstH == src.height) {
    out.data = src.data;
    return;
  }
  if (dstW <= 0 || dstH <= 0 || !src.Valid()) {
    out.data.clear();
    return;
  }
  const int sw = src.width;
  const int sh = src.height;

  sc.xo.resize(dstW);
  sc.cx.resize(static_cast<size_t>(dstW) * 2);
  for (int x = 0; x < dstW; x++) {
    double fx = (static_cast<double>(x) + 0.5) * sw / dstW - 0.5;
    int sx = static_cast<int>(std::floor(fx));
    fx -= sx;
    if (sx < 0) {
      fx = 0;
      sx = 0;
    }
    if (sx >= sw - 1) {
      fx = 0;
      sx = sw - 1;
    }
    sc.xo[x] = sx;
    sc.cx[x * 2] = static_cast<int>(std::lround((1.0 - fx) * kCoefScale));
    sc.cx[x * 2 + 1] = static_cast<int>(std::lround(fx * kCoefScale));
  }

  sc.yo.resize(dstH);
  sc.cy.resize(static_cast<size_t>(dstH) * 2);
  for (int y = 0; y < dstH; y++) {
    double fy = (static_cast<double>(y) + 0.5) * sh / dstH - 0.5;
    int sy = static_cast<int>(std::floor(fy));
    fy -= sy;
    if (sy < 0) {
      fy = 0;
      sy = 0;
    }
    if (sy >= sh - 1) {
      fy = 0;
      sy = sh - 1;
    }
    sc.yo[y] = sy;
    sc.cy[y * 2] = static_cast<int>(std::lround((1.0 - fy) * kCoefScale));
    sc.cy[y * 2 + 1] = static_cast<int>(std::lround(fy * kCoefScale));
  }

  // Horizontal pass: integer intermediate, deliberately left unscaled.
  sc.tmp.resize(static_cast<size_t>(dstW) * sh * 4);
  const int stride = dstW * 4;
  for (int y = 0; y < sh; y++) {
    const uint8_t* row = src.data.data() + static_cast<size_t>(y) * sw * 4;
    int32_t* orow = sc.tmp.data() + static_cast<size_t>(y) * dstW * 4;
    for (int x = 0; x < dstW; x++) {
      // The +1 tap is clamped so a 1-pixel axis cannot read past the buffer; when
      // clamping kicks in the matching coefficient is 0 anyway.
      const uint8_t* s0 = row + static_cast<size_t>(sc.xo[x]) * 4;
      const uint8_t* s1 = row + static_cast<size_t>(std::min(sc.xo[x] + 1, sw - 1)) * 4;
      const int c0 = sc.cx[x * 2];
      const int c1 = sc.cx[x * 2 + 1];
      int32_t* o = orow + static_cast<size_t>(x) * 4;
      o[0] = s0[0] * c0 + s1[0] * c1;
      o[1] = s0[1] * c0 + s1[1] * c1;
      o[2] = s0[2] * c0 + s1[2] * c1;
      o[3] = s0[3] * c0 + s1[3] * c1;
    }
  }

  // Vertical pass: single rounding shift of 2 * kCoefBits. Done in 64-bit: the
  // worst-case intermediate (1,044,480 * 2048) grazes the int32 ceiling, and JS
  // computes it in double anyway.
  out.data.assign(static_cast<size_t>(dstW) * dstH * 4, 0);
  const int64_t round = int64_t(1) << (kCoefBits * 2 - 1);
  for (int y = 0; y < dstH; y++) {
    const int32_t* r0 = sc.tmp.data() + static_cast<size_t>(sc.yo[y]) * stride;
    const int32_t* r1 = sc.tmp.data() + static_cast<size_t>(std::min(sc.yo[y] + 1, sh - 1)) * stride;
    const int c0 = sc.cy[y * 2];
    const int c1 = sc.cy[y * 2 + 1];
    uint8_t* o = out.data.data() + static_cast<size_t>(y) * stride;
    for (int i = 0; i < stride; i++) {
      const int64_t v = (static_cast<int64_t>(r0[i]) * c0 +
                         static_cast<int64_t>(r1[i]) * c1 + round) >> (kCoefBits * 2);
      o[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
  }
}

RgbaImage ResizeLinear(const RgbaImage& src, int dstW, int dstH) {
  ResizeScratch sc;
  RgbaImage out;
  ResizeLinearInto(src, dstW, dstH, out, sc);
  return out;
}

float Iou(const std::vector<float>& a, const std::vector<float>& b) {
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float iw = std::max(0.0f, x2 - x1);
  const float ih = std::max(0.0f, y2 - y1);
  const float inter = iw * ih;
  const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const float areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const float uni = areaA + areaB - inter;
  return uni <= 0 ? 0.0f : inter / uni;
}

/** Gaussian elimination with partial pivoting on an 8x8 system. */
void Solve8(std::vector<std::vector<double>> m, const std::vector<double>& b,
            std::vector<double>& x) {
  const int n = 8;
  for (int i = 0; i < n; i++) {
    m[i].push_back(b[i]);
  }
  for (int col = 0; col < n; col++) {
    int piv = col;
    for (int r = col + 1; r < n; r++) {
      if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) {
        piv = r;
      }
    }
    if (piv != col) {
      std::swap(m[piv], m[col]);
    }
    const double d = m[col][col];
    if (std::fabs(d) < 1e-12) {
      continue;
    }
    for (int r = col + 1; r < n; r++) {
      const double f = m[r][col] / d;
      if (f == 0) {
        continue;
      }
      for (int c = col; c <= n; c++) {
        m[r][c] -= f * m[col][c];
      }
    }
  }
  x.assign(n, 0.0);
  for (int r = n - 1; r >= 0; r--) {
    double s = m[r][n];
    for (int c = r + 1; c < n; c++) {
      s -= m[r][c] * x[c];
    }
    x[r] = std::fabs(m[r][r]) < 1e-12 ? 0.0 : s / m[r][r];
  }
}

/** Homography mapping src -> dst, as a flat 9-vector with h33 = 1. */
void GetPerspectiveTransform(const double src[4][2], const double dst[4][2], double h[9]) {
  std::vector<std::vector<double>> a;
  std::vector<double> b;
  for (int i = 0; i < 4; i++) {
    const double sx = src[i][0];
    const double sy = src[i][1];
    const double dx = dst[i][0];
    const double dy = dst[i][1];
    a.push_back({sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy});
    a.push_back({0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy});
    b.push_back(dx);
    b.push_back(dy);
  }
  std::vector<double> x;
  Solve8(a, b, x);
  for (int i = 0; i < 8; i++) {
    h[i] = x[i];
  }
  h[8] = 1.0;
}

void Invert3x3(const double m[9], double out[9]) {
  const double a = m[0], b = m[1], c = m[2];
  const double d = m[3], e = m[4], f = m[5];
  const double g = m[6], h = m[7], i = m[8];
  const double A = e * i - f * h;
  const double B = -(d * i - f * g);
  const double C = d * h - e * g;
  const double det = a * A + b * B + c * C;
  if (std::fabs(det) < 1e-12) {
    const double ident[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::memcpy(out, ident, sizeof(ident));
    return;
  }
  out[0] = A / det;
  out[1] = -(b * i - c * h) / det;
  out[2] = (b * f - c * e) / det;
  out[3] = B / det;
  out[4] = (a * i - c * g) / det;
  out[5] = -(a * f - c * d) / det;
  out[6] = C / det;
  out[7] = -(a * h - b * g) / det;
  out[8] = (a * e - b * d) / det;
}

/** Bicubic kernel with a = -0.75, matching OpenCV's INTER_CUBIC. */
inline double CubicWeight(double t) {
  const double a = -0.75;
  const double x = std::fabs(t);
  if (x <= 1) {
    return ((a + 2) * x - (a + 3)) * x * x + 1;
  }
  if (x < 2) {
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  }
  return 0;
}

/**
 * cv2.warpPerspective(..., INTER_CUBIC, BORDER_REPLICATE) driven by the inverse
 * homography. The 4x4 tap window is renormalised by the accumulated weight, which
 * is what makes BORDER_REPLICATE behave at the edges.
 */
RgbaImage WarpPerspectiveCubic(const RgbaImage& src, const double hInv[9], int outW, int outH) {
  RgbaImage out;
  out.width = outW;
  out.height = outH;
  out.data.assign(static_cast<size_t>(outW) * outH * 4, 0);
  const int sw = src.width;
  const int sh = src.height;
  const double h11 = hInv[0], h12 = hInv[1], h13 = hInv[2];
  const double h21 = hInv[3], h22 = hInv[4], h23 = hInv[5];
  const double h31 = hInv[6], h32 = hInv[7], h33 = hInv[8];

  for (int y = 0; y < outH; y++) {
    for (int x = 0; x < outW; x++) {
      const double dz = h31 * x + h32 * y + h33;
      const double sx = (h11 * x + h12 * y + h13) / dz;
      const double sy = (h21 * x + h22 * y + h23) / dz;
      const int ix = static_cast<int>(std::floor(sx));
      const int iy = static_cast<int>(std::floor(sy));
      const size_t o = (static_cast<size_t>(y) * outW + x) * 4;

      for (int c = 0; c < 4; c++) {
        double acc = 0;
        double wsum = 0;
        for (int m = -1; m <= 2; m++) {
          const double wy = CubicWeight(sy - (iy + m));
          if (wy == 0) {
            continue;
          }
          int py = iy + m;
          if (py < 0) {
            py = 0;
          } else if (py > sh - 1) {
            py = sh - 1;
          }
          for (int n = -1; n <= 2; n++) {
            const double wx = CubicWeight(sx - (ix + n));
            if (wx == 0) {
              continue;
            }
            int px = ix + n;
            if (px < 0) {
              px = 0;
            } else if (px > sw - 1) {
              px = sw - 1;
            }
            const double wgt = wx * wy;
            acc += src.data[(static_cast<size_t>(py) * sw + px) * 4 + c] * wgt;
            wsum += wgt;
          }
        }
        out.data[o + c] = ClampU8(static_cast<float>(wsum == 0 ? 0.0 : acc / wsum));
      }
      out.data[o + 3] = 255;
    }
  }
  return out;
}

/** np.rot90 on an RGBA buffer (counter-clockwise). */
RgbaImage Rotate90(const RgbaImage& img) {
  const int w = img.width;
  const int h = img.height;
  RgbaImage out;
  out.width = h;
  out.height = w;
  out.data.assign(static_cast<size_t>(w) * h * 4, 0);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const int ny = w - 1 - x;
      const int nx = y;
      const size_t s = (static_cast<size_t>(y) * w + x) * 4;
      const size_t o = (static_cast<size_t>(ny) * out.width + nx) * 4;
      out.data[o] = img.data[s];
      out.data[o + 1] = img.data[s + 1];
      out.data[o + 2] = img.data[s + 2];
      out.data[o + 3] = img.data[s + 3];
    }
  }
  return out;
}

/** Contiguous copy of rows [y0, y1) — numpy's crop[y0:y1, :]. */
RgbaImage SliceRows(const RgbaImage& img, int y0, int y1) {
  RgbaImage out;
  const int a = std::max(0, y0);
  const int b = std::min(img.height, y1);
  const int hh = std::max(0, b - a);
  out.width = img.width;
  out.height = hh;
  out.data.assign(static_cast<size_t>(hh) * img.width * 4, 0);
  if (hh > 0) {
    std::memcpy(out.data.data(), img.data.data() + static_cast<size_t>(a) * img.width * 4,
                static_cast<size_t>(hh) * img.width * 4);
  }
  return out;
}

/** Argmax over the class axis of a [T, C] logit block. */
void ArgmaxRows(const std::vector<float>& logits, int T, int C,
                std::vector<int>& idx, std::vector<float>& prob) {
  idx.assign(T, 0);
  prob.assign(T, 0.0f);
  for (int t = 0; t < T; t++) {
    int best = 0;
    float bv = -INFINITY;
    const size_t base = static_cast<size_t>(t) * C;
    for (int c = 0; c < C; c++) {
      const float v = logits[base + c];
      if (v > bv) {
        bv = v;
        best = c;
      }
    }
    idx[t] = best;
    prob[t] = bv;
  }
}

/**
 * Single-plate recognition: encode -> run -> argmax -> CTC greedy.
 *
 * `recSlot >= 0` 时改走 ncnn 槽位（GPU 唯一通路），否则用 `rec`（MindSpore Lite）。
 * 两条路的输入编码**必须同源**，差别只在内存布局：MS Lite 报 NHWC，ncnn 固定吃 NCHW。
 */
bool Recognise(MsSession* rec, int recSlot, const RgbaImage& crop, std::string& code,
               float& conf, std::vector<std::string>& chars, std::vector<float>& probs,
               std::string& err, std::vector<float>& encScratch,
               std::vector<float>& logitsScratch, ResizeScratch& sc) {
  int encW = 0;
  // Layout follows the runner, not the ONNX graph: MS Lite reports NHWC here,
  // while ncnn's Mat is planar (NCHW).
  const bool nhwc = recSlot >= 0 ? false : (rec->inputFormat == OH_AI_FORMAT_NHWC);
  LprEncodePlateInto(crop, 48, 160, 160, 48, encW, nhwc, encScratch, sc);
  std::vector<float>& logits = logitsScratch;
  std::string shapeKv;
  if (recSlot >= 0) {
    if (!NcnnRunSlot(recSlot, encScratch.data(), 3, 48, 160, logits, shapeKv, err)) {
      return false;
    }
  } else if (!MsRun(rec, encScratch.data(), logits, err)) {
    return false;
  }

  // The class axis MUST come from the model, never from the token table.
  // rpv3_mdict_160_r3 emits [1, 20, 78] — 78 classes — while the charset carried
  // over from pipeline.js has 77 entries. Deriving C from the table mis-slices the
  // logit block by one column per step and decodes into garbage (observed: a
  // 19-character "plate" from a 7-character one).
  int T = 0;
  int C = 0;
  if (recSlot >= 0) {
    // ncnn 侧不自省形状（ncnn::Mat 的 dims 会随优化层变化），改从**实际元素数**
    // 反推：步数 T 是该模型的固定输出（ONNX 声明 [1,20,78]），C = n / T ——
    // 这样 C 仍然来自模型，不会退化成字符表的 77。
    constexpr int kRecT = 20;
    T = kRecT;
    if (logits.size() % static_cast<size_t>(T) == 0) {
      C = static_cast<int>(logits.size() / static_cast<size_t>(T));
    }
  } else if (rec->outputShape.size() == 3) {
    T = static_cast<int>(rec->outputShape[1]);
    C = static_cast<int>(rec->outputShape[2]);
  }
  if (T <= 0 || C <= 0 || static_cast<size_t>(T) * C != logits.size()) {
    // Shape unusable: fall back to the charset size, but say so — this path is the
    // one that decodes wrong, so it must not be silent.
    C = static_cast<int>(LprToken().size());
    T = C > 0 ? static_cast<int>(logits.size() / C) : 0;
    if (T <= 0) {
      err = "recogniser output unusable: elems=" + std::to_string(logits.size());
      return false;
    }
    err.clear();
  }

  std::vector<int> idx;
  std::vector<float> prob;
  ArgmaxRows(logits, T, C, idx, prob);
  LprCtcGreedy(idx, prob, code, conf, chars, probs);
  return true;
}

}  // namespace

// ---------------------------------------------------------------- public API

bool LprRecogniseCrop(const RgbaImage& crop, MsSession* rec, int recSlot,
                      std::string& code, float& conf,
                      std::vector<std::string>& chars, std::vector<float>& probs,
                      std::string& err) {
  if (rec == nullptr && recSlot < 0) {
    err = "missing rec session";
    return false;
  }
  if (!crop.Valid()) {
    err = "invalid crop";
    return false;
  }
  // Scratch is per-call and small (a few index vectors + one resize dest).
  // A batch caller that cares about the allocation can keep its own and call
  // the internal Recognise directly; this entry point is for the probe path
  // where clarity beats the last microsecond.
  ResizeScratch sc;
  std::vector<float> enc;
  std::vector<float> logits;
  return Recognise(rec, recSlot, crop, code, conf, chars, probs, err, enc, logits, sc);
}

const std::vector<std::string>& LprToken() {
  // Must stay index-aligned with TOKEN in assets/js/pipeline.js. Index 0 is the
  // CTC blank; indices 1..44 are the 44 real classes.
  static const std::vector<std::string> kToken = {
      "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
      "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "O", "P",
      "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
      "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民", "沪",
      "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏", "蒙", "藏",
      "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁", "黑", "领", "使", "澳",
  };
  return kToken;
}

/**
 * Letterbox into caller-owned buffers (A18 §1: the 320x320 destination and the
 * resize scratch are reused across frames instead of reallocated).
 */
static void LprLetterBoxInto(const RgbaImage& src, int size, LetterBoxed& lb,
                             ResizeScratch& sc) {
  const int h = src.height;
  const int w = src.width;
  const double r = std::min(static_cast<double>(size) / h, static_cast<double>(size) / w);
  const int newH = static_cast<int>(std::trunc(h * r));
  const int newW = static_cast<int>(std::trunc(w * r));
  const int top = static_cast<int>(std::trunc((size - newH) / 2.0));
  const int left = static_cast<int>(std::trunc((size - newW) / 2.0));

  ResizeLinearInto(src, newW, newH, sc.img, sc);
  lb.img.width = size;
  lb.img.height = size;
  lb.img.data.assign(static_cast<size_t>(size) * size * 4, 0);  // black border
  for (int y = 0; y < newH; y++) {
    const size_t srcOff = static_cast<size_t>(y) * newW * 4;
    const size_t dstOff = (static_cast<size_t>(y + top) * size + left) * 4;
    std::memcpy(lb.img.data.data() + dstOff, sc.img.data.data() + srcOff,
                static_cast<size_t>(newW) * 4);
  }
  lb.r = static_cast<float>(r);
  lb.left = left;
  lb.top = top;
}

LetterBoxed LprLetterBox(const RgbaImage& src, int size) {
  LetterBoxed lb;
  ResizeScratch sc;
  LprLetterBoxInto(src, size, lb, sc);
  return lb;
}

static void LprToNchwInto(const RgbaImage& img, bool swapRB, std::vector<float>& out) {
  const int w = img.width;
  const int h = img.height;
  const size_t plane = static_cast<size_t>(w) * h;
  out.resize(3 * plane);
  for (size_t i = 0, p = 0; i < plane; i++, p += 4) {
    const uint8_t r = img.data[p];
    const uint8_t g = img.data[p + 1];
    const uint8_t b = img.data[p + 2];
    out[i] = (swapRB ? r : b) / 255.0f;
    out[plane + i] = g / 255.0f;
    out[2 * plane + i] = (swapRB ? b : r) / 255.0f;
  }
}

std::vector<float> LprToNchw(const RgbaImage& img, bool swapRB) {
  std::vector<float> out;
  LprToNchwInto(img, swapRB, out);
  return out;
}

static void LprToNhwcInto(const RgbaImage& img, bool swapRB, std::vector<float>& out) {
  const size_t plane = static_cast<size_t>(img.width) * img.height;
  out.resize(3 * plane);
  for (size_t i = 0, p = 0; i < plane; i++, p += 4) {
    const uint8_t r = img.data[p];
    const uint8_t g = img.data[p + 1];
    const uint8_t b = img.data[p + 2];
    out[i * 3] = (swapRB ? r : b) / 255.0f;
    out[i * 3 + 1] = g / 255.0f;
    out[i * 3 + 2] = (swapRB ? b : r) / 255.0f;
  }
}

std::vector<float> LprToNhwc(const RgbaImage& img, bool swapRB) {
  std::vector<float> out;
  LprToNhwcInto(img, swapRB, out);
  return out;
}

std::vector<std::vector<float>> LprDecodeDetections(const std::vector<float>& raw, int rows,
                                                    float confThresh, float iouThresh,
                                                    float r, int left, int top) {
  std::vector<std::vector<float>> cand;
  for (int i = 0; i < rows; i++) {
    const size_t o = static_cast<size_t>(i) * 15;
    if (o + 14 >= raw.size()) {
      break;
    }
    const float obj = raw[o + 4];
    if (!(obj > confThresh)) {
      continue;
    }
    const float s0 = raw[o + 13] * obj;
    const float s1 = raw[o + 14] * obj;
    const float score = std::max(s0, s1);
    const int layer = (s1 > s0) ? 1 : 0;
    const float cx = raw[o];
    const float cy = raw[o + 1];
    const float bw = raw[o + 2];
    const float bh = raw[o + 3];
    std::vector<float> row(14, 0.0f);
    row[0] = cx - bw / 2;
    row[1] = cy - bh / 2;
    row[2] = cx + bw / 2;
    row[3] = cy + bh / 2;
    row[4] = score;
    for (int k = 0; k < 8; k++) {
      row[5 + k] = raw[o + 5 + k];
    }
    row[13] = static_cast<float>(layer);
    cand.push_back(row);
  }
  if (cand.empty()) {
    return {};
  }

  // Greedy NMS, descending score, strict > iouThresh keeps.
  std::vector<int> order(cand.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = static_cast<int>(i);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&cand](int a, int b) { return cand[a][4] > cand[b][4]; });
  std::vector<int> alive = order;
  std::vector<int> keep;
  while (!alive.empty()) {
    const int i = alive.front();
    alive.erase(alive.begin());
    keep.push_back(i);
    std::vector<int> rest;
    for (int j : alive) {
      if (Iou(cand[i], cand[j]) <= iouThresh) {
        rest.push_back(j);
      }
    }
    alive = rest;
  }

  std::vector<std::vector<float>> kept;
  kept.reserve(keep.size());
  for (int i : keep) {
    kept.push_back(cand[i]);
  }

  // restore_box: undo padding then scale. x uses [0,2,5,7,9,11], y uses [1,3,6,8,10,12].
  const int kx[6] = {0, 2, 5, 7, 9, 11};
  const int ky[6] = {1, 3, 6, 8, 10, 12};
  for (auto& row : kept) {
    for (int k = 0; k < 6; k++) {
      row[kx[k]] = (row[kx[k]] - left) / r;
      row[ky[k]] = (row[ky[k]] - top) / r;
    }
  }
  return kept;
}

std::vector<float> LprDecodeBareHead(const std::vector<std::vector<float>>& heads) {
  if (heads.size() != 3) {
    return {};
  }
  // Anchors lifted from graph constants 1005 / 1118 / 1231 (w,h per anchor).
  struct Scale {
    int h;
    int stride;
    float aw[3];
    float ah[3];
  };
  static const Scale kScales[3] = {
      {40, 8, {4.0f, 8.0f, 13.0f}, {5.0f, 10.0f, 16.0f}},
      {20, 16, {23.0f, 43.0f, 73.0f}, {29.0f, 55.0f, 105.0f}},
      {10, 32, {146.0f, 231.0f, 335.0f}, {217.0f, 300.0f, 433.0f}},
  };

  auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

  // Row order matches the in-graph Reshape (scale, anchor, y, x) so the result is
  // byte-comparable with the original model's [1,6300,15] output.
  std::vector<float> rows;
  rows.reserve(static_cast<size_t>(6300) * 15);
  for (int si = 0; si < 3; si++) {
    const Scale& sc = kScales[si];
    const int h = sc.h;
    const std::vector<float>& t = heads[si];
    if (t.size() != static_cast<size_t>(45) * h * h) {
      return {};  // unexpected head layout — caller falls back / errors out
    }
    const size_t chStride = static_cast<size_t>(h) * h;
    for (int a = 0; a < 3; a++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < h; x++) {
          const float* base = t.data() + (static_cast<size_t>(a * 15) * h + y) * h + x;
          auto v = [&](int ch) { return base[static_cast<size_t>(ch) * chStride]; };
          const float gxIdx = static_cast<float>(x);
          const float gyIdx = static_cast<float>(y);
          const float gxPx = static_cast<float>(x * sc.stride);
          const float gyPx = static_cast<float>(y * sc.stride);
          const float aw = sc.aw[a];
          const float ah = sc.ah[a];

          float row[15];
          row[0] = (sig(v(0)) * 2.0f - 0.5f + gxIdx) * static_cast<float>(sc.stride);
          row[1] = (sig(v(1)) * 2.0f - 0.5f + gyIdx) * static_cast<float>(sc.stride);
          const float ew = sig(v(2)) * 2.0f;
          const float eh = sig(v(3)) * 2.0f;
          row[2] = ew * ew * aw;
          row[3] = eh * eh * ah;
          row[4] = sig(v(4));
          // kpt channels are RAW logits: x-side * anchor_w + grid px, y-side * ah.
          for (int k = 0; k < 4; k++) {
            row[5 + 2 * k] = v(5 + 2 * k) * aw + gxPx;
            row[6 + 2 * k] = v(6 + 2 * k) * ah + gyPx;
          }
          row[13] = sig(v(13));
          row[14] = sig(v(14));
          rows.insert(rows.end(), row, row + 15);
        }
      }
    }
  }
  return rows;
}

bool LprRotateCrop(const RgbaImage& src, const int marks[4][2], RgbaImage& out) {
  auto dist = [](const int p[2], const int q[2]) {
    return std::hypot(static_cast<double>(p[0] - q[0]), static_cast<double>(p[1] - q[1]));
  };
  const int cropW = static_cast<int>(std::trunc(std::max(dist(marks[0], marks[1]),
                                                         dist(marks[2], marks[3]))));
  const int cropH = static_cast<int>(std::trunc(std::max(dist(marks[0], marks[3]),
                                                         dist(marks[1], marks[2]))));
  if (cropW <= 0 || cropH <= 0) {
    return false;
  }

  double srcQ[4][2];
  for (int i = 0; i < 4; i++) {
    srcQ[i][0] = marks[i][0];
    srcQ[i][1] = marks[i][1];
  }
  const double dstQ[4][2] = {
      {0, 0}, {static_cast<double>(cropW), 0},
      {static_cast<double>(cropW), static_cast<double>(cropH)},
      {0, static_cast<double>(cropH)}};

  double h[9];
  double hInv[9];
  GetPerspectiveTransform(srcQ, dstQ, h);
  Invert3x3(h, hInv);
  RgbaImage warped = WarpPerspectiveCubic(src, hInv, cropW, cropH);

  // A portrait crop means the plate is vertical: rotate it into reading order.
  if (static_cast<double>(cropH) / cropW >= 1.5) {
    out = Rotate90(warped);
  } else {
    out = warped;
  }
  return true;
}

static void LprEncodePlateInto(const RgbaImage& crop, int imgH, int imgW,
                               int limitedMaxWidth, int limitedMinWidth, int& outW,
                               bool nhwc, std::vector<float>& out, ResizeScratch& sc) {
  const int h = crop.height;
  const int w = crop.width;
  const double maxWhRatio = std::max(static_cast<double>(w) / h,
                                     static_cast<double>(imgW) / imgH);
  int targetW = static_cast<int>(std::trunc(imgH * maxWhRatio));
  targetW = std::max(std::min(targetW, limitedMaxWidth), limitedMinWidth);

  const double ratio = static_cast<double>(w) / h;
  int ratioImgH = static_cast<int>(std::ceil(imgH * ratio));
  ratioImgH = std::max(ratioImgH, limitedMinWidth);
  const int resizedW = ratioImgH > targetW ? targetW : static_cast<int>(std::trunc(ratioImgH));

  RgbaImage& resized = sc.img;
  ResizeLinearInto(crop, resizedW, imgH, resized, sc);

  // Padding 必须显式清零 —— assign 而不是 resize。
  //
  // 参照实现是 `padding_im = np.zeros((3,imgH,imgW))` 后只填 `[:, :, 0:resizedW]`，
  // 即 padding 列恒为 0。这里原来用 resize(3*plane)：只有当**元素数变大**时
  // 才会值初始化，而本函数的 targetW 固定在 [48,160]、crop 尺寸也固定，
  // 3*plane 几乎每次都不变 => resize 是空操作 => 列 [resizedW, targetW)
  // 保留**上一次调用**留下的值。
  //
  // 为什么这会真的算错（2026-09-21 实测，见 docs/t7-rq4-thermal.md）：
  // NHWC 与 NCHW 写的是**不同的下标集合**（NHWC: (y*W+x)*3+c；NCHW: c*plane + y*W+x），
  // 所以空洞里留下的是**另一种布局的残值**，不是"上一次的同一位置"。
  // 生产档 det=CPU / rec=NNRT 本来不该互相影响，但只要中间跑过一次 ncnn 槽位
  // （NCHW，且写满整个缓冲），随后 MS(NHWC) 就会读到被 NCHW 残值污染的输入张量。
  //
  // 真机可复现：全新进程（缓冲全 0）MS 稳定给 `苏ED5172` / rec.l2=4.0489；
  // 一旦跑过「三后端槽位」探针，**切回 MS 也**变成 `苏ED512` / rec.l2=3.0965，
  // 且字符数 6 非法（合法为 7/8）。张量指纹忠实记录了这次输入污染 —— 这正是
  // ADR-0003 的落点自证要抓的东西。
  const size_t plane = static_cast<size_t>(imgH) * targetW;
  out.assign(3 * plane, 0.0f);
  for (int y = 0; y < imgH; y++) {
    for (int x = 0; x < resizedW; x++) {
      const size_t s = (static_cast<size_t>(y) * resizedW + x) * 4;
      const float b = (resized.data[s + 2] - 127.5f) / 127.5f;
      const float g = (resized.data[s + 1] - 127.5f) / 127.5f;
      const float r = (resized.data[s] - 127.5f) / 127.5f;
      if (nhwc) {
        const size_t i = (static_cast<size_t>(y) * targetW + x) * 3;
        out[i] = b;
        out[i + 1] = g;
        out[i + 2] = r;
      } else {
        const size_t i = static_cast<size_t>(y) * targetW + x;
        out[i] = b;
        out[plane + i] = g;
        out[2 * plane + i] = r;
      }
    }
  }
  outW = targetW;
}

std::vector<float> LprEncodePlate(const RgbaImage& crop, int imgH, int imgW,
                                  int limitedMaxWidth, int limitedMinWidth, int& outW,
                                  bool nhwc) {
  std::vector<float> out;
  ResizeScratch sc;
  LprEncodePlateInto(crop, imgH, imgW, limitedMaxWidth, limitedMinWidth, outW, nhwc, out, sc);
  return out;
}

static void LprEncodeClassifyInto(const RgbaImage& crop, int size, bool nhwc,
                                  std::vector<float>& out, ResizeScratch& sc) {
  RgbaImage& resized = sc.img;
  ResizeLinearInto(crop, size, size, resized, sc);
  const size_t plane = static_cast<size_t>(size) * size;
  out.resize(3 * plane);
  for (size_t i = 0; i < plane; i++) {
    const size_t s = i * 4;
    const float b = resized.data[s + 2] / 255.0f;
    const float g = resized.data[s + 1] / 255.0f;
    const float r = resized.data[s] / 255.0f;
    if (nhwc) {
      out[i * 3] = b;
      out[i * 3 + 1] = g;
      out[i * 3 + 2] = r;
    } else {
      out[i] = b;
      out[plane + i] = g;
      out[2 * plane + i] = r;
    }
  }
}

std::vector<float> LprEncodeClassify(const RgbaImage& crop, int size, bool nhwc) {
  std::vector<float> out;
  ResizeScratch sc;
  LprEncodeClassifyInto(crop, size, nhwc, out, sc);
  return out;
}

// ---------------------------------------------------------------- 牌色（像素测量，ADR-0005）
/**
 * RGB -> HSV 色相。OpenCV 口径：H ∈ [0,180)，S/V ∈ [0,255]。
 * 只求 H 就够（判据只看色相带），但为了过滤"近白/近黑"需要 S 与 V。
 */
static void Rgb2Hsv(uint8_t r, uint8_t g, uint8_t b, int& h, int& s, int& v) {
  const int mx = std::max(r, std::max(g, b));
  const int mn = std::min(r, std::min(g, b));
  v = mx;
  const int d = mx - mn;
  s = (mx == 0) ? 0 : (d * 255) / mx;
  if (d == 0) {
    h = 0;
    return;
  }
  // 6 段，结果压到 [0,180) 以对齐 OpenCV
  int hh;
  if (mx == r) {
    hh = 30 * (g - b) / d + (g < b ? 180 : 0);
  } else if (mx == g) {
    hh = 30 * (b - r) / d + 60;
  } else {
    hh = 30 * (r - g) / d + 120;
  }
  h = hh % 360;
  if (h < 0) {
    h += 360;
  }
  h /= 2;  // 360 -> 180
}

/**
 * 牌色：牌面主导饱和色（移植自 plate_face_colour.py）。
 * 阈值与色带均取自那份实现，改动会破坏可复现性。
 */
std::string LprPlateColour(const RgbaImage& crop, float& outConfidence) {
  outConfidence = 0.0f;
  if (!crop.Valid()) {
    return "unknown";
  }
  const int w = crop.width;
  const int h = crop.height;
  // 裁掉边框：上下各 12%、左右各 8%
  const int y0 = static_cast<int>(h * 0.12);
  const int y1 = static_cast<int>(h * 0.88);
  const int x0 = static_cast<int>(w * 0.08);
  const int x1 = static_cast<int>(w * 0.92);

  const int kSMin = 90;   // 饱和下限：滤掉白色字符
  const int kVMin = 45;   // 亮度下限：滤掉黑色字符与阴影
  const int kVMax = 250;  // 亮度上限：滤掉高光

  long n = 0;
  long inGreen = 0, inBlue = 0, inYellow = 0;
  for (int y = y0; y < y1; y++) {
    for (int x = x0; x < x1; x++) {
      const size_t i = (static_cast<size_t>(y) * w + x) * 4;
      const uint8_t r = crop.data[i];
      const uint8_t g = crop.data[i + 1];
      const uint8_t b = crop.data[i + 2];
      int hh = 0, ss = 0, vv = 0;
      Rgb2Hsv(r, g, b, hh, ss, vv);
      if (ss < kSMin || vv < kVMin || vv > kVMax) {
        continue;
      }
      n++;
      if (hh >= 35 && hh <= 95) {
        inGreen++;
      } else if (hh >= 100 && hh <= 135) {
        inBlue++;
      } else if (hh >= 15 && hh <= 34) {
        inYellow++;
      }
    }
  }
  // 样本太少不足以判色（与原实现一致的门槛）
  if (n < 40) {
    return "unknown";
  }
  const double fg = static_cast<double>(inGreen) / n;
  const double fb = static_cast<double>(inBlue) / n;
  const double fy = static_cast<double>(inYellow) / n;
  const double best = std::max(fg, std::max(fb, fy));
  // 必须过半才算数，否则"未知"胜过瞎猜（判色是附带属性，不进识别主链）
  if (best < 0.55) {
    return "unknown";
  }
  outConfidence = static_cast<float>(best);
  if (best == fg) {
    return "green";
  }
  return (best == fb) ? "blue" : "yellow";
}


void LprCtcGreedy(const std::vector<int>& idx, const std::vector<float>& prob,
                  std::string& code, float& conf,
                  std::vector<std::string>& chars, std::vector<float>& probs) {
  chars.clear();
  probs.clear();
  const std::vector<std::string>& token = LprToken();
  for (size_t i = 0; i < idx.size(); i++) {
    const int v = idx[i];
    if (v == 0) {
      continue;  // blank
    }
    if (i > 0 && idx[i - 1] == v) {
      continue;  // repeat
    }
    chars.push_back(static_cast<size_t>(v) < token.size() ? token[v] : "?");
    probs.push_back(i < prob.size() ? prob[i] : 0.0f);
  }
  double sum = 0;
  for (float p : probs) {
    sum += p;
  }
  conf = probs.empty() ? 0.0f : static_cast<float>(sum / probs.size());
  code.clear();
  for (const std::string& c : chars) {
    code += c;
  }
}

/**
 * Per-frame scratch, reused across pipeline calls (A18 §1).
 *
 * LprRunPipeline runs only on the dedicated inference thread (napi_init's
 * InferenceRunner, serial QueueJob) and the LprSessions handed in is rebuilt
 * per call, so a function-local thread_local instance is where scratch can
 * live. Reusing these buffers removes ~1.2 MB (detector pack) + 5.5 MB
 * (resize intermediate) + ~1 MB (letterbox/output tensors) of per-frame
 * allocation and first-touch faults.
 */
struct PipeScratch {
  ResizeScratch rs;
  LetterBoxed lb;
  std::vector<float> detIn;
  std::vector<std::vector<float>> detOuts;
  RgbaImage crop;
  std::vector<float> recIn;
  std::vector<float> recLogits;
  std::vector<float> clsIn;
};

bool LprRunPipeline(const RgbaImage& img, const LprSessions& s,
                    std::vector<PlateResult>& out, std::string& err) {
  static thread_local PipeScratch sc;
  out.clear();
  if (!img.Valid()) {
    err = "invalid RGBA image";
    return false;
  }
  if ((s.det == nullptr && !s.detNcnn) || (s.rec == nullptr && s.recSlot < 0) ||
      (s.cls == nullptr && s.clsSlot < 0)) {
    err = "missing session (det/rec/cls must all be loaded)";
    return false;
  }

  const double t0 = NowMs();

  // ---- detect: letterbox + encode + infer + decode + NMS (拆分成独立计时)
  PIPE_TRACE("E2E preprocess begin ncnn=%{public}d vulkan=%{public}d", s.detNcnn, s.detVulkan);
  
  // Letterbox 单独计时
  const double t_lb_start = NowMs();
  LprLetterBoxInto(img, s.detSize, sc.lb, sc.rs);
  const LetterBoxed& lb = sc.lb;
  if (!lb.img.Valid()) {
    err = "letterbox failed";
    return false;
  }
  const double t_lb_end = NowMs();

  // Encode + Infer 单独计时
  const double t_encode_start = NowMs();
  std::vector<std::vector<float>>& detOuts = sc.detOuts;
  // 2026-09-20：encode（布局打包）与 infer（纯推理）拆开 —— det 段流水线内
  // 24.86 ms 而纯推理 7.13 ms，中间的账一直没拆过（ADR-009 §5 待办）。
  double t_pack_end = t_encode_start;

  if (s.detNcnn) {
    if (!NcnnReload(s.detVulkan, err) || !NcnnDetect(lb, detOuts, err)) {
      return false;
    }
  } else {
    const bool detNhwc = (s.det->inputFormat == OH_AI_FORMAT_NHWC);
    if (detNhwc) {
      LprToNhwcInto(lb.img, true, sc.detIn);
    } else {
      LprToNchwInto(lb.img, true, sc.detIn);
    }
    t_pack_end = NowMs();
    if (!MsRunMulti(s.det, sc.detIn.data(), detOuts, err)) {
      return false;
    }
  }
  const double t_infer_end = NowMs();
  
  if (detOuts.empty()) {
    err = "detector returned no tensors";
    return false;
  }
  PIPE_TRACE("E2E detector returned tensors=%{public}zu", detOuts.size());
  std::vector<float> detRows;
  if (detOuts.size() == 3) {
    detRows = LprDecodeBareHead(detOuts);
    if (detRows.size() != static_cast<size_t>(6300) * 15) {
      err = "bare-head decode produced " + std::to_string(detRows.size()) + " floats";
      return false;
    }
  } else {
    detRows = detOuts[0];
  }
  const int rows = static_cast<int>(detRows.size() / 15);
  const double t_decode_start = NowMs();
  const std::vector<std::vector<float>> dets =
      LprDecodeDetections(detRows, rows, s.confThresh, s.iouThresh, lb.r, lb.left, lb.top);
  const double t_nms_end = NowMs();
  
  const double t1 = NowMs();

  const int kDoubleLayer = 1;
  for (const std::vector<float>& row : dets) {
    // Upstream casts keypoints to int *before* measuring edge lengths; skipping the
    // truncation shifts the crop by up to a pixel and can flip a character.
    int marks[4][2];
    for (int k = 0; k < 4; k++) {
      marks[k][0] = static_cast<int>(std::trunc(row[5 + k * 2]));
      marks[k][1] = static_cast<int>(std::trunc(row[6 + k * 2]));
    }

    RgbaImage& crop = sc.crop;
    if (!LprRotateCrop(img, marks, crop)) {
      continue;
    }
    const double t2 = NowMs();

    PlateResult item;
    for (int k = 0; k < 4; k++) {
      item.rect[k] = static_cast<int>(std::trunc(row[k]));
    }
    item.detScore = row[4];
    item.layer = static_cast<int>(row[13]);
    item.cropH = crop.height;
    item.cropW = crop.width;

    // RGB-only checksum (alpha excluded) — comparable to numpy's BGR sum.
    long long sum = 0;
    for (size_t i = 0; i < crop.data.size(); i++) {
      if (i % 4 != 3) {
        sum += crop.data[i];
      }
    }
    item.cropSum = sum;

    std::string code;
    float conf = 0;
    std::vector<std::string> chars;
    std::vector<float> probs;
    if (item.layer == kDoubleLayer) {
      const int line = static_cast<int>(std::trunc(crop.height * 0.4));
      std::string c0, c1;
      float f0 = 0, f1 = 0;
      std::vector<std::string> ch0, ch1;
      std::vector<float> pr0, pr1;
      if (!Recognise(s.rec, s.recSlot, SliceRows(crop, 0, line), c0, f0, ch0, pr0, err,
                    sc.recIn, sc.recLogits, sc.rs) ||
          !Recognise(s.rec, s.recSlot, SliceRows(crop, line, crop.height), c1, f1, ch1, pr1, err,
                    sc.recIn, sc.recLogits, sc.rs)) {
        return false;
      }
      code = c0 + c1;
      conf = (f0 + f1) / 2;
      chars.insert(chars.end(), ch0.begin(), ch0.end());
      chars.insert(chars.end(), ch1.begin(), ch1.end());
      probs.insert(probs.end(), pr0.begin(), pr0.end());
      probs.insert(probs.end(), pr1.begin(), pr1.end());
    } else {
      if (!Recognise(s.rec, s.recSlot, crop, code, conf, chars, probs, err,
                    sc.recIn, sc.recLogits, sc.rs)) {
        return false;
      }
    }
    const double t3 = NowMs();

    if (code.empty()) {
      continue;
    }

    PIPE_TRACE("E2E recogniser returned code=%{public}s", code.c_str());
    item.code = code;
    item.recConf = conf;
    item.chars = chars;
    item.charProbs = probs;

    // ---- 牌色：像素测量（ADR-0005，2026-09-21）
    // 这里原本跑一个 1.53 MB 的分类模型（litemodel_cls_96x_r1），占 2.69 ms/帧（7.4%）。
    // 它有两个问题：
    //   1) 标签表是旋转的 —— 正确顺序是 blue=0/green=1/yellow=2，源码写的却是
    //      yellow/blue/green。真机可复现：hlpr-test（实为绿牌）显示"蓝牌"、
    //      scene-2 的 藏DT5022（蓝牌）显示"黄牌"。
    //   2) 判色是附带属性，不进入识别主链，不值得占 7.4% 帧预算。
    // 换成像素测量后约 0.1 ms，且无查表依赖。
    item.colour = LprPlateColour(crop, item.colourConfidence);
    const double t4 = NowMs();

    // 分段口径（T2 修正，2026-09-21）：
    //   tDetectMs = 检测【整个阶段】：letterbox + pack + infer + decode/NMS
    //   其余四个是检测段的内部拆分，不得与其相加（会重复计入）
    // 修正前 tDetectMs 与 tLetterboxMs 是同一个表达式，导致「分段之和 ≈ 端到端」
    // 不成立 —— 实测三处相加比 totalMs 多 18.7 ms，正是 letterbox 被重复计入。
    item.tLetterboxMs = static_cast<float>(t_lb_end - t_lb_start);
    item.tPackMs = static_cast<float>(t_pack_end - t_encode_start);
    item.tInferMs = static_cast<float>(t_infer_end - t_pack_end);
    item.tEncodeInferMs = static_cast<float>(t_infer_end - t_encode_start);  // = pack + infer
    item.tDecodeNmsMs = static_cast<float>(t_nms_end - t_decode_start);
    item.tDetectMs = static_cast<float>(t_nms_end - t_lb_start);  // 整个检测阶段
    // 分段口径第二处修正（T2，2026-09-21）：
    //   tRectifyMs 原为 t2 - t_lb_end，而 t2 在 LprRotateCrop 之后才取 ——
    //   中间横跨了整个检测段（encode/infer/decode/NMS），于是"矫正"把检测段
    //   也算了进去，四段之和必然虚高（实测多 25.7 ms）。
    //   矫正段的正确起点是检测段结束处 t_nms_end。
    item.tRectifyMs = static_cast<float>(t2 - t_nms_end);
    item.tRecogMs = static_cast<float>(t3 - t2);
    item.tClsMs = static_cast<float>(t4 - t3);
    out.push_back(item);
  }

  // Best detector score first — the caller shows out[0] as "the" plate.
  std::stable_sort(out.begin(), out.end(),
                   [](const PlateResult& a, const PlateResult& b) {
                     return a.detScore > b.detScore;
                   });
  return true;
}

// ---------------------------------------------------------------- 相机取帧转换

/**
 * 整数 BT.601 有限范围（相机预览的标准约定）：Y' 在 [16,235]，Cb/Cr 在 [16,240]。
 * 用整数近似做定点运算，避免每像素一次浮点乘。
 *
 * 抽成独立函数，是为了让「朴素」与「分块」两条写出路径共用**同一份算术** ——
 * 逐位相等由 tools/verify_nv21_to_rgba.py 守卫（15 组尺寸/stride/旋转角）。
 */
static inline void Nv21PixelToRgb(int Y, int U, int V, int& r, int& g, int& b) {
  // C = Y-16, D = U-128, E = V-128
  const int c = Y - 16;
  const int d = U - 128;
  const int e = V - 128;
  r = (298 * c + 409 * e + 128) >> 8;
  g = (298 * c - 100 * d - 208 * e + 128) >> 8;
  b = (298 * c + 516 * d + 128) >> 8;
  r = r < 0 ? 0 : (r > 255 ? 255 : r);
  g = g < 0 ? 0 : (g > 255 ? 255 : g);
  b = b < 0 ? 0 : (b > 255 ? 255 : b);
}

/**
 * NV21 -> RGBA，整数倍 90 度旋转在写入时一并完成（不额外搬一次内存）。
 *
 * **转置型旋转（90/270）走源空间分块。** 朴素写法内层循环走 x，而目标行号
 * dy 随 x 每步 +1，于是每写 4 字节就换一条新缓存行（利用率 4/64）——
 * 30 万像素 = 19.6 MB 无效写流量，而真实数据只有 1.2 MB。
 * 按 16x16 源块分块后：固定 x 就固定了目标行，内层 y 只在 64 字节
 * （= 正好一条缓存行）内移动；读侧的 16 行也被同一块内的 x 循环复用。
 * 0/180 的内层写出本来就是行内连续的，保持原样。
 *
 * 端侧实测 conv 有 2.2~14.7 ms 的动态范围（docs/notes/camera-npu-headroom.md），
 * 所以**必须用流水线口径 A/B**，不能拿隔离基准外推（纪律 7）。
 */
bool LprNv21ToRgba(const uint8_t* nv21, size_t nv21Size, int width, int height, int stride,
                   int rotation, RgbaImage& out) {
  if (nv21 == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  if (stride <= 0) {
    stride = width;
  }
  const size_t ySize = static_cast<size_t>(stride) * height;
  const size_t uvRows = static_cast<size_t>(height + 1) / 2;
  const size_t need = ySize + static_cast<size_t>(stride) * uvRows;
  if (nv21Size < need) {
    return false;
  }
  const uint8_t* uv = nv21 + ySize;

  const int rot = ((rotation % 360) + 360) % 360;
  const bool swap = (rot == 90 || rot == 270);
  out.width = swap ? height : width;
  out.height = swap ? width : height;
  out.data.assign(static_cast<size_t>(out.width) * out.height * 4, 255);

  // 分块边长（像素）。16 像素 x 4 字节 = 64 字节 = 一条缓存行。
  constexpr int kNv21Tile = 16;

  if (rot == 90 || rot == 270) {
    const int outStride = out.width * 4;
    uint8_t* base = out.data.data();
    for (int y0 = 0; y0 < height; y0 += kNv21Tile) {
      const int y1 = (y0 + kNv21Tile < height) ? y0 + kNv21Tile : height;
      for (int x0 = 0; x0 < width; x0 += kNv21Tile) {
        const int x1 = (x0 + kNv21Tile < width) ? x0 + kNv21Tile : width;
        for (int x = x0; x < x1; x++) {
          const int dy = (rot == 90) ? x : (width - 1 - x);
          const size_t uvIdx = static_cast<size_t>(x / 2) * 2;
          uint8_t* oRow = base + static_cast<size_t>(dy) * outStride;
          for (int y = y0; y < y1; y++) {
            const int Y = nv21[static_cast<size_t>(y) * stride + x];
            // VU 交错：每 2 行共用一个色度行，每 2 列共用一个色度对。
            const uint8_t* uvRow = uv + static_cast<size_t>(y / 2) * stride;
            const int V = uvRow[uvIdx];
            const int U = uvRow[uvIdx + 1];
            int r = 0, g = 0, b = 0;
            Nv21PixelToRgb(Y, U, V, r, g, b);
            const int dx = (rot == 90) ? (height - 1 - y) : y;
            uint8_t* o = oRow + static_cast<size_t>(dx) * 4;
            o[0] = static_cast<uint8_t>(r);
            o[1] = static_cast<uint8_t>(g);
            o[2] = static_cast<uint8_t>(b);
            o[3] = 255;
          }
        }
      }
    }
    return true;
  }

  for (int y = 0; y < height; y++) {
    const uint8_t* yRow = nv21 + static_cast<size_t>(y) * stride;
    const uint8_t* uvRow = uv + static_cast<size_t>(y / 2) * stride;
    for (int x = 0; x < width; x++) {
      const int Y = yRow[x];
      const size_t uvIdx = static_cast<size_t>(x / 2) * 2;
      const int V = uvRow[uvIdx];
      const int U = uvRow[uvIdx + 1];
      int r = 0, g = 0, b = 0;
      Nv21PixelToRgb(Y, U, V, r, g, b);

      // 目标坐标按旋转角算 —— 一次写成，省掉后续的 rotate()。
      int dx = 0;
      int dy = 0;
      if (rot == 180) {
        dx = width - 1 - x;
        dy = height - 1 - y;
      } else {
        dx = x;
        dy = y;
      }
      uint8_t* o = out.data.data() +
                   (static_cast<size_t>(dy) * out.width + dx) * 4;
      o[0] = static_cast<uint8_t>(r);
      o[1] = static_cast<uint8_t>(g);
      o[2] = static_cast<uint8_t>(b);
      o[3] = 255;
    }
  }
  return true;
}

// ============================================================ 车辆检测（T2）
//
// 路线说明见 lpr_pipeline.h 的「车辆检测」段：票面写的 ncnn 改成了
// 「改写 DFL → 转 .ms → 走 MS Lite CPU」，理由与现有 det=CPU 一致。

const std::vector<std::string>& LprCocoNames() {
  // ultralytics/COCO 的固定 80 类顺序，下标即类号。顺序不能改 ——
  // 它与训练时的 names 字典一一对应，改了就会把 bus 叫成 car。
  static const std::vector<std::string> kNames = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
      "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
      "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
      "toothbrush",
  };
  return kNames;
}

bool LprIsVehicleClass(int classId) {
  // car / motorcycle / bus / truck。不含 bicycle(1) —— 那是非机动车，
  // 拍不到车牌；也不含 train(6)。
  return classId == 2 || classId == 3 || classId == 5 || classId == 7;
}

static float BoxIou(const float a[4], const float b[4]) {
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float iw = x2 - x1;
  const float ih = y2 - y1;
  if (!(iw > 0) || !(ih > 0)) {
    return 0;
  }
  const float inter = iw * ih;
  const float aa = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
  const float bb = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
  const float uni = aa + bb - inter;
  return uni > 0 ? inter / uni : 0;
}

/**
 * 车辆框的**跨类别**去重：同一目标被不同类别各检出一个框时，只保留最高分。
 *
 * 为什么必须做：模型对同一辆车可能同时给出 cls=2(car) 与 cls=3(motorcycle)
 * 两个框（实测 IoU 0.99）。NMS 是 **class-wise** 的，不会互相抑制 —— 于是
 * **同一辆车被跑两次车牌检测**，纯浪费且零召回收益。
 *
 * ⚠️ 与 D1「遍历所有车框」不冲突：D1 要的是"不因分数低就丢框"，这里丢的是
 * **空间上重复**的框（同一目标的不同类别标签），不是低分框。去掉它不会让任何
 * 一个**未被覆盖的目标**失去 ROI。
 *
 * 这是 D3 在**车框层面**的对应物：D3 要求对车牌框去重，而重叠车牌框的来源
 * 正是重叠车框 —— 在源头去重，省的是整次车牌检测，而不只是去重那一步。
 *
 * 返回丢掉的框数。副作用：`boxes` 被就地改为去重后（**按分数降序**）。
 */
static int LprDedupeVehicles(std::vector<VehicleBox>& boxes, float iouThresh) {
  if (boxes.size() < 2) {
    return 0;
  }
  std::sort(boxes.begin(), boxes.end(),
            [](const VehicleBox& a, const VehicleBox& b) { return a.score > b.score; });
  std::vector<VehicleBox> keep;
  keep.reserve(boxes.size());
  for (const VehicleBox& b : boxes) {
    bool dup = false;
    for (const VehicleBox& k : keep) {
      if (BoxIou(b.rect, k.rect) >= iouThresh) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      keep.push_back(b);
    }
  }
  const int dropped = static_cast<int>(boxes.size() - keep.size());
  boxes.swap(keep);
  return dropped;
}


bool LprDetectGeometryOf(const MsSession* det, int& outSize, bool& outNhwc, std::string& err) {
  outSize = 0;
  outNhwc = false;
  if (det == nullptr) {
    err = "detectGeometry: null session";
    return false;
  }
  outNhwc = (det->inputFormat == OH_AI_FORMAT_NHWC);
  long long prod = 1;
  for (int64_t d : det->inputShape) {
    if (d <= 0) {
      err = "detectGeometry: 输入形状含动态维";
      return false;
    }
    prod *= static_cast<long long>(d);
  }
  // 不按 layout 猜边长，直接由元素数反推 —— NHWC/NCHW 的报告在本项目里并不稳定，
  // 但 "3 * S * S" 这个事实两种 layout 下都成立。
  if (prod <= 0 || (prod % 3) != 0) {
    err = "detectGeometry: 输入元素数 " + std::to_string(prod) + " 不是 3 的倍数";
    return false;
  }
  const long long area = prod / 3;
  const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(area))));
  if (side <= 0 || static_cast<long long>(side) * side != area) {
    err = "detectGeometry: 输入不是 3xSxS（元素数 " + std::to_string(prod) + "）";
    return false;
  }
  outSize = side;
  return true;
}

std::vector<VehicleBox> LprDecodeYolov5u(const std::vector<float>& raw, float confThresh,
                                         float iouThresh, float r, int left, int top,
                                         bool vehicleOnly, int maxBoxes, bool* outTruncated) {
  if (outTruncated != nullptr) {
    *outTruncated = false;
  }
  constexpr int kClasses = 80;
  constexpr int kChannels = 4 + kClasses;  // 84
  if (raw.empty() || (raw.size() % kChannels) != 0) {
    return {};
  }
  const int anchors = static_cast<int>(raw.size() / kChannels);

  std::vector<VehicleBox> cand;
  cand.reserve(64);
  for (int a = 0; a < anchors; ++a) {
    // 先扫类分数：低于阈值就整条丢，省掉一次坐标计算
    int best = -1;
    float bestScore = 0;
    for (int c = 0; c < kClasses; ++c) {
      const float s = raw[static_cast<size_t>(4 + c) * anchors + a];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (!(bestScore > confThresh)) {
      continue;
    }
    if (vehicleOnly && !LprIsVehicleClass(best)) {
      continue;
    }
    const float cx = raw[static_cast<size_t>(0) * anchors + a];
    const float cy = raw[static_cast<size_t>(1) * anchors + a];
    const float bw = raw[static_cast<size_t>(2) * anchors + a];
    const float bh = raw[static_cast<size_t>(3) * anchors + a];
    VehicleBox b;
    b.rect[0] = cx - bw * 0.5f;
    b.rect[1] = cy - bh * 0.5f;
    b.rect[2] = cx + bw * 0.5f;
    b.rect[3] = cy + bh * 0.5f;
    b.score = bestScore;
    b.classId = best;
    cand.push_back(b);
  }
  if (cand.empty()) {
    return {};
  }

  // 按分数降序；NMS 之前先截一刀，避免低阈值下候选爆到 2100 让 O(n²) 变贵。
  std::vector<int> order(cand.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&cand](int x, int y) { return cand[x].score > cand[y].score; });
  constexpr size_t kMaxCand = 1000;
  if (order.size() > kMaxCand) {
    order.resize(kMaxCand);
    if (outTruncated != nullptr) {
      *outTruncated = true;  // 候选被截断过，结果已不完整，必须如实上报
    }
  }

  // 按类 NMS（YOLO 约定：不同类之间不互相抑制）
  std::vector<char> dead(cand.size(), 0);
  std::vector<VehicleBox> kept;
  for (int oi : order) {
    if (dead[oi]) {
      continue;
    }
    if (maxBoxes > 0 && static_cast<int>(kept.size()) >= maxBoxes) {
      if (outTruncated != nullptr) {
        *outTruncated = true;
      }
      break;
    }
    kept.push_back(cand[oi]);
    for (int oj : order) {
      if (oj == oi || dead[oj] || cand[oj].classId != cand[oi].classId) {
        continue;
      }
      if (BoxIou(cand[oi].rect, cand[oj].rect) > iouThresh) {
        dead[oj] = 1;
      }
    }
  }

  // letterbox 反变换：先减 padding 再除 scale（与 LprDecodeDetections 同一套）
  for (VehicleBox& b : kept) {
    b.rect[0] = (b.rect[0] - static_cast<float>(left)) / r;
    b.rect[1] = (b.rect[1] - static_cast<float>(top)) / r;
    b.rect[2] = (b.rect[2] - static_cast<float>(left)) / r;
    b.rect[3] = (b.rect[3] - static_cast<float>(top)) / r;
  }
  return kept;
}

bool LprVehicleDetect(const RgbaImage& img, MsSession* det, float confThresh, float iouThresh,
                      bool vehicleOnly, std::vector<VehicleBox>& out, bool& outTruncated,
                      float& outInferMs, std::string& err) {
  out.clear();
  outTruncated = false;
  outInferMs = 0;
  if (!img.Valid()) {
    err = "vehicleDetect: invalid image";
    return false;
  }
  int size = 0;
  bool nhwc = false;
  if (!LprDetectGeometryOf(det, size, nhwc, err)) {
    return false;
  }

  // ⚠️ T4 待办：这里的 scratch / 输入缓冲是**每帧新分配**的。A18 §1 记录过
  //    首次触碰的缺页开销（~7 ms/帧），det 段当年就是靠复用 buffer 消掉的。
  //    一期先求正确，等 T4 把「车框→逐框车牌」串起来时再一并搬进共享 scratch。
  ResizeScratch sc;
  LetterBoxed lb;
  LprLetterBoxInto(img, size, lb, sc);
  if (!lb.img.Valid()) {
    err = "vehicleDetect: letterbox failed";
    return false;
  }

  std::vector<float> in;
  if (nhwc) {
    LprToNhwcInto(lb.img, /*swapRB=*/true, in);
  } else {
    LprToNchwInto(lb.img, /*swapRB=*/true, in);
  }

  std::vector<std::vector<float>> outs;
  const double t0 = NowMs();
  if (!MsRunMulti(det, in.data(), outs, err)) {
    return false;
  }
  outInferMs = static_cast<float>(NowMs() - t0);

  if (outs.size() != 1) {
    err = "vehicleDetect: yolov5u 期望单输出，实际 " + std::to_string(outs.size());
    return false;
  }
  const std::vector<float>& raw = outs[0];
  if (raw.empty() || (raw.size() % 84) != 0) {
    err = "vehicleDetect: 输出长度 " + std::to_string(raw.size()) + " 不是 84 的倍数";
    return false;
  }

  out = LprDecodeYolov5u(raw, confThresh, iouThresh, lb.r, lb.left, lb.top, vehicleOnly,
                         kMaxVehicleBoxes, &outTruncated);
  return true;
}

// ---------------------------------------------------------------- ROI 裁剪（T3）

bool RoiRect::ContainsBox(const float box[4]) const {
  if (!valid || imgW <= 0 || imgH <= 0) {
    return false;
  }
  const float fx1 = std::min(box[0], box[2]);
  const float fy1 = std::min(box[1], box[3]);
  const float fx2 = std::max(box[0], box[2]);
  const float fy2 = std::max(box[1], box[3]);
  // 先与图求交 —— 车框可以超出图边界，这时"被覆盖"指的是框与图的交集被覆盖。
  const float cx1 = std::max(0.0f, std::min(fx1, static_cast<float>(imgW)));
  const float cy1 = std::max(0.0f, std::min(fy1, static_cast<float>(imgH)));
  const float cx2 = std::max(0.0f, std::min(fx2, static_cast<float>(imgW)));
  const float cy2 = std::max(0.0f, std::min(fy2, static_cast<float>(imgH)));
  if (!(cx2 > cx1) || !(cy2 > cy1)) {
    return true;  // 框与图无交：空集，任何 ROI 都"覆盖"它
  }
  return x0 <= static_cast<int>(std::floor(cx1)) &&
         y0 <= static_cast<int>(std::floor(cy1)) &&
         (x0 + w) >= static_cast<int>(std::ceil(cx2)) &&
         (y0 + h) >= static_cast<int>(std::ceil(cy2));
}

RoiRect LprRoiFromBox(const float box[4], int imgW, int imgH, float expand) {
  RoiRect r;
  r.imgW = imgW;
  r.imgH = imgH;
  r.expand = expand;
  if (imgW <= 0 || imgH <= 0 || !(expand >= 0)) {
    return r;  // valid=false：非法图尺寸 / 负外扩
  }
  const float fx1 = std::min(box[0], box[2]);
  const float fy1 = std::min(box[1], box[3]);
  const float fx2 = std::max(box[0], box[2]);
  const float fy2 = std::max(box[1], box[3]);
  const float bw = fx2 - fx1;
  const float bh = fy2 - fy1;
  if (!(bw > 0) || !(bh > 0)) {
    return r;  // 零面积框：没有可裁的东西，交给调用方跳过（不产生空 ROI 崩溃）
  }
  const float ex = expand * bw;
  const float ey = expand * bh;
  // 向外取整：左/上 floor，右/下 ceil。
  int x0 = static_cast<int>(std::floor(fx1 - ex));
  int y0 = static_cast<int>(std::floor(fy1 - ey));
  int x1 = static_cast<int>(std::ceil(fx2 + ex));
  int y1 = static_cast<int>(std::ceil(fy2 + ey));
  if (x0 < 0) {
    x0 = 0;
    r.clamped = true;
  }
  if (y0 < 0) {
    y0 = 0;
    r.clamped = true;
  }
  if (x1 > imgW) {
    x1 = imgW;
    r.clamped = true;
  }
  if (y1 > imgH) {
    y1 = imgH;
    r.clamped = true;
  }
  r.x0 = x0;
  r.y0 = y0;
  r.w = x1 - x0;
  r.h = y1 - y0;
  // 框完全在图外时 w/h 会算出负值 —— 这里统一收敛成"不可用"，而不是让它带着负数往下走。
  r.valid = (r.w > 0 && r.h > 0);
  return r;
}

bool LprCropRoi(const RgbaImage& src, const RoiRect& roi, RgbaImage& out, std::string& err) {
  out.data.clear();
  out.width = 0;
  out.height = 0;
  if (!src.Valid()) {
    err = "cropRoi: invalid source";
    return false;
  }
  if (!roi.valid) {
    err = "cropRoi: roi not valid";
    return false;
  }
  if (roi.x0 < 0 || roi.y0 < 0 || roi.w <= 0 || roi.h <= 0 ||
      roi.x0 + roi.w > src.width || roi.y0 + roi.h > src.height) {
    err = "cropRoi: roi out of source";
    return false;
  }
  out.width = roi.w;
  out.height = roi.h;
  out.data.resize(static_cast<size_t>(roi.w) * static_cast<size_t>(roi.h) * 4);
  const size_t rowBytes = static_cast<size_t>(roi.w) * 4;
  for (int y = 0; y < roi.h; y++) {
    const uint8_t* s = src.data.data() +
                       (static_cast<size_t>(roi.y0 + y) * static_cast<size_t>(src.width) +
                        static_cast<size_t>(roi.x0)) * 4;
    std::memcpy(out.data.data() + static_cast<size_t>(y) * rowBytes, s, rowBytes);
  }
  return true;
}

void LprRoiMapRect(int rect[4], const RoiRect& roi) {
  rect[0] += roi.x0;
  rect[1] += roi.y0;
  rect[2] += roi.x0;
  rect[3] += roi.y0;
}

void LprRoiMapRect(float rect[4], const RoiRect& roi) {
  rect[0] += static_cast<float>(roi.x0);
  rect[1] += static_cast<float>(roi.y0);
  rect[2] += static_cast<float>(roi.x0);
  rect[3] += static_cast<float>(roi.y0);
}

void LprRoiUnmapRect(float rect[4], const RoiRect& roi) {
  rect[0] -= static_cast<float>(roi.x0);
  rect[1] -= static_cast<float>(roi.y0);
  rect[2] -= static_cast<float>(roi.x0);
  rect[3] -= static_cast<float>(roi.y0);
}

void LprRoiMapRects(std::vector<VehicleBox>& boxes, const RoiRect& roi) {
  for (VehicleBox& b : boxes) {
    LprRoiMapRect(b.rect, roi);
  }
}

long long LprRgbSum(const RgbaImage& img) {
  long long sum = 0;
  for (size_t i = 0; i < img.data.size(); i++) {
    if (i % 4 != 3) {
      sum += img.data[i];
    }
  }
  return sum;
}

/** 裁出来的字节是否与源图对应区域**逐字节**相同。 */
static bool RoiBytesEqual(const RgbaImage& src, const RgbaImage& crop, const RoiRect& roi) {
  if (crop.width != roi.w || crop.height != roi.h) {
    return false;
  }
  const size_t rowBytes = static_cast<size_t>(roi.w) * 4;
  for (int y = 0; y < roi.h; y++) {
    const uint8_t* s = src.data.data() +
                       (static_cast<size_t>(roi.y0 + y) * static_cast<size_t>(src.width) +
                        static_cast<size_t>(roi.x0)) * 4;
    const uint8_t* c = crop.data.data() + static_cast<size_t>(y) * rowBytes;
    if (std::memcmp(s, c, rowBytes) != 0) {
      return false;
    }
  }
  return true;
}

std::string LprRoiSelfTest(const RgbaImage& img) {
  std::string rep;
  int total = 0;
  int failed = 0;

  RgbaImage src;
  if (img.Valid()) {
    src = img;
  } else {
    // 没拿到图也要能跑完自证：合成一张确定性图案（值只依赖 x/y，可复现）。
    src.width = 320;
    src.height = 320;
    src.data.resize(320 * 320 * 4);
    for (int y = 0; y < 320; y++) {
      for (int x = 0; x < 320; x++) {
        const size_t i = (static_cast<size_t>(y) * 320 + x) * 4;
        src.data[i + 0] = static_cast<uint8_t>((x * 7 + y * 13) & 0xFF);
        src.data[i + 1] = static_cast<uint8_t>((x * 3 + y * 29) & 0xFF);
        src.data[i + 2] = static_cast<uint8_t>((x * 17 + y * 5) & 0xFF);
        src.data[i + 3] = 255;
      }
    }
    rep += "note=src-synth;why=传入图无效，改用确定性图案\n";
  }

  const int W = src.width;
  const int H = src.height;
  rep += "srcSize=" + std::to_string(W) + "x" + std::to_string(H) + "\n";
  rep += "rgbSumAll=" + std::to_string(LprRgbSum(src)) + "\n";
  // 几何用例统一在固定的 320x320 坐标系里断言 —— 期望值是手算硬编码的，
  // 若跟着传入图尺寸走，期望值就得现算，那也就失去"独立于实现"的意义了。
  rep += "geomFrame=320x320\n";

  auto row = [&rep, &total, &failed](const char* name, bool ok, const std::string& detail) {
    total++;
    if (!ok) {
      failed++;
    }
    rep += "case=";
    rep += name;
    rep += ";ok=";
    rep += (ok ? "1" : "0");
    if (!detail.empty()) {
      rep += ";";
      rep += detail;
    }
    rep += "\n";
  };

  // 把 imgW/imgH/expand 一并报出来：PC 侧复核要能**原样复现**这次调用的参数。
  // 否则它只能猜 —— 例如 `roi-bad-src-size` 传的是 0x0，猜成 320x320 就会误判。
  auto roiStr = [](const RoiRect& r) {
    return "x0=" + std::to_string(r.x0) + ";y0=" + std::to_string(r.y0) +
           ";w=" + std::to_string(r.w) + ";h=" + std::to_string(r.h) +
           ";imgW=" + std::to_string(r.imgW) + ";imgH=" + std::to_string(r.imgH) +
           ";clamped=" + std::to_string(r.clamped ? 1 : 0) +
           ";valid=" + std::to_string(r.valid ? 1 : 0) +
           ";expand=" + std::to_string(r.expand);
  };

  auto boxStr = [](const float b[4]) {
    return "box=" + std::to_string(b[0]) + "," + std::to_string(b[1]) + "," +
           std::to_string(b[2]) + "," + std::to_string(b[3]);
  };

  auto expStr = [](int x0, int y0, int w, int h) {
    return "exp=" + std::to_string(x0) + "," + std::to_string(y0) + "," +
           std::to_string(w) + "," + std::to_string(h);
  };

  auto geomOk = [](const RoiRect& r, int x0, int y0, int w, int h) {
    return r.valid && r.x0 == x0 && r.y0 == y0 && r.w == w && r.h == h;
  };

  const float kE = kRoiExpandDefault;  // 0.15

  // ---- 几何：外扩 + 取整方向 + clamp ----
  {
    const float b[4] = {100, 100, 200, 200};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // 100-15=85, 200+15=215 → 130x130
    row("roi-center", geomOk(r, 85, 85, 130, 130) && !r.clamped,
        roiStr(r) + ";" + expStr(85, 85, 130, 130) + ";" + boxStr(b));
  }
  {
    const float b[4] = {2, 50, 102, 150};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // x0 = floor(2-15) = -13 → clamp 0；x1 = ceil(117) = 117
    row("roi-left-edge", geomOk(r, 0, 35, 117, 130) && r.clamped,
        roiStr(r) + ";" + expStr(0, 35, 117, 130) + ";" + boxStr(b));
  }
  {
    const float b[4] = {218, 50, 318, 150};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // x1 = ceil(333) = 333 → clamp 320；x0 = floor(203) = 203
    row("roi-right-edge", geomOk(r, 203, 35, 117, 130) && r.clamped,
        roiStr(r) + ";" + expStr(203, 35, 117, 130) + ";" + boxStr(b));
  }
  {
    const float b[4] = {50, 218, 150, 318};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // y1 = ceil(333) → 320；y0 = floor(203) = 203
    row("roi-bottom-edge", geomOk(r, 35, 203, 130, 117) && r.clamped,
        roiStr(r) + ";" + expStr(35, 203, 130, 117) + ";" + boxStr(b));
  }
  {
    const float b[4] = {50, 60, 150, 160};
    const RoiRect r = LprRoiFromBox(b, 320, 320, 0.0f);
    row("roi-no-expand", geomOk(r, 50, 60, 100, 100) && !r.clamped,
        roiStr(r) + ";" + expStr(50, 60, 100, 100) + ";" + boxStr(b));
  }
  {
    // 小数框、内部、无 clamp —— 专门盯"取整方向"。
    // 向外取整 → 15,25,146,156（131x131）；向内取整会给 145,155（130x130）→ 红。
    const float b[4] = {30.4f, 40.6f, 130.4f, 140.6f};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    row("roi-round-outward", geomOk(r, 15, 25, 131, 131) && !r.clamped,
        roiStr(r) + ";" + expStr(15, 25, 131, 131) + ";" + boxStr(b));
  }
  {
    // x/y 各自按自身边长外扩：bw=50 → ex=7.5，bh=200 → ey=30
    const float b[4] = {10, 10, 60, 210};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // x0=floor(2.5)=2, y0=floor(-20)→0, x1=ceil(67.5)=68, y1=ceil(240)=240
    row("roi-tall-thin", geomOk(r, 2, 0, 66, 240) && r.clamped,
        roiStr(r) + ";" + expStr(2, 0, 66, 240) + ";" + boxStr(b));
  }

  // ---- 退化输入：必须"不可用"，而不是崩或产出空 ROI ----
  {
    const float b[4] = {10, 10, 10, 20};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    row("roi-zero-area", !r.valid, roiStr(r) + ";" + boxStr(b));
  }
  {
    const float b[4] = {400, 400, 500, 500};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    // x0=385 > imgW=320 → w 为负 → 必须收敛成 invalid
    row("roi-fully-outside", !r.valid, roiStr(r) + ";" + boxStr(b));
  }
  {
    const float b[4] = {100, 100, 200, 200};
    const RoiRect r = LprRoiFromBox(b, 0, 0, kE);
    row("roi-bad-src-size", !r.valid, roiStr(r) + ";" + boxStr(b));
  }
  {
    const float b[4] = {100, 100, 200, 200};
    const RoiRect r = LprRoiFromBox(b, 320, 320, -0.1f);
    row("roi-negative-expand", !r.valid, roiStr(r) + ";" + boxStr(b));
  }

  // ---- 覆盖性：ROI 必须盖住 (车框 ∩ 图) ----
  {
    const float b[4] = {100, 100, 200, 200};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    row("cover-center", r.ContainsBox(b), roiStr(r));
  }
  {
    const float b[4] = {2, 50, 102, 150};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    row("cover-left-edge", r.ContainsBox(b), roiStr(r));
  }
  {
    // 框顶出图外：只有 300..320 那 20 px 是真的要盖住的
    const float b[4] = {300, 300, 400, 400};
    const RoiRect r = LprRoiFromBox(b, 320, 320, kE);
    row("cover-poking-out", r.ContainsBox(b), roiStr(r) + ";exp=285,285,35,35");
  }
  {
    // 反向用例：把 ROI 缩到**真的盖不住**车框，覆盖性必须变假 ——
    // 否则这个谓词就是恒真，前三条通过也没有意义。
    //
    // ⚠️ 不能只缩 1 px：ROI 有 15% 的外扩余量（这里每边 15 px），缩 1 px 仍然盖得住，
    // 断言会"假失败"（第一版就是这么写错的，设备上实测 cover-negative;ok=0）。
    // 缩到 x0+w < ceil(box x2) 才真正构成反例。
    const float b[4] = {100, 100, 200, 200};
    RoiRect r = LprRoiFromBox(b, 320, 320, kE);  // → 85,85,130,130
    r.w -= 30;                                   // → 100，x0+w = 185 < 200
    r.h -= 30;                                   // → 100，y0+h = 185 < 200
    row("cover-negative", !r.ContainsBox(b), roiStr(r));
  }
  {
    // 边界语义：x0+w == ceil(box x2) 时算"盖住"（判据用 >=，闭区间）。
    // 把这条边界约定钉下来，免得以后有人把 >= 改成 > 而悄悄放过一条边线像素。
    const float b[4] = {100, 100, 200, 200};
    RoiRect r = LprRoiFromBox(b, 320, 320, kE);  // → 85,85,130,130
    r.w = 200 - r.x0;                            // → 115，x0+w = 200 == box x2
    r.h = 200 - r.y0;
    row("cover-boundary-inclusive", r.ContainsBox(b), roiStr(r));
  }

  // ---- 坐标映射 ----
  const float mbox[4] = {100, 100, 200, 200};
  const RoiRect mroi = LprRoiFromBox(mbox, 320, 320, kE);
  {
    int rc[4] = {0, 0, 10, 10};
    LprRoiMapRect(rc, mroi);
    const bool ok = rc[0] == 85 && rc[1] == 85 && rc[2] == 95 && rc[3] == 95;
    row("map-int", ok, "got=" + std::to_string(rc[0]) + "," + std::to_string(rc[1]) + "," +
                            std::to_string(rc[2]) + "," + std::to_string(rc[3]) +
                            ";exp=85,85,95,95");
  }
  {
    float rf[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    LprRoiMapRect(rf, mroi);
    const bool ok = rf[0] == 86.5f && rf[1] == 87.5f && rf[2] == 88.5f && rf[3] == 89.5f;
    row("map-float", ok, "x1=" + std::to_string(rf[0]) + ";exp=86.5,87.5,88.5,89.5");
  }
  {
    float rf[4] = {3, 4, 7, 9};
    LprRoiMapRect(rf, mroi);
    LprRoiUnmapRect(rf, mroi);
    const bool ok = rf[0] == 3 && rf[1] == 4 && rf[2] == 7 && rf[3] == 9;
    row("map-roundtrip", ok, "x1=" + std::to_string(rf[0]) + ";exp=3");
  }
  {
    std::vector<VehicleBox> bs(2);
    bs[0].rect[0] = 0; bs[0].rect[1] = 0; bs[0].rect[2] = 10; bs[0].rect[3] = 10;
    bs[1].rect[0] = 5; bs[1].rect[1] = 5; bs[1].rect[2] = 15; bs[1].rect[3] = 15;
    LprRoiMapRects(bs, mroi);
    const bool ok = bs[0].rect[0] == 85 && bs[0].rect[3] == 95 && bs[1].rect[0] == 90 &&
                    bs[1].rect[2] == 100;
    row("map-vehicleboxes", ok, "b0x1=" + std::to_string(bs[0].rect[0]) + ";exp=85");
  }

  // ---- 裁剪：逐字节一致（跨实现证据另由 rgbSum 提供）----
  {
    const float b[4] = {W * 0.25f, H * 0.25f, W * 0.75f, H * 0.75f};
    const RoiRect r = LprRoiFromBox(b, W, H, kE);
    RgbaImage crop;
    std::string err;
    const bool okCrop = LprCropRoi(src, r, crop, err);
    const bool ok = okCrop && !r.clamped && RoiBytesEqual(src, crop, r);
    row("crop-interior-bytes", ok,
        roiStr(r) + ";rgbSum=" + std::to_string(LprRgbSum(crop)) + ";" + boxStr(b) +
            ";err=" + err);
  }
  {
    const float b[4] = {0, 0, W * 0.5f, H * 0.5f};
    const RoiRect r = LprRoiFromBox(b, W, H, kE);
    RgbaImage crop;
    std::string err;
    const bool okCrop = LprCropRoi(src, r, crop, err);
    const bool ok = okCrop && r.clamped && RoiBytesEqual(src, crop, r);
    row("crop-clamped-bytes", ok,
        roiStr(r) + ";rgbSum=" + std::to_string(LprRgbSum(crop)) + ";" + boxStr(b) +
            ";err=" + err);
  }
  {
    RoiRect bad;
    RgbaImage crop;
    std::string err;
    const bool ok = !LprCropRoi(src, bad, crop, err) && !err.empty();
    row("crop-invalid-roi", ok, "err=" + err);
  }
  {
    RoiRect bad;
    bad.imgW = W; bad.imgH = H;
    bad.x0 = 0; bad.y0 = 0; bad.w = W + 10; bad.h = 1; bad.valid = true;
    RgbaImage crop;
    std::string err;
    const bool ok = !LprCropRoi(src, bad, crop, err) && !err.empty();
    row("crop-out-of-source", ok, "err=" + err);
  }
  {
    RgbaImage empty;
    const float tiny[4] = {0, 0, 10, 10};
    RoiRect r = LprRoiFromBox(tiny, 10, 10, 0.0f);
    RgbaImage crop;
    std::string err;
    const bool ok = !LprCropRoi(empty, r, crop, err) && !err.empty();
    row("crop-empty-src", ok, "err=" + err);
  }

  rep += "total=" + std::to_string(total) + ";failed=" + std::to_string(failed) + "\n";
  return rep;
}

// ---------------------------------------------------------------- ROI 路径串联（T4）

/** 整数框的 IoU（车牌框是整数像素）。 */
static float PlateRectIou(const int a[4], const int b[4]) {
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

int LprDedupePlates(std::vector<PlateResult>& plates, float iouThresh) {
  const size_t n = plates.size();
  if (n <= 1) {
    return static_cast<int>(n);
  }
  // 先按分数降序 —— 保证"保留高分"且与输入顺序无关。
  std::vector<int> order(n);
  for (size_t i = 0; i < n; i++) {
    order[i] = static_cast<int>(i);
  }
  std::stable_sort(order.begin(), order.end(), [&plates](int a, int b) {
    return plates[a].detScore > plates[b].detScore;
  });
  std::vector<PlateResult> kept;
  kept.reserve(n);
  for (int idx : order) {
    const PlateResult& p = plates[idx];
    bool dup = false;
    for (const PlateResult& k : kept) {
      if (PlateRectIou(p.rect, k.rect) >= iouThresh) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      kept.push_back(p);
    }
  }
  plates.swap(kept);
  return static_cast<int>(plates.size());
}

namespace {
PlateResult MkTestPlate(int x1, int y1, int x2, int y2, float score, int owner = -1) {
  PlateResult p;
  p.rect[0] = x1;
  p.rect[1] = y1;
  p.rect[2] = x2;
  p.rect[3] = y2;
  p.detScore = score;
  p.ownerVeh = owner;
  return p;
}
}  // namespace

std::string LprDedupeSelfTest() {
  std::string rep;
  int total = 0;
  int failed = 0;
  const float kT = 0.5f;  // D3 的阈值

  auto row = [&rep, &total, &failed](const char* name, bool ok, const std::string& detail) {
    total++;
    if (!ok) {
      failed++;
    }
    rep += "case=";
    rep += name;
    rep += ";ok=";
    rep += (ok ? "1" : "0");
    if (!detail.empty()) {
      rep += ";";
      rep += detail;
    }
    rep += "\n";
  };

  auto scores = [](const std::vector<PlateResult>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
      if (i) s += " ";
      s += std::to_string(v[i].detScore);
    }
    return s;
  };

  // 1) 完全重合 → 只留一条，且是高分那条
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.5f), MkTestPlate(0, 0, 100, 100, 0.9f)};
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 1 && v.size() == 1 && v[0].detScore == 0.9f;
    row("dedupe-exact-dup", ok, "kept=" + std::to_string(n) + ";scores=" + scores(v) +
                                    ";exp=1 条且 score=0.9");
  }
  // 2) IoU = 0.667 ≥ 0.5 → 判重（重叠车框的典型情形）
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.9f), MkTestPlate(20, 0, 120, 100, 0.8f)};
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 1 && v[0].detScore == 0.9f;
    row("dedupe-iou-0667", ok, "kept=" + std::to_string(n) + ";scores=" + scores(v) +
                                   ";exp=1（IoU 0.667 >= 0.5）");
  }
  // 3) IoU = 0.333 < 0.5 → 不判重（两条都是真的）
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.9f), MkTestPlate(50, 0, 150, 100, 0.8f)};
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 2;
    row("dedupe-iou-0333", ok, "kept=" + std::to_string(n) + ";exp=2（IoU 0.333 < 0.5）");
  }
  // 4) 三条链式重叠 + 一条独立 → 留 2（A 与 C）
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.9f),     // A
                               MkTestPlate(10, 0, 110, 100, 0.8f),    // B：与 A IoU 0.818 → 丢
                               MkTestPlate(200, 0, 300, 100, 0.7f)};  // C：与 A 无交 → 留
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 2 && v[0].detScore == 0.9f && v[1].detScore == 0.7f;
    row("dedupe-chain", ok, "kept=" + std::to_string(n) + ";scores=" + scores(v) +
                                ";exp=2 条 0.9/0.7");
  }
  // 5) 与输入顺序无关（把第 4 条的顺序打乱）
  {
    std::vector<PlateResult> v{MkTestPlate(200, 0, 300, 100, 0.7f),
                               MkTestPlate(10, 0, 110, 100, 0.8f),
                               MkTestPlate(0, 0, 100, 100, 0.9f)};
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 2 && v[0].detScore == 0.9f && v[1].detScore == 0.7f;
    row("dedupe-order-independent", ok, "kept=" + std::to_string(n) + ";scores=" + scores(v) +
                                           ";exp=2 条 0.9/0.7（与输入顺序无关）");
  }
  // 6) 只挨着、不重叠（IoU = 0）→ 不算重
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.9f), MkTestPlate(100, 0, 200, 100, 0.8f)};
    const int n = LprDedupePlates(v, kT);
    row("dedupe-touch-edges", n == 2, "kept=" + std::to_string(n) +
                                          ";exp=2（边界相接 IoU=0，不是重复）");
  }
  // 7) 空输入不崩
  {
    std::vector<PlateResult> v;
    const int n = LprDedupePlates(v, kT);
    row("dedupe-empty", n == 0, "kept=" + std::to_string(n));
  }
  // 8) 单条输入原样返回
  {
    std::vector<PlateResult> v{MkTestPlate(5, 6, 50, 40, 0.77f)};
    const int n = LprDedupePlates(v, kT);
    row("dedupe-single", n == 1 && v[0].detScore == 0.77f, "kept=" + std::to_string(n));
  }
  // 9) 归属随保留者一起留下（去重不能把 ownerVeh 丢掉，否则界面连不出归属线）
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 100, 100, 0.5f, 3), MkTestPlate(0, 0, 100, 100, 0.9f, 7)};
    const int n = LprDedupePlates(v, kT);
    row("dedupe-keeps-owner", n == 1 && v[0].ownerVeh == 7,
        "kept=" + std::to_string(n) + ";owner=" + std::to_string(v[0].ownerVeh) +
            ";exp=owner=7（高分那条的归属）");
  }
  // 10) 输出保持分数降序
  {
    std::vector<PlateResult> v{MkTestPlate(0, 0, 10, 10, 0.2f), MkTestPlate(50, 50, 60, 60, 0.9f),
                               MkTestPlate(200, 200, 210, 210, 0.5f)};
    const int n = LprDedupePlates(v, kT);
    const bool ok = n == 3 && v[0].detScore == 0.9f && v[1].detScore == 0.5f && v[2].detScore == 0.2f;
    row("dedupe-sorted-desc", ok, "scores=" + scores(v) + ";exp=0.9 0.5 0.2");
  }

  rep += "total=" + std::to_string(total) + ";failed=" + std::to_string(failed) + "\n";
  return rep;
}

bool LprRunRoiPipeline(const RgbaImage& img, const LprSessions& s, MsSession* veh,
                       const RoiPipelineOptions& opt, std::vector<PlateResult>& outPlates,
                       std::vector<VehicleBox>& outVehicles, bool& outVehTruncated,
                       RoiPipelineStats& outStats, std::string& err) {
  outPlates.clear();
  outVehicles.clear();
  outVehTruncated = false;
  outStats = RoiPipelineStats();
  if (!img.Valid()) {
    err = "roiPipeline: invalid image";
    return false;
  }
  const double t0 = NowMs();

  // 1) 车辆检测：D1 的两条优化都在这里（低阈值 + 遍历所有框）
  std::vector<VehicleBox> vehs;
  float vehMs = 0;
  if (!LprVehicleDetect(img, veh, opt.vehConf, opt.vehIou, opt.vehicleOnly, vehs,
                        outVehTruncated, vehMs, err)) {
    return false;
  }
  outStats.vehInferMs = vehMs;
  outStats.vehCount = static_cast<int>(vehs.size());

  // 1.5) 车框预处理：跨类别去重 +（可选）top-N 截断。
  //
  // 去重放在**遍历之前**：省的是整次「裁 ROI + 车牌检测」，而不是最后那步
  // 车牌框去重 —— 同一辆车被跑两遍的代价在遍历里，不在结果里。
  outStats.vehDeduped = LprDedupeVehicles(vehs, opt.vehDedupeIou);
  if (opt.vehMaxBoxes > 0 && static_cast<int>(vehs.size()) > opt.vehMaxBoxes) {
    // LprDedupeVehicles 已按分数降序排好，直接截断即为 top-N。
    outStats.vehTruncatedByN = static_cast<int>(vehs.size()) - opt.vehMaxBoxes;
    vehs.resize(static_cast<size_t>(opt.vehMaxBoxes));
  }
  // outVehicles 给**去重后**的框：界面画的是"实际用于 ROI 的框"。画去重前的
  // 会看到同一辆车叠两个框，既乱、又与 roiTried 的账目对不上。
  outVehicles = vehs;

  // 2) 逐框：裁 ROI → 车牌检测 → 映射回原图坐标
  const double t1 = NowMs();
  for (size_t i = 0; i < vehs.size(); i++) {
    const RoiRect roi = LprRoiFromBox(vehs[i].rect, img.width, img.height, opt.roiExpand);
    if (!roi.valid) {
      outStats.roiSkipped++;
      continue;
    }
    RgbaImage crop;
    std::string cerr;
    if (!LprCropRoi(img, roi, crop, cerr)) {
      outStats.roiSkipped++;
      continue;
    }
    std::vector<PlateResult> hits;
    std::string perr;
    if (!LprRunPipeline(crop, s, hits, perr)) {
      // 单个 ROI 失败不让整帧失败 —— 一帧多车时，一个框出问题不该连累其它车。
      outStats.roiSkipped++;
      continue;
    }
    outStats.roiTried++;
    for (PlateResult& p : hits) {
      // 少了这一步，框会整体偏移 (roi.x0, roi.y0) —— T3 已证过。
      LprRoiMapRect(p.rect, roi);
      p.ownerVeh = static_cast<int>(i);
      outPlates.push_back(p);
    }
  }
  outStats.roiDetectMs = static_cast<float>(NowMs() - t1);

  // 3) 合并去重（D3）
  outStats.rawHits = static_cast<int>(outPlates.size());
  LprDedupePlates(outPlates, opt.dedupeIou);
  outStats.dedupeDropped = outStats.rawHits - static_cast<int>(outPlates.size());

  outStats.totalMs = static_cast<float>(NowMs() - t0);
  return true;
}

std::string LprOverlapDedupeProbe(const RgbaImage& img, const LprSessions& s, MsSession* veh,
                                  int growPx) {
  std::string rep;
  int total = 0;
  int failed = 0;
  auto row = [&rep, &total, &failed](const char* name, bool ok, const std::string& detail) {
    total++;
    if (!ok) {
      failed++;
    }
    rep += "case=";
    rep += name;
    rep += ";ok=";
    rep += (ok ? "1" : "0");
    if (!detail.empty()) {
      rep += ";";
      rep += detail;
    }
    rep += "\n";
  };
  auto finish = [&rep, &total, &failed]() {
    rep += "total=" + std::to_string(total) + ";failed=" + std::to_string(failed) + "\n";
    return rep;
  };

  if (!img.Valid()) {
    row("overlap-detect", false, "why=图无效");
    return finish();
  }

  std::vector<VehicleBox> vehs;
  bool trunc = false;
  float vehMs = 0;
  std::string err;
  if (!LprVehicleDetect(img, veh, 0.05f, 0.5f, /*vehicleOnly=*/true, vehs, trunc, vehMs, err)) {
    row("overlap-detect", false, "err=" + err);
    return finish();
  }
  if (vehs.empty()) {
    row("overlap-detect", false, "why=图上没有车辆框，构造不出重叠车框");
    return finish();
  }

  // 车框 A = 真实最高分框；车框 B = A 向四周各外扩 growPx（人为构造，与 A 必然重叠）
  const VehicleBox a = vehs[0];
  VehicleBox b = vehs[0];
  b.rect[0] -= static_cast<float>(growPx);
  b.rect[1] -= static_cast<float>(growPx);
  b.rect[2] += static_cast<float>(growPx);
  b.rect[3] += static_cast<float>(growPx);
  const VehicleBox boxes[2] = {a, b};

  std::vector<PlateResult> hits;
  std::string detail = "growPx=" + std::to_string(growPx) +
                       ";boxA=" + std::to_string(static_cast<int>(a.rect[0])) + "," +
                       std::to_string(static_cast<int>(a.rect[1])) + "," +
                       std::to_string(static_cast<int>(a.rect[2])) + "," +
                       std::to_string(static_cast<int>(a.rect[3])) + ";boxB=" +
                       std::to_string(static_cast<int>(b.rect[0])) + "," +
                       std::to_string(static_cast<int>(b.rect[1])) + "," +
                       std::to_string(static_cast<int>(b.rect[2])) + "," +
                       std::to_string(static_cast<int>(b.rect[3])) + ";";
  for (int i = 0; i < 2; i++) {
    const RoiRect roi = LprRoiFromBox(boxes[i].rect, img.width, img.height, kRoiExpandDefault);
    if (!roi.valid) {
      detail += "roi" + std::to_string(i) + "=invalid;";
      continue;
    }
    RgbaImage crop;
    std::string cerr;
    if (!LprCropRoi(img, roi, crop, cerr)) {
      detail += "roi" + std::to_string(i) + "=cropfail;";
      continue;
    }
    std::vector<PlateResult> got;
    std::string perr;
    if (!LprRunPipeline(crop, s, got, perr)) {
      detail += "roi" + std::to_string(i) + "=pipefail;";
      continue;
    }
    detail += "roi" + std::to_string(i) + "=" + std::to_string(roi.x0) + "," +
              std::to_string(roi.y0) + "," + std::to_string(roi.w) + "," +
              std::to_string(roi.h) + ";";
    for (PlateResult& p : got) {
      LprRoiMapRect(p.rect, roi);
      p.ownerVeh = i;
      hits.push_back(p);
    }
  }

  const int raw = static_cast<int>(hits.size());
  float iouBefore = 0;
  if (raw >= 2) {
    iouBefore = PlateRectIou(hits[0].rect, hits[1].rect);
  }
  const std::string codeBefore = raw >= 1 ? hits[0].code : std::string();
  const int kept = LprDedupePlates(hits, 0.5f);

  detail += "raw=" + std::to_string(raw) + ";kept=" + std::to_string(kept) +
            ";iouBefore=" + std::to_string(iouBefore) + ";codeBefore=" + codeBefore;
  if (kept >= 1) {
    detail += ";keptCode=" + hits[0].code + ";keptScore=" + std::to_string(hits[0].detScore);
  }
  if (raw < 2) {
    detail += ";why=只有一个 ROI 检出车牌，去重未被触发 —— 这条用例无效，不代表去重实现有问题";
  }
  // 期望：两个重叠 ROI 都检出同一块牌（raw=2），去重后只剩 1 条
  row("overlap-dedupe", raw == 2 && kept == 1, detail);
  return finish();
}