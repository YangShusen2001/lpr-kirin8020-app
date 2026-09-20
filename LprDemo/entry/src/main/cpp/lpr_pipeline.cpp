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

  const size_t plane = static_cast<size_t>(imgH) * targetW;
  out.resize(3 * plane);
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

    // ---- classify
    // MS Lite 报 NHWC；ncnn 的 Mat 是平面 NCHW —— 布局差别只在这一处。
    const bool clsNhwc = s.clsSlot >= 0 ? false : (s.cls->inputFormat == OH_AI_FORMAT_NHWC);
    std::vector<float>& clsIn = sc.clsIn;
    LprEncodeClassifyInto(crop, 96, clsNhwc, clsIn, sc.rs);
    std::vector<float> clsOut;
    if (s.clsSlot >= 0) {
      std::string shapeKv;
      if (!NcnnRunSlot(s.clsSlot, clsIn.data(), 3, 96, 96, clsOut, shapeKv, err)) {
        return false;
      }
    } else if (!MsRun(s.cls, clsIn.data(), clsOut, err)) {
      return false;
    }
    for (int k = 0; k < 3; k++) {
      item.cls[k] = k < static_cast<int>(clsOut.size()) ? clsOut[k] : 0.0f;
    }
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
 * NV21 -> RGBA，整数倍 90 度旋转在写入时一并完成（不额外搬一次内存）。
 *
 * BT.601 有限范围（相机预览的标准约定）：Y' 在 [16,235]，Cb/Cr 在 [16,240]。
 * 用整数近似做定点运算，避免每像素一次浮点乘 —— 640x480 是 30 万像素，
 * 在手机上这点差别是实打实的。
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

  for (int y = 0; y < height; y++) {
    const uint8_t* yRow = nv21 + static_cast<size_t>(y) * stride;
    // VU 交错：每 2 行共用一个色度行，每 2 列共用一个色度对。
    const uint8_t* uvRow = uv + static_cast<size_t>(y / 2) * stride;
    for (int x = 0; x < width; x++) {
      const int Y = yRow[x];
      const size_t uvIdx = static_cast<size_t>(x / 2) * 2;
      const int V = uvRow[uvIdx];
      const int U = uvRow[uvIdx + 1];

      // 整数 BT.601：C = Y-16, D = U-128, E = V-128
      const int c = Y - 16;
      const int d = U - 128;
      const int e = V - 128;
      int r = (298 * c + 409 * e + 128) >> 8;
      int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
      int b = (298 * c + 516 * d + 128) >> 8;
      r = r < 0 ? 0 : (r > 255 ? 255 : r);
      g = g < 0 ? 0 : (g > 255 ? 255 : g);
      b = b < 0 ? 0 : (b > 255 ? 255 : b);

      // 目标坐标按旋转角算 —— 一次写成，省掉后续的 rotate()。
      int dx = 0;
      int dy = 0;
      if (rot == 90) {
        dx = height - 1 - y;
        dy = x;
      } else if (rot == 180) {
        dx = width - 1 - x;
        dy = height - 1 - y;
      } else if (rot == 270) {
        dx = y;
        dy = width - 1 - x;
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