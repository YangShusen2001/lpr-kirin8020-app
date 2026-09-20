#include "ncnn_engine.h"

#include <net.h>
#include <gpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>

#include "lpr_pipeline.h"

namespace {

ncnn::Net g_net;
bool g_loaded = false;
bool g_use_vulkan = false;
/**
 * 权重内存必须存活到网络用完（ncnn 的内存版 load_model **只引用不拷贝**，
 * 见 net.h:106-108）。所以这里持有副本，而不是用调用方的临时 buffer。
 * std::vector 的堆内存由 operator new 分配，满足 32-bit 对齐要求。
 */
std::vector<unsigned char> g_param;
std::vector<unsigned char> g_bin;
/** GPU 实例全局只创建一次；ncnn 要求 create_gpu_instance 先于任何 Vulkan Net。 */
bool g_gpu_instance = false;

std::string Num(double v) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.4f", v);
  return std::string(buf);
}

/** L2 + maxAbs over a ncnn Mat, read as raw floats (pack layout does not matter). */
void Stats(const ncnn::Mat& m, double& l2, double& maxAbs, size_t& n) {
  const float* p = reinterpret_cast<const float*>(m.data);
  n = static_cast<size_t>(m.total()) * m.elempack;
  double acc = 0.0;
  maxAbs = 0.0;
  for (size_t i = 0; i < n; i++) {
    const double v = p[i];
    acc += v * v;
    const double a = std::fabs(v);
    if (a > maxAbs) {
      maxAbs = a;
    }
  }
  l2 = std::sqrt(acc);
}

}  // namespace

bool NcnnLoad(const std::vector<char>& param, const std::vector<char>& bin,
              bool useVulkan, std::string& err) {
  if (param.empty() || bin.empty()) {
    err = "empty param or bin";
    return false;
  }
  g_net.clear();
  g_loaded = false;
  g_use_vulkan = false;

  g_net.opt.num_threads = 4;
  g_net.opt.use_fp16_packed = false;
  g_net.opt.use_fp16_storage = false;
  g_net.opt.use_fp16_arithmetic = false;

  if (useVulkan) {
    // SimpleVK：内部 dlopen("libvulkan.so")，无需链接（ADR-008 已验证可达 Maleoon 920C）
    if (!g_gpu_instance) {
      const int gi = ncnn::create_gpu_instance();
      if (gi != 0) {
        err = "create_gpu_instance failed ret=" + std::to_string(gi);
        return false;
      }
      g_gpu_instance = true;
    }
    if (ncnn::get_gpu_count() <= 0) {
      err = "no vulkan device";
      return false;
    }
    g_net.opt.use_vulkan_compute = true;
    g_net.set_vulkan_device(ncnn::get_default_gpu_index());
  } else {
    g_net.opt.use_vulkan_compute = false;
  }

  // 两个内存 API 的语义**不一样**（net.h:80-111，net.cpp:2636-2648）：
  //   load_param_mem(const char*)      —— 文本 param，要求 NUL 结尾，返回 0 表示成功
  //   load_param(const unsigned char*) —— **二进制** param（内部走 load_param_bin），
  //                                       返回消耗字节数；拿文本喂它必失败
  //   load_model(const unsigned char*) —— 权重，只引用不拷贝，返回消耗字节数，0 才是失败
  g_param.assign(param.begin(), param.end());
  g_param.push_back(0);  // load_param_mem 需要 NUL 结尾
  g_bin.assign(bin.begin(), bin.end());
  const int rp = g_net.load_param_mem(reinterpret_cast<const char*>(g_param.data()));
  if (rp != 0) {
    err = "load_param_mem ret=" + std::to_string(rp) +
          " (param len=" + std::to_string(param.size()) + ")";
    return false;
  }
  const size_t rm = g_net.load_model(g_bin.data());
  if (rm == 0) {
    err = "load_model consumed 0 bytes (bin len=" + std::to_string(bin.size()) + ")";
    return false;
  }
  g_loaded = true;
  g_use_vulkan = useVulkan;
  return true;
}

void NcnnRelease() {
  g_net.clear();
  g_loaded = false;
  g_use_vulkan = false;
  g_param.clear();
  g_bin.clear();
  if (g_gpu_instance) {
    ncnn::destroy_gpu_instance();
    g_gpu_instance = false;
  }
}

bool NcnnLoaded() {
  return g_loaded;
}

std::string NcnnGpuProbe() {
  if (!g_gpu_instance) {
    return "ok=0;error=gpu instance not created";
  }
  const int count = ncnn::get_gpu_count();
  if (count <= 0) {
    return "ok=0;error=no vulkan device;count=" + std::to_string(count);
  }
  const ncnn::GpuInfo& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
  return "ok=1;count=" + std::to_string(count) +
         ";name=" + info.device_name() +
         ";driver=" + info.driver_name() +
         ";api=" + std::to_string(info.api_version()) +
         ";fp16=" + std::to_string(info.support_fp16_packed() ? 1 : 0) +
         std::to_string(info.support_fp16_storage() ? 1 : 0) +
         std::to_string(info.support_fp16_arithmetic() ? 1 : 0);
}

/**
 * 「GPU 到底有没有在算」的可反驳证据。
 *
 * `effectiveVulkan` 只是选项回显 —— 我们自己把 opt 设成 true，它当然是 true。真正说明
 * 问题的是：这台设备上有几张 Vulkan 设备、以及这张图里有多少层带 Vulkan 实现
 * （没有实现的层 ncnn 会静默回落 CPU 跑，那才是「请求了 GPU 却没吃到 GPU」）。
 */
static std::string VkCoverage(const ncnn::Net& net) {
  int vk = 0;
  const std::vector<ncnn::Layer*>& layers = net.layers();
  for (size_t i = 0; i < layers.size(); i++) {
    if (layers[i]->support_vulkan) {
      vk++;
    }
  }
  std::string name = "none";
  const int count = ncnn::get_gpu_count();
  if (count > 0) {
    name = ncnn::get_gpu_info(ncnn::get_default_gpu_index()).device_name();
    for (size_t i = 0; i < name.size(); i++) {
      if (name[i] == ' ') {
        name[i] = '_';
      }
    }
  }
  return "vkLayers=" + std::to_string(vk) + "/" + std::to_string(layers.size()) +
         ";vkGpuCount=" + std::to_string(count) + ";vkGpuName=" + name;
}

std::string NcnnInfo() {
  if (!g_loaded) {
    return "ok=0;error=no network";
  }
  std::string in, out;
  for (size_t i = 0; i < g_net.input_names().size(); i++) {
    if (i > 0) in += "|";
    in += g_net.input_names()[i];
  }
  for (size_t i = 0; i < g_net.output_names().size(); i++) {
    if (i > 0) out += "|";
    out += g_net.output_names()[i];
  }
  return "ok=1;layers=" + std::to_string(g_net.layers().size()) +
         ";inputs=" + in + ";outputs=" + out +
         ";requestedVulkan=" + std::to_string(g_use_vulkan ? 1 : 0) +
         ";effectiveVulkan=" + std::to_string(g_net.opt.use_vulkan_compute ? 1 : 0) +
         ";" + VkCoverage(g_net) +
         ";gpuProbe=" + (g_use_vulkan ? NcnnGpuProbe() : "not requested");
}

std::string NcnnRunRgba(const uint8_t* rgba, int w, int h, int warmup, int repeat) {
  if (!g_loaded) {
    return "ok=0;error=no network";
  }
  if (rgba == nullptr || w <= 0 || h <= 0) {
    return "ok=0;error=bad image";
  }
  if (warmup < 0) warmup = 0;
  if (repeat < 1) repeat = 1;

  // 与 PC 参考同源的预处理：letterbox → NCHW(RGB, /255)
  RgbaImage img;
  img.width = w;
  img.height = h;
  img.data.assign(rgba, rgba + static_cast<size_t>(w) * h * 4);
  if (!img.Valid()) {
    return "ok=0;error=rgba size mismatch";
  }
  const LetterBoxed lb = LprLetterBox(img, 320);
  const std::vector<float> nchw = LprToNchw(lb.img, true);

  ncnn::Mat in(320, 320, 3);
  for (int c = 0; c < 3; c++) {
    ncnn::Mat ch = in.channel(c);
    const float* src = nchw.data() + static_cast<size_t>(c) * 320 * 320;
    for (int y = 0; y < 320; y++) {
      float* row = ch.row(y);
      std::memcpy(row, src + static_cast<size_t>(y) * 320, 320 * sizeof(float));
    }
  }

  double l2[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
  size_t nn[3] = {0, 0, 0};
  std::vector<double> times;
  times.reserve(repeat);

  for (int it = 0; it < warmup + repeat; it++) {
    ncnn::Extractor ex = g_net.create_extractor();
    auto t0 = std::chrono::steady_clock::now();
    ex.input("in0", in);
    ncnn::Mat o0, o1, o2;
    if (ex.extract("out0", o0) != 0 || ex.extract("out1", o1) != 0 ||
        ex.extract("out2", o2) != 0) {
      return "ok=0;error=extract failed at iter " + std::to_string(it);
    }
    auto t1 = std::chrono::steady_clock::now();
    if (it >= warmup) {
      times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      Stats(o0, l2[0], mx[0], nn[0]);
      Stats(o1, l2[1], mx[1], nn[1]);
      Stats(o2, l2[2], mx[2], nn[2]);
    }
  }

  double sum = 0;
  for (double t : times) sum += t;
  std::vector<double> sorted(times);
  std::sort(sorted.begin(), sorted.end());
  const double p50 = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  const double mean = times.empty() ? 0 : sum / static_cast<double>(times.size());

  std::string kv = "ok=1";
  for (int k = 0; k < 3; k++) {
    kv += ";o" + std::to_string(k) + "l2=" + Num(l2[k]) +
          ";o" + std::to_string(k) + "max=" + Num(mx[k]) +
          ";o" + std::to_string(k) + "n=" + std::to_string(nn[k]);
  }
  kv += ";warmup=" + std::to_string(warmup) +
        ";repeat=" + std::to_string(repeat) +
        ";p50Ms=" + Num(p50) +
        ";meanMs=" + Num(mean) +
        ";error=";
  return kv;
}

// ---------------------------------------------------------------- 检测旁路

bool NcnnReload(bool useVulkan, std::string& err) {
  if (g_param.empty() || g_bin.empty()) {
    err = "no cached model (load first)";
    return false;
  }
  if (g_loaded && g_use_vulkan == useVulkan) {
    return true;  // 幂等：同模式不重载
  }
  // NcnnLoad 从 g_param/g_bin 读，所以先把 NUL 尾去掉再回传，
  // 避免 bin 末尾被追加一个多余 0 字节（param 同理，NcnnLoad 内部会再补）。
  std::vector<char> p(g_param.begin(), g_param.end() - (g_param.back() == 0 ? 1 : 0));
  std::vector<char> b(g_bin.begin(), g_bin.end());
  return NcnnLoad(p, b, useVulkan, err);
}

bool NcnnDetect(const LetterBoxed& lb, std::vector<std::vector<float>>& outs,
                std::string& err) {
  outs.clear();
  if (!g_loaded) {
    err = "no network";
    return false;
  }
  if (!lb.img.Valid() || lb.img.width != 320 || lb.img.height != 320) {
    err = "expect 320x320 letterboxed image";
    return false;
  }
  const std::vector<float> nchw = LprToNchw(lb.img, true);

  ncnn::Mat in(320, 320, 3);
  for (int c = 0; c < 3; c++) {
    ncnn::Mat ch = in.channel(c);
    const float* src = nchw.data() + static_cast<size_t>(c) * 320 * 320;
    for (int y = 0; y < 320; y++) {
      std::memcpy(ch.row(y), src + static_cast<size_t>(y) * 320, 320 * sizeof(float));
    }
  }

  ncnn::Extractor ex = g_net.create_extractor();
  ex.input("in0", in);
  const char* names[3] = {"out0", "out1", "out2"};
  for (int k = 0; k < 3; k++) {
    ncnn::Mat om;
    if (ex.extract(names[k], om) != 0) {
      err = std::string("extract ") + names[k] + " failed";
      return false;
    }
    if (om.elempack != 1 || om.elemsize != sizeof(float)) {
      err = std::string(names[k]) + " unexpected pack (elempack=" +
            std::to_string(om.elempack) + " elemsize=" + std::to_string(om.elemsize) + ")";
      return false;
    }
    const int side = 40 >> k;
    if (om.dims != 3 || om.w != side || om.h != side || om.c != 45) {
      err = std::string(names[k]) + " unexpected head shape";
      return false;
    }
    // cstep can include alignment padding: copy logical channel rows only.
    std::vector<float> v(static_cast<size_t>(45) * side * side);
    for (int c = 0; c < 45; ++c) {
      const ncnn::Mat ch = om.channel(c);
      for (int y = 0; y < side; ++y) {
        const float* row = ch.row(y);
        for (int x = 0; x < side; ++x) {
          if (!std::isfinite(row[x])) {
            err = "non-finite ncnn output";
            return false;
          }
        }
        std::memcpy(v.data() + (static_cast<size_t>(c) * side + y) * side,
                    row, side * sizeof(float));
      }
    }
    outs.push_back(std::move(v));
  }
  return true;
}

// ================================================================ 通用模型槽位
//
// 识别（rpv3）与分类（litemodel）走这里，检测旁路完全不受影响（见头文件说明）。

namespace {

/**
 * 一个槽位上的网络。
 *
 * 每槽位独立持有 param/bin 副本：ncnn 的内存版 `load_model` **只引用不拷贝**
 * （net.h:106-108「external memory should be retained」），拿调用方的临时 buffer
 * 会在下一次加载时被释放 —— 这是 ADR-011 记的坑之一。
 */
struct SlotNet {
  ncnn::Net net;
  std::vector<unsigned char> param;  // 已补 NUL 尾（load_param_mem 需要）
  std::vector<unsigned char> bin;
  bool loaded = false;
  bool vulkan = false;
  std::string inName;
  std::string outName;
};

std::map<int, std::shared_ptr<SlotNet>>& Slots() {
  static std::map<int, std::shared_ptr<SlotNet>> m;
  return m;
}

/** 按 ncnn Mat 的内存序拍平（跳过 cstep/行对齐填充），并回报实际形状。 */
void FlattenMat(const ncnn::Mat& m, std::vector<float>& out, std::string& shapeKv) {
  out.clear();
  out.reserve(static_cast<size_t>(m.total()) * m.elempack);
  if (m.dims == 1) {
    const float* p = m;
    out.assign(p, p + m.w * m.elempack);
  } else if (m.dims == 2) {
    for (int y = 0; y < m.h; y++) {
      const float* p = m.row(y);
      out.insert(out.end(), p, p + m.w * m.elempack);
    }
  } else {
    for (int c = 0; c < m.c; c++) {
      const ncnn::Mat ch = m.channel(c);
      for (int y = 0; y < ch.h; y++) {
        const float* p = ch.row(y);
        out.insert(out.end(), p, p + ch.w * ch.elempack);
      }
    }
  }
  shapeKv = "dims=" + std::to_string(m.dims) + ";w=" + std::to_string(m.w) +
            ";h=" + std::to_string(m.h) + ";c=" + std::to_string(m.c) +
            ";pm=" + std::to_string(m.elempack) + ";n=" + std::to_string(out.size());
}

/** NCHW float -> ncnn::Mat(w,h,c)，逐行拷贝（cstep 对齐由 Mat 自己处理）。 */
void ToNcnnMat(const float* nchw, int c, int h, int w, ncnn::Mat& out) {
  out.create(w, h, c);
  for (int ch = 0; ch < c; ch++) {
    const float* src = nchw + static_cast<size_t>(ch) * h * w;
    ncnn::Mat plane = out.channel(ch);
    for (int y = 0; y < h; y++) {
      std::memcpy(plane.row(y), src + static_cast<size_t>(y) * w, w * sizeof(float));
    }
  }
}

}  // namespace

void NcnnReleaseSlot(int slot) {
  Slots().erase(slot);
}

bool NcnnLoadSlot(int slot, const std::vector<char>& param, const std::vector<char>& bin,
                  bool useVulkan, std::string& err) {
  if (param.empty() || bin.empty()) {
    err = "empty param or bin";
    return false;
  }

  auto net = std::make_shared<SlotNet>();
  net->net.opt.num_threads = 4;
  // Vulkan 后端默认 fp16 storage；显式关掉，保证输出是可直接逐值比对的 fp32。
  net->net.opt.use_fp16_packed = false;
  net->net.opt.use_fp16_storage = false;
  net->net.opt.use_fp16_arithmetic = false;

  if (useVulkan) {
    if (!g_gpu_instance) {
      const int gi = ncnn::create_gpu_instance();
      if (gi != 0) {
        err = "create_gpu_instance failed ret=" + std::to_string(gi);
        return false;
      }
      g_gpu_instance = true;
    }
    if (ncnn::get_gpu_count() <= 0) {
      err = "no vulkan device";
      return false;
    }
    net->net.opt.use_vulkan_compute = true;
    net->net.set_vulkan_device(ncnn::get_default_gpu_index());
  } else {
    net->net.opt.use_vulkan_compute = false;
  }

  net->param.assign(param.begin(), param.end());
  net->param.push_back(0);  // load_param_mem 需要 NUL 结尾
  net->bin.assign(bin.begin(), bin.end());

  // 内存 API 语义（ADR-011 §3）：load_param_mem 返回 0 = 成功；
  // load_model 返回「消耗字节数」，0 才是失败 —— 别按 0=成功 去判。
  const int rp = net->net.load_param_mem(reinterpret_cast<const char*>(net->param.data()));
  if (rp != 0) {
    err = "load_param_mem ret=" + std::to_string(rp);
    return false;
  }
  const size_t rm = net->net.load_model(net->bin.data());
  if (rm == 0) {
    err = "load_model consumed 0 bytes";
    return false;
  }
  if (net->net.input_names().empty() || net->net.output_names().empty()) {
    err = "network has no input/output";
    return false;
  }
  net->inName = net->net.input_names()[0];
  net->outName = net->net.output_names()[0];
  net->loaded = true;
  net->vulkan = useVulkan;
  Slots()[slot] = net;
  return true;
}

std::string NcnnInfoSlot(int slot) {
  auto it = Slots().find(slot);
  if (it == Slots().end() || !it->second->loaded) {
    return "ok=0;slot=" + std::to_string(slot) + ";error=not loaded";
  }
  const SlotNet& s = *it->second;
  std::string in, out;
  for (size_t i = 0; i < s.net.input_names().size(); i++) {
    if (i > 0) in += "|";
    in += s.net.input_names()[i];
  }
  for (size_t i = 0; i < s.net.output_names().size(); i++) {
    if (i > 0) out += "|";
    out += s.net.output_names()[i];
  }
  return "ok=1;slot=" + std::to_string(slot) +
         ";layers=" + std::to_string(s.net.layers().size()) +
         ";inputs=" + in + ";outputs=" + out +
         ";requestedVulkan=" + std::to_string(s.vulkan ? 1 : 0) +
         ";effectiveVulkan=" + std::to_string(s.net.opt.use_vulkan_compute ? 1 : 0) +
         ";" + VkCoverage(s.net) +
         ";gpuProbe=" + (s.vulkan ? NcnnGpuProbe() : "not requested");
}

bool NcnnRunSlot(int slot, const float* nchw, int c, int h, int w,
                 std::vector<float>& out, std::string& shapeKv, std::string& err) {
  auto it = Slots().find(slot);
  if (it == Slots().end() || !it->second->loaded) {
    err = "slot " + std::to_string(slot) + " not loaded";
    return false;
  }
  if (nchw == nullptr || c <= 0 || h <= 0 || w <= 0) {
    err = "bad input";
    return false;
  }
  SlotNet& s = *it->second;

  ncnn::Mat in;
  ToNcnnMat(nchw, c, h, w, in);

  ncnn::Extractor ex = s.net.create_extractor();
  ex.input(s.inName.c_str(), in);
  ncnn::Mat om;
  if (ex.extract(s.outName.c_str(), om) != 0) {
    err = "extract " + s.outName + " failed";
    return false;
  }
  if (om.elempack != 1 || om.elemsize != sizeof(float)) {
    err = "unexpected pack (elempack=" + std::to_string(om.elempack) +
          " elemsize=" + std::to_string(om.elemsize) + ")";
    return false;
  }
  FlattenMat(om, out, shapeKv);
  for (float v : out) {
    if (!std::isfinite(v)) {
      err = "non-finite output";
      return false;
    }
  }
  return true;
}

std::string NcnnBenchSlot(int slot, const float* nchw, int c, int h, int w,
                          int warmup, int repeat) {
  auto it = Slots().find(slot);
  if (it == Slots().end() || !it->second->loaded) {
    return "ok=0;error=slot not loaded";
  }
  if (warmup < 0) warmup = 0;
  if (repeat < 1) repeat = 1;
  SlotNet& s = *it->second;

  ncnn::Mat in;
  ToNcnnMat(nchw, c, h, w, in);

  std::vector<double> times;
  times.reserve(repeat);
  std::vector<float> flat;
  std::string shapeKv;
  double l2 = 0, maxAbs = 0;
  size_t n = 0;

  for (int i = 0; i < warmup + repeat; i++) {
    ncnn::Extractor ex = s.net.create_extractor();
    auto t0 = std::chrono::steady_clock::now();
    ex.input(s.inName.c_str(), in);
    ncnn::Mat om;
    if (ex.extract(s.outName.c_str(), om) != 0) {
      return "ok=0;error=extract failed at iter " + std::to_string(i);
    }
    auto t1 = std::chrono::steady_clock::now();
    if (i >= warmup) {
      times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      if (flat.empty()) {
        FlattenMat(om, flat, shapeKv);
        Stats(om, l2, maxAbs, n);
      }
    }
  }

  double sum = 0;
  for (double t : times) sum += t;
  std::vector<double> sorted(times);
  std::sort(sorted.begin(), sorted.end());
  const double p50 = sorted.empty() ? 0 : sorted[sorted.size() / 2];
  const double mean = times.empty() ? 0 : sum / static_cast<double>(times.size());

  return "ok=1;slot=" + std::to_string(slot) +
         ";layers=" + std::to_string(s.net.layers().size()) +
         ";effectiveVulkan=" + std::to_string(s.net.opt.use_vulkan_compute ? 1 : 0) +
         ";warmup=" + std::to_string(warmup) +
         ";repeat=" + std::to_string(repeat) +
         ";p50Ms=" + Num(p50) + ";meanMs=" + Num(mean) +
         ";l2=" + Num(l2) + ";maxAbs=" + Num(maxAbs) + ";n=" + std::to_string(n) +
         ";" + shapeKv + ";error=";
}
