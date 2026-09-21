/**
 * 相机实时识别页 —— 端到端流水线的「活体」入口（2026-09-18 新增）
 *
 * 为什么单独一页：Index.ets 是**离线批处理**（喂 rawfile 静态图 + 跑加速器矩阵探针），
 * 自检流程要跑几十秒。相机是交互场景，生命周期完全不同（Surface 就绪才能开流、
 * 离开页面必须停流释放），塞进同一个 build() 两边都会别扭。
 *
 * ── 数据通路（双路预览）───────────────────────────────────────────────
 * 依据：官方《实现自定义相机》「获取预览帧数据」一节（双路预览），
 *       与官方样例 harmonyos_samples/CustomCamera 的 ImageReceiverManager.ets 同构。
 *
 *   CameraInput
 *     ├─ PreviewOutput#1 ─→ XComponent Surface      只负责「显示」
 *     └─ PreviewOutput#2 ─→ ImageReceiver(JPEG)     只负责「分析」
 *                              ↓ imageArrival 事件
 *                           readLatestImage → getComponent(ComponentType.JPEG)
 *                              ↓ createImageSource → createPixelMap
 *                           rotate(getPreviewRotation(displayRotation))
 *                              ↓ readPixelsToBuffer
 *                           RGBA → lpr.pipelineAsync → 车牌串 + 分段计时
 *
 * 取帧为什么走 JPEG 而不是 YUV：
 *   ImageReceiver 只接受 YCBCR_422_SP 与 JPEG 两种格式。YUV 那一路要自己按 rowStride
 *   裁剪、再做色度上采样才能得到 RGBA，等于在测量链路上又加一段**自己写的**代码；
 *   JPEG 这一路由系统解码器出 RGBA，少一段自研代码就少一个精度争议点。
 *
 * ── 计时口径（别混）──────────────────────────────────────────────────
 *   decodeMs = 取帧 + 解码 + 旋转 + 拷进 ArrayBuffer 的整段
 *   inferMs  = lpr.pipelineAsync 端到端（letterbox→det→NMS→矫正→rec→CTC→cls）
 *   frameMs  = 两者之和 = 一帧真正占用的墙钟时间
 *
 * ⚠️ **`1000/p50(frameMs)` 不是 fps，是服务时间的倒数**（CONTEXT.md：服务时间 /
 * 到达率 / 完成率是三个量，只有后两者可以对外称 fps）。它读出的「40–50 帧」
 * 在相机路上**不可达** —— 相机侧上限实测 ~30 fps（基准档 8 窗口 29.90–30.17）。
 * 真正该看的是**余量 = 33.3 ms − frameMs**，即「我们离吃满相机帧预算还有多远」。
 *
 * ⚠️ **p50 必须分桶，否则是两种分布的混合**（2026-09-21 定位）：
 * 未检出帧不跑矫正/识别/判色，比检出帧便宜得多。同一份日志按 count 分桶实测：
 *   生产档  未检出 infer p50 20.30 ms (n=92)  vs  检出 30.32 ms (n=21)
 *   全 NPU  未检出 infer p50 13.64 ms (n=27)  vs  检出 33.14 ms (n=1)
 * 界面一度显示「延迟 20–25 ms」又「有时 13 ms」，其实是**档位 × 是否检出**
 * 两个变量在同一个窗口里混合，不是抖动。所以现在按 count 分桶统计（见 record）。
 *
 * 丢帧是**有意**的：帧到达时若上一帧还在算，直接 release 不排队。排队会让 fps 变成
 * 「输入帧率」的假象，而这里要测的是**这条流水线能跑多快**。
 */

import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { camera } from '@kit.CameraKit';
import { image } from '@kit.ImageKit';
import { display, router } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@kit.BasicServicesKit';
// 热态与电池温度：native sysroot 没有热态头文件（已查 native/sysroot/usr/include，
// 无 thermal*），只能在 ArkTS 侧取。与 Index.ets 的 RQ4 探针同一套 API。
import { thermal } from '@kit.BasicServicesKit';
import { batteryInfo } from '@kit.BasicServicesKit';
import lpr from 'libentry.so';

const TAG = 'LprCamera';
const DOMAIN = 0xD001;

// ---------------------------------------------------------------- 模型与档位

const MODEL_DET: string = 'models/y5fu_320x_head_fp32.ms';
const MODEL_REC: string = 'models/rpv3_mdict_160_r3.ms';
const MODEL_CLS: string = 'models/litemodel_cls_96x_r1.ms';

/** ncnn 版识别 / 分类（GPU 档位要用，槽位 1 = 识别、2 = 分类）。 */
const NCNN_REC_PARAM: string = 'models/rpv3_mdict_160_r3.ncnn.param';
const NCNN_REC_BIN: string = 'models/rpv3_mdict_160_r3.ncnn.bin';
const NCNN_CLS_PARAM: string = 'models/litemodel_cls_96x_r1.ncnn.param';
const NCNN_CLS_BIN: string = 'models/litemodel_cls_96x_r1.ncnn.bin';
const SLOT_REC: number = 1;
const SLOT_CLS: number = 2;
/**
 * 检测器的 ncnn 版是**全局单例**（native 侧的 g_param/g_bin），不是槽位 ——
 * 只能通过 `lpr.ncnnLoadAsync(param, bin, useVulkan)` 装载，pipeline 的 mode=2
 * 走的就是它。少了这一步，GPU 档每帧都会返回 `no cached model (load first)`。
 */
const NCNN_DET_PARAM: string = 'models/y5fu_320x_head.ncnn.param';
const NCNN_DET_BIN: string = 'models/y5fu_320x_head.ncnn.bin';

/**
 * 三个档位 —— 一次只切一件事，任两档之间的差异只能来自后端。
 *
 *   生产档：det 落 CPU 是**实测取舍**，不是偷懒。裸 head 上 NPU 是通的，但同输入同布局
 *           下 NPU 与 CPU 的 L2 有 0.08% 差，而关键点 = logit × anchor（anchor 最大 433），
 *           0.023 的 logit 差放大后≈10 px 偏移 → 把 苏ED5172 认成 苏E05172（ADR-009）。
 *           矫正对 1 像素都敏感，所以 det 不吃 NPU。
 *   NPU 档：三模型全 nnrt。用来**现场看**上面那条漂移代价 —— 速度涨、车牌串可能变。
 *   GPU 档：三模型全 Vulkan。MS Lite 的 GPU 档是编译期判否（`IsValid# GPU is not
 *           supported`）后静默回落 CPU，麒麟 8020 上唯一的真 GPU 通路是 ncnn-Vulkan
 *           （ADR-008），所以这一档走 ncnn（det 用 pipeline 的 mode=2 旁路）。
 */
const GEAR_PROD: number = 0;
const GEAR_NPU: number = 1;
const GEAR_GPU: number = 2;
/** 基准档：只取帧、只释放，**不推理** —— 用来测相机自己的出帧上限。
 *  有了它才能回答「还能不能更快」：如果这一档也只有 20 fps，那 20 就是天花板。 */
const GEAR_RAW: number = 3;
/** 只数帧档：完全不 `readLatestImage`，只用定时器统计 `imageArrival` 回调间隔分布。
 *  这是「相机上限」的唯一判据：如果 callback 停摆，说明缓冲满；如果间隔拉长，说明背压。
 *  与 GEAR_RAW 的区别：GEAR_RAW 仍然付整帧拷贝 + release 的开销，GEAR_ARRIVE_ONLY 只收 callback。 */
const GEAR_ARRIVE_ONLY: number = 4;
const GEAR_NAMES: string[] =
  ['生产 det=CPU rec=NPU cls=CPU', '全 NPU', '全 GPU(Vulkan)', '基准：只取帧不推理', '只数 callback'];
/** 分段控件用的短名 —— 五段并排会挤爆。 */
const GEAR_SHORT: string[] = ['生产', '全 NPU', '全 GPU', '基准', 'callback'];

// ---------------------------------------------------------------- iOS 配色
// 取 Apple HIG 的 system colors，暗色模式下也认得出来。
const IOS_BLUE: string = '#007AFF';
const IOS_GREEN: string = '#34C759';
const IOS_ORANGE: string = '#FF9500';
const IOS_RED: string = '#FF3B30';
const IOS_LABEL: string = '#000000';
const IOS_LABEL2: string = 'rgba(60,60,67,0.60)';
const IOS_LABEL3: string = 'rgba(60,60,67,0.30)';
const IOS_BG: string = '#F2F2F7';
const IOS_CARD: string = '#FFFFFF';
const IOS_SEP: string = 'rgba(60,60,67,0.12)';
/** 主按钮的浅色底（systemBlue 12%）。 */
const IOS_BLUE_SOFT: string = 'rgba(0,122,255,0.12)';
/** 危险按钮的浅色底（systemRed 12%）。 */
const IOS_RED_SOFT: string = 'rgba(255,59,48,0.12)';
/** 分段控件轨道色（fill/tertiarySystemFill）。 */
const IOS_TRACK: string = 'rgba(118,118,128,0.12)';

/** 分析路的目标宽度（预览 profile 里挑最接近的）。检测器输入只有 320，取太大是浪费。
 *
 * 【2026-09-21 实验结论：保持 640，不要为帧率降分辨率】
 * 试过把宽度降到 480 并把帧率请求提到 60：
 *   STREAM START analyze=480x480    FPS SET active=60-60
 *   实测 ~27 fps —— 与 640x480 下的 28-30 fps **没有差别**（甚至略低）。
 * 即 **~30 fps 是传感器/ISP 的上限，与分辨率无关**：640x480 时设 30 或 60
 * 都只给 ~30，480x480 时设 60 也只给 ~27。
 * 既然降分辨率换不到帧率，就该保持 640x480 以保住小车牌的检出率。
 * 结论：**通过降分辨率提帧率这条路走不通**，不要再试。
 *
 * ⚠️ 【2026-09-21 前提更正：这个常量从来没有过 320】
 * 曾有人记成「刚把分析路改成了 320x320」。**查全 git 历史：`ANALYZE_TARGET_W`
 * 只有 640 一个值，从未出现过 320**；真机预览档列表（`PROFILES[0]`，24 档）
 * 里也**没有 320 宽的档**（最接近是 480x480）。320 是**检测器自己的输入边长**
 * （`MODEL_DET = models/y5fu_320x_head_fp32.ms`，`inputShape=[1,320,320,3]`，
 * native 侧 `LprSessions::detSize = 320`）—— 与本常量无关，且自 T1 起未变。
 * 详见 docs 仓库 `docs/notes/camera-npu-headroom.md`「被纠正的前提」。 */
const ANALYZE_TARGET_W: number = 640;

/** 滑动窗口：帧率与 p50 用最近 N 帧算。 */
const WINDOW: number = 30;

/**
 * 模型装载后的预热帧数：前 N 帧不计入统计。
 * kernel 首次编译 + 内存首次分配会让头几帧慢数倍（实测冷启动首帧 infer=184 ms，
 * 稳态 ~20 ms）。把它们算进 p50，会把假性能当成结论 —— 预热窗口是必须的。
 *
 * ⚠️ 【2026-09-21 前提更正：10 帧远远不够，真正要等的是相机】
 * 实测（`evidence/t3_camera_30fps_release.log`）每轮开流后约 **30 s** 到达率崩到
 * 1.4–5 fps，再花十几秒爬回 28–30。10 帧 @30fps ≈ 0.33 s，**完全盖不住这段**。
 * 且根因不在我们这侧：相机 daemon 的 `setMaxFps()` 在开流后 0.4 s 报
 * `mCurMaxFps=0 -> 27`，切档后才 `27 -> 30` —— 是**相机在爬坡**，不是流水线在预热。
 * 所以现在用 `SETTLE_MS` 做稳态门（见下），`WARMUP_FRAMES` 只负责挡住 kernel 首编。
 */
const WARMUP_FRAMES: number = 10;
/**
 * 稳态门：从 `STREAM START` 起算，前 SETTLE_MS 毫秒的帧**照常消费但不进统计**。
 *
 * 为什么不是「干脆不取帧」：接收器只有 8 格，不消费会在约 0.27 s 内填满，
 * 相机随即**停止回调** `imageArrival`（历史缺陷，见 listenFrames 里 switching 分支
 * 的注释）。一旦停摆就再也没有事件能唤醒循环，读数卡会永远停在「等待首帧」。
 * 所以 settle 期必须继续 readLatestImage + release，只是不结算。
 *
 * 为什么是 45 s：实测崩到谷底再爬回 28+ 要 ~30 s（04:16:51 → 04:17:21），
 * 留 50% 余量。代价是每轮至少跑 45 s + 观测时长。
 */
const SETTLE_MS: number = 45000;
/** 是否在会话启动后主动把预览帧率蹬到设备支持的最高档。
 *  理由：单帧工作耗时已经低于帧间隔，此时唯一的加速空间在**相机出帧率**本身。
 *  实测建流前 PreviewOutput.getSupportedFrameRates() 返回空数组，
 *  必须在 session.start() 之后查（API 文档：during session running）。
 *
 *  【2026-09-21 改为 true】原为 false，理由是「自动档已最优」—— 那个结论是
 *  在 -O0 下得出的，已作废（详见下面 FPS_TARGET 的说明）。修好构建后复测：
 *  自动 20.00 fps，显式设 30-30 得 30.00 fps。**相机自动档并没有跑到传感器
 *  上限**，所以必须主动设。改动收益：基准档 20→30 fps，生产档 7→25–27 fps。 */
const WANT_HIGH_FPS: boolean = true;
/** 【实验】是否把**显示路**（XComponent 那条 960x960 PreviewOutput）从会话里摘掉。
 *  动机：现在是双路预览 —— 显示路 960x960 + 分析路 640x480 同时出流。
 *  基准档测到的 15 fps 仍然带着显示路，所以「显示路是不是在抢
 *  传感器/ISP/带宽」这一格**没被排除**。摘掉它只留分析路，就能区分
 *  「相机物理上限」与「双流互抢」。
 *  代价：摘掉后 XComponent 收不到帧，预览是黑的 —— 仅用于实验，不做产品形态。
 *
 *  【2026-09-18 的结论已作废】当时测「摘掉显示路完全没帮助」，
 *  是在 native 以 **-O0** 构建、流水线只有 ~7 fps 的条件下测的（见
 *  docs/notes/build-mode-o0-regression.md）。那时**相机被我们自己背压**——
 *  我们处理不过来，相机根本发不满，显示路的带宽争用自然看不出来。
 *  该实验因此是**混淆的**，不能用来排除「双流互抢」。
 *
 *  【2026-09-21 复测：摘掉后反而更糟】修好 -O0、并显式设 30-30 之后复测，
 *  单分析路下回调在 ~86 帧后**直接停摆**（LprCamera 再无任何输出），
 *  而双路稳定在 20.00 fps（未设帧率时）。差异方向与预期相反，且伴随
 *  接收器缓冲占满的迹象，说明瓶颈不在「显示路抢带宽」。
 *  **本假设仍未获支持，且单路本身引入了新的停摆问题**，故不采用。
 *
 *  当前默认关闭（产品形态要预览）；留这个开关是为了下次能一分钟内复查。 */
const SKIP_PREVIEW_DISPLAY: boolean = false;
/** 目标帧率（仅当 WANT_HIGH_FPS 为真时生效）。设备报的区间是 [1-30, 60-60]。
 *
 *  【2026-09-18 的结论已作废】当时记「不设（自动）22.1 fps 最优，设 30 反而
 *  降到 18.1–19.4」，同样是在 -O0 下测的 —— 那时是我们的流水线在拖后腿，
 *  设不设帧率都到不了 30，读数反映的是我们的处理速度而非相机。
 *
 *  【2026-09-21 复测（release -O2）】结论完全相反，设 30 明确更快：
 *    不设（相机自动）→ 基准档 20.00 fps（抖动 ±0.1）
 *    设 30-30        → 基准档 30.00 fps（抖动 ±0.1）   ← 采用
 *    设 60-60        → 基准档 ~29.8 fps（active 报 60-60，但传感器仍只给 30）
 *  即 **30 是本分辨率下传感器能给的稳态上限**；60 区间被接受但拿不到 60。
 *  生产档（det=CPU rec=NPU cls=CPU）在 30-30 下能跑到 25–27 fps。 */
const FPS_TARGET: number = 30;

function parseKv(s: string): Map<string, string> {
  const m: Map<string, string> = new Map<string, string>();
  const parts: string[] = s.split(';');
  for (let i = 0; i < parts.length; i++) {
    const p: string = parts[i];
    const eq: number = p.indexOf('=');
    if (eq > 0) {
      m.set(p.substring(0, eq), p.substring(eq + 1));
    }
  }
  return m;
}

function str(m: Map<string, string>, k: string): string {
  const v: string | undefined = m.get(k);
  return v === undefined ? '-' : v;
}

function num(m: Map<string, string>, k: string): number {
  const v: string | undefined = m.get(k);
  if (v === undefined) {
    return -1;
  }
  const n: number = Number.parseFloat(v);
  return Number.isNaN(n) ? -1 : n;
}

/**
 * 解析 native 的 `backends=` 字段为落点面板的显示行（ADR-0003）。
 * 与 Index.ets 里同名函数保持一致的语义；两页各留一份是本文件的既有约定
 * （parseKv / str / num 也是各自复制的）。
 *
 * 字段顺序：role, req, landed, fallback, l2, l2AsFp16, used
 * cls 行跳过：牌色自 2026-09-21 起走像素测量（ADR-0005），没有会话可调。
 */
function evidenceRows(backends: string): string[] {
  const rows: string[] = [];
  if (backends.length === 0) {
    return rows;
  }
  for (const seg of backends.split('|')) {
    const f: string[] = seg.split(',');
    if (f.length < 7) {
      continue;
    }
    const role: string = f[0];
    if (role === 'cls') {
      continue;
    }
    const req: string = f[1];
    const landed: string = f[2];
    const fallback: string = f[3];
    const l2: string = f[4];
    const used: string = f[6];
    const isNpu: boolean = landed.indexOf('NPU') >= 0 || landed.indexOf('NNRT') >= 0;
    let line: string = role + '  ' + (isNpu ? '⚡' : '  ') + 'LANDED=' + landed;
    if (req.length > 0 && req !== landed) {
      line += '  (req=' + req + (fallback.length > 0 ? ', 回落自 ' + fallback : '') + ')';
    }
    line += (used === '1') ? ('  l2=' + l2) : '  (本次未调用)';
    rows.push(line);
  }
  return rows;
}

/** 分段控件选中项的投影。条件表达式直接返回对象字面量在 ArkTS 里类型推断不稳，抽成函数。 */
function segShadow(on: boolean): ShadowOptions {
  if (on) {
    return { radius: 3, color: 'rgba(0,0,0,0.12)', offsetX: 0, offsetY: 1 };
  }
  return { radius: 0, color: 'rgba(0,0,0,0.0)', offsetX: 0, offsetY: 0 };
}

/** p50：窗口内第 50 百分位。窗口小（30），直接排序取中位。 */
function p50(xs: number[]): number {
  if (xs.length === 0) {
    return -1;
  }
  const a: number[] = xs.slice();
  a.sort((x: number, y: number) => x - y);
  return a[Math.floor(a.length / 2)];
}

@Entry
@Component
struct CameraPage {
  @State ready: boolean = false;
  @State status: string = '正在申请相机权限…';
  @State plate: string = '—';
  @State detail: string = '';
  /**
   * 默认档 = 生产档（det=CPU, rec=NPU, cls=CPU）。
   *
   * ⚠️ **默认档必须是这个，不能改成「全 NPU」。** 理由（决定性、逐字符级）：
   * `e2eMatrix` 在黄金图 `assets/samples/hlpr-test.jpg`（期望串 `苏ED5172`）上
   * 逐后端比对，历史日志 `bench30_matrix_20260918.log` 里
   *   role=det backend=cpu/gpu/kirin → LANDED=CPU   → match=1（45 条全对）
   *   role=det backend=nnrt          → LANDED=NPU   → **match=0（45 条全翻）**
   * 即 **det 落 NPU 会把 `苏ED5172` 认成 `苏E05172`**：NPU 的 0.08% 数值漂移
   * 经 anchor（≤433）放大成 ≈10 px 位移，矫正对 1 像素都敏感。
   *
   * 「全 NPU 档更快」是真的（同会话 A/B，release）：
   *   det p50 24.86 → 12.15 ms、infer 34.6 → 23.0 ms、端到端 36.6 → 29.4 ms。
   * **但那个速度是拿正确性换的** —— 相机上换来的提速没有任何意义，
   * 如果它把车牌认错。所以全 NPU 档只保留为**测速用档**，默认停在生产档。
   *
   * 注意：一次相机 A/B 曾显示两档都读对 `皖A40675`（21 次 vs 10 次）——
   * 那是**单个样本、随机车牌**，证据强度不足以推翻上面 45 次的黄金图结论。
   * 换档/换模型后要验收的是**期望串**，不是「碰巧读对一个」。
   */
  @State gear: number = GEAR_PROD;
  @State frames: number = 0;
  @State dropped: number = 0;
  /** 窗口起点（墙钟 ms）。0 = 尚未开始计时。 */
  private winStart: number = 0;
  /**
   * 开流时刻（`STREAM START` 的墙钟 ms）—— 稳态门的计时原点。
   *
   * 用**开流**而不是首次到达作原点：相机爬坡是从 sensor 出流那一刻开始的，
   * 首帧到达本身已经晚了几百毫秒（冷启动首帧 infer 184 ms），拿它当原点会把
   * 爬坡期的头几帧算进稳态。
   */
  private streamStartMs: number = 0;
  /** 稳态门是否已过（过了才开始结算窗口）。 */
  private settled: boolean = false;
  /** 本轮落盘文件路径（filesDir/camera_run.log）。空串 = 拿不到 context，只打 hilog。 */
  private runLogPath: string = '';
  /** 已落盘的运行头（避免每条 RATE 都重复写）。 */
  private runHeaderWritten: boolean = false;
  /** 本轮落盘的行数——用于在 UI/日志里自证「真的在写」，不是静默失败。 */
  private runLogLines: number = 0;
  /** 窗口序号（仅落盘用）。事后核对 hilog 有没有漏行只能靠它。 */
  private winSeq: number = 0;
  /**
   * 热态与电池温度（每窗口采样）。
   *
   * 必须记录：`CONTEXT.md` 定义热态 2→3 时 NPU 延迟 +15~20% 而 CPU <2%，
   * 所以**任何加速比不标热态就不可比**。跨轮比较只允许在同热档内做。
   *
   * native sysroot 里没有热态头文件（已查 `native/sysroot/usr/include`，无
   * `thermal*`），只能在 ArkTS 侧取 —— 与 Index.ets 的 RQ4 探针同一套 API。
   */
  @State thermalLevel: number = -1;
  @State batteryTempC: number = 0;
  /** 热态采样定时器句柄（-1 = 未启动）。 */
  private thermalTimer: number = -1;
  private arrivedInWin: number = 0;
  private doneInWin: number = 0;
  /** 预热剩余帧数：模型装载后前 N 帧不计入统计（kernel 首编不是稳态）。
   *  实测冷启动首帧 infer=184 ms，稳态 ~20 ms —— 混进去会报出假性能。 */
  private warmLeft: number = 0;
  /** 累计到达帧数（含被丢的）。丢帧率的分母必须是它，不是完成帧数。
   *  必须是 @State：它直接渲染在读数卡上（「丢帧 x / N 帧到达」），
   *  普通成员变量不会触发重绘 —— 真机上这行会一直显示切档时清零的那个 0。 */
  @State totalArrived: number = 0;
  /** 当前帧处理期间是否有新帧到达 —— 有则算完立刻接手（见 finishFrame）。 */
  private pending: boolean = false;
  /** 本窗口的真丢帧数（被顶掉的帧）。与 arrivedInWin 同窗口，不做差。 */
  private droppedInWin: number = 0;
  /** 最近一个窗口的实测到达率 / 完成率（fps）。@State：iOS 指标卡要绑它们。 */
  @State arrFps: number = 0;
  @State doneFps: number = 0;
  /** p50 分段（ms）—— 指标卡逐项展示，不再拼成一行长串。 */
  @State frameP50: number = 0;
  @State convP50: number = 0;
  @State inferP50: number = 0;
  /** 最近一次识别的单帧推理耗时（ms），用于「本帧」那一栏。 */
  @State lastInferMs: number = 0;
  /** 本帧是否真的检出车牌 —— 决定读数卡是强调色还是灰色。 */
  @State hit: boolean = false;

  /** 落点自证（ADR-0003）。直接来自 native 的 backends=，不在 UI 层重算。 */
  @State evidence: string[] = [];
  /** 指标卡第二行的分段耗时文本（conv / det / crop / rec / cls）。 */
  @State segText: string = '';
  /** 最近一帧 native 返回的分段串（det|crop|rec|cls），record() 里拼进 segText。 */
  private lastStages: string = '-';

  private ctx: common.UIAbilityContext | undefined = undefined;
  private xc: XComponentController = new XComponentController();
  private surfaceId: string = '';

  private detId: number = -1;
  private recId: number = -1;
  private clsId: number = -1;
  private loadedGear: number = -1;

  private camInput: camera.CameraInput | undefined = undefined;
  private previewOut: camera.PreviewOutput | undefined = undefined;
  private analyzeOut: camera.PreviewOutput | undefined = undefined;
  private session: camera.PhotoSession | undefined = undefined;
  private receiver: image.ImageReceiver | undefined = undefined;

  /** 上一帧还在算时置位 → 新帧直接丢，不排队。@State：状态点要绑它。 */
  @State busy: boolean = false;
  private running: boolean = false;
  /** 档位切换中：帧循环要丢帧，否则新 mode/slot 配旧会话会算出错结果。 */
  @State switching: boolean = false;
  private rotation: number = 0;
  /** 实测拿到的帧格式（JPEG / NV21）—— 与 profile 声明的格式可能不同，要如实记录。 */
  private frameKind: string = '-';
  private firstFrameLogged: boolean = false;
  /** decode 各段耗时：取帧 / 组件 / 建PixelMap / 旋转 / 读像素。 */
  private split: number[] = [0, 0, 0, 0, 0];
  /**
   * 滑窗。**必须分桶**：未检出帧不跑矫正/识别/判色，比检出帧便宜一大截，
   * 混在一个窗口里取 p50 得到的是两种分布的混合物。
   *
   * 命名口径（2026-09-21 更正，见 record 的注释）：
   *   `prep*` = **JS 侧**取帧 + 组件拷贝
   *   `rt*`   = **JS↔native 整段往返**（含 native conv + 流水线 + NAPI）
   *   `frameMsWin` = prep + rt = 一帧实际占用的墙钟（不分桶，吞吐用它）
   *
   * 早先这几个桶叫 `conv*` / `infer*`，与 `STAGE` 行的 **native** `conv=` / `infer=`
   * 同名却不同量，导致同一批数据里两个数并存、看着自相矛盾。名字必须自带口径。
   */
  private frameMsWin: number[] = [];
  private prepMsWin: number[] = [];
  /** 未检出帧的 JS↔native 往返。 */
  private rtEmptyWin: number[] = [];
  /** 检出帧的 JS↔native 往返。 */
  private rtHitWin: number[] = [];
  /** 未检出帧的 JS 侧取帧拷贝。 */
  private prepEmptyWin: number[] = [];
  /** 检出帧的 JS 侧取帧拷贝。 */
  private prepHitWin: number[] = [];

  // ------------------------------------------------------------ 生命周期

  aboutToAppear(): void {
    this.ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.initRunLog();
    this.requestCamera();
  }

  /**
   * 准备本轮证据文件，并起一个热态采样定时器。
   *
   * 文件名带**本轮启动时间戳**：native 的 `appendLine` 只有追加语义（没有截断
   * API），用固定名会把上一轮的窗口混进这一轮 —— 而「一轮 = 一个条件」正是
   * 这套证据的全部价值。每次进页面 = 一份新文件，物理上不可能混。
   * 拉取时按 `camera_run_*.log` 取最新（见 tools/pull_camera_evidence.ps1）。
   */
  private initRunLog(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      hilog.error(DOMAIN, TAG, 'RUNLOG no context — 只打 hilog，本轮无落盘');
      return;
    }
    const stamp: number = Date.now();
    this.runLogPath = c.filesDir + '/camera_run_' + stamp.toString() + '.log';
    hilog.info(DOMAIN, TAG, 'RUNLOG path=%{public}s', this.runLogPath);
    // 热态采样：2 s 一次，与窗口同频。取值本身是同步的（thermal/batteryInfo），
    // 失败只是读不到，不影响相机。
    this.thermalTimer = setInterval(() => {
      this.sampleThermal();
    }, 2000);
    this.sampleThermal();
  }

  /** 采一次热态 / 电池温度。读失败就保持上一次的值（-1 = 从未读到）。 */
  private sampleThermal(): void {
    try {
      this.thermalLevel = thermal.getThermalLevel();
    } catch (e) {
      // 保持原值
    }
    try {
      this.batteryTempC = batteryInfo.batteryTemperature / 10;  // 单位 0.1 ℃
    } catch (e) {
      // 保持原值
    }
  }

  aboutToDisappear(): void {
    if (this.thermalTimer >= 0) {
      clearInterval(this.thermalTimer);
      this.thermalTimer = -1;
    }
    this.stopCamera();
  }

  private sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve: () => void) => {
      setTimeout(() => {
        resolve();
      }, ms);
    });
  }

  private requestCamera(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      return;
    }
    const perms: Permissions[] = ['ohos.permission.CAMERA'];
    try {
      const atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
      atManager.requestPermissionsFromUser(c, perms).then((res) => {
        const auth: number = res.authResults.length > 0 ? res.authResults[0] : -1;
        hilog.info(DOMAIN, TAG, 'CAMPERM auth=%{public}d', auth);
        if (auth === 0) {
          this.status = '权限已授予，等待预览表面…';
          this.tryOpen();
        } else {
          this.status = '相机权限被拒绝（auth=' + auth.toString() + '）';
        }
      }).catch((e: Error) => {
        this.status = '权限申请失败: ' + e.message;
        hilog.error(DOMAIN, TAG, 'CAMPERM FAIL %{public}s', e.message);
      });
    } catch (e) {
      const err = e as Error;
      this.status = '权限异常: ' + err.message;
    }
  }

  /** XComponent 的 Surface 就绪 + 权限到手 → 才能开流。 */
  private tryOpen(): void {
    if (this.surfaceId.length === 0 || this.running) {
      return;
    }
    this.openCamera();
  }

  // ------------------------------------------------------------ 会话加载

  /**
   * 按档位加载三个会话。
   *
   * 原生侧按 (模型文件, 后端) 去重，所以同一份模型换后端 = 新会话。三档最多 9 个会话，
   * 远低于 kMaxSessions=20。切换档位时才重建，避免每帧都查一次。
   */
  private async ensureSessions(): Promise<boolean> {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined) {
      // 这里原来静默 return false，而调用方 applyGear() 会把 status 覆盖成
      // 「档位加载失败」—— 于是真因（拿不到 UIAbilityContext）被抹掉。
      hilog.error(DOMAIN, TAG, 'GEAR %{public}d ENSURE FAIL ctx=undefined', this.gear);
      this.status = '拿不到 UIAbilityContext';
      return false;
    }
    if (this.loadedGear === this.gear && this.detId >= 0 && this.recId >= 0 && this.clsId >= 0) {
      return true;
    }
    this.status = '加载模型（' + GEAR_NAMES[this.gear] + '）…';
    if (this.gear === GEAR_RAW || this.gear === GEAR_ARRIVE_ONLY) {
      // 基准档不推理，无需任何模型。把 id 置 0 让「已加载」判定通过。
      this.detId = 0;
      this.recId = 0;
      this.clsId = 0;
      this.loadedGear = this.gear;
      this.warmLeft = WARMUP_FRAMES;
      this.status = this.gear === GEAR_ARRIVE_ONLY ? '只数 callback' : '基准档：只取帧不推理';
      hilog.info(DOMAIN, TAG, 'GEAR %d %s 就绪（不加载模型）', this.gear, this.status);
      return true;
    }
    try {
      let detBe: string = 'cpu';
      let recBe: string = 'nnrt';
      let clsBe: string = 'cpu';
      if (this.gear === GEAR_NPU) {
        detBe = 'nnrt';
        recBe = 'nnrt';
        clsBe = 'nnrt';
      } else if (this.gear === GEAR_GPU) {
        // GPU 档：det 走 pipeline 的 mode=2 旁路（ncnn-Vulkan），rec/cls 走 ncnn 槽位。
        // MS 会话仍需存在（pipeline 要一个占位 det 会话），但不会被调用。
        detBe = 'cpu';
        recBe = 'cpu';
        clsBe = 'cpu';
      }

      const detBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_DET);
      const detRaw: string = await lpr.loadModelAsync(MODEL_DET, this.toArrayBuffer(detBytes), detBe);
      const d: Map<string, string> = parseKv(detRaw);
      const recBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_REC);
      const recRaw: string = await lpr.loadModelAsync(MODEL_REC, this.toArrayBuffer(recBytes), recBe);
      const r: Map<string, string> = parseKv(recRaw);
      const clsBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_CLS);
      const clsRaw: string = await lpr.loadModelAsync(MODEL_CLS, this.toArrayBuffer(clsBytes), clsBe);
      const k: Map<string, string> = parseKv(clsRaw);
      // 原始返回单独记一行：面板只显示得到一小段，而「raw 为空」与「raw 有内容但
      // 没有 ok 键」是两种完全不同的故障，必须能区分。
      hilog.info(DOMAIN, TAG, 'GEAR %{public}d rawlen det=%{public}d rec=%{public}d cls=%{public}d',
        this.gear, detRaw.length, recRaw.length, clsRaw.length);

      if (d.get('ok') !== '1' || r.get('ok') !== '1' || k.get('ok') !== '1') {
        this.status = '模型加载失败 det=' + str(d, 'error') + ' rec=' + str(r, 'error') +
          ' cls=' + str(k, 'error');
        // 把完整返回也记下来：面板只放得下一小段，hilog 才是可追的现场。
        hilog.error(DOMAIN, TAG, 'GEAR %{public}d LOAD FAIL det_ok=%{public}s rec_ok=%{public}s '
          + 'cls_ok=%{public}s det_err=%{public}s rec_err=%{public}s cls_err=%{public}s',
          this.gear, str(d, 'ok'), str(r, 'ok'), str(k, 'ok'),
          str(d, 'error'), str(r, 'error'), str(k, 'error'));
        return false;
      }
      this.detId = num(d, 'id');
      this.recId = num(r, 'id');
      this.clsId = num(k, 'id');

      if (this.gear === GEAR_GPU) {
        // 检测器的 ncnn 是全局单例（不是槽位）—— 先装它，否则 mode=2 每帧都报
        // `no cached model (load first)`。
        const dp: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_DET_PARAM);
        const db: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_DET_BIN);
        const ld: Map<string, string> = parseKv(await lpr.ncnnLoadAsync(
          this.toArrayBuffer(dp), this.toArrayBuffer(db), true));
        if (ld.get('ok') !== '1') {
          this.status = 'ncnn 检测器加载失败: ' + str(ld, 'error');
          hilog.error(DOMAIN, TAG, 'GEAR GPU DET NCNN FAIL %{public}s', str(ld, 'error'));
          return false;
        }
        const rp: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_REC_PARAM);
        const rb: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_REC_BIN);
        const cp: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_CLS_PARAM);
        const cb: Uint8Array = c.resourceManager.getRawFileContentSync(NCNN_CLS_BIN);
        const lr: Map<string, string> = parseKv(await lpr.ncnnLoadSlotAsync(
          SLOT_REC, this.toArrayBuffer(rp), this.toArrayBuffer(rb), true));
        const lc: Map<string, string> = parseKv(await lpr.ncnnLoadSlotAsync(
          SLOT_CLS, this.toArrayBuffer(cp), this.toArrayBuffer(cb), true));
        if (lr.get('ok') !== '1' || lc.get('ok') !== '1') {
          this.status = 'ncnn 槽位加载失败 rec=' + str(lr, 'error') + ' cls=' + str(lc, 'error');
          hilog.error(DOMAIN, TAG, 'GEAR GPU NCNN FAIL');
          return false;
        }
        hilog.info(DOMAIN, TAG, 'GEAR GPU ncnn rec_vulkan=%{public}s cls_vulkan=%{public}s',
          str(lr, 'effectiveVulkan'), str(lc, 'effectiveVulkan'));
      }

      this.loadedGear = this.gear;
      // 模型刚装载 → 给一段预热预算：kernel 首次编译/内存分配会让前几帧慢数倍，
      // 计进窗口就是假性能（实测冷启动首帧 infer=184 ms vs 稳态 ~20 ms）。
      this.warmLeft = WARMUP_FRAMES;
      hilog.info(DOMAIN, TAG,
        'GEAR %{public}d LOADED det=%{public}s rec=%{public}s cls=%{public}s warmup=%{public}d',
        this.gear, str(d, 'backend'), str(r, 'backend'), str(k, 'backend'), WARMUP_FRAMES);
      return true;
    } catch (e) {
      const err = e as Error;
      this.status = '模型加载异常: ' + err.message;
      hilog.error(DOMAIN, TAG, 'GEAR %{public}d EXCEPTION %{public}s', this.gear, err.message);
      return false;
    }
  }

  private toArrayBuffer(u8: Uint8Array): ArrayBuffer {
    const out: ArrayBuffer = new ArrayBuffer(u8.byteLength);
    new Uint8Array(out).set(u8);
    return out;
  }

  // ------------------------------------------------------------ 相机

  private openCamera(): void {
    const c: common.UIAbilityContext | undefined = this.ctx;
    if (c === undefined || this.running) {
      return;
    }
    this.running = true;
    this.status = '正在打开相机…';
    this.ensureSessions().then((ok: boolean) => {
      if (!ok) {
        this.running = false;
        return;
      }
      this.startPreviewStream(c);
    });
  }

  private startPreviewStream(c: common.UIAbilityContext): void {
    try {
      const mgr: camera.CameraManager = camera.getCameraManager(c);
      const cams: camera.CameraDevice[] = mgr.getSupportedCameras();
      let dev: camera.CameraDevice | undefined = undefined;
      for (let i = 0; i < cams.length; i++) {
        if (cams[i].cameraPosition === camera.CameraPosition.CAMERA_POSITION_BACK) {
          dev = cams[i];
          break;
        }
      }
      if (dev === undefined) {
        dev = cams.length > 0 ? cams[0] : undefined;
      }
      if (dev === undefined) {
        this.status = '没有可用镜头';
        this.running = false;
        return;
      }
      const device: camera.CameraDevice = dev;

      const cap: camera.CameraOutputCapability = mgr.getSupportedOutputCapability(device);
      const profiles: camera.Profile[] = cap.previewProfiles;
      if (profiles.length === 0) {
        this.status = '没有可用预览分辨率';
        this.running = false;
        return;
      }

      // 先如实记录设备给了哪些预览档：`format` 决定 ImageReceiver 能不能取到帧
      // （它只认 YCBCR_422_SP / JPEG），`size` 决定分析路的解码开销。分块打，
      // hilog 单条会截断。
      let dump: string = '';
      for (let i = 0; i < profiles.length; i++) {
        dump += profiles[i].size.width.toString() + 'x' + profiles[i].size.height.toString() +
          '/' + profiles[i].format.toString() + ' ';
      }
      const chunk: number = 350;
      for (let i = 0; i < dump.length; i += chunk) {
        hilog.info(DOMAIN, TAG, 'PROFILES[%{public}d] %{public}s',
          Math.floor(i / chunk), dump.substring(i, Math.min(i + chunk, dump.length)));
      }

      // 显示路：挑一个面积最接近 1280x720 的 profile。
      let showProfile: camera.Profile = profiles[0];
      let bestShow: number = -1;
      // 分析路：宽度贴近 ANALYZE_TARGET_W。
      //
      // 这里**不**要求 profile 是 JPEG：本机（nova 14 Pro）的 24 个预览档全是
      // YUV_420_SP(1003)，一个 JPEG 都没有。官方样例 CustomCamera 的做法正是
      // 「预览档用 YUV_420_SP、ImageReceiver 建 JPEG」—— 接收端声明的格式才是
      // 那条 surface 实际交付的格式。为稳妥，取帧时按魔数判断实际拿到的是 JPEG
      // 还是 NV21，两条路都支持（见 decodeComponent）。
      let analyzeProfile: camera.Profile = profiles[0];
      let bestAnalyze: number = -1;
      for (let i = 0; i < profiles.length; i++) {
        const p: camera.Profile = profiles[i];
        const area: number = p.size.width * p.size.height;
        const dShow: number = Math.abs(area - 1280 * 720);
        if (bestShow < 0 || dShow < bestShow) {
          bestShow = dShow;
          showProfile = p;
        }
        const dAn: number = Math.abs(p.size.width - ANALYZE_TARGET_W);
        if (bestAnalyze < 0 || dAn < bestAnalyze) {
          bestAnalyze = dAn;
          analyzeProfile = p;
        }
      }
      const analyze: camera.Profile = analyzeProfile;

      const input: camera.CameraInput = mgr.createCameraInput(device);
      const show: camera.PreviewOutput =
        mgr.createPreviewOutput(showProfile, this.surfaceId);

      // 相机自己能出多快？如果预览流本身就是 30 fps，那我们的上限就是 30 ——
      // 这一条决定「还能不能再快」这个问题有没有意义。
      try {
        const rates: camera.FrameRateRange[] = show.getSupportedFrameRates();
        let rmin: number = -1;
        let rmax: number = -1;
        for (let i = 0; i < rates.length; i++) {
          if (rmin < 0 || rates[i].min < rmin) {
            rmin = rates[i].min;
          }
          if (rmax < 0 || rates[i].max > rmax) {
            rmax = rates[i].max;
          }
        }
        hilog.info(DOMAIN, TAG, 'FRAME RATES n=%{public}d min=%{public}d max=%{public}d',
          rates.length, rmin, rmax);
      } catch (e) {
        hilog.info(DOMAIN, TAG, 'FRAME RATES unavailable');
      }

      const size: image.Size = { width: analyze.size.width, height: analyze.size.height };
      const recv: image.ImageReceiver =
        image.createImageReceiver(size, image.ImageFormat.JPEG, 8);

      this.camInput = input;
      this.previewOut = show;
      this.receiver = recv;

      recv.getReceivingSurfaceId().then((rid: string) => {
        const analyzeOut: camera.PreviewOutput = mgr.createPreviewOutput(analyze, rid);
        this.analyzeOut = analyzeOut;
        input.open().then(() => {
          const sess: camera.PhotoSession = mgr.createSession(camera.SceneMode.NORMAL_PHOTO);
          this.session = sess;
          sess.beginConfig();
          sess.addInput(input);
          if (!SKIP_PREVIEW_DISPLAY) {
            sess.addOutput(show);
          }
          sess.addOutput(analyzeOut);
          hilog.info(DOMAIN, TAG, 'SESSION CONFIG display=%{public}s',
            SKIP_PREVIEW_DISPLAY ? 'OFF(实验)' : 'ON');
          sess.commitConfig().then(() => {
            sess.start().then(() => {
              // 帧的旋转角要按**分析路**这条 output 问，它才是取帧来源。
              try {
                const disp: display.Display = display.getDefaultDisplaySync();
                const dispRot: number =
                  (disp.rotation === undefined ? 0 : disp.rotation) *
                    camera.ImageRotation.ROTATION_90;
                this.rotation = analyzeOut.getPreviewRotation(dispRot);
              } catch (e) {
                this.rotation = 0;
              }

              // ── 帧率协商 ────────────────────────────────────────────────
              // 预览帧率由相机流决定，不是我们能"算"出来的。到这里完成率已经
              // 追平到达率，唯一的余量就在这条流本身。
              // 关键：**必须等 session.start() 之后**才查得到区间 —— 建流前查
              // 返回空数组（实测 n=0，一度让人以为设备不支持）。
              this.negotiateFrameRate(analyzeOut, show);
              this.ready = true;
              this.status = '实时识别中 · ' + analyze.size.width.toString() + 'x' +
                analyze.size.height.toString() + ' · rot=' + this.rotation.toString();
              hilog.info(DOMAIN, TAG,
                'STREAM START show=%{public}dx%{public}d analyze=%{public}dx%{public}d rot=%{public}d',
                showProfile.size.width, showProfile.size.height,
                analyze.size.width, analyze.size.height, this.rotation);
              // 稳态门的计时原点 = **开流时刻**（不是首帧到达，见 streamStartMs 注释）。
              // 重置门与窗口，这样换档重开流时不会把上一档的 settled 带过来。
              this.streamStartMs = Date.now();
              this.settled = false;
              this.winStart = 0;
              this.winSeq = 0;
              this.writeRunHeader(analyze.size.width, analyze.size.height,
                showProfile.size.width, showProfile.size.height);
              this.dumpLine('STREAM START show=' + showProfile.size.width.toString() + 'x' +
                showProfile.size.height.toString() + ' analyze=' + analyze.size.width.toString() +
                'x' + analyze.size.height.toString() + ' rot=' + this.rotation.toString());
              this.listenFrames(recv);
            }).catch((e: Error) => {
              this.status = '会话启动失败: ' + e.message;
              hilog.error(DOMAIN, TAG, 'SESSION START FAIL %{public}s', e.message);
              this.running = false;
            });
          }).catch((e: Error) => {
            this.status = 'commitConfig 失败: ' + e.message;
            hilog.error(DOMAIN, TAG, 'COMMIT FAIL %{public}s', e.message);
            this.running = false;
          });
        }).catch((e: Error) => {
          this.status = '打开相机失败: ' + e.message;
          hilog.error(DOMAIN, TAG, 'INPUT OPEN FAIL %{public}s', e.message);
          this.running = false;
        });
      }).catch((e: Error) => {
        this.status = '取 SurfaceId 失败: ' + e.message;
        this.running = false;
      });
    } catch (e) {
      const err = e as Error;
      this.status = '相机异常: ' + err.message;
      hilog.error(DOMAIN, TAG, 'CAMERA EXCEPTION %{public}s', err.message);
      this.running = false;
    }
  }

  /** 把预览帧率蹬到设备支持的最高档；两条 output 都试，日志留证。 */
  private negotiateFrameRate(analyzeOut: camera.PreviewOutput,
    show: camera.PreviewOutput): void {
    const outs: camera.PreviewOutput[] = [analyzeOut, show];
    const names: string[] = ['analyze', 'show'];
    for (let i = 0; i < outs.length; i++) {
      try {
        const ranges: camera.FrameRateRange[] = outs[i].getSupportedFrameRates();
        let dump: string = '';
        for (let j = 0; j < ranges.length; j++) {
          dump += ranges[j].min.toString() + '-' + ranges[j].max.toString() + ' ';
        }
        hilog.info(DOMAIN, TAG, 'FPS RANGES %{public}s n=%{public}d [%{public}s]',
          names[i], ranges.length, dump);
        if (!WANT_HIGH_FPS || ranges.length === 0) {
          continue;
        }
        // 选「包含 FPS_TARGET 的区间」；没有就退到 min 最大的那个区间。
        let lo: number = -1;
        let hi: number = -1;
        for (let j = 0; j < ranges.length; j++) {
          if (ranges[j].min <= FPS_TARGET && FPS_TARGET <= ranges[j].max) {
            lo = FPS_TARGET;
            hi = FPS_TARGET;
            break;
          }
        }
        if (lo < 0) {
          let best: camera.FrameRateRange = ranges[0];
          for (let j = 0; j < ranges.length; j++) {
            if (ranges[j].max > best.max) {
              best = ranges[j];
            }
          }
          lo = best.min;
          hi = best.max;
        }
        outs[i].setFrameRate(lo, hi);
        const act: camera.FrameRateRange = outs[i].getActiveFrameRate();
        hilog.info(DOMAIN, TAG,
          'FPS SET %{public}s requested=%{public}d-%{public}d active=%{public}d-%{public}d',
          names[i], lo, hi, act.min, act.max);
      } catch (e) {
        const err = e as BusinessError;
        hilog.info(DOMAIN, TAG, 'FPS %{public}s unavailable code=%{public}d msg=%{public}s',
          names[i], err.code, err.message);
      }
    }
  }

  private listenFrames(recv: image.ImageReceiver): void {
    recv.on('imageArrival', () => {
      // 预热期：模型刚装载、kernel 首次编译的那几帧耗时是稳态的数倍
      // （实测冷启动首帧 infer=184 ms，稳态 ~20 ms）。把它们计进窗口，
      // 会把「丢帧 81/87、推理 p50 54 ms」这种假象报出来 —— 不是真实性能。
      if (this.gear === GEAR_ARRIVE_ONLY) {
        // 只数 callback：统计 arrival 间隔分布，是「相机上限」的判据。
        //
        // 【2026-09-21 更正】这里**必须先记到达**。此前它在自增 totalArrived /
        // arrivedInWin **之前**就 return 了，于是到达数与完成数都只由别的路径更新 ——
        // 而 callback 档下根本没有别的路径。结果读数卡显示的「到达 fps」其实是
        // 上一档残留的窗口值，不是回调率。我据此误判过一次「回调率 = 20 fps」。
        //
        // 到达计数必须在这里，因为「回调到达」正是这一档唯一要测的东西。
        this.totalArrived = this.totalArrived + 1;
        this.arrivedInWin = this.arrivedInWin + 1;
        // 这一档不经过 record()，所以要在这里自己结算窗口，否则永远不打 RATE。
        this.billWindow();
        this.finishFrame();
        // 【2026-09-21 第二处更正：必须把帧抽走】
        //
        // 原注释说「不 readLatestImage，不 release」，理由是「只测回调，不碰帧」。
        // **那是错的，而且它让这一档从未产出过任何证据**：接收器只有 8 格，
        // 30 fps 下约 0.27 s 就填满，相机随即**停止回调** `imageArrival` ——
        // 于是这一档测到的不是「相机能回调多快」，而是「缓冲填满用了几帧」。
        //
        // 实测（2026-09-21 sweep1）：切到本档后 12 s + 45 s + 90 s 全程
        // **一个 RATE 窗口都没打出来**；同时 hilog 里
        // `all buffer are using` 持续刷屏。归档日志里也**从来没有任何 `gear=4` 行** ——
        // 这一档从 T3 加进来起就没成功运行过。
        //
        // 修法：照常 readLatestImage + release（只为腾空缓冲），**不推理、不计入
        // 任何延迟统计**。到达计数已经在上面记过，所以「数回调」这个目的不受影响 ——
        // 被抽走的帧仍然产生过一次 imageArrival 回调。
        this.drainOne(recv);
        return;
      }
      const warm: boolean = this.warmLeft > 0;
      // 每一帧都记到达（**包括被丢的**）—— 「相机给多少」与「我们算多少」是两件事。
      // totalArrived 同时是丢帧率的分母：丢帧数只能与**到达**数比，
      // 拿完成数当分母会读出「丢帧 78 / 76」这种数学上不可能的比值。
      this.totalArrived = this.totalArrived + 1;
      if (!warm) {
        this.arrivedInWin = this.arrivedInWin + 1;
      }
      if (this.busy || this.switching) {
        // 切档中：**必须把帧抽走并释放**，不能像 busy 那样留在缓冲里。
        //
        // 理由：切一次档要连装三个模型（全 NPU 档实测 ~7 s），而接收器只有 8 格。
        // 留帧策略是给「一帧工作略超帧间隔」设计的（见下方注释），能积压的只有
        // 一两帧；但 7 s × 3.4 fps ≈ 24 帧，8 格会在 ~2.4 s 内填满。接收器缓冲
        // 一满，相机**不再回调 imageArrival** —— 于是切档完成后再没有任何事件能
        // 触发 consume，循环永久停摆：画面显示「实时识别中」，却一帧都不出。
        //
        // 真机症状（2026-09-21）：切到「全 NPU」档后 `GEAR 1 LOADED` 与
        // `apply ok=1` 都打出来了，此后 LprCamera 再无一行 STAGE/RATE，
        // 而 CAMERA 打了 8823 行 `all buffer are using`。切档前生产档一切正常。
        //
        // 这些帧本来就要丢掉（新档的 mode/slot 与旧会话不匹配，算出来是错的），
        // 所以「读出来直接 release」既不浪费信息，又让缓冲始终有空位。
        if (this.switching) {
          this.drainOne(recv);
          return;
        }
        // 上一帧还没算完 → 这一帧本轮不处理。
        //
        // 这里**故意不** readLatestImage + release：那样会把接收器缓冲里的帧抽走，
        // 当前帧算完后只能干等下一次 imageArrival。只要单帧工作耗时（实测 24–47 ms）
        // 略微超过帧间隔（1000/24 ≈ 41 ms），干等就会让完成率**直接腰斩**
        // （实测 arrive 24.0 / done 12.5 —— 正好一半，不是巧合）。
        // 改成把帧留在缓冲里（capacity 8 足够，最多积 2 帧），算完立刻接手，
        // 完成率就重新由「工作耗时」决定，而不是由「等待」决定（见 finishFrame）。
        // 只有「已经停了一帧、又来新帧把它顶掉」才是**真丢帧**。
        // 早先这里每遇 busy 就 +1，但加了「算完立刻接手」之后，
        // busy 期间到达的那帧会成为缓冲里最新的一帧被处理，
        // 把它记成丢帧会多算（实测多算 ~1368：日志 n=2760 完成，
        // 读数卡却写「到达 2990 / 丢帧 1598」——1392 ≠ 2760）。
        // 正确关系：到达 = 完成 + 丢帧（被顶掉的帧从未被处理）。
        if (this.pending) {
          this.dropped = this.dropped + 1;
          if (!warm) {
            this.droppedInWin = this.droppedInWin + 1;
          }
        }
        this.pending = true;
        return;
      }
      this.busy = true;
      this.consume(recv);
    });
  }

  /**
   * 一帧处理完毕。若期间有新帧到达（pending），立刻接手最新的那帧。
   *
   * 为什么必须「立刻接手」而不是等下一次 imageArrival：
   *   相机以 ~41 ms 的间隔出帧，单帧工作 ~24–47 ms。工作略超间隔时，
   *   如果在空闲后干等下一帧，那一帧又要再等一整个周期才来 ——
   *   完成率于是被砍到 arrive 的一半（实测 24.0 → 12.5）。
   *   接手缓冲里已有的帧，完成率就回到 ~1000/工作耗时。
   */
  private finishFrame(): void {
    this.busy = false;
    const recv: image.ImageReceiver | undefined = this.receiver;
    if (this.pending && !this.switching && recv !== undefined && this.running) {
      this.pending = false;
      this.busy = true;
      this.consume(recv);
    }
  }

  /**
   * 结算一个 2 s 窗口：算到达率 / 完成率，打 RATE 行，并落盘。
   *
   * 从 `record()` 里抽出来，因为「只数 callback」档（`GEAR_ARRIVE_ONLY`）
   * **根本不会调 `record()`** —— 它拿到回调就把帧丢掉。原来结算逻辑只写在
   * `record()` 里，于是那一档永远不打 RATE，读数卡上也永远不刷新：
   * 表面上像「回调停摆」，实际是**没有任何代码去结算**。
   * 我因此误读过一次，所以把它提成独立方法，让两条路径共用。
   *
   * 【2026-09-21 三处扩充】
   *   1. 稳态门：开流后 `SETTLE_MS` 内的窗口不结算（相机在爬坡，见 SETTLE_MS 注释）。
   *   2. 分桶 p50：检出/未检出分开报，杜绝两种分布混成一个中位数。
   *   3. 热态 + 落盘：hilog 是环形缓冲（16 MB，实测 40 分钟就滚掉），
   *      所以每个窗口都同时写进 filesDir 的文件，事后可 `hdc file recv` 取回。
   */
  private billWindow(): void {
    const now: number = Date.now();
    // ── 稳态门 ────────────────────────────────────────────────────────
    // 注意位置：必须在 winStart 之前。settle 期内 winStart 一直保持 0，
    // 于是「第一次真正开始计时」发生在门开之后，窗口起点天然干净。
    if (!this.settled) {
      if (this.streamStartMs === 0 || now - this.streamStartMs < SETTLE_MS) {
        // 仍在爬坡：重置窗口计数（本窗口的到达/完成数不进入统计），但不碰
        // totalArrived/frames/dropped —— 那些是「本轮一共收了多少帧」的累计量，
        // 与统计窗口是两件事。
        this.arrivedInWin = 0;
        this.doneInWin = 0;
        this.droppedInWin = 0;
        this.winStart = 0;
        return;
      }
      // 门刚开：清空所有统计窗口与累计量，从干净状态起算。
      // 与 warmup 结束时的重置同构（见 record）——两处都清，因为两处都可能先到。
      // 用 resetWindows() 而不是再抄一遍字段列表：抄一遍就会漏字段，
      // 而漏掉的桶会带着爬坡期的值进入稳态统计（这正是本轮要根除的病）。
      this.settled = true;
      this.resetWindows();
      this.winStart = now;
      this.arrivedInWin = 0;
      this.doneInWin = 0;
      this.droppedInWin = 0;
      const settleSec: number = SETTLE_MS / 1000;
      hilog.info(DOMAIN, TAG, 'SETTLE done after %{public}d s — stats start here gear=%{public}d',
        settleSec, this.gear);
      this.dumpLine('SETTLE done after ' + settleSec.toFixed(0) + ' s gear=' + this.gear.toString());
      return;
    }
    if (this.winStart === 0) {
      this.winStart = now;
      return;
    }
    const winMs: number = now - this.winStart;
    if (winMs < 2000) {
      return;
    }
    this.arrFps = this.arrivedInWin * 1000 / winMs;
    this.doneFps = this.doneInWin * 1000 / winMs;
    // 分桶 p50。空桶报 -1（解析侧据此判「样本不足 / 本档不推理」，不当成 0）。
    const rtHit: number = p50(this.rtHitWin);
    const rtEmpty: number = p50(this.rtEmptyWin);
    const prepHit: number = p50(this.prepHitWin);
    const prepEmpty: number = p50(this.prepEmptyWin);
    const thermalNow: number = this.thermalLevel;
    const battNow: number = this.batteryTempC;
    hilog.info(DOMAIN, TAG,
      'RATE arrive=%{public}s fps done=%{public}s fps dropped_in_win=%{public}d ' +
      'cum_arrived=%{public}d cum_done=%{public}d cum_dropped=%{public}d gear=%{public}d ' +
      'hit_n=%{public}d empty_n=%{public}d rt_hit_p50=%{public}s rt_empty_p50=%{public}s ' +
      'prep_hit_p50=%{public}s prep_empty_p50=%{public}s thermal=%{public}d battC=%{public}s',
      this.arrFps.toFixed(2), this.doneFps.toFixed(2), this.droppedInWin,
      this.totalArrived, this.frames, this.dropped, this.gear,
      this.rtHitWin.length, this.rtEmptyWin.length,
      rtHit.toFixed(2), rtEmpty.toFixed(2),
      prepHit.toFixed(2), prepEmpty.toFixed(2),
      thermalNow, battNow.toFixed(1));
    // 落盘：字段与 hilog 行逐字一致，外加窗口序号 —— 事后核对「hilog 有没有漏行」
    // 只能靠它。hilog 会丢（环形缓冲 + 高频），文件不会。
    this.winSeq = this.winSeq + 1;
    this.dumpLine('RATE win=' + this.winSeq.toString() +
      ' arrive=' + this.arrFps.toFixed(2) + ' done=' + this.doneFps.toFixed(2) +
      ' dropped_in_win=' + this.droppedInWin.toString() +
      ' cum_arrived=' + this.totalArrived.toString() +
      ' cum_done=' + this.frames.toString() +
      ' cum_dropped=' + this.dropped.toString() +
      ' gear=' + this.gear.toString() +
      ' hit_n=' + this.rtHitWin.length.toString() +
      ' empty_n=' + this.rtEmptyWin.length.toString() +
      ' rt_hit_p50=' + rtHit.toFixed(2) +
      ' rt_empty_p50=' + rtEmpty.toFixed(2) +
      ' prep_hit_p50=' + prepHit.toFixed(2) +
      ' prep_empty_p50=' + prepEmpty.toFixed(2) +
      ' thermal=' + thermalNow.toString() +
      ' battC=' + battNow.toFixed(1));
    this.winStart = now;
    this.arrivedInWin = 0;
    this.doneInWin = 0;
    this.droppedInWin = 0;
  }

  /**
   * 追加一行到本轮证据文件（filesDir/camera_run.log）。
   *
   * 为什么必须有：hilog 是 16 MB 环形缓冲，实测一轮相机跑完约 40 分钟后
   * `LprCamera` 的行会被完全滚掉（`hilog -x` 里 STAGE = 0 行）。当时只能从
   * 设备的持久化文件 `/data/log/hilog/hilog.NNN.*.gz` 里捞回来 —— 能捞到，
   * 但那是运气（依赖滚动窗口正好覆盖）。主证据必须自己落盘。
   *
   * 失败不抛：拿不到 context / 写失败都只打 hilog，不能让相机页崩掉。
   */
  private dumpLine(line: string): void {
    if (this.runLogPath.length === 0) {
      return;
    }
    try {
      lpr.appendLine(this.runLogPath, line);
      this.runLogLines = this.runLogLines + 1;
    } catch (e) {
      hilog.error(DOMAIN, TAG, 'RUNLOG append failed: %{public}s', (e as Error).message);
    }
  }

  /**
   * 写本轮运行头：把「这批数字是在什么条件下取的」一次写清。
   *
   * 判据全部来自**观测值**，不写请求值：分析路分辨率取自实际选中的 profile，
   * 落点取自 native 的 `backends=`（ADR-0003）。缺任何一条，事后都无法判断
   * 两轮之间到底哪个变量变了。
   */
  private writeRunHeader(analyzeW: number, analyzeH: number, showW: number, showH: number): void {
    if (this.runHeaderWritten) {
      return;
    }
    this.runHeaderWritten = true;
    this.dumpLine('CAMRUN BEGIN ts=' + Date.now().toString() +
      ' gear=' + this.gear.toString() +
      ' analyze=' + analyzeW.toString() + 'x' + analyzeH.toString() +
      ' show=' + showW.toString() + 'x' + showH.toString() +
      ' rotation=' + this.rotation.toString() +
      ' settleMs=' + SETTLE_MS.toString() +
      ' warmupFrames=' + WARMUP_FRAMES.toString() +
      ' window=' + WINDOW.toString() +
      ' detModel=' + MODEL_DET + ' recModel=' + MODEL_REC +
      ' analyzeTargetW=' + ANALYZE_TARGET_W.toString() +
      ' fpsTarget=' + FPS_TARGET.toString());
  }

  /**
   * 切档期间专用：把最新一帧抽出来立刻释放，只为**腾空接收器缓冲**。
   *
   * 不推理、不计任何指标。它的唯一职责是让 `imageArrival` 继续回调 ——
   * 见 listenFrames 里 switching 分支的说明。读不到帧（缓冲空）时静默返回。
   */
  private drainOne(recv: image.ImageReceiver): void {
    recv.readLatestImage().then((img: image.Image) => {
      img.release();
    }).catch(() => {
      // 缓冲空时 readLatestImage 会失败；这正是「已经腾空」的正常情况。
    });
  }

  /**
   * 切档结束后重新接上帧循环。
   *
   * **不能只调 finishFrame()**：切档期间走的是 drainOne 分支，那条路径**不置
   * `pending`**（它只是把帧丢掉），所以 finishFrame 的条件 `pending && ...`
   * 不成立，kick 落空 —— 循环仍然停着。这里显式起一帧。
   */
  private resumeFrames(): void {
    const recv: image.ImageReceiver | undefined = this.receiver;
    if (!this.running || recv === undefined || this.busy || this.switching) {
      return;
    }
    this.pending = false;
    this.busy = true;
    this.consume(recv);
  }

  /** 取一帧 → 直接交给 native（转换+旋转+推理一次做完）。 */
  private consume(recv: image.ImageReceiver): void {
    const t0: number = Date.now();
    recv.readLatestImage().then((img: image.Image) => {
      img.getComponent(image.ComponentType.JPEG).then((comp: image.Component) => {
        const buf: ArrayBuffer | undefined = comp.byteBuffer;
        if (buf === undefined) {
          img.release();
          this.finishFrame();
          return;
        }
        const tA: number = Date.now();
        this.frameKind = comp.rowStride > 0 ? 'NV21' : 'NV21?';
        const stride: number = comp.rowStride > 0 ? comp.rowStride : img.size.width;
        const w: number = img.size.width;
        const h: number = img.size.height;
        // 释放要等 native 读完 buffer —— 用副本，避免 buffer 在异步途中被回收。
        const copy: ArrayBuffer = new ArrayBuffer(buf.byteLength);
        new Uint8Array(copy).set(new Uint8Array(buf));
        const tB: number = Date.now();
        img.release();
        this.split = [tA - t0, tB - tA];
        if (this.gear === GEAR_RAW) {
          // 基准档：拿到帧就结束。这里测到的是**相机 + 取帧路径**的上限，
          // 与推理无关 —— 它给出「再优化推理还有没有意义」的答案。
          // 本档不推理，`hit` 无意义，恒传 false（只进 frameMs 与未检出桶）。
          this.finishFrame();
          this.record(tB - t0, 0, false);
          return;
        }
        if (this.gear === GEAR_ARRIVE_ONLY) {
          // 只数 callback 的档不应该走到这里（已在 imageArrival 入口返回），但防御性处理。
          this.finishFrame();
          return;
        }
        this.infer(copy, w, h, stride, tB - t0);
      }).catch((e: Error) => {
        img.release();
        hilog.error(DOMAIN, TAG, 'COMPONENT FAIL %{public}s', e.message);
        this.finishFrame();
      });
    }).catch(() => {
      this.finishFrame();
    });
  }

  private infer(nv21: ArrayBuffer, w: number, h: number, stride: number, prepMs: number): void {
    const t2: number = Date.now();
    const mode: number = this.gear === GEAR_GPU ? 2 : 0;
    const recSlot: number = this.gear === GEAR_GPU ? SLOT_REC : -1;
    const clsSlot: number = this.gear === GEAR_GPU ? SLOT_CLS : -1;
    lpr.cameraFrameAsync(nv21, w, h, stride, this.rotation,
      this.detId, this.recId, this.clsId, mode, recSlot, clsSlot).then((raw: string) => {
      const roundMs: number = Date.now() - t2;
      const kv: Map<string, string> = parseKv(raw);
      const convMs: number = num(kv, 'convMs');
      const inferMs: number = num(kv, 'inferMs');
      // 本帧是否检出 —— 分桶统计的键，必须在 record 之前定好。
      let frameHit: boolean = false;
      if (kv.get('ok') === '1') {
        const count: number = num(kv, 'count');
        frameHit = count > 0;
        const p0: string[] = str(kv, 'p0').split(',');
        const stages: string = p0.length > 9 ? p0[9] : '-';
        this.lastInferMs = inferMs;
        this.lastStages = stages;
        // 落点自证：相机档同样把 native 的 backends= 渲染出来（ADR-0003）。
        // 相机是一次一帧、每帧都重取，所以这里的值就是本帧的观测值。
        this.evidence = evidenceRows(str(kv, 'backends'));
        if (frameHit && p0.length > 0) {
          this.plate = p0[0];
          this.hit = true;
          // 本帧明细只放**本帧**的 conv/infer —— 分段由下方 segText 统一展示，
          // 两处都写 stages 会重复（实测两条灰线内容一样）。
          this.detail = '本帧 conv ' + convMs.toFixed(1) + ' + infer ' + inferMs.toFixed(1) + ' ms';
        } else {
          this.plate = '（未检测到）';
          this.hit = false;
          this.detail = '本帧 conv ' + convMs.toFixed(1) + ' + infer ' + inferMs.toFixed(1) + ' ms';
        }
        if (this.frames % 30 === 0) {
          hilog.info(DOMAIN, TAG,
            'STAGE n=%{public}d conv=%{public}s infer=%{public}s count=%{public}d ' +
            'wh=%{public}sx%{public}s rgbaSum=%{public}s stages=%{public}s',
            this.frames, str(kv, 'convMs'), str(kv, 'inferMs'), count,
            str(kv, 'w'), str(kv, 'h'), str(kv, 'rgbaSum'), stages);
          // STAGE 与 FRAME 是两条不同的抽样行（一条带分段、一条带 fps），
          // 而分析只吃文件。落盘一份，免得事后只剩混合 p50 可看。
          this.dumpLine('STAGE n=' + this.frames.toString() +
            ' conv=' + str(kv, 'convMs') + ' infer=' + str(kv, 'inferMs') +
            ' count=' + count.toString() +
            ' wh=' + str(kv, 'w') + 'x' + str(kv, 'h') +
            ' rgbaSum=' + str(kv, 'rgbaSum') + ' stages=' + stages);
        }
      } else {
        this.detail = 'infer 失败: ' + str(kv, 'error');
      }
      // 端到端一帧 = JS 侧取帧/拷贝(prepMs) + native 转换(convMs) + 流水线(inferMs)。
      this.record(prepMs, roundMs, frameHit);
      this.finishFrame();
    }).catch((e: Error) => {
      hilog.error(DOMAIN, TAG, 'CAMERA FRAME FAIL %{public}s', e.message);
      this.finishFrame();
    });
  }

  /**
   * 记一帧。`hit` = 本帧**检出**了车牌（native 的 `count > 0`）。
   *
   * ⚠️ **两个入参的口径，别再搞混（2026-09-21 自查发现命名有歧义）**：
   *   `prepMs`   = **JS 侧**取帧 + 组件拷贝的耗时（不含任何 native 工作）
   *   `roundMs`  = `cameraFrameAsync` 的**整段往返**：NAPI 进入 + native 的
   *                NV21→RGBA 转换 + 整条流水线 + NAPI 返回
   *
   * 而 `STAGE` 行里的 `conv=` / `infer=` 是 **native 侧**的 convMs / inferMs。
   * 两者**不是同一个量**：`roundMs` 比 native `inferMs` 多出 native conv 与 NAPI 开销，
   * `prepMs` 与 native `convMs` 也完全不同（一个是 JS 拷贝，一个是 NV21→RGBA）。
   *
   * 这就是为什么 RATE 的桶字段命名为 `prep_*` / `rt_*`（round-trip）而**不叫**
   * `conv_*` / `infer_*` —— 早先叫后者时，同一批数据里
   * `infer_hit_p50=39` 与 `STAGE infer p50=25.58` 并存，看着像自相矛盾，
   * 实际是两个不同口径的量。名字必须自带口径，否则下一轮还会误读。
   *
   * 为什么 `hit` 必须传进来：检出帧要多跑矫正 + 识别 + CTC + 判色，比未检出帧贵
   * 一大截。把两者混进同一个窗口取 p50，得到的是**两种分布的混合物** ——
   * 界面因此一度显示「延迟 20–25 ms」又「有时 13 ms」，看着像抖动，其实条件不同。
   * 现在按 hit 分桶；`frameMsWin` 不分桶（吞吐关心每帧实际占多少墙钟）。
   */
  private record(prepMs: number, roundMs: number, hit: boolean): void {
    const frameMs: number = prepMs + roundMs;
    if (!this.firstFrameLogged) {
      this.firstFrameLogged = true;
      hilog.info(DOMAIN, TAG,
        'FIRST FRAME kind=%{public}s prep=%{public}d round=%{public}d rot=%{public}d',
        this.frameKind, prepMs, roundMs, this.rotation);
    }
    this.frameMsWin.push(frameMs);
    this.prepMsWin.push(prepMs);
    // 基准档（GEAR_RAW）不推理，roundMs 恒为 0。**不能把它推进往返桶** ——
    // 否则 RATE 行会打出 `rt_empty_p50=0.00`，读起来像「往返只要 0 ms」，
    // 而真相是「这一档根本没推理」。空桶报 -1（N/A），由解析侧标注为 N/A。
    const hasInfer: boolean = roundMs > 0;
    if (hasInfer) {
      if (hit) {
        this.rtHitWin.push(roundMs);
        this.prepHitWin.push(prepMs);
      } else {
        this.rtEmptyWin.push(roundMs);
        this.prepEmptyWin.push(prepMs);
      }
    }
    if (this.frameMsWin.length > WINDOW) {
      this.frameMsWin.shift();
      this.prepMsWin.shift();
    }
    if (this.rtHitWin.length > WINDOW) {
      this.rtHitWin.shift();
      this.prepHitWin.shift();
    }
    if (this.rtEmptyWin.length > WINDOW) {
      this.rtEmptyWin.shift();
      this.prepEmptyWin.shift();
    }
    this.frames = this.frames + 1;
    if (this.warmLeft > 0) {
      // 预热帧：只消耗预算，不进统计（见 listenFrames 注释）。
      // 注意它只挡 kernel 首编（~0.33 s），相机爬坡那 ~30 s 由 SETTLE_MS 挡。
      this.warmLeft = this.warmLeft - 1;
      if (this.warmLeft === 0) {
        // 预热结束，窗口从干净状态起算。
        this.resetWindows();
      }
      return;
    }
    this.doneInWin = this.doneInWin + 1;

    // 到达率 vs 完成率：每 ~2 s 结算一个窗口。
    //   到达率 ≈ 完成率  → 相机就是这个速度，我们再快也没用（上限在相机）
    //   到达率 >> 完成率 → 是我们算得慢，还有优化空间
    this.billWindow();

    const f: number = p50(this.frameMsWin);
    if (f > 0) {
      this.frameP50 = f;
      this.convP50 = p50(this.prepMsWin);
      // 指标卡的「推理 p50」显示**检出帧**的那个桶（有车牌时才是用户看到延迟的
      // 场景）；未检出桶单独在 segText 里给，两个数并排，不再混。
      this.inferP50 = p50(this.rtHitWin);
      // 指标卡第二行：本帧分段 + 帧预算。
      // 注意：native 只在**检出车牌**时才回 p0（分段在 p0[9]），空帧没有分段数据，
      // 所以这里要区分「本帧未检出」和「有数据」，别把前者显示成 '-' 像故障。
      const emptyP50: number = p50(this.rtEmptyWin);
      const bucketText: string = ' · 往返 检出 ' + (this.inferP50 > 0 ? this.inferP50.toFixed(1) : '-') +
        ' / 未检出 ' + (emptyP50 > 0 ? emptyP50.toFixed(1) : '-') + ' ms';
      if (this.gear === GEAR_RAW) {
        this.segText = '本档只取帧不推理 · 相机出帧上限';
      } else if (this.lastStages === '-') {
        this.segText = 'prep ' + p50(this.prepMsWin).toFixed(1) +
          ' ms · 本帧未检出（分段仅在检出时记录）' + bucketText;
      } else {
        this.segText = 'prep ' + p50(this.prepMsWin).toFixed(1) +
          ' ms · 分段 det|crop|rec|cls = ' + this.lastStages + bucketText;
      }
    }
    if (this.frames % 30 === 0) {
      // FRAME 行也带上分桶值：它每 30 帧打一条，是最常被引用的抽样行，
      // 只报混合 p50 会再次造成「20–25 又 13」的误读。
      hilog.info(DOMAIN, TAG,
        'FRAME n=%{public}d fps=%{public}s plate=%{public}s split=%{public}s ' +
        'rt_hit_p50=%{public}s rt_empty_p50=%{public}s hit_n=%{public}d empty_n=%{public}d',
        this.frames, (1000 / f).toFixed(2), this.plate, this.split.join('|'),
        p50(this.rtHitWin).toFixed(2), p50(this.rtEmptyWin).toFixed(2),
        this.rtHitWin.length, this.rtEmptyWin.length);
    }
  }

  /** 清空全部统计窗口与累计量（预热结束 / 稳态门开启时共用）。 */
  private resetWindows(): void {
    this.frameMsWin = [];
    this.prepMsWin = [];
    this.rtEmptyWin = [];
    this.rtHitWin = [];
    this.prepEmptyWin = [];
    this.prepHitWin = [];
    this.winStart = 0;
    this.arrivedInWin = 0;
    this.doneInWin = 0;
    this.frames = 0;
    this.dropped = 0;
    this.totalArrived = 0;
    this.droppedInWin = 0;
  }

  private stopCamera(): void {
    this.running = false;
    try {
      if (this.receiver !== undefined) {
        this.receiver.off('imageArrival');
        this.receiver.release();
        this.receiver = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.session !== undefined) {
        this.session.stop();
        this.session.release();
        this.session = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.previewOut !== undefined) {
        this.previewOut.release();
        this.previewOut = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.analyzeOut !== undefined) {
        this.analyzeOut.release();
        this.analyzeOut = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.camInput !== undefined) {
        this.camInput.close();
        this.camInput = undefined;
      }
    } catch (e) {
    }
  }

  private switchGear(g: number): void {
    if (this.gear === g || this.switching) {
      return;
    }
    this.gear = g;
    // 切档期间必须丢帧：新档位的 mode/slot 与旧会话不匹配，放过去会算出错的东西。
    // 早先这里写的是 `|| this.busy` 就 return —— 30 fps 下 busy 几乎恒为真，
    // 按钮实际点不动。改成置 switching 标志、由帧循环让路。
    this.switching = true;
    this.pending = false;
    this.resetWindows();
    // 切档 = 新条件，稳态门必须重走一遍。这里的 settled 复位只是兜底：
    // 真正重开流时 STREAM START 还会再设一次（见 openCamera 的 start 回调）。
    // 但切档**不重开流**（会话是原地重建的），所以这里不复位就永远不会重新 settle，
    // 新档的前几十秒爬坡会被算成稳态 —— 这正是上一轮「全 NPU 30 fps」的成因之一。
    this.settled = false;
    this.streamStartMs = Date.now();
    this.arrFps = 0;
    this.doneFps = 0;
    // 上一档的读数必须清掉：否则切档后标题/读数卡还挂着旧档的 conv/infer 与旧车牌，
    // 看起来像新档已经出数了。plate 清成占位、hit 复位。
    this.detail = '';
    this.segText = '';
    this.lastStages = '-';
    this.frameP50 = 0;
    this.convP50 = 0;
    this.inferP50 = 0;
    this.lastInferMs = 0;
    this.plate = '（未检测到）';
    this.hit = false;
    this.status = '切换档位 → ' + GEAR_NAMES[g];
    // 新档开跑前先记一行：档位边界是事后切分证据的唯一依据。
    this.dumpLine('GEAR SWITCH -> ' + g.toString() + ' (' + GEAR_NAMES[g] + ') settle restarts');
    this.applyGear();
  }

  /** 等当前帧跑完再重建会话，避免与在飞的推理抢会话。 */
  private applyGear(): void {
    if (this.busy) {
      setTimeout(() => {
        this.applyGear();
      }, 20);
      return;
    }
    this.ensureSessions().then((ok: boolean) => {
      this.switching = false;
      this.ready = ok;
      if (ok) {
        this.status = this.gear === GEAR_ARRIVE_ONLY ? '只数 callback' : '实时识别中 · ' + GEAR_NAMES[this.gear];
      } else if (this.status.indexOf('失败') < 0 && this.status.indexOf('异常') < 0) {
        // ensureSessions 已经写过具体原因（读不到 rawfile / 原生 loadModel 报错…）
        // 时就保留它；只有它没写原因时才兜底成这句泛化文案。
        // 原实现无条件覆盖，把真因抹掉了 —— 现场只剩「档位加载失败」四个字。
        this.status = '档位加载失败';
      }
      hilog.info(DOMAIN, TAG, 'GEAR %{public}d apply ok=%{public}d status=%{public}s',
        this.gear, ok ? 1 : 0, this.status);
      // 切档完成 → 主动重新接上帧循环。
      //
      // 为什么必须显式做：切档期间走 drainOne 分支，那条路径只是把帧丢掉、
      // **不置 pending**，所以 finishFrame 里 `pending && !switching` 的接手条件
      // 不成立 —— 光调 finishFrame 会落空，循环就永久停着（画面「实时识别中」
      // 但一帧不出）。必须直接起一帧。
      //
      // 另一层原因：切档要连装三个模型（全 NPU 档实测 ~7 s），接收器只有 8 格。
      // 若切档期间完全不读，8 格会在 ~2.4 s 内填满，相机随即**停止回调**
      // imageArrival，之后没有任何事件能唤醒循环。
      if (ok) {
        this.resumeFrames();
      } else if (this.loadedGear >= 0 && this.loadedGear !== this.gear) {
        // 加载失败：会话还是上一档的，而 `this.gear` 已被 switchGear 改成新档 ——
        // 这个组合会拿旧会话去跑新档语义（mode/slot 不匹配），静默算出错的东西。
        // 把 gear 退回上一档，让「档位」与「实际会话」重新一致，再恢复帧循环，
        // 相机保持可用；面板上的 status 已经写明失败原因。
        hilog.warn(DOMAIN, TAG, 'GEAR %{public}d load failed, reverting to %{public}d',
          this.gear, this.loadedGear);
        this.gear = this.loadedGear;
        this.ready = true;
        this.resumeFrames();
      } else {
        // 没有可退回的档位（首次加载就失败）：不恢复。
        hilog.error(DOMAIN, TAG, 'GEAR %{public}d load failed with no fallback gear', this.gear);
      }
    });
  }

  // ------------------------------------------------------------ 界面

  build() {
    Column({ space: 0 }) {
      // ── 导航栏（iOS 大标题风格）
      Row() {
        Text('‹ 返回')
          .fontSize(17)
          .fontColor(IOS_BLUE)
          .onClick(() => {
            this.stopCamera();
            router.back();
          })
        Blank()
        // 状态点：跑着是绿的，切档是橙的，停了是灰的
        Row({ space: 5 }) {
          Circle()
            .width(7)
            .height(7)
            .fill(this.switching ? IOS_ORANGE :
              (this.ready && this.running ? IOS_GREEN : IOS_LABEL3))
          Text(this.switching ? '切换中' : (this.ready ? '识别中' : '未就绪'))
            .fontSize(12)
            .fontColor(IOS_LABEL2)
        }
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 10 })
      .backgroundColor(IOS_CARD)

      // ── 大标题区
      Column({ space: 3 }) {
        Text('相机实时识别')
          .fontSize(28)
          .fontWeight(FontWeight.Bold)
          .fontColor(IOS_LABEL)
        Text(this.status)
          .fontSize(12)
          .fontColor(IOS_LABEL2)
          .maxLines(2)
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start)
      .padding({ left: 16, right: 16, bottom: 12 })
      .backgroundColor(IOS_CARD)

      // ── 预览区：XComponent 只负责显示，取帧走第二路 ImageReceiver。
      XComponent({
        type: XComponentType.SURFACE,
        controller: this.xc
      })
        .onLoad(() => {
          this.surfaceId = this.xc.getXComponentSurfaceId();
          hilog.info(DOMAIN, TAG, 'SURFACE %{public}s', this.surfaceId);
          this.tryOpen();
        })
        .onDestroy(() => {
          this.stopCamera();
        })
        .width('100%')
        .layoutWeight(1)
        .backgroundColor('#000000')

      // ── 读数卡：车牌放大显示，检出错才用强调色
      Column({ space: 2 }) {
        Text(this.plate)
          .fontSize(38)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.hit ? IOS_LABEL : IOS_LABEL3)
          .letterSpacing(2)
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .width('100%')
          .textAlign(TextAlign.Center)
        Text(this.hit ? '识别结果' : '把车牌对准取景框')
          .fontSize(11)
          .fontColor(IOS_LABEL2)
        // 本帧明细：conv / infer 与分段。放在读数卡里，标题下那行只留状态，不再重复。
        Text(this.detail.length > 0 ? this.detail : '等待首帧…')
          .fontSize(11)
          .fontColor(IOS_LABEL3)
          .maxLines(2)
          .width('100%')
          .textAlign(TextAlign.Center)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 14, bottom: 12 })
      .backgroundColor(IOS_CARD)
      .margin({ top: 8 })

      // ── 指标卡：四格直接内联绑定 @State。
      // 注意：不能用 @Builder 传值参数渲染这些数字 —— ArkTS 的 @Builder
      // 按值传参只在首帧求值一次，之后 @State 变化不会重渲染该格
      // （实测四个格子冻在 0.0，而同一屏内联的 Text 正常刷新）。
      Column() {
        Row() {
          Column({ space: 1 }) {
            Row({ space: 2 }) {
              Text(this.arrFps.toFixed(1))
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .fontColor(IOS_BLUE)
                .maxLines(1)
            }
            .alignItems(VerticalAlign.Bottom)
            Text('到达 fps')
              .fontSize(11)
              .fontColor(IOS_LABEL2)
              .maxLines(1)
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)

          Divider().vertical(true).height(28).color(IOS_SEP).strokeWidth(0.5)

          Column({ space: 1 }) {
            Text(this.doneFps.toFixed(1))
              .fontSize(20)
              .fontWeight(FontWeight.Bold)
              .fontColor(IOS_GREEN)
              .maxLines(1)
            Text('完成 fps')
              .fontSize(11)
              .fontColor(IOS_LABEL2)
              .maxLines(1)
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding({ left: 10 })

          Divider().vertical(true).height(28).color(IOS_SEP).strokeWidth(0.5)

          Column({ space: 1 }) {
            Row({ space: 2 }) {
              Text(this.frameP50.toFixed(1))
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .fontColor(IOS_LABEL)
                .maxLines(1)
              Text('ms')
                .fontSize(11)
                .fontColor(IOS_LABEL3)
                .margin({ bottom: 2 })
            }
            .alignItems(VerticalAlign.Bottom)
            Text(this.gear === GEAR_RAW ? '取帧 p50' : '帧 p50')
              .fontSize(11)
              .fontColor(IOS_LABEL2)
              .maxLines(1)
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding({ left: 10 })

          Divider().vertical(true).height(28).color(IOS_SEP).strokeWidth(0.5)

          Column({ space: 1 }) {
            Row({ space: 2 }) {
              Text(this.inferP50 > 0 ? this.inferP50.toFixed(1) : '-')
                .fontSize(20)
                .fontWeight(FontWeight.Bold)
                .fontColor(IOS_ORANGE)
                .maxLines(1)
              Text('ms')
                .fontSize(11)
                .fontColor(IOS_LABEL3)
                .margin({ bottom: 2 })
            }
            .alignItems(VerticalAlign.Bottom)
            // 标签写明是**检出帧**的桶、以及这是**往返**（不是 native 单段）：
            // 混桶读数是上一轮误判「20–25 又 13」的根源；而把往返当 native infer
            // 会让 RATE 与 STAGE 两个数看着自相矛盾。
            Text('往返 p50·检出')
              .fontSize(11)
              .fontColor(IOS_LABEL2)
              .maxLines(1)
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding({ left: 10 })
        }
        .width('100%')

        // 第二行：丢帧计数 + 分段耗时（换行展示，不再挤成一行被截断）
        Row({ space: 10 }) {
          // 分母用「到达」而不是「完成」：丢帧数可能大于完成数，
          // 写成 x/frames 会显示「丢帧 78 / 76」这种不可能的比例。
          Text('丢帧 ' + this.dropped.toString() + ' / ' +
            this.totalArrived.toString() + ' 帧到达')
            .fontSize(11)
            .fontColor(IOS_LABEL2)
          Blank()
          Text(this.segText)
            .fontSize(11)
            .fontColor(IOS_LABEL3)
            .maxLines(3)
            .layoutWeight(1)
            .textAlign(TextAlign.End)
        }
        .width('100%')
        .alignItems(VerticalAlign.Top)
        .padding({ top: 10 })

        // 第三行：稳态门 + 热态 + 落盘自证。
        // 「观测代码有没有被执行」必须能当场看见：settle 未过时数字全是 0，
        // 若不在界面上说清，会和「回调停摆」长得一模一样（历史上误判过一次）。
        Row({ space: 10 }) {
          Text(this.settled
            ? ('稳态已开 · 窗口 ' + this.winSeq.toString() + ' · 落盘 ' +
               this.runLogLines.toString() + ' 行')
            : ('稳态门中 · 相机爬坡不计入（' + (SETTLE_MS / 1000).toFixed(0) + ' s）'))
            .fontSize(10)
            .fontColor(this.settled ? IOS_GREEN : IOS_ORANGE)
          Blank()
          Text('热档 ' + this.thermalLevel.toString() + ' · ' +
            this.batteryTempC.toFixed(1) + '℃')
            .fontSize(10)
            .fontColor(IOS_LABEL2)
        }
        .width('100%')
        .padding({ top: 6 })

        // 第三行：落点自证（ADR-0003）
        // 相机是榨干硬件的**真战场**（热、功耗、掉帧都在这里出现），
        // 所以落点必须和吞吐指标并排显示，而不是藏在日志里。
        if (this.evidence.length > 0) {
          Divider()
            .color(IOS_SEP)
            .margin({ top: 8, bottom: 8 })
          Text('落点自证 · LANDED')
            .fontSize(10)
            .fontColor(IOS_LABEL3)
            .width('100%')
          ForEach(this.evidence, (row: string) => {
            Text(row)
              .fontSize(10)
              .fontFamily('monospace')
              .fontColor(IOS_LABEL2)
              .width('100%')
          }, (row: string) => row)
        }
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor(IOS_CARD)
      .margin({ top: 8 })

      // ── 分段控件：四档后端
      Row() {
        ForEach(GEAR_SHORT, (name: string, idx: number) => {
          Text(name)
            .fontSize(13)
            .fontWeight(this.gear === idx ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.gear === idx ? IOS_LABEL : IOS_LABEL2)
            .textAlign(TextAlign.Center)
            .layoutWeight(1)
            .padding({ top: 7, bottom: 7 })
            .backgroundColor(this.gear === idx ? IOS_CARD : 'rgba(0,0,0,0)')
            .borderRadius(7)
            .shadow(segShadow(this.gear === idx))
            .onClick(() => {
              this.switchGear(idx);
            })
        }, (name: string) => name)
      }
      .width('100%')
      .padding(2)
      .backgroundColor('#E9E9EB')
      .borderRadius(9)
      .margin({ left: 16, right: 16, top: 10 })

      // ── 底部说明
      Text(GEAR_NAMES[this.gear])
        .fontSize(11)
        .fontColor(IOS_LABEL2)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .width('100%')
        .textAlign(TextAlign.Center)
        .padding({ left: 16, right: 16, top: 6, bottom: 14 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor(IOS_BG)
  }
}
