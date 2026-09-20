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
 *   fps      = 1000 / p50(frameMs)
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

/** 分析路的目标宽度（预览 profile 里挑最接近的）。检测器输入只有 320，取太大是浪费。 */
const ANALYZE_TARGET_W: number = 640;

/** 滑动窗口：帧率与 p50 用最近 N 帧算。 */
const WINDOW: number = 30;

/**
 * 模型装载后的预热帧数：前 N 帧不计入统计。
 * kernel 首次编译 + 内存首次分配会让头几帧慢数倍（实测冷启动首帧 infer=184 ms，
 * 稳态 ~20 ms）。把它们算进 p50，会把假性能当成结论 —— 预热窗口是必须的。
 */
const WARMUP_FRAMES: number = 10;
/** 是否在会话启动后主动把预览帧率蹬到设备支持的最高档。
 *  理由：单帧工作耗时（≈25–35 ms）已经低于帧间隔，完成率
 *  等于到达率 —— 此时唯一的加速空间在**相机出帧率**本身。
 *  实测建流前 PreviewOutput.getSupportedFrameRates() 返回空数组，
 *  必须在 session.start() 之后查（API 文档：during session running）。 */
const WANT_HIGH_FPS: boolean = false;
/** 【实验】是否把**显示路**（XComponent 那条 960x960 PreviewOutput）从会话里摘掉。
 *  动机：现在是双路预览 —— 显示路 960x960 + 分析路 640x480 同时出流。
 *  基准档测到的 15 fps 仍然带着显示路，所以「显示路是不是在抢
 *  传感器/ISP/带宽」这一格**没被排除**。摘掉它只留分析路，就能区分
 *  「相机物理上限」与「双流互抢」。
 *  代价：摘掉后 XComponent 收不到帧，预览是黑的 —— 仅用于实验，不做产品形态。
 *
 *  【实测结论 2026-09-18】摘掉显示路**完全没有帮助**：
 *    显示路 ON ：生产档 arrive 14.7–15.3 · 基准档 14.9–15.1
 *    显示路 OFF：生产档 arrive 14.8–15.4 · 基准档 14.9–15.1
 *  即「双流互抢」被排除。当前 15.00 fps（抖动仅 ±0.1）是相机在暗场景
 *  把帧率**砍半换曝光**（30→15）的典型行为，是传感器决策，不是我们的开销。
 *  因此保持关闭；留这个开关是为了下次能一分钟内复查同一假设。 */
const SKIP_PREVIEW_DISPLAY: boolean = false;
/** 目标帧率（仅当 WANT_HIGH_FPS 为真时生效）。设备报的区间是 [1-30, 60-60]。
 *  A/B 实测（同一场景、同一 release 构建）：
 *    不设（相机自动）→ arrive 22.1 fps
 *    设 30-30        → arrive 18.1–19.4 fps
 *    设 60-60        → arrive 18.5 fps
 *  也就是说**相机自动档已经是最优**，强制固定帧率会让传感器换一套曝光/出帧
 *  配置、反而更慢。因此默认关闭，只保留查询日志作为证据。 */
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
  @State timing: string = '';
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
  /** 到达率相关：每个窗口的帧到达时间戳（含被丢的帧），用来区分
   *  「我们算得慢」与「相机只给这么多」。 */
  private arrivals: number[] = [];
  private winStart: number = 0;
  private arrivedInWin: number = 0;
  private doneInWin: number = 0;
  /** 预热剩余帧数：模型装载后前 N 帧不计入统计（kernel 首编不是稳态）。
   *  实测冷启动首帧 infer=184 ms，稳态 ~20 ms —— 混进去会报出假性能。 */
  private warmLeft: number = 0;
  /** 累计到达帧数（含被丢的）。丢帧率的分母必须是它，不是完成帧数。 */
  private totalArrived: number = 0;
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
  private frameMsWin: number[] = [];
  private decodeMsWin: number[] = [];
  private inferMsWin: number[] = [];

  // ------------------------------------------------------------ 生命周期

  aboutToAppear(): void {
    this.ctx = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.requestCamera();
  }

  aboutToDisappear(): void {
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
      const d: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_DET, this.toArrayBuffer(detBytes), detBe));
      const recBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_REC);
      const r: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_REC, this.toArrayBuffer(recBytes), recBe));
      const clsBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_CLS);
      const k: Map<string, string> = parseKv(
        await lpr.loadModelAsync(MODEL_CLS, this.toArrayBuffer(clsBytes), clsBe));

      if (d.get('ok') !== '1' || r.get('ok') !== '1' || k.get('ok') !== '1') {
        this.status = '模型加载失败 det=' + str(d, 'error') + ' rec=' + str(r, 'error') +
          ' cls=' + str(k, 'error');
        hilog.error(DOMAIN, TAG, 'GEAR %{public}d LOAD FAIL', this.gear);
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
        // 只数 callback：不 readLatestImage，不 release，只统计 arrival 间隔分布。
        // 如果 callback 停摆 → 缓冲满；如果间隔拉长 → 背压。这是「相机上限」的唯一判据。
        this.finishFrame();
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
        // 上一帧还没算完（或正在切档）→ 这一帧本轮不处理。
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
          this.finishFrame();
          this.record(tB - t0, 0);
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
      if (kv.get('ok') === '1') {
        const count: number = num(kv, 'count');
        const p0: string[] = str(kv, 'p0').split(',');
        const stages: string = p0.length > 9 ? p0[9] : '-';
        this.lastInferMs = inferMs;
        this.lastStages = stages;
        // 落点自证：相机档同样把 native 的 backends= 渲染出来（ADR-0003）。
        // 相机是一次一帧、每帧都重取，所以这里的值就是本帧的观测值。
        this.evidence = evidenceRows(str(kv, 'backends'));
        if (count > 0 && p0.length > 0) {
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
        }
      } else {
        this.detail = 'infer 失败: ' + str(kv, 'error');
      }
      // 端到端一帧 = JS 侧取帧/拷贝(prepMs) + native 转换(convMs) + 流水线(inferMs)。
      this.record(prepMs, roundMs);
      this.finishFrame();
    }).catch((e: Error) => {
      hilog.error(DOMAIN, TAG, 'CAMERA FRAME FAIL %{public}s', e.message);
      this.finishFrame();
    });
  }

  private record(decodeMs: number, inferMs: number): void {
    const frameMs: number = decodeMs + inferMs;
    if (!this.firstFrameLogged) {
      this.firstFrameLogged = true;
      hilog.info(DOMAIN, TAG,
        'FIRST FRAME kind=%{public}s decode=%{public}d infer=%{public}d rot=%{public}d',
        this.frameKind, decodeMs, inferMs, this.rotation);
    }
    this.frameMsWin.push(frameMs);
    this.decodeMsWin.push(decodeMs);
    this.inferMsWin.push(inferMs);
    if (this.frameMsWin.length > WINDOW) {
      this.frameMsWin.shift();
      this.decodeMsWin.shift();
      this.inferMsWin.shift();
    }
    this.frames = this.frames + 1;
    if (this.warmLeft > 0) {
      // 预热帧：只消耗预算，不进统计（见 listenFrames 注释）。
      this.warmLeft = this.warmLeft - 1;
      if (this.warmLeft === 0) {
        // 预热结束，窗口从干净状态起算。
        this.frameMsWin = [];
        this.decodeMsWin = [];
        this.inferMsWin = [];
        this.winStart = 0;
        this.arrivedInWin = 0;
        this.doneInWin = 0;
        this.frames = 0;
        this.dropped = 0;
        this.totalArrived = 0;
        this.droppedInWin = 0;
      }
      return;
    }
    this.doneInWin = this.doneInWin + 1;

    // 到达率 vs 完成率：每 ~2 s 结算一个窗口。
    //   到达率 ≈ 完成率  → 相机就是这个速度，我们再快也没用（上限在相机）
    //   到达率 >> 完成率 → 是我们算得慢，还有优化空间
    const now: number = Date.now();
    if (this.winStart === 0) {
      this.winStart = now;
    }
    const winMs: number = now - this.winStart;
    if (winMs >= 2000) {
      this.arrFps = this.arrivedInWin * 1000 / winMs;
      this.doneFps = this.doneInWin * 1000 / winMs;
      hilog.info(DOMAIN, TAG,
        'RATE arrive=%{public}s fps done=%{public}s fps dropped_in_win=%{public}d ' +
        'cum_arrived=%{public}d cum_done=%{public}d cum_dropped=%{public}d',
        this.arrFps.toFixed(2), this.doneFps.toFixed(2), this.droppedInWin,
        this.totalArrived, this.frames, this.dropped);
      this.winStart = now;
      this.arrivedInWin = 0;
      this.doneInWin = 0;
      this.droppedInWin = 0;
    }

    const f: number = p50(this.frameMsWin);
    if (f > 0) {
      this.frameP50 = f;
      this.convP50 = p50(this.decodeMsWin);
      this.inferP50 = p50(this.inferMsWin);
      this.timing = (1000 / f).toFixed(1) + ' fps · frame ' + f.toFixed(1) +
        ' ms (decode ' + p50(this.decodeMsWin).toFixed(1) +
        ' / infer ' + p50(this.inferMsWin).toFixed(1) + ')' +
        ' · 到达 ' + this.arrFps.toFixed(1) + ' / 完成 ' + this.doneFps.toFixed(1) +
        ' · 丢帧 ' + this.dropped.toString() + '/' + this.totalArrived.toString();
      // 指标卡第二行：本帧分段 + 帧预算。
      // 注意：native 只在**检出车牌**时才回 p0（分段在 p0[9]），空帧没有分段数据，
      // 所以这里要区分「本帧未检出」和「有数据」，别把前者显示成 '-' 像故障。
      if (this.gear === GEAR_RAW) {
        this.segText = '本档只取帧不推理 · 相机出帧上限';
      } else if (this.lastStages === '-') {
        this.segText = 'conv ' + p50(this.decodeMsWin).toFixed(1) +
          ' ms · 本帧未检出（分段仅在检出时记录）';
      } else {
        this.segText = 'conv ' + p50(this.decodeMsWin).toFixed(1) +
          ' ms · 分段 det|crop|rec|cls = ' + this.lastStages;
      }
    }
    if (this.frames % 30 === 0) {
      hilog.info(DOMAIN, TAG,
        'FRAME n=%{public}d fps=%{public}s plate=%{public}s split=%{public}s',
        this.frames, (1000 / f).toFixed(2), this.plate, this.split.join('|'));
    }
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
    this.frameMsWin = [];
    this.decodeMsWin = [];
    this.inferMsWin = [];
    this.frames = 0;
    this.dropped = 0;
    this.totalArrived = 0;
    this.droppedInWin = 0;
    this.arrivedInWin = 0;
    this.doneInWin = 0;
    this.winStart = 0;
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
    this.timing = '';
    this.status = '切换档位 → ' + GEAR_NAMES[g];
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
      } else {
        this.status = '档位加载失败';
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
              Text(this.inferP50.toFixed(1))
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
            Text('推理 p50')
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
            .maxLines(2)
            .layoutWeight(1)
            .textAlign(TextAlign.End)
        }
        .width('100%')
        .alignItems(VerticalAlign.Top)
        .padding({ top: 10 })

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
