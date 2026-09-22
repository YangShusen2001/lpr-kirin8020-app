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
import { componentUtils, display, router } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@kit.BasicServicesKit';
// 热态与电池温度：native sysroot 没有热态头文件（已查 native/sysroot/usr/include，
// 无 thermal*），只能在 ArkTS 侧取。与 Index.ets 的 RQ4 探针同一套 API。
import { thermal } from '@kit.BasicServicesKit';
import { batteryInfo } from '@kit.BasicServicesKit';
// 【2026-09-23 实验】视频档（5/6）需要 AVRecorder（提供 VideoOutput 的编码 surface）
// 与 fileIo（给它一个 fd:// 输出文件）。video-only 录制不配音频源，不需要麦克风权限。
import { media } from '@kit.MediaKit';
import { fileIo } from '@kit.CoreFileKit';
import lpr from 'libentry.so';

const TAG = 'LprCamera';
const DOMAIN = 0xD001;

// ---------------------------------------------------------------- 模型与档位

const MODEL_DET: string = 'models/y5fu_320x_head_fp32.ms';
const MODEL_REC: string = 'models/rpv3_mdict_160_r3.ms';
const MODEL_CLS: string = 'models/litemodel_cls_96x_r1.ms';
/**
 * T5：车辆检测模型（YOLOv5u/ultralytics，320，80 类 COCO，DFL 改写后转的 .ms）。
 *
 * **只在走 ROI 路径时加载**：它 36 MB，直检模式下不该白付这份加载与内存。
 * 它是**另一个模型**，与 MODEL_DET（y5fu_320x 车牌检测器）不是同一个 ——
 * 混用会得到「yolov5u 期望单输出，实际 3」。
 */
const MODEL_VEH: string = 'models/yolov5su_320_veh_fp32.ms';

/**
 * 【调试入口】喂图自证的输入：一张预编码的 NV21。
 *
 * 由 `_veh/make_nv21_fixture.py` 从 `veh320.png` 编出（`stride == width`、无旋转），
 * 尺寸与源图一致。喂进去走的是**相机帧同一条入口**（`cameraFrameAsync`），
 * 所以验的是相机页本身 —— 详见 `feedTestImage()`。
 */
const FEED_NV21: string = 'assets/veh320/veh320.nv21';
const FEED_W: number = 320;
const FEED_H: number = 320;
const FEED_STRIDE: number = 320;

/** 预览区容器的组件 id —— 自证脚本靠它反查预览区的**实时**屏幕几何。 */
const PREVIEW_ID: string = 'lprPreview';

/**
 * 相机帧预算（ms）。30 fps → 33.3 ms/帧。
 *
 * ⚠️ 真正该看的指标是 **余量 = 33.3 − frameMs**，不是 `1000/p50(frameMs)` ——
 * 后者是**服务时间的倒数**，不是 fps（CONTEXT.md：服务时间 ≠ 帧率）。
 * 余量为负表示我们已经吃满并超过预算，相机只能靠丢帧跟上。
 */
const FRAME_BUDGET_MS: number = 33.3;

/**
 * 归属配色：车辆框与它的车牌框用**同一个颜色 + 同一个编号**。
 *
 * 用编号而不是连线来表达「车牌属于哪辆车」—— 连线要算角度与端点，
 * 而它的正确性同样只能靠看图验证（本项目看不了图）。编号+同色是可程序化核对的。
 */
const VEH_COLOURS: string[] = ['#FF9500', '#34C759', '#007AFF', '#AF52DE', '#FF3B30', '#5AC8FA'];

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
/** 【2026-09-23 实验】视频档：会话用 `SceneMode.NORMAL_VIDEO`（VideoSession），
 *  预览/分析流 `setFrameRate(60,60)`。动机：`camera-recording.md` L74-78 的官方
 *  双流帧率耦合规则 + `arkts-apis-camera-videosession.md` L17 明说录像流支持
 *  「30fps、60fps」—— 这是文档面上唯一没被真机证伪的 >30fps 路径
 *  （见 research 仓库 docs/notes/camera-60fps-alternatives-research.md）。
 *  本档是**单流版**：只换会话模式 + 请求 60，不加 VideoOutput。
 *  判定只用 `RATE arrive=`（真到达率），不信 `getActiveFrameRate()`。 */
const GEAR_VIDEO: number = 5;
/** 【2026-09-23 实验】视频+录像档：在档 5 之上再加一条 60fps `VideoOutput`
 *  （AVRecorder 提供 surface）。按耦合规则：录像流固定 60-60 → 预览流可跟设
 *  60-60。如果 HAL 的 60fps 传感器档位只在录像配置下激活，只有本档能验证到。
 *  与档 3/5 一样**只取帧不推理** —— 它测的是相机到达率，不是流水线。 */
const GEAR_VIDEO_VOUT: number = 6;
/** 【T9 / 2026-09-23】60fps 供给 + **真实推理**。会话配置与档 6 完全相同
 *  （NORMAL_VIDEO + 预览 60-60 + VideoOutput 60-60），唯一差别是**不在
 *  `isFrameOnlyGear` 族里** —— 它会正常加载模型并走 `infer()`。
 *
 *  为什么需要这一档：相机页**有意丢帧不排队**，故 `done ≡ min(arrive, 产能)`。
 *  在 ~30 fps 供给下只要流水线快于相机，`done` 就恒被**相机**卡住，产能被
 *  完全掩盖（T3 的 `arrive == done` 即此）。供给提到 47~58 后，`done` 第一次
 *  成为**流水线产能的直接读数** —— 这是此前结构上无法测量的量。
 *
 *  判据仍只用 `RATE arrive/done/dropped`，且**必须分桶**（hit/empty 分开）。
 *  预期：生产档（det=CPU）done 明显低于 arrive；全 NPU 档更高。
 *  见 lpr-kirin8020 issue #10。 */
const GEAR_VIDEO_INFER: number = 7;
/** 【T9】档 7 的全 NPU 对照：同样 60fps 供给，但 det/rec/cls 全走 NNRT。
 *  两档只差**检测器的落点**这一个变量 —— 这正是「NPU 的帧率」要回答的问题：
 *  在 30 fps 供给下 det 落点只影响「能不能贴住 30」，而在 47~58 供给下
 *  它直接决定**产能**。⚠️ 引用时必须同时给出 `苏E05172` 这个读错结果
 *  （`det-backend-alters-result.md`：全 NPU 档读错参考图）。 */
const GEAR_VIDEO_INFER_NPU: number = 8;
const GEAR_NAMES: string[] =
  ['生产 det=CPU rec=NPU cls=CPU', '全 NPU', '全 GPU(Vulkan)', '基准：只取帧不推理', '只数 callback',
    '视频档：NORMAL_VIDEO 预览请求60', '视频+录像档：60fps VideoOutput',
    '视频+推理(生产后端)：60fps 供给下测产能', '视频+推理(全 NPU)：60fps 供给下测产能'];
/** 分段控件用的短名 —— 段数一多就会挤爆，字号已随段数收窄。 */
const GEAR_SHORT: string[] =
  ['生产', '全NPU', '全GPU', '基准', 'cb', '视频60', '视频+录', '视频+推理', '视频+全NPU'];

/** 这三档只取帧不推理（相机上限仪器族）。
 *  ⚠️ 档 7/8 **故意不在**此列 —— 它们要跑推理，那正是它们存在的理由。 */
function isFrameOnlyGear(g: number): boolean {
  return g === GEAR_RAW || g === GEAR_ARRIVE_ONLY || g === GEAR_VIDEO || g === GEAR_VIDEO_VOUT;
}

/** 用 `NORMAL_VIDEO` 会话（VideoSession）的档：5/6/7/8。 */
function isVideoGear(g: number): boolean {
  return g === GEAR_VIDEO || g === GEAR_VIDEO_VOUT ||
    g === GEAR_VIDEO_INFER || g === GEAR_VIDEO_INFER_NPU;
}

/** 跑真实推理的「视频+推理」档：7/8。 */
function isVideoInferGear(g: number): boolean {
  return g === GEAR_VIDEO_INFER || g === GEAR_VIDEO_INFER_NPU;
}

/** 需要那条 60fps `VideoOutput`（帧率耦合规则的另一腿）的档：6/7/8。
 *  档 5 是单流对照组 —— 它证明「只换会话模式不加 VideoOutput」并不提速
 *  （实测 25.98 < 基准 30.01）。 */
function wantsVideoOutput(g: number): boolean {
  return g === GEAR_VIDEO_VOUT || isVideoInferGear(g);
}

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

/**
 * 【实验开关】相机路径里 det 的 CPU 后端档。默认 `'cpu'`（不设亲和绑定）。
 *
 * 为什么需要这个开关：跑完 CPU 线程扫描与亲和扫描后发现 —— 同一份
 * `y5fu_320x_head_fp32.ms`、同一后端、同样 4 线程：
 *
 *   | 条件 | p50 |
 *   |---|---|
 *   | 隔离基准，无绑定（紧循环） | **7.67 ms** |
 *   | 隔离基准，`cpu_a1` 大核优先 | 11.36 ms |
 *   | 隔离基准，`cpu_a2` 小核优先 | **19.45 ms** |
 *   | **相机流水线内**（sweep2, 热档 3） | **19.51 ms** |
 *
 * 流水线内 19.51 与「小核优先」19.45 **几乎相同**，而比「无绑定」慢 2.5 倍。
 * 这提示：流水线里的 det 很可能**实际跑在小核上**（或等效的低频状态），
 * 而不是算子本身慢。设为 `'cpu_a1'` 就能**在真实产品条件下**直接验证这一点：
 * 若 tInferMs 从 ~19 ms 掉到 ~11 ms，则「调度/核位」是主因，且**不写一行算子代码**。
 *
 * 注意 `cpu_a1` 在**隔离**基准里反而比无绑定慢（11.36 vs 7.67）——
 * 所以这不是「无脑更快」的档，必须在流水线里实测。测完请改回 `'cpu'`。
 * 详见 docs/notes/camera-npu-headroom.md §3.2c / §5.2。
 */
const DET_CPU_BACKEND: string = 'cpu';

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
  /** T5：车辆检测会话（仅 ROI 路径需要，懒加载）。-1 = 未加载。 */
  private vehId: number = -1;
  private loadedGear: number = -1;

  private camInput: camera.CameraInput | undefined = undefined;
  private previewOut: camera.PreviewOutput | undefined = undefined;
  private analyzeOut: camera.PreviewOutput | undefined = undefined;
  /** 档 5/6 用 VideoSession，其余档用 PhotoSession —— 两者都 extends Session，
   *  公共流程（beginConfig/commitConfig/addOutput/start/stop/release）走基类类型。 */
  private session: camera.Session | undefined = undefined;
  private receiver: image.ImageReceiver | undefined = undefined;
  /** 档 6 的 60fps 录像输出与它的 AVRecorder（提供编码 surface）。 */
  private videoOut: camera.VideoOutput | undefined = undefined;
  private avRecorder: media.AVRecorder | undefined = undefined;
  /** 录像测试文件的 fd（stopCamera 时关闭并删文件）。-1 = 没有。 */
  private videoFd: number = -1;
  /** 当前相机会话是否为 NORMAL_VIDEO（档 5/6 时 openCamera 置位）。
   *  applyGear 靠它判断「切档是否需要整流重启」—— 相机会话只在 openCamera 建一次。 */
  private camVideoMode: boolean = false;
  /** 当前会话是否带 60fps VideoOutput（档 6 且建流成功时置位）。 */
  private camHasVout: boolean = false;

  /** 上一帧还在算时置位 → 新帧直接丢，不排队。@State：状态点要绑它。 */
  @State busy: boolean = false;
  private running: boolean = false;
  /** 档位切换中：帧循环要丢帧，否则新 mode/slot 配旧会话会算出错结果。 */
  @State switching: boolean = false;

  // ─────────────────────────────────────────────────────── T5：两路径 + 叠加层
  /**
   * 走 ROI 路径（true）还是全图直检（false）。
   *
   * 为什么两条路都要留（spec D2）：ROI 路径在 CCPD 上实测 −9.0%，但用户的真实场景
   * （远距离、多车、强干扰）**未被该数据集覆盖**。保留直检既是对照，也是安全网 ——
   * 真机上的线上对比数据靠这个开关长出来。
   */
  @State useRoi: boolean = false;
  /** 本帧车辆框：[x1,y1,x2,y2,cls,score]（源图坐标）。直检模式下为空。 */
  @State vehBoxes: number[][] = [];
  /** 本帧车牌框：[x1,y1,x2,y2,owner]（源图坐标）。owner = -1 表示无归属。 */
  @State plateBoxes: number[][] = [];
  /** 本帧源图尺寸（native 转换后的 w/h），叠加层按它换算。 */
  @State srcW: number = 0;
  @State srcH: number = 0;
  /** 预览区实测尺寸（vp）—— 叠加层的换算基准，由 onAreaChange 写入。 */
  @State viewW: number = 0;
  @State viewH: number = 0;
  /** 本帧走的是哪条路（来自 native 的 useRoi 回执，不是界面自己的意图）。 */
  @State landedRoi: boolean = false;
  /** 本帧车牌颜色（像素测量结论，ADR-0005）。 */
  @State plateColour: string = 'unknown';
  /** 本帧车牌归属的车辆下标（-1 = 无归属）。 */
  @State plateOwner: number = -1;
  /**
   * 帧预算余量（ms）= 33.3 − frameP50。**A4 要求的显示项**。
   * 负数表示已超预算；`-1` 表示样本不足（不是"余量 -1 ms"）。
   */
  @State budgetLeft: number = -1;
  /** ROI 路径的分段读数（veh / roi 两段），直检模式下为空串。 */
  @State roiSegText: string = '';
  /** 预览区在屏幕上的原点（vp）。叠加框坐标相对预览区，布局树坐标是屏幕绝对，
   *  两者比对必须先减掉这个原点。 */
  private viewX: number = 0;
  private viewY: number = 0;
  /** 叠加层的**屏幕坐标**绘制表：[x, y, w, h, 配色下标]。由 rebuildOverlay 预计算。 */
  @State vehDraw: number[][] = [];
  @State plateDraw: number[][] = [];
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
   * 加载车辆检测会话（T5 懒加载）。36 MB，只在走 ROI 路径时付这份代价。
   *
   * 抽成独立方法，是为了让「三个主会话已就绪、只差车辆模型」这条路能**单独补**
   * —— 见 `ensureSessions` 早退条件里的注释（曾因此漏加载，native 报
   * `bad vehicle session id`）。
   */
  private async loadVehSession(c: common.UIAbilityContext): Promise<boolean> {
    const vehBytes: Uint8Array = c.resourceManager.getRawFileContentSync(MODEL_VEH);
    const vehRaw: string = await lpr.loadModelAsync(MODEL_VEH,
      this.toArrayBuffer(vehBytes), 'cpu');
    const vk: Map<string, string> = parseKv(vehRaw);
    if (vk.get('ok') !== '1') {
      this.status = '车辆检测模型加载失败 ' + str(vk, 'error');
      hilog.error(DOMAIN, TAG, 'VEH LOAD FAIL %{public}s', str(vk, 'error'));
      return false;
    }
    this.vehId = num(vk, 'id');
    hilog.info(DOMAIN, TAG, 'VEH LOAD ok id=%{public}d LANDED=%{public}s',
      this.vehId, str(vk, 'backend'));
    return true;
  }

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
    // ⚠️ 早退条件**必须包含车辆模型**。它是懒加载的（只有 ROI 路径要），
    // 但「三个主会话已就绪」不足以说明一切就绪 —— 从直检切到 ROI 时，三个主会话
    // 早在直检阶段就加载好了，早退会直接返回 true ⇒ veh 永不加载 ⇒
    // `useRoi=true` 而 `vehId=-1` ⇒ native 报 `bad vehicle session id`，
    // 而界面只会说"喂图失败"，真因被埋掉。
    const vehReady: boolean = !this.useRoi || this.vehId >= 0;
    if (this.loadedGear === this.gear && this.detId >= 0 && this.recId >= 0 &&
        this.clsId >= 0 && vehReady) {
      return true;
    }
    if (this.loadedGear === this.gear && this.detId >= 0 && this.recId >= 0 &&
        this.clsId >= 0) {
      // 只差车辆模型：单独补它，不重跑三个主会话。
      return this.loadVehSession(c);
    }
    this.status = '加载模型（' + GEAR_NAMES[this.gear] + '）…';
    if (isFrameOnlyGear(this.gear)) {
      // 基准/只数/视频档不推理，无需任何模型。把 id 置 0 让「已加载」判定通过。
      this.detId = 0;
      this.recId = 0;
      this.clsId = 0;
      this.loadedGear = this.gear;
      this.warmLeft = WARMUP_FRAMES;
      if (this.gear === GEAR_ARRIVE_ONLY) {
        this.status = '只数 callback';
      } else if (this.gear === GEAR_VIDEO) {
        this.status = '视频档：NORMAL_VIDEO · 预览请求 60';
      } else if (this.gear === GEAR_VIDEO_VOUT) {
        this.status = '视频+录像档：60fps VideoOutput';
      } else {
        this.status = '基准档：只取帧不推理';
      }
      hilog.info(DOMAIN, TAG, 'GEAR %d %s 就绪（不加载模型）', this.gear, this.status);
      return true;
    }
    try {
      let detBe: string = DET_CPU_BACKEND;
      let recBe: string = 'nnrt';
      let clsBe: string = 'cpu';
      if (this.gear === GEAR_NPU || this.gear === GEAR_VIDEO_INFER_NPU) {
        // 档 8 = 60fps 供给 + 全 NPU。与档 1 同一后端配置，只多一条 60fps 会话。
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

      // T5：车辆检测模型**懒加载** —— 只有走 ROI 路径时才需要它。
      // 36 MB，直检模式下不该白付这份加载与内存。
      if (this.useRoi && this.vehId < 0) {
        if (!await this.loadVehSession(c)) {
          return false;
        }
      }

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

      // 【2026-09-23 实验】档 5/6 用 NORMAL_VIDEO 能力集。前序研究
      // （camera-60fps-research.md §4.2 未验证项 3）指出：profile 缓存是**按
      // SceneMode 分派**的（camera_manager.cpp L2651-2656），NORMAL_VIDEO 下的
      // 预览/录像 profile 集合可能与 NORMAL_PHOTO 不同 —— 本轮顺带把这个问题钉死。
      const gearAtOpen: number = this.gear;
      const videoMode: boolean = isVideoGear(gearAtOpen);
      const cap: camera.CameraOutputCapability = videoMode
        ? mgr.getSupportedOutputCapability(device, camera.SceneMode.NORMAL_VIDEO)
        : mgr.getSupportedOutputCapability(device, camera.SceneMode.NORMAL_PHOTO);
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

      // 【2026-09-23 实验】NORMAL_VIDEO 下的录像 profile 全量 dump（fps 三元组），
      // 并挑「帧率范围含 60 且面积最小」的一档给档 6 的 VideoOutput 用。
      // 这份数据同时回答 camera-60fps-research.md §4.2 未验证项 2：
      // 「60-60 那条能力挂在哪个 (format, size, fps) 三元组上」。
      let vProfileChosen: camera.VideoProfile | undefined = undefined;
      if (videoMode) {
        const vps: camera.VideoProfile[] = cap.videoProfiles;
        let vdump: string = '';
        for (let i = 0; i < vps.length; i++) {
          const v: camera.VideoProfile = vps[i];
          vdump += v.size.width.toString() + 'x' + v.size.height.toString() +
            '/' + v.format.toString() + '@' + v.frameRateRange.min.toString() +
            '-' + v.frameRateRange.max.toString() + ' ';
        }
        for (let i = 0; i < vdump.length; i += chunk) {
          hilog.info(DOMAIN, TAG, 'VIDPROFILES[%{public}d] %{public}s',
            Math.floor(i / chunk), vdump.substring(i, Math.min(i + chunk, vdump.length)));
        }
        for (let i = 0; i < vps.length; i++) {
          const v: camera.VideoProfile = vps[i];
          if (v.frameRateRange.max < 60 || v.frameRateRange.min > 60) {
            continue;
          }
          if (vProfileChosen === undefined ||
            v.size.width * v.size.height < vProfileChosen.size.width * vProfileChosen.size.height) {
            vProfileChosen = v;
          }
        }
        hilog.info(DOMAIN, TAG, 'VPROFILE PICK %{public}s',
          vProfileChosen === undefined ? 'none(60fps 档不存在)'
            : vProfileChosen.size.width.toString() + 'x' + vProfileChosen.size.height.toString() +
              '@' + vProfileChosen.frameRateRange.min.toString() + '-' +
              vProfileChosen.frameRateRange.max.toString());
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

      recv.getReceivingSurfaceId().then(async (rid: string) => {
        const analyzeOut: camera.PreviewOutput = mgr.createPreviewOutput(analyze, rid);
        this.analyzeOut = analyzeOut;
        // 【2026-09-23 实验】档 6：开流前先把 AVRecorder 备好。
        // 时序按官方录像样例：prepare → getInputSurface → createVideoOutput →
        // （beginConfig 窗口内）addOutput；avRecorder.start() 等 session.start() 成功后。
        const wantVout: boolean = videoMode && wantsVideoOutput(gearAtOpen) &&
          vProfileChosen !== undefined;
        if (wantVout) {
          const vp: camera.VideoProfile = vProfileChosen as camera.VideoProfile;
          try {
            const rec: media.AVRecorder = await media.createAVRecorder();
            const vpath: string = c.filesDir + '/video60_' + Date.now().toString() + '.mp4';
            const f: fileIo.File = fileIo.openSync(vpath,
              fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.CREATE);
            // 60fps 视频码率给足 8 Mbps，避免码率不足反过来钳帧率。
            const vcfgProfile: media.AVRecorderProfile = {
              fileFormat: media.ContainerFormatType.CFT_MPEG_4,
              videoBitrate: 8000000,
              videoCodec: media.CodecMimeType.VIDEO_AVC,
              videoFrameWidth: vp.size.width,
              videoFrameHeight: vp.size.height,
              videoFrameRate: 60,
            };
            const vcfg: media.AVRecorderConfig = {
              // video-only：不配 audioSourceType，无需麦克风权限。
              videoSourceType: media.VideoSourceType.VIDEO_SOURCE_TYPE_SURFACE_YUV,
              profile: vcfgProfile,
              url: 'fd://' + f.fd.toString(),
              rotation: 0,
            };
            await rec.prepare(vcfg);
            const vsid: string = await rec.getInputSurface();
            this.videoOut = mgr.createVideoOutput(vp, vsid);
            this.avRecorder = rec;
            this.videoFd = f.fd;
            hilog.info(DOMAIN, TAG, 'VOUT READY %{public}dx%{public}d fps=%{public}d-%{public}d',
              vp.size.width, vp.size.height, vp.frameRateRange.min, vp.frameRateRange.max);
          } catch (e) {
            const err = e as BusinessError;
            // 退化：VideoOutput 建不起来就退回档 5 的形态（纯预览视频会话），
            // 实验照常跑，只是少了耦合规则那条腿。
            hilog.error(DOMAIN, TAG, 'VOUT FAIL code=%{public}d msg=%{public}s —— 退化为纯预览',
              err.code, err.message);
            this.videoOut = undefined;
            this.avRecorder = undefined;
          }
        }
        input.open().then(() => {
          const sess: camera.Session = videoMode
            ? mgr.createSession<camera.VideoSession>(camera.SceneMode.NORMAL_VIDEO)
            : mgr.createSession<camera.PhotoSession>(camera.SceneMode.NORMAL_PHOTO);
          this.session = sess;
          // 记录本流的真实条件（证据文件里每条流一行，事后切分靠它）。
          const fpsTargetAtOpen: number = videoMode ? 60 : FPS_TARGET;
          this.camVideoMode = videoMode;
          this.camHasVout = wantVout && this.videoOut !== undefined;
          this.dumpLine('SESSION MODE gear=' + gearAtOpen.toString() +
            ' mode=' + (videoMode ? 'NORMAL_VIDEO' : 'NORMAL_PHOTO') +
            ' vout=' + (this.camHasVout ? '1' : '0') +
            ' fpsTarget=' + fpsTargetAtOpen.toString());
          sess.beginConfig();
          sess.addInput(input);
          if (!SKIP_PREVIEW_DISPLAY) {
            sess.addOutput(show);
          }
          sess.addOutput(analyzeOut);
          if (this.camHasVout && this.videoOut !== undefined) {
            // 双流帧率耦合规则（camera-recording.md L74-78）的「录像流」这一腿。
            sess.addOutput(this.videoOut);
          }
          hilog.info(DOMAIN, TAG,
            'SESSION CONFIG display=%{public}s mode=%{public}s vout=%{public}d',
            SKIP_PREVIEW_DISPLAY ? 'OFF(实验)' : 'ON',
            videoMode ? 'NORMAL_VIDEO' : 'NORMAL_PHOTO', this.camHasVout ? 1 : 0);
          sess.commitConfig().then(() => {
            sess.start().then(() => {
              if (this.camHasVout && this.avRecorder !== undefined) {
                this.avRecorder.start().then(() => {
                  hilog.info(DOMAIN, TAG, 'VOUT REC START');
                }).catch((e: BusinessError) => {
                  hilog.error(DOMAIN, TAG, 'VOUT REC START FAIL %{public}d %{public}s',
                    e.code, e.message);
                });
              }
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
              this.negotiateFrameRate(analyzeOut, show, videoMode ? 60 : FPS_TARGET);
              this.ready = true;
              this.status = '实时识别中 · ' + analyze.size.width.toString() + 'x' +
                analyze.size.height.toString() + ' · rot=' + this.rotation.toString();
              hilog.info(DOMAIN, TAG,
                'STREAM START show=%{public}dx%{public}d analyze=%{public}dx%{public}d rot=%{public}d ' +
                'mode=%{public}s vout=%{public}d fpsTarget=%{public}d',
                showProfile.size.width, showProfile.size.height,
                analyze.size.width, analyze.size.height, this.rotation,
                videoMode ? 'NORMAL_VIDEO' : 'NORMAL_PHOTO', this.camHasVout ? 1 : 0,
                videoMode ? 60 : FPS_TARGET);
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
                'x' + analyze.size.height.toString() + ' rot=' + this.rotation.toString() +
                ' mode=' + (videoMode ? 'NORMAL_VIDEO' : 'NORMAL_PHOTO') +
                ' vout=' + (this.camHasVout ? '1' : '0') +
                ' fpsTarget=' + (videoMode ? '60' : FPS_TARGET.toString()));
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

  /** 把预览帧率蹬到设备支持的最高档；两条 output 都试，日志留证。
   *  target：目标帧率（PHOTO 档 30，视频档 60）。60-60 被 API 接受 ≠ 传感器兑现，
   *  判定一律以 `RATE arrive=` 为准 —— 这是 camera-fps-ceiling.md 的教训。 */
  private negotiateFrameRate(analyzeOut: camera.PreviewOutput,
    show: camera.PreviewOutput, target: number): void {
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
        // 选「包含 target 的区间」；没有就退到 min 最大的那个区间。
        let lo: number = -1;
        let hi: number = -1;
        for (let j = 0; j < ranges.length; j++) {
          if (ranges[j].min <= target && target <= ranges[j].max) {
            lo = target;
            hi = target;
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
        if (isFrameOnlyGear(this.gear) && this.gear !== GEAR_ARRIVE_ONLY) {
          // 只取帧族（基准/视频5/视频6）：拿到帧就结束。这里测到的是
          // **相机 + 取帧路径**的上限，与推理无关 —— 它给出「再优化推理还有没有意义」的答案。
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
      this.detId, this.recId, this.clsId, mode, recSlot, clsSlot,
      this.vehId, this.useRoi).then((raw: string) => {
      const roundMs: number = Date.now() - t2;
      const frameHit: boolean = this.applyFrameResult(raw);
      // 端到端一帧 = JS 侧取帧/拷贝(prepMs) + native 转换(convMs) + 流水线(inferMs)。
      this.record(prepMs, roundMs, frameHit);
      this.finishFrame();
    }).catch((e: Error) => {
      hilog.error(DOMAIN, TAG, 'CAMERA FRAME FAIL %{public}s', e.message);
      this.finishFrame();
    });
  }

  /**
   * 把一条 `cameraFrameAsync` 的结果串**落成界面状态**（叠加层 / 读数卡 / 分段读数）。
   *
   * 为什么单独抽出来：相机页有**两个**数据源 —— 真实相机帧，以及「喂图自证」的
   * 预编码 NV21。两者格式与入口完全相同（都过 `cameraFrameAsync`），只有「帧从哪来」
   * 不同。抽成一个方法，喂图验的就是**相机页本身**，而不是一条旁路副本。
   *
   * ⚠️ 本方法**不碰** `record()` / `finishFrame()`：帧统计（fps / p50 / 丢帧）只能由
   * 真实相机帧驱动 —— 否则喂一次图就把统计口径污染了。调用方自己决定要不要记帧。
   *
   * 返回本帧是否检出（`count > 0`）：检出帧要多跑矫正 + 识别 + CTC + 判色，
   * 与未检出帧是两种分布，调用方要用这个值分桶（见 `record` 的注释）。
   */
  private applyFrameResult(raw: string): boolean {
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

      // ── T5：叠加层数据（源图坐标，换算交给 overlayRect）
      this.srcW = num(kv, 'w');
      this.srcH = num(kv, 'h');
      // 用 native 的**回执**而不是界面的意图：开关点了但 native 因缺 vehId 报错时，
      // landedRoi 会是 0，界面就能如实显示"实际走的还是直检"。
      this.landedRoi = str(kv, 'useRoi') === '1';
      const nVeh: number = num(kv, 'vehCount');
      const vb: number[][] = [];
      for (let i = 0; i < nVeh; i++) {
        const f: string[] = str(kv, 'v' + i.toString()).split(',');
        // 只依赖前 3 段（classId,score,rect）—— 第 4 段是类别名，叠加层不画它。
        //
        // ⚠️ 这里曾写成 `< 4`：当时 camera 路径只给 3 段，于是 4 个车框被**静默丢光**
        // —— native 报 `vehCount=4`，界面 `vehDraw=0`，一个框都画不出来且不报错。
        // 判定要按「这段代码实际用几段」写，不能按「另一条路径给了几段」写。
        if (f.length < 3) {
          continue;
        }
        const rc: string[] = f[2].split('|');
        if (rc.length !== 4) {
          continue;
        }
        vb.push([parseFloat(rc[0]), parseFloat(rc[1]), parseFloat(rc[2]), parseFloat(rc[3]),
          parseFloat(f[0]), parseFloat(f[1])]);
      }
      this.vehBoxes = vb;
      const pb: number[][] = [];
      for (let i = 0; i < count; i++) {
        const f: string[] = str(kv, 'p' + i.toString()).split(',');
        // p<i> 的末位（下标 11）是 ownerVeh —— T5 追加，既有下标含义不变。
        if (f.length < 12) {
          continue;
        }
        const rc: string[] = f[4].split('|');
        if (rc.length !== 4) {
          continue;
        }
        pb.push([parseFloat(rc[0]), parseFloat(rc[1]), parseFloat(rc[2]), parseFloat(rc[3]),
          parseFloat(f[11])]);
      }
      this.plateBoxes = pb;
      if (count > 0) {
        const f0: string[] = str(kv, 'p0').split(',');
        this.plateOwner = f0.length > 11 ? parseFloat(f0[11]) : -1;
        const col: string[] = (f0.length > 6 ? f0[6] : '').split('|');
        this.plateColour = col.length > 0 && col[0].length > 0 ? col[0] : 'unknown';
      } else {
        this.plateOwner = -1;
        this.plateColour = 'unknown';
      }
      // ROI 路的分段读数（直检模式下清空，免得挂着上一路的数）
      if (this.landedRoi) {
        this.roiSegText = 'veh ' + str(kv, 'vehInferMs') + ' + roi ' + str(kv, 'roiDetectMs') +
          ' ms · 车框 ' + str(kv, 'vehCount') + ' · 去重丢 ' + str(kv, 'dedupeDropped');
      } else {
        this.roiSegText = '';
      }
      this.rebuildOverlay();
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
    return frameHit;
  }

  /**
   * 喂图自证的一行证据。**两条通道都写**：
   *   ① `hilog` —— hdc 能抓（真机私有目录里的运行日志拉不出来，见项目约定）；
   *   ② `dumpLine` —— 进本轮运行日志，界面上可直接复制。
   * 只写 ② 会让自证无法从 PC 侧取证 —— 这一条是踩过才知道的。
   */
  private feedLog(line: string): void {
    hilog.info(DOMAIN, TAG, 'FEED %{public}s', line);
    this.dumpLine(line);
  }

  /**
   * 【调试入口】喂一张**预编码的 NV21** 测试图，走与相机帧完全相同的入口。
   *
   * 为什么需要：T5 验的是呈现层（叠加框 / 归属 / 读数卡 / 余量），而真机上让相机
   * **碰巧拍到车牌**不可复现 —— 朝向、光照、焦距都不可控。这里把
   * `assets/veh320/veh320.nv21` 当作一帧相机数据喂进去：**同格式、同入口
   * （`cameraFrameAsync`）、同后处理（`applyFrameResult`）**，所以验的是相机页
   * 本身，而不是一条旁路副本。
   *
   * ⚠️ 喂图前必须**停相机**：否则相机的帧回调会在 ~33 ms 内把刚写进去的状态覆盖掉，
   * 验证窗口根本不存在。停流后**不自动恢复** —— 切档或重进页面即可恢复。
   *
   * ⚠️ **不记帧**：这一帧不是相机给的，进 fps / p50 / 丢帧统计会污染口径。
   *
   * 触发：长按大标题「相机实时识别」。
   */
  private async feedTestImage(): Promise<void> {
    // 记下喂图前相机是否在跑：喂图必须停相机（否则帧回调会在 ~33 ms 内覆盖掉
    // 刚写进去的状态），但**停完不恢复会让用户以为 App 卡死了**（预览区静止、
    // 计时不动）。所以停之前先记住，喂完延迟恢复。
    const wasRunning: boolean = this.running;
    if (wasRunning) {
      this.stopCamera();
    }
    try {
      // 喂图要跑推理，而「基准 / 只数 / 视频」这几档不加载模型 —— 先如实说清楚，
      // 而不是让它去跑一条 detId=0 的假流水线。
      if (isFrameOnlyGear(this.gear)) {
        this.feedLog('SKIP 当前档不推理（' + GEAR_NAMES[this.gear] + '），先切到推理档');
        this.status = '喂图自证：当前档不推理';
        return;
      }
      const ok: boolean = await this.ensureSessions();
      if (!ok) {
        this.feedLog('FAIL 会话未就绪');
        return;
      }
      const c: common.UIAbilityContext | undefined = this.ctx;
      if (c === undefined) {
        this.feedLog('FAIL 无 UIAbilityContext');
        return;
      }
      const bytes: Uint8Array = c.resourceManager.getRawFileContentSync(FEED_NV21);
      const expect: number = FEED_STRIDE * FEED_H * 3 / 2;
      if (bytes.length !== expect) {
        // 长度不符就没必要喂 —— 让 native 去撞越界检查只会得到一条含糊的报错。
        this.feedLog('FAIL nv21 长度 ' + bytes.length.toString() + ' != ' + expect.toString());
        return;
      }
      const raw: string = await lpr.cameraFrameAsync(this.toArrayBuffer(bytes),
        FEED_W, FEED_H, FEED_STRIDE, 0,
        this.detId, this.recId, this.clsId, 0, -1, -1, this.vehId, this.useRoi);
      const hit: boolean = this.applyFrameResult(raw);
      const kv: Map<string, string> = parseKv(raw);
      // 一行把判据需要的都摊开：源图尺寸（叠加层换算基准）、框数、路径回执、耗时。
      this.feedLog('ok=' + str(kv, 'ok') + ' count=' + str(kv, 'count') +
        ' hit=' + (hit ? '1' : '0') +
        ' veh=' + str(kv, 'vehCount') + ' useRoi=' + str(kv, 'useRoi') +
        ' src=' + str(kv, 'w') + 'x' + str(kv, 'h') +
        ' convMs=' + str(kv, 'convMs') + ' inferMs=' + str(kv, 'inferMs') +
        // 失败时真因在 error 里。**必须打出来** —— 曾因为没打，只能靠
        // 布局树里那行 `infer 失败: xxx` 反推，绕了一大圈。
        ' error=' + str(kv, 'error'));
      // ── 呈现层的可核对证据 ──────────────────────────────────────
      // 叠加层坐标是**预计算**出来的（overlayRect 是纯函数）。这里把输入与输出
      // 一起摊开，验证脚本就能：① 独立重算 src/view → scale → draw，逐框比对；
      // ② 拿 draw 去和布局树 dump 出的**实际**矩形比对。
      // 两处都对，才算「框画对了」，而不是「我觉得画对了」。
      const sc: number = (this.srcW > 0
        ? Math.min(this.viewW / this.srcW, this.viewH / this.srcH) : 0);
      this.feedLog('GEOM src=' + this.srcW.toString() + 'x' + this.srcH.toString() +
        ' view=' + this.viewW.toFixed(2) + 'x' + this.viewH.toFixed(2) +
        ' scale=' + sc.toFixed(6) +
        ' vehDraw=' + this.vehDraw.length.toString() +
        ' plateDraw=' + this.plateDraw.length.toString() +
        ' density=' + (vp2px(1)).toFixed(4) +
        ' viewPos=' + px2vp(this.viewX).toFixed(2) + ',' + px2vp(this.viewY).toFixed(2) +
        ' viewPosPx=' + this.viewX.toFixed(1) + ',' + this.viewY.toFixed(1));
      for (let i = 0; i < this.vehDraw.length; i++) {
        const b: number[] = this.vehDraw[i];
        const sb: number[] = this.vehBoxes[i];
        // src 与 draw 必须同行给出：验证脚本要拿 src 独立重算 draw，
        // 再拿 draw 与布局树的实际矩形比对 —— 少任何一个都只剩"自说自话"。
        this.feedLog('BOX veh i=' + i.toString() +
          ' src=' + sb[0].toFixed(2) + ',' + sb[1].toFixed(2) + ',' +
          sb[2].toFixed(2) + ',' + sb[3].toFixed(2) +
          ' draw=' + b[0].toFixed(2) + ',' + b[1].toFixed(2) + ',' +
          (b[0] + b[2]).toFixed(2) + ',' + (b[1] + b[3]).toFixed(2) +
          ' drawPx=' + vp2px(b[0]).toFixed(1) + ',' + vp2px(b[1]).toFixed(1) + ',' +
          vp2px(b[0] + b[2]).toFixed(1) + ',' + vp2px(b[1] + b[3]).toFixed(1));
      }
      for (let i = 0; i < this.plateDraw.length; i++) {
        const b: number[] = this.plateDraw[i];
        const sb: number[] = this.plateBoxes[i];
        this.feedLog('BOX plate i=' + i.toString() + ' owner=' + b[4].toFixed(0) +
          ' src=' + sb[0].toFixed(2) + ',' + sb[1].toFixed(2) + ',' +
          sb[2].toFixed(2) + ',' + sb[3].toFixed(2) +
          ' draw=' + b[0].toFixed(2) + ',' + b[1].toFixed(2) + ',' +
          (b[0] + b[2]).toFixed(2) + ',' + (b[1] + b[3]).toFixed(2) +
          ' drawPx=' + vp2px(b[0]).toFixed(1) + ',' + vp2px(b[1]).toFixed(1) + ',' +
          vp2px(b[0] + b[2]).toFixed(1) + ',' + vp2px(b[1] + b[3]).toFixed(1));
      }
      this.feedLog('RESULT plate=' + this.plate + ' colour=' + this.plateColour +
        ' owner=' + this.plateOwner.toString() + ' hit=' + (hit ? '1' : '0') +
        ' seg=' + (this.roiSegText.length > 0 ? this.roiSegText : '-'));
      this.status = (hit ? '喂图自证：检出' : '喂图自证：未检出') +
        (wasRunning ? '（10 s 后自动恢复相机）' : '');
      if (wasRunning) {
        // 延迟恢复：留出取证窗口（PC 侧要 dump 布局树 + 截图，实测几秒内跑完），
        // 之后自动回到活体状态。10 s 是实测折中。
        setTimeout(() => {
          this.tryOpen();
        }, 10000);
      }
    } catch (e) {
      const err = e as Error;
      this.feedLog('EXCEPTION ' + err.message);
    }
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
      // A4：余量 = 33.3 − frameMs。**这才是该看的指标**（不是 1000/p50）。
      this.budgetLeft = FRAME_BUDGET_MS - f;
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
      if (isFrameOnlyGear(this.gear) && this.gear !== GEAR_ARRIVE_ONLY) {
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
    // 【2026-09-23 实验】档 6 的录像资源：先停 recorder（它会持续吃 VideoOutput 的
    // buffer），再关 fd。mp4 **保留**在 filesDir（可 hdc file recv 取回作证据），
    // 文件名 video60_*.mp4，需要清理时按时间戳删。
    try {
      if (this.avRecorder !== undefined) {
        this.avRecorder.stop().catch(() => {
        });
        this.avRecorder.release().catch(() => {
        });
        this.avRecorder = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.videoOut !== undefined) {
        this.videoOut.release();
        this.videoOut = undefined;
      }
    } catch (e) {
    }
    try {
      if (this.videoFd >= 0) {
        fileIo.closeSync(this.videoFd);
        this.videoFd = -1;
      }
    } catch (e) {
    }
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
    // 【2026-09-23 实验】档 5/6/7/8 需要 NORMAL_VIDEO 会话（6/7/8 另加 VideoOutput），
    // 而相机会话只在 openCamera 建一次 —— 目标形态与当前不一致时必须**整流重启**。
    // 这也是为什么视频实验不能做成普通档位：普通档只换模型会话，不动相机流。
    //
    // 用 `isVideoGear` / `wantsVideoOutput` 而不是枚举档号：档 7/8 与档 6 的
    // **会话形态完全相同**，差别只在是否推理，所以「需不需要重启」只看形态。
    // 枚举写法每加一档就要改一处，正是 t8 那次静默回滚的温床。
    const wantVideo: boolean = isVideoGear(this.gear);
    const wantVout: boolean = wantsVideoOutput(this.gear);
    const needRestart: boolean = wantVideo !== this.camVideoMode ||
      (wantVideo && wantVout !== this.camHasVout);
    if (needRestart) {
      this.stopCamera();
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
      // 【2026-09-23 实验】跨形态切档：整流重启。openCamera 按当前 gear 重建会话
      // （mode/vout 在里面置位），STREAM START 后 listenFrames 重新挂回调，
      // 帧循环自然恢复，不走 resumeFrames（旧 receiver 已 release）。
      if (ok && needRestart) {
        this.tryOpen();
        return;
      }
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

  // ────────────────────────────────────────────── T5：两路径切换 + 叠加层

  /**
   * 切「ROI 路径 / 全图直检」。
   *
   * 与 `switchGear` 的区别：**不重建会话、不重开流** —— 两条路复用同一份 NV21→RGBA
   * 与同一批车牌会话，差别只在后段（ROI 多一次车辆检测 + N 次逐框车牌检测）。
   * 所以这里只改一个布尔 + 重置统计。
   *
   * 统计窗口**必须重置**：两条路的 frameMs 是两种分布（ROI 多一段车辆检测 + N 次
   * 车牌检测），混在一个窗口里取 p50 就是混合分布 —— 切换前后的读数会没有意义。
   */
  private switchPath(useRoi: boolean): void {
    if (this.useRoi === useRoi || this.switching) {
      return;
    }
    this.useRoi = useRoi;
    this.resetWindows();
    // 上一路的读数清掉：否则切换后还挂着旧路的 frameMs/p50，看着像新路已经出数。
    this.frameP50 = 0;
    this.convP50 = 0;
    this.inferP50 = 0;
    this.budgetLeft = -1;
    this.roiSegText = '';
    this.lastInferMs = 0;
    this.lastStages = '-';
    this.segText = '';
    this.detail = '';
    this.plate = '（未检测到）';
    this.hit = false;
    this.plateOwner = -1;
    this.plateColour = 'unknown';
    this.vehBoxes = [];
    this.plateBoxes = [];
    this.vehDraw = [];
    this.plateDraw = [];
    // 新路的爬坡不能被算成稳态 —— 与切档同一套纪律。
    this.settled = false;
    this.streamStartMs = Date.now();
    this.arrFps = 0;
    this.doneFps = 0;
    this.status = '路径 → ' + (useRoi ? 'ROI（车辆检测 → 逐框车牌）' : '全图直检');
    this.dumpLine('PATH SWITCH -> ' + (useRoi ? 'roi' : 'direct') + ' settle restarts');
    if (useRoi && this.vehId < 0) {
      // 车辆模型懒加载：第一次切到 ROI 才付这份 36 MB 的加载与内存。
      this.ensureSessions().then((ok: boolean) => {
        if (!ok) {
          this.status = 'ROI 路径不可用：车辆模型未就绪（已回落到直检）';
          this.dumpLine('PATH SWITCH roi FAILED: veh model not ready');
        }
      });
    }
  }

  /**
   * 源图坐标 → 预览区坐标（vp），按 **contain** 语义等比缩放并居中。
   *
   * ⚠️ 这是叠加层**唯一**的几何假设。若 Surface 实际按 cover/fill 显示，
   * 叠加框会系统性偏移 —— 而那只可能靠**看图**发现，本项目看不了图。
   * 所以：① 这里写成纯函数，便于单独核对；② 每次重建都把
   * `src/view` 与框数打进 hilog，事后可拿 `hdc shell snapshot_display` 的截图
   * 逐像素核对。
   */
  private overlayRect(x1: number, y1: number, x2: number, y2: number): number[] {
    if (this.srcW <= 0 || this.srcH <= 0 || this.viewW <= 0 || this.viewH <= 0) {
      return [0, 0, 0, 0];
    }
    const sc: number = Math.min(this.viewW / this.srcW, this.viewH / this.srcH);
    const offX: number = (this.viewW - this.srcW * sc) / 2;
    const offY: number = (this.viewH - this.srcH * sc) / 2;
    return [offX + x1 * sc, offY + y1 * sc, offX + x2 * sc, offY + y2 * sc];
  }

  /**
   * 把源图坐标的框换算成屏幕绘制表。
   *
   * 为什么预计算而不是在 @Builder 里现算：@Builder 体内做 4 次函数调用 × N 个框，
   * 且 ArkTS 的 @Builder 不保证在 @State 变化时按预期重算（本项目已踩过一次
   * 「@Builder 传值参数只在首帧求值」）。预计算成 @State 数组最稳。
   */
  /**
   * 取预览区的**实时**几何：vp 尺寸（供 overlayRect）+ 屏幕 px 原点（供自证）。
   *
   * ⚠️ 为什么不能只用 `onAreaChange` 缓存的值：它给的是**变化那一刻的快照**。
   * 预览区高度由 `layoutWeight(1)` 分配、会随读数卡行数变化，实测抓到过
   * viewH 停在 315.53 vp 而实际已是 341.31 vp（差 25.8 vp ≈ 74 px）——
   * scale 因此算小，4 个框**全部系统性偏移**，且表现为「整体缩放」而非
   * 「整体平移」（因为 scale 是按错的 viewH 算的），极易被误读成公式写错。
   *
   * `onAreaChange` 仍然保留：它负责在尺寸变化时触发一次重算；本方法负责
   * 在**每次**重算时把值取准。两者分工不同，缺一不可。
   */
  private refreshViewGeom(): void {
    try {
      const info: componentUtils.ComponentInfo | undefined =
        this.getUIContext().getComponentUtils().getRectangleById(PREVIEW_ID);
      if (info === undefined) {
        return;
      }
      this.viewW = px2vp(info.size.width);
      this.viewH = px2vp(info.size.height);
      // screenOffset 是**屏幕 px**，与布局树 dump 的 bounds 同一坐标系 ——
      // 自证脚本据此把「相对预览区」的框坐标换算成屏幕绝对坐标直接比对，
      // 全程不需要任何 density 猜测。
      this.viewX = info.screenOffset.x;
      this.viewY = info.screenOffset.y;
    } catch (e) {
      // 拿不到就沿用旧值；GEOM 行的 view 会与布局树对不上，自证脚本能看出来。
      hilog.warn(DOMAIN, TAG, 'refreshViewGeom failed: %{public}s', (e as Error).message);
    }
  }

  private rebuildOverlay(): void {
    this.refreshViewGeom();
    const vd: number[][] = [];
    for (let i = 0; i < this.vehBoxes.length; i++) {
      const b: number[] = this.vehBoxes[i];
      const r: number[] = this.overlayRect(b[0], b[1], b[2], b[3]);
      vd.push([r[0], r[1], r[2] - r[0], r[3] - r[1], i]);
    }
    this.vehDraw = vd;
    const pd: number[][] = [];
    for (let i = 0; i < this.plateBoxes.length; i++) {
      const b: number[] = this.plateBoxes[i];
      const r: number[] = this.overlayRect(b[0], b[1], b[2], b[3]);
      pd.push([r[0], r[1], r[2] - r[0], r[3] - r[1], b[4]]);
    }
    this.plateDraw = pd;
    if ((this.vehDraw.length > 0 || this.plateDraw.length > 0) && this.frames % 30 === 0) {
      hilog.info(DOMAIN, TAG,
        'OVERLAY src=%{public}dx%{public}d view=%{public}sx%{public}s veh=%{public}d ' +
        'plate=%{public}d scale=%{public}s',
        this.srcW, this.srcH, this.viewW.toFixed(1), this.viewH.toFixed(1),
        this.vehDraw.length, this.plateDraw.length,
        (this.srcW > 0 ? Math.min(this.viewW / this.srcW, this.viewH / this.srcH) : 0).toFixed(4));
    }
  }

  /**
   * 叠加层：车辆框 + 车牌框 + 归属编号。
   *
   * 车辆框左上角一个编号徽标（`#0`/`#1`…），车牌框用**同一编号同一颜色**描边 ——
   * 这就是「标明车牌归属哪辆车」。无归属的车牌（直检路径，owner=-1）用红色描边。
   */
  @Builder
  overlayLayer() {
    Stack() {
      ForEach(this.vehDraw, (b: number[]) => {
        Column() {
          Text('#' + b[4].toString())
            .fontSize(10)
            .fontColor('#FFFFFF')
            .backgroundColor(VEH_COLOURS[b[4] % VEH_COLOURS.length])
            .padding({ left: 3, right: 3 })
          Blank()
        }
        .width(b[2])
        .height(b[3])
        .alignItems(HorizontalAlign.Start)
        .border({ width: 1.5, color: VEH_COLOURS[b[4] % VEH_COLOURS.length] })
        .position({ x: b[0], y: b[1] })
      }, (b: number[]) => 'v' + b[4].toString() + '-' + b[0].toFixed(1))
      ForEach(this.plateDraw, (b: number[]) => {
        Row()
          .width(b[2])
          .height(b[3])
          .border({
            width: 2,
            color: b[4] >= 0 ? VEH_COLOURS[b[4] % VEH_COLOURS.length] : '#FF3B30'
          })
          .position({ x: b[0], y: b[1] })
      }, (b: number[]) => 'p' + b[4].toString() + '-' + b[0].toFixed(1))
    }
    .width('100%')
    .height('100%')
  }

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
          // 【调试入口】长按标题 → 喂预编码 NV21 做呈现层自证（见 feedTestImage）。
          // 不做成按钮：仪器面板上的常驻控件应当都是线上要看的东西，调试入口不是。
          .gesture(LongPressGesture().onAction(() => {
            this.feedTestImage();
          }))
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
      // T5：外面套一层 Stack，把叠加层（车辆框 / 车牌框）盖在预览上。
      Stack() {
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
          .height('100%')
          .backgroundColor('#000000')
        this.overlayLayer()
      }
      .width('100%')
      .layoutWeight(1)
      .id(PREVIEW_ID)
      // 叠加层的换算基准只能来自**实测**尺寸：预览区高度由 layoutWeight 分配，
      // 拿屏幕宽高去猜必然错。
      // 尺寸变化时触发一次重算。**具体的值由 refreshViewGeom 现取**，
      // 这里不缓存 —— 缓存会过期（见 refreshViewGeom 的注释）。
      .onAreaChange(() => {
        this.rebuildOverlay();
      })

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
        // T5：颜色 + 归属（票面：结果区显示车牌串 + 颜色，并标明车牌归属哪辆车）
        Text(this.hit
          ? ('颜色 ' + this.plateColour + ' · 归属 ' +
            (this.plateOwner >= 0 ? '车辆#' + this.plateOwner.toString() : '无归属（直检）'))
          : '—')
          .fontSize(11)
          .fontColor(IOS_LABEL2)
        // A4：帧预算余量。**不得把 1000/p50 叫 fps** —— 余量才是该看的指标。
        Row({ space: 6 }) {
          Text('余量 ' + (this.budgetLeft < 0 ? '—' : this.budgetLeft.toFixed(1) + ' ms'))
            .fontSize(12)
            .fontColor(this.budgetLeft < 0 ? IOS_LABEL3 :
              (this.budgetLeft >= 0 ? IOS_GREEN : IOS_ORANGE))
          Text('= 33.3 − frameMs p50（服务时间，非 fps）')
            .fontSize(10)
            .fontColor(IOS_LABEL3)
        }
        // T5：两路径切换。半透明一律写 rgba()（8 位 hex 在 ArkUI 是 ARGB，不是 CSS RGBA）
        Row({ space: 6 }) {
          Text('全图直检')
            .fontSize(12)
            .fontColor(this.useRoi ? IOS_LABEL2 : '#FFFFFF')
            .backgroundColor(this.useRoi ? 'rgba(0,0,0,0)' : IOS_BLUE)
            .padding({ left: 10, right: 10, top: 4, bottom: 4 })
            .borderRadius(7)
            .onClick(() => {
              this.switchPath(false);
            })
          Text('ROI 路径')
            .fontSize(12)
            .fontColor(this.useRoi ? '#FFFFFF' : IOS_LABEL2)
            .backgroundColor(this.useRoi ? IOS_BLUE : 'rgba(0,0,0,0)')
            .padding({ left: 10, right: 10, top: 4, bottom: 4 })
            .borderRadius(7)
            .onClick(() => {
              this.switchPath(true);
            })
          // 显示 native 的**回执**而不是界面的意图：开关点了但 native 报错时，
          // 这里会如实显示"实际：直检"，开关不会变成假的。
          Text(this.landedRoi ? '实际：ROI' : '实际：直检')
            .fontSize(10)
            .fontColor(IOS_LABEL3)
        }
        Text(this.roiSegText.length > 0 ? this.roiSegText : ' ')
          .fontSize(10)
          .fontColor(IOS_LABEL3)
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
            Text(isFrameOnlyGear(this.gear) && this.gear !== GEAR_ARRIVE_ONLY
              ? '取帧 p50' : '帧 p50')
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

      // ── 分段控件：逐档后端
      //
      // ⚠️ 档位一多，等宽分段会挤到点不准。**不要在这里写死点击坐标** ——
      // 坐标由 uitest dumpLayout 现场读（tools/run_camera_sweep.ps1 的
      // -Gears 从 bounds 反算中心）。字号随段数收窄，保证 9 段仍能显示全。
      Row() {
        ForEach(GEAR_SHORT, (name: string, idx: number) => {
          Text(name)
            .fontSize(GEAR_SHORT.length > 6 ? 10 : 13)
            .fontWeight(this.gear === idx ? FontWeight.Bold : FontWeight.Normal)
            .fontColor(this.gear === idx ? IOS_LABEL : IOS_LABEL2)
            .textAlign(TextAlign.Center)
            .maxLines(1)
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
