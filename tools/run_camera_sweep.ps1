<#
.SYNOPSIS
    按固定协议跑一轮相机复采：逐档点选 -> 等稳态门 -> 观测 -> 落盘。

.DESCRIPTION
    **为什么档位是外部点击而不是 App 内的自动扫档**：App 只加仪表（分桶 / 稳态门 /
    热态 / 落盘），协议由外部驱动。这样「跑协议」这件事不会和「相机页状态机」
    耦合 —— 历史上切档停摆（ade5555）就是状态机自身的问题，再往里面塞一个
    自动扫档例程，会把两类缺陷混在一起，出了事分不清是谁的。

    每档的时序：
        click -> (切档加载 ~7 s) -> settle 45 s -> 观测 observeSec -> 下一档

    稳态门由 App 侧强制（`SETTLE_MS`），本脚本只是**等够时间**；即使脚本等早了，
    App 也不会把爬坡期的窗口记进统计。这是有意的双保险：时间由脚本控制，
    但「什么算稳态」的判定权在仪器里，不在驱动脚本里。

.PARAMETER Tag
    本轮标签，用于归档（如 `sweep1`）。

.PARAMETER ObserveSec
    每档稳态观测时长（默认 75 s，约 37 个窗口）。

.PARAMETER Gears
    要跑的档位（默认 0,1,3,4：生产 / 全 NPU / 基准 / callback）。
    2（全 GPU）默认不跑 —— Vulkan 三模型都比 CPU 慢，已有结论。

.EXAMPLE
    pwsh tools/run_camera_sweep.ps1 -Tag sweep1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [int]$ObserveSec = 75,
    # ⚠️ 必须是**字符串**而不是 [int[]]：用 `powershell -File script.ps1 -Gears 0,1,3,4` 调用时，
    # -File 不做 PowerShell 的参数解析，`0,1,3,4` 会被当成**一个**字符串，
    # 而 [int[]] 转换会把逗号剥掉变成数字 **134** —— 于是 $GearXY[134] 为 null，
    # 报「Cannot index into a null array」，且错得完全看不出原因（实测踩到）。
    # 收字符串再自己 split，两种调用方式（-File 与 -Command）都对。
    [string]$Gears = '0,1,3,4'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Hdc = 'D:\IDE\DevEco_Studio\sdk\default\openharmony\toolchains\hdc.exe'
$Bundle = 'com.shusen.lprdemo'

# 分段控件的点击坐标（1224x2688 屏幕，来自 uitest dumpLayout 的实测 bounds）
$GearXY = @{
    0 = @(126, 2541)    # 生产    bounds=[5,2499][248,2583]
    1 = @(369, 2541)    # 全 NPU  bounds=[248,2499][491,2583]
    2 = @(612, 2541)    # 全 GPU  bounds=[491,2499][734,2583]
    3 = @(854, 2541)    # 基准    bounds=[733,2499][976,2583]
    4 = @(1097, 2541)   # callback bounds=[976,2499][1219,2583]
}
$GearName = @{ 0 = '生产'; 1 = '全 NPU'; 2 = '全 GPU'; 3 = '基准'; 4 = 'callback' }

function Invoke-Hdc { param([string]$Cmd) & $Hdc shell $Cmd 2>&1 | Out-String }

function Get-CurrentGear {
    # 从 App 落盘文件最后一行 RATE / GEAR SWITCH 读当前档位。
    # 不猜：直接问仪器自己报了什么。
    $out = Invoke-Hdc "tail -3 /data/app/el2/100/base/$Bundle/haps/entry/files/camera_run_*.log"
    if ($out -match 'GEAR SWITCH -> (\d)') { return [int]$Matches[1] }
    if ($out -match 'gear=(\d)') { return [int]$Matches[1] }
    return -1
}

# ── 0. 前置检查 ──────────────────────────────────────────────────────────────
$targets = (& $Hdc list targets 2>&1) | Where-Object { $_ -and $_ -notmatch 'Empty' }
if (-not $targets) { throw '没有连接的设备' }
Write-Host "[device] $($targets -join ', ')"

$chk = Join-Path $PSScriptRoot 'check_native_build_flags.py'
$out = & python $chk 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host $out
    throw 'native 优化标志检查未通过 —— 本轮性能数字不可用（-O0 慢 4 倍且无报错）'
}
Write-Host '[build] native 优化标志 OK'

# ── 1. 启动并进入相机页 ──────────────────────────────────────────────────────
Write-Host '[step] 启动 App'
Invoke-Hdc "aa force-stop $Bundle" | Out-Null
Start-Sleep -Seconds 2
Invoke-Hdc "aa start -a EntryAbility -b $Bundle" | Out-Null
Start-Sleep -Seconds 8

# 「相机实时识别」按钮 bounds=[85,533][1139,659]
Invoke-Hdc 'uitest uiInput click 612 596' | Out-Null
Start-Sleep -Seconds 6

# 权限弹窗：重装 App 会重置授权，首次进相机页必弹「允许…访问你的相机？」。
# 不处理的话 dumpLayout 只看到弹窗、STREAM START 永远不出现 ——
# 而症状（没有 STREAM START）与「相机起不来」一模一样，容易误诊。
# 按文案找「允许」按钮并点击；找不到就当作已经授权过。
function Approve-CameraPermission {
    for ($i = 0; $i -lt 3; $i++) {
        Invoke-Hdc 'uitest dumpLayout -p /data/local/tmp/perm.json' | Out-Null
        $tmp = Join-Path $Root '_scratch\perm.json'
        & $Hdc file recv /data/local/tmp/perm.json $tmp 2>&1 | Out-Null
        if (-not (Test-Path $tmp)) { return $false }
        $json = Get-Content $tmp -Raw -Encoding UTF8
        if ($json -notmatch '访问你的相机') { return $false }
        # 从 layout 里取含「允许」节点的 bounds 中心。
        #
        # ⚠️ 正则不能贪心跨窗口匹配：实测 dumpLayout 里「允许」这个 Text 节点
        # 前面还有一个标题节点，文案是「允许"车牌识别"访问你的相机？」。
        # 用 `"允许` 前缀 + `[\s\S]{0,400}?bounds` 会先匹配到**标题**，
        # 算出 (612,1315) —— 那个点落在对话框正文中部，点了没反应，
        # 症状是 STREAM START 永不出现，跟「相机起不来」一模一样（二次踩坑）。
        # 真正的按钮是独立节点：text 严格等于「允许」，bounds=[831,1567][937,1629]
        # → 中心 (884,1598)。改为**整节点匹配**：先拿到含 允许 的完整节点对象，
        # 再要求它的 "text" 严格是 允许（不是标题那种长句）。
        $nodes = [regex]::Matches($json, '\{[^{}]*"text"\s*:\s*"[^"]*允许[^"]*"[^{}]*\}')
        $btn = $null
        foreach ($n in $nodes) {
            $t = [regex]::Match($n.Value, '"text"\s*:\s*"([^"]*)"').Groups[1].Value
            if ($t -eq '允许') { $btn = $n; break }
        }
        if ($btn) {
            $m = [regex]::Match($btn.Value, '"bounds"\s*:\s*"\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
            if ($m.Success) {
                $cx = [int](([int]$m.Groups[1].Value + [int]$m.Groups[3].Value) / 2)
                $cy = [int](([int]$m.Groups[2].Value + [int]$m.Groups[4].Value) / 2)
                Write-Host "[perm] 点击「允许」($cx,$cy)"
                Invoke-Hdc ("uitest uiInput click {0} {1}" -f $cx, $cy) | Out-Null
                Start-Sleep -Seconds 3
                return $true
            }
        }
        Write-Host '[perm] 检测到权限弹窗但没找到「允许」坐标 —— 请手动授权后重跑'
        return $false
    }
    return $false
}
Approve-CameraPermission | Out-Null
Start-Sleep -Seconds 10

$layout = Invoke-Hdc 'uitest dumpLayout -p /data/local/tmp/sweep.json'
if ($layout -notmatch 'saved') { throw 'dumpLayout 失败' }
& $Hdc file recv /data/local/tmp/sweep.json (Join-Path $Root '_scratch\sweep_layout.json') 2>&1 | Out-Null

# 确认真的在相机页（看到档位分段控件才算）
$cam = Invoke-Hdc "hilog -x -e 'STREAM START'"
if ($cam -notmatch 'STREAM START') {
    throw '没看到 STREAM START —— 相机页没起来（权限？Surface？）。检查 UI 后再跑。'
}
Write-Host '[step] 相机页已就绪'

# ── 2. 逐档采集 ──────────────────────────────────────────────────────────────
$settleSec = 45
# 解析档位列表（见 param 里关于为什么收字符串的说明）。
$gearList = @()
foreach ($part in ($Gears -split ',')) {
    $t = $part.Trim()
    if ($t -eq '') { continue }
    $v = 0
    if (-not [int]::TryParse($t, [ref]$v)) { throw "无法解析档位 '$t'（-Gears 应为逗号分隔，如 0,1,3,4）" }
    if (-not $GearXY.ContainsKey($v)) { throw "未知档位 $v（可选 0,1,2,3,4）" }
    $gearList += $v
}
if ($gearList.Count -eq 0) { throw '-Gears 为空' }
Write-Host "[plan] 档位 $($gearList -join ', ') · 每档 settle ${settleSec}s + 观测 ${ObserveSec}s"

$log = @()
foreach ($g in $gearList) {
    $xy = $GearXY[$g]
    Write-Host ''
    Write-Host ("[gear {0}] {1} 点击 ({2},{3})" -f $g, $GearName[$g], $xy[0], $xy[1])
    Invoke-Hdc ("uitest uiInput click {0} {1}" -f $xy[0], $xy[1]) | Out-Null

    # 切档加载（全 NPU 档要连装 3 个模型，实测 ~7 s）+ 稳态门
    $switchWait = 12
    Write-Host ("[gear {0}] 切档等待 {1} s ..." -f $g, $switchWait)
    Start-Sleep -Seconds $switchWait
    Write-Host ("[gear {0}] 稳态门 {1} s ..." -f $g, $settleSec)
    Start-Sleep -Seconds $settleSec
    Write-Host ("[gear {0}] 观测 {1} s ..." -f $g, $ObserveSec)
    Start-Sleep -Seconds $ObserveSec

    # 自证：这一档到底出了多少窗口
    $tail = Invoke-Hdc "grep 'gear=$g' /data/app/el2/100/base/$Bundle/haps/entry/files/camera_run_*.log | tail -40"
    $rateN = ([regex]::Matches($tail, 'RATE win=')).Count
    $hits = 0
    foreach ($m in [regex]::Matches($tail, 'hit_n=(\d+)')) { $hits += [int]$m.Groups[1].Value }
    Write-Host ("[gear {0}] 最近 40 行里 RATE 窗口 {1} 个，检出帧合计 {2}" -f $g, $rateN, $hits)
    $log += "gear=$g rate_windows_in_tail=$rateN hit_frames_in_tail=$hits"
}

# ── 3. 收尾：拉证据 ──────────────────────────────────────────────────────────
Write-Host ''
Write-Host '[step] 拉取证据'
# 用当前宿主的 powershell，不要写死 `pwsh`（Windows PowerShell 5.1 上没有这个别名）。
$psExe = (Get-Process -Id $PID).Path
& $psExe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'pull_camera_evidence.ps1') `
    -Tag $Tag -SkipBuildCheck

$log | Out-File -Encoding utf8 (Join-Path $Root "evidence\camera_$Tag.sweep.txt")
Write-Host ''
Write-Host '[done] 下一步：python tools/parse_camera_run.py evidence/camera_<tag>.log'
