<#
.SYNOPSIS
    从真机取回一轮相机复采的证据（App 落盘文件 + 覆盖该时间窗的 hilog 持久化文件）。

.DESCRIPTION
    为什么需要这个脚本 —— 2026-09-21 踩到的两个坑，都靠它固化：

    1. **hilog 是 16 MB 环形缓冲，会滚掉。** 06:04 那一轮相机数据在 06:47 已从
       缓冲里完全消失（`hilog -x` 里 `LprCamera: STAGE` = 0 行），当时只能从设备
       的持久化文件 `/data/log/hilog/hilog.NNN.*.gz` 里捞回来。能捞到是运气
       （依赖滚动窗口正好覆盖），不能当成常态。
       所以 App 侧已经把每个 RATE/STAGE 窗口写进 `filesDir/camera_run_<ts>.log`
       （主证据），hilog 只作辅证。

    2. **原生优化标志必须先查。** `-O0` 会让全部 native 慢 4 倍，且**没有任何
       报错**（见 docs/notes/build-mode-o0-regression.md）。构建模式不对的话，
       这一轮所有性能数字都不可用 —— 所以在拉数据**之前**先验证。

    设备侧持久化文件是 **gzip**，但 hdc 拉回来的是原始字节流（`hdc file recv`
    不做传输解码）。本脚本用 .NET GZipStream 解压后按 tag 过滤。

.PARAMETER Tag
    本轮证据的标签，用于命名归档文件（如 `prod-r1`、`npu-r2`）。

.PARAMETER SkipBuildCheck
    跳过 native 优化标志检查（仅在明知构建模式时使用）。

.PARAMETER KeepRaw
    保留解压后的完整 hilog 文本（默认只留过滤后的行）。

.EXAMPLE
    pwsh tools/pull_camera_evidence.ps1 -Tag prod-r1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [switch]$SkipBuildCheck,
    [switch]$KeepRaw
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot          # 仓库根
$Evidence = Join-Path $Root 'evidence'
$Scratch = Join-Path $Root '_scratch\hilog_pull'
$Hdc = 'D:\IDE\DevEco_Studio\sdk\default\openharmony\toolchains\hdc.exe'
$Bundle = 'com.shusen.lprdemo'
$AppFiles = "/data/app/el2/100/base/$Bundle/haps/entry/files"

foreach ($d in @($Evidence, $Scratch)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
if (-not (Test-Path $Hdc)) { throw "找不到 hdc: $Hdc" }

# ── 0. 设备在不在 ────────────────────────────────────────────────────────────
$targets = (& $Hdc list targets 2>&1) | Where-Object { $_ -and $_ -notmatch 'Empty' }
if (-not $targets) { throw '没有连接的设备（hdc list targets 为空）' }
Write-Host "[device] $($targets -join ', ')"

# ── 1. native 优化标志（纪律 1）──────────────────────────────────────────────
if (-not $SkipBuildCheck) {
    $chk = Join-Path $PSScriptRoot 'check_native_build_flags.py'
    if (Test-Path $chk) {
        $out = & python $chk 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-Host $out
            throw 'native 优化标志检查未通过 —— 这一轮的性能数字不可用（-O0 慢 4 倍且无报错）。先按 buildMode=release 重建。'
        }
        Write-Host '[build] native 优化标志 OK'
    } else {
        Write-Warning "找不到 $chk，跳过优化标志检查"
    }
}

# ── 2. 拉 App 落盘文件（主证据）──────────────────────────────────────────────
$listing = (& $Hdc shell "ls -t $AppFiles/camera_run_*.log" 2>&1) |
    Where-Object { $_ -match 'camera_run_\d+\.log' }
if (-not $listing) {
    throw "设备上没有 camera_run_*.log —— App 的落盘没生效（检查 hilog 里的 RUNLOG 行）。"
}
$newest = ($listing | Select-Object -First 1).Trim()
$localLog = Join-Path $Evidence "camera_$Tag.log"
& $Hdc file recv $newest $localLog 2>&1 | Select-Object -Last 1 | ForEach-Object { Write-Host "[app] $_" }
Write-Host "[app] $newest -> $localLog"

# 同时把该目录下所有 camera_run 文件的**清单**记下来，方便判断有没有多轮混在一起
(& $Hdc shell "ls -l $AppFiles/camera_run_*.log" 2>&1) |
    ForEach-Object { $_.Trim() } |
    Out-File -Encoding utf8 (Join-Path $Evidence "camera_$Tag.files.txt")

# ── 3. 拉覆盖该时间窗的 hilog 持久化文件（辅证）──────────────────────────────
# 持久化任务是 4 MB 滚动，约每 2-3 分钟一个文件；取最近 12 个足够覆盖一轮。
$recent = (& $Hdc shell "ls -t /data/log/hilog/hilog.*.gz 2>/dev/null | head -12" 2>&1) |
    Where-Object { $_ -match '\.gz' }
if (-not $recent) {
    Write-Warning '设备上没有 /data/log/hilog/hilog.*.gz —— 无法取 hilog 辅证（App 落盘仍是主证据）'
} else {
    $pulled = @()
    foreach ($f in $recent) {
        $name = [System.IO.Path]::GetFileName($f.Trim())
        $dest = Join-Path $Scratch $name
        if (-not (Test-Path $dest)) {
            & $Hdc file recv $f.Trim() $dest 2>&1 | Out-Null
        }
        if (Test-Path $dest) { $pulled += $dest }
    }
    Write-Host "[hilog] 拉回 $($pulled.Count) 个持久化文件"

    # 解压 + 过滤。gzip 里是**二进制混合流**（多个进程的日志交错），
    # 不能按行解析，只能按字节找锚点再截取上下文。
    $hilogOut = Join-Path $Evidence "camera_$Tag.hilog.txt"
    $rawDir = Join-Path $Scratch 'raw'
    if (-not (Test-Path $rawDir)) { New-Item -ItemType Directory -Force -Path $rawDir | Out-Null }

    $anchors = @('RATE arrive', 'STAGE n=', 'FRAME n=', 'GEAR ', 'STREAM START',
                 'FPS SET', 'FPS RANGES', 'SETTLE done', 'RUNLOG path', 'CAMRUN BEGIN',
                 'GEAR SWITCH', 'FIRST FRAME')
    $sb = New-Object System.Text.StringBuilder
    foreach ($gz in $pulled) {
        try {
            $in = [System.IO.File]::OpenRead($gz)
            $zs = New-Object System.IO.Compression.GZipStream($in, [System.IO.Compression.CompressionMode]::Decompress)
            $ms = New-Object System.IO.MemoryStream
            $zs.CopyTo($ms)
            $bytes = $ms.ToArray()
            $zs.Close(); $in.Close()
        } catch {
            Write-Warning "解压失败 $gz : $($_.Exception.Message)"
            continue
        }
        if ($KeepRaw) {
            $rawPath = Join-Path $rawDir ([System.IO.Path]::GetFileNameWithoutExtension($gz) + '.bin')
            [System.IO.File]::WriteAllBytes($rawPath, $bytes)
        }
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
        foreach ($a in $anchors) {
            $idx = 0
            while (($idx = $text.IndexOf($a, $idx)) -ge 0) {
                $len = [Math]::Min(240, $text.Length - $idx)
                $chunk = $text.Substring($idx, $len)
                # 二进制流里混着非文本字节，过滤掉不可打印字符
                $clean = -join ($chunk.ToCharArray() | ForEach-Object {
                    if ([char]::IsControl($_) -and $_ -ne "`t") { ' ' } else { $_ }
                })
                [void]$sb.AppendLine($clean.Trim())
                $idx += $a.Length
            }
        }
        Write-Host "[hilog] $(Split-Path -Leaf $gz) -> $($bytes.Length) bytes"
    }
    $sb.ToString() | Out-File -Encoding utf8 $hilogOut
    Write-Host "[hilog] 过滤结果 -> $hilogOut"
}

# ── 4. 自证：这一轮到底有没有数据 ────────────────────────────────────────────
$lines = Get-Content $localLog -ErrorAction SilentlyContinue
$rates = @($lines | Where-Object { $_ -match '^RATE ' })
$settle = @($lines | Where-Object { $_ -match '^SETTLE done' })
$switch = @($lines | Where-Object { $_ -match '^GEAR SWITCH' })
Write-Host ''
Write-Host '── 本轮自证 ──────────────────────────────'
Write-Host "  落盘行数      : $($lines.Count)"
Write-Host "  稳态窗口(RATE): $($rates.Count)"
Write-Host "  稳态门通过    : $($settle.Count)"
Write-Host "  档位切换      : $($switch.Count)"

# 每档的检出/未检出样本数 —— 样本不足的档位不报 p50，这里先亮出来
$byGear = @{}
foreach ($r in $rates) {
    if ($r -match 'gear=(\d+)') { $g = $Matches[1] } else { continue }
    if ($r -match 'hit_n=(\d+)') { $h = [int]$Matches[1] } else { $h = 0 }
    if ($r -match 'empty_n=(\d+)') { $e = [int]$Matches[1] } else { $e = 0 }
    if (-not $byGear.ContainsKey($g)) { $byGear[$g] = @{ hit = 0; empty = 0; win = 0 } }
    $byGear[$g].hit += $h
    $byGear[$g].empty += $e
    $byGear[$g].win += 1
}
foreach ($g in ($byGear.Keys | Sort-Object)) {
    $v = $byGear[$g]
    Write-Host ("  gear={0}  窗口={1}  检出帧={2}  未检出帧={3}" -f $g, $v.win, $v.hit, $v.empty)
}
Write-Host '──────────────────────────────────────────'

if ($rates.Count -lt 20) {
    Write-Warning "稳态窗口只有 $($rates.Count) 个（期望 ≥20）—— 这轮时长不够或相机爬坡没走完，建议重跑。"
}
$anyHit = $byGear.Values | Where-Object { $_.hit -gt 0 }
if (-not $anyHit) {
    Write-Warning '本轮**没有任何检出帧** —— 检出桶无样本，无法回答「检出时的延迟」。对着车牌重跑这一档。'
}
