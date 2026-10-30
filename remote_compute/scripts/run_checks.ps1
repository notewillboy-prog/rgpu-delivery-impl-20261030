# RGPU 一键自检：跑全部 unittest（含端到端，server/client 未落地时 e2e 自动 skip）。
#
# 用法（项目根目录或任意目录）：
#   pwsh -File scripts\run_checks.ps1            # 全部（含 e2e）
#   pwsh -File scripts\run_checks.ps1 -SkipE2E   # 只跑离线单测
#   pwsh -File scripts\run_checks.ps1 -VerboseTests
#
# 注意（本机实测）：Windows PowerShell 默认执行策略会拦 .ps1（报
# "AuthorizationManager 检查失败"），用下面的方式运行即可：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_checks.ps1
# 脚本文件本身带 UTF-8 BOM，兼容 Windows PowerShell 5.1 下中文注释的解析。
#
# 退出码：0 = 全部通过；非 0 = 有失败（直接透传 unittest 的退出码）。
param(
    [switch]$SkipE2E,
    [switch]$VerboseTests
)

$ErrorActionPreference = 'Stop'

# --- UTF-8：控制台读写 + Python 子进程 --------------------------------------
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
} catch {
    Write-Host "警告: 设置 UTF-8 输出失败（不影响测试执行）: $($_.Exception.Message)" -ForegroundColor Yellow
}

if ($SkipE2E) {
    $env:RGPU_SKIP_E2E = '1'
} else {
    $env:RGPU_SKIP_E2E = '0'
}

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "找不到项目 venv Python: $python" -ForegroundColor Red
    Write-Host "（本项目所有命令统一用 .venv 的绝对路径解释器，不要依赖 PATH）" -ForegroundColor Yellow
    exit 1
}

Push-Location -LiteralPath $root
try {
    Write-Host "== RGPU 自检 ==" -ForegroundColor Cyan
    Write-Host "项目根 : $root"
    Write-Host "解释器 : $python"
    Write-Host "e2e    : $(if ($SkipE2E) { '跳过（-SkipE2E）' } else { '启用（server/client 缺失时自动 skip）' })"
    Write-Host ''

    $unittestArgs = @('-m', 'unittest', 'discover', '-s', 'tests')
    if ($VerboseTests) { $unittestArgs += '-v' }

    & $python @unittestArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ''
if ($code -ne 0) {
    Write-Host "检查失败：unittest 退出码 $code" -ForegroundColor Red
    exit $code
}
Write-Host '全部检查通过（unittest discover -s tests）' -ForegroundColor Green
exit 0
