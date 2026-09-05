[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Formal")]
    [string]$Mode = "Formal",
    [string]$Python = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$RunRoot = "",
    [switch]$Background,
    [switch]$SkipTests,
    [switch]$SkipPhysical,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python was not found: $Python"
}
if (-not $RunRoot) {
    throw "Historical runner: specify -RunRoot explicitly. Current experiments use run_stage123_unknown_upper.ps1."
}
$ResolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RunRoot))
New-Item -ItemType Directory -Path $ResolvedRoot -Force | Out-Null

if (-not $SkipTests) {
    Write-Host "[SALDAE-DO] Running solver and identity-constraint tests" -ForegroundColor Green
    & $Python -m pytest `
        tests/test_stage3_saldae.py `
        tests/test_stage3_identity_blotto.py `
        tests/test_stage3_identity_reserve.py `
        -q
    if ($LASTEXITCODE -ne 0) {
        throw "SALDAE-DO tests failed with exit code $LASTEXITCODE"
    }
}

$Arguments = [System.Collections.Generic.List[string]]::new()
$Arguments.Add("-u")
$Arguments.Add("scripts\evaluate_stage3_saldae.py")
$Arguments.Add("--config")
$Arguments.Add("configs\stage3_saldae.yaml")
$Arguments.Add("--output-dir")
$Arguments.Add($ResolvedRoot)
$Arguments.Add("--device")
$Arguments.Add($Device)
if ($Mode -eq "Smoke") {
    $Arguments.Add("--smoke")
}
if ($SkipPhysical) {
    $Arguments.Add("--skip-physical")
}

if ($Background) {
    $Stdout = Join-Path $ResolvedRoot "launcher_stdout.log"
    $Stderr = Join-Path $ResolvedRoot "launcher_stderr.log"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments.ToArray() `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru
    Write-Host "[SALDAE-DO] Background experiment started. PID=$($Process.Id)" -ForegroundColor Green
    Write-Host "Progress: $ResolvedRoot\live_progress.log" -ForegroundColor Cyan
    Write-Host "Status:   $ResolvedRoot\pipeline_status.json" -ForegroundColor Cyan
    Write-Host "Report:   $ResolvedRoot\stage3_saldae_report.md" -ForegroundColor Cyan
    exit 0
}

Write-Host "[SALDAE-DO] Running in foreground: $Mode" -ForegroundColor Green
& $Python @($Arguments.ToArray())
if ($LASTEXITCODE -ne 0) {
    throw "SALDAE-DO experiment failed with exit code $LASTEXITCODE"
}
