[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Formal")]
    [string]$Mode = "Formal",
    [string]$Python = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$RunRoot = "",
    [switch]$Background,
    [switch]$SkipTests,
    [switch]$ResumeStage3
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

$Arguments = [System.Collections.Generic.List[string]]::new()
$Arguments.Add("-u")
$Arguments.Add("scripts\run_stage23_pipeline.py")
$Arguments.Add("--mode")
$Arguments.Add($Mode.ToLowerInvariant())
$Arguments.Add("--run-root")
$Arguments.Add($ResolvedRoot)
if ($SkipTests) {
    $Arguments.Add("--skip-tests")
}
if ($ResumeStage3) {
    if ($Mode -ne "Formal") {
        throw "-ResumeStage3 can only be used with -Mode Formal"
    }
    $Arguments.Add("--reuse-stage2")
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
    if ($ResumeStage3) {
        Write-Host "[Stage3 recovery] Background process started. PID=$($Process.Id)" -ForegroundColor Green
        Write-Host "Stage2 will be hash-validated and reused; collection/training will not run." -ForegroundColor Yellow
    } else {
        Write-Host "[Stage2->Stage3] Background process started. PID=$($Process.Id)" -ForegroundColor Green
    }
    Write-Host "Progress: $ResolvedRoot\pipeline.log" -ForegroundColor Cyan
    Write-Host "Status:   $ResolvedRoot\pipeline_status.json" -ForegroundColor Cyan
    Write-Host "Report:   $ResolvedRoot\stage23_report.md" -ForegroundColor Cyan
    exit 0
}

if ($ResumeStage3) {
    Write-Host "[Stage3 recovery] Validating and reusing accepted Stage2" -ForegroundColor Green
} else {
    Write-Host "[Stage2->Stage3] Running in foreground: $Mode" -ForegroundColor Green
}
& $Python @($Arguments.ToArray())
if ($LASTEXITCODE -ne 0) {
    throw "Aligned Stage2->Stage3 pipeline failed with exit code $LASTEXITCODE"
}
