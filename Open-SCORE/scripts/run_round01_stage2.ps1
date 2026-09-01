param(
    [switch]$DryRun,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project

$condaHook = "D:\Software\Anaconda\shell\condabin\conda-hook.ps1"
if (-not (Test-Path -LiteralPath $condaHook)) {
    throw "Conda PowerShell hook not found: $condaHook"
}

& $condaHook
conda activate torch310
$env:PYTHONUTF8 = "1"

Write-Host "Conda environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Cyan
Write-Host "Round-01 Stage 2 output: outputs\round_01_mvp\stage2\final" -ForegroundColor Cyan
$stage2Args = @(
    "-u",
    "scripts\run_round01_stage2.py",
    "--config",
    "configs\stage2_round01_final.yaml"
)
if ($DryRun) {
    $stage2Args += "--dry-run"
}
if ($Smoke) {
    $stage2Args += "--smoke"
}
python @stage2Args

if ($LASTEXITCODE -ne 0) {
    throw "Round 01 Stage 2 failed with exit code $LASTEXITCODE"
}
