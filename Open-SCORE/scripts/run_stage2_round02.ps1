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
Write-Host "Stage 2 output: outputs\round_02_stage2_dynamic" -ForegroundColor Cyan
$stage2Args = @(
    "-u",
    "scripts\run_stage2_round02.py",
    "--config",
    "configs\stage2_round02_dynamic.yaml"
)
if ($DryRun) {
    $stage2Args += "--dry-run"
}
if ($Smoke) {
    $stage2Args += "--smoke"
}
python @stage2Args

if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 round-02 failed with exit code $LASTEXITCODE"
}
