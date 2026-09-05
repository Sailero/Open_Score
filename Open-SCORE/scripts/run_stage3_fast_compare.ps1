[CmdletBinding()]
param(
    [switch]$Background,
    [ValidateRange(1,8)][int]$Workers = 4,
    [string]$Python = 'D:\Software\Anaconda\envs\torch310\python.exe'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot 'outputs\stage123_unknown_upper_v1'
$FastRoot = Join-Path $RunRoot 'stage3\fast_compare_v2'
$ScriptPath = Join-Path $PSScriptRoot 'run_stage3_fast_compare.py'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
New-Item -ItemType Directory -Path $FastRoot -Force | Out-Null
if ($Background) {
    $TaskProcess = Start-Process -FilePath $Python -ArgumentList @('-u', ('"' + $ScriptPath + '"'), '--workers', "$Workers") -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $FastRoot 'launcher_stdout.log') -RedirectStandardError (Join-Path $FastRoot 'launcher_stderr.log') -PassThru
    [pscustomobject]@{ PID=$TaskProcess.Id; Progress=(Join-Path $RunRoot 'pipeline.log'); Status=(Join-Path $RunRoot 'status.json'); Report=(Join-Path $RunRoot 'experiment_report.md'); Results=$FastRoot } | Format-List
    return
}
Push-Location -LiteralPath $ProjectRoot
try {
    & $Python -u $ScriptPath --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Fast comparison failed with code $LASTEXITCODE; inspect pipeline.log" }
} finally { Pop-Location }
