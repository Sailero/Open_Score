[CmdletBinding()]
param(
    [ValidateSet("Formal")]
    [string]$Mode = "Formal",
    [switch]$Background,
    [ValidateRange(1,16)]
    [int]$Workers = 2,
    [string]$Python = "D:\Software\Anaconda\envs\torch310\python.exe"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $ProjectRoot "outputs\stage123_unknown_upper_v1"
$ScriptPath = Join-Path $PSScriptRoot "run_stage123_unknown_upper.py"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
if ($Background) {
    $Process = Start-Process -FilePath $Python `
        -ArgumentList @("-u", ('"' + $ScriptPath + '"'), "--workers", "$Workers") `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RunRoot "launcher_stdout.log") `
        -RedirectStandardError (Join-Path $RunRoot "launcher_stderr.log") -PassThru
    [pscustomobject]@{
        PID = $Process.Id
        Progress = Join-Path $RunRoot "pipeline.log"
        Status = Join-Path $RunRoot "status.json"
        Report = Join-Path $RunRoot "experiment_report.md"
    } | Format-List
    return
}
Push-Location -LiteralPath $ProjectRoot
try {
    & $Python -u $ScriptPath --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Formal pipeline failed (exit $LASTEXITCODE); rerun the same command to resume." }
} finally {
    Pop-Location
}
