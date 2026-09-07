[CmdletBinding()]
param([switch]$StatusOnly)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Join-Path $workspaceRoot 'Open-SCORE'
$pythonExecutable = 'D:\Software\Anaconda\envs\torch310\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $projectRoot

if ($StatusOnly) {
    & $pythonExecutable -u scripts/run_research_v5.py status --watch --run-dir outputs/v5
}
else {
    & $pythonExecutable -u scripts/run_research_v5.py resume --run-dir outputs/v5
}
exit $LASTEXITCODE