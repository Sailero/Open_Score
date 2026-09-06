[CmdletBinding()]
param(
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot 'Open-SCORE'))
$runDirectory = Join-Path $projectRoot 'outputs\v5_parallel\v5_20260907_main'
$runnerLockPath = Join-Path $runDirectory '.runner.lock'
$pythonExecutable = 'D:\Software\Anaconda\envs\torch310\python.exe'

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Required Python executable is missing: $pythonExecutable"
}

$runnerIsActive = $false
$runnerProcessId = 0
if (Test-Path -LiteralPath $runnerLockPath -PathType Leaf) {
    try {
        $runnerLock = Get-Content -LiteralPath $runnerLockPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $runnerProcessId = [int]$runnerLock.pid
    }
    catch {
        throw "Cannot verify the runner lock. Retry after checking $runnerLockPath. $($_.Exception.Message)"
    }
    if ($runnerProcessId -gt 0) {
        $runnerProcess = Get-Process -Id $runnerProcessId -ErrorAction SilentlyContinue
        if ($null -ne $runnerProcess) {
            $runnerDetails = Get-CimInstance Win32_Process -Filter "ProcessId = $runnerProcessId"
            $runnerCommandLine = [string]$runnerDetails.CommandLine
            $runnerIsActive = (
                $runnerCommandLine -match '(?i)(open_score\.research_v5(?:\.orchestrate)?|run_research_v5\.py)' -and
                $runnerCommandLine -match '(?i)(?:^|[\s"])run-all(?:[\s"]|$)'
            )
        }
    }
}

if ($Once -and -not $runnerIsActive) {
    throw 'No verified active v5 run-all process. -Once only monitors; it never starts training.'
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonArguments = @('-X', 'utf8', '-u', '-m', 'open_score.research_v5.orchestrate')
if ($runnerIsActive) {
    Write-Host "[v5.1] Attaching progress monitor to existing runner PID $runnerProcessId."
    $pythonArguments += @('watch', '--run-dir', $runDirectory)
    if ($Once) {
        $pythonArguments += '--once'
    }
}
else {
    Write-Host '[v5.1] Starting or resuming the frozen six-task run.'
    $pythonArguments += @('run-all', '--run-dir', $runDirectory, '--parallel-tasks', '6')
}

& $pythonExecutable @pythonArguments
exit $LASTEXITCODE
