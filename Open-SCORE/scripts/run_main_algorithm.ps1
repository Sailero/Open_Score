param(
    [string]$Python = 'D:\Software\Anaconda\envs\torch310\python.exe',
    [double]$TrainMinutes = 5,
    [int]$TrainSteps = 2000,
    [int]$EvalEpisodes = 20,
    [double]$EvalMinutes = 5,
    [string]$Output = 'outputs/v2_main',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [ValidateSet('reactive', 'concentrated', 'balanced')][string]$Opponent = 'reactive',
    [int]$Seed = 20260905,
    [switch]$Resume,
    [switch]$EvaluateOnly
)
$ErrorActionPreference = 'Stop'
if ($TrainMinutes -le 0 -or $TrainSteps -lt 0 -or $EvalEpisodes -lt 1 -or $EvalMinutes -le 0) {
    throw 'Budgets must be positive; TrainSteps=0 means time budget only.'
}
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python not found: $Python. Pass -Python with your PyTorch environment executable."
}
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOldPythonPath = $env:PYTHONPATH
$taskMode = if ($EvaluateOnly) { 'evaluate' } else { 'run' }
$taskArguments = @('-m', 'open_score', $taskMode, '--config', 'configs/known_opponent_v2.yaml',
    '--profile', 'briefing', '--method', 'selective', '--device', $Device, '--opponent', $Opponent,
    '--seed', "$Seed", '--steps', "$TrainSteps", '--train-seconds', "$($TrainMinutes * 60)",
    '--eval-seconds', "$($EvalMinutes * 60)", '--eval-episodes', "$EvalEpisodes", '--output', $Output)
if ($Resume) { $taskArguments += '--resume' }
Push-Location -LiteralPath $taskRoot
try {
    $env:PYTHONPATH = Join-Path $taskRoot 'src'
    & $Python @taskArguments
    if ($LASTEXITCODE -ne 0) { throw "Main algorithm exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $taskOldPythonPath
    Pop-Location
}
