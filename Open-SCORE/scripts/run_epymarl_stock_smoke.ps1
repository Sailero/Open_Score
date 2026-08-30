<#
.SYNOPSIS
Run official EPyMARL QMIX, VDN and MAPPO entry points on stock SMAClite.

.DESCRIPTION
The defaults are intentionally a pipeline smoke (50 requested environment
steps), not a convergence experiment. Sacred writes the full config, source
commit, host information and metrics under the ignored EPyMARL results folder.
Use -UseStockTrainingDefaults for a longer check without the smoke-only replay
and batch overrides. Long runs schedule evaluation at 90% of the requested
budget unless TestInterval is explicit, avoiding a final episode that skips
the only post-training test. Neither mode is a convergence claim by itself.
#>

param(
    [ValidateSet("all", "qmix", "vdn", "mappo")]
    [string]$Algorithm = "all",
    [string]$MapName = "2s_vs_1sc",
    [int]$TMax = 50,
    [int]$TimeLimit = 25,
    [int]$TestEpisodes = 1,
    [int]$TestInterval = 0,
    [int]$LogInterval = 0,
    [int]$Seed = 20260830,
    [switch]$Cpu,
    [switch]$UseStockTrainingDefaults,
    [switch]$SaveModel,
    [switch]$ShowWarnings,
    [string]$PythonPath = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$GitPath = "D:\Software\Git\cmd\git.exe",
    [string]$EPyMARLPath = ""
)

$ErrorActionPreference = "Stop"
$ExpectedCommit = "cbc38c09588064eab978501d0f12c2cf58fa7fc2"
if ($TMax -lt 1 -or $TimeLimit -lt 1 -or $TestEpisodes -lt 1 -or $TestInterval -lt 0 -or $LogInterval -lt 0) {
    throw "TMax, TimeLimit and TestEpisodes must be positive; TestInterval and LogInterval must be non-negative"
}
$EffectiveLogInterval = if ($LogInterval -eq 0) { $TimeLimit } else { $LogInterval }
$EffectiveTestInterval = if ($TestInterval -gt 0) {
    $TestInterval
} elseif ($TMax -le 50) {
    $TMax
} else {
    [Math]::Max(1, [int][Math]::Floor(0.9 * $TMax))
}
$UseConfiguredTimeLimit = $UseStockTrainingDefaults -and -not $PSBoundParameters.ContainsKey("TimeLimit")
if ([string]::IsNullOrWhiteSpace($EPyMARLPath)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $EPyMARLPath = Join-Path $ProjectRoot "upstream\external\epymarl"
} else {
    $EPyMARLPath = [System.IO.Path]::GetFullPath($EPyMARLPath)
}
foreach ($RequiredFile in @($PythonPath, $GitPath, (Join-Path $EPyMARLPath "src\main.py"))) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile. Run install_epymarl_windows.ps1 first."
    }
}
$ActualCommit = (& $GitPath -C $EPyMARLPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $ExpectedCommit) {
    throw "Expected clean EPyMARL commit $ExpectedCommit, found $ActualCommit"
}
if (& $GitPath -C $EPyMARLPath status --porcelain --untracked-files=no) {
    throw "EPyMARL tracked sources are dirty; stock reproduction requires an unmodified checkout"
}

$env:GIT_PYTHON_GIT_EXECUTABLE = $GitPath
$env:Path = (Split-Path -Parent $GitPath) + ";" + $env:Path
$env:PYGAME_HIDE_SUPPORT_PROMPT = "1"
if (-not $ShowWarnings) {
    $env:PYTHONWARNINGS = "ignore::UserWarning"
}
$Algorithms = if ($Algorithm -eq "all") { @("qmix", "vdn", "mappo") } else { @($Algorithm) }
$CudaValue = if ($Cpu) { "False" } else { "True" }
$SaveModelValue = if ($SaveModel) { "True" } else { "False" }

Push-Location $EPyMARLPath
try {
    foreach ($Name in $Algorithms) {
        if ($UseStockTrainingDefaults -and $Name -eq "mappo" -and ($TestEpisodes % 10) -ne 0) {
            throw "Official MAPPO uses batch_size_run=10, so TestEpisodes must be divisible by 10"
        }
        $RunLabel = if ($TMax -le 50 -and -not $UseStockTrainingDefaults) {
            "pipeline smoke"
        } elseif ($UseStockTrainingDefaults -and $UseConfiguredTimeLimit) {
            "stock-horizon and algorithm-default short-budget check"
        } elseif ($UseStockTrainingDefaults) {
            "algorithm-default check with explicit time_limit=$TimeLimit"
        } else {
            "extended smoke-hyperparameter diagnostic"
        }
        Write-Output "Running stock SMAClite $MapName with official EPyMARL $Name ($RunLabel)"
        $RunArguments = @(
            "src\main.py",
            "--config=$Name",
            "--env-config=smaclite",
            "with",
            "env_args.map_name=$MapName",
            "seed=$Seed",
            "t_max=$TMax",
            "test_interval=$EffectiveTestInterval",
            "test_nepisode=$TestEpisodes",
            "log_interval=$EffectiveLogInterval",
            "runner_log_interval=$EffectiveLogInterval",
            "learner_log_interval=$EffectiveLogInterval",
            "use_cuda=$CudaValue",
            "save_model=$SaveModelValue"
        )
        if (-not $UseConfiguredTimeLimit) {
            $RunArguments += "env_args.time_limit=$TimeLimit"
        }
        if (-not $UseStockTrainingDefaults) {
            if ($Name -eq "mappo") {
                $BatchSize = 1
                $BufferSize = 2
            } else {
                $BatchSize = 2
                $BufferSize = 10
            }
            $RunArguments += @(
                "batch_size_run=1",
                "batch_size=$BatchSize",
                "buffer_size=$BufferSize"
            )
        }
        & $PythonPath @RunArguments
        if ($LASTEXITCODE -ne 0) {
            throw "EPyMARL $Name smoke failed with exit code $LASTEXITCODE"
        }
    }
} finally {
    Pop-Location
}

$CompletedLabel = if ($TMax -le 50 -and -not $UseStockTrainingDefaults) {
    "pipeline smoke"
} elseif ($UseStockTrainingDefaults -and $UseConfiguredTimeLimit) {
    "stock-horizon and algorithm-default short-budget check"
} elseif ($UseStockTrainingDefaults) {
    "algorithm-default check with explicit time_limit=$TimeLimit"
} else {
    "extended smoke-hyperparameter diagnostic"
}
Write-Output "Completed EPyMARL stock $CompletedLabel. These runs do not establish convergence."
