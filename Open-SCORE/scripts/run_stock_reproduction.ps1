<#
.SYNOPSIS
Run the preregistered stock SMAClite reproduction matrix on Windows.

.DESCRIPTION
Reads a committed JSON preregistration, verifies both pinned upstream
checkouts, the installed C++ RVO2 binary, and all registered source hashes,
then launches EPyMARL without editing its checkout. Checkpoints are saved with
a Windows-safe timestamp compatibility overlay. Runtime overrides are allowed only with -AllowProtocolOverride and
are labelled override_non_reproduction in the run record.

Resume uses EPyMARL's checkpoint loader.  It restores networks and optimizers,
but not replay, target-network, environment, or RNG state; resumed runs are
therefore always diagnostic rather than registered reproductions.
#>

param(
    [string]$ManifestPath = "",
    [string]$Profile = "",
    [ValidateSet("primary", "cross_algorithm_matrix")]
    [string]$Suite = "primary",
    [ValidateSet("all", "qmix", "vdn", "mappo")]
    [string]$Algorithm = "all",
    [string]$MapName = "",
    [int[]]$Seeds = @(),
    [int]$TrainingSteps = 0,
    [int]$TestEpisodes = 0,
    [int]$TestInterval = 0,
    [int]$TimeLimit = 0,
    [int]$SaveModelInterval = 0,
    [switch]$Cpu,
    [switch]$UseCppRvo2,
    [switch]$UseNumpyRvo2,
    [switch]$Resume,
    [string]$ResumeFrom = "",
    [int]$LoadStep = 0,
    [switch]$AllowProtocolOverride,
    [switch]$DryRun,
    [switch]$ContinueOnError,
    [string]$PythonPath = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$GitPath = "D:\Software\Git\cmd\git.exe",
    [string]$EPyMARLPath = "",
    [string]$SMAClitePath = "",
    [string]$RVO2Path = "",
    [string]$OutputRoot = "",
    [Int64]$CpuAffinityMask = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $ProjectRoot "configs\stock_reproduction\smaclite_aamas2023_epymarl_v3.json"
}
$ManifestPath = [System.IO.Path]::GetFullPath($ManifestPath)
if ([string]::IsNullOrWhiteSpace($EPyMARLPath)) {
    $EPyMARLPath = Join-Path $ProjectRoot "upstream\external\epymarl"
}
if ([string]::IsNullOrWhiteSpace($SMAClitePath)) {
    $SMAClitePath = Join-Path $ProjectRoot "upstream\external\smaclite-v2.0.0"
}
if ([string]::IsNullOrWhiteSpace($RVO2Path)) {
    $RVO2Path = Join-Path $ProjectRoot "upstream\external\SMAClite-Python-RVO2"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "outputs\stock_reproduction"
}
$EPyMARLPath = [System.IO.Path]::GetFullPath($EPyMARLPath)
$SMAClitePath = [System.IO.Path]::GetFullPath($SMAClitePath)
$RVO2Path = [System.IO.Path]::GetFullPath($RVO2Path)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$WindowsSiteDirectory = Join-Path $PSScriptRoot "epymarl_windows_site"
$WindowsSiteCustomise = Join-Path $WindowsSiteDirectory "sitecustomize.py"
$RuntimeCpuProbe = Join-Path $ProjectRoot ([string](
    (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).source_pins.runtime_cpu_probe.path
))
$EPyMARLMain = Join-Path $EPyMARLPath "src\main.py"

foreach ($RequiredFile in @(
    $ManifestPath,
    $PythonPath,
    $GitPath,
    $WindowsSiteCustomise,
    $RuntimeCpuProbe,
    $EPyMARLMain
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $SMAClitePath ".git") -PathType Container)) {
    throw "Pinned SMAClite checkout not found: $SMAClitePath"
}
if (-not (Test-Path -LiteralPath (Join-Path $RVO2Path ".git") -PathType Container)) {
    throw "Pinned SMAClite-Python-RVO2 checkout not found: $RVO2Path"
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($Manifest.schema_version -ne 1) {
    throw "Unsupported manifest schema_version: $($Manifest.schema_version)"
}
if ([string]::IsNullOrWhiteSpace($Profile)) {
    $Profile = [string]$Manifest.default_profile
}
$ProfileConfig = $Manifest.profiles.PSObject.Properties[$Profile].Value
if ($null -eq $ProfileConfig) {
    throw "Unknown profile '$Profile'"
}
if (-not $ProfileConfig.runnable) {
    throw "Profile '$Profile' is non-runnable: $($ProfileConfig.reason)"
}

$CurrentProcess = [System.Diagnostics.Process]::GetCurrentProcess()
$AvailableCpuAffinityMask = [Int64]$CurrentProcess.ProcessorAffinity.ToInt64()
$RequireSingleCoreAffinity = [bool]$ProfileConfig.require_single_core_affinity
$AppliedCpuAffinityMask = $null
$AppliedCpuLogicalIndex = $null
if ($CpuAffinityMask -lt 0) {
    throw "CpuAffinityMask must be non-negative"
}
if ($CpuAffinityMask -eq 0 -and $RequireSingleCoreAffinity) {
    $CpuAffinityMask = $AvailableCpuAffinityMask -band (-$AvailableCpuAffinityMask)
}
if ($CpuAffinityMask -gt 0) {
    if (($CpuAffinityMask -band $AvailableCpuAffinityMask) -ne $CpuAffinityMask) {
        throw "CpuAffinityMask $CpuAffinityMask selects a logical processor outside the launcher's available affinity $AvailableCpuAffinityMask"
    }
    if ($RequireSingleCoreAffinity -and (($CpuAffinityMask -band ($CpuAffinityMask - 1)) -ne 0)) {
        throw "The registered profile requires a one-bit CpuAffinityMask"
    }
    $CurrentProcess.ProcessorAffinity = [IntPtr]$CpuAffinityMask
    $AppliedCpuAffinityMask = [Int64]$CpuAffinityMask
    $ProbeMask = [Int64]$CpuAffinityMask
    $ProbeIndex = 0
    while (($ProbeMask -band 1) -eq 0) {
        $ProbeMask = $ProbeMask -shr 1
        $ProbeIndex += 1
    }
    $AppliedCpuLogicalIndex = $ProbeIndex
}
if ($RequireSingleCoreAffinity -and $null -eq $AppliedCpuAffinityMask) {
    throw "The registered profile requires a single-core CPU affinity"
}

function Assert-CleanPinnedCheckout {
    param([string]$Path, [string]$ExpectedCommit, [string]$Name)
    $ActualCommit = (& $GitPath -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $ExpectedCommit) {
        throw "$Name commit mismatch: expected $ExpectedCommit, found $ActualCommit"
    }
    $TrackedChanges = & $GitPath -C $Path status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $TrackedChanges) {
        throw "$Name has tracked local changes; refusing stock reproduction"
    }
}

Assert-CleanPinnedCheckout -Path $EPyMARLPath `
    -ExpectedCommit ([string]$Manifest.source_pins.epymarl.commit) -Name "EPyMARL"
Assert-CleanPinnedCheckout -Path $SMAClitePath `
    -ExpectedCommit ([string]$Manifest.source_pins.smaclite.commit) -Name "SMAClite"
Assert-CleanPinnedCheckout -Path $RVO2Path `
    -ExpectedCommit ([string]$Manifest.source_pins.smaclite_python_rvo2.commit) -Name "SMAClite-Python-RVO2"

foreach ($HashProperty in $Manifest.source_pins.source_sha256.PSObject.Properties) {
    $SourcePath = Join-Path $EPyMARLPath $HashProperty.Name
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "Pinned source file missing: $SourcePath"
    }
    $ActualHash = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne ([string]$HashProperty.Value).ToLowerInvariant()) {
        throw "Pinned source hash mismatch: $($HashProperty.Name)"
    }
}

$RegisteredBuildPatch = Join-Path $ProjectRoot ([string]$Manifest.source_pins.smaclite_python_rvo2.windows_build_patch)
if (-not (Test-Path -LiteralPath $RegisteredBuildPatch -PathType Leaf)) {
    throw "Registered RVO2 Windows build patch is missing: $RegisteredBuildPatch"
}
$BuildPatchHash = (Get-FileHash -LiteralPath $RegisteredBuildPatch -Algorithm SHA256).Hash.ToLowerInvariant()
if ($BuildPatchHash -ne ([string]$Manifest.source_pins.smaclite_python_rvo2.windows_build_patch_sha256).ToLowerInvariant()) {
    throw "Registered RVO2 Windows build patch hash mismatch"
}
$SiteCustomiseHash = (Get-FileHash -LiteralPath $WindowsSiteCustomise -Algorithm SHA256).Hash.ToLowerInvariant()
if ($SiteCustomiseHash -ne ([string]$Manifest.source_pins.windows_compatibility_overlay.sha256).ToLowerInvariant()) {
    throw "Windows compatibility overlay hash mismatch"
}
$RuntimeCpuProbeHash = (Get-FileHash -LiteralPath $RuntimeCpuProbe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($RuntimeCpuProbeHash -ne ([string]$Manifest.source_pins.runtime_cpu_probe.sha256).ToLowerInvariant()) {
    throw "Runtime CPU probe hash mismatch"
}
$SacredCaptureMode = [string]$Manifest.source_pins.windows_compatibility_overlay.sacred_capture_mode
if ($SacredCaptureMode -ne "sys") {
    throw "The registered Windows launcher requires Sacred capture mode 'sys'"
}

$Experiments = @()
if ($Suite -eq "primary") {
    foreach ($Entry in $Manifest.suites.primary) {
        $Experiments += [pscustomobject]@{
            algorithm = [string]$Entry.algorithm
            map = [string]$Entry.map
            paper_reference = $Entry.paper_reference
        }
    }
} else {
    foreach ($AlgName in $Manifest.suites.cross_algorithm_matrix.algorithms) {
        foreach ($Scenario in $Manifest.suites.cross_algorithm_matrix.maps) {
            $Reference = $null
            foreach ($Entry in $Manifest.suites.primary) {
                if ($Entry.algorithm -eq $AlgName -and $Entry.map -eq $Scenario) {
                    $Reference = $Entry.paper_reference
                }
            }
            $Experiments += [pscustomobject]@{
                algorithm = [string]$AlgName
                map = [string]$Scenario
                paper_reference = $Reference
            }
        }
    }
}
if ($Algorithm -ne "all") {
    $Experiments = @($Experiments | Where-Object { $_.algorithm -eq $Algorithm })
}
if (-not [string]::IsNullOrWhiteSpace($MapName)) {
    $Experiments = @($Experiments | Where-Object { $_.map -ieq $MapName })
}
if ($Experiments.Count -eq 0) {
    throw "The requested algorithm/map filters select no experiment in suite '$Suite'"
}

$RegisteredSeeds = @($ProfileConfig.seeds | ForEach-Object { [int]$_ })
$SelectedSeeds = if ($Seeds.Count -gt 0) { @($Seeds) } else { $RegisteredSeeds }
if ($SelectedSeeds.Count -eq 0) {
    throw "At least one seed is required"
}
$NonRegisteredSeeds = @($SelectedSeeds | Where-Object { $_ -notin $RegisteredSeeds })

$EffectiveTrainingSteps = if ($TrainingSteps -gt 0) { $TrainingSteps } else { [int]$ProfileConfig.training_steps }
$EffectiveTestEpisodes = if ($TestEpisodes -gt 0) { $TestEpisodes } else { [int]$ProfileConfig.test_episodes }
$EffectiveTestInterval = if ($TestInterval -gt 0) { $TestInterval } else { [int]$ProfileConfig.test_interval }
$EffectiveTimeLimit = if ($TimeLimit -gt 0) { $TimeLimit } else { [int]$ProfileConfig.time_limit }
$EffectiveSaveInterval = if ($SaveModelInterval -gt 0) { $SaveModelInterval } else { [int]$ProfileConfig.save_model_interval }
$EffectiveUseCuda = [bool]$ProfileConfig.use_cuda
$EffectiveUseCppRvo2 = [bool]$ProfileConfig.use_cpp_rvo2
$CpuThreadsPerRun = if ($null -ne $ProfileConfig.cpu_threads_per_run) {
    [int]$ProfileConfig.cpu_threads_per_run
} else {
    1
}
if ($UseCppRvo2 -and $UseNumpyRvo2) {
    throw "-UseCppRvo2 and -UseNumpyRvo2 are mutually exclusive"
}
if ($Cpu) { $EffectiveUseCuda = $false }
if ($UseCppRvo2) { $EffectiveUseCppRvo2 = $true }
if ($UseNumpyRvo2) { $EffectiveUseCppRvo2 = $false }

foreach ($Positive in @($EffectiveTrainingSteps, $EffectiveTestEpisodes, $EffectiveTestInterval, $EffectiveTimeLimit, $EffectiveSaveInterval)) {
    if ($Positive -lt 1) { throw "Training, evaluation, horizon and save values must be positive" }
}
if ($CpuThreadsPerRun -lt 1) { throw "cpu_threads_per_run must be positive" }

$OverrideReasons = @()
if ($TrainingSteps -gt 0 -and $EffectiveTrainingSteps -ne [int]$ProfileConfig.training_steps) { $OverrideReasons += "training_steps" }
if ($TestEpisodes -gt 0 -and $EffectiveTestEpisodes -ne [int]$ProfileConfig.test_episodes) { $OverrideReasons += "test_episodes" }
if ($TestInterval -gt 0 -and $EffectiveTestInterval -ne [int]$ProfileConfig.test_interval) { $OverrideReasons += "test_interval" }
if ($TimeLimit -gt 0 -and $EffectiveTimeLimit -ne [int]$ProfileConfig.time_limit) { $OverrideReasons += "time_limit" }
if ($SaveModelInterval -gt 0 -and $EffectiveSaveInterval -ne [int]$ProfileConfig.save_model_interval) { $OverrideReasons += "save_model_interval" }
if ($Cpu -and [bool]$ProfileConfig.use_cuda) { $OverrideReasons += "device" }
if ($UseCppRvo2 -and -not [bool]$ProfileConfig.use_cpp_rvo2) { $OverrideReasons += "collision_backend" }
if ($UseNumpyRvo2 -and [bool]$ProfileConfig.use_cpp_rvo2) { $OverrideReasons += "collision_backend" }
if ($NonRegisteredSeeds.Count -gt 0) { $OverrideReasons += "unregistered_seed" }
if ($Resume -or -not [string]::IsNullOrWhiteSpace($ResumeFrom)) { $OverrideReasons += "warm_resume" }
$OverrideReasons = @($OverrideReasons | Sort-Object -Unique)
if ($OverrideReasons.Count -gt 0 -and -not $AllowProtocolOverride) {
    throw "Runtime changes [$($OverrideReasons -join ', ')] require -AllowProtocolOverride and cannot count as registered reproduction"
}
$Compliance = if ($OverrideReasons.Count -eq 0) { "registered" } else { "override_non_reproduction" }

# Bind every registered run to the exact committed Open-SCORE orchestration
# source.  Runtime artifacts live under ignored output directories, so they do
# not make an otherwise clean worktree dirty.  Development smokes can still be
# launched from a dirty tree, but only under the explicit non-reproduction
# override label.
$RepositoryRoot = (& $GitPath -C $ProjectRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    throw "Could not resolve the Open-SCORE Git repository root"
}
$OpenScoreCommit = (& $GitPath -C $RepositoryRoot rev-parse HEAD).Trim()
$OpenScoreBranch = (& $GitPath -C $RepositoryRoot branch --show-current).Trim()
$OpenScoreStatus = @(& $GitPath -C $RepositoryRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0 -or $OpenScoreCommit -notmatch '^[0-9a-fA-F]{40}$') {
    throw "Could not audit the Open-SCORE Git revision"
}
$OpenScoreDirty = $OpenScoreStatus.Count -gt 0
if ($Compliance -eq "registered" -and $OpenScoreDirty) {
    throw "Registered stock reproduction requires a clean committed Open-SCORE worktree"
}
$RepositoryUri = [Uri](([string]$RepositoryRoot).TrimEnd([char[]]'\/') + [IO.Path]::DirectorySeparatorChar)
$ManifestUri = [Uri]$ManifestPath
$ManifestRepositoryRelativePath = [Uri]::UnescapeDataString(
    $RepositoryUri.MakeRelativeUri($ManifestUri).ToString()
).Replace('\', '/')
$ManifestTracked = $false
if (-not $ManifestRepositoryRelativePath.StartsWith('../')) {
    & $GitPath -C $RepositoryRoot ls-files --error-unmatch -- $ManifestRepositoryRelativePath 2>$null | Out-Null
    $ManifestTracked = $LASTEXITCODE -eq 0
}
$ManifestGitBlobObjectId = $null
if ($ManifestTracked) {
    $WorkingBlob = (& $GitPath -C $RepositoryRoot hash-object -- $ManifestPath).Trim()
    $CommittedBlob = (& $GitPath -C $RepositoryRoot rev-parse "$OpenScoreCommit`:$ManifestRepositoryRelativePath").Trim()
    if ($LASTEXITCODE -eq 0 -and $WorkingBlob -eq $CommittedBlob) {
        $ManifestGitBlobObjectId = $CommittedBlob
    }
}
if ($Compliance -eq "registered" -and (-not $ManifestTracked -or $null -eq $ManifestGitBlobObjectId)) {
    throw "Registered stock reproduction requires a tracked manifest identical to its HEAD blob"
}
$ManifestSha256 = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$LauncherSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$OpenScoreProvenance = [ordered]@{
    repository_root = [string]$RepositoryRoot
    commit = [string]$OpenScoreCommit
    branch = [string]$OpenScoreBranch
    dirty = [bool]$OpenScoreDirty
    status_porcelain = @($OpenScoreStatus)
}
$ManifestProvenance = [ordered]@{
    path = $ManifestPath
    repository_relative_path = $ManifestRepositoryRelativePath
    tracked = $ManifestTracked
    git_blob_object_id = $ManifestGitBlobObjectId
    sha256 = $ManifestSha256
    outer_commit = $OpenScoreCommit
}

$RuntimeRvo2 = [ordered]@{
    backend = if ($EffectiveUseCppRvo2) { "cpp" } else { "numpy" }
    source_commit = [string]$Manifest.source_pins.smaclite_python_rvo2.commit
    source_checkout_tracked_clean = $true
    binary_path = $null
    binary_filename = $null
    binary_sha256 = $null
}
if ($EffectiveUseCppRvo2) {
    $RvoProbe = "import hashlib,json,pathlib,rvo2; p=pathlib.Path(rvo2.__file__).resolve(); print(json.dumps({'path':str(p),'filename':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
    $RvoProbeJson = (& $PythonPath -c $RvoProbe)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($RvoProbeJson)) {
        throw "The selected profile requires SMAClite_plus, but the rvo2 extension could not be imported"
    }
    try {
        $RvoProbeResult = $RvoProbeJson | ConvertFrom-Json
    } catch {
        throw "Could not parse the installed rvo2 binary audit: $RvoProbeJson"
    }
    if ([string]$RvoProbeResult.filename -ne [string]$Manifest.source_pins.smaclite_python_rvo2.binary_filename) {
        throw "Installed rvo2 binary filename mismatch: $($RvoProbeResult.filename)"
    }
    if ([string]$RvoProbeResult.sha256 -ne [string]$Manifest.source_pins.smaclite_python_rvo2.binary_sha256) {
        throw "Installed rvo2 binary SHA256 mismatch: $($RvoProbeResult.sha256)"
    }
    $RuntimeRvo2.binary_path = [string]$RvoProbeResult.path
    $RuntimeRvo2.binary_filename = [string]$RvoProbeResult.filename
    $RuntimeRvo2.binary_sha256 = [string]$RvoProbeResult.sha256
}

if (-not [string]::IsNullOrWhiteSpace($ResumeFrom) -and (($Experiments.Count * $SelectedSeeds.Count) -ne 1)) {
    throw "-ResumeFrom requires exactly one selected algorithm/map/seed run"
}
if ($LoadStep -lt 0) { throw "LoadStep must be non-negative" }

function Convert-ToSacredValue {
    param($Value)
    if ($Value -is [bool]) {
        if ($Value) { return "True" } else { return "False" }
    }
    if ($Value -is [double] -or $Value -is [single] -or $Value -is [decimal]) {
        return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return [string]$Value
}

function Find-LatestCheckpointRoot {
    param([string]$AlgorithmName, [string]$Scenario, [int]$Seed)
    $ModelsRoot = Join-Path $OutputRoot "artifacts\models"
    if (-not (Test-Path -LiteralPath $ModelsRoot -PathType Container)) { return $null }
    $Prefix = "${AlgorithmName}_seed${Seed}_${Scenario}_"
    $Candidates = @(Get-ChildItem -LiteralPath $ModelsRoot -Directory |
        Where-Object { $_.Name.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase) } |
        Sort-Object LastWriteTimeUtc -Descending)
    if ($Candidates.Count -eq 0) { return $null }
    return $Candidates[0].FullName
}

function Find-LatestSacredRun {
    param([string]$AlgorithmName, [string]$Scenario, [int]$Seed, [DateTime]$StartedUtc)
    $SacredRoot = Join-Path $EPyMARLPath "results\sacred\$AlgorithmName\$Scenario"
    if (-not (Test-Path -LiteralPath $SacredRoot -PathType Container)) { return $null }
    $Candidates = @(Get-ChildItem -LiteralPath $SacredRoot -Directory |
        Where-Object {
            $_.Name -match '^\d+$' -and
            $_.LastWriteTimeUtc -ge $StartedUtc.AddMinutes(-1)
        } |
        Sort-Object LastWriteTimeUtc -Descending)
    foreach ($Candidate in $Candidates) {
        $ConfigPath = Join-Path $Candidate.FullName "config.json"
        if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { continue }
        try {
            $CandidateConfig = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
            if ([int]$CandidateConfig.seed -eq $Seed -and
                [string]$CandidateConfig.label -eq [string]$Manifest.protocol_id -and
                [int]$CandidateConfig.t_max -eq $EffectiveTrainingSteps) {
                return $Candidate.FullName
            }
        } catch {
            continue
        }
    }
    return $null
}

function Get-CheckpointRoots {
    param([string]$AlgorithmName, [string]$Scenario, [int]$Seed)
    $ModelsRoot = Join-Path $OutputRoot "artifacts\models"
    if (-not (Test-Path -LiteralPath $ModelsRoot -PathType Container)) { return @() }
    $Prefix = "${AlgorithmName}_seed${Seed}_${Scenario}_"
    return @(Get-ChildItem -LiteralPath $ModelsRoot -Directory |
        Where-Object { $_.Name.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase) } |
        ForEach-Object { $_.FullName })
}

function Get-SacredRunIds {
    param([string]$AlgorithmName, [string]$Scenario)
    $Root = Join-Path $EPyMARLPath "results\sacred\$AlgorithmName\$Scenario"
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $Root -Directory |
        Where-Object { $_.Name -match '^\d+$' } |
        ForEach-Object { $_.Name })
}

function Get-CheckpointEvidence {
    param([string]$Root, [string]$AlgorithmName, [int64]$MinimumStep)
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw "Missing checkpoint root: $Root" }
    $Steps = @(Get-ChildItem -LiteralPath $Root -Directory |
        Where-Object { $_.Name -match '^\d+$' } |
        ForEach-Object { [int64]$_.Name } | Sort-Object)
    if ($Steps.Count -eq 0 -or $Steps[-1] -lt $MinimumStep) {
        throw "Checkpoint is below terminal threshold $MinimumStep under $Root"
    }
    $LatestStep = $Steps[-1]
    $LatestDirectory = Join-Path $Root ([string]$LatestStep)
    $Required = if ($AlgorithmName -eq 'mappo') {
        @('agent.th', 'critic.th', 'agent_opt.th', 'critic_opt.th')
    } else {
        @('agent.th', 'mixer.th', 'opt.th')
    }
    $Hashes = [ordered]@{}
    foreach ($Name in $Required) {
        $File = Join-Path $LatestDirectory $Name
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) {
            throw "Required checkpoint file missing: $File"
        }
    }
    foreach ($File in @(Get-ChildItem -LiteralPath $LatestDirectory -File -Recurse | Sort-Object FullName)) {
        $Relative = $File.FullName.Substring($LatestDirectory.Length).TrimStart([char[]]'\/').Replace('\', '/')
        $Hashes[$Relative] = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $CombinedText = (($Hashes.Keys | ForEach-Object { "$_`0$($Hashes[$_])`n" }) -join '')
    $CombinedBytes = [Text.Encoding]::UTF8.GetBytes($CombinedText)
    $Sha = [Security.Cryptography.SHA256]::Create()
    try { $CombinedHash = ([BitConverter]::ToString($Sha.ComputeHash($CombinedBytes))).Replace('-', '').ToLowerInvariant() }
    finally { $Sha.Dispose() }
    return [ordered]@{
        checkpoint_root = $Root
        latest_step = $LatestStep
        latest_directory = $LatestDirectory
        file_sha256 = $Hashes
        combined_sha256 = $CombinedHash
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$RecordsRoot = Join-Path $OutputRoot "run_records"
$ArtifactsRoot = Join-Path $OutputRoot "artifacts"
New-Item -ItemType Directory -Force -Path $RecordsRoot, $ArtifactsRoot | Out-Null

$env:GIT_PYTHON_GIT_EXECUTABLE = $GitPath
$env:Path = (Split-Path -Parent $GitPath) + ";" + $env:Path
$env:PYGAME_HIDE_SUPPORT_PROMPT = "1"
$env:PYTHONWARNINGS = "ignore::UserWarning"
$PythonPathEntries = @($WindowsSiteDirectory, (Join-Path $EPyMARLPath "src"))
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $PythonPathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $PythonPathEntries -join ";"
$env:OPEN_SCORE_EPYMARL_WINDOWS_SAFE = "1"
$env:OPEN_SCORE_SMACLITE_CHECKOUT = $SMAClitePath
$env:OMP_NUM_THREADS = [string]$CpuThreadsPerRun
$env:MKL_NUM_THREADS = [string]$CpuThreadsPerRun
$env:OPENBLAS_NUM_THREADS = [string]$CpuThreadsPerRun
$env:NUMEXPR_NUM_THREADS = [string]$CpuThreadsPerRun
$env:VECLIB_MAXIMUM_THREADS = [string]$CpuThreadsPerRun
$RuntimeCpuProbeJson = & $PythonPath $RuntimeCpuProbe
if ($LASTEXITCODE -ne 0) {
    throw "Runtime CPU probe failed"
}
$RuntimeCpuProbeResult = $RuntimeCpuProbeJson | ConvertFrom-Json
if ([int]$RuntimeCpuProbeResult.torch_num_threads -ne $CpuThreadsPerRun) {
    throw "Child Python torch thread count mismatch: expected $CpuThreadsPerRun, found $($RuntimeCpuProbeResult.torch_num_threads)"
}
if ($null -ne $AppliedCpuAffinityMask -and
    [Int64]$RuntimeCpuProbeResult.process_affinity_mask -ne $AppliedCpuAffinityMask) {
    throw "Child Python affinity mismatch: expected $AppliedCpuAffinityMask, found $($RuntimeCpuProbeResult.process_affinity_mask)"
}
if ($RuntimeCpuProbeResult.smaclite_imported_from_expected_checkout -ne $true) {
    throw "Imported smaclite does not resolve inside the pinned SMAClite checkout"
}
if ([int]$RuntimeCpuProbeResult.stock_scenario_file_count -ne [int]$Manifest.source_pins.smaclite.stock_scenario_file_count -or
    [string]$RuntimeCpuProbeResult.stock_scenario_combined_sha256 -ne [string]$Manifest.source_pins.smaclite.stock_scenario_combined_sha256) {
    throw "Runtime SMAClite stock scenario fingerprint mismatch"
}
$Failures = @()

foreach ($Experiment in $Experiments) {
    foreach ($Seed in $SelectedSeeds) {
        $AlgName = [string]$Experiment.algorithm
        $Scenario = [string]$Experiment.map
        if ($AlgName -eq "mappo" -and ($EffectiveTestEpisodes % 10) -ne 0) {
            throw "EPyMARL-MAPPO batch_size_run=10 requires TestEpisodes divisible by 10"
        }
        $CheckpointRoot = $null
        if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
            $CheckpointRoot = [System.IO.Path]::GetFullPath($ResumeFrom)
        } elseif ($Resume) {
            $CheckpointRoot = Find-LatestCheckpointRoot -AlgorithmName $AlgName -Scenario $Scenario -Seed $Seed
            if ($null -eq $CheckpointRoot) {
                throw "No checkpoint root found for $AlgName/$Scenario/seed=$Seed under $ArtifactsRoot"
            }
        }
        if ($null -ne $CheckpointRoot) {
            if (-not (Test-Path -LiteralPath $CheckpointRoot -PathType Container)) {
                throw "Checkpoint root not found: $CheckpointRoot"
            }
            $ExpectedPrefix = "${AlgName}_seed${Seed}_${Scenario}_"
            if (-not (Split-Path -Leaf $CheckpointRoot).StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Checkpoint path name does not match $AlgName/$Scenario/seed=${Seed}: $CheckpointRoot"
            }
        }

        $CudaValue = if ($EffectiveUseCuda) { "True" } else { "False" }
        $CppValue = if ($EffectiveUseCppRvo2) { "True" } else { "False" }
        $SacredArguments = @(
            $EPyMARLMain,
            "--config=$AlgName",
            "--env-config=smaclite",
            "--capture=$SacredCaptureMode",
            "with",
            "env_args.map_name=$Scenario",
            "env_args.time_limit=$EffectiveTimeLimit",
            "env_args.use_cpp_rvo2=$CppValue",
            "seed=$Seed",
            "t_max=$EffectiveTrainingSteps",
            "test_interval=$EffectiveTestInterval",
            "test_nepisode=$EffectiveTestEpisodes",
            "log_interval=$([int]$ProfileConfig.log_interval)",
            "runner_log_interval=$([int]$ProfileConfig.runner_log_interval)",
            "learner_log_interval=$([int]$ProfileConfig.learner_log_interval)",
            "use_cuda=$CudaValue",
            "save_model=True",
            "save_model_interval=$EffectiveSaveInterval",
            "local_results_path=$($ArtifactsRoot.Replace('\', '/'))",
            "label=$($Manifest.protocol_id)"
        )
        $AlgorithmOverrides = $ProfileConfig.algorithm_overrides.PSObject.Properties[$AlgName].Value
        foreach ($Property in $AlgorithmOverrides.PSObject.Properties) {
            $SacredArguments += "$($Property.Name)=$(Convert-ToSacredValue $Property.Value)"
        }
        if ($null -ne $CheckpointRoot) {
            $SacredArguments += "checkpoint_path=$($CheckpointRoot.Replace('\', '/'))"
            $SacredArguments += "load_step=$LoadStep"
        }

        $CheckpointRootsBefore = @(Get-CheckpointRoots -AlgorithmName $AlgName -Scenario $Scenario -Seed $Seed)
        $SacredIdsBefore = @(Get-SacredRunIds -AlgorithmName $AlgName -Scenario $Scenario)
        $Now = [DateTime]::UtcNow
        $Token = $Now.ToString("yyyyMMddTHHmmssfffZ") + "_" + ([Guid]::NewGuid().ToString("N").Substring(0, 8))
        $RecordDirectory = Join-Path $RecordsRoot "$Profile\$AlgName\$Scenario\seed_$Seed\$Token"
        New-Item -ItemType Directory -Force -Path $RecordDirectory | Out-Null
        $RecordPath = Join-Path $RecordDirectory "run_record.json"
        $Record = [ordered]@{
            schema_version = 1
            protocol_id = [string]$Manifest.protocol_id
            manifest_path = $ManifestPath
            manifest_provenance = $ManifestProvenance
            launcher_sha256 = $LauncherSha256
            profile = $Profile
            suite = $Suite
            protocol_compliance = $Compliance
            override_reasons = @($OverrideReasons)
            algorithm_label = "EPyMARL-$($AlgName.ToUpperInvariant())"
            algorithm = $AlgName
            map = $Scenario
            seed = [int]$Seed
            training_steps = $EffectiveTrainingSteps
            time_limit = $EffectiveTimeLimit
            test_interval = $EffectiveTestInterval
            test_episodes = $EffectiveTestEpisodes
            use_cuda = $EffectiveUseCuda
            cpu_threads_per_run = $CpuThreadsPerRun
            require_single_core_affinity = $RequireSingleCoreAffinity
            cpu_affinity_mask = $AppliedCpuAffinityMask
            cpu_affinity_logical_index = $AppliedCpuLogicalIndex
            thread_environment = [ordered]@{
                OMP_NUM_THREADS = $env:OMP_NUM_THREADS
                MKL_NUM_THREADS = $env:MKL_NUM_THREADS
                OPENBLAS_NUM_THREADS = $env:OPENBLAS_NUM_THREADS
                NUMEXPR_NUM_THREADS = $env:NUMEXPR_NUM_THREADS
                VECLIB_MAXIMUM_THREADS = $env:VECLIB_MAXIMUM_THREADS
            }
            runtime_cpu_probe = [ordered]@{
                path = $RuntimeCpuProbe
                sha256 = $RuntimeCpuProbeHash
                scope = [string]$Manifest.source_pins.runtime_cpu_probe.scope
                result = $RuntimeCpuProbeResult
            }
            use_cpp_rvo2 = $EffectiveUseCppRvo2
            paper_reference = $Experiment.paper_reference
            source_pins = $Manifest.source_pins
            open_score_git = $OpenScoreProvenance
            source_checkout_tracked_clean = $true
            runtime_rvo2 = $RuntimeRvo2
            windows_compatibility_overlay = [ordered]@{
                path = $WindowsSiteCustomise
                sha256 = $SiteCustomiseHash
                scope = [string]$Manifest.source_pins.windows_compatibility_overlay.scope
                sacred_capture_mode = $SacredCaptureMode
            }
            algorithm_overrides = $AlgorithmOverrides
            command = @($PythonPath) + @($SacredArguments)
            checkpoint_path = $CheckpointRoot
            load_step = $LoadStep
            resume_limitations = @($Manifest.resume_policy.limitations)
            started_at = $Now.ToString("o")
            finished_at = $null
            exit_code = $null
            status = if ($DryRun) { "dry_run" } else { "running" }
            pre_run_artifact_snapshot = [ordered]@{
                checkpoint_roots = @($CheckpointRootsBefore)
                sacred_run_ids = @($SacredIdsBefore)
            }
        }
        $Record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $RecordPath -Encoding UTF8
        Write-Output "[$($Record.status)] $($Record.algorithm_label) map=$Scenario seed=$Seed compliance=$Compliance"
        Write-Output "run record: $RecordPath"

        if (-not $DryRun) {
            $ExitCode = 1
            $PostAuditFailures = @()
            try {
                Push-Location $EPyMARLPath
                try {
                    & $PythonPath @SacredArguments
                    $ExitCode = $LASTEXITCODE
                } finally {
                    Pop-Location
                }
            } catch {
                $Record.error = $_.Exception.Message
                $ExitCode = 1
            }
            $Record.exit_code = $ExitCode
            $Record.finished_at = [DateTime]::UtcNow.ToString("o")
            if ($ExitCode -eq 0) {
                try {
                    $NewCheckpointRoots = @(Get-CheckpointRoots -AlgorithmName $AlgName -Scenario $Scenario -Seed $Seed |
                        Where-Object { $_ -notin $CheckpointRootsBefore })
                    if ($NewCheckpointRoots.Count -ne 1) {
                        throw "Expected exactly one new checkpoint root, found $($NewCheckpointRoots.Count)"
                    }
                    $MinimumTerminalStep = [int64][Math]::Ceiling(0.975 * $EffectiveTrainingSteps)
                    $CheckpointEvidence = Get-CheckpointEvidence -Root $NewCheckpointRoots[0] `
                        -AlgorithmName $AlgName -MinimumStep $MinimumTerminalStep
                    $Record.produced_checkpoint_root = $NewCheckpointRoots[0]
                    $Record.latest_checkpoint_step = $CheckpointEvidence.latest_step
                    $Record.produced_checkpoint = $CheckpointEvidence
                } catch {
                    $PostAuditFailures += "checkpoint:$($_.Exception.Message)"
                }
                try {
                    $SacredRoot = Join-Path $EPyMARLPath "results\sacred\$AlgName\$Scenario"
                    $NewSacredCandidates = @()
                    foreach ($CandidateId in @(Get-SacredRunIds -AlgorithmName $AlgName -Scenario $Scenario |
                        Where-Object { $_ -notin $SacredIdsBefore })) {
                        $Candidate = Join-Path $SacredRoot $CandidateId
                        $ConfigPath = Join-Path $Candidate "config.json"
                        if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { continue }
                        $CandidateConfig = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
                        if ([int]$CandidateConfig.seed -eq $Seed -and
                            [string]$CandidateConfig.label -eq [string]$Manifest.protocol_id -and
                            [int]$CandidateConfig.t_max -eq $EffectiveTrainingSteps) {
                            $NewSacredCandidates += $Candidate
                        }
                    }
                    if ($NewSacredCandidates.Count -ne 1) {
                        throw "Expected exactly one new matching Sacred run, found $($NewSacredCandidates.Count)"
                    }
                    $Record.sacred_run_path = $NewSacredCandidates[0]
                } catch {
                    $PostAuditFailures += "sacred:$($_.Exception.Message)"
                }
            }
            try {
                Assert-CleanPinnedCheckout -Path $EPyMARLPath `
                    -ExpectedCommit ([string]$Manifest.source_pins.epymarl.commit) -Name "EPyMARL"
                Assert-CleanPinnedCheckout -Path $SMAClitePath `
                    -ExpectedCommit ([string]$Manifest.source_pins.smaclite.commit) -Name "SMAClite"
                Assert-CleanPinnedCheckout -Path $RVO2Path `
                    -ExpectedCommit ([string]$Manifest.source_pins.smaclite_python_rvo2.commit) -Name "SMAClite-Python-RVO2"
                $PostOuterCommit = (& $GitPath -C $RepositoryRoot rev-parse HEAD).Trim()
                $PostOuterStatus = @(& $GitPath -C $RepositoryRoot status --porcelain=v1 --untracked-files=all)
                if ($PostOuterCommit -ne $OpenScoreCommit -or ($Compliance -eq 'registered' -and $PostOuterStatus.Count -gt 0)) {
                    throw "Open-SCORE HEAD/clean state changed during the run"
                }
            } catch {
                $PostAuditFailures += "source:$($_.Exception.Message)"
                $PostOuterCommit = (& $GitPath -C $RepositoryRoot rev-parse HEAD).Trim()
                $PostOuterStatus = @(& $GitPath -C $RepositoryRoot status --porcelain=v1 --untracked-files=all)
            }
            $Record.post_run_source_audit = [ordered]@{
                passed = ($PostAuditFailures.Count -eq 0)
                outer_commit = $PostOuterCommit
                outer_dirty = ($PostOuterStatus.Count -gt 0)
                outer_status_porcelain = @($PostOuterStatus)
                upstream = [ordered]@{
                    epymarl = [ordered]@{ commit = [string]$Manifest.source_pins.epymarl.commit; tracked_clean = ($PostAuditFailures.Count -eq 0) }
                    smaclite = [ordered]@{ commit = [string]$Manifest.source_pins.smaclite.commit; tracked_clean = ($PostAuditFailures.Count -eq 0) }
                    rvo2 = [ordered]@{ commit = [string]$Manifest.source_pins.smaclite_python_rvo2.commit; tracked_clean = ($PostAuditFailures.Count -eq 0) }
                }
                failures = @($PostAuditFailures)
            }
            $Record.status = if ($ExitCode -eq 0 -and $PostAuditFailures.Count -eq 0) { "completed" } else { "failed" }
            if ($PostAuditFailures.Count -gt 0) { $Record.error = "post_audit_failed: $($PostAuditFailures -join '; ')" }
            $Record | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $RecordPath -Encoding UTF8
            if ($Record.status -ne 'completed') {
                $Failures += "$AlgName/$Scenario/seed=$Seed"
                if (-not $ContinueOnError) {
                    throw "Stock reproduction run failed: $($Failures[-1]); see $RecordPath"
                }
            }
        }
    }
}

if ($Failures.Count -gt 0) {
    throw "Completed with failed runs: $($Failures -join ', ')"
}
Write-Output "Stock reproduction launcher completed. No pinned upstream tracked file was modified."
