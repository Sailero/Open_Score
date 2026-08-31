<#
.SYNOPSIS
Build and install SMAClite's pinned C++ RVO2 backend for torch310 on Windows.

.DESCRIPTION
The upstream extension is Linux-oriented. This script clones the exact audited
commit and applies the tracked patch that changes build flags and library paths
only. It does not alter RVO2 runtime/physics sources. CMake and the Visual C++
workload may be installed explicitly with -InstallBuildTools.
#>

param(
    [string]$PythonPath = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$GitPath = "D:\Software\Git\cmd\git.exe",
    [string]$CheckoutPath = "",
    [switch]$InstallBuildTools,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"
$ExpectedCommit = "a693b272e387bcf02a8f41ef295d7c3e1b19abb0"
$Repository = "https://github.com/uoe-agents/SMAClite-Python-RVO2.git"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PatchPath = Join-Path $ProjectRoot "upstream\patches\smaclite-python-rvo2-windows.patch"
$CMakeExe = "C:\Program Files\CMake\bin\cmake.exe"
$VsWhereExe = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$WingetExe = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\winget.exe"

if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "This installer is intentionally Windows-only."
}
foreach ($required in @($PythonPath, $GitPath, $PatchPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}

if ($InstallBuildTools) {
    if (-not (Test-Path -LiteralPath $WingetExe -PathType Leaf)) {
        throw "winget was not found: $WingetExe"
    }
    if (-not (Test-Path -LiteralPath $CMakeExe -PathType Leaf)) {
        & $WingetExe install --id Kitware.CMake --exact --source winget --silent `
            --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw "CMake installation failed" }
    }
    $HasVCTools = $false
    if (Test-Path -LiteralPath $VsWhereExe -PathType Leaf) {
        $VsPath = (& $VsWhereExe -latest -products Microsoft.VisualStudio.Product.BuildTools `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        $HasVCTools = -not [string]::IsNullOrWhiteSpace($VsPath)
    }
    if (-not $HasVCTools) {
        & $WingetExe install --id Microsoft.VisualStudio.2022.BuildTools --exact --source winget `
            --silent --accept-source-agreements --accept-package-agreements `
            --override "--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        if ($LASTEXITCODE -ne 0) { throw "Visual Studio C++ Build Tools installation failed" }
    }
}

if (-not (Test-Path -LiteralPath $CMakeExe -PathType Leaf)) {
    throw "CMake not found. Re-run with -InstallBuildTools."
}
if (-not (Test-Path -LiteralPath $VsWhereExe -PathType Leaf)) {
    throw "Visual Studio Build Tools not found. Re-run with -InstallBuildTools."
}
$VsPath = (& $VsWhereExe -latest -products Microsoft.VisualStudio.Product.BuildTools `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if ([string]::IsNullOrWhiteSpace($VsPath)) {
    throw "The Visual C++ x64 workload was not found. Re-run with -InstallBuildTools."
}

if ([string]::IsNullOrWhiteSpace($CheckoutPath)) {
    $CheckoutPath = Join-Path $ProjectRoot "upstream\external\SMAClite-Python-RVO2"
} else {
    $CheckoutPath = [System.IO.Path]::GetFullPath($CheckoutPath)
}
$CheckoutParent = Split-Path -Parent $CheckoutPath
New-Item -ItemType Directory -Force -Path $CheckoutParent | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $CheckoutPath ".git") -PathType Container)) {
    if (Test-Path -LiteralPath $CheckoutPath) {
        throw "Checkout path exists but is not a Git checkout: $CheckoutPath"
    }
    & $GitPath clone $Repository $CheckoutPath
    if ($LASTEXITCODE -ne 0) { throw "RVO2 clone failed" }
}
$ActualCommit = (& $GitPath -C $CheckoutPath rev-parse HEAD).Trim()
if ($ActualCommit -ne $ExpectedCommit) {
    throw "Expected RVO2 $ExpectedCommit, found $ActualCommit. Preserve it and choose a clean path."
}

& $GitPath -C $CheckoutPath apply --reverse --check $PatchPath 2>$null
$PatchAlreadyApplied = $LASTEXITCODE -eq 0
if (-not $PatchAlreadyApplied) {
    $Dirty = & $GitPath -C $CheckoutPath status --porcelain --untracked-files=no
    if ($Dirty) {
        throw "RVO2 checkout has unrelated tracked changes; refusing to patch it."
    }
    & $GitPath -C $CheckoutPath apply --check $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "Windows build patch does not apply cleanly" }
    & $GitPath -C $CheckoutPath apply $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "Windows build patch failed" }
}

try {
    & $PythonPath -m pip install --disable-pip-version-check "Cython==0.29.32"
    if ($LASTEXITCODE -ne 0) { throw "Cython installation failed" }
    $env:PATH = (Split-Path -Parent $CMakeExe) + ";" + $env:PATH
    $WheelPath = Join-Path $ProjectRoot "upstream\wheels\rvo2-cp310"
    New-Item -ItemType Directory -Force -Path $WheelPath | Out-Null
    Push-Location $CheckoutPath
    try {
        & $PythonPath -m pip wheel . --no-deps --no-build-isolation --wheel-dir $WheelPath
        if ($LASTEXITCODE -ne 0) { throw "RVO2 wheel build failed" }
    } finally {
        Pop-Location
    }
} finally {
    # The checkout is evidence as well as source. Restore it to the exact
    # upstream tree after producing the wheel; only the built binary persists.
    & $GitPath -C $CheckoutPath apply --reverse --check $PatchPath 2>$null
    if ($LASTEXITCODE -eq 0) {
        & $GitPath -C $CheckoutPath apply --reverse $PatchPath
        if ($LASTEXITCODE -ne 0) { throw "Could not restore the clean upstream RVO2 checkout" }
    }
}
$Wheel = Get-ChildItem -LiteralPath $WheelPath -Filter "pyrvo2-0.0.0-cp310-cp310-win_amd64.whl" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $Wheel) { throw "The expected CPython 3.10 x64 wheel was not produced" }

# Avoid needlessly replacing a loaded Windows .pyd. The comparison uses the
# extension bytes inside the newly built wheel, not package metadata alone.
& $PythonPath -c "import hashlib,pathlib,sys,zipfile; w=pathlib.Path(sys.argv[1]); import rvo2; installed=hashlib.sha256(pathlib.Path(rvo2.__file__).read_bytes()).digest(); z=zipfile.ZipFile(w); member=next(n for n in z.namelist() if n.endswith('.pyd')); expected=hashlib.sha256(z.read(member)).digest(); raise SystemExit(0 if installed == expected else 1)" $Wheel.FullName 2>$null
$InstalledWheelMatches = $LASTEXITCODE -eq 0
if ($ForceReinstall -or -not $InstalledWheelMatches) {
    & $PythonPath -m pip install --force-reinstall --no-deps $Wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "RVO2 wheel installation failed" }
} else {
    Write-Output "The installed RVO2 binary already matches the built wheel; reinstall skipped."
}

& $PythonPath -c "import gymnasium as gym, hashlib, rvo2, smaclite; p=rvo2.__file__; e=gym.make('smaclite/2s_vs_1sc-v0', use_cpp_rvo2=True); o,i=e.reset(seed=20260831); print({'rvo2_binary':p,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'agents':len(o)}); e.close()"
if ($LASTEXITCODE -ne 0) { throw "Installed C++ RVO2 backend failed its SMAClite smoke test" }

Write-Output "Installed SMAClite C++ RVO2 commit $ActualCommit using build-only patch $PatchPath"
