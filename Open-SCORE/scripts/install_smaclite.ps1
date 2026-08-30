<#
.SYNOPSIS
Install the audited SMAClite v2.0.0 source into the torch310 environment.

.DESCRIPTION
SMAClite v2.0.0's ordinary wheel omits map/unit subdirectories.  This script
therefore clones the exact audited commit and performs an editable install,
which is also appropriate for upstream's source-tree workflow.  It never edits
the upstream checkout.
#>

param(
    [string]$PythonPath = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$GitPath = "D:\Software\Git\cmd\git.exe",
    [string]$CheckoutPath = ""
)

$ErrorActionPreference = "Stop"
$ExpectedCommit = "e936d9dbf4f85551d6fd445a6c1150867bc79c55"
$Repository = "https://github.com/uoe-agents/smaclite.git"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $GitPath -PathType Leaf)) {
    $GitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $GitCommand) {
        throw "Git not found. Pass -GitPath with the absolute git.exe path."
    }
    $GitPath = $GitCommand.Source
}
if ([string]::IsNullOrWhiteSpace($CheckoutPath)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $CheckoutPath = Join-Path $ProjectRoot "upstream\external\smaclite-v2.0.0"
} else {
    $CheckoutPath = [System.IO.Path]::GetFullPath($CheckoutPath)
}
$CheckoutParent = Split-Path -Parent $CheckoutPath
New-Item -ItemType Directory -Force -Path $CheckoutParent | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $CheckoutPath ".git") -PathType Container)) {
    if (Test-Path -LiteralPath $CheckoutPath) {
        throw "Checkout path exists but is not a Git checkout: $CheckoutPath"
    }
    & $GitPath clone --branch v2.0.0 --depth 1 $Repository $CheckoutPath
    if ($LASTEXITCODE -ne 0) { throw "SMAClite clone failed" }
}

$ActualCommit = (& $GitPath -C $CheckoutPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $ExpectedCommit) {
    throw "Expected SMAClite $ExpectedCommit, found $ActualCommit. Preserve the checkout and choose a clean path."
}
$Dirty = & $GitPath -C $CheckoutPath status --porcelain
if ($Dirty) {
    throw "Pinned SMAClite checkout has local modifications; refusing to install it."
}

# Freeze the compatibility-tested Windows dependencies without replacing the
# already working NumPy/PyTorch stack.
& $PythonPath -m pip install --disable-pip-version-check `
    "numpy==1.26.4" `
    "gymnasium==1.2.2" `
    "Rtree==1.0.0" `
    "pygame==2.6.1" `
    "scikit-learn==1.7.2" `
    "setuptools==80.9.0"
if ($LASTEXITCODE -ne 0) { throw "SMAClite dependency installation failed" }

& $PythonPath -m pip install --disable-pip-version-check --no-deps --editable $CheckoutPath
if ($LASTEXITCODE -ne 0) { throw "SMAClite editable installation failed" }

& $PythonPath -c "import gymnasium as gym, smaclite; e=gym.make('smaclite/3s5z-v0'); o,i=e.reset(seed=7); print({'module': smaclite.__file__, 'agents': len(o), 'obs': o[0].shape, 'info': i}); e.close()"
if ($LASTEXITCODE -ne 0) { throw "Installed SMAClite failed its stock smoke test" }

Write-Output "Installed audited SMAClite commit $ActualCommit from $CheckoutPath"
