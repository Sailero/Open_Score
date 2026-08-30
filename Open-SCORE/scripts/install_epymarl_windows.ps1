<#
.SYNOPSIS
Install the audited EPyMARL runner used for stock SMAClite baselines.

.DESCRIPTION
Clones one exact EPyMARL commit under the ignored upstream/external folder,
installs the Windows/Python-3.10 compatibility overlay, and delegates the
SMAClite installation to install_smaclite.ps1.  The upstream checkout is never
patched.  In particular, PyYAML 6.0.3 is retained instead of the legacy
PyYAML 5.3.1 requirement, which has no reliable Python-3.10 Windows wheel.
#>

param(
    [string]$PythonPath = "D:\Software\Anaconda\envs\torch310\python.exe",
    [string]$GitPath = "D:\Software\Git\cmd\git.exe",
    [string]$CheckoutPath = ""
)

$ErrorActionPreference = "Stop"
$ExpectedCommit = "cbc38c09588064eab978501d0f12c2cf58fa7fc2"
$Repository = "https://github.com/uoe-agents/epymarl.git"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python interpreter not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $GitPath -PathType Leaf)) {
    throw "Git executable not found: $GitPath"
}
if ([string]::IsNullOrWhiteSpace($CheckoutPath)) {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $CheckoutPath = Join-Path $ProjectRoot "upstream\external\epymarl"
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
    if ($LASTEXITCODE -ne 0) { throw "EPyMARL clone failed" }
    & $GitPath -C $CheckoutPath checkout --detach $ExpectedCommit
    if ($LASTEXITCODE -ne 0) { throw "EPyMARL commit checkout failed" }
}

$ActualCommit = (& $GitPath -C $CheckoutPath rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $ExpectedCommit) {
    throw "Expected EPyMARL $ExpectedCommit, found $ActualCommit. Use a separate clean path."
}
$Dirty = & $GitPath -C $CheckoutPath status --porcelain --untracked-files=no
if ($Dirty) {
    throw "Pinned EPyMARL checkout has tracked local modifications; refusing to use it."
}

& (Join-Path $PSScriptRoot "install_smaclite.ps1") `
    -PythonPath $PythonPath -GitPath $GitPath
if ($LASTEXITCODE -ne 0) { throw "SMAClite setup failed" }

# Sacred 0.8.7 imports pkg_resources, removed from newer setuptools releases.
# These are the packages exercised by the stock QMIX/VDN/MAPPO smoke runs;
# optional SC2, VMAS and video environments are deliberately not installed.
& $PythonPath -m pip install --disable-pip-version-check `
    "setuptools==80.9.0" `
    "PyYAML==6.0.3" `
    "sacred==0.8.7" `
    "tensorboard-logger==0.1.0" `
    "probscale==0.2.5" `
    "scikit-video==1.1.11" `
    "snakeviz==2.2.2" `
    "portpicker==1.6.0" `
    "whichcraft==0.6.1" `
    "wrapt==1.17.3" `
    "mpyq==0.2.5"
if ($LASTEXITCODE -ne 0) { throw "EPyMARL compatibility dependency installation failed" }

$env:GIT_PYTHON_GIT_EXECUTABLE = $GitPath
$env:Path = (Split-Path -Parent $GitPath) + ";" + $env:Path
& $PythonPath -c "import gymnasium, numpy, sacred, smaclite, torch, yaml; print({'epymarl_commit': '$ActualCommit', 'python_ok': True, 'numpy': numpy.__version__, 'gymnasium': gymnasium.__version__, 'sacred': sacred.__version__, 'torch': torch.__version__, 'yaml': yaml.__version__})"
if ($LASTEXITCODE -ne 0) { throw "EPyMARL dependency import smoke failed" }

Write-Output "Installed the audited EPyMARL compatibility overlay for commit $ActualCommit"
