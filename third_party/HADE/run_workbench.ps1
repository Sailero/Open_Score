param([string]$Episode = '')
$ErrorActionPreference = 'Stop'
$HADPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $HADPython)) {
    throw 'Create .venv and install .[viewer] first; see README.md.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Episode) {
        & $HADPython -B -m had_env.workbench view $Episode
    } else {
        & $HADPython -B -m had_env.workbench live
    }
    if ($LASTEXITCODE -ne 0) { throw "HAD Workbench exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
