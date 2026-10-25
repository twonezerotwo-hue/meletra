<#
.SYNOPSIS
    DEVELOPER SETUP (Windows) - for authorized collaborators of the private source repository. NOT the way to install MELETRA.

.DESCRIPTION
    End users install MELETRA from the public download page or with the one-command installer (see the README); they never need this repository.
    This script prepares a development checkout: it delegates to setup_windows.ps1 (Python 3.11 virtual environment .venv with the pinned dependencies,
    input/ and outputs/ folders, GPU diagnostic) and then checks that the test runner works.

.PARAMETER InstallMissing
    Install a missing Python 3.11 / FFmpeg with winget (user scope) - passed on to setup_windows.ps1.
.PARAMETER SkipGpuTest
    Skip the GPU diagnostic (it downloads a ~75 MB test model once) - passed on to setup_windows.ps1.
.PARAMETER RunTests
    Run the full test suite at the end (about 15-20 minutes).
#>
[CmdletBinding()]
param(
    [switch]$InstallMissing,
    [switch]$SkipGpuTest,
    [switch]$RunTests
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root

Write-Host 'DEVELOPER SETUP - this is not the installation for users (see the README: "Install MELETRA").' -ForegroundColor Yellow
if (-not (Test-Path (Join-Path $root 'setup_windows.ps1'))) { throw 'setup_windows.ps1 not found: run this script from a checkout of the source repository.' }

$setupArgs = @()
if ($InstallMissing) { $setupArgs += '-InstallMissing' }
if ($SkipGpuTest) { $setupArgs += '-SkipGpuTest' }
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'setup_windows.ps1') @setupArgs
if ($LASTEXITCODE -ne 0) { throw "setup_windows.ps1 failed with exit code $LASTEXITCODE" }

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'The virtual environment .venv was not created.' }
& $python -c "import sys; print('Python', sys.version.split()[0])"
& $python -m pytest --version
if ($RunTests) {
    & $python -m pytest -q -p no:cacheprovider
    exit $LASTEXITCODE
}
Write-Host 'Done. Run the tests with:  .venv\Scripts\python.exe -m pytest -q' -ForegroundColor Green
