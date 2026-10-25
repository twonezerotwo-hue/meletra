<#
.SYNOPSIS
    Sets up medical-lecture-transcriber on Windows 11 (NVIDIA GPU, Python 3.11).

.DESCRIPTION
    Creates .venv, installs the pinned dependencies (including the CUDA 12.6 build of PyTorch),
    creates the input/ and outputs/ folders, validates the config and runs the GPU diagnostic.

    It never installs NVIDIA drivers or changes system-level settings. Python 3.11 and FFmpeg are
    only installed (per user, via winget) when you pass -InstallMissing.

.PARAMETER InstallMissing
    Install a missing Python 3.11 / FFmpeg with winget (user scope).
.PARAMETER WithWhisperX
    Also create the separate .venv-whisperx environment (optional alignment / diarization).
.PARAMETER SkipGpuTest
    Skip scripts/check_gpu.py at the end (it downloads the ~75 MB "tiny" model once).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -InstallMissing -WithWhisperX
#>
[CmdletBinding()]
param(
    [switch]$InstallMissing,
    [switch]$WithWhisperX,
    [switch]$SkipGpuTest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step([string]$Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text)   { Write-Host "[ OK ] $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "[WARN] $Text" -ForegroundColor Yellow }
function Write-Fail([string]$Text) { Write-Host "[FAIL] $Text" -ForegroundColor Red }

function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

function Find-Python311 {
    $candidates = @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe", "$env:ProgramFiles\Python311\python.exe")
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $path = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $path) { return $path.Trim() }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($ver -eq "3.11") { return $python.Source }
    }
    return $null
}

function Find-FFmpeg {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return $true }
    $hit = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "ffmpeg.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    return [bool]$hit
}

# ---------------------------------------------------------------- 1. Python 3.11
Write-Step "Checking Python 3.11"
$python = Find-Python311
if (-not $python -and $InstallMissing) {
    Write-Host "Installing Python 3.11 (user scope) with winget ..."
    winget install --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
    Update-SessionPath
    $python = Find-Python311
}
if (-not $python) {
    Write-Fail "Python 3.11 not found. Install it (https://www.python.org/downloads/release/python-3119/) or re-run with -InstallMissing."
    exit 1
}
Write-Ok "Python: $python  ($(& $python --version))"

# ---------------------------------------------------------------- 2. FFmpeg
Write-Step "Checking FFmpeg"
if (-not (Find-FFmpeg) -and $InstallMissing) {
    Write-Host "Installing FFmpeg (user scope) with winget ..."
    winget install --id Gyan.FFmpeg --scope user --silent --accept-package-agreements --accept-source-agreements
    Update-SessionPath
}
if (Find-FFmpeg) { Write-Ok "FFmpeg found" } else {
    Write-Warn "FFmpeg not found. Install: winget install Gyan.FFmpeg  (then open a NEW terminal)."
}

# ---------------------------------------------------------------- 3. NVIDIA driver (never installed here)
Write-Step "Checking NVIDIA driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
    Write-Ok "GPU: $gpu"
    Write-Host "     (If VRAM 'used' is high, close other GPU programs, e.g. 'ollama stop <model>'.)"
} else {
    Write-Warn "nvidia-smi not found. Install/update the NVIDIA driver yourself; this script never does."
}

# ---------------------------------------------------------------- 4. Virtual environment + dependencies
Write-Step "Creating virtual environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $python -m venv .venv }
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install -q --upgrade pip
Write-Ok "venv ready"

Write-Step "Installing dependencies (PyTorch CUDA build is ~2.4 GB - this can take a while)"
& $venvPython -m pip install -q --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install failed"; exit 1 }
Write-Ok "dependencies installed"

# ---------------------------------------------------------------- 5. Folders + config
Write-Step "Preparing folders and checking configuration"
New-Item -ItemType Directory -Force -Path "input", "outputs" | Out-Null
& $venvPython scripts\check_dependencies.py
if ($LASTEXITCODE -ne 0) { Write-Warn "check_dependencies reported problems (see above)" }

# ---------------------------------------------------------------- 6. Optional WhisperX environment
if ($WithWhisperX) {
    Write-Step "Creating the separate WhisperX environment (.venv-whisperx)"
    if (-not (Test-Path ".venv-whisperx\Scripts\python.exe")) { & $python -m venv .venv-whisperx }
    $wxPython = Join-Path $PSScriptRoot ".venv-whisperx\Scripts\python.exe"
    & $wxPython -m pip install --upgrade pip
    & $wxPython -m pip install -r requirements-whisperx.txt
    if ($LASTEXITCODE -ne 0) { Write-Warn "WhisperX install failed - the main app is NOT affected." } else { Write-Ok "WhisperX environment ready" }
}

# ---------------------------------------------------------------- 7. GPU diagnostic
if (-not $SkipGpuTest) {
    Write-Step "Running the GPU diagnostic"
    & $venvPython scripts\check_gpu.py
    if ($LASTEXITCODE -ne 0) { Write-Fail "GPU diagnostic did not report READY - see README, section 'Common CUDA errors'."; exit 1 }
}

Write-Host "`nSetup finished. Put media files in the 'input' folder and run START_QUALITY.bat" -ForegroundColor Green
