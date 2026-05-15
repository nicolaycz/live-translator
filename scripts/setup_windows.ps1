# Setup script for Windows 10 / 11 (64-bit, native — not WSL).
#
# Everything Python lives inside .venv\ at the repo root — your system Python
# is NOT touched. To use the installed package later, run:
#     .\.venv\Scripts\Activate.ps1
#
# Steps performed:
#   1. Verify prerequisites (Python 3.11/3.12, cmake, MSVC build tools, git)
#   2. Create an isolated Python virtualenv at .venv\
#   3. Install Python dependencies into that .venv\ via pip install -e .
#   4. Compile whisper.cpp with MSVC (CPU; no Metal/CUDA)
#   5. Download Whisper, Silero VAD and Piper models into models\
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 [whisper_model]
#
#   whisper_model: tiny | base | small | medium  (default: small)
#
# Re-running is safe: anything already downloaded or built is skipped.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('tiny', 'base', 'small', 'medium')]
    [string]$WhisperModel = 'small'
)

$ErrorActionPreference = 'Stop'

# --- Paths ---
$RepoRoot   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VendorDir  = Join-Path $RepoRoot 'vendor'
$ModelsDir  = Join-Path $RepoRoot 'models'
$VenvDir    = Join-Path $RepoRoot '.venv'
$WhisperDir = Join-Path $VendorDir 'whisper.cpp'
$PiperVoice = 'en_US-lessac-medium'

Write-Host "==> live-translator setup for Windows (model: $WhisperModel)"

# --- Sanity ---
if ($IsLinux -or $IsMacOS) {
    Write-Error "This script targets Windows. For macOS run scripts/setup_mac.sh, for Raspberry Pi run scripts/setup_rpi.sh."
    exit 1
}

if ([Environment]::Is64BitOperatingSystem -eq $false) {
    Write-Error "live-translator requires a 64-bit Windows. PyTorch and ONNX wheels are not published for 32-bit Windows."
    exit 1
}

function Assert-Cmd {
    param([string]$Name, [string]$HelpHint)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Error "$Name not found on PATH. $HelpHint"
        exit 1
    }
    return $cmd.Source
}

# --- Pick Python ---
# Prefer the `py` launcher (ships with the python.org installer) so we can
# request a specific version. Fall back to whatever `python` resolves to.
$PythonBin = $null
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    foreach ($ver in @('-3.12', '-3.11')) {
        try {
            $check = & py $ver -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $check) {
                $PythonBin = $check.Trim()
                break
            }
        } catch { }
    }
}
if (-not $PythonBin) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $PythonBin = $pythonCmd.Source
    }
}
if (-not $PythonBin) {
    Write-Error @"
No Python interpreter found.
Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
(or run: winget install --id Python.Python.3.11)
Check the 'Add python.exe to PATH' box during install.
"@
    exit 1
}

# Verify version range
$pyOk = & $PythonBin -c "import sys; print(1 if (3,11) <= sys.version_info[:2] < (3,13) else 0)"
if ($pyOk.Trim() -ne '1') {
    $ver = & $PythonBin --version
    Write-Error "Found $PythonBin ($ver), but need Python 3.11 or 3.12. Install with: winget install --id Python.Python.3.11"
    exit 1
}
Write-Host "==> Using $PythonBin ($(& $PythonBin --version))"

# --- cmake, git, MSVC ---
Assert-Cmd 'cmake' 'Install with: winget install --id Kitware.CMake'
Assert-Cmd 'git'   'Install with: winget install --id Git.Git'

# MSVC: cmake will pick up the latest Visual Studio install automatically as
# long as Visual Studio Build Tools 2022 (or later) with the "Desktop
# development with C++" workload is installed. We can't reliably test for
# MSVC from PowerShell without invoking vswhere, so we just warn here and
# let cmake produce a clear error if it's missing.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $vsInstall) {
        Write-Warning "Visual Studio C++ tools not detected. If the whisper.cpp build fails, install: winget install --id Microsoft.VisualStudio.2022.BuildTools (then add the 'Desktop development with C++' workload)."
    } else {
        Write-Host "==> Found MSVC install at: $vsInstall"
    }
} else {
    Write-Warning "Could not find vswhere.exe — skipping MSVC detection. Make sure Visual Studio Build Tools 2022 with 'Desktop development with C++' is installed."
}

# --- Create directories ---
New-Item -ItemType Directory -Force -Path $VendorDir,
    (Join-Path $ModelsDir 'whisper'),
    (Join-Path $ModelsDir 'vad'),
    (Join-Path $ModelsDir 'piper') | Out-Null

# --- whisper.cpp ---
if (-not (Test-Path $WhisperDir)) {
    Write-Host "==> Cloning whisper.cpp"
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp $WhisperDir
}

# MSVC is a multi-config generator → binaries land in build\bin\Release\
$WhisperExe = Join-Path $WhisperDir 'build\bin\Release\whisper-cli.exe'

if (-not (Test-Path $WhisperExe)) {
    Write-Host "==> Building whisper.cpp (CPU, MSVC)"
    cmake -S $WhisperDir -B (Join-Path $WhisperDir 'build') `
        -DGGML_METAL=OFF `
        -DGGML_CUDA=OFF `
        -DWHISPER_BUILD_TESTS=OFF `
        -DWHISPER_BUILD_EXAMPLES=ON
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
    cmake --build (Join-Path $WhisperDir 'build') --config Release -j
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
} else {
    Write-Host "==> whisper.cpp already built"
}

if (-not (Test-Path $WhisperExe)) {
    Write-Error "Build finished but $WhisperExe is missing. Check the cmake output above."
    exit 1
}

# --- Whisper model ---
$WhisperModelFile = Join-Path $ModelsDir "whisper\ggml-$WhisperModel.bin"
if (-not (Test-Path $WhisperModelFile)) {
    Write-Host "==> Downloading Whisper model: $WhisperModel"
    Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WhisperModel.bin" `
                      -OutFile $WhisperModelFile -UseBasicParsing
} else {
    Write-Host "==> Whisper model already present: ggml-$WhisperModel.bin"
}

# Silero VAD ships inside the silero-vad PyPI package — no separate download needed.

# --- Piper voice ---
$PiperOnnx = Join-Path $ModelsDir "piper\$PiperVoice.onnx"
$PiperJson = Join-Path $ModelsDir "piper\$PiperVoice.onnx.json"
if (-not (Test-Path $PiperOnnx)) {
    Write-Host "==> Downloading Piper voice: $PiperVoice"
    Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$PiperVoice.onnx" `
                      -OutFile $PiperOnnx -UseBasicParsing
}
if (-not (Test-Path $PiperJson)) {
    Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$PiperVoice.onnx.json" `
                      -OutFile $PiperJson -UseBasicParsing
}
Write-Host "==> Piper voice ready"

# --- Python venv ---
Set-Location $RepoRoot

if (-not (Test-Path $VenvDir)) {
    Write-Host "==> Creating isolated Python virtualenv at: $VenvDir"
    & $PythonBin -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "==> Reusing existing Python virtualenv at: $VenvDir"
}

$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip    = Join-Path $VenvDir 'Scripts\pip.exe'

Write-Host "==> Installing Python dependencies into .venv\ (not your system Python)"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $VenvPip install -e .
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "==> Dependencies installed in $VenvDir"

Write-Host ""
Write-Host "OK Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "Activate the virtualenv in your shell:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "List audio devices:"
Write-Host "  live-translator devices"
Write-Host ""
Write-Host "Verify your mic before going live:"
Write-Host "  live-translator mic-test"
Write-Host ""
Write-Host "Live mode (mic to speaker):"
Write-Host "  live-translator run"
