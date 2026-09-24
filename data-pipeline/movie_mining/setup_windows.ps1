<#
  One-time setup for movie mining on Windows (PowerShell, from anywhere):

      powershell -ExecutionPolicy Bypass -File data-pipeline\movie_mining\setup_windows.ps1

  What it does:
    1. Checks Python, the NVIDIA GPU, git and ffmpeg (installs ffmpeg via winget if missing).
    2. Picks where the big stuff goes. If C: has less than -MinFreeGB free and a D:
       drive exists, the venv, model caches and media all go to D:\podtekst-mm.
       Otherwise the venv sits in data-pipeline\.venv-mm and media in raw-media\.
    3. Saves PODTEKST_MEDIA_ROOT / HF_HOME / TORCH_HOME as user environment
       variables, so later terminals use the same locations.
    4. Creates the venv, installs CUDA PyTorch + the text and audio requirements.
    5. Downloads the whisper.cpp CUDA build (sets WHISPER_CPP_BIN).
    6. Pre-downloads all models (~6 GB incl. Whisper large-v3) and the OpenSubtitles zip (~1 GB).
    7. Runs the unit tests and a GPU check.

  Re-running is safe: finished steps are skipped or are no-ops.
  Diarization uses the transformers backend natively on Windows. The NeMo
  backend is optional and meant for WSL2.
#>
param(
    [int]$MinFreeGB = 40,              # rough need: ~12 GB tools/models + films and clips
    [string]$CudaTag = "cu128",        # PyTorch CUDA wheel index (RTX 40-series works with cu12x)
    [string]$Drive = "",               # force "C" or "D"
    [switch]$SkipData                  # skip the OpenSubtitles download
)
$ErrorActionPreference = "Stop"
$MM   = $PSScriptRoot                              # ...\data-pipeline\movie_mining
$DP   = Split-Path $MM -Parent                     # ...\data-pipeline
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Run {                  # run a native command, stop on failure (plain $args so "-m" passes through)
    $ErrorActionPreference = "Continue"   # function-local: pip/git notices on stderr must not abort; exit code decides
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "Failed ($LASTEXITCODE): $($args -join ' ')" }
}
function FreeGB($letter) {
    $d = Get-PSDrive -Name $letter -ErrorAction SilentlyContinue
    if ($d) { [math]::Round($d.Free / 1GB, 1) } else { $null }
}

Step "Checking prerequisites"
$pyExe = "python"; $pyArgs = @()
foreach ($v in "3.12", "3.11", "3.10") {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-$v"); break } } catch {}
}
Run $pyExe @pyArgs --version
try { nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader } catch { Write-Warning "nvidia-smi not found -- GPU driver missing? Mining will fall back to CPU (slow)." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is required (for the transformers install). Install Git for Windows first." }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg not found -- installing with winget (Gyan.FFmpeg)..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Warning "ffmpeg installed but not on PATH yet -- open a new terminal after setup." }
}

Step "Choosing storage location"
$freeC = FreeGB "C"; $freeD = FreeGB "D"
Write-Host "Free space: C: $freeC GB$(if ($freeD -ne $null) { ", D: $freeD GB" })"
$useD = $false
if ($Drive -eq "D") { $useD = $true }
elseif ($Drive -eq "C") { $useD = $false }
elseif ($freeC -lt $MinFreeGB -and $freeD -ne $null -and $freeD -gt $freeC) { $useD = $true }
if ($useD) {
    $Base  = "D:\podtekst-mm"
    $Venv  = "$Base\venv"
    $Media = "$Base\media"
    $HF    = "$Base\hf-cache"
    $Torch = "$Base\torch-cache"
} else {
    $Venv  = "$DP\.venv-mm"
    $Media = "$DP\raw-media"
    $HF    = $env:HF_HOME; if (-not $HF) { $HF = "$env:USERPROFILE\.cache\huggingface" }
    $Torch = $env:TORCH_HOME; if (-not $Torch) { $Torch = "$env:USERPROFILE\.cache\torch" }
}
New-Item -ItemType Directory -Force -Path $Media, "$Media\films", $HF, $Torch | Out-Null
foreach ($kv in @(@("PODTEKST_MEDIA_ROOT", $Media), @("HF_HOME", $HF), @("TORCH_HOME", $Torch))) {
    [Environment]::SetEnvironmentVariable($kv[0], $kv[1], "User")
    Set-Item -Path "env:$($kv[0])" -Value $kv[1]
}
Write-Host "venv:   $Venv"
Write-Host "media:  $Media   (put films in $Media\films)"
Write-Host "models: $HF"

Step "Creating venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) { Run $pyExe @pyArgs -m venv $Venv }
$vpy = "$Venv\Scripts\python.exe"
Run $vpy -m pip install --upgrade pip wheel --quiet

Step "Installing PyTorch ($CudaTag)"
$ErrorActionPreference = "Continue"   # PS 5.1 makes native stderr fatal under "Stop"; this probe fails by design
& $vpy -c "import torch" 2>$null
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    Run $vpy -m pip install --no-cache-dir torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaTag"
}

Step "Installing mining requirements"
Run $vpy -m pip install --no-cache-dir -r "$MM\requirements-text.txt" soundfile librosa demucs huggingface_hub
Run $vpy -m pip install --no-cache-dir "gigaam @ git+https://github.com/salute-developers/GigaAM.git"
# The model card installs transformers from git for Nemotron 3 Diarization support.
Run $vpy -m pip install --no-cache-dir --upgrade "git+https://github.com/huggingface/transformers"

Step "Installing whisper.cpp (CUDA build)"
$WDir = if ($useD) { "$Base\whisper.cpp" } else { "$env:LOCALAPPDATA\podtekst\whisper.cpp" }
$cli = Get-ChildItem -Path $WDir -Recurse -Filter "whisper-cli.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $cli) {
    $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest" -Headers @{ "User-Agent" = "podtekst-setup" }
    $zips = $rel.assets | Where-Object { $_.name -like "*.zip" -and $_.name -match "x64" }
    $asset = $zips | Where-Object { $_.name -match "cublas|cuda" -and $_.name -match "12" } | Sort-Object name -Descending | Select-Object -First 1
    if (-not $asset) {
        Write-Warning "No CUDA 12 Windows build in $($rel.tag_name) -- using the CPU build (slower)."
        $asset = $zips | Where-Object { $_.name -notmatch "cublas|cuda|arm" } | Select-Object -First 1
    }
    if (-not $asset) { throw "No Windows x64 whisper.cpp build found in release $($rel.tag_name)" }
    Write-Host "Downloading $($asset.name) ($($rel.tag_name))"
    New-Item -ItemType Directory -Force -Path $WDir | Out-Null
    $zip = Join-Path $WDir $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    Expand-Archive $zip -DestinationPath $WDir -Force
    Remove-Item $zip
    $cli = Get-ChildItem -Path $WDir -Recurse -Filter "whisper-cli.exe" | Select-Object -First 1
}
if (-not $cli) { throw "whisper-cli.exe not found after install in $WDir" }
[Environment]::SetEnvironmentVariable("WHISPER_CPP_BIN", $cli.FullName, "User")
$env:WHISPER_CPP_BIN = $cli.FullName
Write-Host "whisper-cli: $($cli.FullName)"

Step "GPU check"
Run $vpy -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Step "Downloading models$(if (-not $SkipData) { ' + OpenSubtitles' })"
Push-Location $DP
try {
    if ($SkipData) { Run $vpy -m movie_mining.prefetch --no-data } else { Run $vpy -m movie_mining.prefetch }
    Step "Running tests"
    Run $vpy -m unittest discover -s movie_mining/tests -t .
} finally { Pop-Location }

Step "Done"
Write-Host "Activate:  & '$Venv\Scripts\Activate.ps1'   (then cd $DP)"
Write-Host "Text:      python -m movie_mining.mine_subtitles --name subs1 --max-lines 3000000"
Write-Host "Audio:     drop the film or audio file (+ film.ru.srt / film.en.srt if you have them) in $Media\films,"
Write-Host "           then: python -m movie_mining.run_film   (no Russian subtitle -> whisper.cpp transcribes it)"
