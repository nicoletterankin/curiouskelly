# --- PIPER TTS: One-Paste Bootstrap for Windows + RTX 5090 (PowerShell) ---
$ErrorActionPreference = "Stop"
$host.ui.RawUI.WindowTitle = "Piper TTS Bootstrap"

# === Project settings ===
$WORKDIR   = "$HOME\UI-TARS-desktop\synthetic_tts"
$REPO_URL  = "https://github.com/rhasspy/piper.git"
$REPO_DIR  = Join-Path $WORKDIR "piper_training"
$ENV_DIR   = Join-Path $REPO_DIR "src\python\.venv"
$PYTHON_EXE = ""
$CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"  # CUDA 12.1 builds (works on 5090/Blackwell via compatibility); fallback logic below
$DATASET   = "Kelly25"
$DATA_ROOT = Join-Path $REPO_DIR "data\$DATASET"

function Section($t){ Write-Host "`n==== $t ====" -ForegroundColor Cyan }
function Note($t){ Write-Host "  - $t" -ForegroundColor DarkGray }
function Ok($t){ Write-Host "✔ $t" -ForegroundColor Green }
function Warn($t){ Write-Host "⚠ $t" -ForegroundColor Yellow }
function Fail($t){ Write-Host "✘ $t" -ForegroundColor Red }

# 0) Ensure working directory exists
Section "Prepare workspace"
New-Item -ItemType Directory -Force -Path $WORKDIR | Out-Null
Set-Location $WORKDIR
Ok "Workspace: $WORKDIR"

# 1) Clone/refresh repo
Section "Clone Piper (training)"
if (!(Test-Path $REPO_DIR)) {
  git clone $REPO_URL $REPO_DIR
  Ok "Cloned to $REPO_DIR"
} else {
  Set-Location $REPO_DIR
  git fetch --all --prune
  git pull --rebase
  Ok "Updated existing repo"
}
Set-Location $REPO_DIR

# 2) Create venv (under src/python/.venv to match docs)
Section "Create & activate virtual environment"
$PY_SRC = Join-Path $REPO_DIR "src\python"
New-Item -ItemType Directory -Force -Path $PY_SRC | Out-Null
Set-Location $PY_SRC

if (!(Test-Path $ENV_DIR)) {
  python -m venv ".venv"
  Ok "Created venv at $ENV_DIR"
} else {
  Ok "Venv already exists"
}

# Activate for this session
$activate = Join-Path $ENV_DIR "Scripts\Activate.ps1"
. $activate
Ok "Activated venv"

# Get python path inside venv
$PYTHON_EXE = (Get-Command python).Source
Note "Python in venv: $PYTHON_EXE"

# 3) Robust pip toolchain upgrade (Windows-safe)
Section "Upgrade pip, wheel, setuptools"
python -m pip install --upgrade pip wheel setuptools
Ok "pip toolchain upgraded"

# 4) Install PyTorch (+CUDA) with fallback
Section "Install PyTorch with CUDA"
$torchOk = $false
try {
  python - <<'PY'
import sys, subprocess
def pipi(args):
    subprocess.check_call([sys.executable,"-m","pip","install",*args])
# Try CUDA 12.1 wheels first
try:
    pipi(["--index-url","https://download.pytorch.org/whl/cu121","torch","torchvision","torchaudio"])
    print("CUDA121_OK")
except Exception as e:
    print("CUDA121_FAIL", e)
    # Fallback to default index (may install CPU build)
    pipi(["torch","torchvision","torchaudio"])
    print("DEFAULT_OK")
PY
  $torchOk = $true
} catch {
  $torchOk = $false
}
if ($torchOk) { Ok "PyTorch installed" } else { Fail "PyTorch install failed"; exit 1 }

# 5) Install Piper training requirements (if present)
Section "Install Piper requirements"
Set-Location $REPO_DIR
# Some repos keep requirements in root or in training submodules; try common spots.
$reqs = @("requirements.txt","src\python\requirements.txt","trainer\requirements.txt") | ForEach-Object { Join-Path $REPO_DIR $_ } | Where-Object { Test-Path $_ }
if ($reqs.Count -gt 0) {
  foreach ($r in $reqs) {
    Note "Installing -r $(Split-Path $r -Leaf)"
    python -m pip install -r $r
  }
  Ok "Piper requirements installed"
} else {
  Warn "No requirements.txt found; continuing (the repo may vendor deps)."
}

# 6) Extra deps commonly needed for training (phonemizer, datasets, aligners helpers)
Section "Install common training extras"
python -m pip install phonemizer==3.* torchaudio tensorboard datasets pandas tqdm numba soundfile unidecode librosa
Ok "Installed extras"

# 7) Verify GPU availability
Section "Verify CUDA & GPU"
python - <<'PY'
import torch
print("torch.__version__:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version reported by PyTorch:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: CUDA not available. You may have installed CPU wheels.")
PY

# 8) Scaffold Kelly25 dataset layout
Section "Scaffold dataset folders"
New-Item -ItemType Directory -Force -Path $DATA_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DATA_ROOT "wavs") | Out-Null
$meta = Join-Path $DATA_ROOT "metadata.csv"
if (!(Test-Path $meta)) {
  @"
# LJSpeech-style CSV: <utt_id>|<normalized_text>|<raw_text>
# Example:
# kelly_0001|HELLO I AM KELLY.|Hello, I'm Kelly.
"@ | Set-Content -Encoding UTF8 $meta
  Ok "Created $meta"
} else {
  Ok "Found $meta"
}
Ok "Dataset root: $DATA_ROOT"
Note "Put your 22.05kHz or 24kHz mono WAVs in /wavs and fill metadata.csv"

# 9) Quick CLI sanity checks
Section "Sanity checks"
# Help screens (these shouldn't error)
try { python -m piper_train --help | Out-Null; Ok "piper_train CLI reachable" } catch { Warn "piper_train help failed (module name may differ in this repo)" }
try { python -c "import phonemizer; print('phonemizer OK')" | Out-Null; Ok "phonemizer import OK" } catch { Warn "phonemizer not importable" }

# 10) (Optional) TensorBoard log dir + example training command stencil
Section "Training command stencil"
$LOGDIR = Join-Path $REPO_DIR "runs\$DATASET"
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
$train_cmd = @"
# --- EXAMPLE TRAIN START (copy, edit, and run when data ready) ---
# Activate venv if not already:
#   `.` "$ENV_DIR\Scripts\Activate.ps1"
#
# Start TensorBoard (optional, new terminal):
#   tensorboard --logdir "$LOGDIR" --port 6006
#
# Example Piper training command (adjust paths/sample rate/hparams to your data):
python -m piper_train `
  --dataset-dir "$DATA_ROOT" `
  --metadata-file "$meta" `
  --output-dir "$REPO_DIR\models\$DATASET" `
  --log-dir "$LOGDIR" `
  --epochs 200 `
  --batch-size 32 `
  --learning-rate 1e-3 `
  --sample-rate 22050
# ---------------------------------------------------------------
"@
$train_cmd | Set-Content -Encoding UTF8 (Join-Path $REPO_DIR "TRAIN_$($DATASET).ps1")
Ok "Wrote training stencil: $(Join-Path $REPO_DIR "TRAIN_$($DATASET).ps1")"

# 11) Final summary
Section "Done"
Ok "Environment ready. Next steps:"
Note "1) Add WAVs to: $($DATA_ROOT)\wavs"
Note "2) Edit $meta with utterance rows (utt_id|normalized|raw)"
Note "3) Optional: open TensorBoard at $LOGDIR"
Note "4) Start training by editing & running TRAIN_$($DATASET).ps1"





































