#!/usr/bin/env bash
# Setup MuseTalk V1.5 for lip-sync animation (replaces SadTalker)
# Run from the project root:  bash scripts/setup_musetalk.sh
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
MODELS_DIR="$BACKEND_DIR/models"
MUSETALK_DIR="$MODELS_DIR/MuseTalk"
VENV_PYTHON="$BACKEND_DIR/venv/bin/python"
SENTINEL="$PROJECT_ROOT/.musetalk_ready"

if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

echo "=== MuseTalk V1.5 Setup ==="
echo "Project : $PROJECT_ROOT"
echo "Backend : $BACKEND_DIR"
echo "Python  : $VENV_PYTHON"
echo ""

# ── 1. Clone MuseTalk ────────────────────────────────────────────────────────
if [ -d "$MUSETALK_DIR/.git" ]; then
  echo "[1/5] MuseTalk already cloned — pulling latest..."
  git -C "$MUSETALK_DIR" pull --ff-only
else
  echo "[1/5] Cloning MuseTalk..."
  mkdir -p "$MODELS_DIR"
  git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
fi

# ── 2. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "[2/5] Installing MuseTalk requirements..."

# Install requirements, skipping packages incompatible with Python 3.12
# (mmpose/mmcv have no 3.12 wheels; tensorflow is optional for our use case)
"$VENV_PYTHON" -m pip install -q \
  "diffusers==0.32.2" \
  "accelerate==0.28.0" \
  "transformers==4.39.2" \
  "opencv-python==4.9.0.80" \
  "soundfile==0.12.1" \
  "librosa==0.11.0" \
  "einops==0.8.1" \
  "omegaconf" \
  "pyyaml" \
  "imageio" \
  "imageio[ffmpeg]" \
  "ffmpeg-python" \
  "moviepy<2" \
  "mediapipe" \
  "face-alignment" \
  "safetensors" \
  "timm"

# ── 3. Replace preprocessing.py with CPU-compatible version ─────────────────
echo ""
echo "[3/5] Writing CPU-compatible preprocessing.py (replaces mmpose with face_alignment)..."

PREPROCESS="$MUSETALK_DIR/musetalk/utils/preprocessing.py"
# Always overwrite — idempotent, works on first run and re-runs
cp "$PREPROCESS" "${PREPROCESS}.orig" 2>/dev/null || true
# Shared with the backend Dockerfile so local and container installs match.
cp "$BACKEND_DIR/musetalk_preprocessing.py" "$PREPROCESS"
echo "  preprocessing.py replaced ✓"

# ── 3b. Install our custom persistent worker ────────────────────────────────
# musetalk_worker.py is OUR driver (loads the models once and serves inference
# jobs over stdin/stdout), not part of upstream MuseTalk — so the git clone
# above doesn't include it. Copy the tracked copy from the backend into the
# clone's scripts/ dir. The backend can also run it from its tracked location,
# but copying keeps everything self-contained under the MuseTalk repo.
echo ""
echo "[3b/5] Installing persistent MuseTalk worker (musetalk_worker.py)..."
WORKER_SRC="$BACKEND_DIR/musetalk_worker.py"
WORKER_DST="$MUSETALK_DIR/scripts/musetalk_worker.py"
if [ -f "$WORKER_SRC" ]; then
  cp "$WORKER_SRC" "$WORKER_DST"
  echo "  musetalk_worker.py installed ✓"
else
  echo "  ERROR: $WORKER_SRC not found — is the repo checkout complete?" >&2
  exit 1
fi

# ── 4. Download model weights from HuggingFace ──────────────────────────────
echo ""
echo "[4/5] Downloading MuseTalk model weights (~8.8 GB total)..."

# gdown is required for the Google Drive face parsing model
"$VENV_PYTHON" -m pip install -q gdown

# Shared with the container entrypoint (first-boot download).
"$VENV_PYTHON" "$BACKEND_DIR/download_musetalk_weights.py" "$MUSETALK_DIR"

# ── 5. Verify and write sentinel ─────────────────────────────────────────────
echo ""
echo "[5/5] Verifying installation..."

MISSING=0

if [ ! -f "$MUSETALK_DIR/scripts/inference.py" ]; then
  echo "  MISSING: scripts/inference.py"
  MISSING=1
else
  echo "  scripts/inference.py ✓"
fi

if [ ! -f "$MUSETALK_DIR/scripts/musetalk_worker.py" ]; then
  echo "  MISSING: scripts/musetalk_worker.py"
  MISSING=1
else
  echo "  scripts/musetalk_worker.py ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/musetalkV15/unet.pth" ]; then
  echo "  MISSING: models/musetalkV15/unet.pth"
  MISSING=1
else
  echo "  models/musetalkV15/unet.pth ✓"
fi

if [ ! -d "$MUSETALK_DIR/models/whisper" ]; then
  echo "  MISSING: models/whisper/"
  MISSING=1
else
  echo "  models/whisper/ ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/sd-vae/config.json" ]; then
  echo "  MISSING: models/sd-vae/config.json"
  MISSING=1
else
  echo "  models/sd-vae/ ✓"
fi

if [ ! -f "$MUSETALK_DIR/models/face-parse-bisent/resnet18-5c106cde.pth" ]; then
  echo "  MISSING: models/face-parse-bisent/resnet18-5c106cde.pth"
  MISSING=1
elif [ ! -f "$MUSETALK_DIR/models/face-parse-bisent/79999_iter.pth" ]; then
  echo "  MISSING: models/face-parse-bisent/79999_iter.pth"
  MISSING=1
else
  echo "  models/face-parse-bisent/ ✓"
fi

if [ "$MISSING" -eq 0 ]; then
  touch "$SENTINEL"
  echo ""
  echo "  Sentinel written: $SENTINEL"
  echo ""
  echo "=== MuseTalk setup complete! ==="
  echo "Set AVATAR_ENGINE=musetalk in .env and restart the backend."
else
  echo ""
  echo "Some checks failed — see above. Fix and re-run."
  exit 1
fi
