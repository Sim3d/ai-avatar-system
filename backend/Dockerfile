FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.10 + system dependencies.
#
# Wrapped in a retry loop because this layer is the build's most fragile step:
# it runs right after the multi-GB CUDA base image is pulled, then fetches
# ~270 packages (mostly ffmpeg's dependency tree) over plain HTTP. Three
# consecutive CI runs failed here against three different archive.ubuntu.com
# mirror IPs — sometimes on a package, once on the index fetch itself — after
# ~9 minutes of work that then had to be thrown away.
#
# Acquire::Retries alone was not enough: it retries within a single apt
# invocation, so a mirror that is unreachable for a stretch still fails the
# whole command. Retrying the command itself, with a pause between attempts,
# rides out that window.
#
# Deliberately NOT pinning an Azure/region-specific mirror: it would suit
# GitHub's runners but make the Dockerfile worse to build anywhere else.
# Deliberately NOT using --no-install-recommends: it would shrink the image,
# but ffmpeg's recommended packages carry optional codecs, and a missing codec
# surfaces at runtime in the video pipeline rather than at build time.
RUN set -eux; \
    apt_ok=0; \
    for attempt in 1 2 3 4 5; do \
        if apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=30 \
            && apt-get install -y \
                -o Acquire::Retries=5 \
                -o Acquire::http::Timeout=30 \
                python3.10 \
                python3.10-dev \
                python3-pip \
                ffmpeg \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                git \
                wget \
                curl; \
        then apt_ok=1; break; fi; \
        echo "apt attempt ${attempt} failed; retrying in $((attempt * 15))s" >&2; \
        sleep $((attempt * 15)); \
    done; \
    [ "$apt_ok" = 1 ] || { echo "apt failed after 5 attempts" >&2; exit 1; }; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf python3.10 /usr/bin/python3; \
    ln -sf python3 /usr/bin/python; \
    ln -sf pip3 /usr/bin/pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 (matches base image)
RUN pip install --no-cache-dir \
    torch==2.2.0+cu118 \
    torchvision==0.17.0+cu118 \
    torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# MuseTalk lip-sync code, pinned to a known commit. Only the code lives in the
# image: the ~9 GB of weights are downloaded on first boot by entrypoint.sh into
# models/MuseTalk/models, which is meant to be a persistent volume.
ARG MUSETALK_COMMIT=0a89dec45a0192b824e3cf4daf96c239440c5ed8
RUN git init -q models/MuseTalk \
    && git -C models/MuseTalk fetch -q --depth 1 https://github.com/TMElyralab/MuseTalk.git "${MUSETALK_COMMIT}" \
    && git -C models/MuseTalk checkout -q FETCH_HEAD \
    && rm -rf models/MuseTalk/.git \
    && pip install --no-cache-dir einops gdown huggingface_hub

# Optional GFPGAN face restoration (enabled with FACE_RESTORE=gfpgan).
# basicsr builds from source: --no-build-isolation makes it reuse the torch
# already installed instead of pulling another one, and the constraints file
# keeps gfpgan's deps from moving the torch/numpy/opencv pins.
# Weights are baked under /app/gfpgan rather than the models volume so they are
# usable from the very first boot. facexlib helpers go to gfpgan/weights under
# the MuseTalk dir because GFPGANer resolves that path relative to the worker cwd.
COPY requirements-face-restore.txt .
ENV FACE_RESTORE_MODEL=/app/gfpgan/GFPGANv1.4.pth
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy|opencv-python)==' > /tmp/pins.txt \
    && pip install --no-cache-dir --no-build-isolation -c /tmp/pins.txt -r requirements-face-restore.txt \
    && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__); assert torch.__version__.startswith('2.2.0'), 'gfpgan install moved torch'" \
    && sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/' \
        /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py \
    && mkdir -p /app/gfpgan \
    && curl -fSL --retry 5 --retry-delay 5 -o /app/gfpgan/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
    && cd models/MuseTalk \
    && python -c "from facexlib.detection import init_detection_model; from facexlib.parsing import init_parsing_model; init_detection_model('retinaface_resnet50', half=False, device='cpu', model_rootpath='gfpgan/weights'); init_parsing_model(model_name='parsenet', half=False, device='cpu', model_rootpath='gfpgan/weights')" \
    && python -c "from gfpgan import GFPGANer" \
    && rm /tmp/pins.txt

# Copy application code
COPY . .

# Swap in the mmpose-free preprocessing and the persistent worker script.
RUN cp musetalk_preprocessing.py models/MuseTalk/musetalk/utils/preprocessing.py \
    && cp musetalk_worker.py models/MuseTalk/scripts/musetalk_worker.py

# Create necessary directories
RUN mkdir -p /tmp/avatars /tmp/videos /tmp/audio voice_profiles

# Entrypoint: run migrations then start server
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 8000

# Health check — longer start-period for model loading on first boot
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run application via entrypoint (handles migrations automatically)
CMD ["./entrypoint.sh"]
