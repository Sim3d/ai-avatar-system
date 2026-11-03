"""
Download the MuseTalk V1.5 model weights (~8.8 GB) into <musetalk_dir>/models.

Shared by scripts/setup_musetalk.sh (local installs) and the container
entrypoint (first boot on a fresh persistent volume). Idempotent: already
present pieces are skipped, so an interrupted download resumes on the next run.

The BiSeNet face parser is downloaded LAST and written atomically, so its
presence means every earlier piece finished — AvatarAnimator relies on that
to decide when MuseTalk is usable.

Usage: python download_musetalk_weights.py <musetalk_dir>
"""
import os
import sys
import urllib.request


def _has_files(path: str) -> bool:
    return os.path.isdir(path) and bool(os.listdir(path))


def download(musetalk_dir: str) -> None:
    from huggingface_hub import snapshot_download

    models_target = os.path.join(musetalk_dir, "models")
    os.makedirs(models_target, exist_ok=True)

    # ── MuseTalk weights (unet + VAE) ────────────────────────────────────────
    if os.path.isfile(os.path.join(models_target, "musetalkV15", "unet.pth")):
        print("  MuseTalk weights already present — skipping.")
    else:
        print("  Downloading TMElyralab/MuseTalk weights (~7 GB)...")
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=models_target,
            ignore_patterns=["*.md", "*.txt", "*.gitattributes"],
        )
        print("  MuseTalk weights done.")

    # ── Whisper-tiny (audio feature extractor, ~150 MB) ──────────────────────
    whisper_target = os.path.join(models_target, "whisper")
    if not _has_files(whisper_target):
        print("  Downloading openai/whisper-tiny (~150 MB)...")
        snapshot_download(
            repo_id="openai/whisper-tiny",
            local_dir=whisper_target,
            ignore_patterns=["*.md", "*.gitattributes", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("  Whisper-tiny done.")
    else:
        print("  Whisper-tiny already present — skipping.")

    # ── SD-VAE (stabilityai/sd-vae-ft-mse, ~335 MB) ──────────────────────────
    vae_target = os.path.join(models_target, "sd-vae")
    if not _has_files(vae_target):
        print("  Downloading stabilityai/sd-vae-ft-mse (~335 MB)...")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=vae_target,
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        print("  SD-VAE done.")
    else:
        print("  SD-VAE already present — skipping.")

    # ── face-parse-bisent (BiSeNet face segmentation) ────────────────────────
    bisent_dir = os.path.join(models_target, "face-parse-bisent")
    os.makedirs(bisent_dir, exist_ok=True)

    resnet_path = os.path.join(bisent_dir, "resnet18-5c106cde.pth")
    if not os.path.isfile(resnet_path):
        print("  Downloading ResNet18 backbone (~45 MB)...")
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            resnet_path + ".part",
        )
        os.replace(resnet_path + ".part", resnet_path)
        print("  ResNet18 done.")
    else:
        print("  resnet18-5c106cde.pth already present — skipping.")

    bisenet_path = os.path.join(bisent_dir, "79999_iter.pth")
    if not os.path.isfile(bisenet_path):
        print("  Downloading BiSeNet face parser via gdown (~53 MB)...")
        import gdown

        if not gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=bisenet_path + ".part", quiet=False):
            raise RuntimeError("BiSeNet download failed")
        os.replace(bisenet_path + ".part", bisenet_path)
        print("  BiSeNet done.")
    else:
        print("  79999_iter.pth already present — skipping.")

    print("  All model downloads complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <musetalk_dir>")
    download(sys.argv[1])
