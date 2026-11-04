"""
Persistent MuseTalk worker.

Lifecycle:
  1. Start: reads one JSON line {
       "unet_model_path": ...,
       "unet_config": ...,
       "whisper_dir": ...,
       "vae_type": ...,
       "use_float16": true/false   <- optional, default false (enable on GPU for ~2x speedup)
     }
  2. Loads all models once, prints "READY\\n" to stdout.
  3. Loop: reads one JSON job line {
           "image": ..., "audio": ...,
           "output": ..., "coord_cache": ...}
           runs inference, prints {"status":"ok","output":...}\\n
           or {"status":"error","msg":...}\\n

All paths may be absolute or relative to cwd (the MuseTalk repo root).
"""

import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from typing import Any, List

# stdout is the parent<->worker protocol channel, but upstream MuseTalk and
# its helpers print() diagnostics (e.g. "load unet model from ..."). Keep a
# private handle for protocol messages and route every other print to stderr,
# which the parent captures to worker_stderr.log.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

import cv2
import numpy as np
import torch
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import face_seg, get_crop_box, get_image_blending
from PIL import Image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, load_all_model
from transformers import WhisperModel

FPS = 25
EXTRA_MARGIN = 10
BATCH_SIZE = 8
AUDIO_PAD_L = 2
AUDIO_PAD_R = 2


def _reply(obj: dict):
    _PROTOCOL_OUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_OUT.flush()


def _load_face_restorer(mode: str, model_path: str, device):
    """
    Load GFPGAN once, or return None.

    Never raises. Restoration is a quality nicety; a missing weight file or a
    dependency that moved must degrade to plain MuseTalk output rather than
    take down lip-sync, which is the actual feature. Failures are reported on
    stderr (which the parent captures to worker_stderr.log) and then dropped.
    """
    if mode != "gfpgan":
        return None
    if device.type != "cuda":
        # On CPU a MuseTalk turn already takes 30-90 s; adding a per-frame
        # restoration pass would make it unusable rather than nicer.
        sys.stderr.write("INFO: face restoration skipped — requires CUDA\n")
        sys.stderr.flush()
        return None
    if not os.path.isfile(model_path):
        sys.stderr.write(
            f"WARN: face restoration disabled — weights not found at {model_path}. "
            f"Run scripts/setup_face_restore.sh\n"
        )
        sys.stderr.flush()
        return None
    try:
        from gfpgan import GFPGANer

        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,  # compositing is MuseTalk's job; only restore detail
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,  # background is untouched original pixels already
        )
        sys.stderr.write("INFO: face restoration enabled (gfpgan)\n")
        sys.stderr.flush()
        return restorer
    except Exception as e:
        sys.stderr.write(f"WARN: face restoration disabled — {type(e).__name__}: {e}\n")
        sys.stderr.flush()
        return None


def _restore(restorer, frame):
    """
    Restore the face in a composited frame, or return it unchanged.

    Applied after MuseTalk has blended its generated mouth back in, so GFPGAN
    sees a complete face to detect. Running it on the raw generated region
    instead would be cheaper, but that region is a jaw/mouth crop and the face
    detector frequently finds nothing in it.

    only_center_face keeps it to the avatar rather than anyone in the
    background; paste_back composites the restored face into the frame.
    """
    if restorer is None:
        return frame
    try:
        _, _, restored = restorer.enhance(
            frame, has_aligned=False, only_center_face=True, paste_back=True
        )
        return restored if restored is not None else frame
    except Exception as e:
        # Per-frame failure must not abort the turn; one bad frame is invisible,
        # a failed turn is not.
        sys.stderr.write(f"WARN: face restoration failed on a frame: {e}\n")
        sys.stderr.flush()
        return frame


def _match_tone(generated, original):
    """
    Shift the generated face crop's colour onto the original crop's.

    MuseTalk's VAE re-renders the lower face with a slightly different skin
    tone, which shows as a visible patch while talking that vanishes when the
    idle image comes back. Only the lower half is compared (that is what the
    jaw mask pastes back), and the median per-pixel LAB difference is used so
    the open mouth and teeth, a minority of pixels, don't skew the offset.
    """
    half = generated.shape[0] // 2
    gen_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.int16)
    ori_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = ori_lab[half:] - gen_lab[half:]
    offset = np.median(diff.reshape(-1, 3), axis=0).astype(np.int16)
    shifted = np.clip(gen_lab + offset, 0, 255).astype(np.uint8)
    return cv2.cvtColor(shifted, cv2.COLOR_LAB2BGR)


def _prepare_blend_mask(image, face_box, fp, mode="jaw", upper_boundary_ratio=0.5, expand=1.5):
    """
    The blend mask of upstream `get_image`, computed once instead of per frame.

    The mask is derived from the original frame only, never from the generated
    mouth, so for a fixed frame and bbox it is constant across the whole clip.
    Upstream's `get_image_prepare_material` is not used because it blurs with a
    0.1 kernel ratio while `get_image` uses 0.05; this keeps output identical.
    """
    body = Image.fromarray(image[:, :, ::-1])
    x, y, x1, y1 = face_box
    crop_box, _ = get_crop_box(face_box, expand)
    x_s, y_s, _, _ = crop_box
    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new("L", ori_shape, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified = Image.new("L", ori_shape, 0)
    modified.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur = int(0.05 * ori_shape[0] // 2 * 2) + 1
    return cv2.GaussianBlur(np.array(modified), (blur, blur), 0), crop_box


def _run_job(job, vae, unet, pe, audio_processor, whisper, fp, timesteps, device, restorer=None):
    image_path = job["image"]
    audio_path = job["audio"]
    output_path = job["output"]
    coord_cache = job.get("coord_cache")
    timings = {}
    t0 = time.perf_counter()

    input_img_list = [image_path]

    # ── face coordinates (cached per avatar image) ───────────────────────────
    coord_list: List[Any]
    frame_list: List[Any]
    if coord_cache and os.path.exists(coord_cache):
        with open(coord_cache, "rb") as f:
            coord_list = list(pickle.load(f))
        frame_list = list(read_imgs(input_img_list))
    else:
        _coords, _frames = get_landmark_and_bbox(input_img_list, 0)
        coord_list, frame_list = list(_coords), list(_frames)
        if coord_cache:
            os.makedirs(os.path.dirname(coord_cache), exist_ok=True)
            with open(coord_cache, "wb") as f:
                pickle.dump(coord_list, f)

    if not frame_list or all(c == coord_placeholder for c in coord_list):
        raise RuntimeError("No face detected in avatar image")

    # ── audio features ───────────────────────────────────────────────────────
    weight_dtype = unet.model.dtype
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=FPS,
        audio_padding_length_left=AUDIO_PAD_L,
        audio_padding_length_right=AUDIO_PAD_R,
    )

    # ── VAE-encode frames ────────────────────────────────────────────────────
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(crop))  # type: ignore[arg-type]

    if not input_latent_list:
        raise RuntimeError("No valid face crops produced")

    frame_list_cycle: List[Any] = frame_list + list(reversed(frame_list))  # type: ignore[operator]
    coord_list_cycle: List[Any] = coord_list + list(reversed(coord_list))  # type: ignore[operator]
    latent_list_cycle: List[Any] = input_latent_list + list(reversed(input_latent_list))

    timings["prep"] = time.perf_counter() - t0
    t0 = time.perf_counter()

    # ── UNet inference ───────────────────────────────────────────────────────
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=latent_list_cycle,
        batch_size=BATCH_SIZE,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    for whisper_batch, latent_batch in gen:
        audio_feat = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
        for f in vae.decode_latents(pred):
            res_frame_list.append(f)

    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["unet"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    # The rest is CPU-only unless face restoration runs on the GPU; tell the
    # backend so it can start the next sentence's TTS without contending.
    if restorer is None:
        _reply({"id": job.get("id"), "status": "gpu_done"})

    # ── blend back, streamed straight into one ffmpeg pass (video + audio) ──
    height, width = frame_list_cycle[0].shape[:2]
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS),
            "-i", "-",
            "-i", audio_path,
            # Encode and tag BT.709 explicitly: untagged output is converted
            # with BT.601 but decoded by browsers as BT.709, shifting the
            # colours of the whole clip away from the idle image.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart",
            output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,  # stdout is the worker protocol channel
    )
    masks = {}
    try:
        for i, res_frame in enumerate(res_frame_list):
            idx = i % len(coord_list_cycle)
            x1, y1, x2, y2 = coord_list_cycle[idx]
            ori = frame_list_cycle[i % len(frame_list_cycle)]
            y2 = min(y2 + EXTRA_MARGIN, ori.shape[0])
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except Exception:
                continue
            res_frame = _match_tone(res_frame, ori[y1:y2, x1:x2])
            if idx not in masks:
                masks[idx] = _prepare_blend_mask(ori, [x1, y1, x2, y2], fp)
            mask_array, crop_box = masks[idx]
            combined = get_image_blending(ori, res_frame, [x1, y1, x2, y2], mask_array, crop_box)
            combined = _restore(restorer, combined)
            ffmpeg.stdin.write(np.ascontiguousarray(combined).tobytes())
    finally:
        ffmpeg.stdin.close()
        rc = ffmpeg.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}")
    timings["blend_encode"] = time.perf_counter() - t0
    timings["frames"] = len(res_frame_list)
    return timings


def main():
    # ── read init config ──────────────────────────────────────────────────────
    init = json.loads(sys.stdin.readline())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_float16 = init.get("use_float16", False)

    vae, unet, pe = load_all_model(
        unet_model_path=init["unet_model_path"],
        vae_type=init["vae_type"],
        unet_config=init["unet_config"],
        device=device,
    )

    # float16 on GPU = ~2× faster via Tensor Cores (A10G / L4 / V100 all support it)
    if use_float16 and device.type == "cuda":
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        sys.stderr.write("INFO: float16 enabled — ~2× faster on GPU\n")
        sys.stderr.flush()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    weight_dtype = unet.model.dtype
    audio_processor = AudioProcessor(feature_extractor_path=init["whisper_dir"])
    whisper = WhisperModel.from_pretrained(init["whisper_dir"])
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing()
    timesteps = torch.tensor([0], device=device)

    # Loaded once, like every other model here — the worker is persistent
    # precisely so per-turn cost is inference only.
    restorer = _load_face_restorer(
        init.get("face_restore", "off"), init.get("face_restore_model", ""), device
    )

    _PROTOCOL_OUT.write("READY\n")
    _PROTOCOL_OUT.flush()

    # ── job loop ──────────────────────────────────────────────────────────────
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            job = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            timings = _run_job(
                job, vae, unet, pe, audio_processor, whisper, fp, timesteps, device, restorer
            )
            _reply(
                {"id": job.get("id"), "status": "ok", "output": job["output"], "timings": timings}
            )
        except Exception as e:
            _reply(
                {"id": job.get("id"), "status": "error", "msg": str(e), "tb": traceback.format_exc()}
            )


if __name__ == "__main__":
    main()
