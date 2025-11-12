"""
Kelly Asset Generator

Reads a YAML preset describing an asset to generate, calls a placeholder
image generation backend (to be wired to Vertex AI Imagen), performs
optional upscaling/matting stubs, and writes outputs plus a manifest.

This first version focuses on structure, IO, and manifesting so we can
swap the backend safely once credentials are ready.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from PIL import Image
import io
import subprocess
import math
import shutil


@dataclass
class Preset:
    asset_type: str
    view: str
    lighting: str
    output: Dict[str, Any]
    prompt: str
    negative_prompt: Optional[str] = None
    refs: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    backend: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    image_path: Path
    width: int
    height: int
    seed: Optional[int]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_preset(preset_path: Path) -> Preset:
    data = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    return Preset(**data)


def ensure_dirs(outdir: Path) -> None:
    (outdir / "renders").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)


def _placeholder_generate_image(width: int, height: int) -> Image.Image:
    # Placeholder: creates a neutral gradient image to validate the pipeline
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    r = (xv * 255).astype(np.uint8)
    g = (yv * 255).astype(np.uint8)
    b = ((1 - xv) * 255).astype(np.uint8)
    arr = np.stack([r, g, b, np.full_like(r, 255)], axis=-1)
    return Image.fromarray(arr, mode="RGBA")


def _map_aspect_ratio(width: int, height: int) -> str:
    ratio = width / max(1, height)
    # Map to common Aspect Ratios supported by Imagen
    if abs(ratio - 1.0) < 0.02:
        return "1:1"
    if abs(ratio - (16/9)) < 0.02:
        return "16:9"
    if abs(ratio - (9/16)) < 0.02:
        return "9:16"
    if abs(ratio - (4/3)) < 0.02:
        return "4:3"
    if abs(ratio - (3/4)) < 0.02:
        return "3:4"
    # Fallback to square
    return "1:1"


def _try_vertex_imagen(preset: Preset, width: int, height: int) -> Optional[Image.Image]:
    try:
        from vertexai import init as vertex_init
        from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
    except Exception:
        return None

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not project:
        # Without a project, let caller fall back gracefully
        return None

    try:
        vertex_init(project=project, location=location)
        model_name = (preset.backend or {}).get("model", "imagen-3.0-generate-001")
        model = ImageGenerationModel.from_pretrained(model_name)

        ref_images = []
        if preset.refs and isinstance(preset.refs.get("images"), list):
            for item in preset.refs["images"]:
                path = item.get("path") if isinstance(item, dict) else None
                if path and Path(path).exists():
                    try:
                        ref_images.append(VertexImage.load_from_file(path))
                    except Exception:
                        continue

        params: Dict[str, Any] = {
            "prompt": preset.prompt,
            "number_of_images": 1,
            "aspect_ratio": _map_aspect_ratio(width, height),
        }
        if preset.negative_prompt:
            params["negative_prompt"] = preset.negative_prompt
        if preset.seed is not None:
            params["seed"] = int(preset.seed)
        guidance = (preset.backend or {}).get("guidance_scale")
        if guidance is not None:
            params["guidance_scale"] = guidance
        if ref_images:
            params["reference_images"] = ref_images

        images = model.generate_images(**params)
        if not images:
            return None
        img0 = images[0]

        # Attempt to obtain bytes or save directly
        pil_img: Optional[Image.Image] = None
        try:
            if hasattr(img0, "image_bytes") and img0.image_bytes:
                pil_img = Image.open(io.BytesIO(img0.image_bytes)).convert("RGBA")
        except Exception:
            pil_img = None
        if pil_img is None:
            try:
                # If object supports save(), write to memory
                buf = io.BytesIO()
                img0.save(buf, format="PNG")  # type: ignore[attr-defined]
                buf.seek(0)
                pil_img = Image.open(buf).convert("RGBA")
            except Exception:
                pil_img = None

        if pil_img is None:
            return None

        # Resize to target exactly (Imagen picks from fixed sizes by aspect)
        if pil_img.size != (width, height):
            pil_img = pil_img.resize((width, height), Image.LANCZOS)

        return pil_img
    except Exception:
        return None


def generate_with_backend(preset: Preset, outdir: Path) -> GenerationResult:
    target = preset.output or {}
    width = int(target.get("width", 2048))
    height = int(target.get("height", 2048))

    # Enforce exact 8K targets when requested via profiles
    profile = str(target.get("profile", "")).lower()
    if profile == "8k_square":
        width, height = 8192, 8192
    elif profile == "8k_16x9":
        width, height = 7680, 4320

    # Try Vertex Imagen (if configured); otherwise use placeholder
    img = _try_vertex_imagen(preset, width, height)
    if img is None:
        img = _placeholder_generate_image(width, height)

    # File naming
    name = f"kelly_{preset.asset_type}_{preset.view}_{preset.lighting}"
    version = target.get("version", 1)
    filename = f"{name}_v{version:03d}.png"
    image_path = outdir / "renders" / filename
    img.save(image_path)

    # Optional: Real-ESRGAN upscale to enforce exact 8K detail when source < target
    # Supports two strategies: local ncnn executable (faster, no CUDA req) or Python lib if available.
    upscale = (preset.backend or {}).get("upscale", {}).get("enabled", False)
    if upscale:
        factor = max(1.0, max(width, height) / max(1, max(img.size)))
        # If img smaller than target, attempt a 4x pass using realsr-ncnn-vulkan or realesrgan-ncnn-vulkan
        if max(img.size) < max(width, height):
            exe = os.environ.get("REAL_ESRGAN_EXE", "realesrgan-ncnn-vulkan")
            if shutil.which(exe) is not None:  # type: ignore[name-defined]
                tmp_in = image_path
                tmp_out = image_path.with_name(image_path.stem + "_esrgan.png")
                try:
                    subprocess.run([exe, "-i", str(tmp_in), "-o", str(tmp_out), "-s", "4"], check=True)
                    # Replace original with upscaled then resize exact to target
                    up = Image.open(tmp_out).convert("RGBA")
                    up = up.resize((width, height), Image.LANCZOS)
                    up.save(image_path)
                    try:
                        tmp_out.unlink()
                    except Exception:
                        pass
                except Exception:
                    # Silent fallback; keep original
                    pass
            else:
                # Try python package if available
                try:
                    from realesrgan import RealESRGAN
                    import torch
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = RealESRGAN(device, scale=4)
                    model.load_weights("weights/RealESRGAN_x4.pth")
                    up = model.predict(np.array(img.convert("RGB")))
                    up_img = Image.fromarray(up).convert("RGBA").resize((width, height), Image.LANCZOS)
                    up_img.save(image_path)
                except Exception:
                    pass

    return GenerationResult(image_path=image_path, width=width, height=height, seed=preset.seed)


def write_manifest(preset: Preset, result: GenerationResult, outdir: Path) -> Path:
    now = datetime.utcnow().isoformat() + "Z"
    # Lightweight QC metrics on the saved file
    qc = _compute_qc_metrics(result.image_path)
    verdict = _qc_verdict(qc, preset)

    manifest = {
        "schema": "kelly.asset.manifest/v1",
        "created_at": now,
        "preset": asdict(preset),
        "result": {
            "path": str(result.image_path.as_posix()),
            "width": result.width,
            "height": result.height,
            "seed": result.seed,
            "sha256": _sha256_file(result.image_path),
        },
        "lineage": {
            "prompt": preset.prompt,
            "negative_prompt": preset.negative_prompt,
            "refs": preset.refs or {},
        },
        "qc": {"metrics": qc, "verdict": verdict},
    }
    base = Path(result.image_path.name).with_suffix("")
    manifest_path = outdir / "manifests" / f"{base}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _laplacian_sharpness(pil: Image.Image) -> float:
    arr = np.array(pil.convert("L"), dtype=np.float32)
    # 3x3 Laplacian
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # simple conv
    from scipy.signal import convolve2d  # type: ignore
    lap = convolve2d(arr, k, mode="same", boundary="symm")
    return float(np.var(lap))


def _compute_qc_metrics(image_path: Path) -> Dict[str, Any]:
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Brightness as mean luminance
        y = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        brightness = float(y.mean())
        # Contrast as stddev of luminance
        contrast = float(y.std())
        # Colorfulness metric (Hasler–Süsstrunk approximation)
        rg = arr[..., 0] - arr[..., 1]
        yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
        colorfulness = float(np.sqrt(rg.var() + yb.var()))
        # Simple sharpness via Laplacian variance (fallback if scipy missing)
        try:
            sharpness = _laplacian_sharpness(img)
        except Exception:
            # Sobel fallback
            gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
            gy = gx.T
            gray = (y * 255.0).astype(np.float32)
            from scipy.signal import convolve2d  # type: ignore
            sx = convolve2d(gray, gx, mode="same", boundary="symm")
            sy = convolve2d(gray, gy, mode="same", boundary="symm")
            sharpness = float(np.var(np.hypot(sx, sy)))
        h, w = img.size[1], img.size[0]
        return {
            "resolution": {"width": int(w), "height": int(h)},
            "brightness_mean": round(brightness, 6),
            "contrast_std": round(contrast, 6),
            "colorfulness": round(colorfulness, 6),
            "sharpness_var_laplace": round(sharpness, 3),
        }
    except Exception:
        return {"error": "qc_failed"}


def _qc_verdict(qc: Dict[str, Any], preset: Preset) -> Dict[str, Any]:
    # Defaults, can be overridden via preset.backend.qc_thresholds
    thresholds = {
        "brightness_min": 0.10,
        "brightness_max": 0.90,
        "contrast_min": 0.05,
        "sharpness_min": 50.0,
    }
    user = ((preset.backend or {}).get("qc_thresholds") or {})
    thresholds.update({k: v for k, v in user.items() if isinstance(v, (int, float))})

    if "error" in qc:
        return {"pass": False, "reason": "qc_failed", "thresholds": thresholds}

    b = qc.get("brightness_mean", 0.0)
    c = qc.get("contrast_std", 0.0)
    s = qc.get("sharpness_var_laplace", 0.0)

    reasons = []
    if b < thresholds["brightness_min"]:
        reasons.append("too_dark")
    if b > thresholds["brightness_max"]:
        reasons.append("too_bright")
    if c < thresholds["contrast_min"]:
        reasons.append("low_contrast")
    if s < thresholds["sharpness_min"]:
        reasons.append("soft")

    ok = len(reasons) == 0
    return {"pass": ok, "reasons": reasons, "thresholds": thresholds}


def run(preset_path: Path, outdir: Path) -> Dict[str, Any]:
    preset = load_preset(preset_path)
    ensure_dirs(outdir)
    result = generate_with_backend(preset, outdir)
    manifest_path = write_manifest(preset, result, outdir)
    return {
        "image": str(result.image_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kelly asset generator")
    parser.add_argument("preset", type=Path, help="Path to YAML preset")
    parser.add_argument("--outdir", type=Path, default=Path("projects/Kelly/assets"))
    args = parser.parse_args()

    out = run(args.preset, args.outdir)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


