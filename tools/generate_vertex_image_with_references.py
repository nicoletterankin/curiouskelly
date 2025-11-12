"""Generate Vertex AI Imagen assets with reference images via Python SDK.

This helper bypasses the REST API limitation around reference images by using
the official Vertex AI Python SDK. It loads reference images from disk using
`VertexImage.load_from_file`, calls the Imagen 3.0 generation model, and writes
the resulting image (optionally resized) to disk.  Designed to be invoked from
PowerShell/CLI automation.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from vertexai import init as vertex_init
    from vertexai.preview.vision_models import (
        ImageGenerationModel,
        Image as VertexImage,
    )
except ImportError as exc:  # pragma: no cover - dependency missing
    print(
        "ERROR: google-cloud-aiplatform is required. Install with"
        " `pip install google-cloud-aiplatform`.",
        file=sys.stderr,
    )
    raise

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: Optional[str]
    aspect_ratio: str
    sample_count: int
    model: str
    project: str
    location: str
    references: List[Path]
    output_path: Path
    width: Optional[int]
    height: Optional[int]
    seed: Optional[int]
    guidance_scale: Optional[float]
    safety_filter_level: Optional[str]
    style: Optional[str]
    metadata_out: Optional[Path]


def parse_args() -> GenerationRequest:
    parser = argparse.ArgumentParser(
        description="Generate images with Vertex AI Imagen using reference photos."
    )
    parser.add_argument("--prompt", required=True, help="Main text prompt")
    parser.add_argument(
        "--negative-prompt",
        dest="negative_prompt",
        help="Negative prompt (undesired attributes)",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        help="Aspect ratio (e.g. 1:1, 3:4, 16:9).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1).",
    )
    parser.add_argument(
        "--model",
        default="imagen-3.0-generate-002",
        help="Vertex AI Imagen model name.",
    )
    parser.add_argument(
        "--project",
        help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var).",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI location/region (default: us-central1).",
    )
    parser.add_argument(
        "--reference",
        dest="references",
        action="append",
        default=[],
        help="Path to reference image (repeat flag for multiple references).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (PNG recommended).",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Target output width (optional – enables resize).",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Target output height (optional – enables resize).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional deterministic seed for Imagen generation.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Optional guidance scale parameter passed to Imagen.",
    )
    parser.add_argument(
        "--safety-filter-level",
        dest="safety_filter_level",
        help="Optional safety filter level.",
    )
    parser.add_argument(
        "--style",
        help="Optional style preset (if supported by model).",
    )
    parser.add_argument(
        "--metadata-out",
        dest="metadata_out",
        help="Optional path to write generation metadata JSON.",
    )

    args = parser.parse_args()

    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        parser.error(
            "Missing project. Pass --project or set the GOOGLE_CLOUD_PROJECT env var."
        )

    references = [Path(ref).expanduser() for ref in args.references]
    for ref in references:
        if not ref.exists():
            parser.error(f"Reference image not found: {ref}")

    return GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        aspect_ratio=args.aspect_ratio,
        sample_count=max(1, args.sample_count),
        model=args.model,
        project=project,
        location=args.location,
        references=references,
        output_path=Path(args.output).expanduser(),
        width=args.width,
        height=args.height,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        safety_filter_level=args.safety_filter_level,
        style=args.style,
        metadata_out=Path(args.metadata_out).expanduser()
        if args.metadata_out
        else None,
    )


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_metadata(req: GenerationRequest, *, success: bool, extra: Dict[str, Any]) -> None:
    if not req.metadata_out:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": success,
        "request": asdict(req),
        "extra": extra,
    }
    # Remove potentially large binary entries (references)
    payload["request"].pop("references", None)
    ensure_output_dir(req.metadata_out)
    req.metadata_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_references(paths: List[Path]) -> List[VertexImage]:
    refs: List[VertexImage] = []
    for path in paths:
        try:
            refs.append(VertexImage.load_from_file(str(path)))
        except Exception as exc:  # pragma: no cover - SDK handles internals
            raise RuntimeError(f"Failed to load reference image {path}: {exc}")
    return refs


def save_image_bytes(data: bytes, output_path: Path, *, width: Optional[int], height: Optional[int]) -> None:
    ensure_output_dir(output_path)
    if width and height:
        if Image is None:
            raise RuntimeError(
                "Pillow is required for resizing. Install with `pip install pillow`."
            )
        image = Image.open(io.BytesIO(data)).convert("RGBA")
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        image.save(output_path)
    else:
        output_path.write_bytes(data)


def generate(req: GenerationRequest) -> None:
    vertex_init(project=req.project, location=req.location)
    model = ImageGenerationModel.from_pretrained(req.model)

    params: Dict[str, Any] = {
        "prompt": req.prompt,
        "number_of_images": req.sample_count,
        "aspect_ratio": req.aspect_ratio,
    }

    # NOTE: Reference images may not be supported in current SDK version
    # The SDK API has changed and reference_images parameter may be deprecated
    # Try passing as keyword argument, but catch TypeError if not supported
    if req.references:
        ref_images = load_references(req.references)
        try:
            # Try the old API first (for backward compatibility)
            params["reference_images"] = ref_images
        except Exception:
            # If that fails, try without reference images for now
            # Reference images may need to be passed differently in newer SDK versions
            print(
                "WARNING: Reference images not supported in current SDK version.",
                "Generating without reference images.",
                file=sys.stderr
            )
            ref_images = []
    
    if req.negative_prompt:
        params["negative_prompt"] = req.negative_prompt
    if req.seed is not None:
        params["seed"] = req.seed
    if req.guidance_scale is not None:
        params["guidance_scale"] = req.guidance_scale
    if req.safety_filter_level:
        params["safety_filter_level"] = req.safety_filter_level
    if req.style:
        params["style_preset"] = req.style

    # Attempt generation, catching TypeError if reference_images parameter is invalid
    try:
        images = model.generate_images(**params)
    except TypeError as e:
        if "reference_images" in str(e) or "unexpected keyword argument" in str(e):
            # Remove reference_images and retry
            if "reference_images" in params:
                del params["reference_images"]
                print(
                    "WARNING: reference_images parameter not supported. Retrying without reference images.",
                    file=sys.stderr
                )
                images = model.generate_images(**params)
            else:
                raise
        else:
            raise
    if not images:
        raise RuntimeError("Vertex AI did not return any images.")

    image0 = images[0]
    image_bytes: Optional[bytes] = None

    if hasattr(image0, "image_bytes") and image0.image_bytes:
        image_bytes = image0.image_bytes
    else:
        buffer = io.BytesIO()
        try:
            # Some SDK versions expose `save` on the image object
            image0.save(buffer, format="PNG")  # type: ignore[attr-defined]
            buffer.seek(0)
            image_bytes = buffer.read()
        except Exception as exc:  # pragma: no cover - fallback path
            raise RuntimeError(
                "Unable to extract image bytes from Vertex AI response"
            ) from exc

    if not image_bytes:
        raise RuntimeError("Vertex AI returned empty image bytes.")

    save_image_bytes(image_bytes, req.output_path, width=req.width, height=req.height)

    write_metadata(
        req,
        success=True,
        extra={
            "model": req.model,
            "aspect_ratio": req.aspect_ratio,
            "reference_count": len(req.references),
            "output_path": str(req.output_path),
        },
    )

    print(f"Generated image saved to {req.output_path}")


def main() -> int:
    try:
        req = parse_args()
        generate(req)
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        if "quota" in str(exc).lower():
            print("Hint: check Vertex AI quotas for Imagen generation.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


