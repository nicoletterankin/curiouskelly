"""Generate local 16:9 Kelly persona images from square persona PNGs.

Why:
- The app currently falls back to a photoreal placeholder image.
- We want a fully-local fallback set derived from our Kelly persona images.

Inputs:
- public/assets/kelly/personas/*.png

Outputs:
- public/images/kelly/personas-16x9/<persona>.jpg
- public/images/kelly-placeholder-16x9.jpg (copied from scientist)

This script does NOT call any external APIs.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "public" / "assets" / "kelly" / "personas"
OUT_DIR = ROOT / "public" / "images" / "kelly" / "personas-16x9"
PLACEHOLDER_OUT = ROOT / "public" / "images" / "kelly-placeholder-16x9.jpg"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def make_vertical_gradient(size: tuple[int, int], top_rgb: tuple[int, int, int], bottom_rgb: tuple[int, int, int]) -> Image.Image:
    w, h = size
    bg = Image.new("RGB", (w, h), top_rgb)
    px = bg.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_rgb[0] * (1 - t) + bottom_rgb[0] * t)
        g = int(top_rgb[1] * (1 - t) + bottom_rgb[1] * t)
        b = int(top_rgb[2] * (1 - t) + bottom_rgb[2] * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return bg


def add_vignette(img: Image.Image, strength: float = 0.55) -> Image.Image:
    # Darken edges to keep focus on subject.
    w, h = img.size
    overlay = Image.new("L", (w, h), 0)
    opx = overlay.load()
    cx, cy = w / 2.0, h / 2.0
    maxd = math.sqrt(cx * cx + cy * cy)
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / maxd
            v = int(_clamp((d ** 1.6) * 255 * strength, 0, 255))
            opx[x, y] = v
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(8, int(min(w, h) * 0.015))))
    return Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, overlay)


def white_key_alpha(src_rgb: Image.Image, t0: float = 8.0, t1: float = 70.0) -> Image.Image:
    """Create an alpha matte that removes near-white backgrounds.

    t0: distance-to-white below which alpha=0
    t1: distance-to-white above which alpha=255
    """

    rgb = src_rgb.convert("RGB")
    w, h = rgb.size
    ap = Image.new("L", (w, h), 0)
    rp = rgb.load()
    apx = ap.load()
    for y in range(h):
        for x in range(w):
            r, g, b = rp[x, y]
            d = math.sqrt((255 - r) ** 2 + (255 - g) ** 2 + (255 - b) ** 2)
            a = (d - t0) / max(1e-6, (t1 - t0))
            apx[x, y] = int(_clamp(a * 255, 0, 255))
    return ap.filter(ImageFilter.GaussianBlur(radius=1.2))


def crop_to_alpha(src_rgba: Image.Image, pad: int = 20) -> Image.Image:
    alpha = src_rgba.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return src_rgba
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(src_rgba.width, r + pad)
    b = min(src_rgba.height, b + pad)
    return src_rgba.crop((l, t, r, b))


def make_subject_rgba(src_path: Path) -> Image.Image:
    src = Image.open(src_path)
    alpha = None
    if src.mode in ("RGBA", "LA"):
        alpha = src.getchannel("A")
    # Many persona PNGs are RGB on pure white; key them.
    key_alpha = white_key_alpha(src)
    if alpha is None:
        alpha = key_alpha
    else:
        # Combine existing alpha with white-key alpha.
        alpha = ImageChops.multiply(alpha, key_alpha)
    rgba = src.convert("RGBA")
    rgba.putalpha(alpha)
    rgba = crop_to_alpha(rgba, pad=24)
    return rgba


def render_persona_16x9(persona_rgba: Image.Image, out_size: tuple[int, int]) -> Image.Image:
    w, h = out_size

    # Background gradient tuned to the app's dark/glass aesthetic.
    bg = make_vertical_gradient((w, h), top_rgb=(8, 10, 18), bottom_rgb=(22, 12, 34))
    bg = add_vignette(bg, strength=0.65)

    # Scale subject to occupy ~92% of height.
    target_h = int(h * 0.92)
    scale = target_h / max(1, persona_rgba.height)
    new_w = int(persona_rgba.width * scale)
    new_h = int(persona_rgba.height * scale)
    subj = persona_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Shadow
    shadow = Image.new("RGBA", subj.size, (0, 0, 0, 0))
    shadow.putalpha(subj.getchannel("A"))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(6, int(h * 0.02))))

    # Position slightly lower than center to feel grounded.
    x = (w - new_w) // 2
    y = int(h * 0.08)

    canvas = bg.convert("RGBA")
    canvas.alpha_composite(shadow, (x + int(w * 0.01), y + int(h * 0.02)))
    canvas.alpha_composite(subj, (x, y))

    # Flatten to JPEG
    return canvas.convert("RGB")


def main() -> int:
    if not IN_DIR.exists():
        raise SystemExit(f"Missing input directory: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    persona_paths = sorted(IN_DIR.glob("*.png"))
    if not persona_paths:
        raise SystemExit(f"No persona PNGs found in: {IN_DIR}")

    out_size = (1280, 720)
    generated = []

    for p in persona_paths:
        persona_id = p.stem
        print(f"Generating 16:9 persona: {persona_id}")

        src = Image.open(p).convert("RGB")
        alpha = white_key_alpha(src)
        rgba = src.convert("RGBA")
        rgba.putalpha(alpha)
        rgba = crop_to_alpha(rgba, pad=24)

        out = render_persona_16x9(rgba, out_size=out_size)
        out_path = OUT_DIR / f"{persona_id}.jpg"
        out.save(out_path, format="JPEG", quality=92, optimize=True, progressive=True)
        generated.append((persona_id, out_path))

    # Keep existing app contract: placeholder path stays stable.
    scientist = OUT_DIR / "scientist.jpg"
    if scientist.exists():
        print(f"Updating placeholder: {PLACEHOLDER_OUT} <- {scientist}")
        Image.open(scientist).save(PLACEHOLDER_OUT, format="JPEG", quality=92, optimize=True, progressive=True)

    print(f"Done. Wrote {len(generated)} persona images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
