"""
Generate iOS/Android icon PNGs from a single square source image.

Usage (PowerShell):
  python scripts/mobile/generate_app_store_icons.py --src public/images/brand/kelly-logo-square.png --out public/icons

Notes:
- Apple App Store icon must be 1024x1024 PNG with no alpha. We flatten alpha onto a solid background.
- This script is intentionally dependency-light (Pillow only).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class IconSpec:
    rel_path: str
    size: int


IOS_ICONS: list[IconSpec] = [
    IconSpec("ios/Icon-App-20x20@1x.png", 20),
    IconSpec("ios/Icon-App-20x20@2x.png", 40),
    IconSpec("ios/Icon-App-20x20@3x.png", 60),
    IconSpec("ios/Icon-App-29x29@1x.png", 29),
    IconSpec("ios/Icon-App-29x29@2x.png", 58),
    IconSpec("ios/Icon-App-29x29@3x.png", 87),
    IconSpec("ios/Icon-App-40x40@1x.png", 40),
    IconSpec("ios/Icon-App-40x40@2x.png", 80),
    IconSpec("ios/Icon-App-40x40@3x.png", 120),
    IconSpec("ios/Icon-App-60x60@2x.png", 120),
    IconSpec("ios/Icon-App-60x60@3x.png", 180),
    IconSpec("ios/Icon-App-76x76@1x.png", 76),
    IconSpec("ios/Icon-App-76x76@2x.png", 152),
    IconSpec("ios/Icon-App-83.5x83.5@2x.png", 167),
    IconSpec("ios/Icon-App-1024x1024@1x.png", 1024),
]

ANDROID_ICONS: list[IconSpec] = [
    # Play Store listing icon is 512x512. PWA icons are also 192/512.
    IconSpec("android/icon-192.png", 192),
    IconSpec("android/icon-512.png", 512),
    IconSpec("android/play-store-icon-512.png", 512),
]


def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Expected #RRGGBB")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _flatten_alpha(im: Image.Image, bg_rgb: tuple[int, int, int]) -> Image.Image:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        base = Image.new("RGB", im.size, bg_rgb)
        base.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        return base
    return im.convert("RGB")


def _center_crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side))


def _write_icons(im_rgb_square: Image.Image, out_dir: str, specs: Iterable[IconSpec]) -> None:
    for spec in specs:
        out_path = os.path.join(out_dir, spec.rel_path)
        _ensure_dir(os.path.dirname(out_path))
        resized = im_rgb_square.resize((spec.size, spec.size), resample=Image.LANCZOS)
        resized.save(out_path, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to square(ish) source PNG")
    parser.add_argument("--out", required=True, help="Output directory (e.g., public/icons)")
    parser.add_argument(
        "--bg",
        default="#0f0f1a",
        help="Background color used to flatten alpha (default: #0f0f1a)",
    )
    args = parser.parse_args()

    bg_rgb = _parse_hex_color(args.bg)

    src = args.src
    out_dir = args.out

    im = Image.open(src)
    im = _center_crop_square(im)
    im_rgb = _flatten_alpha(im, bg_rgb)

    _write_icons(im_rgb, out_dir, IOS_ICONS)
    _write_icons(im_rgb, out_dir, ANDROID_ICONS)

    # Convenience copies for the PWA manifest + iOS touch icon used by the web app
    _ensure_dir(out_dir)
    im_rgb.resize((192, 192), resample=Image.LANCZOS).save(os.path.join(out_dir, "icon-192.png"), format="PNG", optimize=True)
    im_rgb.resize((512, 512), resample=Image.LANCZOS).save(os.path.join(out_dir, "icon-512.png"), format="PNG", optimize=True)
    im_rgb.resize((180, 180), resample=Image.LANCZOS).save(os.path.join(out_dir, "apple-touch-icon.png"), format="PNG", optimize=True)

    print(f"Wrote iOS icons: {os.path.join(out_dir, 'ios')}")
    print(f"Wrote Android icons: {os.path.join(out_dir, 'android')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







