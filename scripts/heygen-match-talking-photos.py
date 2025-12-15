#!/usr/bin/env python3
"""
Match HeyGen talking photos to local Kelly persona images.

Problem:
- HeyGen creation endpoints may be unavailable for this API key/account.
- We still can *list* talking photos and use their preview images.

This script:
- Computes a simple perceptual hash (average hash / aHash) for:
  1) Local persona head images (e.g. generated-images/kelly-archetypes-head-only/age/mature/*.png)
  2) HeyGen talking photo preview images (from generated-images/kelly-talking-photos.json)
- Greedily matches local images to HeyGen talking photos by minimum Hamming distance.
- Writes a mapping file: { archetype: talking_photo_id | null }

Usage:
  python scripts/heygen-match-talking-photos.py --age mature
  python scripts/heygen-match-talking-photos.py --age elder

Notes:
- Requires Pillow (PIL). If missing, install it (it's already pinned in tools/social-media-automation/requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen, Request

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Install with: pip install pillow\n"
        "Tip: it's pinned in tools/social-media-automation/requirements.txt"
    ) from e


ARCHETYPES = [
    "scientist",
    "explorer",
    "rebel",
    "architect",
    "diplomat",
    "empath",
    "macgyver",
    "mystic",
    "provider",
    "storyteller",
    "strategist",
    "survivor",
]


def ahash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Average hash (aHash), returned as an integer bitset.
    We crop to the upper ~60% first to emphasize head accessories.
    """
    w, h = image.size
    crop_h = int(h * 0.6)
    cropped = image.crop((0, 0, w, crop_h))
    im = cropped.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(im.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_to_cache(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    headers = {
        "User-Agent": "UI-TARS/heygen-match-talking-photos",
        "Accept": "*/*",
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
    safe_mkdir(out_path.parent)
    out_path.write_bytes(data)


@dataclass(frozen=True)
class TalkingPhoto:
    id: str
    image_url: str


def load_talking_photos(path: Path) -> List[TalkingPhoto]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    photos: List[TalkingPhoto] = []
    for item in raw:
        pid = item.get("id") or item.get("talking_photo_id")
        url = item.get("image_url") or item.get("preview_image_url")
        if pid and url:
            photos.append(TalkingPhoto(id=str(pid), image_url=str(url)))
    return photos


def load_excluded_ids(paths: List[Path]) -> set[str]:
    excluded: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, str) and v:
                        excluded.add(v)
            elif isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        vid = item.get("id") or item.get("talking_photo_id")
                        if isinstance(vid, str) and vid:
                            excluded.add(vid)
        except Exception:
            continue
    return excluded


def local_image_paths(repo_root: Path, age: str) -> Dict[str, Path]:
    base = repo_root / "generated-images" / "kelly-archetypes-head-only" / "age" / age
    out: Dict[str, Path] = {}
    for arch in ARCHETYPES:
        p = base / f"kelly_{arch}_head.png"
        if p.exists():
            out[arch] = p
    return out


def compute_local_hashes(paths: Dict[str, Path]) -> Dict[str, int]:
    hashes: Dict[str, int] = {}
    for arch, p in paths.items():
        with Image.open(p) as im:
            hashes[arch] = ahash(im)
    return hashes


def compute_talking_photo_hashes(photos: List[TalkingPhoto], cache_dir: Path) -> Dict[str, int]:
    hashes: Dict[str, int] = {}
    for tp in photos:
        # Some URLs include query strings; cache by ID as stable key
        cache_path = cache_dir / f"{tp.id}.img"
        download_to_cache(tp.image_url, cache_path)
        with Image.open(cache_path) as im:
            hashes[tp.id] = ahash(im)
    return hashes


def greedy_match(
    local_hashes: Dict[str, int],
    tp_hashes: Dict[str, int],
    max_distance: int,
) -> Tuple[Dict[str, str | None], List[Tuple[str, str, int]]]:
    """
    Greedy global matching by smallest distance first.
    Returns mapping and a list of chosen assignments (arch, id, distance).
    """
    distances: List[Tuple[int, str, str]] = []
    for arch, h in local_hashes.items():
        for tp_id, tp_h in tp_hashes.items():
            d = hamming_distance(h, tp_h)
            distances.append((d, arch, tp_id))

    distances.sort(key=lambda x: x[0])

    assigned_arch = set()
    assigned_tp = set()
    chosen: List[Tuple[str, str, int]] = []
    mapping: Dict[str, str | None] = {a: None for a in local_hashes.keys()}

    for d, arch, tp_id in distances:
        if d > max_distance:
            break
        if arch in assigned_arch or tp_id in assigned_tp:
            continue
        assigned_arch.add(arch)
        assigned_tp.add(tp_id)
        mapping[arch] = tp_id
        chosen.append((arch, tp_id, d))

        if len(assigned_arch) == len(local_hashes):
            break

    return mapping, chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", choices=["mature", "elder"], required=True)
    parser.add_argument(
        "--talking-photos",
        default="generated-images/kelly-talking-photos.json",
        help="Path to JSON list of talking photos (id + image_url).",
    )
    parser.add_argument(
        "--cache-dir",
        default="generated-images/_heygen_talking_photo_cache",
        help="Where to cache downloaded preview images.",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=8,
        help="Max Hamming distance to accept as a match (0-64 for 8x8 aHash).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output mapping file path. Default: generated-images/.../age/<age>/heygen_talking_photo_ids.json",
    )
    parser.add_argument(
        "--exclude-map",
        action="append",
        default=[],
        help="Path(s) to JSON mapping files whose IDs should be excluded from matching (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    talking_photos_path = (repo_root / args.talking_photos).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()

    if not talking_photos_path.exists():
        raise SystemExit(f"Talking photos file not found: {talking_photos_path}")

    local_paths = local_image_paths(repo_root, args.age)
    if len(local_paths) == 0:
        raise SystemExit(f"No local images found for age={args.age} in generated-images/kelly-archetypes-head-only/age/{args.age}")

    photos = load_talking_photos(talking_photos_path)
    if len(photos) == 0:
        raise SystemExit("No talking photos found in JSON (expected list of {id,image_url}).")

    print(f"Local images: {len(local_paths)}")
    print(f"HeyGen talking photos: {len(photos)}")
    print(f"Cache dir: {cache_dir}")

    local_hashes = compute_local_hashes(local_paths)
    tp_hashes_all = compute_talking_photo_hashes(photos, cache_dir)

    excluded_paths = [(repo_root / p).resolve() for p in (args.exclude_map or [])]
    excluded_ids = load_excluded_ids(excluded_paths)
    if excluded_ids:
        print(f"Excluded IDs: {len(excluded_ids)}")
    tp_hashes = {k: v for (k, v) in tp_hashes_all.items() if k not in excluded_ids}
    print(f"Candidate talking photos after exclusions: {len(tp_hashes)}")

    # Always show the best candidate per archetype (useful when no exact match exists).
    print("\nBest candidate per archetype (min distance across all talking photos):")
    for arch, h in sorted(local_hashes.items(), key=lambda kv: kv[0]):
        best_id = None
        best_d = 10**9
        for tp_id, tp_h in tp_hashes.items():
            d = hamming_distance(h, tp_h)
            if d < best_d:
                best_d = d
                best_id = tp_id
        print(f"  - {arch:12s} best_distance={best_d} best_id={best_id}")

    mapping, chosen = greedy_match(local_hashes, tp_hashes, max_distance=args.max_distance)
    chosen_sorted = sorted(chosen, key=lambda x: x[0])

    print("\nMatches (best effort):")
    for arch, tp_id, dist in chosen_sorted:
        print(f"  - {arch:12s} -> {tp_id} (distance={dist})")

    missing = [a for a, v in mapping.items() if v is None]
    if missing:
        print("\nUnmatched archetypes (not found within threshold):")
        for a in missing:
            print(f"  - {a}")

    out_path = Path(args.out) if args.out else (repo_root / "generated-images" / "kelly-archetypes-head-only" / "age" / args.age / "heygen_talking_photo_ids.json")
    out_path = out_path.resolve()

    if args.dry_run:
        print(f"\nDRY RUN: would write {out_path}")
        return

    safe_mkdir(out_path.parent)
    out_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()




