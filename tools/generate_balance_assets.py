#!/usr/bin/env python3
"""
generate_balance_assets.py

Utility that ingests lesson-player/balance-visual-prompts.json and
produces both on-disk asset scaffolding plus a machine-readable manifest
(`balance-visual-assets.json`) that downstream systems (Unity, HTML player,
CDN uploaders) can consume.

The script supports two modes:
1. --simulate (default): creates placeholder files that contain the prompt
   text. Use this when you want to preview the directory layout or hand off
   prompts to a manual artist/AI workflow.
2. --invoke-provider vertx|google : wires in the actual generation pipeline.
   Hooks are stubbed below—plug in the real SDK once credentials are available.

Example usage:
  python tools/generate_balance_assets.py ^
      --prompts lesson-player/balance-visual-prompts.json ^
      --output lesson-player/balance-visual-assets.json ^
      --asset-root digital-kelly/content/balance ^
      --cdn-base https://assets.curiouskelly.com/lessons/balance ^
      --batch avatars diagrams
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CATEGORY_KEYS = {
    "kelly_avatars": ("avatars", "glb"),
    "animations": ("animations", "glb"),
    "diagrams": ("diagrams", "svg"),
    "backgrounds": ("backgrounds", "webp"),
    "interactive_elements": ("interactive", "html"),
    "ui_elements": ("ui", "svg"),
    "supporting_visuals": ("diagrams", "webp"),
    "mathematical_visuals": ("diagrams", "svg"),
    "physics_demonstrations": ("animations", "mp4"),
    "audio_sync": ("sync", "json"),
    "expression_cues": ("sync", "json"),
    "emotion_expressions": ("sync", "json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balance lesson assets/manifest.")
    parser.add_argument("--prompts", default="lesson-player/balance-visual-prompts.json")
    parser.add_argument("--output", default="lessons/manifests/balance-visual-assets.json")
    parser.add_argument("--asset-root", default="digital-kelly/content/balance")
    parser.add_argument("--cdn-base", default="https://assets.curiouskelly.com/lessons/balance")
    parser.add_argument(
        "--batch",
        nargs="*",
        help="Optional list of categories to process (e.g., avatars diagrams ui). "
        "Defaults to every category in the prompts file.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Create placeholder files instead of calling the provider.",
    )
    parser.add_argument(
        "--invoke-provider",
        choices=["vertx", "google"],
        help="Choose a provider integration. Requires credentials; currently stubbed.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> Dict[str, List[dict]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_cdn_url(cdn_base: str, rel_path: str) -> str:
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"{cdn_base.rstrip('/')}/{rel}"


def simulate_file(path: Path, payload: dict) -> None:
    if path.exists():
        return
    contents = [
        f"ID: {payload.get('id')}",
        f"Type: {payload.get('type')}",
        f"Prompt: {payload.get('prompt')}",
        "",
        "This is a placeholder file. Replace with generated asset.",
    ]
    path.write_text("\n".join(contents), encoding="utf-8")


def call_provider(provider: str, payload: dict, destination: Path) -> None:
    raise NotImplementedError(
        f"Provider '{provider}' integration not implemented. "
        "Add API client logic here once credentials are available."
    )


def derive_category_key(key: str) -> Optional[tuple[str, str]]:
    return CATEGORY_KEYS.get(key)


def flatten_entries(raw: Dict[str, List[dict]], batch: Optional[Iterable[str]]) -> List[tuple[str, dict]]:
    entries: List[tuple[str, dict]] = []
    include_keys = set(batch) if batch else set(raw.keys())
    for key, items in raw.items():
        if not isinstance(items, list):
            continue
        if key not in include_keys:
            continue
        entries.extend((key, item) for item in items)
    return entries


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    asset_root = Path(args.asset_root)
    ensure_dir(asset_root)

    prompts = load_prompts(prompts_path)
    entries = flatten_entries(prompts, args.batch)
    manifest: List[dict] = []

    for category_key, payload in entries:
        mapping = derive_category_key(category_key)
        if not mapping:
            print(f"[WARN] Unknown category '{category_key}' – skipping entry {payload.get('id')}")
            continue
        subdir, default_ext = mapping
        asset_id = payload.get("id")
        ext = payload.get("ext", default_ext)
        rel_path = f"{subdir}/{asset_id}.{ext}".replace("//", "/")
        dest_path = asset_root / rel_path
        ensure_dir(dest_path.parent)

        if args.simulate or not args.invoke_provider:
            simulate_file(dest_path, payload)
            status = "placeholder"
        else:
            call_provider(args.invoke_provider, payload, dest_path)
            status = "generated"

        manifest.append(
            {
                "id": asset_id,
                "type": payload.get("type"),
                "category": category_key,
                "ageBuckets": payload.get("ageBuckets"),
                "prompt": payload.get("prompt"),
                "style": payload.get("style"),
                "technical": payload.get("technical"),
                "localPath": str(dest_path.as_posix()),
                "cdnPath": build_cdn_url(args.cdn_base, rel_path),
                "status": status,
            }
        )

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"assets": manifest}, fp, indent=2)
        fp.write("\n")

    print(f"[done] Processed {len(manifest)} assets. Manifest written to {output_path}")


if __name__ == "__main__":
    main()

