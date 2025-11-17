#!/usr/bin/env python3
"""
Generate Kelly clip audio files from assets/kelly_clips/v1/scripts.json.

Output WAV files land in assets/kelly_clips/v1/audio/, ready for iClone AccuLips.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_FILE = REPO_ROOT / "assets" / "kelly_clips" / "v1" / "scripts.json"
AUDIO_DIR = REPO_ROOT / "assets" / "kelly_clips" / "v1" / "audio"

DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE = "wAdymQH5YucAkXwmrdL0"  # Kelly voice used across the repo


def load_clips() -> List[Dict]:
    if not SCRIPTS_FILE.exists():
        raise FileNotFoundError(f"Missing scripts file: {SCRIPTS_FILE}")
    data = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not clips:
        raise ValueError(f"No clips found in {SCRIPTS_FILE}")
    return clips


def ensure_audio_dir() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def elevenlabs_request(text: str, clip_id: str, fmt: str, dry_run: bool = False) -> Path:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise EnvironmentError("ELEVENLABS_API_KEY not set in environment or .env.local")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
    model_id = os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL)

    headers = {
        "Accept": f"audio/{fmt}",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": float(os.getenv("ELEVENLABS_STABILITY", 0.58)),
            "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY", 0.72)),
            "style": float(os.getenv("ELEVENLABS_STYLE", 0.0)),
            "use_speaker_boost": os.getenv("ELEVENLABS_SPEAKER_BOOST", "true").lower()
            != "false",
        },
    }

    extension = "wav" if fmt == "wav" else "mp3"
    output_path = AUDIO_DIR / f"{clip_id}.{extension}"

    if dry_run:
        print(f"[dry-run] Would request {clip_id} -> {output_path.name}")
        return output_path

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs error ({response.status_code}) for {clip_id}: {response.text[:200]}"
        )

    output_path.write_bytes(response.content)
    return output_path


def iter_target_clips(all_clips: List[Dict], include: Iterable[str] | None) -> List[Dict]:
    if not include:
        return all_clips
    wanted = set(include)
    missing = wanted - {clip["id"] for clip in all_clips}
    if missing:
        raise ValueError(f"Unknown clip ids: {', '.join(sorted(missing))}")
    return [clip for clip in all_clips if clip["id"] in wanted]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        help="Generate audio for a specific clip ID (default: all clips listed in scripts.json)",
    )
    parser.add_argument(
        "--format",
        choices=("wav", "mpeg"),
        default="wav",
        help="Audio format to request from ElevenLabs (default: wav)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without calling the API")
    args = parser.parse_args(argv)

    clips = load_clips()
    targets = iter_target_clips(clips, args.clip)
    ensure_audio_dir()

    print(f"🎬 Generating {len(targets)} clip{'s' if len(targets) != 1 else ''} using ElevenLabs")
    print(f"📁 Output directory: {AUDIO_DIR}")
    completed = []
    for clip in targets:
        clip_id = clip["id"]
        text = clip["script"]
        print(f"\n→ {clip_id}: {text[:70]}{'…' if len(text) > 70 else ''}")
        try:
            output_path = elevenlabs_request(text, clip_id, args.format, args.dry_run)
            completed.append(output_path)
            print(f"   ✅ Saved {output_path.name}")
        except Exception as exc:
            print(f"   ❌ Failed: {exc}")

    print("\nSummary")
    print("-------")
    print(f"Generated {len([p for p in completed if not args.dry_run])} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




