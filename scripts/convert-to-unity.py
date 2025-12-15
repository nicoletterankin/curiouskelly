import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VISEME_MAP = {
    "A": "viseme_aa",
    "B": "viseme_PP",
    "C": "viseme_CH",
    "D": "viseme_DD",
    "E": "viseme_E",
    "F": "viseme_FF",
    "G": "viseme_I",
    "H": "viseme_O",
    "X": "viseme_sil",
}


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _curve(frames: List[Dict[str, Any]], key: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for fr in frames:
        if key in fr:
            out.append({"time": float(fr["timestamp"]), "value": float(fr[key])})
    return out


def convert_motion_dir(motion_dir: Path, unity_out_dir: Path) -> Path:
    name = motion_dir.name

    viseme_file = motion_dir / f"{name}_visemes.json"
    face_file = motion_dir / f"{name}_face.json"
    pose_file = motion_dir / f"{name}_pose.json"

    visemes: List[Dict[str, Any]] = []
    if viseme_file.exists():
        rh = _load_json(viseme_file)
        for cue in rh.get("mouthCues", []):
            visemes.append(
                {
                    "time": float(cue["start"]),
                    "duration": float(cue["end"] - cue["start"]),
                    "viseme": VISEME_MAP.get(cue.get("value"), "viseme_sil"),
                }
            )

    face_data: Dict[str, Any] = {}
    face_frames: List[Dict[str, Any]] = []
    if face_file.exists():
        face_data = _load_json(face_file)
        face_frames = face_data.get("frames", []) or []

    pose_data: Dict[str, Any] = {}
    pose_frames: List[Dict[str, Any]] = []
    if pose_file.exists():
        pose_data = _load_json(pose_file)
        pose_frames = pose_data.get("frames", []) or []

    duration = float(face_data.get("duration", pose_data.get("duration", 10.0)) or 10.0)
    fps = float(face_data.get("fps", pose_data.get("fps", 30.0)) or 30.0)

    unity_data = {
        "clipName": name,
        "duration": duration,
        "fps": fps,
        "visemes": visemes,
        "expressions": face_frames,
        "pose": {
            "subset": pose_data.get("subset", []),
            "frames": pose_frames,
        }
        if pose_file.exists()
        else None,
        "curves": {
            "mouthOpen": _curve(face_frames, "mouthOpen"),
            "smile": _curve(face_frames, "smile"),
            "leftBrowRaise": _curve(face_frames, "leftBrowRaise"),
            "rightBrowRaise": _curve(face_frames, "rightBrowRaise"),
            "leftEyeOpen": _curve(face_frames, "leftEyeOpen"),
            "rightEyeOpen": _curve(face_frames, "rightEyeOpen"),
            "mouthWidth": _curve(face_frames, "mouthWidth"),
            "headYaw": _curve(face_frames, "headYaw"),
            "headPitch": _curve(face_frames, "headPitch"),
            "headRoll": _curve(face_frames, "headRoll"),
        },
    }

    unity_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = unity_out_dir / f"{name}_unity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(unity_data, f, indent=2)

    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extracted motion data to Unity-ready JSON.")
    parser.add_argument("--motion-dir", default="./local/motion-data", help="Input dir (default: ./local/motion-data)")
    parser.add_argument(
        "--output-dir", default="./local/unity-animations", help="Output dir (default: ./local/unity-animations)"
    )
    args = parser.parse_args()

    motion_root = Path(args.motion_dir)
    out_root = Path(args.output_dir)

    motion_dirs = [d for d in motion_root.iterdir() if d.is_dir()] if motion_root.exists() else []
    print(f"Converting {len(motion_dirs)} motion set(s) -> {out_root}")

    converted = 0
    for d in motion_dirs:
        out = convert_motion_dir(d, out_root)
        converted += 1
        print(f"[OK] {d.name} -> {out.name}")

    print(f"\nUnity animations saved to: {out_root} (converted {converted})")


if __name__ == "__main__":
    main()


