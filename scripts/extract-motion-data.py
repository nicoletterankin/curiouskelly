import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _ensure_tensorflow_doc_controls_stub() -> None:
    """
    MediaPipe's Tasks layer may import tensorflow.tools.docs.doc_controls for docs decorators.
    In environments where TensorFlow is installed but incompatible with our protobuf pin,
    importing tensorflow can fail and prevent MediaPipe from importing at all.

    We don't need TensorFlow for FaceMesh/Pose extraction, so we provide a tiny stub that
    satisfies the optional import.
    """

    if "tensorflow" in sys.modules:
        return

    tf = types.ModuleType("tensorflow")
    tools = types.ModuleType("tensorflow.tools")
    docs = types.ModuleType("tensorflow.tools.docs")
    doc_controls = types.ModuleType("tensorflow.tools.docs.doc_controls")

    # MediaPipe uses these as decorators. Implement as no-ops.
    def _noop_decorator(fn=None, **_kwargs):
        if fn is None:
            return lambda x: x
        return fn

    doc_controls.do_not_generate_docs = _noop_decorator
    doc_controls.do_not_doc_inheritable = _noop_decorator
    doc_controls.do_not_generate_docs_inheritable = _noop_decorator

    docs.doc_controls = doc_controls
    tools.docs = docs
    tf.tools = tools

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.tools"] = tools
    sys.modules["tensorflow.tools.docs"] = docs
    sys.modules["tensorflow.tools.docs.doc_controls"] = doc_controls


_ensure_tensorflow_doc_controls_stub()

# Delayed until stub is in place
import mediapipe as mp  # noqa: E402


@dataclass
class Paths:
    ffmpeg_exe: str
    rhubarb_path: Path
    video_dir: Path
    output_dir: Path


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resolve_ffmpeg_exe() -> str:
    """
    Prefer:
    - FFMPEG_PATH env var
    - ffmpeg on PATH
    - WinGet package location (best-effort)
    """
    env = os.environ.get("FFMPEG_PATH")
    if env:
        return env

    which = shutil.which("ffmpeg")
    if which:
        return which

    # WinGet commonly installs here (we pick the first match)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates = glob.glob(
            os.path.join(local, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "bin", "ffmpeg.exe"),
            recursive=True,
        )
        if candidates:
            return candidates[0]

    return "ffmpeg"


def extract_audio(ffmpeg_exe: str, video_path: Path, output_wav_path: Path) -> None:
    """
    Extract audio to mono 16k WAV for Rhubarb.
    """
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        str(output_wav_path),
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path.name}:\n{result.stderr}")


def extract_visemes(rhubarb_exe: Path, audio_path: Path, output_json_path: Path) -> Tuple[bool, str]:
    """
    Run Rhubarb to get viseme timeline. Returns (ok, error_message).
    """
    cmd = [str(rhubarb_exe), str(audio_path), "-f", "json", "-o", str(output_json_path)]
    result = _run(cmd)
    if result.returncode == 0 and output_json_path.exists():
        return True, ""
    return False, result.stderr.strip() or "Unknown Rhubarb error"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute_expression_values(landmarks) -> Dict[str, float]:
    """
    Extremely lightweight heuristic blend values from MediaPipe FaceMesh landmarks.
    Outputs are normalized 0..1.
    """
    # Mouth
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # Eyes
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    # Brows
    left_brow = landmarks[66]
    right_brow = landmarks[296]
    left_eye_inner = landmarks[133]
    right_eye_inner = landmarks[362]

    mouth_open = abs(upper_lip.y - lower_lip.y) * 10.0
    mouth_width = abs(left_mouth.x - right_mouth.x) * 5.0

    left_eye_open = abs(left_eye_top.y - left_eye_bottom.y) * 20.0
    right_eye_open = abs(right_eye_top.y - right_eye_bottom.y) * 20.0

    left_brow_raise = (left_eye_inner.y - left_brow.y) * 15.0
    right_brow_raise = (right_eye_inner.y - right_brow.y) * 15.0

    mouth_center_y = (upper_lip.y + lower_lip.y) / 2.0
    smile = ((mouth_center_y - left_mouth.y) + (mouth_center_y - right_mouth.y)) * 10.0

    return {
        "mouthOpen": _clip01(mouth_open),
        "mouthWidth": _clip01(mouth_width),
        "leftEyeOpen": _clip01(left_eye_open),
        "rightEyeOpen": _clip01(right_eye_open),
        "leftBrowRaise": _clip01(left_brow_raise),
        "rightBrowRaise": _clip01(right_brow_raise),
        "smile": _clip01(smile),
    }


def compute_head_rotation(landmarks) -> Dict[str, float]:
    """
    Rough head rotation estimate (degrees). This is a heuristic, not a calibrated solvePnP.
    """
    nose = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    yaw = (nose.x - 0.5) * 90.0

    face_height = abs(forehead.y - chin.y)
    nose_relative = (nose.y - forehead.y) / face_height if face_height > 1e-6 else 0.5
    pitch = (nose_relative - 0.4) * 60.0

    roll = (left_ear.y - right_ear.y) * 45.0

    return {"headYaw": float(yaw), "headPitch": float(pitch), "headRoll": float(roll)}


POSE_KEYS = {
    "nose": mp.solutions.pose.PoseLandmark.NOSE,
    "left_shoulder": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
}


def _extract_pose_subset(pose_landmarks) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, idx in POSE_KEYS.items():
        lm = pose_landmarks.landmark[idx]
        out[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "v": float(lm.visibility),
        }
    return out


def extract_face_and_pose_data(
    video_path: Path,
    face_output_path: Path,
    pose_output_path: Optional[Path],
    include_pose: bool,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract per-frame facial blend-ish values + (optional) small pose subset.
    Writes:
      - <name>_face.json (always)
      - <name>_pose.json (if include_pose)
    Returns summary dict used for end-of-run report.
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose = None
    if include_pose:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    face_frames: List[Dict[str, Any]] = []
    pose_frames: List[Dict[str, Any]] = []

    frame_idx = 0
    processed = 0
    face_detected = 0
    pose_detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames is not None and frame_idx >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        t = frame_idx / float(fps)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            face_frames.append(
                {
                    "frame": frame_idx,
                    "timestamp": t,
                    **compute_expression_values(landmarks),
                    **compute_head_rotation(landmarks),
                }
            )
            face_detected += 1

        if include_pose and pose is not None:
            pose_results = pose.process(rgb)
            if pose_results.pose_landmarks:
                pose_frames.append(
                    {
                        "frame": frame_idx,
                        "timestamp": t,
                        "landmarks": _extract_pose_subset(pose_results.pose_landmarks),
                    }
                )
                pose_detected += 1

        processed += 1
        frame_idx += 1

        if processed % 60 == 0 and total_frames:
            print(f"    Processing frame {processed}/{total_frames}", end="\r")

    cap.release()
    face_mesh.close()
    if pose is not None:
        pose.close()

    face_payload = {
        "fps": float(fps),
        "totalFrames": int(total_frames),
        "duration": float((total_frames / float(fps)) if fps else 0.0),
        "frames": face_frames,
    }

    with open(face_output_path, "w", encoding="utf-8") as f:
        json.dump(face_payload, f, indent=2)

    if include_pose and pose_output_path is not None:
        pose_payload = {
            "fps": float(fps),
            "totalFrames": int(total_frames),
            "duration": float((total_frames / float(fps)) if fps else 0.0),
            "frames": pose_frames,
            "subset": list(POSE_KEYS.keys()),
        }
        with open(pose_output_path, "w", encoding="utf-8") as f:
            json.dump(pose_payload, f, indent=2)

    print(
        f"  [FACE] Frames: {len(face_frames)} (detected {face_detected}/{processed}) | "
        f"[POSE] Frames: {len(pose_frames) if include_pose else 0} "
        f"(detected {pose_detected}/{processed})"
    )

    return {
        "fps": float(fps),
        "frames_processed": processed,
        "face_frames": len(face_frames),
        "pose_frames": len(pose_frames),
        "face_output": str(face_output_path),
        "pose_output": str(pose_output_path) if include_pose and pose_output_path else None,
    }


def process_video(paths: Paths, video_path: Path, include_pose: bool, max_frames: Optional[int]) -> Dict[str, Any]:
    name = video_path.stem
    print(f"\n[VIDEO] Processing: {name}")

    video_output_dir = paths.output_dir / name
    _ensure_dir(video_output_dir)

    # 1) Audio
    audio_path = video_output_dir / f"{name}.wav"
    try:
        extract_audio(paths.ffmpeg_exe, video_path, audio_path)
        print(f"  [AUDIO] Extracted: {audio_path.name}")
    except Exception as e:
        print(f"  [ERROR] Audio extraction failed: {e}")

    # 2) Visemes
    viseme_ok = False
    viseme_err = ""
    viseme_path = video_output_dir / f"{name}_visemes.json"
    if audio_path.exists():
        if not paths.rhubarb_path.exists():
            viseme_ok = False
            viseme_err = f"Rhubarb not found at: {paths.rhubarb_path}"
            print(f"  [WARN] Viseme extraction skipped: {viseme_err}")
        else:
            viseme_ok, viseme_err = extract_visemes(paths.rhubarb_path, audio_path, viseme_path)
            if viseme_ok:
                print(f"  [VISEMES] Extracted: {viseme_path.name}")
            else:
                print(f"  [ERROR] Viseme extraction failed: {viseme_err}")

    # 3) Face + Pose
    face_path = video_output_dir / f"{name}_face.json"
    pose_path = video_output_dir / f"{name}_pose.json" if include_pose else None
    summary = extract_face_and_pose_data(
        video_path=video_path,
        face_output_path=face_path,
        pose_output_path=pose_path,
        include_pose=include_pose,
        max_frames=max_frames,
    )

    return {
        "video": str(video_path),
        "name": name,
        "audio": str(audio_path) if audio_path.exists() else None,
        "visemes": str(viseme_path) if viseme_ok and viseme_path.exists() else None,
        "visemes_ok": bool(viseme_ok),
        "visemes_error": viseme_err if not viseme_ok else "",
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract visemes + face + pose data from HeyGen videos.")
    parser.add_argument(
        "--video-dir",
        default="./local/heygen-videos",
        help="Directory containing input .mp4 files (default: ./local/heygen-videos)",
    )
    parser.add_argument(
        "--output-dir",
        default="./local/motion-data",
        help="Output directory (default: ./local/motion-data)",
    )
    parser.add_argument(
        "--rhubarb",
        default=os.environ.get("RHUBARB_PATH", r"C:\tools\rhubarb\rhubarb.exe"),
        help=r"Path to rhubarb.exe (default: RHUBARB_PATH env or C:\tools\rhubarb\rhubarb.exe)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=os.environ.get("FFMPEG_PATH", None),
        help="Path to ffmpeg executable (default: FFMPEG_PATH env, else auto-detect)",
    )
    parser.add_argument(
        "--one",
        default=None,
        help="Process a single video path (relative to --video-dir, or absolute).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .mp4 videos in --video-dir",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose extraction (FaceMesh only).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap processing to N frames (debug).",
    )
    args = parser.parse_args()

    rhubarb_path = Path(args.rhubarb)
    if not rhubarb_path.exists():
        # Auto-detect if user has extracted Rhubarb into C:\tools\rhubarb\<version>\rhubarb.exe
        candidates = glob.glob(r"C:\tools\rhubarb\**\rhubarb.exe", recursive=True)
        if candidates:
            rhubarb_path = Path(candidates[0])

    paths = Paths(
        ffmpeg_exe=str(args.ffmpeg) if args.ffmpeg else resolve_ffmpeg_exe(),
        rhubarb_path=rhubarb_path,
        video_dir=Path(args.video_dir),
        output_dir=Path(args.output_dir),
    )
    _ensure_dir(paths.output_dir)

    include_pose = not args.no_pose

    # Collect targets
    targets: List[Path] = []
    if args.one:
        one = str(args.one)

        # If user passes something like "day_001" (no extension), treat it as a prefix match.
        if not one.lower().endswith(".mp4") and ("/" not in one and "\\" not in one):
            # Prefer recursive search because downloads may be nested (e.g. day_001/scientist_adult/main/*.mp4)
            matches = [p for p in sorted(paths.video_dir.rglob("*.mp4")) if p.name.lower().startswith(one.lower())]
            if not matches:
                # Fallback: substring match
                matches = [p for p in sorted(paths.video_dir.rglob("*.mp4")) if one.lower() in p.as_posix().lower()]
            if matches:
                targets = [matches[0]]
            else:
                targets = [paths.video_dir / f"{one}.mp4"]
        else:
            p = Path(one)
            if not p.is_absolute():
                p = paths.video_dir / p
            targets = [p]
    elif args.all:
        targets = sorted(paths.video_dir.rglob("*.mp4"))
    else:
        # Default behavior: "ONE video first" (first mp4 found)
        candidates = sorted(paths.video_dir.rglob("*.mp4"))
        if candidates:
            targets = [candidates[0]]
        else:
            print(f"[ERROR] No .mp4 found in {paths.video_dir}")
            return

    print(f"Found {len(targets)} video(s) to process")
    print(f"Input:  {paths.video_dir}")
    print(f"Output: {paths.output_dir}")
    print(f"FFmpeg: {paths.ffmpeg_exe}")
    print(f"Rhubarb: {paths.rhubarb_path}")
    print(f"Pose:   {'ON' if include_pose else 'OFF'}")

    results: List[Dict[str, Any]] = []
    for i, vid in enumerate(targets, 1):
        if not vid.exists():
            print(f"\n[{i}/{len(targets)}] [ERROR] Missing: {vid}")
            continue
        print(f"\n[{i}/{len(targets)}]", end="")
        results.append(process_video(paths, vid, include_pose=include_pose, max_frames=args.max_frames))

    # Write a simple run report
    report_path = paths.output_dir / "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    processed = len(results)
    viseme_files = sum(1 for r in results if r.get("visemes"))
    face_files = sum(1 for r in results if r.get("face_output") and Path(r["face_output"]).exists())
    pose_files = sum(1 for r in results if r.get("pose_output") and Path(r["pose_output"]).exists())

    print("\n" + "=" * 72)
    print("EXTRACTION REPORT")
    print("=" * 72)
    print(f"Videos processed: {processed}")
    print(f"Viseme files:     {viseme_files}")
    print(f"Face files:       {face_files}")
    print(f"Pose files:       {pose_files}")
    print(f"Report:           {report_path}")


if __name__ == "__main__":
    main()


