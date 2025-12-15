import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


FEATURES = [
    "mouthOpen",
    "mouthWidth",
    "smile",
    "leftEyeOpen",
    "rightEyeOpen",
    "leftBrowRaise",
    "rightBrowRaise",
    "headYaw",
    "headPitch",
    "headRoll",
]


@dataclass
class ClipStats:
    name: str
    fps: float
    duration: float
    n_frames: int


def _percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _summarize_feature(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": _percentile(values, 5),
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p95": _percentile(values, 95),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def _head_movement_signature(frames: List[dict], fps: float) -> Dict[str, float]:
    """
    Simple movement stats:
    - per-axis range (p95-p05)
    - per-axis avg angular speed (deg/s) based on finite differences
    """
    if not frames or fps <= 0:
        return {"frames": 0}

    yaw = np.array([f.get("headYaw", 0.0) for f in frames], dtype=np.float32)
    pitch = np.array([f.get("headPitch", 0.0) for f in frames], dtype=np.float32)
    roll = np.array([f.get("headRoll", 0.0) for f in frames], dtype=np.float32)

    dt = 1.0 / float(fps)
    # deg/s (absolute)
    yaw_speed = np.abs(np.diff(yaw) / dt) if yaw.size > 1 else np.array([], dtype=np.float32)
    pitch_speed = np.abs(np.diff(pitch) / dt) if pitch.size > 1 else np.array([], dtype=np.float32)
    roll_speed = np.abs(np.diff(roll) / dt) if roll.size > 1 else np.array([], dtype=np.float32)

    return {
        "frames": int(len(frames)),
        "yaw_range_p95_p05": float(np.percentile(yaw, 95) - np.percentile(yaw, 5)),
        "pitch_range_p95_p05": float(np.percentile(pitch, 95) - np.percentile(pitch, 5)),
        "roll_range_p95_p05": float(np.percentile(roll, 95) - np.percentile(roll, 5)),
        "yaw_speed_mean_deg_s": float(yaw_speed.mean()) if yaw_speed.size else 0.0,
        "pitch_speed_mean_deg_s": float(pitch_speed.mean()) if pitch_speed.size else 0.0,
        "roll_speed_mean_deg_s": float(roll_speed.mean()) if roll_speed.size else 0.0,
        "yaw_speed_p95_deg_s": float(np.percentile(yaw_speed, 95)) if yaw_speed.size else 0.0,
        "pitch_speed_p95_deg_s": float(np.percentile(pitch_speed, 95)) if pitch_speed.size else 0.0,
        "roll_speed_p95_deg_s": float(np.percentile(roll_speed, 95)) if roll_speed.size else 0.0,
    }


def _gesture_signature(pose_frames: List[dict], fps: float) -> Dict[str, float]:
    """
    Uses the extracted pose subset (wrists only) to get a rough "gesture activity" score.
    - wrist_speed_mean (normalized units per second)
    - gesture_peak_rate: peaks per minute above a threshold
    """
    if not pose_frames or fps <= 0:
        return {"frames": 0}

    dt = 1.0 / float(fps)

    def _wrist_xy(name: str) -> np.ndarray:
        pts = []
        for fr in pose_frames:
            lm = (fr.get("landmarks") or {}).get(name) or {}
            pts.append((float(lm.get("x", 0.0)), float(lm.get("y", 0.0))))
        return np.array(pts, dtype=np.float32)

    lw = _wrist_xy("left_wrist")
    rw = _wrist_xy("right_wrist")

    def _speed(pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] < 2:
            return np.array([], dtype=np.float32)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return d / dt

    lw_s = _speed(lw)
    rw_s = _speed(rw)
    speed = np.maximum(lw_s, rw_s) if (lw_s.size and rw_s.size) else (lw_s if lw_s.size else rw_s)

    if speed.size == 0:
        return {"frames": int(len(pose_frames))}

    # Threshold chosen empirically for normalized coords; tune later.
    thresh = float(np.percentile(speed, 95)) * 0.6
    peaks = 0
    for i in range(1, speed.size - 1):
        if speed[i] > thresh and speed[i] > speed[i - 1] and speed[i] > speed[i + 1]:
            peaks += 1

    duration_s = (len(pose_frames) / float(fps)) if fps else 0.0
    peaks_per_min = (peaks / duration_s) * 60.0 if duration_s > 1e-6 else 0.0

    return {
        "frames": int(len(pose_frames)),
        "wrist_speed_mean": float(speed.mean()),
        "wrist_speed_p95": float(np.percentile(speed, 95)),
        "gesture_peaks": int(peaks),
        "gesture_peaks_per_min": float(peaks_per_min),
        "threshold_used": float(thresh),
    }


def main() -> None:
    motion_dir = Path("local/motion-data")
    out_path = motion_dir / "scientist_profile.json"

    face_files = sorted(motion_dir.glob("*/*_face.json"))
    if not face_files:
        print("[ERROR] No face files found under local/motion-data/*/*_face.json")
        return

    all_values: Dict[str, List[float]] = {k: [] for k in FEATURES}
    clip_summaries: List[ClipStats] = []
    head_sigs: List[dict] = []
    gesture_sigs: List[dict] = []

    for face_file in face_files:
        clip_name = face_file.parent.name
        face = json.loads(face_file.read_text(encoding="utf-8"))
        fps = float(face.get("fps") or 0.0)
        duration = float(face.get("duration") or 0.0)
        frames = face.get("frames") or []
        n = int(len(frames))

        clip_summaries.append(ClipStats(name=clip_name, fps=fps, duration=duration, n_frames=n))

        for f in frames:
            for k in FEATURES:
                if k in f:
                    all_values[k].append(float(f[k]))

        head_sigs.append({"clip": clip_name, **_head_movement_signature(frames, fps)})

        pose_file = face_file.with_name(f"{clip_name}_pose.json")
        if pose_file.exists():
            pose = json.loads(pose_file.read_text(encoding="utf-8"))
            pose_frames = pose.get("frames") or []
            gesture_sigs.append({"clip": clip_name, **_gesture_signature(pose_frames, fps)})

    summary = {
        "scope": {
            "persona": "scientist_adult",
            "clips": len(clip_summaries),
            "total_face_frames": int(sum(c.n_frames for c in clip_summaries)),
        },
        "clips": [
            {"clip": c.name, "fps": c.fps, "duration": c.duration, "frames": c.n_frames} for c in clip_summaries
        ],
        "expressions": {k: _summarize_feature(np.array(v, dtype=np.float32)) for k, v in all_values.items()},
        "head_movement": {
            "aggregate": {
                "yaw_range_p95_p05_mean": float(np.mean([h["yaw_range_p95_p05"] for h in head_sigs])),
                "pitch_range_p95_p05_mean": float(np.mean([h["pitch_range_p95_p05"] for h in head_sigs])),
                "roll_range_p95_p05_mean": float(np.mean([h["roll_range_p95_p05"] for h in head_sigs])),
                "yaw_speed_mean_deg_s_mean": float(np.mean([h["yaw_speed_mean_deg_s"] for h in head_sigs])),
                "pitch_speed_mean_deg_s_mean": float(np.mean([h["pitch_speed_mean_deg_s"] for h in head_sigs])),
                "roll_speed_mean_deg_s_mean": float(np.mean([h["roll_speed_mean_deg_s"] for h in head_sigs])),
            },
            "per_clip": head_sigs,
        },
        "gesture_activity": {
            "aggregate": {
                "gesture_peaks_per_min_mean": float(np.mean([g.get("gesture_peaks_per_min", 0.0) for g in gesture_sigs]))
                if gesture_sigs
                else 0.0,
                "wrist_speed_mean_mean": float(np.mean([g.get("wrist_speed_mean", 0.0) for g in gesture_sigs]))
                if gesture_sigs
                else 0.0,
            },
            "per_clip": gesture_sigs,
        },
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Scientist profile written:", out_path)
    print("Clips:", summary["scope"]["clips"])
    print("Total face frames:", summary["scope"]["total_face_frames"])
    print("Head movement (avg p95-p05 ranges):", summary["head_movement"]["aggregate"])
    print("Gesture peaks/min (avg):", summary["gesture_activity"]["aggregate"].get("gesture_peaks_per_min_mean", 0.0))


if __name__ == "__main__":
    main()


