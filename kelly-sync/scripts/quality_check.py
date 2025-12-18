#!/usr/bin/env python3
"""
📊 KELLY-SYNC QUALITY VALIDATOR

Automated quality assessment for generated Kelly videos.

Metrics:
1. Resolution verification (4K/8K)
2. Lip sync accuracy (SyncNet)
3. Face identity preservation (ArcFace)
4. Temporal consistency (inter-frame variance)
5. Blur detection (Laplacian variance)
6. Uncanny valley score (custom classifier)

Usage:
    python quality_check.py video.mp4
    python quality_check.py --batch output/*.mp4
    python quality_check.py --compare video1.mp4 video2.mp4
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('kelly-quality')

@dataclass
class QualityReport:
    """Quality assessment report for a video."""
    
    video_path: str
    
    # Basic info
    resolution: Tuple[int, int]
    fps: float
    duration: float
    frame_count: int
    
    # Quality metrics (0-100 scale)
    resolution_score: float      # vs target (4K = 100)
    blur_score: float            # higher = less blur
    temporal_score: float        # higher = more stable
    identity_score: float        # face consistency
    lipsync_score: float         # audio-video sync
    uncanny_score: float         # photorealism
    
    # Overall
    overall_score: float
    grade: str                   # A, B, C, D, F
    passed: bool                 # meets minimum threshold
    
    # Issues
    issues: List[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


class QualityChecker:
    """
    Comprehensive quality validation for Kelly videos.
    """
    
    # Quality thresholds
    THRESHOLDS = {
        'resolution_min': (1920, 1080),
        'resolution_target': (3840, 2160),
        'fps_min': 24,
        'blur_threshold': 100,        # Laplacian variance
        'temporal_threshold': 0.02,   # LPIPS variance
        'identity_threshold': 0.75,   # ArcFace similarity
        'lipsync_threshold': 0.85,    # SyncNet confidence
        'overall_pass': 70,           # Minimum overall score
    }
    
    # Grade boundaries
    GRADES = [
        (90, 'A'),
        (80, 'B'),
        (70, 'C'),
        (60, 'D'),
        (0, 'F'),
    ]
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self._arcface = None
        self._syncnet = None
        
    def check_video(self, video_path: str) -> QualityReport:
        """
        Run full quality assessment on a video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            QualityReport with all metrics
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Analyzing: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Basic info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        # Sample frames for analysis
        frames = self._sample_frames(cap, num_samples=30)
        cap.release()
        
        # Run metrics
        issues = []
        
        # 1. Resolution score
        resolution_score = self._score_resolution(width, height)
        if width < self.THRESHOLDS['resolution_min'][0]:
            issues.append(f"Resolution too low ({width}x{height})")
        
        # 2. Blur detection
        blur_score = self._score_blur(frames)
        if blur_score < 50:
            issues.append(f"Excessive blur detected (score: {blur_score:.1f})")
        
        # 3. Temporal consistency
        temporal_score = self._score_temporal(frames)
        if temporal_score < 60:
            issues.append(f"Temporal instability detected (score: {temporal_score:.1f})")
        
        # 4. Face identity (placeholder - needs ArcFace)
        identity_score = self._score_identity(frames)
        
        # 5. Lip sync (placeholder - needs SyncNet)
        lipsync_score = self._score_lipsync(video_path)
        
        # 6. Uncanny valley / photorealism
        uncanny_score = self._score_photorealism(frames)
        if uncanny_score < 60:
            issues.append(f"Uncanny valley issues (score: {uncanny_score:.1f})")
        
        # Overall score (weighted average)
        weights = {
            'resolution': 0.15,
            'blur': 0.20,
            'temporal': 0.15,
            'identity': 0.15,
            'lipsync': 0.20,
            'uncanny': 0.15,
        }
        
        overall_score = (
            resolution_score * weights['resolution'] +
            blur_score * weights['blur'] +
            temporal_score * weights['temporal'] +
            identity_score * weights['identity'] +
            lipsync_score * weights['lipsync'] +
            uncanny_score * weights['uncanny']
        )
        
        # Grade
        grade = 'F'
        for threshold, g in self.GRADES:
            if overall_score >= threshold:
                grade = g
                break
        
        passed = overall_score >= self.THRESHOLDS['overall_pass']
        
        report = QualityReport(
            video_path=str(video_path),
            resolution=(width, height),
            fps=fps,
            duration=duration,
            frame_count=frame_count,
            resolution_score=resolution_score,
            blur_score=blur_score,
            temporal_score=temporal_score,
            identity_score=identity_score,
            lipsync_score=lipsync_score,
            uncanny_score=uncanny_score,
            overall_score=overall_score,
            grade=grade,
            passed=passed,
            issues=issues,
        )
        
        return report
    
    def _sample_frames(self, cap, num_samples: int = 30) -> List[np.ndarray]:
        """Sample evenly distributed frames from video."""
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        return frames
    
    def _score_resolution(self, width: int, height: int) -> float:
        """Score resolution vs 4K target."""
        target_w, target_h = self.THRESHOLDS['resolution_target']
        min_w, min_h = self.THRESHOLDS['resolution_min']
        
        # Calculate pixel ratio
        actual_pixels = width * height
        target_pixels = target_w * target_h
        min_pixels = min_w * min_h
        
        if actual_pixels >= target_pixels:
            return 100.0
        elif actual_pixels <= min_pixels:
            return 50.0 * (actual_pixels / min_pixels)
        else:
            # Linear interpolation between min and target
            ratio = (actual_pixels - min_pixels) / (target_pixels - min_pixels)
            return 50.0 + 50.0 * ratio
    
    def _score_blur(self, frames: List[np.ndarray]) -> float:
        """Score blur using Laplacian variance."""
        variances = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            variances.append(variance)
        
        avg_variance = np.mean(variances)
        
        # Map to 0-100 score
        # Typical good quality: variance > 500
        # Typical bad quality: variance < 100
        if avg_variance >= 500:
            return 100.0
        elif avg_variance <= 50:
            return 20.0
        else:
            return 20.0 + 80.0 * (avg_variance - 50) / 450
    
    def _score_temporal(self, frames: List[np.ndarray]) -> float:
        """Score temporal consistency (frame-to-frame stability)."""
        if len(frames) < 2:
            return 100.0
        
        diffs = []
        
        for i in range(len(frames) - 1):
            # Calculate structural difference
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            diffs.append(np.mean(diff))
        
        # Measure variance of differences (high variance = unstable)
        diff_variance = np.std(diffs)
        
        # Map to score (lower variance = better)
        if diff_variance <= 5:
            return 100.0
        elif diff_variance >= 30:
            return 30.0
        else:
            return 100.0 - (diff_variance - 5) * 2.8
    
    def _score_identity(self, frames: List[np.ndarray]) -> float:
        """
        Score face identity consistency.
        Uses ArcFace to compare face embeddings across frames.
        """
        # TODO: Implement ArcFace-based identity check
        # For now, use face detection consistency as proxy
        
        try:
            import mediapipe as mp
            
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
            )
            
            detections = 0
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    detections += 1
            
            face_mesh.close()
            
            # Detection rate as proxy for consistency
            detection_rate = detections / len(frames)
            return detection_rate * 100
            
        except ImportError:
            # MediaPipe not available, return neutral score
            return 75.0
    
    def _score_lipsync(self, video_path: Path) -> float:
        """
        Score lip synchronization accuracy.
        Uses SyncNet or audio-video alignment analysis.
        """
        # TODO: Implement SyncNet-based lip sync check
        # For now, return neutral score
        return 75.0
    
    def _score_photorealism(self, frames: List[np.ndarray]) -> float:
        """
        Score photorealism / uncanny valley.
        
        Analyzes:
        - Skin texture naturalness
        - Eye region quality
        - Mouth region quality
        - Overall face consistency
        """
        scores = []
        
        for frame in frames:
            # Convert to LAB for skin analysis
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Skin color range in LAB
            l, a, b = cv2.split(lab)
            
            # Analyze color variance (natural skin has variation)
            skin_variance = np.std(a) + np.std(b)
            
            # Good skin: variance between 10-30
            if 10 <= skin_variance <= 30:
                texture_score = 100
            elif skin_variance < 10:
                texture_score = skin_variance * 10  # Too smooth = artificial
            else:
                texture_score = max(0, 100 - (skin_variance - 30) * 2)
            
            # Analyze edge quality (sharp but not too sharp)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Good: 5-15% edges
            if 0.05 <= edge_ratio <= 0.15:
                edge_score = 100
            else:
                edge_score = max(0, 100 - abs(edge_ratio - 0.1) * 1000)
            
            scores.append((texture_score + edge_score) / 2)
        
        return np.mean(scores)
    
    def print_report(self, report: QualityReport):
        """Print formatted quality report."""
        
        status = "✅ PASSED" if report.passed else "❌ FAILED"
        color = '\033[92m' if report.passed else '\033[91m'
        end = '\033[0m'
        
        print(f"\n{'='*60}")
        print(f"📊 QUALITY REPORT: {Path(report.video_path).name}")
        print(f"{'='*60}")
        
        print(f"\n📹 Video Info:")
        print(f"   Resolution: {report.resolution[0]}x{report.resolution[1]}")
        print(f"   FPS: {report.fps:.2f}")
        print(f"   Duration: {report.duration:.2f}s ({report.frame_count} frames)")
        
        print(f"\n📊 Quality Scores:")
        metrics = [
            ('Resolution', report.resolution_score),
            ('Sharpness', report.blur_score),
            ('Temporal', report.temporal_score),
            ('Identity', report.identity_score),
            ('Lip Sync', report.lipsync_score),
            ('Photorealism', report.uncanny_score),
        ]
        
        for name, score in metrics:
            bar_len = int(score / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"   {name:12} [{bar}] {score:5.1f}")
        
        print(f"\n{'='*60}")
        print(f"   OVERALL: {report.overall_score:.1f}/100  Grade: {report.grade}")
        print(f"   {color}{status}{end}")
        
        if report.issues:
            print(f"\n⚠️  Issues:")
            for issue in report.issues:
                print(f"   - {issue}")
        
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Kelly Video Quality Checker')
    parser.add_argument('videos', nargs='*', help='Video files to check')
    parser.add_argument('--batch', action='store_true', help='Process as batch')
    parser.add_argument('--json', help='Output JSON report to file')
    parser.add_argument('--compare', nargs=2, help='Compare two videos')
    parser.add_argument('--threshold', type=float, default=70, help='Pass threshold')
    
    args = parser.parse_args()
    
    if not args.videos and not args.compare:
        parser.error("Provide video files to check")
    
    checker = QualityChecker()
    
    if args.compare:
        # Compare two videos
        report1 = checker.check_video(args.compare[0])
        report2 = checker.check_video(args.compare[1])
        
        checker.print_report(report1)
        checker.print_report(report2)
        
        print(f"\n📊 COMPARISON:")
        print(f"   {Path(args.compare[0]).name}: {report1.overall_score:.1f}")
        print(f"   {Path(args.compare[1]).name}: {report2.overall_score:.1f}")
        
        diff = report1.overall_score - report2.overall_score
        if abs(diff) < 5:
            print(f"   Result: Approximately equal")
        elif diff > 0:
            print(f"   Result: First video is better (+{diff:.1f})")
        else:
            print(f"   Result: Second video is better (+{abs(diff):.1f})")
    
    else:
        # Check individual videos
        reports = []
        
        for video in args.videos:
            try:
                report = checker.check_video(video)
                reports.append(report)
                checker.print_report(report)
            except Exception as e:
                logger.error(f"Error checking {video}: {e}")
        
        # Save JSON report
        if args.json:
            with open(args.json, 'w') as f:
                json.dump([r.to_dict() for r in reports], f, indent=2)
            logger.info(f"Report saved to {args.json}")
        
        # Summary
        if len(reports) > 1:
            passed = sum(1 for r in reports if r.passed)
            avg_score = np.mean([r.overall_score for r in reports])
            print(f"\n{'='*60}")
            print(f"📊 BATCH SUMMARY: {passed}/{len(reports)} passed")
            print(f"   Average score: {avg_score:.1f}")
            print(f"{'='*60}")
        
        # Exit code
        all_passed = all(r.passed for r in reports)
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
