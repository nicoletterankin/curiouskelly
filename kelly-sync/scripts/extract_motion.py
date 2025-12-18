#!/usr/bin/env python3
"""
🎭 EXTRACT MOTION FROM HEYGEN VIDEOS

Extracts motion patterns from archived HeyGen videos to create
Kelly-specific motion templates for the local pipeline.

This is a ONE-TIME operation to build the motion library.

Usage:
    # Extract from all Day 351 videos
    python extract_motion.py --input generated-videos/heygen-archive

    # Extract from specific video
    python extract_motion.py --video day-351-scientist-xyz.mp4

    # List extracted motions
    python extract_motion.py --list
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('kelly-motion')

# Output directory for motion templates
MOTION_DIR = Path(__file__).parent.parent / 'assets' / 'motion_templates'


class MotionExtractor:
    """
    Extracts motion data from Kelly videos.
    
    Captures:
    - Face landmarks (68 points)
    - Head pose (rotation, translation)
    - Optical flow (pixel motion)
    - Expression coefficients
    
    Output is saved as compressed numpy files for efficient storage.
    """
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self._face_mesh = None
        
        # Ensure output directory exists
        MOTION_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("MotionExtractor initialized")
    
    @property
    def face_mesh(self):
        """Lazy load MediaPipe Face Mesh."""
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._face_mesh
    
    def extract_from_video(
        self,
        video_path: str,
        output_dir: str = None,
    ) -> Dict:
        """
        Extract motion data from a single video.
        
        Args:
            video_path: Path to video file
            output_dir: Where to save motion data
        
        Returns:
            Dict with extraction metadata
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"\n📹 Extracting motion from: {video_path.name}")
        
        # Parse archetype from filename (day-351-scientist-xyz.mp4)
        parts = video_path.stem.split('-')
        archetype = parts[2] if len(parts) >= 4 else 'unknown'
        day = parts[1] if len(parts) >= 2 else '0'
        
        logger.info(f"   Archetype: {archetype}")
        logger.info(f"   Day: {day}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"   Resolution: {width}x{height}")
        logger.info(f"   Duration: {duration:.2f}s ({frame_count} frames)")
        
        # Extract data
        landmarks_sequence = []
        optical_flow_sequence = []
        prev_gray = None
        
        pbar = tqdm(total=frame_count, desc="   Extracting")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract face landmarks
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                # Convert to numpy array (478 landmarks × 3 coordinates)
                lm_array = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in landmarks.landmark
                ])
                landmarks_sequence.append(lm_array)
            else:
                # Use previous landmarks if face not detected
                if landmarks_sequence:
                    landmarks_sequence.append(landmarks_sequence[-1])
                else:
                    landmarks_sequence.append(np.zeros((478, 3)))
            
            # Calculate optical flow (skip first frame)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                
                # Downsample flow for storage efficiency
                flow_small = cv2.resize(flow, (64, 64))
                optical_flow_sequence.append(flow_small)
            
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        # Convert to numpy arrays
        landmarks_array = np.array(landmarks_sequence)
        flow_array = np.array(optical_flow_sequence) if optical_flow_sequence else None
        
        logger.info(f"   Landmarks shape: {landmarks_array.shape}")
        if flow_array is not None:
            logger.info(f"   Flow shape: {flow_array.shape}")
        
        # Calculate statistics
        # Mouth openness from landmarks (distance between upper and lower lip)
        upper_lip_idx = 13  # MediaPipe landmark index
        lower_lip_idx = 14
        mouth_openness = landmarks_array[:, lower_lip_idx, 1] - landmarks_array[:, upper_lip_idx, 1]
        
        # Head rotation (simplified from nose and eye positions)
        nose_idx = 1
        left_eye_idx = 33
        right_eye_idx = 263
        
        eye_vector = landmarks_array[:, right_eye_idx, :2] - landmarks_array[:, left_eye_idx, :2]
        head_rotation = np.arctan2(eye_vector[:, 1], eye_vector[:, 0])
        
        # Save motion data
        output_dir = Path(output_dir) if output_dir else MOTION_DIR / archetype
        output_dir.mkdir(parents=True, exist_ok=True)
        
        motion_file = output_dir / f"day-{day}-motion.npz"
        
        np.savez_compressed(
            motion_file,
            landmarks=landmarks_array,
            optical_flow=flow_array,
            mouth_openness=mouth_openness,
            head_rotation=head_rotation,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        
        file_size = motion_file.stat().st_size / (1024 * 1024)
        logger.info(f"   ✅ Saved: {motion_file} ({file_size:.2f}MB)")
        
        # Save metadata
        metadata = {
            'source_video': str(video_path),
            'archetype': archetype,
            'day': day,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'resolution': [width, height],
            'landmarks_shape': list(landmarks_array.shape),
            'extracted': datetime.now().isoformat(),
        }
        
        meta_file = output_dir / f"day-{day}-motion.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def extract_from_directory(
        self,
        input_dir: str,
        pattern: str = "*.mp4",
    ) -> List[Dict]:
        """
        Extract motion from all videos in a directory.
        
        Args:
            input_dir: Directory containing videos
            pattern: Glob pattern for video files
        
        Returns:
            List of extraction metadata
        """
        input_dir = Path(input_dir)
        videos = list(input_dir.glob(pattern))
        
        if not videos:
            logger.warning(f"No videos found matching {pattern} in {input_dir}")
            return []
        
        logger.info(f"\n📁 Found {len(videos)} videos in {input_dir}")
        
        results = []
        
        for video_path in videos:
            try:
                metadata = self.extract_from_video(video_path)
                results.append(metadata)
            except Exception as e:
                logger.error(f"   ❌ Error processing {video_path.name}: {e}")
        
        return results
    
    def get_motion_for_archetype(
        self,
        archetype: str,
        day: int = None,
    ) -> Optional[Dict]:
        """
        Load extracted motion data for an archetype.
        
        Args:
            archetype: Kelly archetype name
            day: Specific day (or None for any available)
        
        Returns:
            Dict with motion data or None if not found
        """
        archetype_dir = MOTION_DIR / archetype
        
        if not archetype_dir.exists():
            return None
        
        if day:
            motion_file = archetype_dir / f"day-{day}-motion.npz"
        else:
            # Get any available motion file
            motion_files = list(archetype_dir.glob("day-*-motion.npz"))
            if not motion_files:
                return None
            motion_file = motion_files[0]
        
        if not motion_file.exists():
            return None
        
        data = np.load(motion_file)
        
        return {
            'landmarks': data['landmarks'],
            'optical_flow': data.get('optical_flow'),
            'mouth_openness': data['mouth_openness'],
            'head_rotation': data['head_rotation'],
            'fps': float(data['fps']),
            'frame_count': int(data['frame_count']),
        }
    
    def list_available_motions(self) -> Dict[str, List[str]]:
        """List all available motion templates."""
        motions = {}
        
        for archetype_dir in MOTION_DIR.iterdir():
            if archetype_dir.is_dir():
                days = []
                for motion_file in archetype_dir.glob("day-*-motion.npz"):
                    day = motion_file.stem.split('-')[1]
                    days.append(day)
                
                if days:
                    motions[archetype_dir.name] = sorted(days)
        
        return motions


def main():
    parser = argparse.ArgumentParser(description='Extract motion from Kelly videos')
    parser.add_argument('--input', '-i', help='Input directory with videos')
    parser.add_argument('--video', '-v', help='Single video to process')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--list', '-l', action='store_true', help='List available motions')
    parser.add_argument('--archetype', help='Extract for specific archetype')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║  🎭 KELLY MOTION EXTRACTOR                                 ║
║  Building motion templates from HeyGen videos              ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    extractor = MotionExtractor()
    
    if args.list:
        # List available motions
        motions = extractor.list_available_motions()
        
        if not motions:
            print("No motion templates found.")
            print(f"Run extraction first with --input <heygen-archive-dir>")
        else:
            print(f"📁 Motion templates in {MOTION_DIR}:\n")
            for archetype, days in sorted(motions.items()):
                print(f"  {archetype}:")
                for day in days:
                    print(f"    - Day {day}")
        
        return 0
    
    if args.video:
        # Single video extraction
        try:
            metadata = extractor.extract_from_video(
                args.video,
                output_dir=args.output,
            )
            print(f"\n✅ Extraction complete")
            return 0
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
    
    if args.input:
        # Directory extraction
        input_dir = Path(args.input)
        
        if not input_dir.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return 1
        
        results = extractor.extract_from_directory(input_dir)
        
        print(f"\n{'='*60}")
        print(f"📊 EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Processed: {len(results)} videos")
        print(f"  Output: {MOTION_DIR}")
        
        # Group by archetype
        by_archetype = {}
        for r in results:
            arch = r.get('archetype', 'unknown')
            by_archetype[arch] = by_archetype.get(arch, 0) + 1
        
        print(f"\n  By archetype:")
        for arch, count in sorted(by_archetype.items()):
            print(f"    {arch}: {count}")
        
        return 0
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
