#!/usr/bin/env python3
"""
GENERATE VIDEO SAFE ZONE MANIFESTS

Analyzes Kelly HD videos and generates safe zone manifests.
Runs during video production pipeline (after MiniMax + Sync Labs).

Approaches:
1. Manual annotation (fast, accurate for known content)
2. ML-based pose detection (MediaPipe, OpenPose)
3. Timeline markers from animation data

Usage:
    python generate-video-safe-zones.py --video day-001-phase-01-hook.mp4 --output day-001-phase-01-hook-safe-zones.json
"""

import argparse
import json
import cv2
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Tuple

class VideoSafeZoneGenerator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_id = Path(video_path).stem
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"[SafeZoneGen] Loaded video: {self.video_id}")
        print(f"[SafeZoneGen] Duration: {self.duration:.2f}s, FPS: {self.fps}, Size: {self.width}x{self.height}")
    
    def normalize_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized (0-1) coordinates"""
        return (x / self.width, y / self.height)
    
    def detect_face_box(self, landmarks) -> Dict:
        """Calculate face bounding box from pose landmarks"""
        # Get key facial landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        # Calculate center and size
        center_x = (left_eye.x + right_eye.x) / 2
        center_y = (left_eye.y + right_eye.y + nose.y) / 3
        
        # Face width (ear to ear) + padding
        face_width = abs(right_ear.x - left_ear.x) * 1.5
        face_height = face_width * 1.3  # Face is taller than wide
        
        return {
            "x": max(0, center_x - face_width / 2),
            "y": max(0, center_y - face_height / 2),
            "width": min(1.0, face_width),
            "height": min(1.0, face_height)
        }
    
    def detect_hand_boxes(self, landmarks) -> List[Dict]:
        """Calculate hand bounding boxes from pose landmarks"""
        hands = []
        
        # Left wrist
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        if left_wrist.visibility > 0.5:
            hands.append({
                "x": max(0, left_wrist.x - 0.04),
                "y": max(0, left_wrist.y - 0.04),
                "width": 0.08,
                "height": 0.08,
                "gesture": self.detect_gesture(landmarks, "left")
            })
        
        # Right wrist
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        if right_wrist.visibility > 0.5:
            hands.append({
                "x": max(0, right_wrist.x - 0.04),
                "y": max(0, right_wrist.y - 0.04),
                "width": 0.08,
                "height": 0.08,
                "gesture": self.detect_gesture(landmarks, "right")
            })
        
        return hands
    
    def detect_gesture(self, landmarks, side: str) -> str:
        """Detect gesture type (pointing, idle, etc.)"""
        if side == "left":
            wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        else:
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Check if arm is extended (pointing)
        arm_extension = abs(wrist.x - shoulder.x)
        if arm_extension > 0.3:  # Arm extended
            return "pointing"
        elif wrist.y < shoulder.y:  # Hand raised
            return "raised"
        else:
            return "idle"
    
    def calculate_safe_zones(self, face_box: Dict, hand_boxes: List[Dict]) -> List[Dict]:
        """Calculate safe zones based on blocked zones"""
        # Define potential safe zone areas
        potential_zones = [
            {"name": "top-left", "x": 0.05, "y": 0.05, "width": 0.30, "height": 0.25},
            {"name": "top-right", "x": 0.65, "y": 0.05, "width": 0.30, "height": 0.25},
            {"name": "left-mid", "x": 0.05, "y": 0.40, "width": 0.25, "height": 0.20},
            {"name": "right-mid", "x": 0.70, "y": 0.40, "width": 0.25, "height": 0.20},
            {"name": "bottom-left", "x": 0.05, "y": 0.75, "width": 0.30, "height": 0.20},
            {"name": "bottom-right", "x": 0.65, "y": 0.75, "width": 0.30, "height": 0.20}
        ]
        
        # Check each zone for overlap with face/hands
        safe_zones = []
        blocked_zones = [face_box] + hand_boxes
        
        for zone in potential_zones:
            overlap = False
            for blocked in blocked_zones:
                if self.boxes_overlap(zone, blocked):
                    overlap = True
                    break
            
            if not overlap:
                # Calculate score (distance from blocked zones)
                score = self.calculate_zone_score(zone, blocked_zones)
                safe_zones.append({**zone, "score": score})
        
        # Sort by score (highest first)
        safe_zones.sort(key=lambda z: z["score"], reverse=True)
        
        return safe_zones
    
    def boxes_overlap(self, box1: Dict, box2: Dict) -> bool:
        """Check if two boxes overlap"""
        return not (
            box1["x"] + box1["width"] < box2["x"] or
            box2["x"] + box2["width"] < box1["x"] or
            box1["y"] + box1["height"] < box2["y"] or
            box2["y"] + box2["height"] < box1["y"]
        )
    
    def calculate_zone_score(self, zone: Dict, blocked_zones: List[Dict]) -> float:
        """Calculate score for a safe zone (higher = safer)"""
        # Calculate minimum distance to any blocked zone
        min_distance = float('inf')
        
        zone_center_x = zone["x"] + zone["width"] / 2
        zone_center_y = zone["y"] + zone["height"] / 2
        
        for blocked in blocked_zones:
            blocked_center_x = blocked["x"] + blocked["width"] / 2
            blocked_center_y = blocked["y"] + blocked["height"] / 2
            
            distance = ((zone_center_x - blocked_center_x) ** 2 + 
                       (zone_center_y - blocked_center_y) ** 2) ** 0.5
            
            min_distance = min(min_distance, distance)
        
        # Convert distance to score (0-1)
        # Max distance on screen is ~1.4 (corner to corner)
        return min(1.0, min_distance / 1.4)
    
    def process_video(self, sample_rate: int = 30) -> Dict:
        """Process video and generate safe zone manifest"""
        print(f"[SafeZoneGen] Processing video (sampling every {sample_rate} frames)...")
        
        manifest = {
            "video_id": self.video_id,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "safe_zones": []
        }
        
        frame_count = 0
        last_segment = None
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Sample frames (not every frame, too expensive)
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Detect face and hands
                face_box = self.detect_face_box(landmarks)
                hand_boxes = self.detect_hand_boxes(landmarks)
                
                # Calculate safe zones
                safe_zones = self.calculate_safe_zones(face_box, hand_boxes)
                
                # Create segment
                time_start = frame_count / self.fps
                time_end = (frame_count + sample_rate) / self.fps
                
                segment = {
                    "time_start": round(time_start, 2),
                    "time_end": round(time_end, 2),
                    "kelly_face": face_box,
                    "kelly_hands": hand_boxes,
                    "safe_zones": safe_zones
                }
                
                # Merge with last segment if similar (reduce manifest size)
                if last_segment and self.segments_similar(last_segment, segment):
                    last_segment["time_end"] = segment["time_end"]
                else:
                    if last_segment:
                        manifest["safe_zones"].append(last_segment)
                    last_segment = segment
            
            frame_count += 1
            
            # Progress
            if frame_count % (sample_rate * 10) == 0:
                progress = (frame_count / self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100
                print(f"[SafeZoneGen] Progress: {progress:.1f}%")
        
        # Add last segment
        if last_segment:
            manifest["safe_zones"].append(last_segment)
        
        self.cap.release()
        print(f"[SafeZoneGen] ✅ Generated {len(manifest['safe_zones'])} segments")
        
        return manifest
    
    def segments_similar(self, seg1: Dict, seg2: Dict, threshold: float = 0.1) -> bool:
        """Check if two segments are similar enough to merge"""
        # Compare face positions
        face1 = seg1["kelly_face"]
        face2 = seg2["kelly_face"]
        
        face_diff = (
            abs(face1["x"] - face2["x"]) +
            abs(face1["y"] - face2["y"])
        ) / 2
        
        return face_diff < threshold
    
    def save_manifest(self, manifest: Dict, output_path: str):
        """Save manifest to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"[SafeZoneGen] ✅ Saved manifest: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate video safe zone manifests')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--sample-rate', type=int, default=30, help='Sample every N frames (default: 30)')
    
    args = parser.parse_args()
    
    # Generate manifest
    generator = VideoSafeZoneGenerator(args.video)
    manifest = generator.process_video(sample_rate=args.sample_rate)
    generator.save_manifest(manifest, args.output)
    
    print(f"\n[SafeZoneGen] 🎉 Done! Manifest ready for: {manifest['video_id']}")
    print(f"[SafeZoneGen] Duration: {manifest['duration']:.2f}s")
    print(f"[SafeZoneGen] Segments: {len(manifest['safe_zones'])}")
    print(f"[SafeZoneGen] Average segment length: {manifest['duration'] / len(manifest['safe_zones']):.2f}s")

if __name__ == '__main__':
    main()







