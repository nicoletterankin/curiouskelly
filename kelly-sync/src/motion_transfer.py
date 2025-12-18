#!/usr/bin/env python3
"""
🎭 KELLY-SYNC Motion Transfer

Extracts motion patterns from HeyGen reference videos
and applies them to newly generated frames.

Uses First Order Motion Model (FOMM) for motion extraction and transfer.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch

logger = logging.getLogger('kelly-sync.motion')


class MotionTransfer:
    """
    Motion transfer using First Order Motion Model.
    
    Purpose:
    - Extract Kelly's characteristic gestures from HeyGen videos
    - Apply consistent motion patterns to new lip-synced content
    - Maintain natural head movement and expressions
    
    This ensures local-generated videos match HeyGen quality.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        checkpoint_path: str = "models/fomm/vox-adv-cpk.pth.tar",
        config_path: str = "models/fomm/vox-256.yaml",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        
        self._generator = None
        self._kp_detector = None
        
        logger.info(f"MotionTransfer initialized on {self.device}")
    
    def _load_model(self):
        """Load First Order Motion Model."""
        logger.info("Loading First Order Motion Model...")
        
        try:
            import yaml
            from scipy.spatial import ConvexHull
            
            # FOMM architecture imports
            # Note: Requires FOMM codebase to be available
            from demo import load_checkpoints, make_animation
            from demo import OcclusionAwareGenerator, KPDetector
            
            # Load config
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            
            # Initialize models
            self._generator = OcclusionAwareGenerator(
                **config['model_params']['generator_params'],
                **config['model_params']['common_params']
            )
            self._kp_detector = KPDetector(
                **config['model_params']['kp_detector_params'],
                **config['model_params']['common_params']
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self._generator.load_state_dict(checkpoint['generator'])
            self._kp_detector.load_state_dict(checkpoint['kp_detector'])
            
            self._generator.to(self.device).eval()
            self._kp_detector.to(self.device).eval()
            
            logger.info("  FOMM model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"FOMM not available: {e}")
            logger.warning("Motion transfer will be disabled")
            self._generator = None
            self._kp_detector = None
    
    def extract_motion(
        self,
        video_path: str,
        max_frames: int = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract motion keypoints from a video.
        
        Args:
            video_path: Path to reference video
            max_frames: Maximum frames to process
        
        Returns:
            Dictionary of motion data:
            - keypoints: [N, K, 2] array of keypoint positions
            - jacobians: [N, K, 2, 2] array of local jacobians
        """
        if self._kp_detector is None:
            self._load_model()
        
        if self._kp_detector is None:
            logger.warning("Motion extraction unavailable")
            return {}
        
        logger.info(f"Extracting motion from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        keypoints = []
        jacobians = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (256, 256))
            frame_tensor = (
                torch.from_numpy(frame_resized)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )
            
            # Extract keypoints
            with torch.no_grad():
                kp = self._kp_detector(frame_tensor)
            
            keypoints.append(kp['value'].cpu().numpy())
            jacobians.append(kp['jacobian'].cpu().numpy())
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"  Extracted motion from {frame_count} frames")
        
        return {
            'keypoints': np.concatenate(keypoints, axis=0),
            'jacobians': np.concatenate(jacobians, axis=0),
        }
    
    def apply_motion(
        self,
        source_image: np.ndarray,
        motion_data: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        """
        Apply extracted motion to a source image.
        
        Args:
            source_image: Source face image (RGB, 256x256)
            motion_data: Motion data from extract_motion()
        
        Returns:
            List of animated frames
        """
        if self._generator is None:
            self._load_model()
        
        if self._generator is None:
            logger.warning("Motion application unavailable")
            return [source_image] * len(motion_data.get('keypoints', [1]))
        
        # Preprocess source
        source_resized = cv2.resize(source_image, (256, 256))
        source_tensor = (
            torch.from_numpy(source_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )
        
        # Get source keypoints
        with torch.no_grad():
            source_kp = self._kp_detector(source_tensor)
        
        frames = []
        keypoints = motion_data['keypoints']
        jacobians = motion_data['jacobians']
        
        for i in range(len(keypoints)):
            # Construct driving keypoints
            driving_kp = {
                'value': torch.from_numpy(keypoints[i:i+1]).to(self.device),
                'jacobian': torch.from_numpy(jacobians[i:i+1]).to(self.device),
            }
            
            # Generate frame
            with torch.no_grad():
                out = self._generator(source_tensor, source_kp, driving_kp)
            
            # Convert to numpy
            frame = (
                out['prediction']
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            
            frames.append(frame)
        
        return frames
    
    def apply(
        self,
        source_frames: List[np.ndarray],
        motion_template: str,
    ) -> List[np.ndarray]:
        """
        Apply motion from template video to source frames.
        
        This blends the lip movements from source_frames
        with the head motion from motion_template.
        
        Args:
            source_frames: Lip-synced frames
            motion_template: Path to HeyGen reference video
        
        Returns:
            Frames with combined lip sync and motion
        """
        logger.info("Applying motion template...")
        
        # Extract motion from template
        motion_data = self.extract_motion(
            motion_template,
            max_frames=len(source_frames),
        )
        
        if not motion_data:
            logger.warning("No motion data extracted, returning original frames")
            return source_frames
        
        # For now, return source frames with motion blended
        # Full implementation would:
        # 1. Separate lip region from head motion
        # 2. Keep lip sync from source_frames
        # 3. Apply head motion from template
        
        # TODO: Implement proper motion blending
        logger.warning("Full motion blending not yet implemented")
        return source_frames


class MotionLibrary:
    """
    Library of pre-extracted motion templates from HeyGen videos.
    
    Organizes motion by:
    - Archetype (scientist, explorer, etc.)
    - Motion type (A=welcoming, B=teaching, C=filler)
    - Phase (hook, cliff, fact, wisdom, outro)
    """
    
    def __init__(self, library_dir: str = "assets/motion_templates"):
        self.library_dir = Path(library_dir)
        self._cache = {}
        
        logger.info(f"MotionLibrary initialized: {library_dir}")
    
    def get_motion(
        self,
        archetype: str,
        motion_type: str = "B",
        phase: str = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Get motion data for a specific archetype and type.
        
        Args:
            archetype: Kelly archetype (scientist, explorer, etc.)
            motion_type: Motion category (A, B, or C)
            phase: Lesson phase (hook, cliff, etc.)
        
        Returns:
            Motion data dict or None if not found
        """
        cache_key = f"{archetype}_{motion_type}_{phase or 'full'}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Look for saved motion file
        motion_file = self.library_dir / archetype / f"motion_{motion_type}.npz"
        
        if not motion_file.exists():
            logger.warning(f"Motion template not found: {motion_file}")
            return None
        
        # Load motion data
        data = np.load(motion_file)
        motion_data = {
            'keypoints': data['keypoints'],
            'jacobians': data['jacobians'],
        }
        
        self._cache[cache_key] = motion_data
        return motion_data
    
    def extract_all(
        self,
        heygen_archive_dir: str,
        output_dir: str = None,
    ):
        """
        Extract motion templates from all archived HeyGen videos.
        
        This is a one-time operation to build the motion library.
        """
        archive_dir = Path(heygen_archive_dir)
        output_dir = Path(output_dir or self.library_dir)
        
        transfer = MotionTransfer()
        
        for video_file in archive_dir.glob("*.mp4"):
            # Parse filename: day-351-scientist-videoid.mp4
            parts = video_file.stem.split('-')
            if len(parts) >= 4:
                archetype = parts[2]
                
                logger.info(f"Extracting motion from {video_file.name}...")
                
                motion_data = transfer.extract_motion(str(video_file))
                
                if motion_data:
                    out_dir = output_dir / archetype
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    out_file = out_dir / "motion_B.npz"  # Assume main teaching
                    np.savez(out_file, **motion_data)
                    
                    logger.info(f"  Saved to {out_file}")
