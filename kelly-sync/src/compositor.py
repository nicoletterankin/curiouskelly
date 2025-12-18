#!/usr/bin/env python3
"""
🎬 KELLY-SYNC Compositor

Final video compositing and export.
Handles blending, color matching, and encoding.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import subprocess
import tempfile
import shutil

logger = logging.getLogger('kelly-sync.compositor')


class Compositor:
    """
    Final compositing and video export.
    
    Features:
    - Poisson blending for seamless face integration
    - Color matching between face and background
    - Temporal smoothing for stable output
    - High-quality H.264/H.265 encoding
    """
    
    def __init__(
        self,
        output_format: str = "mp4",
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "slow",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
    ):
        self.output_format = output_format
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        
        logger.info("Compositor initialized")
        logger.info(f"  Codec: {codec}, CRF: {crf}, Preset: {preset}")
    
    def blend_face(
        self,
        face: np.ndarray,
        background: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        method: str = "poisson",
    ) -> np.ndarray:
        """
        Blend face region into background.
        
        Args:
            face: Face image (RGB)
            background: Background image (RGB)
            face_bbox: (x1, y1, x2, y2) position in background
            method: Blending method (alpha, poisson, seamless)
        
        Returns:
            Composited image
        """
        x1, y1, x2, y2 = face_bbox
        
        # Resize face to target region
        target_w = x2 - x1
        target_h = y2 - y1
        face_resized = cv2.resize(face, (target_w, target_h))
        
        if method == "alpha":
            return self._alpha_blend(face_resized, background, (x1, y1))
        
        elif method == "poisson":
            return self._poisson_blend(face_resized, background, (x1, y1))
        
        elif method == "seamless":
            return self._seamless_clone(face_resized, background, (x1, y1))
        
        else:
            # Simple paste
            result = background.copy()
            result[y1:y2, x1:x2] = face_resized
            return result
    
    def _alpha_blend(
        self,
        face: np.ndarray,
        background: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Alpha blend with soft edges."""
        x, y = position
        h, w = face.shape[:2]
        
        # Create soft edge mask
        mask = np.ones((h, w), dtype=np.float32)
        feather = min(10, h // 10, w // 10)
        
        # Feather edges
        for i in range(feather):
            alpha = i / feather
            mask[i, :] *= alpha
            mask[-(i+1), :] *= alpha
            mask[:, i] *= alpha
            mask[:, -(i+1)] *= alpha
        
        mask = np.stack([mask] * 3, axis=-1)
        
        # Blend
        result = background.copy().astype(np.float32)
        result[y:y+h, x:x+w] = (
            face.astype(np.float32) * mask +
            result[y:y+h, x:x+w] * (1 - mask)
        )
        
        return result.astype(np.uint8)
    
    def _poisson_blend(
        self,
        face: np.ndarray,
        background: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """Poisson blending for seamless integration."""
        x, y = position
        h, w = face.shape[:2]
        
        # Create mask (all white for full face)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Add soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Center point for seamlessClone
        center = (x + w // 2, y + h // 2)
        
        # Ensure background is large enough
        bg_h, bg_w = background.shape[:2]
        if center[0] >= bg_w or center[1] >= bg_h:
            logger.warning("Face extends beyond background, using simple paste")
            result = background.copy()
            result[y:y+h, x:x+w] = face
            return result
        
        # Convert to BGR for OpenCV
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        bg_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        
        try:
            result_bgr = cv2.seamlessClone(
                face_bgr,
                bg_bgr,
                mask,
                center,
                cv2.MIXED_CLONE,  # Better for faces than NORMAL_CLONE
            )
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            logger.warning(f"Poisson blend failed: {e}, using alpha blend")
            return self._alpha_blend(face, background, position)
    
    def _seamless_clone(
        self,
        face: np.ndarray,
        background: np.ndarray,
        position: Tuple[int, int],
    ) -> np.ndarray:
        """OpenCV seamlessClone with normal mode."""
        x, y = position
        h, w = face.shape[:2]
        
        mask = np.ones((h, w), dtype=np.uint8) * 255
        center = (x + w // 2, y + h // 2)
        
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        bg_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        
        result_bgr = cv2.seamlessClone(
            face_bgr,
            bg_bgr,
            mask,
            center,
            cv2.NORMAL_CLONE,
        )
        
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    def match_color(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        method: str = "histogram",
    ) -> np.ndarray:
        """
        Match color of source to reference.
        
        Args:
            source: Source image to adjust
            reference: Reference image for color matching
            method: Matching method (histogram, mean, mkl)
        
        Returns:
            Color-matched image
        """
        if method == "mean":
            # Simple mean/std matching
            source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            for i in range(3):
                src_mean = source_lab[:, :, i].mean()
                src_std = source_lab[:, :, i].std()
                ref_mean = ref_lab[:, :, i].mean()
                ref_std = ref_lab[:, :, i].std()
                
                source_lab[:, :, i] = (source_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean
            
            source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)
        
        elif method == "histogram":
            # Histogram matching per channel
            result = np.zeros_like(source)
            for i in range(3):
                result[:, :, i] = self._match_histogram_channel(
                    source[:, :, i],
                    reference[:, :, i],
                )
            return result
        
        return source
    
    def _match_histogram_channel(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Match histogram of single channel."""
        # Get CDFs
        src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
        
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normalize
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Build mapping
        lookup = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf[j] < src_cdf[i]:
                j += 1
            lookup[i] = j
        
        return lookup[source]
    
    def apply_temporal_smoothing(
        self,
        frames: List[np.ndarray],
        window_size: int = 5,
    ) -> List[np.ndarray]:
        """
        Apply temporal smoothing to reduce flicker.
        
        Uses weighted average of nearby frames.
        """
        if window_size < 2:
            return frames
        
        logger.info(f"Applying temporal smoothing (window={window_size})...")
        
        n_frames = len(frames)
        smoothed = []
        
        # Gaussian weights
        half_window = window_size // 2
        weights = np.exp(-0.5 * (np.arange(-half_window, half_window + 1) ** 2) / (half_window / 2) ** 2)
        weights /= weights.sum()
        
        for i in range(n_frames):
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            
            window_frames = frames[start:end]
            window_weights = weights[
                (half_window - (i - start)):
                (half_window + (end - i))
            ]
            window_weights = window_weights / window_weights.sum()
            
            # Weighted average
            result = np.zeros_like(frames[0], dtype=np.float32)
            for frame, w in zip(window_frames, window_weights):
                result += frame.astype(np.float32) * w
            
            smoothed.append(result.astype(np.uint8))
        
        return smoothed
    
    def export(
        self,
        frames: List[np.ndarray],
        audio_path: str,
        output_path: str,
        fps: int = 30,
    ) -> str:
        """
        Export video with audio.
        
        Args:
            frames: List of RGB frames
            audio_path: Path to audio file
            output_path: Output video path
            fps: Frame rate
        
        Returns:
            Path to output video
        """
        logger.info(f"Exporting video ({len(frames)} frames at {fps} fps)...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for frame images
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Write frames as images
            logger.info("  Writing frames...")
            for i, frame in enumerate(frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(temp_dir / f"frame_{i:06d}.png"), frame_bgr)
            
            # Use FFmpeg to combine frames and audio
            logger.info("  Encoding video...")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", str(temp_dir / "frame_%06d.png"),
                "-i", audio_path,
                "-c:v", self.codec,
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                "-c:a", self.audio_codec,
                "-b:a", self.audio_bitrate,
                "-shortest",  # Match shortest stream
                str(output_path),
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError("FFmpeg encoding failed")
            
            logger.info(f"  ✅ Exported to: {output_path}")
            
            # Get file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  File size: {size_mb:.1f} MB")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return str(output_path)
    
    def export_frames_only(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        format: str = "png",
    ) -> str:
        """
        Export frames as individual images.
        
        Useful for debugging or further processing.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"frame_{i:06d}.{format}"), frame_bgr)
        
        logger.info(f"Exported {len(frames)} frames to {output_dir}")
        return str(output_dir)
