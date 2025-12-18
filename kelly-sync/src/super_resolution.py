#!/usr/bin/env python3
"""
📐 KELLY-SYNC Super Resolution

4K/8K upscaling using Real-ESRGAN.
Optimized for face content with anime variant blending.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
import torch

logger = logging.getLogger('kelly-sync.superres')


class SuperResolver:
    """
    Super resolution using Real-ESRGAN.
    
    Model selection:
    - RealESRGAN_x4plus: General purpose, good for backgrounds
    - RealESRGAN_x4plus_anime_6B: Sharper edges, better for faces
    
    We use a blend of both for optimal Kelly quality:
    - Anime model for face region (sharper lips, eyes)
    - Standard model for background (natural textures)
    """
    
    def __init__(
        self,
        device: torch.device = None,
        model_name: str = "RealESRGAN_x4plus",
        tile_size: int = 400,
        tile_pad: int = 40,
        half_precision: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.half_precision = half_precision and torch.cuda.is_available()
        
        self._upsampler = None
        self._upsampler_anime = None
        
        logger.info(f"SuperResolver initialized on {self.device}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Half precision: {half_precision}")
    
    def _load_model(self, model_name: str):
        """Load Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            models_dir = Path("models/real_esrgan")
            
            if model_name == "RealESRGAN_x4plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                model_path = models_dir / "RealESRGAN_x4plus.pth"
                netscale = 4
                
            elif model_name == "RealESRGAN_x4plus_anime_6B":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,
                    num_grow_ch=32,
                    scale=4,
                )
                model_path = models_dir / "RealESRGAN_x4plus_anime_6B.pth"
                netscale = 4
                
            elif model_name == "RealESRGAN_x2plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                model_path = models_dir / "RealESRGAN_x2plus.pth"
                netscale = 2
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}\n"
                    f"Run: python scripts/download_models.py"
                )
            
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                dni_weight=None,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=0,
                half=self.half_precision,
                device=self.device,
            )
            
            logger.info(f"  Loaded {model_name}")
            return upsampler
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install realesrgan basicsr")
            raise
    
    @property
    def upsampler(self):
        """Lazy load main upsampler."""
        if self._upsampler is None:
            self._upsampler = self._load_model(self.model_name)
        return self._upsampler
    
    @property
    def upsampler_anime(self):
        """Lazy load anime upsampler for face regions."""
        if self._upsampler_anime is None:
            self._upsampler_anime = self._load_model("RealESRGAN_x4plus_anime_6B")
        return self._upsampler_anime
    
    def upscale(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = None,
        use_anime_model: bool = False,
    ) -> np.ndarray:
        """
        Upscale image to target resolution.
        
        Args:
            image: Input image (RGB, 0-255)
            target_size: Target (width, height), or None for default 4x
            use_anime_model: Use anime model for sharper face details
        
        Returns:
            Upscaled image (RGB, 0-255)
        """
        # Convert to BGR for Real-ESRGAN
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Choose model
        model = self.upsampler_anime if use_anime_model else self.upsampler
        
        # Upscale
        output, _ = model.enhance(img_bgr, outscale=4)
        
        # Resize to exact target if specified
        if target_size is not None:
            target_w, target_h = target_size
            h, w = output.shape[:2]
            
            if (w, h) != (target_w, target_h):
                output = cv2.resize(
                    output,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )
        
        # Convert back to RGB
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    def upscale_with_face_blend(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int] = None,
        target_size: Tuple[int, int] = None,
        blend_margin: int = 20,
    ) -> np.ndarray:
        """
        Upscale with different models for face vs background.
        
        Uses anime model for face region (sharper) and
        standard model for background (more natural).
        
        Args:
            image: Input image (RGB)
            face_bbox: Face bounding box (x1, y1, x2, y2)
            target_size: Target resolution
            blend_margin: Pixels to blend at face/background boundary
        
        Returns:
            Upscaled image with blended models
        """
        # Get both upscaled versions
        upscaled_standard = self.upscale(image, target_size, use_anime_model=False)
        upscaled_anime = self.upscale(image, target_size, use_anime_model=True)
        
        if face_bbox is None:
            # No face bbox, use anime model (better for faces)
            return upscaled_anime
        
        # Create blend mask
        h, w = upscaled_standard.shape[:2]
        scale = w / image.shape[1]
        
        # Scale bbox to output resolution
        x1, y1, x2, y2 = [int(v * scale) for v in face_bbox]
        
        # Expand with margin
        x1 = max(0, x1 - blend_margin)
        y1 = max(0, y1 - blend_margin)
        x2 = min(w, x2 + blend_margin)
        y2 = min(h, y2 + blend_margin)
        
        # Create mask with soft edges
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        
        # Gaussian blur for soft blend
        mask = cv2.GaussianBlur(mask, (blend_margin * 2 + 1, blend_margin * 2 + 1), 0)
        mask = np.stack([mask] * 3, axis=-1)
        
        # Blend: anime in face region, standard elsewhere
        result = (upscaled_anime * mask + upscaled_standard * (1 - mask)).astype(np.uint8)
        
        return result
    
    def upscale_video_frames(
        self,
        frames: list,
        target_size: Tuple[int, int] = None,
        batch_size: int = 1,
    ) -> list:
        """
        Upscale a sequence of video frames.
        
        Note: Real-ESRGAN processes one frame at a time,
        so we just iterate. Future: implement temporal consistency.
        """
        from tqdm import tqdm
        
        logger.info(f"Upscaling {len(frames)} frames to {target_size or '4x'}...")
        
        results = []
        for frame in tqdm(frames, desc="Upscaling"):
            upscaled = self.upscale(frame, target_size, use_anime_model=True)
            results.append(upscaled)
        
        return results
