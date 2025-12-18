#!/usr/bin/env python3
"""
✨ KELLY-SYNC Face Restorer

Uses CodeFormer for production-quality face restoration.
Critical for removing blur and artifacts from lip synthesis.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import torch

logger = logging.getLogger('kelly-sync.restore')


class FaceRestorer:
    """
    Face restoration using CodeFormer.
    
    CodeFormer advantages over GFPGAN:
    - Controllable fidelity vs quality tradeoff
    - Better identity preservation
    - Handles partial occlusion (speech)
    - More natural skin texture
    
    Fidelity weight:
    - 0.0 = Maximum quality, less faithful to input
    - 1.0 = Maximum fidelity to input, less enhancement
    - 0.5-0.7 = Balanced (recommended for Kelly)
    """
    
    def __init__(
        self,
        device: torch.device = None,
        fidelity_weight: float = 0.7,
        models_dir: str = "models/codeformer",
        upscale: int = 2,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fidelity_weight = fidelity_weight
        self.models_dir = Path(models_dir)
        self.upscale = upscale
        
        self._model = None
        self._face_helper = None
        
        logger.info(f"FaceRestorer initialized on {self.device}")
        logger.info(f"  Fidelity weight: {fidelity_weight}")
    
    def _load_model(self):
        """Load CodeFormer model and face helper."""
        logger.info("Loading CodeFormer model...")
        
        try:
            from basicsr.utils import imwrite
            from basicsr.archs.codeformer_arch import CodeFormer
            from basicsr.utils.registry import ARCH_REGISTRY
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Initialize face helper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=self.upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                device=self.device,
            )
            
            # Load CodeFormer model
            checkpoint_path = self.models_dir / 'codeformer.pth'
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"CodeFormer checkpoint not found at {checkpoint_path}\n"
                    f"Run: python scripts/download_models.py"
                )
            
            self._model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256'],
            ).to(self.device)
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self._model.load_state_dict(checkpoint['params_ema'])
            self._model.eval()
            
            logger.info("  CodeFormer model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Run: pip install basicsr facexlib")
            raise
    
    def restore(
        self,
        image: np.ndarray,
        return_face_only: bool = False,
    ) -> np.ndarray:
        """
        Restore face quality in image.
        
        Args:
            image: Input image (RGB, 0-255)
            return_face_only: If True, return only the cropped face
        
        Returns:
            Restored image (RGB, 0-255)
        """
        if self._model is None:
            self._load_model()
        
        # Convert to BGR for processing
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Initialize face helper with this image
        self._face_helper.clean_all()
        self._face_helper.read_image(img_bgr)
        
        # Detect and align faces
        num_faces = self._face_helper.get_face_landmarks_5(
            only_center_face=True,  # Kelly is always the center face
            resize=640,
            eye_dist_threshold=5,
        )
        
        if num_faces == 0:
            logger.warning("No face detected, returning original image")
            return image
        
        # Align and warp faces
        self._face_helper.align_warp_face()
        
        # Process each cropped face
        for idx, cropped_face in enumerate(self._face_helper.cropped_faces):
            # Normalize to [-1, 1]
            cropped_face_t = (
                torch.from_numpy(cropped_face.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )
            cropped_face_t = (cropped_face_t - 0.5) / 0.5
            
            # Inference
            with torch.no_grad():
                output = self._model(
                    cropped_face_t,
                    w=self.fidelity_weight,
                    adain=True,
                )[0]
                
                # Denormalize
                restored_face = (output.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
                restored_face = (
                    restored_face.permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            
            # Add to face helper for later pasting
            self._face_helper.add_restored_face(restored_face)
        
        if return_face_only:
            # Return just the restored face
            result = cv2.cvtColor(self._face_helper.restored_faces[0], cv2.COLOR_BGR2RGB)
        else:
            # Paste faces back to original image
            self._face_helper.get_inverse_affine(None)
            result = self._face_helper.paste_faces_to_input_image()
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return result
    
    def restore_batch(
        self,
        images: list,
        batch_size: int = 4,
    ) -> list:
        """
        Restore a batch of images efficiently.
        
        For video processing, this is more efficient than frame-by-frame.
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img in batch:
                restored = self.restore(img)
                results.append(restored)
        
        return results


class FaceRestorerSimple:
    """
    Simplified face restoration using GFPGAN.
    Fallback if CodeFormer is unavailable.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        logger.info(f"FaceRestorerSimple (GFPGAN) initialized on {self.device}")
    
    def _load_model(self):
        """Load GFPGAN model."""
        try:
            from gfpgan import GFPGANer
            
            model_path = Path("models/video_retalking/GFPGANv1.3.pth")
            
            self._model = GFPGANer(
                model_path=str(model_path),
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                device=self.device,
            )
            
            logger.info("  GFPGAN model loaded")
            
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise
    
    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore face using GFPGAN."""
        if self._model is None:
            self._load_model()
        
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        _, _, output = self._model.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
