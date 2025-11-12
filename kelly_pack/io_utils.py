"""
I/O utilities for loading and saving images
"""
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image


def load_image(path: str, mode: str = "RGB") -> Optional[np.ndarray]:
    """
    Load an image from path.
    
    Args:
        path: Path to image file
        mode: PIL mode (RGB, RGBA, L, etc.)
    
    Returns:
        numpy array [H, W, C] or None if file not found
    """
    if not os.path.exists(path):
        return None
    
    try:
        img = Image.open(path).convert(mode)
        return np.array(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def save_image(img: np.ndarray, path: str, mode: str = "RGB"):
    """
    Save numpy array as image.
    
    Args:
        img: numpy array [H, W, C] or [H, W]
        path: Output path
        mode: PIL mode
    """
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Clamp to valid range
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img, mode=mode)
    pil_img.save(path)
    print(f"Saved: {path} ({pil_img.size[0]}x{pil_img.size[1]})")


def find_first_existing(*paths: str) -> Optional[str]:
    """
    Return the first path that exists, or None.
    """
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


