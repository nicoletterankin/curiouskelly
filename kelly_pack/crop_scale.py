"""
Crop and scale utilities for precise 16:9 and square framing
"""
from typing import Tuple
import numpy as np
from PIL import Image


def crop_to_aspect(img: np.ndarray, aspect_w: int, aspect_h: int) -> np.ndarray:
    """
    Center-crop image to target aspect ratio.
    
    Args:
        img: Input image [H, W, C]
        aspect_w: Target aspect width
        aspect_h: Target aspect height
    
    Returns:
        Cropped image
    """
    h, w = img.shape[:2]
    target_ratio = aspect_w / aspect_h
    current_ratio = w / h
    
    if current_ratio > target_ratio:
        # Too wide, crop sides
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img[:, left:left + new_w]
    else:
        # Too tall, crop top/bottom
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img[top:top + new_h, :]


def resize_lanczos(img: np.ndarray, size: tuple, mode: str = "RGB") -> np.ndarray:
    """
    Resize image using Lanczos (high quality).
    
    Args:
        img: Input image [H, W, C]
        size: (width, height)
        mode: PIL mode
    
    Returns:
        Resized image
    """
    pil_img = Image.fromarray(img, mode=mode)
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    return np.array(pil_img)


def prepare_16_9_hero(img: np.ndarray, target_w: int = 7680, target_h: int = 4320) -> np.ndarray:
    """
    Prepare 16:9 hero image at 8K resolution.
    
    Args:
        img: Input image
        target_w: Target width (default 7680)
        target_h: Target height (default 4320)
    
    Returns:
        Processed image at target resolution
    """
    # Crop to 16:9
    img = crop_to_aspect(img, 16, 9)
    
    # Resize to target
    if img.shape[2] == 4:  # RGBA
        return resize_lanczos(img, (target_w, target_h), mode="RGBA")
    else:
        return resize_lanczos(img, (target_w, target_h), mode="RGB")


def prepare_square_sprite(img: np.ndarray, 
                         alpha: np.ndarray,
                         canvas_size: int = 8192,
                         padding_frac: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center subject on square canvas with padding.
    
    Args:
        img: RGB image [H, W, 3]
        alpha: Alpha channel [H, W]
        canvas_size: Output size (square)
        padding_frac: Padding fraction (0.10 = 10% on each side)
    
    Returns:
        (rgb_canvas, alpha_canvas) both [canvas_size, canvas_size]
    """
    # Find subject bounds from alpha
    rows = np.any(alpha > 0.01, axis=1)
    cols = np.any(alpha > 0.01, axis=0)
    
    if not rows.any() or not cols.any():
        # Empty alpha, just center the image
        y1, y2 = 0, img.shape[0]
        x1, x2 = 0, img.shape[1]
    else:
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Crop to subject
    subject_rgb = img[y1:y2+1, x1:x2+1]
    subject_alpha = alpha[y1:y2+1, x1:x2+1]
    
    # Compute scale to fit in (1 - 2*padding) of canvas
    target_size = int(canvas_size * (1 - 2 * padding_frac))
    h, w = subject_rgb.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize subject
    subject_rgb_resized = resize_lanczos(subject_rgb, (new_w, new_h), mode="RGB")
    subject_alpha_resized = resize_lanczos(
        (subject_alpha * 255).astype(np.uint8), 
        (new_w, new_h), 
        mode="L"
    ).astype(np.float32) / 255.0
    
    # Create canvas
    rgb_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    
    # Center on canvas
    y_offset = (canvas_size - new_h) // 2
    x_offset = (canvas_size - new_w) // 2
    
    rgb_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = subject_rgb_resized
    alpha_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = subject_alpha_resized
    
    return rgb_canvas, alpha_canvas

