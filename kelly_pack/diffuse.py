"""
Diffuse texture neutralization: channel balancing and contrast flattening
"""
import numpy as np


def gray_world_balance(img: np.ndarray) -> np.ndarray:
    """
    Apply gray-world color balance.
    
    Args:
        img: RGB image [H, W, 3] in [0, 255]
    
    Returns:
        Balanced RGB image
    """
    img_float = img.astype(np.float32)
    
    # Compute mean for each channel
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])
    
    # Target gray (average of means)
    gray = (mean_r + mean_g + mean_b) / 3.0
    
    # Scale each channel
    if mean_r > 0:
        img_float[:, :, 0] *= gray / mean_r
    if mean_g > 0:
        img_float[:, :, 1] *= gray / mean_g
    if mean_b > 0:
        img_float[:, :, 2] *= gray / mean_b
    
    return np.clip(img_float, 0, 255).astype(np.uint8)


def flatten_contrast(img: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """
    Reduce contrast by moving values toward mid-gray.
    
    Args:
        img: RGB image [H, W, 3] in [0, 255]
        amount: Contrast reduction amount (0.15 = 15% reduction)
    
    Returns:
        Flattened RGB image
    """
    img_float = img.astype(np.float32) / 255.0
    
    # Mid-gray reference
    mid_gray = 0.5
    
    # Move toward mid-gray
    img_flat = img_float * (1 - amount) + mid_gray * amount
    
    return (np.clip(img_flat, 0, 1) * 255).astype(np.uint8)


def neutralize_diffuse(img: np.ndarray, contrast_flatten: float = 0.15) -> np.ndarray:
    """
    Full diffuse neutralization pipeline.
    
    Args:
        img: RGB image [H, W, 3]
        contrast_flatten: Contrast flattening amount
    
    Returns:
        Neutralized RGB image
    """
    # Step 1: Gray-world balance
    balanced = gray_world_balance(img)
    
    # Step 2: Flatten contrast
    flattened = flatten_contrast(balanced, contrast_flatten)
    
    return flattened


