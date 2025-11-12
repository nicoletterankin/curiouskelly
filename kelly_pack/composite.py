"""
Compositing utilities: gradients, alpha blending
"""
import numpy as np


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: "#RRGGBB" format
    
    Returns:
        (R, G, B) tuple, values in [0, 255]
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_vertical_gradient(width: int, 
                             height: int,
                             top_color: str = "#22262A",
                             bottom_color: str = "#080808") -> np.ndarray:
    """
    Create vertical gradient background.
    
    Args:
        width: Width in pixels
        height: Height in pixels
        top_color: Top color (hex)
        bottom_color: Bottom color (hex)
    
    Returns:
        RGB gradient [H, W, 3]
    """
    top_rgb = np.array(hex_to_rgb(top_color), dtype=np.float32)
    bottom_rgb = np.array(hex_to_rgb(bottom_color), dtype=np.float32)
    
    # Create gradient
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        t = y / (height - 1)  # Interpolation factor [0, 1]
        color = top_rgb * (1 - t) + bottom_rgb * t
        gradient[y, :] = color
    
    return gradient.astype(np.uint8)


def composite_over_background(rgb: np.ndarray,
                              alpha: np.ndarray,
                              background: np.ndarray) -> np.ndarray:
    """
    Composite RGBA over background using alpha blending.
    
    Args:
        rgb: Foreground RGB [H, W, 3]
        alpha: Foreground alpha [H, W] in [0, 1]
        background: Background RGB [H, W, 3]
    
    Returns:
        Composited RGB [H, W, 3]
    """
    # Ensure shapes match
    assert rgb.shape[:2] == alpha.shape, "RGB and alpha must have same dimensions"
    assert rgb.shape == background.shape, "RGB and background must have same dimensions"
    
    # Expand alpha to 3 channels
    alpha_3ch = alpha[:, :, np.newaxis]
    
    # Alpha blend: result = fg * alpha + bg * (1 - alpha)
    rgb_float = rgb.astype(np.float32)
    bg_float = background.astype(np.float32)
    
    result = rgb_float * alpha_3ch + bg_float * (1 - alpha_3ch)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_dark_hero(rgb: np.ndarray,
                    alpha: np.ndarray,
                    top_color: str = "#22262A",
                    bottom_color: str = "#080808") -> np.ndarray:
    """
    Create dark-mode hero by compositing over dark gradient.
    
    Args:
        rgb: Foreground RGB [H, W, 3]
        alpha: Foreground alpha [H, W]
        top_color: Gradient top color
        bottom_color: Gradient bottom color
    
    Returns:
        Composited RGB [H, W, 3]
    """
    h, w = rgb.shape[:2]
    gradient = create_vertical_gradient(w, h, top_color, bottom_color)
    return composite_over_background(rgb, alpha, gradient)


