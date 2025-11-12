"""
Alpha channel utilities: soft/tight variants, edge extraction, morphology
"""
import numpy as np
import cv2
from typing import Tuple


def apply_gaussian_blur(alpha: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply Gaussian blur to alpha channel.
    
    Args:
        alpha: Alpha channel [H, W]
        radius: Blur radius in pixels
    
    Returns:
        Blurred alpha
    """
    if radius <= 0:
        return alpha
    
    ksize = int(radius * 2) * 2 + 1  # Ensure odd
    return cv2.GaussianBlur(alpha, (ksize, ksize), radius)


def apply_morphology(alpha: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    """
    Apply morphological operation to alpha.
    
    Args:
        alpha: Alpha channel [H, W] in [0, 1]
        operation: "erode" or "dilate"
        kernel_size: Kernel size (odd integer)
    
    Returns:
        Processed alpha
    """
    if kernel_size <= 0:
        return alpha
    
    # Convert to uint8 for morphology
    alpha_8bit = (alpha * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == "erode":
        result = cv2.erode(alpha_8bit, kernel, iterations=1)
    elif operation == "dilate":
        result = cv2.dilate(alpha_8bit, kernel, iterations=1)
    else:
        result = alpha_8bit
    
    return result.astype(np.float32) / 255.0


def generate_soft_alpha(alpha: np.ndarray,
                       blur_radius: float = 2.0,
                       bias: float = 0.05) -> np.ndarray:
    """
    Generate soft alpha with halo (good for light UIs).
    
    Args:
        alpha: Base alpha [H, W]
        blur_radius: Blur radius in pixels
        bias: Positive bias to expand edges
    
    Returns:
        Soft alpha [H, W]
    """
    soft = apply_gaussian_blur(alpha, blur_radius)
    soft = np.clip(soft + bias, 0, 1)
    return soft


def generate_tight_alpha(alpha: np.ndarray,
                        blur_radius: float = 1.0,
                        bias: float = -0.03,
                        erode_size: int = 1) -> np.ndarray:
    """
    Generate tight alpha with no halo (good for dark UIs).
    
    Args:
        alpha: Base alpha [H, W]
        blur_radius: Blur radius in pixels
        bias: Negative bias to contract edges
        erode_size: Erosion kernel size
    
    Returns:
        Tight alpha [H, W]
    """
    tight = apply_gaussian_blur(alpha, blur_radius)
    tight = np.clip(tight + bias, 0, 1)
    
    if erode_size > 0:
        tight = apply_morphology(tight, "erode", erode_size)
    
    return tight


def generate_edge_matte(soft: np.ndarray, tight: np.ndarray) -> np.ndarray:
    """
    Generate edge-only matte (soft minus tight).
    
    Args:
        soft: Soft alpha [H, W]
        tight: Tight alpha [H, W]
    
    Returns:
        Edge matte [H, W]
    """
    edge = soft - tight
    return np.clip(edge, 0, 1)


def generate_alpha_variants(base_alpha: np.ndarray,
                           soft_blur: float = 2.0,
                           soft_bias: float = 0.05,
                           tight_blur: float = 1.0,
                           tight_bias: float = -0.03,
                           tight_erode: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all three alpha variants from base alpha.
    
    Args:
        base_alpha: Base alpha matte [H, W]
        soft_blur: Soft alpha blur radius
        soft_bias: Soft alpha bias
        tight_blur: Tight alpha blur radius
        tight_bias: Tight alpha bias
        tight_erode: Tight alpha erosion size
    
    Returns:
        (alpha_soft, alpha_tight, alpha_edge) tuple
    """
    alpha_soft = generate_soft_alpha(base_alpha, soft_blur, soft_bias)
    alpha_tight = generate_tight_alpha(base_alpha, tight_blur, tight_bias, tight_erode)
    alpha_edge = generate_edge_matte(alpha_soft, alpha_tight)
    
    return alpha_soft, alpha_tight, alpha_edge


