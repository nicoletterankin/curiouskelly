"""
Portrait/hair matting using model-based and heuristic approaches
"""
import os
import numpy as np
from typing import Tuple, Optional
import cv2


# Model cache
_model_cache = {}


def download_u2net_weights(weights_dir: str = "./weights") -> str:
    """
    Download U²-Net portrait weights if not present.
    
    Returns:
        Path to weights file
    """
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "u2net_portrait.pth")
    
    if os.path.exists(weights_path):
        return weights_path
    
    print("Downloading U²-Net portrait weights (~4.7 MB)...")
    try:
        import urllib.request
        url = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net_portrait/u2net_portrait.pth"
        urllib.request.urlretrieve(url, weights_path)
        print(f"Downloaded: {weights_path}")
        return weights_path
    except Exception as e:
        print(f"Warning: Could not download weights: {e}")
        return None


def model_based_matting(img: np.ndarray, 
                       device: str = "cpu",
                       weights_dir: str = "./weights") -> Optional[np.ndarray]:
    """
    Generate alpha matte using U²-Net portrait model.
    
    Args:
        img: RGB image [H, W, 3]
        device: "cpu" or "cuda"
        weights_dir: Directory for model weights
    
    Returns:
        Alpha matte [H, W] in range [0, 1] or None if failed
    """
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        
        # Download weights if needed
        weights_path = download_u2net_weights(weights_dir)
        if not weights_path:
            return None
        
        # Load model (cached)
        if "u2net" not in _model_cache:
            # Simple U²-Net-like model (we'll use a lightweight version)
            # For production, use the full U²-Net implementation
            print("Loading U²-Net model...")
            try:
                from u2net import U2NET  # Will fail if not installed, that's ok
                model = U2NET(3, 1)
                model.load_state_dict(torch.load(weights_path, map_location=device))
            except:
                print("U²-Net module not available, falling back to heuristic")
                return None
            
            model.eval()
            model.to(device)
            _model_cache["u2net"] = model
        
        model = _model_cache["u2net"]
        
        # Prepare input
        orig_h, orig_w = img.shape[:2]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            d0, *_ = model(input_tensor)
            pred = torch.sigmoid(d0)
        
        # Post-process
        alpha = pred.squeeze().cpu().numpy()
        alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1)
        
        return alpha
        
    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return None
    except Exception as e:
        print(f"Model-based matting failed: {e}")
        return None


def heuristic_matting(img: np.ndarray,
                     white_threshold: float = 0.75,
                     chroma_weight: float = 0.6) -> np.ndarray:
    """
    Heuristic alpha estimation for white/light backgrounds.
    
    Args:
        img: RGB image [H, W, 3], values in [0, 255]
        white_threshold: Lower threshold for white detection
        chroma_weight: Weight for chroma in whiteness computation
    
    Returns:
        Alpha matte [H, W] in range [0, 1]
    """
    # Normalize to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Compute luminance (ITU-R BT.709)
    luminance = 0.2126 * img_float[:, :, 0] + 0.7152 * img_float[:, :, 1] + 0.0722 * img_float[:, :, 2]
    
    # Compute chroma (saturation)
    max_rgb = np.max(img_float, axis=2)
    min_rgb = np.min(img_float, axis=2)
    chroma = max_rgb - min_rgb
    
    # Whiteness = high luminance + low chroma
    whiteness = luminance - chroma_weight * chroma
    
    # Convert to alpha via smoothstep
    t0 = white_threshold - 0.15
    t1 = white_threshold + 0.05
    
    alpha = 1.0 - np.clip((whiteness - t0) / (t1 - t0), 0, 1)
    
    # Smooth smoothstep
    alpha = alpha * alpha * (3 - 2 * alpha)
    
    return alpha


def generate_alpha(img: np.ndarray,
                  use_model: bool = True,
                  device: str = "cpu",
                  weights_dir: str = "./weights") -> np.ndarray:
    """
    Generate base alpha matte with automatic fallback.
    
    Args:
        img: RGB image [H, W, 3]
        use_model: Try model-based first
        device: "cpu" or "cuda"
        weights_dir: Model weights directory
    
    Returns:
        Alpha matte [H, W] in [0, 1]
    """
    alpha = None
    
    if use_model:
        print("Attempting model-based matting...")
        alpha = model_based_matting(img, device=device, weights_dir=weights_dir)
    
    if alpha is None:
        print("Using heuristic matting (white background estimation)...")
        alpha = heuristic_matting(img)
    
    return alpha


def guided_upsample_alpha(alpha_low: np.ndarray,
                         guide_rgb: np.ndarray,
                         target_size: Tuple[int, int]) -> np.ndarray:
    """
    Edge-aware upsample of alpha using RGB guide.
    
    Args:
        alpha_low: Low-res alpha [H_low, W_low]
        guide_rgb: High-res RGB guide [H_high, W_high, 3]
        target_size: (width, height) for output
    
    Returns:
        Upsampled alpha [H_high, W_high]
    """
    # First, bilinear upsample
    alpha_up = cv2.resize(alpha_low, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply guided filter to preserve edges
    try:
        guide_gray = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY)
        alpha_up = cv2.ximgproc.guidedFilter(
            guide_gray.astype(np.float32),
            alpha_up.astype(np.float32),
            radius=8,
            eps=1e-4
        )
    except:
        # Fallback: bilateral filter
        alpha_up_8bit = (alpha_up * 255).astype(np.uint8)
        alpha_up = cv2.bilateralFilter(alpha_up_8bit, 9, 75, 75).astype(np.float32) / 255.0
    
    return np.clip(alpha_up, 0, 1)


