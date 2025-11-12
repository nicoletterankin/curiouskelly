"""
Video frame extraction utilities
"""
import numpy as np
from typing import Optional


def extract_midframe(video_path: str, target_second: float = 2.0) -> Optional[np.ndarray]:
    """
    Extract a frame from video at specified timestamp.
    
    Args:
        video_path: Path to video file
        target_second: Timestamp in seconds
    
    Returns:
        RGB frame [H, W, 3] or None if failed
    """
    try:
        import imageio.v3 as iio
        
        # Read video metadata
        props = iio.improps(video_path)
        fps = props.fps if hasattr(props, 'fps') else 30
        
        # Calculate target frame
        target_frame = int(target_second * fps)
        
        # Read frame
        frame = iio.imread(video_path, index=target_frame, plugin="pyav")
        
        # Convert to RGB if needed
        if frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]
        
        return frame
        
    except ImportError:
        print("Warning: imageio not installed, cannot extract video frame")
        return None
    except Exception as e:
        print(f"Warning: Could not extract video frame: {e}")
        return None


