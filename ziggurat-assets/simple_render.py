"""Simple LED renderer - direct drawing without complex blending"""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = img.size

# Rainbow colors (purple top to red bottom)
rainbow = [
    (139, 92, 246),   # Purple - Level 7 (crown)
    (59, 130, 246),   # Blue
    (6, 182, 212),    # Cyan
    (34, 197, 94),    # Green
    (234, 179, 8),    # Yellow
    (249, 115, 22),   # Orange
    (239, 68, 68),    # Red - Level 1 (base)
]

# Terrace coordinates (y, left_x, right_x) - CORRECTED +450 pixels
# Building crown is at ~Y=1180, base at ~Y=1520
terraces = [
    (1180, 1500, 2000),  # Level 7 (crown)
    (1230, 1380, 2120),  # Level 6
    (1285, 1250, 2250),  # Level 5
    (1345, 1110, 2390),  # Level 4
    (1410, 970, 2540),   # Level 3
    (1475, 820, 2695),   # Level 2
    (1545, 670, 2850),   # Level 1 (base)
]

# Create night version
arr = np.array(img).astype(np.float32)

# Darken for night
arr = arr * 0.3  # Make it darker

# Draw LED bands
for i, ((y, lx, rx), color) in enumerate(zip(terraces, rainbow)):
    thickness = 15 + i * 4
    y1, y2 = y - thickness//2, y + thickness//2
    
    # Create glow (larger area, semi-transparent)
    for glow_size in [40, 30, 20, 10]:
        gy1 = max(0, y - glow_size)
        gy2 = min(h, y + glow_size)
        glx = max(0, lx - glow_size)
        grx = min(w, rx + glow_size)
        alpha = 0.15 * (40 - glow_size) / 40
        arr[gy1:gy2, glx:grx, 0] += color[0] * alpha
        arr[gy1:gy2, glx:grx, 1] += color[1] * alpha
        arr[gy1:gy2, glx:grx, 2] += color[2] * alpha
    
    # Draw solid LED band
    arr[y1:y2, lx:rx, 0] = color[0]
    arr[y1:y2, lx:rx, 1] = color[1]
    arr[y1:y2, lx:rx, 2] = color[2]
    
    # White center line
    cy = y
    arr[cy-2:cy+2, lx:rx, :] = 255

# Clamp and save
arr = np.clip(arr, 0, 255)
result = Image.fromarray(arr.astype(np.uint8))
result.save(os.path.join(script_dir, "ziggurat-simple-night.png"))

# Also save before
img.save(os.path.join(script_dir, "ziggurat-simple-before.png"))

print(f"Saved ziggurat-simple-night.png and ziggurat-simple-before.png")
print(f"LED bands at Y positions: {[t[0] for t in terraces]}")
