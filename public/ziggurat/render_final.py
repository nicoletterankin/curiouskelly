"""
Final Ziggurat LED Mockup Renderer
Produces stakeholder-ready before/after images
"""
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = img.size
print(f"Image size: {w}x{h}")

# Rainbow colors (purple crown to red base)
rainbow = [
    (139, 92, 246),   # Purple - Level 7 (crown)
    (59, 130, 246),   # Blue
    (6, 182, 212),    # Cyan
    (34, 197, 94),    # Green
    (234, 179, 8),    # Yellow
    (249, 115, 22),   # Orange
    (239, 68, 68),    # Red - Level 1 (base)
]

# VERIFIED terrace coordinates (y, left_x, right_x)
terraces = [
    (1180, 1500, 2000),  # Level 7 (crown)
    (1230, 1380, 2120),  # Level 6
    (1285, 1250, 2250),  # Level 5
    (1345, 1110, 2390),  # Level 4
    (1410, 970, 2540),   # Level 3
    (1475, 820, 2695),   # Level 2
    (1545, 670, 2850),   # Level 1 (base)
]

def create_night_scene(base_img):
    """Create nighttime atmosphere"""
    arr = np.array(base_img).astype(np.float32)
    
    # Darken and add blue tint for night
    arr[:,:,0] *= 0.25  # Red
    arr[:,:,1] *= 0.28  # Green
    arr[:,:,2] *= 0.35  # Blue (less darkened = blue tint)
    
    return arr

def add_led_bands(arr, glow_intensity=1.0):
    """Add LED bands with glow effect"""
    h, w = arr.shape[:2]
    
    for i, ((y, lx, rx), color) in enumerate(zip(terraces, rainbow)):
        thickness = 12 + i * 3
        y1, y2 = y - thickness//2, y + thickness//2
        
        # Multi-layer glow (outer to inner)
        for glow_size in [60, 45, 30, 18, 8]:
            gy1 = max(0, y - glow_size)
            gy2 = min(h, y + glow_size)
            glx = max(0, lx - glow_size//2)
            grx = min(w, rx + glow_size//2)
            
            alpha = 0.12 * (60 - glow_size) / 60 * glow_intensity
            arr[gy1:gy2, glx:grx, 0] += color[0] * alpha
            arr[gy1:gy2, glx:grx, 1] += color[1] * alpha
            arr[gy1:gy2, glx:grx, 2] += color[2] * alpha
        
        # Core LED band
        arr[y1:y2, lx:rx, 0] = np.maximum(arr[y1:y2, lx:rx, 0], color[0] * 0.9)
        arr[y1:y2, lx:rx, 1] = np.maximum(arr[y1:y2, lx:rx, 1], color[1] * 0.9)
        arr[y1:y2, lx:rx, 2] = np.maximum(arr[y1:y2, lx:rx, 2], color[2] * 0.9)
        
        # White-hot center line
        cy = y
        arr[cy-1:cy+1, lx:rx, :] = 255
    
    return arr

def add_sky_glow(arr):
    """Add subtle glow reflection in sky"""
    h, w = arr.shape[:2]
    
    # Gradient glow above building
    for y in range(800, 1180):
        factor = (1180 - y) / 380 * 0.08
        arr[y, 1000:2500, 0] += 180 * factor
        arr[y, 1000:2500, 1] += 120 * factor
        arr[y, 1000:2500, 2] += 200 * factor
    
    return arr

# === RENDER NIGHT VERSION ===
print("Rendering night version...")
night_arr = create_night_scene(img)
night_arr = add_led_bands(night_arr, glow_intensity=1.2)
night_arr = add_sky_glow(night_arr)
night_arr = np.clip(night_arr, 0, 255)
night_img = Image.fromarray(night_arr.astype(np.uint8))

# === RENDER DUSK VERSION ===
print("Rendering dusk version...")
dusk_arr = np.array(img).astype(np.float32)
# Warm dusk tones
dusk_arr[:,:,0] *= 0.65
dusk_arr[:,:,1] *= 0.55
dusk_arr[:,:,2] *= 0.45
dusk_arr = add_led_bands(dusk_arr, glow_intensity=0.8)
dusk_arr = np.clip(dusk_arr, 0, 255)
dusk_img = Image.fromarray(dusk_arr.astype(np.uint8))

# === SAVE OUTPUTS ===
out_dir = script_dir

# Before
before_path = os.path.join(out_dir, "BEFORE.png")
img.save(before_path, quality=95)
print(f"Saved: {before_path}")

# Night
night_path = os.path.join(out_dir, "AFTER-night.png")
night_img.save(night_path, quality=95)
print(f"Saved: {night_path}")

# Dusk
dusk_path = os.path.join(out_dir, "AFTER-dusk.png")
dusk_img.save(dusk_path, quality=95)
print(f"Saved: {dusk_path}")

# === CREATE SIDE-BY-SIDE ===
print("Creating side-by-side comparison...")
comparison = Image.new('RGB', (w * 2, h), (0, 0, 0))
comparison.paste(img, (0, 0))
comparison.paste(night_img, (w, 0))

# Add labels
draw = ImageDraw.Draw(comparison)
draw.rectangle([0, 0, 300, 60], fill=(0, 0, 0, 180))
draw.rectangle([w, 0, w + 300, 60], fill=(0, 0, 0, 180))
draw.text((20, 15), "BEFORE", fill=(255, 255, 255))
draw.text((w + 20, 15), "AFTER", fill=(255, 255, 255))

comparison_path = os.path.join(out_dir, "BEFORE-AFTER.png")
comparison.save(comparison_path, quality=95)
print(f"Saved: {comparison_path}")

print("\n=== COMPLETE ===")
print(f"Files in: {out_dir}")
