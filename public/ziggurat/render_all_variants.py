"""
Ziggurat LED Vision — Complete Variant Batch Renderer
Generates all color schemes × time-of-day × resolutions
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import os
from scipy.signal import convolve2d

# === SETUP ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "variants")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT LED VISION — BATCH VARIANT RENDERER")
print("=" * 60)

# Load source
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

arr = np.array(before).astype(np.float32)

# === HSV MASK ===
def rgb_to_hsv(a):
    a = a / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    diff = mx - mn
    hue = np.zeros_like(mx)
    sat = np.zeros_like(mx)
    val = mx
    mask = diff != 0
    rm = (mx == r) & mask
    gm = (mx == g) & mask
    bm = (mx == b) & mask
    hue[rm] = (60 * ((g[rm] - b[rm]) / diff[rm]) + 360) % 360
    hue[gm] = (60 * ((b[gm] - r[gm]) / diff[gm]) + 120) % 360
    hue[bm] = (60 * ((r[bm] - g[bm]) / diff[bm]) + 240) % 360
    sat[mx != 0] = diff[mx != 0] / mx[mx != 0]
    return hue, sat, val

H, S, V = rgb_to_hsv(arr)
mask = (H > 20) & (H < 80) & (S > 0.06) & (V > 0.28) & (V < 0.96)
mask &= ~((H > 80) & (H < 170) & (S > 0.14))
mask &= (V > 0.32)

mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
mask_clean = (np.array(mask_img) > 95).astype(np.uint8) * 255
mask_clean_img = Image.fromarray(mask_clean).filter(ImageFilter.GaussianBlur(2))
maskf = (np.array(mask_clean_img) / 255.0).astype(np.float32)

# === EDGE DETECTION ===
print("Detecting terrace edges...")
gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
gy = convolve2d(gray, ky, mode='same', boundary='symm')
edge_h = np.abs(gy) * maskf

row_strength = edge_h.mean(axis=1)
ymin, ymax = 1180, 1550
roi = row_strength.copy()
roi[:ymin] = 0
roi[ymax + 1:] = 0

N, min_sep = 7, 40
peaks = []
roi_copy = roi.copy()
for _ in range(N * 3):
    y = int(np.argmax(roi_copy))
    if roi_copy[y] <= 0:
        break
    if all(abs(y - p) > min_sep for p in peaks):
        peaks.append(y)
        if len(peaks) >= N:
            break
    roi_copy[max(0, y - min_sep):min(h, y + min_sep)] = 0

peaks = sorted(peaks)
print(f"Terrace edges: {peaks}")

# === COLOR PALETTES ===
PALETTES = {
    'rainbow': [
        np.array([139, 92, 246]),   # Purple
        np.array([59, 130, 246]),   # Blue
        np.array([6, 182, 212]),    # Cyan
        np.array([34, 197, 94]),    # Green
        np.array([234, 179, 8]),    # Yellow
        np.array([249, 115, 22]),   # Orange
        np.array([239, 68, 68]),    # Red
    ],
    'cool': [
        np.array([139, 92, 246]),   # Purple
        np.array([99, 102, 241]),   # Indigo
        np.array([59, 130, 246]),   # Blue
        np.array([14, 165, 233]),   # Sky
        np.array([6, 182, 212]),    # Cyan
        np.array([20, 184, 166]),   # Teal
        np.array([16, 185, 129]),   # Emerald
    ],
    'warm': [
        np.array([251, 191, 36]),   # Amber
        np.array([245, 158, 11]),   # Amber-dark
        np.array([249, 115, 22]),   # Orange
        np.array([239, 68, 68]),    # Red
        np.array([236, 72, 153]),   # Pink
        np.array([217, 70, 239]),   # Fuchsia
        np.array([168, 85, 247]),   # Purple
    ],
    'mono-white': [
        np.array([255, 255, 255]),
        np.array([245, 245, 245]),
        np.array([235, 235, 235]),
        np.array([225, 225, 225]),
        np.array([215, 215, 215]),
        np.array([205, 205, 205]),
        np.array([195, 195, 195]),
    ],
    'mono-gold': [
        np.array([255, 215, 0]),
        np.array([255, 205, 0]),
        np.array([255, 195, 0]),
        np.array([255, 185, 0]),
        np.array([255, 175, 0]),
        np.array([255, 165, 0]),
        np.array([255, 155, 0]),
    ],
    'mono-cyan': [
        np.array([0, 255, 255]),
        np.array([0, 235, 235]),
        np.array([0, 215, 215]),
        np.array([0, 195, 195]),
        np.array([0, 175, 175]),
        np.array([0, 155, 155]),
        np.array([0, 135, 135]),
    ],
    'mono-purple': [
        np.array([168, 85, 247]),
        np.array([158, 80, 237]),
        np.array([148, 75, 227]),
        np.array([138, 70, 217]),
        np.array([128, 65, 207]),
        np.array([118, 60, 197]),
        np.array([108, 55, 187]),
    ],
    'usa': [
        np.array([239, 68, 68]),    # Red
        np.array([255, 255, 255]),  # White
        np.array([59, 130, 246]),   # Blue
        np.array([255, 255, 255]),  # White
        np.array([239, 68, 68]),    # Red
        np.array([255, 255, 255]),  # White
        np.array([59, 130, 246]),   # Blue
    ],
    'sunset': [
        np.array([239, 68, 68]),    # Red
        np.array([249, 115, 22]),   # Orange
        np.array([251, 146, 60]),   # Orange-light
        np.array([253, 186, 116]),  # Peach
        np.array([254, 215, 170]),  # Cream
        np.array([255, 237, 213]),  # Light
        np.array([255, 247, 237]),  # Near white
    ],
    'ocean': [
        np.array([30, 58, 138]),    # Blue-900
        np.array([29, 78, 216]),    # Blue-700
        np.array([37, 99, 235]),    # Blue-600
        np.array([59, 130, 246]),   # Blue-500
        np.array([96, 165, 250]),   # Blue-400
        np.array([147, 197, 253]),  # Blue-300
        np.array([191, 219, 254]),  # Blue-200
    ],
}

# === TIME OF DAY SETTINGS ===
TIME_SETTINGS = {
    'night': {
        'grade': (0.25, 0.30, 0.42),
        'brightness': 180,
        'contrast': 1.08,
        'saturation': 0.95,
    },
    'late-night': {
        'grade': (0.18, 0.20, 0.30),
        'brightness': 200,
        'contrast': 1.12,
        'saturation': 0.90,
    },
    'dusk': {
        'grade': (0.70, 0.60, 0.50),
        'brightness': 120,
        'contrast': 1.04,
        'saturation': 1.05,
    },
    'golden': {
        'grade': (0.85, 0.75, 0.60),
        'brightness': 90,
        'contrast': 1.03,
        'saturation': 1.10,
    },
    'day': {
        'grade': (0.95, 0.95, 0.95),
        'brightness': 60,
        'contrast': 1.02,
        'saturation': 0.98,
    },
    'overcast': {
        'grade': (0.80, 0.82, 0.88),
        'brightness': 70,
        'contrast': 1.01,
        'saturation': 0.90,
    },
}

# === RENDER FUNCTION ===
def render_variant(palette_name, time_name):
    palette = PALETTES[palette_name]
    settings = TIME_SETTINGS[time_name]
    
    # Apply time grading
    result = arr.copy()
    result[:, :, 0] *= settings['grade'][0]
    result[:, :, 1] *= settings['grade'][1]
    result[:, :, 2] *= settings['grade'][2]
    
    # Create emission map
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    for i, y in enumerate(peaks):
        thickness = 14 + i * 3
        band = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, y - thickness // 2), min(h, y + thickness // 2)
        band[y0:y1, :] = 1.0
        band *= maskf
        band = np.array(Image.fromarray((band * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(4))) / 255.0
        
        col = palette[min(i, len(palette) - 1)].astype(np.float32)
        emit += band[..., None] * (col / 255.0) * settings['brightness']
        
        # White-hot center for night modes
        if settings['brightness'] > 100:
            center = np.zeros((h, w), dtype=np.float32)
            center[y-1:y+1, :] = 1.0
            center *= maskf
            emit += center[..., None] * 0.4 * settings['brightness']
    
    # Screen blend
    a = result / 255.0
    b = np.clip(emit, 0, 255) / 255.0
    blended = 1 - (1 - a) * (1 - b)
    
    # Spill guard
    spill = maskf[..., None] * 0.92 + 0.08
    blended = a * (1 - maskf[..., None]) + blended * spill
    
    out = np.clip(blended * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    
    # Post-processing
    img = ImageEnhance.Contrast(img).enhance(settings['contrast'])
    img = ImageEnhance.Color(img).enhance(settings['saturation'])
    
    return img

# === RESOLUTIONS ===
RESOLUTIONS = {
    'full': (w, h),           # 4032×2217
    '4k': (3840, 2111),       # 4K
    '2k': (2560, 1407),       # 2K
    '1080p': (1920, 1055),    # 1080p
    '720p': (1280, 703),      # 720p
    'thumb': (640, 352),      # Thumbnail
}

# === BATCH RENDER ===
total_variants = len(PALETTES) * len(TIME_SETTINGS)
print(f"\nRendering {total_variants} variants × {len(RESOLUTIONS)} resolutions...")
print(f"Total files: {total_variants * len(RESOLUTIONS) + len(RESOLUTIONS)}")  # + before images

# Save before at all resolutions
print("\n[BEFORE] Saving source images...")
for res_name, (rw, rh) in RESOLUTIONS.items():
    before_r = before.resize((rw, rh), Image.Resampling.LANCZOS)
    path = os.path.join(out_dir, f"before-{res_name}.jpg")
    before_r.save(path, quality=92)
    print(f"  before-{res_name}.jpg")

# Render all variants
count = 0
for palette_name in PALETTES:
    for time_name in TIME_SETTINGS:
        count += 1
        variant_name = f"{palette_name}-{time_name}"
        print(f"\n[{count}/{total_variants}] {variant_name}")
        
        # Render full resolution
        img = render_variant(palette_name, time_name)
        
        # Save at all resolutions
        for res_name, (rw, rh) in RESOLUTIONS.items():
            img_r = img.resize((rw, rh), Image.Resampling.LANCZOS)
            path = os.path.join(out_dir, f"{variant_name}-{res_name}.jpg")
            img_r.save(path, quality=92)
            print(f"  {variant_name}-{res_name}.jpg")

# === GENERATE MANIFEST ===
print("\n\nGenerating manifest...")
manifest = {
    "palettes": list(PALETTES.keys()),
    "times": list(TIME_SETTINGS.keys()),
    "resolutions": list(RESOLUTIONS.keys()),
    "terraceEdges": peaks,
    "variants": []
}

for palette_name in PALETTES:
    for time_name in TIME_SETTINGS:
        variant_name = f"{palette_name}-{time_name}"
        manifest["variants"].append({
            "id": variant_name,
            "palette": palette_name,
            "time": time_name,
            "files": {
                res: f"{variant_name}-{res}.jpg" 
                for res in RESOLUTIONS
            }
        })

import json
manifest_path = os.path.join(out_dir, "manifest.json")
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"Saved: {manifest_path}")

# === SUMMARY ===
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"Output directory: {out_dir}")
print(f"Palettes: {len(PALETTES)}")
print(f"Time settings: {len(TIME_SETTINGS)}")
print(f"Resolutions: {len(RESOLUTIONS)}")
print(f"Total variants: {total_variants}")
print(f"Total files: {total_variants * len(RESOLUTIONS) + len(RESOLUTIONS)}")

# List files
files = sorted(os.listdir(out_dir))
total_size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
