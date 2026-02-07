"""
Ziggurat LED Vision — Refined Polygon Renderer
Corrected: wider base (90%), lower tier positions
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os
import json

# === SETUP ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "refined")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT — REFINED POLYGON RENDERER")
print("=" * 60)

# Load source
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

# === SCALE FACTORS ===
sx = w / 1920
sy = h / 1080
print(f"Scale: X={sx:.3f}, Y={sy:.3f}")

# === REFINED BUILDING POLYGON ===
# Base is 90% width (5% to 95%), tiers shifted down
# Apex moved down, all tiers proportionally adjusted

# For 1920×1080 reference:
# - Base: X = 5% to 95% = 96 to 1824 (was 442-1478)
# - Apex: shifted down from Y=475 to Y=520
# - Base bottom: shifted down from Y=810 to Y=850

BUILDING_POLYGON_1080 = [
    # Apex (narrower, centered)
    (960, 520),
    # Right side - widening as we go down
    (1050, 545),   # T7 right
    (1150, 575),   # T6 right
    (1260, 610),   # T5 right
    (1380, 650),   # T4 right
    (1500, 695),   # T3 right
    (1620, 745),   # T2 right
    (1740, 800),   # T1 right
    (1824, 850),   # Base right (95%)
    # Bottom edge
    (96, 850),     # Base left (5%)
    # Left side - ascending
    (180, 800),    # T1 left
    (300, 745),    # T2 left
    (420, 695),    # T3 left
    (540, 650),    # T4 left
    (660, 610),    # T5 left
    (770, 575),    # T6 left
    (870, 545),    # T7 left
]

BUILDING_POLYGON = [(int(x * sx), int(y * sy)) for x, y in BUILDING_POLYGON_1080]
print(f"Building polygon: apex=({BUILDING_POLYGON[0]}), base width={BUILDING_POLYGON[8][0] - BUILDING_POLYGON[9][0]}px")

# === REFINED TIER DEFINITIONS ===
# Each tier is a trapezoid slice with proper width progression
TIERS_1080 = [
    # T7 (apex) - narrowest
    {"name": "T7", "top": 520, "bottom": 545, 
     "left_top": 960, "right_top": 960,  # Apex point
     "left_bot": 870, "right_bot": 1050},
    
    # T6
    {"name": "T6", "top": 545, "bottom": 575, 
     "left_top": 870, "right_top": 1050,
     "left_bot": 770, "right_bot": 1150},
    
    # T5
    {"name": "T5", "top": 575, "bottom": 610, 
     "left_top": 770, "right_top": 1150,
     "left_bot": 660, "right_bot": 1260},
    
    # T4
    {"name": "T4", "top": 610, "bottom": 650, 
     "left_top": 660, "right_top": 1260,
     "left_bot": 540, "right_bot": 1380},
    
    # T3
    {"name": "T3", "top": 650, "bottom": 695, 
     "left_top": 540, "right_top": 1380,
     "left_bot": 420, "right_bot": 1500},
    
    # T2
    {"name": "T2", "top": 695, "bottom": 745, 
     "left_top": 420, "right_top": 1500,
     "left_bot": 300, "right_bot": 1620},
    
    # T1 (widest tier before base)
    {"name": "T1", "top": 745, "bottom": 800, 
     "left_top": 300, "right_top": 1620,
     "left_bot": 180, "right_bot": 1740},
    
    # Base
    {"name": "BASE", "top": 800, "bottom": 850, 
     "left_top": 180, "right_top": 1740,
     "left_bot": 96, "right_bot": 1824},
]

# Scale tiers
TIERS = [{
    "name": t["name"],
    "top": int(t["top"] * sy), "bottom": int(t["bottom"] * sy),
    "left_top": int(t["left_top"] * sx), "right_top": int(t["right_top"] * sx),
    "left_bot": int(t["left_bot"] * sx), "right_bot": int(t["right_bot"] * sx),
} for t in TIERS_1080]

for t in TIERS:
    width = t["right_bot"] - t["left_bot"]
    pct = width / w * 100
    print(f"  {t['name']}: Y={t['top']}-{t['bottom']}, width={width}px ({pct:.0f}%)")

# === CREATE MASKS ===
print("\nCreating masks...")

# Building mask
building_mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(building_mask).polygon(BUILDING_POLYGON, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(3))
building_maskf = np.array(building_mask).astype(np.float32) / 255.0

# Tier masks (trapezoids)
tier_masks = []
for t in TIERS:
    mask = Image.new('L', (w, h), 0)
    poly = [
        (t["left_top"], t["top"]),
        (t["right_top"], t["top"]),
        (t["right_bot"], t["bottom"]),
        (t["left_bot"], t["bottom"]),
    ]
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    tier_masks.append(np.array(mask).astype(np.float32) / 255.0)

# === PALETTES ===
PALETTES = {
    'rainbow': {
        'label': 'Rainbow',
        'colors': [[139,92,246], [59,130,246], [6,182,212], [34,197,94], [234,179,8], [249,115,22], [239,68,68], [220,38,38]],
    },
    'cool': {
        'label': 'Cool Tones',
        'colors': [[139,92,246], [99,102,241], [59,130,246], [14,165,233], [6,182,212], [20,184,166], [16,185,129], [5,150,105]],
    },
    'warm': {
        'label': 'Warm Tones',
        'colors': [[251,191,36], [245,158,11], [249,115,22], [239,68,68], [236,72,153], [217,70,239], [168,85,247], [139,92,246]],
    },
    'white': {
        'label': 'Warm White',
        'colors': [[255,250,245], [255,248,240], [255,245,235], [255,242,230], [255,240,225], [255,238,220], [255,235,215], [255,232,210]],
    },
    'gold': {
        'label': 'Gold',
        'colors': [[255,223,0], [255,215,0], [255,205,0], [255,195,0], [255,185,0], [255,175,0], [255,165,0], [255,155,0]],
    },
    'cyan': {
        'label': 'Cyan',
        'colors': [[0,255,255], [0,245,245], [0,235,235], [0,220,220], [0,205,205], [0,190,190], [0,175,175], [0,160,160]],
    },
    'purple': {
        'label': 'Purple',
        'colors': [[168,85,247], [158,80,237], [148,75,227], [138,70,217], [128,65,207], [118,60,197], [108,55,187], [98,50,177]],
    },
    'usa': {
        'label': 'USA',
        'colors': [[239,68,68], [255,255,255], [59,130,246], [255,255,255], [239,68,68], [255,255,255], [59,130,246], [239,68,68]],
    },
}

# === TIME SETTINGS ===
TIMES = {
    'night': {'label': 'Night', 'grade': (0.20, 0.24, 0.36), 'brightness': 1.0, 'sky_dark': 0.22},
    'late-night': {'label': 'Late Night', 'grade': (0.14, 0.16, 0.26), 'brightness': 1.1, 'sky_dark': 0.15},
    'twilight': {'label': 'Twilight', 'grade': (0.32, 0.32, 0.48), 'brightness': 0.85, 'sky_dark': 0.38},
    'dusk': {'label': 'Dusk', 'grade': (0.62, 0.52, 0.45), 'brightness': 0.65, 'sky_dark': 0.68},
}

# === RESOLUTIONS ===
RESOLUTIONS = {
    'full': (w, h),
    '4k': (3840, 2111),
    '1080p': (1920, 1055),
    'thumb': (480, 264),
}

# === RENDER FUNCTION ===
def render(palette_key, time_key):
    palette = [np.array(c) for c in PALETTES[palette_key]['colors']]
    settings = TIMES[time_key]
    
    arr = np.array(before).astype(np.float32)
    arr[:, :, 0] *= settings['grade'][0]
    arr[:, :, 1] *= settings['grade'][1]
    arr[:, :, 2] *= settings['grade'][2]
    
    # Darken sky (above apex)
    apex_y = int(520 * sy)
    for y in range(apex_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / apex_y) * 0.5
        arr[y, :, :] *= factor
    
    # Emission per tier
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    # Brightness gradient: T7=brightest, BASE=dimmest
    brightness_scale = [1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76, 0.72]
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)].astype(np.float32)
        
        # LED band at bottom edge of tier
        band_y = t["bottom"]
        thickness = int(14 * sy)  # Slightly thicker
        
        band = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, band_y - thickness), min(h, band_y + 6)
        band[y0:y1, :] = 1.0
        band *= tmask
        
        # Blur for glow
        band_img = Image.fromarray((band * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(10 * sx)))
        band = np.array(band_img).astype(np.float32) / 255.0
        band *= building_maskf
        
        intensity = 220 * settings['brightness'] * brightness_scale[i]
        emit += band[..., None] * (col / 255.0) * intensity
        
        # White-hot center line
        center = np.zeros((h, w), dtype=np.float32)
        center[band_y-3:band_y+3, :] = 1.0
        center *= tmask * building_maskf
        emit += center[..., None] * intensity * 0.45
    
    # Screen blend
    base = arr / 255.0
    light = np.clip(emit, 0, 255) / 255.0
    blended = 1 - (1 - base) * (1 - light)
    
    # Spill control
    spill = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill = spill.filter(ImageFilter.GaussianBlur(int(35 * sx)))
    spillf = np.array(spill).astype(np.float32) / 255.0
    
    final = base * (1 - spillf[..., None]) + blended * spillf[..., None]
    
    # Sky glow above apex
    glow = np.zeros((h, w), dtype=np.float32)
    for dy in range(int(120 * sy)):
        y = apex_y - dy
        if y < 0: break
        glow[y, :] = (1.0 - dy / (120 * sy)) * 0.12
    
    top_col = palette[0].astype(np.float32) / 255.0
    final = final + glow[..., None] * top_col * 0.25
    
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.06)
    return img

# === BATCH RENDER ===
total = len(PALETTES) * len(TIMES)
print(f"\nRendering {total} variants × {len(RESOLUTIONS)} resolutions...")

# Before images
print("\n[BEFORE]")
for res, (rw, rh) in RESOLUTIONS.items():
    path = os.path.join(out_dir, f"before-{res}.jpg")
    before.resize((rw, rh), Image.Resampling.LANCZOS).save(path, quality=92)
    print(f"  before-{res}.jpg")

# All variants
count = 0
for p in PALETTES:
    for t in TIMES:
        count += 1
        variant = f"{p}-{t}"
        print(f"\n[{count}/{total}] {variant}")
        
        img = render(p, t)
        for res, (rw, rh) in RESOLUTIONS.items():
            path = os.path.join(out_dir, f"{variant}-{res}.jpg")
            img.resize((rw, rh), Image.Resampling.LANCZOS).save(path, quality=92)
            print(f"  {variant}-{res}.jpg")

# === MANIFEST ===
manifest = {
    "palettes": {k: v['label'] for k, v in PALETTES.items()},
    "times": {k: v['label'] for k, v in TIMES.items()},
    "resolutions": list(RESOLUTIONS.keys()),
    "tiers": 8,
    "baseWidth": "90%",
}
with open(os.path.join(out_dir, "manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2)

# === SUMMARY ===
files = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
print(f"\n{'='*60}")
print(f"COMPLETE: {len(files)} files, {size/1024/1024:.1f} MB")
print(f"Output: {out_dir}")
print("=" * 60)
