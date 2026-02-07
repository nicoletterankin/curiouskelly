"""
Ziggurat LED Vision — Precision Edge Tracing
Sub-pixel accurate tier boundaries from visual analysis
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "precision")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT — PRECISION EDGE TRACING")
print("=" * 60)

before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

# Reference: 1920×1055 (the actual scaled output size)
sx = w / 1920
sy = h / 1055
print(f"Scale: X={sx:.4f}, Y={sy:.4f}")

# === PRECISION TRACED COORDINATES ===
# 7 visible stepped tiers (not 8 - base is separate structure)
# Building center: X ≈ 890 (46.4% from left)
# Asymmetric due to camera angle

TIERS_1920 = [
    # T7 (Crown/Apex) - small mechanical level at top
    {"name": "T7", "top": 474, "bottom": 498,
     "left_top": 855, "right_top": 928,
     "left_bot": 805, "right_bot": 980},
    
    # T6
    {"name": "T6", "top": 498, "bottom": 528,
     "left_top": 805, "right_top": 980,
     "left_bot": 752, "right_bot": 1038},
    
    # T5
    {"name": "T5", "top": 528, "bottom": 562,
     "left_top": 752, "right_top": 1038,
     "left_bot": 692, "right_bot": 1100},
    
    # T4
    {"name": "T4", "top": 562, "bottom": 600,
     "left_top": 692, "right_top": 1100,
     "left_bot": 625, "right_bot": 1168},
    
    # T3
    {"name": "T3", "top": 600, "bottom": 645,
     "left_top": 625, "right_top": 1168,
     "left_bot": 550, "right_bot": 1245},
    
    # T2
    {"name": "T2", "top": 645, "bottom": 695,
     "left_top": 550, "right_top": 1245,
     "left_bot": 468, "right_bot": 1332},
    
    # T1 (widest stepped tier)
    {"name": "T1", "top": 695, "bottom": 750,
     "left_top": 468, "right_top": 1332,
     "left_bot": 378, "right_bot": 1425},
]

# Scale to actual image size
TIERS = [{
    "name": t["name"],
    "top": int(t["top"] * sy),
    "bottom": int(t["bottom"] * sy),
    "left_top": int(t["left_top"] * sx),
    "right_top": int(t["right_top"] * sx),
    "left_bot": int(t["left_bot"] * sx),
    "right_bot": int(t["right_bot"] * sx),
} for t in TIERS_1920]

print("\nPrecision tier measurements:")
for t in TIERS:
    width = t["right_bot"] - t["left_bot"]
    center = (t["left_bot"] + t["right_bot"]) / 2
    print(f"  {t['name']}: Y={t['top']:4d}-{t['bottom']:4d}, W={width:4d}px ({width/w*100:5.1f}%), center={center/w*100:.1f}%")

# Build hull for building mask
HULL = []
# Top edge (apex)
HULL.append((TIERS[0]["left_top"], TIERS[0]["top"]))
HULL.append((TIERS[0]["right_top"], TIERS[0]["top"]))
# Right side descending
for t in TIERS:
    HULL.append((t["right_bot"], t["bottom"]))
# Left side ascending
for t in reversed(TIERS):
    HULL.append((t["left_bot"], t["bottom"]))

print(f"\nHull vertices: {len(HULL)}")

# === CREATE MASKS ===
print("Creating masks...")

# Building hull mask
building_mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(building_mask).polygon(HULL, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(2))
building_maskf = np.array(building_mask, dtype=np.float32) / 255.0

# Individual tier trapezoid masks
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
    tier_masks.append(np.array(mask, dtype=np.float32) / 255.0)

# === PALETTES (refined colors) ===
PALETTES = {
    'rainbow': [[147,51,234], [59,130,246], [6,182,212], [34,197,94], [234,179,8], [249,115,22], [239,68,68]],
    'cool': [[147,51,234], [99,102,241], [59,130,246], [14,165,233], [6,182,212], [20,184,166], [16,185,129]],
    'warm': [[251,191,36], [245,158,11], [249,115,22], [239,68,68], [236,72,153], [217,70,239], [168,85,247]],
    'white': [[255,253,250]] * 7,
    'gold': [[255,215,0], [255,207,0], [255,198,0], [255,189,0], [255,180,0], [255,170,0], [255,160,0]],
    'cyan': [[0,255,255], [0,240,250], [0,225,240], [0,210,230], [0,195,220], [0,180,210], [0,165,200]],
    'usa': [[239,68,68], [255,255,255], [59,130,246], [255,255,255], [239,68,68], [255,255,255], [59,130,246]],
}

TIMES = {
    'night': {'grade': (0.14, 0.18, 0.30), 'brightness': 1.0, 'sky_dark': 0.15},
    'late-night': {'grade': (0.08, 0.10, 0.20), 'brightness': 1.2, 'sky_dark': 0.08},
    'twilight': {'grade': (0.26, 0.26, 0.42), 'brightness': 0.75, 'sky_dark': 0.30},
    'dusk': {'grade': (0.52, 0.42, 0.38), 'brightness': 0.52, 'sky_dark': 0.58},
}

RESOLUTIONS = {
    'full': (w, h),
    '4k': (3840, 2111),
    '1080p': (1920, 1055),
    'thumb': (480, 264),
}

# === RENDER FUNCTION ===
def render(palette_key, time_key):
    palette = [np.array(c, dtype=np.float32) for c in PALETTES[palette_key]]
    settings = TIMES[time_key]
    
    arr = np.array(before, dtype=np.float32)
    
    # Time grading
    arr[:, :, 0] *= settings['grade'][0]
    arr[:, :, 1] *= settings['grade'][1]
    arr[:, :, 2] *= settings['grade'][2]
    
    # Sky darkening above building
    apex_y = TIERS[0]["top"]
    for y in range(apex_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / apex_y) ** 0.7
        arr[y, :, :] *= factor
    
    # Create emission map
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    # Brightness falloff: apex brightest
    brightness_scale = [1.0, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58]
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)]
        
        # LED band at tier bottom (overhang shadow line)
        band_y = t["bottom"]
        thickness = int(18 * sy)
        
        # Create horizontal band
        band = np.zeros((h, w), dtype=np.float32)
        y0 = max(0, band_y - thickness)
        y1 = min(h, band_y + 5)
        band[y0:y1, :] = 1.0
        
        # Constrain to tier trapezoid
        band *= tmask
        
        # Gaussian glow
        band_img = Image.fromarray((band * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(14 * sx)))
        band = np.array(band_img, dtype=np.float32) / 255.0
        band *= building_maskf
        
        # Add emission
        intensity = 280 * settings['brightness'] * brightness_scale[i]
        emit += band[..., None] * (col / 255.0) * intensity
        
        # Hot center line (crisp LED edge)
        center = np.zeros((h, w), dtype=np.float32)
        cy = band_y
        center[max(0, cy-5):min(h, cy+5), :] = 1.0
        center *= tmask * building_maskf
        emit += center[..., None] * intensity * 0.30
    
    # Screen blend
    base = arr / 255.0
    light = np.clip(emit / 255.0, 0, 1)
    blended = 1 - (1 - base) * (1 - light)
    
    # Spill control
    spill = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill = spill.filter(ImageFilter.GaussianBlur(int(55 * sx)))
    spillf = np.array(spill, dtype=np.float32) / 255.0
    
    final = base * (1 - spillf[..., None]) + blended * spillf[..., None]
    
    # Sky glow
    glow = np.zeros((h, w), dtype=np.float32)
    glow_h = int(70 * sy)
    for dy in range(glow_h):
        y = apex_y - dy
        if y < 0: break
        falloff = (1.0 - dy / glow_h) ** 1.8
        glow[y, :] = falloff * 0.06
    
    top_col = palette[0] / 255.0
    final = final + glow[..., None] * top_col * 0.15
    
    # Output
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.03)
    return img

# === BATCH RENDER ===
total = len(PALETTES) * len(TIMES)
print(f"\nRendering {total} variants × {len(RESOLUTIONS)} resolutions...")

# Before images
for res, (rw, rh) in RESOLUTIONS.items():
    before.resize((rw, rh), Image.Resampling.LANCZOS).save(
        os.path.join(out_dir, f"before-{res}.jpg"), quality=93)
print("  before saved")

# Variants
for p in PALETTES:
    for t in TIMES:
        v = f"{p}-{t}"
        print(f"  {v}")
        img = render(p, t)
        for res, (rw, rh) in RESOLUTIONS.items():
            img.resize((rw, rh), Image.Resampling.LANCZOS).save(
                os.path.join(out_dir, f"{v}-{res}.jpg"), quality=93)

# Summary
files = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
print(f"\n{'='*60}")
print(f"COMPLETE: {len(files)} images, {size/1024/1024:.1f} MB")
print(f"Output: {out_dir}")
print("="*60)
