"""
Ziggurat LED Vision — Accurate Building Trace
Pixel-precise coordinates traced from aerial.jpg
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "accurate")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT — ACCURATE BUILDING TRACE")
print("=" * 60)

before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

# Scale from 1920×1055 reference to actual image
sx = w / 1920
sy = h / 1055  # Using 1055 as reference height (16:9 aspect from 1920)
print(f"Scale: X={sx:.3f}, Y={sy:.3f}")

# === PIXEL-TRACED TIER COORDINATES (1920×1055) ===
# Building center is at approximately X=890 (NOT 960 - building is left of center)
# Traced from actual building edges visible in image

TIERS_REF = [
    # T7 (Crown/Apex) - narrowest, mechanical level
    {"name": "T7",
     "top": 478, "bottom": 505,
     "left_top": 830, "right_top": 950,
     "left_bot": 780, "right_bot": 1000},
    
    # T6
    {"name": "T6",
     "top": 505, "bottom": 538,
     "left_top": 780, "right_top": 1000,
     "left_bot": 720, "right_bot": 1065},
    
    # T5
    {"name": "T5",
     "top": 538, "bottom": 575,
     "left_top": 720, "right_top": 1065,
     "left_bot": 650, "right_bot": 1140},
    
    # T4
    {"name": "T4",
     "top": 575, "bottom": 615,
     "left_top": 650, "right_top": 1140,
     "left_bot": 575, "right_bot": 1220},
    
    # T3
    {"name": "T3",
     "top": 615, "bottom": 660,
     "left_top": 575, "right_top": 1220,
     "left_bot": 495, "right_bot": 1305},
    
    # T2
    {"name": "T2",
     "top": 660, "bottom": 710,
     "left_top": 495, "right_top": 1305,
     "left_bot": 410, "right_bot": 1395},
    
    # T1 (widest stepped tier)
    {"name": "T1",
     "top": 710, "bottom": 760,
     "left_top": 410, "right_top": 1395,
     "left_bot": 330, "right_bot": 1480},
    
    # BASE (flat base structure below ziggurat)
    {"name": "BASE",
     "top": 760, "bottom": 800,
     "left_top": 330, "right_top": 1480,
     "left_bot": 280, "right_bot": 1535},
]

# Scale to actual image
TIERS = [{
    "name": t["name"],
    "top": int(t["top"] * sy),
    "bottom": int(t["bottom"] * sy),
    "left_top": int(t["left_top"] * sx),
    "right_top": int(t["right_top"] * sx),
    "left_bot": int(t["left_bot"] * sx),
    "right_bot": int(t["right_bot"] * sx),
} for t in TIERS_REF]

print("\nTier dimensions (actual pixels):")
for t in TIERS:
    width_top = t["right_top"] - t["left_top"]
    width_bot = t["right_bot"] - t["left_bot"]
    print(f"  {t['name']}: Y={t['top']}-{t['bottom']}, W={width_bot}px ({width_bot/w*100:.0f}%)")

# Build polygon from tier corners
BUILDING_POLYGON = []
# Right side (top to bottom)
for t in TIERS:
    BUILDING_POLYGON.append((t["right_top"], t["top"]))
for t in TIERS:
    BUILDING_POLYGON.append((t["right_bot"], t["bottom"]))
# Bottom
BUILDING_POLYGON.append((TIERS[-1]["left_bot"], TIERS[-1]["bottom"]))
# Left side (bottom to top)
for t in reversed(TIERS):
    BUILDING_POLYGON.append((t["left_bot"], t["bottom"]))
for t in reversed(TIERS):
    BUILDING_POLYGON.append((t["left_top"], t["top"]))

# Simplify to just the outer hull
HULL = [
    (TIERS[0]["left_top"], TIERS[0]["top"]),  # Apex left
    (TIERS[0]["right_top"], TIERS[0]["top"]), # Apex right
]
for t in TIERS:
    HULL.append((t["right_bot"], t["bottom"]))
HULL.append((TIERS[-1]["left_bot"], TIERS[-1]["bottom"]))
for t in reversed(TIERS):
    HULL.append((t["left_bot"], t["bottom"]))

print(f"\nBuilding hull: {len(HULL)} vertices")

# === CREATE MASKS ===
print("Creating masks...")

# Building mask from hull
building_mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(building_mask).polygon(HULL, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(2))
building_maskf = np.array(building_mask).astype(np.float32) / 255.0

# Tier masks (individual trapezoids)
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
    'rainbow': [[139,92,246], [59,130,246], [6,182,212], [34,197,94], [234,179,8], [249,115,22], [239,68,68], [220,38,38]],
    'cool': [[139,92,246], [99,102,241], [59,130,246], [14,165,233], [6,182,212], [20,184,166], [16,185,129], [5,150,105]],
    'warm': [[251,191,36], [245,158,11], [249,115,22], [239,68,68], [236,72,153], [217,70,239], [168,85,247], [139,92,246]],
    'white': [[255,252,248]] * 8,
    'gold': [[255,223,0], [255,215,0], [255,205,0], [255,195,0], [255,185,0], [255,175,0], [255,165,0], [255,155,0]],
    'usa': [[239,68,68], [255,255,255], [59,130,246], [255,255,255], [239,68,68], [255,255,255], [59,130,246], [239,68,68]],
}

TIMES = {
    'night': {'grade': (0.16, 0.20, 0.32), 'brightness': 1.0, 'sky_dark': 0.18},
    'late-night': {'grade': (0.10, 0.12, 0.22), 'brightness': 1.15, 'sky_dark': 0.10},
    'twilight': {'grade': (0.28, 0.28, 0.44), 'brightness': 0.78, 'sky_dark': 0.32},
    'dusk': {'grade': (0.55, 0.45, 0.40), 'brightness': 0.55, 'sky_dark': 0.62},
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
    
    # Apply time grading
    arr[:, :, 0] *= settings['grade'][0]
    arr[:, :, 1] *= settings['grade'][1]
    arr[:, :, 2] *= settings['grade'][2]
    
    # Darken sky above building
    apex_y = TIERS[0]["top"]
    for y in range(apex_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / apex_y) * 0.4
        arr[y, :, :] *= factor
    
    # Create emission map
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    # Brightness gradient: apex brightest, base dimmest
    brightness_scale = [1.0, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64, 0.58]
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)]
        
        # LED band at bottom of tier (where shadow/overhang is)
        band_y_center = t["bottom"]
        band_thickness = int(16 * sy)
        
        # Create band
        band = np.zeros((h, w), dtype=np.float32)
        y0 = max(0, band_y_center - band_thickness)
        y1 = min(h, band_y_center + 4)
        band[y0:y1, :] = 1.0
        
        # Clip to tier trapezoid
        band *= tmask
        
        # Gaussian blur for glow
        band_img = Image.fromarray((band * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(12 * sx)))
        band = np.array(band_img, dtype=np.float32) / 255.0
        
        # Re-clip to building mask
        band *= building_maskf
        
        # Add to emission
        intensity = 260 * settings['brightness'] * brightness_scale[i]
        emit += band[..., None] * (col / 255.0) * intensity
        
        # White-hot center line
        center = np.zeros((h, w), dtype=np.float32)
        cy = band_y_center
        center[max(0,cy-4):min(h,cy+4), :] = 1.0
        center *= tmask * building_maskf
        emit += center[..., None] * intensity * 0.35
    
    # Screen blend
    base = arr / 255.0
    light = np.clip(emit / 255.0, 0, 1)
    blended = 1 - (1 - base) * (1 - light)
    
    # Spill control - blend only within building + soft edge
    spill = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill = spill.filter(ImageFilter.GaussianBlur(int(50 * sx)))
    spillf = np.array(spill, dtype=np.float32) / 255.0
    
    final = base * (1 - spillf[..., None]) + blended * spillf[..., None]
    
    # Sky glow above apex
    glow = np.zeros((h, w), dtype=np.float32)
    glow_height = int(80 * sy)
    for dy in range(glow_height):
        y = apex_y - dy
        if y < 0:
            break
        falloff = 1.0 - (dy / glow_height)
        falloff = falloff ** 1.5  # Faster falloff
        glow[y, :] = falloff * 0.08
    
    top_col = palette[0] / 255.0
    final = final + glow[..., None] * top_col * 0.18
    
    # Final output
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.04)
    
    return img

# === BATCH RENDER ===
print(f"\nRendering {len(PALETTES) * len(TIMES)} variants...")

# Before
for res, (rw, rh) in RESOLUTIONS.items():
    before.resize((rw, rh), Image.Resampling.LANCZOS).save(
        os.path.join(out_dir, f"before-{res}.jpg"), quality=92)
print("  before images saved")

# Variants
for p in PALETTES:
    for t in TIMES:
        variant = f"{p}-{t}"
        print(f"  {variant}")
        img = render(p, t)
        for res, (rw, rh) in RESOLUTIONS.items():
            img.resize((rw, rh), Image.Resampling.LANCZOS).save(
                os.path.join(out_dir, f"{variant}-{res}.jpg"), quality=92)

# Summary
files = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
print(f"\nCOMPLETE: {len(files)} files, {size/1024/1024:.1f} MB")
print(f"Output: {out_dir}")
