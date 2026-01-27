"""
Ziggurat LED Vision — Calibrated to Actual Building
Based on visual analysis of before image
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os
import json

# === SETUP ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "calibrated")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT — CALIBRATED RENDERER")
print("=" * 60)

# Load source
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

sx = w / 1920
sy = h / 1080
print(f"Scale: X={sx:.3f}, Y={sy:.3f}")

# === CALIBRATED BUILDING MEASUREMENTS ===
# From visual analysis of before image at 1920×1080:
# - Building base left edge: ~320px (17%)
# - Building base right edge: ~1520px (79%)  
# - Building width at base: ~1200px (62% of frame)
# - Apex Y: ~510px (47%)
# - Base Y: ~820px (76%)

# The ziggurat has a shallower slope than 45° - more like 20-25°
# Each tier is ~8% narrower than the one below

CENTER_X = 960  # Center of frame

# Tier widths (half-width from center, in pixels for 1920px)
# Base is widest, apex is narrowest
TIER_HALF_WIDTHS = [
    85,    # T7 (apex) - very narrow
    140,   # T6
    210,   # T5
    290,   # T4
    380,   # T3
    470,   # T2
    550,   # T1
    600,   # BASE - 600px half-width = 1200px total = 62.5% of 1920
]

# Tier Y positions (for 1920×1080)
TIER_Y_BOTTOMS = [
    548,   # T7 bottom
    582,   # T6 bottom  
    622,   # T5 bottom
    668,   # T4 bottom
    718,   # T3 bottom
    768,   # T2 bottom
    818,   # T1 bottom
    855,   # BASE bottom
]

TIER_Y_TOPS = [
    510,   # T7 top (apex)
    548,   # T6 top
    582,   # T5 top
    622,   # T4 top
    668,   # T3 top
    718,   # T2 top
    768,   # T1 top
    818,   # BASE top
]

# Build tier definitions
TIERS_1080 = []
for i in range(8):
    name = f"T{7-i}" if i < 7 else "BASE"
    hw_top = TIER_HALF_WIDTHS[max(0, i-1)] if i > 0 else 50
    hw_bot = TIER_HALF_WIDTHS[i]
    TIERS_1080.append({
        "name": name,
        "top": TIER_Y_TOPS[i],
        "bottom": TIER_Y_BOTTOMS[i],
        "left_top": CENTER_X - hw_top,
        "right_top": CENTER_X + hw_top,
        "left_bot": CENTER_X - hw_bot,
        "right_bot": CENTER_X + hw_bot,
    })

# Scale to actual image
TIERS = [{
    "name": t["name"],
    "top": int(t["top"] * sy), "bottom": int(t["bottom"] * sy),
    "left_top": int(t["left_top"] * sx), "right_top": int(t["right_top"] * sx),
    "left_bot": int(t["left_bot"] * sx), "right_bot": int(t["right_bot"] * sx),
} for t in TIERS_1080]

print("\nTier dimensions:")
for t in TIERS:
    width = t["right_bot"] - t["left_bot"]
    pct = width / w * 100
    print(f"  {t['name']}: Y={t['top']}-{t['bottom']}, width={width}px ({pct:.0f}%)")

# Build building polygon
BUILDING_POLYGON = []
# Right side (top to bottom)
for t in TIERS:
    BUILDING_POLYGON.append((t["right_bot"], t["bottom"]))
# Bottom edge
BUILDING_POLYGON.append((TIERS[-1]["left_bot"], TIERS[-1]["bottom"]))
# Left side (bottom to top)
for t in reversed(TIERS):
    BUILDING_POLYGON.append((t["left_bot"], t["bottom"]))
# Close at apex
BUILDING_POLYGON.append((TIERS[0]["left_top"], TIERS[0]["top"]))

print(f"\nBuilding polygon: {len(BUILDING_POLYGON)} vertices")
print(f"Base width: {TIERS[-1]['right_bot'] - TIERS[-1]['left_bot']}px ({(TIERS[-1]['right_bot'] - TIERS[-1]['left_bot'])/w*100:.0f}%)")

# === CREATE MASKS ===
print("\nCreating masks...")

# Building mask
building_mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(building_mask).polygon(BUILDING_POLYGON, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(4))
building_maskf = np.array(building_mask).astype(np.float32) / 255.0

# Tier masks
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
    'night': {'grade': (0.18, 0.22, 0.34), 'brightness': 1.0, 'sky_dark': 0.20},
    'late-night': {'grade': (0.12, 0.14, 0.24), 'brightness': 1.15, 'sky_dark': 0.12},
    'twilight': {'grade': (0.30, 0.30, 0.46), 'brightness': 0.80, 'sky_dark': 0.35},
    'dusk': {'grade': (0.58, 0.48, 0.42), 'brightness': 0.60, 'sky_dark': 0.65},
}

RESOLUTIONS = {
    'full': (w, h),
    '4k': (3840, 2111),
    '1080p': (1920, 1055),
    'thumb': (480, 264),
}

# === RENDER FUNCTION ===
def render(palette_key, time_key):
    palette = [np.array(c) for c in PALETTES[palette_key]]
    settings = TIMES[time_key]
    
    arr = np.array(before).astype(np.float32)
    arr[:, :, 0] *= settings['grade'][0]
    arr[:, :, 1] *= settings['grade'][1]
    arr[:, :, 2] *= settings['grade'][2]
    
    # Darken sky
    apex_y = int(510 * sy)
    for y in range(apex_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / apex_y) * 0.5
        arr[y, :, :] *= factor
    
    # Emission
    emit = np.zeros((h, w, 3), dtype=np.float32)
    brightness_scale = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)].astype(np.float32)
        
        band_y = t["bottom"]
        thickness = int(14 * sy)
        
        band = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, band_y - thickness), min(h, band_y + 5)
        band[y0:y1, :] = 1.0
        band *= tmask
        
        band_img = Image.fromarray((band * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(10 * sx)))
        band = np.array(band_img).astype(np.float32) / 255.0
        band *= building_maskf
        
        intensity = 240 * settings['brightness'] * brightness_scale[i]
        emit += band[..., None] * (col / 255.0) * intensity
        
        # White-hot center
        center = np.zeros((h, w), dtype=np.float32)
        center[band_y-3:band_y+3, :] = 1.0
        center *= tmask * building_maskf
        emit += center[..., None] * intensity * 0.4
    
    # Screen blend
    base = arr / 255.0
    light = np.clip(emit, 0, 255) / 255.0
    blended = 1 - (1 - base) * (1 - light)
    
    # Spill control
    spill = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill = spill.filter(ImageFilter.GaussianBlur(int(40 * sx)))
    spillf = np.array(spill).astype(np.float32) / 255.0
    
    final = base * (1 - spillf[..., None]) + blended * spillf[..., None]
    
    # Sky glow
    glow = np.zeros((h, w), dtype=np.float32)
    for dy in range(int(100 * sy)):
        y = apex_y - dy
        if y < 0: break
        glow[y, :] = (1.0 - dy / (100 * sy)) * 0.10
    
    top_col = palette[0].astype(np.float32) / 255.0
    final = final + glow[..., None] * top_col * 0.2
    
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.05)
    return img

# === BATCH RENDER ===
print(f"\nRendering {len(PALETTES) * len(TIMES)} variants...")

# Before
for res, (rw, rh) in RESOLUTIONS.items():
    before.resize((rw, rh), Image.Resampling.LANCZOS).save(
        os.path.join(out_dir, f"before-{res}.jpg"), quality=92)

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
