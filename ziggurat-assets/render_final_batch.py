"""
Ziggurat LED Vision — Final Batch Renderer
Polygon-constrained, all palettes × times × resolutions
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os
import json

# === SETUP ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "final")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT LED VISION — FINAL BATCH RENDER")
print("=" * 60)

# Load source
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

# === SCALE FACTORS ===
sx = w / 1920
sy = h / 1080
print(f"Scale: X={sx:.3f}, Y={sy:.3f}")

# === BUILDING POLYGON ===
BUILDING_POLYGON_1080 = [
    (960, 475), (1114, 497), (1190, 529), (1248, 562), (1306, 605),
    (1363, 648), (1402, 702), (1440, 756), (1478, 810), (442, 810),
    (480, 756), (518, 702), (557, 648), (614, 605), (672, 562),
    (730, 529), (806, 497),
]
BUILDING_POLYGON = [(int(x * sx), int(y * sy)) for x, y in BUILDING_POLYGON_1080]

# === TIER DEFINITIONS ===
TIERS_1080 = [
    {"name": "T7", "top": 475, "bottom": 497, "left_top": 806, "right_top": 1114, "left_bot": 730, "right_bot": 1190},
    {"name": "T6", "top": 497, "bottom": 529, "left_top": 730, "right_top": 1190, "left_bot": 672, "right_bot": 1248},
    {"name": "T5", "top": 529, "bottom": 562, "left_top": 672, "right_top": 1248, "left_bot": 614, "right_bot": 1306},
    {"name": "T4", "top": 562, "bottom": 605, "left_top": 614, "right_top": 1306, "left_bot": 557, "right_bot": 1363},
    {"name": "T3", "top": 605, "bottom": 648, "left_top": 557, "right_top": 1363, "left_bot": 518, "right_bot": 1402},
    {"name": "T2", "top": 648, "bottom": 702, "left_top": 518, "right_top": 1402, "left_bot": 480, "right_bot": 1440},
    {"name": "T1", "top": 702, "bottom": 756, "left_top": 480, "right_top": 1440, "left_bot": 442, "right_bot": 1478},
]

TIERS = [{
    "name": t["name"],
    "top": int(t["top"] * sy), "bottom": int(t["bottom"] * sy),
    "left_top": int(t["left_top"] * sx), "right_top": int(t["right_top"] * sx),
    "left_bot": int(t["left_bot"] * sx), "right_bot": int(t["right_bot"] * sx),
} for t in TIERS_1080]

# === CREATE MASKS ===
print("Creating masks...")
building_mask = Image.new('L', (w, h), 0)
ImageDraw.Draw(building_mask).polygon(BUILDING_POLYGON, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(3))
building_maskf = np.array(building_mask).astype(np.float32) / 255.0

tier_masks = []
for t in TIERS:
    mask = Image.new('L', (w, h), 0)
    poly = [(t["left_top"], t["top"]), (t["right_top"], t["top"]),
            (t["right_bot"], t["bottom"]), (t["left_bot"], t["bottom"])]
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    tier_masks.append(np.array(mask).astype(np.float32) / 255.0)

# === PALETTES ===
PALETTES = {
    'rainbow': {
        'label': 'Rainbow',
        'colors': [[139,92,246], [59,130,246], [6,182,212], [34,197,94], [234,179,8], [249,115,22], [239,68,68]],
    },
    'cool': {
        'label': 'Cool Tones',
        'colors': [[139,92,246], [99,102,241], [59,130,246], [14,165,233], [6,182,212], [20,184,166], [16,185,129]],
    },
    'warm': {
        'label': 'Warm Tones',
        'colors': [[251,191,36], [245,158,11], [249,115,22], [239,68,68], [236,72,153], [217,70,239], [168,85,247]],
    },
    'white': {
        'label': 'Warm White',
        'colors': [[255,248,240]] * 7,
    },
    'gold': {
        'label': 'Gold',
        'colors': [[255,215,0], [255,205,0], [255,195,0], [255,185,0], [255,175,0], [255,165,0], [255,155,0]],
    },
    'cyan': {
        'label': 'Cyan',
        'colors': [[0,255,255], [0,235,235], [0,215,215], [0,195,195], [0,175,175], [0,155,155], [0,135,135]],
    },
    'purple': {
        'label': 'Purple',
        'colors': [[168,85,247], [158,80,237], [148,75,227], [138,70,217], [128,65,207], [118,60,197], [108,55,187]],
    },
    'usa': {
        'label': 'USA',
        'colors': [[239,68,68], [255,255,255], [59,130,246], [255,255,255], [239,68,68], [255,255,255], [59,130,246]],
    },
    'sunset': {
        'label': 'Sunset',
        'colors': [[239,68,68], [249,115,22], [251,146,60], [253,186,116], [254,215,170], [255,237,213], [255,247,237]],
    },
    'ocean': {
        'label': 'Ocean',
        'colors': [[30,58,138], [29,78,216], [37,99,235], [59,130,246], [96,165,250], [147,197,253], [191,219,254]],
    },
}

# === TIME SETTINGS ===
TIMES = {
    'night': {'label': 'Night', 'grade': (0.22, 0.26, 0.38), 'brightness': 1.0, 'sky_dark': 0.25},
    'late-night': {'label': 'Late Night', 'grade': (0.15, 0.18, 0.28), 'brightness': 1.1, 'sky_dark': 0.18},
    'twilight': {'label': 'Twilight', 'grade': (0.35, 0.35, 0.50), 'brightness': 0.85, 'sky_dark': 0.40},
    'dusk': {'label': 'Dusk', 'grade': (0.65, 0.55, 0.48), 'brightness': 0.65, 'sky_dark': 0.70},
    'golden': {'label': 'Golden Hour', 'grade': (0.80, 0.70, 0.55), 'brightness': 0.50, 'sky_dark': 0.85},
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
    
    # Darken sky
    apex_y = int(475 * sy)
    for y in range(apex_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / apex_y) * 0.5
        arr[y, :, :] *= factor
    
    # Emission per tier
    emit = np.zeros((h, w, 3), dtype=np.float32)
    brightness_scale = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)].astype(np.float32)
        band_y = t["bottom"]
        thickness = int(12 * sy)
        
        band = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, band_y - thickness), min(h, band_y + 4)
        band[y0:y1, :] = 1.0
        band *= tmask
        
        band_img = Image.fromarray((band * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(8 * sx)))
        band = np.array(band_img).astype(np.float32) / 255.0
        band *= building_maskf
        
        intensity = 200 * settings['brightness'] * brightness_scale[i]
        emit += band[..., None] * (col / 255.0) * intensity
        
        # White-hot center
        center = np.zeros((h, w), dtype=np.float32)
        center[band_y-2:band_y+2, :] = 1.0
        center *= tmask * building_maskf
        emit += center[..., None] * intensity * 0.5
    
    # Screen blend
    base = arr / 255.0
    light = np.clip(emit, 0, 255) / 255.0
    blended = 1 - (1 - base) * (1 - light)
    
    # Spill control
    spill = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill = spill.filter(ImageFilter.GaussianBlur(int(30 * sx)))
    spillf = np.array(spill).astype(np.float32) / 255.0
    
    final = base * (1 - spillf[..., None]) + blended * spillf[..., None]
    
    # Sky glow
    glow = np.zeros((h, w), dtype=np.float32)
    for dy in range(int(100 * sy)):
        y = apex_y - dy
        if y < 0: break
        glow[y, :] = (1.0 - dy / (100 * sy)) * 0.15
    
    top_col = palette[0].astype(np.float32) / 255.0
    final = final + glow[..., None] * top_col * 0.3
    
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    img = ImageEnhance.Contrast(img).enhance(1.08)
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
print("\n[MANIFEST]")
manifest = {
    "palettes": {k: v['label'] for k, v in PALETTES.items()},
    "times": {k: v['label'] for k, v in TIMES.items()},
    "resolutions": list(RESOLUTIONS.keys()),
    "variants": [f"{p}-{t}" for p in PALETTES for t in TIMES],
    "total": total,
}
with open(os.path.join(out_dir, "manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2)
print("  manifest.json")

# === SUMMARY ===
files = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in files)
print(f"\n{'='*60}")
print(f"COMPLETE: {len(files)} files, {size/1024/1024:.1f} MB")
print(f"Output: {out_dir}")
print("=" * 60)
