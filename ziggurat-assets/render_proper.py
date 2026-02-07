"""
Ziggurat LED Vision — Proper Polygon-Constrained Renderer
Uses the actual building silhouette to contain LED bands
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import os

# === SETUP ===
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "proper")
os.makedirs(out_dir, exist_ok=True)

print("=" * 60)
print("ZIGGURAT — POLYGON-CONSTRAINED RENDERER")
print("=" * 60)

# Load source
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}×{h}")

# === SCALE FACTORS ===
# Reference coordinates are for 1920×1080
# Scale to actual image size
sx = w / 1920  # 4032/1920 = 2.1
sy = h / 1080  # 2217/1080 = 2.0528

print(f"Scale factors: X={sx:.3f}, Y={sy:.3f}")

# === BUILDING POLYGON (from spatial diagram, scaled) ===
# Original coordinates for 1920×1080, now scaled
BUILDING_POLYGON_1080 = [
    (960, 475),    # Apex
    (1114, 497),   # T7 right
    (1190, 529),   # T6 right
    (1248, 562),   # T5 right
    (1306, 605),   # T4 right
    (1363, 648),   # T3 right
    (1402, 702),   # T2 right
    (1440, 756),   # T1 right
    (1478, 810),   # Base right
    (442, 810),    # Base left
    (480, 756),    # T1 left
    (518, 702),    # T2 left
    (557, 648),    # T3 left
    (614, 605),    # T4 left
    (672, 562),    # T5 left
    (730, 529),    # T6 left
    (806, 497),    # T7 left
]

# Scale to actual image size
BUILDING_POLYGON = [(int(x * sx), int(y * sy)) for x, y in BUILDING_POLYGON_1080]
print(f"Building polygon scaled: apex=({BUILDING_POLYGON[0]})")

# === TIER DEFINITIONS (Y boundaries in 1080p, then scaled) ===
TIERS_1080 = [
    {"name": "T7", "top": 475, "bottom": 497, "left_top": 806, "right_top": 1114, "left_bot": 730, "right_bot": 1190},
    {"name": "T6", "top": 497, "bottom": 529, "left_top": 730, "right_top": 1190, "left_bot": 672, "right_bot": 1248},
    {"name": "T5", "top": 529, "bottom": 562, "left_top": 672, "right_top": 1248, "left_bot": 614, "right_bot": 1306},
    {"name": "T4", "top": 562, "bottom": 605, "left_top": 614, "right_top": 1306, "left_bot": 557, "right_bot": 1363},
    {"name": "T3", "top": 605, "bottom": 648, "left_top": 557, "right_top": 1363, "left_bot": 518, "right_bot": 1402},
    {"name": "T2", "top": 648, "bottom": 702, "left_top": 518, "right_top": 1402, "left_bot": 480, "right_bot": 1440},
    {"name": "T1", "top": 702, "bottom": 756, "left_top": 480, "right_top": 1440, "left_bot": 442, "right_bot": 1478},
]

# Scale tiers
TIERS = []
for t in TIERS_1080:
    TIERS.append({
        "name": t["name"],
        "top": int(t["top"] * sy),
        "bottom": int(t["bottom"] * sy),
        "left_top": int(t["left_top"] * sx),
        "right_top": int(t["right_top"] * sx),
        "left_bot": int(t["left_bot"] * sx),
        "right_bot": int(t["right_bot"] * sx),
    })

for t in TIERS:
    print(f"  {t['name']}: Y={t['top']}-{t['bottom']}, X={t['left_bot']}-{t['right_bot']}")

# === CREATE BUILDING MASK ===
print("\nCreating building polygon mask...")
building_mask = Image.new('L', (w, h), 0)
draw = ImageDraw.Draw(building_mask)
draw.polygon(BUILDING_POLYGON, fill=255)
building_mask = building_mask.filter(ImageFilter.GaussianBlur(3))
building_maskf = np.array(building_mask).astype(np.float32) / 255.0

# === CREATE TIER MASKS ===
print("Creating per-tier trapezoid masks...")
tier_masks = []
for t in TIERS:
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    # Trapezoid for this tier
    poly = [
        (t["left_top"], t["top"]),
        (t["right_top"], t["top"]),
        (t["right_bot"], t["bottom"]),
        (t["left_bot"], t["bottom"]),
    ]
    draw.polygon(poly, fill=255)
    tier_masks.append(np.array(mask).astype(np.float32) / 255.0)

# === COLOR PALETTES ===
PALETTES = {
    'rainbow': [
        np.array([139, 92, 246]),   # Purple - T7
        np.array([59, 130, 246]),   # Blue - T6
        np.array([6, 182, 212]),    # Cyan - T5
        np.array([34, 197, 94]),    # Green - T4
        np.array([234, 179, 8]),    # Yellow - T3
        np.array([249, 115, 22]),   # Orange - T2
        np.array([239, 68, 68]),    # Red - T1
    ],
    'warm-white': [
        np.array([255, 248, 240]),  # Warm white for all
        np.array([255, 248, 240]),
        np.array([255, 248, 240]),
        np.array([255, 248, 240]),
        np.array([255, 248, 240]),
        np.array([255, 248, 240]),
        np.array([255, 248, 240]),
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
    'cool': [
        np.array([139, 92, 246]),   # Purple
        np.array([99, 102, 241]),   # Indigo
        np.array([59, 130, 246]),   # Blue
        np.array([14, 165, 233]),   # Sky
        np.array([6, 182, 212]),    # Cyan
        np.array([20, 184, 166]),   # Teal
        np.array([16, 185, 129]),   # Emerald
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
}

# === TIME SETTINGS ===
TIME_SETTINGS = {
    'night': {
        'grade': (0.22, 0.26, 0.38),
        'brightness': 1.0,
        'sky_dark': 0.25,
    },
    'dusk': {
        'grade': (0.65, 0.55, 0.48),
        'brightness': 0.7,
        'sky_dark': 0.70,
    },
    'twilight': {
        'grade': (0.35, 0.35, 0.50),
        'brightness': 0.85,
        'sky_dark': 0.40,
    },
}

# === RENDER FUNCTION ===
def render_variant(palette_name, time_name):
    palette = PALETTES[palette_name]
    settings = TIME_SETTINGS[time_name]
    
    arr = np.array(before).astype(np.float32)
    
    # Apply time grading
    arr[:, :, 0] *= settings['grade'][0]
    arr[:, :, 1] *= settings['grade'][1]
    arr[:, :, 2] *= settings['grade'][2]
    
    # Darken sky more (above building)
    sky_y = int(475 * sy)  # Apex Y
    for y in range(sky_y):
        factor = settings['sky_dark'] + (1 - settings['sky_dark']) * (y / sky_y) * 0.5
        arr[y, :, :] *= factor
    
    # Create emission for each tier
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    brightness_per_tier = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]  # T7 brightest
    
    for i, (t, tmask) in enumerate(zip(TIERS, tier_masks)):
        col = palette[min(i, len(palette) - 1)].astype(np.float32)
        
        # Create LED band at bottom edge of tier (where overhang would be lit)
        band_y = t["bottom"]
        band_thickness = int(12 * sy)
        
        band_mask = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, band_y - band_thickness), min(h, band_y + 4)
        band_mask[y0:y1, :] = 1.0
        
        # Multiply by tier trapezoid to clip to building bounds
        band_mask *= tmask
        
        # Blur for glow
        band_img = Image.fromarray((band_mask * 255).astype(np.uint8))
        band_img = band_img.filter(ImageFilter.GaussianBlur(int(8 * sx)))
        band_mask = np.array(band_img).astype(np.float32) / 255.0
        
        # Re-clip to building bounds
        band_mask *= building_maskf
        
        # Add emission
        intensity = 200 * settings['brightness'] * brightness_per_tier[i]
        emit += band_mask[..., None] * (col / 255.0) * intensity
        
        # Add white-hot center line
        center_mask = np.zeros((h, w), dtype=np.float32)
        center_mask[band_y-2:band_y+2, :] = 1.0
        center_mask *= tmask
        center_mask *= building_maskf
        emit += center_mask[..., None] * intensity * 0.5
    
    # Screen blend emission onto base
    base = arr / 255.0
    light = np.clip(emit, 0, 255) / 255.0
    
    # Screen: 1 - (1-a)(1-b)
    blended = 1 - (1 - base) * (1 - light)
    
    # Constrain blend to building area + subtle spill
    spill_mask = Image.fromarray((building_maskf * 255).astype(np.uint8))
    spill_mask = spill_mask.filter(ImageFilter.GaussianBlur(int(30 * sx)))
    spill_maskf = np.array(spill_mask).astype(np.float32) / 255.0
    
    # Blend: full effect inside building, fading outside
    final = base * (1 - spill_maskf[..., None]) + blended * spill_maskf[..., None]
    
    # Add subtle glow above building (light bleeding into sky)
    glow = np.zeros((h, w), dtype=np.float32)
    apex_y = int(475 * sy)
    for dy in range(int(100 * sy)):
        y = apex_y - dy
        if y < 0:
            break
        falloff = 1.0 - (dy / (100 * sy))
        glow[y, :] = falloff * 0.15
    
    # Tint glow with palette's top color
    top_color = palette[0].astype(np.float32) / 255.0
    glow_rgb = glow[..., None] * top_color * 0.3
    final = final + glow_rgb
    
    out = np.clip(final * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    
    # Post-process
    img = ImageEnhance.Contrast(img).enhance(1.08)
    
    return img

# === RESOLUTIONS ===
RESOLUTIONS = {
    'full': (w, h),
    '4k': (3840, 2111),
    '1080p': (1920, 1055),
    'thumb': (640, 352),
}

# === RENDER ALL COMBINATIONS ===
combos = [
    ('rainbow', 'night'),
    ('rainbow', 'twilight'),
    ('rainbow', 'dusk'),
    ('warm-white', 'night'),
    ('warm-white', 'twilight'),
    ('mono-gold', 'night'),
    ('cool', 'night'),
    ('usa', 'night'),
]

print(f"\nRendering {len(combos)} variants...")

# Save before images
print("\n[BEFORE]")
for res_name, (rw, rh) in RESOLUTIONS.items():
    img = before.resize((rw, rh), Image.Resampling.LANCZOS)
    path = os.path.join(out_dir, f"before-{res_name}.jpg")
    img.save(path, quality=92)
    print(f"  before-{res_name}.jpg")

# Render variants
for palette, time in combos:
    variant = f"{palette}-{time}"
    print(f"\n[{variant}]")
    
    img = render_variant(palette, time)
    
    for res_name, (rw, rh) in RESOLUTIONS.items():
        img_r = img.resize((rw, rh), Image.Resampling.LANCZOS)
        path = os.path.join(out_dir, f"{variant}-{res_name}.jpg")
        img_r.save(path, quality=92)
        print(f"  {variant}-{res_name}.jpg")

# === CREATE COMPARISON ===
print("\n[COMPARISON]")
before_1080 = before.resize((1920, 1055), Image.Resampling.LANCZOS)
after_1080 = render_variant('rainbow', 'night').resize((1920, 1055), Image.Resampling.LANCZOS)

comparison = Image.new('RGB', (1920 * 2, 1055))
comparison.paste(before_1080, (0, 0))
comparison.paste(after_1080, (1920, 0))
comparison.save(os.path.join(out_dir, "comparison-rainbow-night.jpg"), quality=92)
print("  comparison-rainbow-night.jpg")

# Warm white comparison
after_ww = render_variant('warm-white', 'night').resize((1920, 1055), Image.Resampling.LANCZOS)
comparison_ww = Image.new('RGB', (1920 * 2, 1055))
comparison_ww.paste(before_1080, (0, 0))
comparison_ww.paste(after_ww, (1920, 0))
comparison_ww.save(os.path.join(out_dir, "comparison-warmwhite-night.jpg"), quality=92)
print("  comparison-warmwhite-night.jpg")

print("\n" + "=" * 60)
print("COMPLETE — Polygon-constrained renders saved to:", out_dir)
print("=" * 60)
