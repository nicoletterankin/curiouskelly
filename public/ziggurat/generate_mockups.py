from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import os
import math
import random

# Paths adjusted for Windows
script_dir = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(script_dir, "aerial.jpg")
out_dir = script_dir

img = Image.open(in_path).convert("RGB")
w, h = img.size

arr = np.array(img).astype(np.float32)

# Build a soft mask for the yellow/tan facade panels using HSV thresholds
def rgb_to_hsv(a):
    a = a / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx
    # Hue
    mask = diff != 0
    rm = (mx == r) & mask
    gm = (mx == g) & mask
    bm = (mx == b) & mask
    h[rm] = (60 * ((g[rm] - b[rm]) / diff[rm]) + 360) % 360
    h[gm] = (60 * ((b[gm] - r[gm]) / diff[gm]) + 120) % 360
    h[bm] = (60 * ((r[bm] - g[bm]) / diff[bm]) + 240) % 360
    # Saturation
    s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    return h, s, v

H, S, V = rgb_to_hsv(arr)

# Yellow/tan ranges (empirically) + value to avoid sky
mask = (H > 25) & (H < 75) & (S > 0.08) & (V > 0.25) & (V < 0.95)

# Remove vegetation greens and shadows by excluding greenish hues and very dark
mask = mask & ~((H > 80) & (H < 170) & (S > 0.15)) & (V > 0.32)

mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

# Clean mask: blur then threshold, then smooth edges
mask_blur = mask_img.filter(ImageFilter.GaussianBlur(radius=4))
mask_np = np.array(mask_blur)
mask_clean = (mask_np > 90).astype(np.uint8) * 255
mask_clean_img = Image.fromarray(mask_clean, mode="L").filter(ImageFilter.GaussianBlur(radius=3))

# USE MANUALLY TRACED COORDINATES for the ziggurat pyramid
# The pyramid is at x: 680-2820, y: 755-1100 (traced from image)
x0, y0, x1, y1 = 680, 755, 2820, 1100
bbox = (x0, y0, x1, y1)

# Restrict mask to only the pyramid area
pyramid_mask = np.zeros((h, w), dtype=np.uint8)
pyramid_mask[y0:y1, x0:x1] = 255
mask_clean_img = Image.fromarray(
    np.minimum(np.array(mask_clean_img), pyramid_mask), mode="L"
)

print(f"Using pyramid bbox: {bbox}")
print(f"Image size: {w}x{h}")

# Sample sky region for reflections (top 20% of image)
sky = img.crop((0, 0, w, int(h * 0.22))).filter(ImageFilter.GaussianBlur(radius=10))
sky_ref = sky.resize((w, h), resample=Image.Resampling.BICUBIC)

# Create "glass/metal" base: desaturate the facade area into cool neutral and overlay reflection
base = img.copy()
base_arr = np.array(base).astype(np.float32)
ref_arr = np.array(sky_ref).astype(np.float32)

# Neutralize tan -> aluminum/glass tone
cool = np.zeros_like(base_arr)
cool[..., 0] = 190
cool[..., 1] = 198
cool[..., 2] = 210  # cool gray
m = (np.array(mask_clean_img) / 255.0)[..., None]

# add gentle reflection gradient: stronger toward top of facade
yy = np.linspace(0, 1, h)[:, None]
facade_grad = np.clip((yy - (y0 / h)) * 3.0, 0, 1)
facade_grad = facade_grad[..., None]
ref_strength = 0.18 + 0.22 * facade_grad
ref_strength = np.clip(ref_strength, 0.18, 0.40)

metal_glass = base_arr * (1 - m) + ((base_arr * (1 - 0.55) + cool * 0.55)) * m
metal_glass = metal_glass * (1 - m * ref_strength) + ref_arr * (m * ref_strength)

# Add subtle panel seam texture: horizontal micro-lines within facade bbox
texture = np.zeros((h, w, 3), dtype=np.float32)
for y in range(y0, y1, 18):
    texture[y:y + 1, x0:x1, :] = 1.0
texture_img = Image.fromarray(np.clip(texture * 255, 0, 255).astype(np.uint8))
texture_img = texture_img.filter(ImageFilter.GaussianBlur(radius=0.6))
tex_arr = np.array(texture_img).astype(np.float32)

metal_glass = metal_glass + (tex_arr - 128) * 0.06 * m
metal_glass = np.clip(metal_glass, 0, 255)

base2 = Image.fromarray(metal_glass.astype(np.uint8))

# LED band zones: use PRECISE traced terrace coordinates
# Each terrace has specific left and right X positions
# Format: (y_position, left_x, right_x, thickness, color_rgb)
rainbow_colors = [
    (139, 92, 246),   # Purple - Level 7 (crown)
    (59, 130, 246),   # Blue - Level 6
    (6, 182, 212),    # Cyan - Level 5
    (34, 197, 94),    # Green - Level 4
    (234, 179, 8),    # Yellow - Level 3
    (249, 115, 22),   # Orange - Level 2
    (239, 68, 68),    # Red - Level 1 (base)
]

# Precisely traced terraces (y, left_x, right_x) - FINAL CORRECTED
# Building pyramid is at approximately Y = 730 (crown) to Y = 1080 (base)
terraces = [
    (730, 1480, 2010),   # Level 7 (crown) - very top
    (780, 1360, 2130),   # Level 6
    (835, 1230, 2260),   # Level 5
    (895, 1095, 2400),   # Level 4
    (960, 955, 2550),    # Level 3
    (1025, 810, 2700),   # Level 2
    (1095, 660, 2855),   # Level 1 (base)
]

band_specs = []
for i, (y, lx, rx) in enumerate(terraces):
    thickness = 14 + (i * 3)  # Thicker bands at lower levels
    color = rainbow_colors[i]
    band_specs.append((y, lx, rx, thickness, color))

print(f"LED band positions: {band_specs}")


def apply_led_variant(mode):
    """
    mode: 'day', 'dusk', 'night', 'final'
    """
    out = base2.copy()
    out_arr = np.array(out).astype(np.float32)

    # Create LED emission map
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    for idx, (y, lx, rx, th, color) in enumerate(band_specs):
        y1b = max(0, y - th // 2)
        y2b = min(h, y + th // 2)
        
        # Create band mask only within the terrace bounds
        band_mask = np.zeros((h, w), dtype=np.float32)
        band_mask[y1b:y2b, lx:rx] = 1.0
        
        # Blur for soft glow
        band_mask = np.array(Image.fromarray((band_mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=8))) / 255.0
        
        # Use the rainbow color
        col = np.array(color, dtype=np.float32)

        # brightness per mode
        if mode == 'day':
            b = 25
        elif mode == 'dusk':
            b = 70
        elif mode == 'night':
            b = 120
        else:  # final
            b = 95

        emit += band_mask[..., None] * (col / 255.0) * b

    # Apply emission with additive/screen blend
    emit_rgb = np.clip(emit, 0, 255)
    a = out_arr / 255.0
    b_emit = emit_rgb / 255.0
    
    # Additive blend for bright LEDs
    blended = np.clip(a + b_emit * 0.8, 0, 1)

    # Slight darkening for night mood
    if mode == 'night':
        blended = blended * 0.95
    if mode == 'final':
        blended = blended * 0.97

    return Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8))


print("Generating day variant...")
day = apply_led_variant('day')
print("Generating dusk variant...")
dusk = apply_led_variant('dusk')
print("Generating night variant...")
night = apply_led_variant('night')
print("Generating final variant...")
final = apply_led_variant('final')

# Create zoning diagram overlay
zoning = img.copy()
overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Facade outline bbox for context
draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 180), width=2)

# Define zones
zones = [
    ("LED reveal bands (night-only, low-nit)", (0, 180, 255, 120)),
    ("Embedded glass curtain bands", (140, 200, 255, 80)),
    ("Opaque architectural panels (ceramic/aluminum)", (255, 255, 255, 35)),
    ("Entry wayfinding pylons", (255, 200, 0, 120)),
]

# Mark LED bands with their actual colors
for (y, lx, rx, th, color) in band_specs:
    draw.rectangle([lx, y - th // 2, rx, y + th // 2], fill=(color[0], color[1], color[2], 150))

# Mark glass bands
glass_regions = [
    (x0, int(y0 + (y1 - y0) * 0.27), x1, int(y0 + (y1 - y0) * 0.33)),
    (x0, int(y0 + (y1 - y0) * 0.59), x1, int(y0 + (y1 - y0) * 0.64)),
]
for r in glass_regions:
    draw.rectangle(r, fill=(140, 200, 255, 70))

# Mark opaque panels
draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 28))

# Pylons
pylons = [(int(w * 0.19), int(h * 0.695)), (int(w * 0.76), int(h * 0.695))]
for (px, py) in pylons:
    draw.rectangle([px - 6, py - 32, px + 6, py + 32], fill=(255, 200, 0, 120))

zoning = Image.alpha_composite(zoning.convert("RGBA"), overlay).convert("RGB")

# Add legend box
legend = zoning.copy()
ld = ImageDraw.Draw(legend)
box_w, box_h = int(w * 0.34), int(h * 0.19)
bx0, by0 = int(w * 0.02), int(h * 0.03)
ld.rectangle([bx0, by0, bx0 + box_w, by0 + box_h], fill=(10, 10, 14, 160), outline=(255, 255, 255, 140), width=2)

# Font - try to load, fallback to default
try:
    font = ImageFont.truetype("arial.ttf", 22)
    font_s = ImageFont.truetype("arial.ttf", 18)
except:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_s = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_s = font

ld.text((bx0 + 14, by0 + 10), "LED Integration Zoning", fill=(255, 255, 255, 230), font=font)
ycur = by0 + 46
for name, color in zones:
    sw = 20
    ld.rectangle([bx0 + 14, ycur + 5, bx0 + 14 + sw, ycur + 5 + sw], fill=color)
    ld.text((bx0 + 14 + sw + 10, ycur + 3), name, fill=(235, 235, 235, 230), font=font_s)
    ycur += 28

# Save outputs
paths = {
    "zoning": os.path.join(out_dir, "ziggurat-led-zoning.png"),
    "day": os.path.join(out_dir, "ziggurat-led-day.png"),
    "dusk": os.path.join(out_dir, "ziggurat-led-dusk.png"),
    "night": os.path.join(out_dir, "ziggurat-led-night.png"),
    "final": os.path.join(out_dir, "ziggurat-led-final.png"),
    "before": os.path.join(out_dir, "ziggurat-before.png"),
}

print("Saving outputs...")
legend.save(paths["zoning"])
day.save(paths["day"])
dusk.save(paths["dusk"])
night.save(paths["night"])
final.save(paths["final"])
img.save(paths["before"])

print(f"\nGenerated files:")
for name, path in paths.items():
    print(f"  {name}: {path}")

print("\nDone!")
