"""
Final Pitch-Ready Ziggurat Renderer
Produces stakeholder-ready assets at multiple resolutions
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import os
from scipy.signal import convolve2d

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "pitch-assets")
os.makedirs(out_dir, exist_ok=True)

before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Source: {w}x{h}")

arr = np.array(before).astype(np.float32)

# === HSV MASK FOR FACADE ===
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
print("Finding terrace edges...")
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

# === RENDER FUNCTION ===
def render_leds(base_arr, mode='night'):
    result = base_arr.copy()
    
    # Color grading
    if mode == 'night':
        result[:, :, 0] *= 0.28
        result[:, :, 1] *= 0.32
        result[:, :, 2] *= 0.42
    elif mode == 'dusk':
        result[:, :, 0] *= 0.75
        result[:, :, 1] *= 0.68
        result[:, :, 2] *= 0.58
    else:  # day
        result[:, :, 0] *= 0.95
        result[:, :, 1] *= 0.96
        result[:, :, 2] *= 0.97
    
    emit = np.zeros((h, w, 3), dtype=np.float32)
    
    # Rainbow palette
    rainbow = [
        np.array([139, 92, 246]),   # Purple
        np.array([59, 130, 246]),   # Blue
        np.array([6, 182, 212]),    # Cyan
        np.array([34, 197, 94]),    # Green
        np.array([234, 179, 8]),    # Yellow
        np.array([249, 115, 22]),   # Orange
        np.array([239, 68, 68]),    # Red
    ]
    
    brightness = {'night': 160, 'dusk': 100, 'day': 50}[mode]
    
    for i, y in enumerate(peaks):
        thickness = 14 + i * 3
        band = np.zeros((h, w), dtype=np.float32)
        y0, y1 = max(0, y - thickness // 2), min(h, y + thickness // 2)
        band[y0:y1, :] = 1.0
        band *= maskf
        band = np.array(Image.fromarray((band * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(4))) / 255.0
        
        col = rainbow[min(i, len(rainbow) - 1)].astype(np.float32)
        emit += band[..., None] * (col / 255.0) * brightness
        
        # White-hot center
        if mode == 'night':
            center = np.zeros((h, w), dtype=np.float32)
            center[y-1:y+1, :] = 1.0
            center *= maskf
            emit += center[..., None] * 0.5 * brightness
    
    # Screen blend
    a = result / 255.0
    b = np.clip(emit, 0, 255) / 255.0
    blended = 1 - (1 - a) * (1 - b)
    
    # Spill guard
    spill = maskf[..., None] * 0.92 + 0.08
    blended = a * (1 - maskf[..., None]) + blended * spill
    
    return np.clip(blended * 255, 0, 255).astype(np.uint8)

# === GENERATE VERSIONS ===
print("Rendering night...")
night = Image.fromarray(render_leds(arr, 'night'))
night = ImageEnhance.Contrast(night).enhance(1.06)
night = ImageEnhance.Sharpness(night).enhance(1.1)

print("Rendering dusk...")
dusk = Image.fromarray(render_leds(arr, 'dusk'))
dusk = ImageEnhance.Contrast(dusk).enhance(1.04)

print("Rendering day...")
day = Image.fromarray(render_leds(arr, 'day'))

# === SAVE AT MULTIPLE RESOLUTIONS ===
resolutions = {
    'full': (w, h),
    '4k': (3840, int(3840 * h / w)),
    '1080p': (1920, int(1920 * h / w)),
}

for name, (rw, rh) in resolutions.items():
    print(f"Exporting {name} ({rw}x{rh})...")
    
    before_r = before.resize((rw, rh), Image.Resampling.LANCZOS)
    night_r = night.resize((rw, rh), Image.Resampling.LANCZOS)
    dusk_r = dusk.resize((rw, rh), Image.Resampling.LANCZOS)
    day_r = day.resize((rw, rh), Image.Resampling.LANCZOS)
    
    before_r.save(os.path.join(out_dir, f"before-{name}.jpg"), quality=92)
    night_r.save(os.path.join(out_dir, f"after-night-{name}.jpg"), quality=92)
    dusk_r.save(os.path.join(out_dir, f"after-dusk-{name}.jpg"), quality=92)
    day_r.save(os.path.join(out_dir, f"after-day-{name}.jpg"), quality=92)

# === SIDE-BY-SIDE COMPARISONS ===
print("Creating comparisons...")

def make_comparison(before_img, after_img, label_after):
    gap = 20
    cw = before_img.width * 2 + gap
    ch = before_img.height
    canvas = Image.new("RGB", (cw, ch), (10, 10, 12))
    canvas.paste(before_img, (0, 0))
    canvas.paste(after_img, (before_img.width + gap, 0))
    
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(24, before_img.height // 40))
    except:
        font = ImageFont.load_default()
    
    pad = 16
    for x, text in [(pad, "BEFORE"), (before_img.width + gap + pad, label_after)]:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2], bbox[3]
        draw.rounded_rectangle([x, pad, x + tw + 20, pad + th + 12], 
                               radius=8, fill=(0, 0, 0, 200), outline=(255, 255, 255), width=1)
        draw.text((x + 10, pad + 6), text, fill=(255, 255, 255), font=font)
    
    return canvas

# 1080p comparisons
before_1080 = before.resize((1920, int(1920 * h / w)), Image.Resampling.LANCZOS)
night_1080 = night.resize((1920, int(1920 * h / w)), Image.Resampling.LANCZOS)
dusk_1080 = dusk.resize((1920, int(1920 * h / w)), Image.Resampling.LANCZOS)

comp_night = make_comparison(before_1080, night_1080, "AFTER (NIGHT)")
comp_dusk = make_comparison(before_1080, dusk_1080, "AFTER (DUSK)")

comp_night.save(os.path.join(out_dir, "comparison-night.jpg"), quality=92)
comp_dusk.save(os.path.join(out_dir, "comparison-dusk.jpg"), quality=92)

# === HERO IMAGE (NIGHT FULL RES) ===
print("Creating hero image...")
night.save(os.path.join(out_dir, "HERO-night-full.png"), quality=95)
before.save(os.path.join(out_dir, "HERO-before-full.png"), quality=95)

print(f"\n=== COMPLETE ===")
print(f"Assets saved to: {out_dir}")
print(f"Files generated:")
for f in sorted(os.listdir(out_dir)):
    size = os.path.getsize(os.path.join(out_dir, f)) / 1024 / 1024
    print(f"  {f} ({size:.1f} MB)")
