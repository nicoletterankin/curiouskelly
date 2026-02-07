"""
Enhanced Ziggurat LED Renderer
Uses edge detection to find actual terrace seams
"""
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import os
from scipy.signal import convolve2d

script_dir = os.path.dirname(os.path.abspath(__file__))
before = Image.open(os.path.join(script_dir, "aerial.jpg")).convert("RGB")
w, h = before.size
print(f"Image size: {w}x{h}")

arr = np.array(before).astype(np.float32)

# HSV conversion
def rgb_to_hsv(a):
    a = a / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx
    mask = diff != 0
    rm = (mx == r) & mask
    gm = (mx == g) & mask
    bm = (mx == b) & mask
    h[rm] = (60 * ((g[rm] - b[rm]) / diff[rm]) + 360) % 360
    h[gm] = (60 * ((b[gm] - r[gm]) / diff[gm]) + 120) % 360
    h[bm] = (60 * ((r[bm] - g[bm]) / diff[bm]) + 240) % 360
    s[mx != 0] = diff[mx != 0] / mx[mx != 0]
    return h, s, v

H, S, V = rgb_to_hsv(arr)

# Facade mask (yellow/tan building panels)
mask = (H > 20) & (H < 80) & (S > 0.06) & (V > 0.28) & (V < 0.96)
mask &= ~((H > 80) & (H < 170) & (S > 0.14))  # remove vegetation
mask &= (V > 0.32)

mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
mask_np = np.array(mask_img)
mask_clean = (mask_np > 95).astype(np.uint8) * 255
mask_clean_img = Image.fromarray(mask_clean).filter(ImageFilter.GaussianBlur(2))
maskf = (np.array(mask_clean_img) / 255.0).astype(np.float32)

# Find horizontal edges using Sobel filter
print("Detecting horizontal edges...")
gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
gy = convolve2d(gray, ky, mode='same', boundary='symm')
edge_h = np.abs(gy) * maskf

# Summarize edge strength by row
row_strength = edge_h.mean(axis=1)

# Restrict to pyramid terraces only (not the whole building)
# Pyramid is at Y ~1180 to ~1550
ymin, ymax = 1180, 1550
print(f"Searching pyramid Y range: {ymin} to {ymax}")

# Zero out non-pyramid rows
roi = row_strength.copy()
roi[:ymin] = 0
roi[ymax + 1:] = 0

# Pick top N peaks with minimum separation
N = 7  # 7 terrace levels
min_sep = 35
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
    # suppress neighborhood
    y0 = max(0, y - min_sep)
    y1 = min(h, y + min_sep)
    roi_copy[y0:y1] = 0

peaks = sorted(peaks)
print(f"Detected {len(peaks)} terrace edges at Y: {peaks}")

# Build enhanced after from before
base = arr.copy()

# Slight night/dusk grading
base[:, :, 0] *= 0.85
base[:, :, 1] *= 0.88
base[:, :, 2] *= 0.92

emit = np.zeros((h, w, 3), dtype=np.float32)

# Rainbow palette (purple top to red bottom)
rainbow = [
    np.array([139, 92, 246], dtype=np.float32),   # Purple
    np.array([59, 130, 246], dtype=np.float32),   # Blue
    np.array([6, 182, 212], dtype=np.float32),    # Cyan
    np.array([34, 197, 94], dtype=np.float32),    # Green
    np.array([234, 179, 8], dtype=np.float32),    # Yellow
    np.array([249, 115, 22], dtype=np.float32),   # Orange
    np.array([239, 68, 68], dtype=np.float32),    # Red
]

for i, y in enumerate(peaks):
    # LED band - thicker for visibility
    thickness = 16 + i * 4
    band = np.zeros((h, w), dtype=np.float32)
    y0 = max(0, y - thickness // 2)
    y1 = min(h, y + thickness // 2)
    band[y0:y1, :] = 1.0
    band *= maskf
    
    # Blur for glow
    band = np.array(Image.fromarray((band * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(5))) / 255.0
    
    # Pick color from rainbow
    col_idx = min(i, len(rainbow) - 1)
    col = rainbow[col_idx]
    
    # Brightness - strong enough to see clearly
    b = 140
    emit += band[..., None] * (col / 255.0) * b

# Add wayfinding pylons
pylons = [(int(w * 0.20), int(h * 0.70)), (int(w * 0.77), int(h * 0.70))]
for (px, py) in pylons:
    rect = np.zeros((h, w), dtype=np.float32)
    rect[max(0, py - 22):min(h, py + 22), max(0, px - 4):min(w, px + 4)] = 1.0
    rect *= maskf
    rect = np.array(Image.fromarray((rect * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2.0))) / 255.0
    emit += rect[..., None] * (np.array([235, 240, 245], dtype=np.float32) / 255.0) * 28

# Screen blend
a = base / 255.0
b = np.clip(emit, 0, 255) / 255.0
blended = 1 - (1 - a) * (1 - b)

# Spill guard
spill_guard = (maskf[..., None] * 0.95 + 0.05)
blended = a * (1 - maskf[..., None]) + blended * spill_guard

out = np.clip(blended * 255, 0, 255).astype(np.uint8)
out_img = Image.fromarray(out)

# Post-processing
out_img = ImageEnhance.Color(out_img).enhance(0.94)
out_img = ImageEnhance.Contrast(out_img).enhance(1.04)
out_img = ImageEnhance.Sharpness(out_img).enhance(1.12)

# Save enhanced after
enh_path = os.path.join(script_dir, "ziggurat-after-enhanced.png")
out_img.save(enh_path, quality=95)
print(f"Saved: {enh_path}")

# Create before-after composite
gap = 24
canvas = Image.new("RGB", (w * 2 + gap, h), (15, 15, 18))
canvas.paste(before, (0, 0))
canvas.paste(out_img, (w + gap, 0))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font_s = ImageFont.truetype("arial.ttf", 30)
except:
    try:
        font_s = ImageFont.truetype("DejaVuSans.ttf", 30)
    except:
        font_s = ImageFont.load_default()

pad = 18

def tag(x, y, text):
    bbox = draw.textbbox((0, 0), text, font=font_s)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx0, by0 = x, y
    bx1, by1 = x + tw + 26, y + th + 18
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.text((x + 13, y + 8), text, fill=(255, 255, 255), font=font_s)

tag(pad, pad, "BEFORE")
tag(w + gap + pad, pad, "AFTER (ENHANCED)")

ba_path = os.path.join(script_dir, "ziggurat-before-after-enhanced.png")
canvas.save(ba_path, quality=95)
print(f"Saved: {ba_path}")

print(f"\nDetected terrace seams at Y positions: {peaks}")
print("Done!")
