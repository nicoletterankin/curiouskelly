# Kelly Asset Pack Generator - Demo

This document demonstrates the toolkit capabilities with example outputs.

## Existing Assets

The `synthetic_tts/` directory already contains:
- `kelly_directors_chair_8k_light.png` — Existing Kelly image (light background)
- `kelly_front_square_8k_transparent.png` — Existing square sprite

## Demo Scenarios

### Scenario 1: Generate All Assets from Existing Image

If you have the director's chair image:

```bash
# Option 1: Auto-detect
python -m kelly_pack.cli build --outdir ./demo_output

# Option 2: Explicit path
python -m kelly_pack.cli build \
  --chair "synthetic_tts/kelly_directors_chair_8k_light.png" \
  --outdir ./demo_output
```

**Expected outputs in `./demo_output/`:**
```
kelly_directors_chair_8k_transparent.png  (7680×4320 RGBA)
kelly_directors_chair_8k_dark.png         (7680×4320 RGB, dark gradient)
kelly_front_square_8k_transparent.png     (8192×8192 RGBA)
kelly_diffuse_neutral_8k.png              (8192×8192 RGB)
kelly_chair_diffuse_neutral_8k.png        (7680×4320 RGB)
kelly_alpha_soft_8k.png                   (7680×4320 L)
kelly_alpha_tight_8k.png                  (7680×4320 L)
kelly_hair_edge_matte_8k.png              (7680×4320 L)
kelly_physics_reference_sheet.pdf         (PDF document)
```

---

### Scenario 2: Regenerate Dark Hero with Custom Colors

```bash
python -m kelly_pack.cli dark-hero \
  --outdir ./demo_output \
  --grad-top "#1A1E22" \
  --grad-bottom "#000000"
```

**Output:** Updated `kelly_directors_chair_8k_dark.png` with new gradient.

---

### Scenario 3: Fine-Tune Hair for Light UI

```bash
python -m kelly_pack.cli hair \
  --chair "synthetic_tts/kelly_directors_chair_8k_light.png" \
  --outdir ./demo_output \
  --soft-blur 3.0 \
  --soft-bias 0.08
```

**Output:** Updated alpha mattes with more generous halo for light backgrounds.

---

### Scenario 4: Tight Hair for Dark UI

```bash
python -m kelly_pack.cli hair \
  --chair "synthetic_tts/kelly_directors_chair_8k_light.png" \
  --outdir ./demo_output \
  --tight-blur 0.5 \
  --tight-bias -0.05 \
  --tight-erode 2
```

**Output:** Updated alpha mattes with tighter edges for dark backgrounds.

---

### Scenario 5: Use GPU Acceleration

```bash
python -m kelly_pack.cli build \
  --chair "synthetic_tts/kelly_directors_chair_8k_light.png" \
  --device cuda \
  --outdir ./demo_output
```

**Note:** Requires PyTorch with CUDA support.

---

### Scenario 6: Heuristic-Only (No PyTorch)

```bash
python -m kelly_pack.cli build \
  --chair "synthetic_tts/kelly_directors_chair_8k_light.png" \
  --no-torch \
  --outdir ./demo_output
```

**Use case:** Environments without PyTorch, or for white-background images where heuristic works well.

---

## Visual Comparison

### Hair Quality Comparison

| Mode | Alpha Variant | Use Case | Hair Edge Quality |
|------|---------------|----------|-------------------|
| Light UI | `alpha_soft` | Light backgrounds | Graceful halo with wispy strands |
| Dark UI | `alpha_tight` | Dark backgrounds | No visible white halo |
| Edge Only | `hair_edge_matte` | Compositing control | Halo-only channel |

### Dark Hero Gradient

The dark hero uses a vertical gradient:
- **Top (#22262A):** Subtle blue-gray for depth
- **Bottom (#080808):** Nearly black for cinematic feel

You can customize:
```bash
--grad-top "#2A2E32" --grad-bottom "#0A0A0A"
```

---

## Performance Notes

### Processing Times (Approximate)

| Task | Resolution | CPU (i7) | GPU (RTX 3080) |
|------|-----------|----------|----------------|
| Matting (heuristic) | 2K | ~2s | N/A |
| Matting (U²-Net) | 2K | ~15s | ~3s |
| Upsample to 8K | 8K | ~5s | ~5s |
| Total pipeline | 8K | ~30s | ~15s |

### Memory Usage

- **Peak RAM:** ~4 GB (for 8K processing)
- **VRAM (GPU):** ~2 GB (for model inference)

---

## Validation Checklist

After running the toolkit, validate these aspects:

### ✅ Hair on Dark Mode
Open `kelly_directors_chair_8k_dark.png` in image viewer.
- **Check:** No white halo around hair edges
- **Expected:** Clean dark gradient background

### ✅ Hair on Light Mode
Open `kelly_alpha_soft_8k.png` as overlay on white background.
- **Check:** Graceful wispy halo
- **Expected:** Natural hair falloff

### ✅ Transparent Hero Alignment
Overlay `kelly_directors_chair_8k_transparent.png` on dark hero.
- **Check:** Pixel-perfect alignment
- **Expected:** Same crop/scale

### ✅ Square Sprite Padding
Open `kelly_front_square_8k_transparent.png`.
- **Check:** Subject centered, ~10% padding each side
- **Expected:** Subject occupies ~80% of canvas

### ✅ Diffuse Neutral Quality
Open `kelly_diffuse_neutral_8k.png`.
- **Check:** Evenly balanced channels, reduced mid-tone contrast
- **Expected:** "Flat" diffuse texture suitable for relighting

### ✅ Physics Reference
Open `kelly_physics_reference_sheet.pdf`.
- **Check:** All iClone physics specs present
- **Expected:** Chair, fabric, hair, camera specs

---

## Troubleshooting

### Issue: Hair edges too harsh on light UI
**Solution:**
```bash
python -m kelly_pack.cli hair --soft-blur 3.0 --soft-bias 0.10
```

### Issue: Hair halo visible on dark UI
**Solution:**
```bash
python -m kelly_pack.cli hair --tight-bias -0.05 --tight-erode 2
```

### Issue: Subject too small in square sprite
**Solution:**
```bash
python -m kelly_pack.cli sprite --padding-frac 0.05
```

### Issue: Diffuse too flat
**Solution:**
```bash
python -m kelly_pack.cli build --contrast-flatten 0.08
```

---

## Next Steps

1. **Integrate into pipeline:** Use generated assets in your digital human workflow
2. **Customize parameters:** Adjust alpha tuning for your specific UI design
3. **Automate:** Add to CI/CD or batch processing scripts
4. **Extend:** Fork and add custom matting models or output formats

---

**Ready to generate production-quality 8K assets!** 🎬


