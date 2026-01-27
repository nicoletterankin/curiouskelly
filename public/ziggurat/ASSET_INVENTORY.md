# Ziggurat Asset Inventory

**Location:** `public/ziggurat/`  
**Total Files:** 72  
**Total Size:** ~175 MB  
**Generated:** January 24, 2026

---

## Source Photography

| File | Size | Description |
|------|------|-------------|
| `aerial.jpg` | 1.3 MB | **PRIMARY SOURCE** - Hero aerial photo (4032×2217) |
| `aerial.png` | 127 KB | PNG version (smaller) |
| `before/hero-01-quarter-view.jpg` | 401 KB | Alternate angle 1 |
| `before/hero-02-frontal.jpg` | 419 KB | Alternate angle 2 |
| `before/hero-03-corner.jpg` | 512 KB | Alternate angle 3 |
| `before/hero-04-detail.jpg` | 492 KB | Alternate angle 4 |

---

## Rendered Mockups (Full Resolution)

### Primary Outputs
| File | Size | Description |
|------|------|-------------|
| `BEFORE.png` | 6.4 MB | Original photo (4032×2217) |
| `AFTER-night.png` | 4.8 MB | Night rainbow LEDs |
| `AFTER-dusk.png` | 6.2 MB | Dusk variant |
| `BEFORE-AFTER.png` | 11.6 MB | Side-by-side composite |

### Enhanced Edge-Detection Version
| File | Size | Description |
|------|------|-------------|
| `ziggurat-after-enhanced.png` | 8.2 MB | Edge-aligned LED bands |
| `ziggurat-before-after-enhanced.png` | 14.6 MB | Enhanced comparison |

### Color Variants (from generate_mockups.py)
| File | Size | Description |
|------|------|-------------|
| `ziggurat-before.png` | 6.4 MB | Before baseline |
| `ziggurat-led-day.png` | 6.4 MB | Day LEDs (subtle) |
| `ziggurat-led-dusk.png` | 6.5 MB | Dusk LEDs |
| `ziggurat-led-night.png` | 7.0 MB | Night LEDs |
| `ziggurat-led-final.png` | 6.9 MB | Final composite |
| `ziggurat-led-zoning.png` | 6.3 MB | Zoning diagram overlay |

### Simple Render Tests
| File | Size | Description |
|------|------|-------------|
| `ziggurat-simple-before.png` | 6.4 MB | Simple before |
| `ziggurat-simple-night.png` | 4.7 MB | Simple night render |

---

## Pitch-Ready Assets (`pitch-assets/`)

### Hero Images (Full 4032×2217)
| File | Size | Format |
|------|------|--------|
| `HERO-before-full.png` | 6.4 MB | PNG |
| `HERO-night-full.png` | 6.4 MB | PNG |

### Multi-Resolution Exports
| Variant | Full | 4K | 1080p |
|---------|------|-------|-------|
| Before | `before-full.jpg` (1.3 MB) | `before-4k.jpg` (1.2 MB) | `before-1080p.jpg` (460 KB) |
| Night | `after-night-full.jpg` (932 KB) | `after-night-4k.jpg` (843 KB) | `after-night-1080p.jpg` (293 KB) |
| Dusk | `after-dusk-full.jpg` (1.2 MB) | `after-dusk-4k.jpg` (1.1 MB) | `after-dusk-1080p.jpg` (410 KB) |
| Day | `after-day-full.jpg` (1.3 MB) | `after-day-4k.jpg` (1.3 MB) | `after-day-1080p.jpg` (464 KB) |

### Comparison Images
| File | Size | Description |
|------|------|-------------|
| `comparison-night.jpg` | 763 KB | Before + After (Night) side-by-side |
| `comparison-dusk.jpg` | 880 KB | Before + After (Dusk) side-by-side |

---

## Debug/Development Images

| File | Size | Purpose |
|------|------|---------|
| `debug-grid.png` | 6.4 MB | Y-coordinate calibration |
| `debug-night-grid.png` | 7.0 MB | Night render with grid |
| `simple-with-grid.png` | 4.8 MB | Grid overlay test |
| `test-band-730.png` | 6.4 MB | Band position test |

---

## HTML Pages

### Production Ready
| File | Purpose |
|------|---------|
| `pitch-assets/index.html` | **Stakeholder pitch presentation** |
| `view.html` | Interactive before/after slider |
| `final.html` | Single-page presentation |

### Tools & Editors
| File | Purpose |
|------|---------|
| `generate.html` | One-click asset generator |
| `mockup-v2.html` | LED mockup editor |
| `trace-walls.html` | Manual coordinate tracer |
| `7-layers.html` | 7-layer visualization |
| `compositor.html` | Image compositor |

### Presentations & Demos
| File | Purpose |
|------|---------|
| `index.html` | Landing page |
| `pitch.html` | Pitch deck |
| `presentation.html` | Full presentation |
| `experience.html` | Interactive experience |

### Utilities
| File | Purpose |
|------|---------|
| `results.html` | Results viewer |
| `test.html` | Test page |
| `check.html` | Check page |
| `tools.html` | Tool index |
| `floors.html` | Floor visualization |
| `photo-overlay.html` | Photo overlay tool |
| `panel-planner.html` | Panel planning tool |
| `display-simulator.html` | Display simulator |
| `content-simulator.html` | Content simulator |
| `export-mockups.html` | Export tool |
| `mockup.html` | Original mockup |

---

## Python Renderers

| File | Size | Purpose |
|------|------|---------|
| `render_pitch.py` | 7.8 KB | **PRIMARY** - Multi-resolution pitch assets |
| `render_enhanced.py` | 6.4 KB | Edge-detection LED placement |
| `render_final.py` | 4.7 KB | Night/dusk render |
| `generate_mockups.py` | 10 KB | HSV-masked LED generator |
| `simple_render.py` | 2.4 KB | Simple direct render |

---

## Data Files

| File | Size | Purpose |
|------|------|---------|
| `TRACED_COORDINATES.json` | 3.5 KB | Terrace pixel coordinates |
| `manifest.json` | 7.4 KB | Asset manifest with metadata |

---

## Documentation

| File | Size | Purpose |
|------|------|---------|
| `ZIGGURAT_PLATFORM_SPEC.md` | 13 KB | Platform specification |
| `ZIGGURAT_SPECIFICATION.md` | 22 KB | LED technical specification |
| `LAYER_ARCHITECTURE.md` | 7.8 KB | 7-layer architecture |
| `README.md` | 1.8 KB | Project readme |
| `ASSET_INVENTORY.md` | — | This file |

---

## Key Coordinates

**Detected Terrace Edges (Y pixels, image height 2217):**
```
Y = [1222, 1264, 1336, 1393, 1435, 1492, 1540]
```

**Normalized (0-1):**
```
Y = [0.551, 0.570, 0.602, 0.628, 0.647, 0.673, 0.694]
```

**Building Bounds:**
- Crown: Y ≈ 1180 (53%)
- Base: Y ≈ 1550 (70%)
- Left X: ~660 (16%)
- Right X: ~2850 (71%)

---

## Recommended for Pitch

**Minimum set for stakeholder presentation:**

1. `pitch-assets/index.html` — Interactive web page
2. `pitch-assets/HERO-night-full.png` — Hero after (6.4 MB)
3. `pitch-assets/HERO-before-full.png` — Hero before (6.4 MB)
4. `pitch-assets/comparison-night.jpg` — Side-by-side (763 KB)
5. `pitch-assets/after-night-4k.jpg` — 4K version (843 KB)

**Total pitch package: ~15 MB**

---

## Quick Access URLs

When served locally (`npx serve public`):

```
http://localhost:3000/ziggurat/pitch-assets/          # Pitch presentation
http://localhost:3000/ziggurat/view.html              # Slider viewer
http://localhost:3000/ziggurat/HERO-night-full.png    # Hero image
http://localhost:3000/ziggurat/comparison-night.jpg   # Comparison
```
