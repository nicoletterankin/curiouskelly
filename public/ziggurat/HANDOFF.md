# Ziggurat LED Vision — v0.app Handoff

## Quick Start

```bash
# Assets are in precision/ folder
open public/ziggurat/index.html  # Master landing page
open public/ziggurat/precision/  # Full gallery with all variants
```

## Asset Inventory

### Source Image
- `aerial.jpg` — 4032×2217 hero photo of Chet Holifield Federal Building

### Rendered Variants (precision/)
- **28 variants** = 7 palettes × 4 times
- **4 resolutions each** = full, 4k, 1080p, thumb
- **Total: 116 images, ~52 MB**

### Palettes
| Key | Label | Description |
|-----|-------|-------------|
| rainbow | Rainbow | Purple→Blue→Cyan→Green→Yellow→Orange→Red |
| cool | Cool | Purple→Blue→Cyan→Teal gradient |
| warm | Warm | Yellow→Orange→Red→Pink→Purple |
| white | White | Warm white (Apple Park aesthetic) |
| gold | Gold | Golden yellow gradient |
| cyan | Cyan | Aqua/teal gradient |
| usa | USA | Red/White/Blue alternating |

### Times
| Key | Label | Description |
|-----|-------|-------------|
| night | Night | Deep blue night grading |
| late-night | Late Night | Darker, higher LED brightness |
| twilight | Twilight | Purple dusk sky |
| dusk | Dusk | Golden hour, warmer tones |

### Resolutions
| Key | Dimensions | Use Case |
|-----|------------|----------|
| full | 4032×2217 | Print, large displays |
| 4k | 3840×2111 | 4K monitors |
| 1080p | 1920×1055 | Web, presentations |
| thumb | 480×264 | Thumbnails, previews |

## Building Geometry (Precision-Traced)

### Tier Measurements (1920×1055 reference)
| Tier | Top Y | Bottom Y | Left X | Right X | Width % | Center % |
|------|-------|----------|--------|---------|---------|----------|
| T7 | 474 | 498 | 855→805 | 928→980 | 9.1% | 46.5% |
| T6 | 498 | 528 | 805→752 | 980→1038 | 14.9% | 46.6% |
| T5 | 528 | 562 | 752→692 | 1038→1100 | 21.3% | 46.7% |
| T4 | 562 | 600 | 692→625 | 1100→1168 | 28.3% | 46.7% |
| T3 | 600 | 645 | 625→550 | 1168→1245 | 36.2% | 46.7% |
| T2 | 645 | 695 | 550→468 | 1245→1332 | 45.0% | 46.9% |
| T1 | 695 | 750 | 468→378 | 1332→1425 | 54.5% | 46.9% |

### Key Insights
- Building center is at **46.9%** from left (not 50% — camera angle)
- 7 visible stepped tiers (8th "base" is separate structure)
- Tier width progression: 9.1% → 54.5% of frame width

## File Naming Convention

```
{palette}-{time}-{resolution}.jpg

Examples:
  gold-night-1080p.jpg
  rainbow-twilight-4k.jpg
  usa-dusk-full.jpg
  before-1080p.jpg
```

## Rendering Pipeline

The Python renderer (`render_precision.py`) uses:

1. **Polygon masks** — Per-tier trapezoid constraints
2. **Time grading** — RGB channel multipliers for sky/building
3. **Emission mapping** — LED bands at tier bottoms
4. **Screen blend** — `1 - (1 - base) * (1 - light)`
5. **Spill control** — Gaussian-blurred building mask for soft edge
6. **Sky glow** — Gradient above apex

## Integration Points

### For Next.js/React
```tsx
// Image paths
const getVariantUrl = (palette: string, time: string, res: string) =>
  `/ziggurat/precision/${palette}-${time}-${res}.jpg`;

// Palette/time configs importable from render_precision.py structure
```

### For Supabase (if storing metadata)
```sql
CREATE TABLE ziggurat_variants (
  id SERIAL PRIMARY KEY,
  palette VARCHAR(20) NOT NULL,
  time_of_day VARCHAR(20) NOT NULL,
  resolution VARCHAR(10) NOT NULL,
  url TEXT NOT NULL,
  width INT,
  height INT
);
```

## What v0.app Could Build

1. **Interactive Configurator** — Live palette/time picker with before/after slider
2. **Animation Mode** — Cycle through palettes in sequence
3. **Admin Panel** — Upload new source images, regenerate variants
4. **Stakeholder Portal** — Password-protected gallery with download tracking

## Files to Deploy

```
public/ziggurat/
├── index.html              # Master landing page
├── aerial.jpg              # Source image
├── HANDOFF.md              # This file
├── render_precision.py     # Regeneration script
└── precision/
    ├── index.html          # Full gallery
    ├── before-*.jpg        # 4 resolutions
    └── {palette}-{time}-{resolution}.jpg  # 112 variants
```

## Credits

- Building: Chet Holifield Federal Building, Laguna Niguel, CA
- Architect: William Pereira (1971)
- LED Vision: Precision-traced rendering pipeline
