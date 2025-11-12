# Kelly Asset Pack Generator - Project Structure

Complete directory layout and file descriptions.

```
kelly_pack/                           [Main package]
├── __init__.py                       Package metadata and version
├── __main__.py                       Make package directly executable
├── cli.py                            CLI interface with subcommands
│                                     - build: Generate all assets
│                                     - hair: Regenerate hair alphas
│                                     - dark-hero: Regenerate dark hero
│                                     - sprite: Regenerate square sprite
├── io_utils.py                       Image I/O utilities
│                                     - load_image(): Load with PIL
│                                     - save_image(): Save with mode handling
│                                     - find_first_existing(): Auto-detect inputs
├── crop_scale.py                     Crop and scale operations
│                                     - crop_to_aspect(): Center-crop to aspect ratio
│                                     - resize_lanczos(): High-quality resize
│                                     - prepare_16_9_hero(): Full 16:9 pipeline
│                                     - prepare_square_sprite(): Square canvas + padding
├── matting.py                        Hair/portrait matting
│                                     - model_based_matting(): U²-Net inference
│                                     - heuristic_matting(): White-bg estimator
│                                     - generate_alpha(): Auto-fallback wrapper
│                                     - guided_upsample_alpha(): Edge-aware upsample
├── alpha_tools.py                    Alpha channel utilities
│                                     - generate_soft_alpha(): Light UI variant
│                                     - generate_tight_alpha(): Dark UI variant
│                                     - generate_edge_matte(): Soft minus tight
│                                     - apply_gaussian_blur(): Gaussian blur
│                                     - apply_morphology(): Erode/dilate
├── composite.py                      Compositing and gradients
│                                     - create_vertical_gradient(): Gradient bg
│                                     - composite_over_background(): Alpha blend
│                                     - create_dark_hero(): Full dark-mode pipeline
│                                     - hex_to_rgb(): Color conversion
├── diffuse.py                        Diffuse neutralization
│                                     - gray_world_balance(): Channel balance
│                                     - flatten_contrast(): Reduce contrast
│                                     - neutralize_diffuse(): Full pipeline
├── sprite.py                         Square sprite (re-exports crop_scale)
├── physics_sheet.py                  Physics reference generation
│                                     - generate_physics_pdf(): PDF with reportlab
│                                     - generate_physics_png(): PNG fallback
└── video_frame.py                    Video utilities
                                      - extract_midframe(): Frame extraction

scripts/                              [Orchestration scripts]
└── build_all.py                      Wrapper script for CLI

tests/                                [Test suite]
├── __init__.py                       Test package
└── test_shapes_and_files.py          PyTest validation
                                      - Test output dimensions
                                      - Test alpha variants
                                      - Test file generation
                                      - Test image modes

Documentation:
├── README.md                         Complete documentation
├── QUICKSTART.md                     5-minute getting started guide
├── DEMO.md                           Example scenarios and validation
├── CHANGELOG.md                      Version history
├── PROJECT_SUMMARY.md                Technical overview and architecture
├── STRUCTURE.md                      This file (project layout)
└── LICENSE                           Apache 2.0 license

Configuration:
├── requirements.txt                  Python dependencies
├── setup.py                          Package installation script
├── Makefile                          Convenience targets (build, test, clean)
├── .gitignore                        Git ignore patterns
└── verify_installation.py            Installation verification script

Examples:
└── example_usage.py                  Programmatic usage examples

Output Directories (generated):
├── output/                           Default output directory
├── weights/                          Downloaded model weights
└── example_output/                   Example script outputs
```

---

## File Sizes (Approximate)

| File/Directory | Lines | Size | Description |
|----------------|-------|------|-------------|
| `cli.py` | ~500 | 20 KB | CLI interface |
| `matting.py` | ~200 | 10 KB | Matting algorithms |
| `crop_scale.py` | ~150 | 8 KB | Crop/scale utilities |
| `alpha_tools.py` | ~150 | 7 KB | Alpha operations |
| `composite.py` | ~120 | 6 KB | Compositing |
| `diffuse.py` | ~80 | 4 KB | Diffuse neutralization |
| `physics_sheet.py` | ~150 | 8 KB | PDF/PNG generation |
| `test_shapes_and_files.py` | ~250 | 12 KB | Tests |
| `README.md` | ~450 | 25 KB | Documentation |
| **Total (core)** | ~2,500 | 150 KB | Entire project |

---

## Generated Assets

### Typical Output Directory

```
output/
├── kelly_directors_chair_8k_transparent.png    (7680×4320, RGBA, ~50 MB)
├── kelly_directors_chair_8k_dark.png           (7680×4320, RGB, ~40 MB)
├── kelly_front_square_8k_transparent.png       (8192×8192, RGBA, ~80 MB)
├── kelly_diffuse_neutral_8k.png                (8192×8192, RGB, ~60 MB)
├── kelly_chair_diffuse_neutral_8k.png          (7680×4320, RGB, ~40 MB)
├── kelly_alpha_soft_8k.png                     (7680×4320, L, ~20 MB)
├── kelly_alpha_tight_8k.png                    (7680×4320, L, ~20 MB)
├── kelly_hair_edge_matte_8k.png                (7680×4320, L, ~15 MB)
├── kelly_physics_reference_sheet.pdf           (1 page, ~50 KB)
└── kelly_video_midframe_8k.png                 (7680×4320, RGB, ~40 MB, optional)
```

**Total output size:** ~365 MB per character

---

## Model Weights

### Downloaded Automatically (if using model-based matting)

```
weights/
└── u2net_portrait.pth                          (~4.7 MB)
```

**Note:** Weights are cached locally. Download happens once on first run.

---

## Data Flow

```
Input Image(s)
    │
    ├──> [crop_scale] ──> 16:9 Hero RGB (7680×4320)
    │         │
    │         └──> [matting] ──> Base Alpha
    │                   │
    │                   ├──> Model-based (U²-Net, 2K → 8K guided upsample)
    │                   └──> Heuristic (luminance-chroma, smoothstep)
    │                         │
    │                         └──> [alpha_tools] ──> Soft/Tight/Edge variants
    │                                   │
    │                                   ├──> Soft (blur + bias)
    │                                   ├──> Tight (blur + bias + erode)
    │                                   └──> Edge (soft - tight)
    │
    ├──> [composite] ──> Transparent Hero (RGBA)
    │                 └──> Dark Hero (RGB over gradient)
    │
    ├──> [diffuse] ──> Chair Diffuse Neutral (gray-world + flatten)
    │
    ├──> [sprite] ──> Square Sprite (8192², center + padding)
    │                 └──> Diffuse Neutral (square)
    │
    └──> [physics_sheet] ──> PDF Reference Sheet
```

---

## Dependency Graph

```
cli.py
 ├─ io_utils (load/save)
 ├─ crop_scale (16:9, square)
 ├─ matting (alpha generation)
 ├─ alpha_tools (soft/tight/edge)
 ├─ composite (gradients, blend)
 ├─ diffuse (neutralization)
 ├─ physics_sheet (PDF/PNG)
 └─ video_frame (optional)

matting.py
 ├─ cv2 (guided filter)
 ├─ torch (optional, U²-Net)
 └─ numpy

alpha_tools.py
 ├─ cv2 (blur, morphology)
 └─ numpy

composite.py
 └─ numpy

diffuse.py
 └─ numpy

physics_sheet.py
 ├─ reportlab (PDF)
 └─ matplotlib (PNG fallback)

video_frame.py
 └─ imageio (frame extraction)
```

---

## Extension Points

### Adding New Matting Models

1. Implement in `matting.py`:
   ```python
   def modnet_matting(img: np.ndarray) -> np.ndarray:
       # Load MODNet model
       # Run inference
       # Return alpha
   ```

2. Add to `generate_alpha()`:
   ```python
   if use_modnet:
       alpha = modnet_matting(img)
   elif use_u2net:
       alpha = model_based_matting(img)
   else:
       alpha = heuristic_matting(img)
   ```

3. Add CLI flag:
   ```python
   parser.add_argument("--use-modnet", action="store_true")
   ```

### Adding New Output Formats

1. Create new function in appropriate module:
   ```python
   def generate_webp_hero(rgb, alpha, output_path):
       # Convert to WebP with alpha
   ```

2. Call from `cli.py` in `build_all()`:
   ```python
   generate_webp_hero(hero_rgb, alpha_tight, f"{args.outdir}/kelly_hero.webp")
   ```

### Adding New Subcommands

1. Define command function in `cli.py`:
   ```python
   def preview_mode(args):
       # Generate 2K previews for quick iteration
   ```

2. Add subparser:
   ```python
   preview_parser = subparsers.add_parser("preview", help="Generate 2K previews")
   # Add args
   ```

3. Route in `main()`:
   ```python
   elif args.command == "preview":
       return preview_mode(args)
   ```

---

## Testing Strategy

### Unit Tests
- Individual functions (crop, blur, gradient, etc.)
- Use small test images (100×100)
- Fast execution (<1s per test)

### Integration Tests
- Full pipeline with small images
- Verify file generation
- Check output dimensions

### Validation Tests
- Manual QA on real 8K outputs
- Hair quality inspection (light/dark UI)
- Pixel alignment checks

---

## Continuous Integration (CI)

### Recommended GitHub Actions Workflow

```yaml
name: Test Kelly Pack

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

---

## Deployment

### Local Installation

```bash
pip install -e .                 # Editable mode
python -m kelly_pack.cli build  # Run CLI
```

### PyPI Distribution

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

### Docker (Future)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e .
ENTRYPOINT ["python", "-m", "kelly_pack.cli"]
```

---

**Complete project structure documented!** 📁


