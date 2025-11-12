# Quick Start Guide

Get up and running with Kelly Asset Pack Generator in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration and video support:
```bash
pip install torch torchvision imageio imageio-ffmpeg
```

## 2. Place Input Images

Put one or more of these in your project directory:
- `kelly2-directors-chair.jpeg` (preferred)
- `Kelly Source.jpeg` (fallback)
- `reference_kelly_image.png` (for square sprite)

## 3. Generate Assets

### Option A: Simple (auto-detect inputs)
```bash
python -m kelly_pack.cli build
```

### Option B: Explicit inputs
```bash
python -m kelly_pack.cli build \
  --chair "kelly2-directors-chair.jpeg" \
  --portrait "reference_kelly_image.png" \
  --outdir "./output"
```

### Option C: Use Makefile
```bash
make build
```

## 4. Check Outputs

Your assets are in `./output/`:
- `kelly_directors_chair_8k_transparent.png` — RGBA hero
- `kelly_directors_chair_8k_dark.png` — Dark mode version
- `kelly_front_square_8k_transparent.png` — Square sprite
- And 7 more files...

## Common Adjustments

### Hair too harsh on light backgrounds?
```bash
python -m kelly_pack.cli build --soft-blur 3.0 --soft-bias 0.10
```

### Hair halo on dark backgrounds?
```bash
python -m kelly_pack.cli build --tight-bias -0.05 --tight-erode 2
```

### Use GPU acceleration?
```bash
python -m kelly_pack.cli build --device cuda
```

### Force heuristic matting (no PyTorch)?
```bash
python -m kelly_pack.cli build --no-torch
```

## Testing

```bash
make test
# or
pytest tests/ -v
```

## Cleanup

```bash
make clean
```

---

**Need help?** See [README.md](README.md) for full documentation.


