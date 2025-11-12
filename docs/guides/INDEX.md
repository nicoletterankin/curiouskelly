# Kelly Asset Pack Generator - Documentation Index

Quick navigation to all project documentation.

## 🚀 Getting Started

1. **[QUICKSTART.md](QUICKSTART.md)** — Get up and running in 5 minutes
2. **[verify_installation.py](verify_installation.py)** — Verify your installation
3. **[example_usage.py](example_usage.py)** — Programmatic usage examples

## 📖 Core Documentation

- **[README.md](README.md)** — Complete documentation (CLI, features, troubleshooting)
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — Technical overview and architecture
- **[STRUCTURE.md](STRUCTURE.md)** — Project layout and file descriptions
- **[DEMO.md](DEMO.md)** — Example scenarios and validation checklist
- **[Kelly_CC5_Headshot2_Operator_Manual.md](docs/Kelly_CC5_Headshot2_Operator_Manual.md)** — CC5 + Headshot 2 + iClone 8 operator guide

## 📦 Project Files

### Configuration
- **[requirements.txt](requirements.txt)** — Python dependencies
- **[setup.py](setup.py)** — Package installation script
- **[Makefile](Makefile)** — Convenience targets (build, test, clean)
- **[.gitignore](.gitignore)** — Git ignore patterns

### Legal
- **[LICENSE](LICENSE)** — Apache 2.0 license
- **[CHANGELOG.md](CHANGELOG.md)** — Version history

## 🔧 Source Code

### Main Package (`kelly_pack/`)
- **[__init__.py](kelly_pack/__init__.py)** — Package metadata
- **[__main__.py](kelly_pack/__main__.py)** — Direct execution entry point
- **[cli.py](kelly_pack/cli.py)** — CLI interface with subcommands
- **[io_utils.py](kelly_pack/io_utils.py)** — Image I/O utilities
- **[crop_scale.py](kelly_pack/crop_scale.py)** — Crop and resize operations
- **[matting.py](kelly_pack/matting.py)** — Hair/portrait matting (model + heuristic)
- **[alpha_tools.py](kelly_pack/alpha_tools.py)** — Alpha channel utilities
- **[composite.py](kelly_pack/composite.py)** — Compositing and gradients
- **[diffuse.py](kelly_pack/diffuse.py)** — Diffuse neutralization
- **[sprite.py](kelly_pack/sprite.py)** — Square sprite utilities
- **[physics_sheet.py](kelly_pack/physics_sheet.py)** — Physics reference PDF/PNG
- **[video_frame.py](kelly_pack/video_frame.py)** — Video frame extraction

### Scripts (`scripts/`)
- **[build_all.py](scripts/build_all.py)** — Orchestration wrapper

### Tests (`tests/`)
- **[test_shapes_and_files.py](tests/test_shapes_and_files.py)** — PyTest validation suite

## 📚 Quick Reference

### CLI Commands

```bash
# Build all assets
python -m kelly_pack.cli build --outdir ./output

# Regenerate hair alphas only
python -m kelly_pack.cli hair --chair "image.jpg"

# Regenerate dark hero only
python -m kelly_pack.cli dark-hero

# Regenerate square sprite only
python -m kelly_pack.cli sprite --portrait "image.png"

# Get help
python -m kelly_pack.cli --help
python -m kelly_pack.cli build --help
```

### Make Targets

```bash
make install    # Install dependencies
make build      # Build all assets
make test       # Run tests
make clean      # Remove generated files
make help       # Show available targets
```

### Installation

```bash
# Basic installation
pip install -r requirements.txt

# With GPU support
pip install -r requirements.txt torch torchvision

# With video support
pip install -r requirements.txt imageio imageio-ffmpeg

# Development installation
pip install -e .
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run verification
python verify_installation.py

# Run examples
python example_usage.py
```

## 🎯 Common Use Cases

### Generate Assets from Chair Image

See: [QUICKSTART.md](QUICKSTART.md)

```bash
python -m kelly_pack.cli build \
  --chair "kelly2-directors-chair.jpeg" \
  --outdir ./output
```

### Fine-Tune Hair for Light UI

See: [DEMO.md](DEMO.md#scenario-3-fine-tune-hair-for-light-ui)

```bash
python -m kelly_pack.cli hair \
  --soft-blur 3.0 \
  --soft-bias 0.10
```

### Customize Dark Gradient

See: [DEMO.md](DEMO.md#scenario-2-regenerate-dark-hero-with-custom-colors)

```bash
python -m kelly_pack.cli dark-hero \
  --grad-top "#1A1E22" \
  --grad-bottom "#000000"
```

## 🔍 Troubleshooting

See: [README.md - Troubleshooting](README.md#troubleshooting)

Common issues:
- **"No chair image found"** → Place input image or use `--chair`
- **"PyTorch not available"** → Use `--no-torch` or install PyTorch
- **Hair edges harsh** → Adjust `--soft-blur` and `--soft-bias`
- **Hair halo on dark** → Adjust `--tight-bias` and `--tight-erode`

## 📊 Outputs

### Generated Files (10 total)

1. `kelly_directors_chair_8k_transparent.png` (7680×4320, RGBA)
2. `kelly_directors_chair_8k_dark.png` (7680×4320, RGB)
3. `kelly_front_square_8k_transparent.png` (8192×8192, RGBA)
4. `kelly_diffuse_neutral_8k.png` (8192×8192, RGB)
5. `kelly_chair_diffuse_neutral_8k.png` (7680×4320, RGB)
6. `kelly_alpha_soft_8k.png` (7680×4320, L)
7. `kelly_alpha_tight_8k.png` (7680×4320, L)
8. `kelly_hair_edge_matte_8k.png` (7680×4320, L)
9. `kelly_physics_reference_sheet.pdf` (PDF)
10. `kelly_video_midframe_8k.png` (7680×4320, RGB, optional)

See: [README.md - Output Files](README.md#output-files)

## 🏗️ Architecture

### Data Flow

```
Input → Crop/Scale → Matting → Alpha Variants → Outputs
                                    ├─ Transparent Hero
                                    ├─ Dark Hero
                                    ├─ Square Sprite
                                    ├─ Diffuse Neutrals
                                    └─ Alpha Utilities
```

See: [STRUCTURE.md - Data Flow](STRUCTURE.md#data-flow)

### Module Dependencies

```
cli.py
 ├─ io_utils
 ├─ crop_scale
 ├─ matting
 ├─ alpha_tools
 ├─ composite
 ├─ diffuse
 ├─ physics_sheet
 └─ video_frame
```

See: [STRUCTURE.md - Dependency Graph](STRUCTURE.md#dependency-graph)

## 🎓 Learning Resources

### Understanding Hair Matting
- Model-based: Uses deep learning (U²-Net) for portrait segmentation
- Heuristic: Uses luminance-chroma analysis for white backgrounds
- Both produce soft/tight variants optimized for different UIs

### Understanding Alpha Variants
- **Soft Alpha**: Positive bias + blur → graceful halo for light UIs
- **Tight Alpha**: Negative bias + erosion → no halo for dark UIs
- **Edge Matte**: Soft − tight → isolated halo for compositing control

### Understanding Diffuse Neutralization
- **Gray-world**: Balances RGB channels to neutral gray average
- **Contrast flatten**: Reduces mid-tone contrast for relighting

## 🔗 External Resources

- **U²-Net Paper**: https://arxiv.org/abs/2005.09007
- **OpenCV Guided Filter**: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- **Pillow Documentation**: https://pillow.readthedocs.io/
- **NumPy Documentation**: https://numpy.org/doc/

## 📝 Version History

See: [CHANGELOG.md](CHANGELOG.md)

**Current Version:** 1.0.0 (2025-10-12)

## 🤝 Contributing

### Reporting Issues
1. Check existing issues on GitHub
2. Provide input files and error messages
3. Include system info (OS, Python version, GPU)

### Submitting Pull Requests
1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Update documentation
6. Submit PR with clear description

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions focused

## 📧 Support

For questions, issues, or contributions:
- GitHub Issues: (link to repo)
- Email: support@ui-tars.com
- Documentation: This index!

---

## 🎬 Quick Start (30 seconds)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place image
# Put kelly2-directors-chair.jpeg in current directory

# 3. Generate
python -m kelly_pack.cli build

# 4. Check outputs
ls -lh *.png *.pdf
```

**Done!** See [QUICKSTART.md](QUICKSTART.md) for details.

---

**Complete documentation index!** 📚

**Last Updated:** 2025-10-12  
**Version:** 1.0.0  
**Project:** Kelly Asset Pack Generator


