# Changelog

All notable changes to the Kelly Asset Pack Generator will be documented in this file.

## [1.0.0] - 2025-10-12

### Added
- Initial release
- Complete 8K asset generation pipeline
- Model-based hair matting (U²-Net support)
- Heuristic fallback matting for white backgrounds
- Soft and tight alpha variants
- Dark-mode hero composite with gradient
- Square sprite generation with configurable padding
- Diffuse texture neutralization (gray-world + contrast flatten)
- Edge-aware alpha upsampling with guided filter
- Physics reference sheet PDF generation
- Video mid-frame extraction support
- Comprehensive CLI with subcommands
- PyTest test suite
- Complete documentation
- Makefile for convenience
- Example usage scripts

### Features
- 16:9 hero exports (7680×4320)
- Square sprite (8192×8192)
- Alpha utilities (soft/tight/edge)
- Diffuse neutral textures
- 100% open-source dependencies
- Offline-ready after weight download
- CUDA acceleration support
- Configurable alpha tuning parameters
- Automatic input file discovery


