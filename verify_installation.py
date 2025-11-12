#!/usr/bin/env python3
"""
Verify Kelly Asset Pack Generator installation and dependencies.
Run this after pip install to ensure everything is working.
"""
import sys
import importlib
from pathlib import Path


def check_import(module_name, optional=False):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name:20s} — OK")
        return True
    except ImportError as e:
        if optional:
            print(f"⊘ {module_name:20s} — Optional (not installed)")
            return True
        else:
            print(f"✗ {module_name:20s} — MISSING: {e}")
            return False


def check_kelly_pack():
    """Check kelly_pack modules."""
    modules = [
        "kelly_pack",
        "kelly_pack.cli",
        "kelly_pack.io_utils",
        "kelly_pack.crop_scale",
        "kelly_pack.matting",
        "kelly_pack.alpha_tools",
        "kelly_pack.composite",
        "kelly_pack.diffuse",
        "kelly_pack.sprite",
        "kelly_pack.physics_sheet",
        "kelly_pack.video_frame",
    ]
    
    all_ok = True
    for module in modules:
        if not check_import(module):
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check required and optional dependencies."""
    print("\n" + "=" * 60)
    print("Checking Required Dependencies")
    print("=" * 60)
    
    required = [
        "PIL",
        "numpy",
        "cv2",
        "reportlab",
        "matplotlib",
    ]
    
    all_ok = True
    for module in required:
        if not check_import(module):
            all_ok = False
    
    print("\n" + "=" * 60)
    print("Checking Optional Dependencies")
    print("=" * 60)
    
    optional = [
        "torch",
        "torchvision",
        "imageio",
    ]
    
    for module in optional:
        check_import(module, optional=True)
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU Support")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print("⊘ CUDA not available (CPU-only mode)")
    except ImportError:
        print("⊘ PyTorch not installed (GPU check skipped)")


def check_cli():
    """Check CLI executable."""
    print("\n" + "=" * 60)
    print("Checking CLI")
    print("=" * 60)
    
    try:
        from kelly_pack.cli import main
        print("✓ CLI module imports successfully")
        print("  Run with: python -m kelly_pack.cli --help")
        return True
    except Exception as e:
        print(f"✗ CLI module failed: {e}")
        return False


def check_project_structure():
    """Check project structure."""
    print("\n" + "=" * 60)
    print("Checking Project Structure")
    print("=" * 60)
    
    required_files = [
        "kelly_pack/__init__.py",
        "kelly_pack/cli.py",
        "requirements.txt",
        "README.md",
    ]
    
    optional_files = [
        "tests/test_shapes_and_files.py",
        "scripts/build_all.py",
        "Makefile",
        "setup.py",
    ]
    
    all_ok = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file:40s} — OK")
        else:
            print(f"✗ {file:40s} — MISSING")
            all_ok = False
    
    for file in optional_files:
        if Path(file).exists():
            print(f"✓ {file:40s} — OK")
        else:
            print(f"⊘ {file:40s} — Optional (missing)")
    
    return all_ok


def run_simple_test():
    """Run a simple functionality test."""
    print("\n" + "=" * 60)
    print("Running Simple Functionality Test")
    print("=" * 60)
    
    try:
        import numpy as np
        from kelly_pack import alpha_tools, composite, diffuse
        
        # Test alpha variants
        print("Testing alpha variants...")
        base_alpha = np.random.rand(100, 100).astype(np.float32)
        alpha_soft, alpha_tight, alpha_edge = alpha_tools.generate_alpha_variants(base_alpha)
        assert alpha_soft.shape == (100, 100)
        assert alpha_tight.shape == (100, 100)
        assert alpha_edge.shape == (100, 100)
        print("  ✓ Alpha variants work")
        
        # Test gradient
        print("Testing gradient generation...")
        gradient = composite.create_vertical_gradient(100, 100, "#22262A", "#080808")
        assert gradient.shape == (100, 100, 3)
        print("  ✓ Gradient generation works")
        
        # Test diffuse
        print("Testing diffuse neutralization...")
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        neutral = diffuse.neutralize_diffuse(img, 0.15)
        assert neutral.shape == (100, 100, 3)
        print("  ✓ Diffuse neutralization works")
        
        print("\n✓ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Kelly Asset Pack Generator - Installation Verification")
    print("=" * 60)
    
    results = {
        "Kelly Pack Modules": check_kelly_pack(),
        "Dependencies": check_dependencies(),
        "Project Structure": check_project_structure(),
        "CLI": check_cli(),
        "Functionality": run_simple_test(),
    }
    
    check_gpu()
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:25s} — {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou're ready to use Kelly Asset Pack Generator!")
        print("\nNext steps:")
        print("  1. Place input images (kelly2-directors-chair.jpeg, etc.)")
        print("  2. Run: python -m kelly_pack.cli build")
        print("  3. Check outputs in ./output/")
        print("\nFor help: python -m kelly_pack.cli --help")
        print("For docs: see README.md and QUICKSTART.md")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())


