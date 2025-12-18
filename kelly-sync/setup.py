#!/usr/bin/env python3
"""
🔧 KELLY-SYNC Setup Script

Automated setup for the production video pipeline.
Run this once to prepare all dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

KELLY_SYNC_DIR = Path(__file__).parent
PYTHON = sys.executable


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and return success status."""
    print(f"\n▶ {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
        )
        print(f"  ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e.stderr[:200] if e.stderr else 'No error output'}")
        return False


def check_gpu():
    """Check CUDA availability."""
    print_header("GPU CHECK")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {gpu_mem:.1f} GB")
            return True
        else:
            print("  ⚠️ CUDA not available")
            print("     GPU processing will be disabled")
            return False
    except ImportError:
        print("  ⚠️ PyTorch not installed")
        return False


def check_ffmpeg():
    """Check FFmpeg installation."""
    print_header("FFMPEG CHECK")
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        version_line = result.stdout.split("\n")[0]
        print(f"  ✅ FFmpeg installed: {version_line}")
        return True
    except FileNotFoundError:
        print("  ❌ FFmpeg not found")
        print("     Please install FFmpeg: https://ffmpeg.org/download.html")
        return False


def create_directories():
    """Create required directories."""
    print_header("CREATING DIRECTORIES")
    
    dirs = [
        KELLY_SYNC_DIR / "models",
        KELLY_SYNC_DIR / "models" / "video_retalking",
        KELLY_SYNC_DIR / "models" / "codeformer",
        KELLY_SYNC_DIR / "models" / "real_esrgan",
        KELLY_SYNC_DIR / "models" / "fomm",
        KELLY_SYNC_DIR / "assets",
        KELLY_SYNC_DIR / "assets" / "motion_templates",
        KELLY_SYNC_DIR / "output",
        KELLY_SYNC_DIR / "temp",
        KELLY_SYNC_DIR / "logs",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d.relative_to(KELLY_SYNC_DIR)}")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print_header("INSTALLING DEPENDENCIES")
    
    requirements_file = KELLY_SYNC_DIR / "requirements.txt"
    
    if not requirements_file.exists():
        print("  ❌ requirements.txt not found")
        return False
    
    # Install PyTorch with CUDA first
    if platform.system() == "Windows":
        pytorch_cmd = [
            PYTHON, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ]
    else:
        pytorch_cmd = [
            PYTHON, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
        ]
    
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA"):
        print("  ⚠️ PyTorch installation may have issues")
    
    # Install other requirements
    pip_cmd = [PYTHON, "-m", "pip", "install", "-r", str(requirements_file)]
    return run_command(pip_cmd, "Installing requirements", check=False)


def download_models():
    """Download model weights."""
    print_header("DOWNLOADING MODELS")
    
    download_script = KELLY_SYNC_DIR / "scripts" / "download_models.py"
    
    if not download_script.exists():
        print("  ❌ download_models.py not found")
        return False
    
    return run_command(
        [PYTHON, str(download_script)],
        "Downloading model weights (~15GB)",
    )


def setup_kelly_reference():
    """Set up Kelly reference image."""
    print_header("KELLY REFERENCE IMAGE")
    
    # Check for Kelly reference
    kelly_ref = Path("C:/iLearnStudio/projects/Kelly/Ref/Best Character Reference/head and shoulders without chair.png")
    kelly_local = KELLY_SYNC_DIR / "assets" / "kelly_reference_4k.png"
    
    if kelly_ref.exists():
        print(f"  ✅ Found Kelly reference: {kelly_ref}")
        
        # Copy to local assets
        if not kelly_local.exists():
            import shutil
            shutil.copy(kelly_ref, kelly_local)
            print(f"  ✅ Copied to: {kelly_local}")
        
        return True
    else:
        print(f"  ⚠️ Kelly reference not found at: {kelly_ref}")
        print("     Please provide Kelly reference image")
        return False


def run_verification():
    """Run a quick verification test."""
    print_header("VERIFICATION")
    
    try:
        # Test imports
        print("  Testing imports...")
        
        import torch
        print(f"    ✅ PyTorch {torch.__version__}")
        
        import cv2
        print(f"    ✅ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"    ✅ NumPy {np.__version__}")
        
        try:
            import whisper
            print(f"    ✅ Whisper installed")
        except ImportError:
            print(f"    ⚠️ Whisper not installed (optional for phoneme extraction)")
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            print(f"    ✅ BasicSR installed")
        except ImportError:
            print(f"    ⚠️ BasicSR not installed")
        
        try:
            from realesrgan import RealESRGANer
            print(f"    ✅ Real-ESRGAN installed")
        except ImportError:
            print(f"    ⚠️ Real-ESRGAN not installed")
        
        print("\n  ✅ Basic verification passed")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Verification failed: {e}")
        return False


def main():
    """Main setup routine."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  🎬 KELLY-SYNC PRODUCTION PIPELINE SETUP                   ║")
    print("║  4K/8K Photorealistic Video Generation                     ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Run all setup steps
    results["directories"] = create_directories()
    results["ffmpeg"] = check_ffmpeg()
    results["dependencies"] = install_dependencies()
    results["gpu"] = check_gpu()
    results["kelly_ref"] = setup_kelly_reference()
    
    # Ask about model download (large)
    print("\n" + "-" * 60)
    print("Model download is ~15GB. This may take a while.")
    response = input("Download models now? [y/N]: ").strip().lower()
    
    if response == "y":
        results["models"] = download_models()
    else:
        results["models"] = None
        print("  Skipped. Run 'python scripts/download_models.py' later.")
    
    # Verification
    results["verify"] = run_verification()
    
    # Summary
    print_header("SETUP SUMMARY")
    
    all_ok = True
    for step, result in results.items():
        if result is None:
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ OK"
        else:
            status = "❌ FAILED"
            all_ok = False
        
        print(f"  {status}: {step}")
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("🎉 Setup complete! Ready for production.")
        print("\nNext steps:")
        print("  1. Run: python scripts/download_models.py (if skipped)")
        print("  2. Test: python scripts/generate_video.py --help")
        print("  3. Generate: python scripts/generate_video.py --audio <file>")
    else:
        print("⚠️ Some steps failed. Please review and retry.")
        print("\nFor help, check the README.md or logs/")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
