#!/usr/bin/env python3
"""
🔧 SETUP LOCAL VIDEO PIPELINE
==============================
Installs and configures all dependencies for local Kelly video generation.

Requirements:
- NVIDIA GPU with 12GB+ VRAM (24GB+ recommended)
- CUDA 12.x installed
- Python 3.10+
- 50GB+ free disk space for models

Usage:
    python setup_local_pipeline.py --install-all
    python setup_local_pipeline.py --check-only
    python setup_local_pipeline.py --download-models
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# Configuration
MODELS_DIR = Path(__file__).parent.parent / "models"
SADTALKER_REPO = "https://github.com/OpenTalker/SadTalker.git"
TORTOISE_REPO = "https://github.com/neonbjb/tortoise-tts.git"


def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability and VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No CUDA GPU detected"
        
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if vram < 8:
            return False, f"Insufficient VRAM: {vram:.1f}GB (need 8GB+)"
        
        return True, f"{gpu_name} ({vram:.1f}GB VRAM)"
    except ImportError:
        return False, "PyTorch not installed"


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check all required dependencies."""
    results = []
    
    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_ok = sys.version_info >= (3, 10)
    results.append(("Python 3.10+", py_ok, py_ver))
    
    # CUDA
    cuda_ver = os.environ.get('CUDA_VERSION', 'unknown')
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_ver = result.stdout.split('release')[-1].split(',')[0].strip()
    except:
        pass
    cuda_ok = cuda_ver != 'unknown'
    results.append(("CUDA", cuda_ok, cuda_ver))
    
    # PyTorch
    try:
        import torch
        torch_ok = True
        torch_ver = f"{torch.__version__} (CUDA: {torch.version.cuda})"
    except ImportError:
        torch_ok = False
        torch_ver = "Not installed"
    results.append(("PyTorch", torch_ok, torch_ver))
    
    # GPU
    gpu_ok, gpu_info = check_gpu()
    results.append(("GPU", gpu_ok, gpu_info))
    
    # FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        ffmpeg_ok = result.returncode == 0
        ffmpeg_ver = result.stdout.split('ffmpeg version')[1].split()[0] if ffmpeg_ok else "Not found"
    except:
        ffmpeg_ok = False
        ffmpeg_ver = "Not installed"
    results.append(("FFmpeg", ffmpeg_ok, ffmpeg_ver))
    
    # Check key packages
    packages = [
        ('numpy', 'numpy'),
        ('opencv', 'cv2'),
        ('PIL/Pillow', 'PIL'),
        ('librosa', 'librosa'),
        ('torchaudio', 'torchaudio'),
    ]
    
    for name, module in packages:
        try:
            __import__(module)
            results.append((name, True, "OK"))
        except ImportError:
            results.append((name, False, "Not installed"))
    
    return results


def install_base_requirements():
    """Install base Python requirements."""
    print("\n📦 Installing base requirements...")
    
    requirements_file = Path(__file__).parent.parent / "requirements-local.txt"
    
    if not requirements_file.exists():
        print(f"  ❌ Requirements file not found: {requirements_file}")
        return False
    
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def clone_sadtalker():
    """Clone and setup SadTalker."""
    print("\n🎬 Setting up SadTalker...")
    
    sadtalker_dir = MODELS_DIR / "SadTalker"
    
    if sadtalker_dir.exists():
        print(f"  ✓ SadTalker already exists at {sadtalker_dir}")
        return True
    
    # Clone repo
    print(f"  Cloning from {SADTALKER_REPO}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        ['git', 'clone', '--depth', '1', SADTALKER_REPO, str(sadtalker_dir)],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"  ❌ Clone failed: {result.stderr}")
        return False
    
    # Install SadTalker requirements
    sadtalker_req = sadtalker_dir / "requirements.txt"
    if sadtalker_req.exists():
        print("  Installing SadTalker requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(sadtalker_req)])
    
    print(f"  ✓ SadTalker installed at {sadtalker_dir}")
    return True


def download_sadtalker_models():
    """Download SadTalker model weights."""
    print("\n⬇️ Downloading SadTalker models...")
    
    sadtalker_dir = MODELS_DIR / "SadTalker"
    checkpoints_dir = sadtalker_dir / "checkpoints"
    
    if not sadtalker_dir.exists():
        print("  ❌ SadTalker not installed. Run with --install-all first.")
        return False
    
    # Check if models exist
    expected_models = [
        "SadTalker_V0.0.2_256.safetensors",
        "SadTalker_V0.0.2_512.safetensors",
        "mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar",
    ]
    
    missing = []
    for model in expected_models:
        if not (checkpoints_dir / model).exists():
            missing.append(model)
    
    if not missing:
        print("  ✓ All SadTalker models present")
        return True
    
    print(f"  Missing models: {missing}")
    print("\n  Download models manually from:")
    print("  https://github.com/OpenTalker/SadTalker#-2-download-models")
    print(f"\n  Place in: {checkpoints_dir}")
    
    return False


def install_tortoise():
    """Install Tortoise TTS."""
    print("\n🔊 Setting up Tortoise TTS...")
    
    try:
        import tortoise
        print("  ✓ Tortoise TTS already installed")
        return True
    except ImportError:
        pass
    
    print("  Installing tortoise-tts...")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'tortoise-tts'],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"  ❌ Installation failed: {result.stderr}")
        return False
    
    print("  ✓ Tortoise TTS installed")
    return True


def print_status(results: List[Tuple[str, bool, str]]):
    """Print status table."""
    print("\n" + "=" * 50)
    print("DEPENDENCY STATUS")
    print("=" * 50)
    
    max_name = max(len(r[0]) for r in results)
    
    for name, ok, info in results:
        icon = "✓" if ok else "❌"
        print(f"  {icon} {name.ljust(max_name)} : {info}")
    
    print("=" * 50)
    
    all_ok = all(r[1] for r in results)
    if all_ok:
        print("✅ All dependencies satisfied!")
    else:
        print("❌ Some dependencies missing. Run with --install-all")
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Local Video Pipeline")
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies')
    parser.add_argument('--install-all', action='store_true', help='Install everything')
    parser.add_argument('--download-models', action='store_true', help='Download model weights')
    
    args = parser.parse_args()
    
    print("🔧 LOCAL VIDEO PIPELINE SETUP")
    print("=" * 50)
    
    # Check dependencies
    results = check_dependencies()
    all_ok = print_status(results)
    
    if args.check_only:
        return 0 if all_ok else 1
    
    if args.install_all:
        # Install base requirements
        if not install_base_requirements():
            print("❌ Failed to install base requirements")
            return 1
        
        # Install Tortoise TTS
        if not install_tortoise():
            print("⚠️ Tortoise TTS installation failed")
        
        # Clone SadTalker
        if not clone_sadtalker():
            print("❌ Failed to setup SadTalker")
            return 1
        
        # Re-check
        print("\n📋 Final status check...")
        results = check_dependencies()
        print_status(results)
    
    if args.download_models:
        download_sadtalker_models()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Download SadTalker models (see links above)")
    print("  2. Test pipeline: python local_video_pipeline.py --test")
    print("  3. Generate Day 51: python local_video_pipeline.py --day 51 --phase hook")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
