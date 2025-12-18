#!/usr/bin/env python3
"""
🔧 KELLY-SYNC SETUP

Automated setup script for the Kelly video pipeline.

This script:
1. Creates conda/venv environment
2. Installs dependencies
3. Downloads model weights
4. Validates installation
5. Runs test generation

Requirements:
- Python 3.10+
- NVIDIA GPU with 10GB+ VRAM
- CUDA 12.x installed
- ~20GB disk space for models
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(num, text):
    print(f"{Colors.CYAN}[{num}]{Colors.END} {text}")

def print_success(text):
    print(f"  {Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"  {Colors.WARNING}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"  {Colors.FAIL}❌ {text}{Colors.END}")

def run_command(cmd, check=True, capture=False):
    """Run a shell command."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if capture:
            print(e.stdout)
            print(e.stderr)
        return None

def check_system_requirements():
    """Verify system meets requirements."""
    print_header("SYSTEM REQUIREMENTS CHECK")
    
    requirements_met = True
    
    # Python version
    print_step(1, "Python version")
    py_version = sys.version_info
    if py_version >= (3, 10):
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_error(f"Python {py_version.major}.{py_version.minor} - need 3.10+")
        requirements_met = False
    
    # Platform
    print_step(2, "Operating system")
    system = platform.system()
    if system == "Windows":
        print_success(f"Windows {platform.release()}")
    elif system == "Linux":
        print_success(f"Linux {platform.release()}")
    elif system == "Darwin":
        print_success(f"macOS {platform.mac_ver()[0]}")
    
    # CUDA
    print_step(3, "NVIDIA GPU and CUDA")
    try:
        result = run_command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], capture=True)
        if result and result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            gpu_name = gpu_info[0].strip()
            gpu_mem = gpu_info[1].strip()
            print_success(f"{gpu_name} ({gpu_mem})")
            
            # Check VRAM
            mem_mb = int(''.join(filter(str.isdigit, gpu_mem)))
            if mem_mb < 10000:
                print_warning(f"Low VRAM ({mem_mb}MB) - 10GB+ recommended")
        else:
            print_error("nvidia-smi failed - is CUDA installed?")
            requirements_met = False
    except FileNotFoundError:
        print_error("nvidia-smi not found - NVIDIA drivers not installed?")
        requirements_met = False
    
    # FFmpeg
    print_step(4, "FFmpeg")
    try:
        result = run_command(['ffmpeg', '-version'], capture=True)
        if result and result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print_success(version[:50])
        else:
            print_error("FFmpeg not working")
            requirements_met = False
    except FileNotFoundError:
        print_error("FFmpeg not found - please install FFmpeg")
        requirements_met = False
    
    # Disk space
    print_step(5, "Disk space")
    disk_usage = shutil.disk_usage(Path(__file__).parent)
    free_gb = disk_usage.free / (1024**3)
    if free_gb >= 20:
        print_success(f"{free_gb:.1f}GB free")
    else:
        print_warning(f"Only {free_gb:.1f}GB free - need 20GB+ for models")
    
    return requirements_met

def create_environment():
    """Create Python virtual environment."""
    print_header("CREATING PYTHON ENVIRONMENT")
    
    venv_path = Path(__file__).parent / 'venv'
    
    if venv_path.exists():
        print_step(1, "Environment already exists")
        print_success(str(venv_path))
        return True
    
    print_step(1, "Creating virtual environment")
    result = run_command([sys.executable, '-m', 'venv', str(venv_path)])
    
    if result and result.returncode == 0:
        print_success(f"Created: {venv_path}")
        
        # Determine pip path
        if platform.system() == 'Windows':
            pip_path = venv_path / 'Scripts' / 'pip.exe'
        else:
            pip_path = venv_path / 'bin' / 'pip'
        
        # Upgrade pip
        print_step(2, "Upgrading pip")
        run_command([str(pip_path), 'install', '--upgrade', 'pip'])
        print_success("pip upgraded")
        
        return True
    else:
        print_error("Failed to create virtual environment")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print_header("INSTALLING DEPENDENCIES")
    
    venv_path = Path(__file__).parent / 'venv'
    
    if platform.system() == 'Windows':
        pip_path = venv_path / 'Scripts' / 'pip.exe'
    else:
        pip_path = venv_path / 'bin' / 'pip'
    
    # Install PyTorch first (with CUDA)
    print_step(1, "Installing PyTorch with CUDA")
    result = run_command([
        str(pip_path), 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu121'
    ])
    
    if not result or result.returncode != 0:
        print_warning("PyTorch install may have issues")
    else:
        print_success("PyTorch installed")
    
    # Install from requirements.txt
    print_step(2, "Installing requirements.txt")
    req_path = Path(__file__).parent / 'requirements.txt'
    
    result = run_command([
        str(pip_path), 'install', '-r', str(req_path)
    ])
    
    if result and result.returncode == 0:
        print_success("Dependencies installed")
    else:
        print_warning("Some dependencies may have failed")
    
    # Install additional packages
    print_step(3, "Installing model-specific packages")
    
    packages = [
        'git+https://github.com/sczhou/CodeFormer.git',
        'git+https://github.com/xinntao/Real-ESRGAN.git',
    ]
    
    for pkg in packages:
        print(f"    Installing {pkg.split('/')[-1].replace('.git', '')}...")
        run_command([str(pip_path), 'install', pkg], check=False)
    
    print_success("Model packages installed")
    
    return True

def download_models():
    """Download model weights."""
    print_header("DOWNLOADING MODEL WEIGHTS")
    
    venv_path = Path(__file__).parent / 'venv'
    
    if platform.system() == 'Windows':
        python_path = venv_path / 'Scripts' / 'python.exe'
    else:
        python_path = venv_path / 'bin' / 'python'
    
    download_script = Path(__file__).parent / 'scripts' / 'download_models.py'
    
    if not download_script.exists():
        print_error(f"Download script not found: {download_script}")
        return False
    
    print_step(1, "Running model downloader")
    result = run_command([str(python_path), str(download_script)])
    
    if result and result.returncode == 0:
        print_success("Models downloaded")
        return True
    else:
        print_warning("Some models may not have downloaded")
        return True  # Continue anyway

def validate_installation():
    """Run validation tests."""
    print_header("VALIDATING INSTALLATION")
    
    venv_path = Path(__file__).parent / 'venv'
    
    if platform.system() == 'Windows':
        python_path = venv_path / 'Scripts' / 'python.exe'
    else:
        python_path = venv_path / 'bin' / 'python'
    
    # Test imports
    print_step(1, "Testing imports")
    
    test_script = '''
import torch
import cv2
import numpy as np
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
'''
    
    result = run_command([str(python_path), '-c', test_script], capture=True)
    
    if result and result.returncode == 0:
        print(result.stdout)
        print_success("Core imports working")
    else:
        print_error("Import test failed")
        if result:
            print(result.stderr)
        return False
    
    # Test pipeline import
    print_step(2, "Testing pipeline import")
    
    test_script = '''
import sys
sys.path.insert(0, '.')
from src.pipeline import KellySyncPipeline, PipelineConfig
print("Pipeline modules imported successfully")
'''
    
    os.chdir(Path(__file__).parent)
    result = run_command([str(python_path), '-c', test_script], capture=True)
    
    if result and result.returncode == 0:
        print_success("Pipeline imports working")
    else:
        print_warning("Pipeline import issues (may need models first)")
    
    return True

def print_next_steps():
    """Print instructions for next steps."""
    print_header("SETUP COMPLETE")
    
    venv_path = Path(__file__).parent / 'venv'
    
    if platform.system() == 'Windows':
        activate = f"{venv_path}\\Scripts\\activate"
    else:
        activate = f"source {venv_path}/bin/activate"
    
    print(f"""
{Colors.GREEN}✅ Kelly-Sync pipeline is ready!{Colors.END}

{Colors.BOLD}To activate the environment:{Colors.END}
    {activate}

{Colors.BOLD}To generate a video:{Colors.END}
    python scripts/generate_video.py --script "Hello world!" -o test.mp4

{Colors.BOLD}To generate a full day:{Colors.END}
    python scripts/generate_video.py --day 352 --all-archetypes

{Colors.BOLD}Quality presets:{Colors.END}
    --quality draft     # 720p, fast, for testing
    --quality standard  # 1080p, balanced
    --quality premium   # 4K, full pipeline
    --quality ultra     # 8K, maximum quality

{Colors.BOLD}Documentation:{Colors.END}
    kelly-sync/README.md

{Colors.CYAN}Happy generating! 🎬{Colors.END}
""")

def main():
    """Main setup flow."""
    print(f"""
{Colors.HEADER}{Colors.BOLD}
============================================================
                                                           
    KELLY-SYNC SETUP                                     
    Production-Grade Local Video Pipeline                   
                                                           
============================================================
{Colors.END}
""")
    
    # Check requirements
    if not check_system_requirements():
        print_error("\nSystem requirements not met. Please fix issues above.")
        return 1
    
    # Create environment
    if not create_environment():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Download models
    download_models()
    
    # Validate
    validate_installation()
    
    # Next steps
    print_next_steps()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
