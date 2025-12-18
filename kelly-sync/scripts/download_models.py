#!/usr/bin/env python3
"""
🎬 KELLY-SYNC Model Downloader
Downloads all required model weights for production pipeline.

Total download size: ~15GB
Disk space required: ~20GB (with extraction)
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import gzip
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"

# Model definitions with checksums for verification
MODELS = {
    "video_retalking": {
        "description": "VideoReTalking - State-of-the-art lip synthesis",
        "files": [
            {
                "name": "30_net_gen.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/30_net_gen.pth",
                "size_mb": 717,
                "md5": None,  # Add if available
            },
            {
                "name": "BFM.zip",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/BFM.zip",
                "size_mb": 153,
                "extract": True,
            },
            {
                "name": "DNet.pt",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/DNet.pt",
                "size_mb": 149,
            },
            {
                "name": "ENet.pt",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/ENet.pt",
                "size_mb": 4,
            },
            {
                "name": "expression.mat",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/expression.mat",
                "size_mb": 1,
            },
            {
                "name": "face3d_pretrain_epoch_20.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/face3d_pretrain_epoch_20.pth",
                "size_mb": 286,
            },
            {
                "name": "GFPGANv1.3.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/GFPGANv1.3.pth",
                "size_mb": 348,
            },
            {
                "name": "GPEN-BFR-512.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/GPEN-BFR-512.pth",
                "size_mb": 268,
            },
            {
                "name": "LNet.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/LNet.pth",
                "size_mb": 178,
            },
            {
                "name": "ParseNet-latest.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/ParseNet-latest.pth",
                "size_mb": 85,
            },
            {
                "name": "RetinaFace-R50.pth",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/RetinaFace-R50.pth",
                "size_mb": 104,
            },
            {
                "name": "shape_predictor_68_face_landmarks.dat",
                "url": "https://huggingface.co/vinthony/video-retalking/resolve/main/shape_predictor_68_face_landmarks.dat",
                "size_mb": 100,
            },
        ]
    },
    "codeformer": {
        "description": "CodeFormer - Face restoration for photorealistic output",
        "files": [
            {
                "name": "codeformer.pth",
                "url": "https://huggingface.co/NCKU-NVIDIA/CodeFormer/resolve/main/codeformer.pth",
                "size_mb": 376,
            },
            {
                "name": "facelib/detection_Resnet50_Final.pth",
                "url": "https://huggingface.co/NCKU-NVIDIA/CodeFormer/resolve/main/facelib/detection_Resnet50_Final.pth",
                "size_mb": 104,
            },
            {
                "name": "facelib/parsing_parsenet.pth",
                "url": "https://huggingface.co/NCKU-NVIDIA/CodeFormer/resolve/main/facelib/parsing_parsenet.pth",
                "size_mb": 85,
            },
        ]
    },
    "real_esrgan": {
        "description": "Real-ESRGAN - 4K/8K super resolution",
        "files": [
            {
                "name": "RealESRGAN_x4plus.pth",
                "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus.pth",
                "size_mb": 64,
            },
            {
                "name": "RealESRGAN_x4plus_anime_6B.pth",
                "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4plus_anime_6B.pth",
                "size_mb": 17,
            },
            {
                "name": "RealESRGAN_x2plus.pth",
                "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2plus.pth",
                "size_mb": 64,
            },
        ]
    },
    "fomm": {
        "description": "First Order Motion Model - Motion transfer from HeyGen",
        "files": [
            {
                "name": "vox-adv-cpk.pth.tar",
                "url": "https://huggingface.co/spaces/PAIR/Text2Video-Zero/resolve/main/fomm/vox-adv-cpk.pth.tar",
                "size_mb": 229,
            },
        ]
    },
    "face_detection": {
        "description": "Face detection and alignment models",
        "files": [
            {
                "name": "s3fd-619a316812.pth",
                "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                "size_mb": 89,
            },
            {
                "name": "2DFAN4-cd938726ad.zip",
                "url": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
                "size_mb": 96,
                "extract": True,
            },
        ]
    },
}


def download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract zip file."""
    print(f"  📦 Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    # Optionally remove zip after extraction
    # zip_path.unlink()


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    if not expected_md5:
        return True
    
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    
    return md5.hexdigest() == expected_md5


def download_model_group(group_name: str, group_config: dict) -> bool:
    """Download all files for a model group."""
    print(f"\n{'='*60}")
    print(f"📥 {group_config['description']}")
    print(f"{'='*60}")
    
    group_dir = MODELS_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    
    success = True
    for file_config in group_config['files']:
        file_name = file_config['name']
        file_url = file_config['url']
        file_path = group_dir / file_name
        
        # Check if already exists
        if file_path.exists():
            print(f"  ✅ {file_name} (already exists)")
            continue
        
        # Create subdirectories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download
        print(f"  📥 Downloading {file_name}...")
        if not download_file(file_url, file_path, file_name):
            success = False
            continue
        
        # Verify checksum
        if file_config.get('md5'):
            if not verify_checksum(file_path, file_config['md5']):
                print(f"  ❌ Checksum mismatch for {file_name}")
                success = False
                continue
        
        # Extract if needed
        if file_config.get('extract') and file_name.endswith('.zip'):
            extract_zip(file_path, group_dir)
        
        print(f"  ✅ {file_name}")
    
    return success


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  🎬 KELLY-SYNC MODEL DOWNLOADER                            ║")
    print("║  Production-grade video pipeline models                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Calculate total size
    total_mb = sum(
        f.get('size_mb', 0) 
        for group in MODELS.values() 
        for f in group['files']
    )
    print(f"\n📊 Total download size: ~{total_mb / 1024:.1f} GB")
    print(f"📁 Models directory: {MODELS_DIR}")
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download each model group
    results = {}
    for group_name, group_config in MODELS.items():
        results[group_name] = download_model_group(group_name, group_config)
    
    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    
    all_success = True
    for group_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {group_name}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print("   Run 'python scripts/generate_video.py --help' to get started.")
    else:
        print("\n⚠️  Some downloads failed. Please retry or download manually.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
