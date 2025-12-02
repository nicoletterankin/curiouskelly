#!/usr/bin/env python3
"""
Kelly Production Asset Optimizer
Compresses and converts Kelly images to web-ready formats
"""

import os
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(r"C:\Users\user\UI-TARS-desktop\public\assets\kelly\production")
ORIGINAL_DIR = BASE_DIR / "original"
WEBP_DIR = BASE_DIR / "webp"
JPEG_DIR = BASE_DIR / "jpeg"

# Target sizes (width in pixels)
SIZES = [640, 1280, 1920]

# Quality settings
JPEG_QUALITY = 85
WEBP_QUALITY = 80

# Kelly assets mapping (friendly name -> filename)
KELLY_ASSETS = {
    "hello": "kelly-hello.jpeg",
    "thinking": "kelly-thinking.jpeg", 
    "pointing-left": "kelly-pointing-left.jpeg",
    "pointing-right": "kelly-pointing-right.jpeg",
    "in-left": "in-left.jpeg",
    "in-right": "in-right.jpeg",
    "mid-left": "mid-left.jpeg",
    "mid-right": "mid-right.jpeg",
    "out-left": "out-left.jpeg",
    "out-right": "out-right.jpeg"
}

def check_imagemagick():
    """Check if ImageMagick is available"""
    try:
        result = subprocess.run(["magick", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ImageMagick found")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ ImageMagick not found. Please install it:")
    print("   winget install ImageMagick.ImageMagick")
    return False

def optimize_image(input_path, output_path, width, quality, format_type):
    """Optimize a single image using ImageMagick"""
    cmd = [
        "magick",
        str(input_path),
        "-resize", f"{width}x>",  # Resize only if larger
        "-strip",  # Remove metadata
        "-quality", str(quality),
    ]
    
    if format_type == "webp":
        cmd.extend(["-define", "webp:lossless=false"])
    
    cmd.append(str(output_path))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Get file sizes
            original_size = input_path.stat().st_size / 1024  # KB
            new_size = output_path.stat().st_size / 1024  # KB
            reduction = (1 - new_size / original_size) * 100
            print(f"  ✅ {output_path.name}: {new_size:.0f}KB ({reduction:.0f}% smaller)")
            return True
        else:
            print(f"  ❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def process_all_assets():
    """Process all Kelly assets"""
    if not check_imagemagick():
        return False
    
    print("\n🎨 Processing Kelly Production Assets...")
    print("=" * 50)
    
    total_original = 0
    total_optimized = 0
    
    for name, filename in KELLY_ASSETS.items():
        input_path = ORIGINAL_DIR / filename
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {name}: {filename} not found")
            continue
        
        original_size = input_path.stat().st_size / 1024
        total_original += original_size
        print(f"\n📷 {name} ({original_size:.0f}KB original)")
        
        # Create WebP versions
        for width in SIZES:
            output_name = f"{name}-{width}.webp"
            output_path = WEBP_DIR / output_name
            if optimize_image(input_path, output_path, width, WEBP_QUALITY, "webp"):
                total_optimized += output_path.stat().st_size / 1024
        
        # Create JPEG versions (just the main size for fallback)
        output_name = f"{name}.jpeg"
        output_path = JPEG_DIR / output_name
        if optimize_image(input_path, output_path, 1280, JPEG_QUALITY, "jpeg"):
            pass  # Don't double count
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Original total: {total_original/1024:.1f}MB")
    print(f"   Optimized WebP: {total_optimized/1024:.1f}MB")
    print(f"   Reduction: {(1 - total_optimized/total_original)*100:.0f}%")
    
    return True

def create_manifest():
    """Create a JSON manifest of all assets"""
    import json
    
    manifest = {
        "version": "1.0",
        "assets": {}
    }
    
    for name in KELLY_ASSETS.keys():
        manifest["assets"][name] = {
            "webp": {
                "640": f"/assets/kelly/production/webp/{name}-640.webp",
                "1280": f"/assets/kelly/production/webp/{name}-1280.webp",
                "1920": f"/assets/kelly/production/webp/{name}-1920.webp"
            },
            "jpeg": f"/assets/kelly/production/jpeg/{name}.jpeg"
        }
    
    manifest_path = BASE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest created: {manifest_path}")
    return manifest

if __name__ == "__main__":
    print("🚀 Kelly Production Asset Optimizer")
    print("=" * 50)
    
    # Ensure directories exist
    WEBP_DIR.mkdir(parents=True, exist_ok=True)
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process assets
    if process_all_assets():
        create_manifest()
        print("\n✅ All assets optimized and ready for production!")
    else:
        print("\n❌ Optimization failed. Please install ImageMagick.")



