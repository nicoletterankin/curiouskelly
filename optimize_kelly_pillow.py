#!/usr/bin/env python3
"""
Kelly Production Asset Optimizer (Pillow Version)
Compresses and converts Kelly images to web-ready formats
"""

from PIL import Image
from pathlib import Path
import json

# Paths
BASE_DIR = Path(r"C:\Users\user\UI-TARS-desktop\public\assets\kelly\production")
ORIGINAL_DIR = BASE_DIR / "original"
WEBP_DIR = BASE_DIR / "webp"
JPEG_DIR = BASE_DIR / "jpeg"

# Target sizes (width in pixels)
SIZES = [640, 1280, 1920]

# Quality settings
JPEG_QUALITY = 85
WEBP_QUALITY = 82

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

def optimize_image(input_path, output_path, target_width, quality, is_webp=False):
    """Optimize a single image"""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if width > target_width:
                ratio = target_width / width
                new_height = int(height * ratio)
                img = img.resize((target_width, new_height), Image.LANCZOS)
            
            # Save with optimization
            if is_webp:
                img.save(output_path, 'WEBP', quality=quality, method=6)
            else:
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            # Report sizes
            original_size = input_path.stat().st_size / 1024
            new_size = output_path.stat().st_size / 1024
            reduction = (1 - new_size / original_size) * 100
            print(f"  ✅ {output_path.name}: {new_size:.0f}KB ({reduction:.0f}% smaller)")
            return new_size
    except Exception as e:
        print(f"  ❌ Error processing {input_path.name}: {e}")
        return 0

def process_all_assets():
    """Process all Kelly assets"""
    print("\n🎨 Processing Kelly Production Assets with Pillow...")
    print("=" * 50)
    
    total_original = 0
    total_webp = 0
    total_jpeg = 0
    
    for name, filename in KELLY_ASSETS.items():
        input_path = ORIGINAL_DIR / filename
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {name}: {filename} not found")
            continue
        
        original_size = input_path.stat().st_size / 1024
        total_original += original_size
        print(f"\n📷 {name} ({original_size:.0f}KB original)")
        
        # Create WebP versions for all sizes
        for width in SIZES:
            output_name = f"{name}-{width}.webp"
            output_path = WEBP_DIR / output_name
            size = optimize_image(input_path, output_path, width, WEBP_QUALITY, is_webp=True)
            total_webp += size
        
        # Create single JPEG fallback (1280px)
        output_name = f"{name}.jpeg"
        output_path = JPEG_DIR / output_name
        size = optimize_image(input_path, output_path, 1280, JPEG_QUALITY, is_webp=False)
        total_jpeg += size
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Original total: {total_original/1024:.2f}MB")
    print(f"   WebP total: {total_webp/1024:.2f}MB")
    print(f"   JPEG fallbacks: {total_jpeg/1024:.2f}MB")
    print(f"   WebP reduction: {(1 - total_webp/total_original)*100:.0f}%")
    
    return True

def create_manifest():
    """Create a JSON manifest of all assets"""
    manifest = {
        "version": "1.0",
        "description": "Kelly Production Assets - Web Optimized",
        "states": {
            "hello": {
                "description": "Kelly greeting/celebrating",
                "use": "welcome, lesson complete, celebrations"
            },
            "thinking": {
                "description": "Kelly contemplating",
                "use": "waiting for user choice, processing"
            },
            "pointing-left": {
                "description": "Kelly pointing left",
                "use": "Choice A indicator"
            },
            "pointing-right": {
                "description": "Kelly pointing right", 
                "use": "Choice B indicator"
            },
            "in-left": {
                "description": "Close-up, looking left",
                "use": "Intimate engagement, Choice A"
            },
            "in-right": {
                "description": "Close-up, looking right",
                "use": "Intimate engagement, Choice B"
            },
            "mid-left": {
                "description": "Mid shot, looking left",
                "use": "Standard teaching, Choice A"
            },
            "mid-right": {
                "description": "Mid shot, looking right",
                "use": "Standard teaching, Choice B"
            },
            "out-left": {
                "description": "Full body, pointing left",
                "use": "Full context, Choice A"
            },
            "out-right": {
                "description": "Full body, pointing right",
                "use": "Full context, Choice B"
            }
        },
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
    print("🚀 Kelly Production Asset Optimizer (Pillow)")
    print("=" * 50)
    
    # Ensure directories exist
    WEBP_DIR.mkdir(parents=True, exist_ok=True)
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process assets
    if process_all_assets():
        create_manifest()
        print("\n✅ All assets optimized and ready for production!")
        print("\n📁 Output locations:")
        print(f"   WebP: {WEBP_DIR}")
        print(f"   JPEG: {JPEG_DIR}")

