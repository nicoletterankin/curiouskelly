#!/usr/bin/env python3
"""
Kelly Production Asset Optimizer - HIGH QUALITY VERSION
Preserves 4K quality while optimizing for web
"""

from PIL import Image
from pathlib import Path
import json

# Paths
BASE_DIR = Path(r"C:\Users\user\UI-TARS-desktop\public\assets\kelly\production")
ORIGINAL_DIR = BASE_DIR / "original"
WEBP_DIR = BASE_DIR / "webp"
JPEG_DIR = BASE_DIR / "jpeg"

# Target sizes - include full resolution for 4K displays
SIZES = [640, 1280, 1920, 2560]  # Added 2560 for 4K/Retina

# HIGH QUALITY settings (was 82/85, now 92/95)
JPEG_QUALITY = 95
WEBP_QUALITY = 92  # Higher quality, larger files but SHARP

# Kelly assets mapping
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
    """Optimize a single image with HIGH QUALITY"""
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
                # Use LANCZOS for best quality downscaling
                img = img.resize((target_width, new_height), Image.LANCZOS)
            
            # Save with HIGH QUALITY optimization
            if is_webp:
                # method=4 is good balance (not max compression)
                # lossless=False but high quality
                img.save(output_path, 'WEBP', quality=quality, method=4)
            else:
                # subsampling=0 = 4:4:4 (no chroma subsampling = sharper)
                img.save(output_path, 'JPEG', quality=quality, optimize=True, subsampling=0)
            
            # Report sizes
            original_size = input_path.stat().st_size / 1024
            new_size = output_path.stat().st_size / 1024
            print(f"  ✅ {output_path.name}: {new_size:.0f}KB")
            return new_size
    except Exception as e:
        print(f"  ❌ Error processing {input_path.name}: {e}")
        return 0

def process_all_assets():
    """Process all Kelly assets with HIGH QUALITY"""
    print("\n🎨 Processing Kelly Assets - HIGH QUALITY MODE")
    print("=" * 50)
    print(f"WebP Quality: {WEBP_QUALITY}%")
    print(f"JPEG Quality: {JPEG_QUALITY}%")
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
        
        # Create JPEG fallback at 1920px (good for most screens)
        output_name = f"{name}.jpeg"
        output_path = JPEG_DIR / output_name
        size = optimize_image(input_path, output_path, 1920, JPEG_QUALITY, is_webp=False)
        total_jpeg += size
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Original total: {total_original/1024:.2f}MB")
    print(f"   WebP total: {total_webp/1024:.2f}MB")
    print(f"   JPEG fallbacks: {total_jpeg/1024:.2f}MB")
    print(f"   Size reduction: {(1 - (total_webp + total_jpeg)/(total_original*2))*100:.0f}%")
    
    return True

def create_manifest():
    """Create a JSON manifest of all assets"""
    manifest = {
        "version": "2.0",
        "description": "Kelly Production Assets - HIGH QUALITY Web Optimized",
        "quality": {
            "webp": WEBP_QUALITY,
            "jpeg": JPEG_QUALITY
        },
        "sizes": SIZES,
        "states": {
            "hello": {"description": "Kelly greeting", "use": "welcome, celebrations"},
            "thinking": {"description": "Kelly contemplating", "use": "waiting, processing"},
            "pointing-left": {"description": "Pointing left", "use": "Choice A"},
            "pointing-right": {"description": "Pointing right", "use": "Choice B"},
            "in-left": {"description": "Close-up left", "use": "Intimate, Choice A"},
            "in-right": {"description": "Close-up right", "use": "Intimate, Choice B"},
            "mid-left": {"description": "Mid shot left", "use": "Teaching, Choice A"},
            "mid-right": {"description": "Mid shot right", "use": "Teaching, Choice B"},
            "out-left": {"description": "Full body left", "use": "Overview, Choice A"},
            "out-right": {"description": "Full body right", "use": "Overview, Choice B"}
        },
        "assets": {}
    }
    
    for name in KELLY_ASSETS.keys():
        manifest["assets"][name] = {
            "webp": {str(s): f"/assets/kelly/production/webp/{name}-{s}.webp" for s in SIZES},
            "jpeg": f"/assets/kelly/production/jpeg/{name}.jpeg"
        }
    
    manifest_path = BASE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest updated: {manifest_path}")
    return manifest

if __name__ == "__main__":
    print("🚀 Kelly HQ Asset Optimizer")
    print("=" * 50)
    
    # Ensure directories exist
    WEBP_DIR.mkdir(parents=True, exist_ok=True)
    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process assets
    if process_all_assets():
        create_manifest()
        print("\n✅ HIGH QUALITY assets ready!")
        print("\n📁 Output locations:")
        print(f"   WebP: {WEBP_DIR}")
        print(f"   JPEG: {JPEG_DIR}")




