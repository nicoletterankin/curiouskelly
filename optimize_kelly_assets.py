#!/usr/bin/env python3
"""
Kelly Asset Optimization - Creative Agency Standard
Optimizes images for web delivery while maintaining visual quality
"""

from PIL import Image
import os
from pathlib import Path

def optimize_image(input_path, output_path, max_width=None, quality=88, progressive=True):
    """
    Optimize a single image for web delivery
    
    Args:
        input_path: Source image path
        output_path: Destination path
        max_width: Maximum width (maintains aspect ratio). None = keep original
        quality: JPEG quality (85-95 recommended, 88 is sweet spot)
        progressive: Use progressive JPEG encoding
    """
    print(f"\n📸 Processing: {Path(input_path).name}")
    
    # Open image
    img = Image.open(input_path)
    original_size = os.path.getsize(input_path)
    
    # Convert RGBA to RGB if needed
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    
    # Resize if needed
    if max_width and img.width > max_width:
        aspect_ratio = img.height / img.width
        new_height = int(max_width * aspect_ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        print(f"   ✓ Resized: {img.width}x{img.height}")
    else:
        print(f"   • Original size: {img.width}x{img.height}")
    
    # Save optimized
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(
        output_path,
        'JPEG',
        quality=quality,
        optimize=True,
        progressive=progressive
    )
    
    # Report results
    optimized_size = os.path.getsize(output_path)
    reduction = ((original_size - optimized_size) / original_size) * 100
    
    print(f"   • Original: {original_size / 1024 / 1024:.2f} MB")
    print(f"   • Optimized: {optimized_size / 1024:.1f} KB")
    print(f"   ✓ Reduction: {reduction:.1f}%")
    
    return optimized_size

# ============================================================================
# HOMEPAGE HERO
# ============================================================================
print("=" * 70)
print("🎨 KELLY ASSET OPTIMIZATION - Creative Agency Standard")
print("=" * 70)

hero_source = "daily-lesson-marketing/public/lessons/images/walk/open-walk.jpeg"
hero_dest = "public/images/kelly-homepage-hero.jpeg"

hero_size = optimize_image(
    hero_source,
    hero_dest,
    max_width=1200,  # Perfect for hero images (retina-ready)
    quality=88,      # Sweet spot: imperceptible quality loss, huge size reduction
    progressive=True
)

# ============================================================================
# LESSON AVATAR - POINTING IMAGES
# ============================================================================
pointing_source_dir = "daily-lesson-marketing/public/lessons/images/top-bottom"
pointing_dest_dir = "public/images/kelly"

# Point Up
point_up_size = optimize_image(
    f"{pointing_source_dir}/top-choice.jpeg",
    f"{pointing_dest_dir}/kelly-point-up.jpeg",
    max_width=800,   # Lesson avatar doesn't need full resolution
    quality=88,
    progressive=True
)

# Point Down
point_down_size = optimize_image(
    f"{pointing_source_dir}/bottom-choice.jpeg",
    f"{pointing_dest_dir}/kelly-point-down.jpeg",
    max_width=800,
    quality=88,
    progressive=True
)

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("✅ OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"\n📦 Assets Ready for Production:")
print(f"   • Homepage Hero: {hero_size / 1024:.1f} KB")
print(f"   • Point Up Avatar: {point_up_size / 1024:.1f} KB")
print(f"   • Point Down Avatar: {point_down_size / 1024:.1f} KB")
print(f"\n🚀 Total payload: {(hero_size + point_up_size + point_down_size) / 1024:.1f} KB")
print("\n✓ All images optimized for web delivery")
print("✓ Visual quality maintained (JPEG Q88)")
print("✓ Progressive encoding enabled (faster perceived load)")
print("\n" + "=" * 70)




