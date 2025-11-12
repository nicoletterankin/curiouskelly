#!/usr/bin/env python3
"""
Example usage of Kelly Asset Pack Generator

This script demonstrates how to use the toolkit programmatically.
"""
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from kelly_pack import io_utils, crop_scale, matting, alpha_tools, composite, diffuse
import numpy as np


def example_basic():
    """Basic example: load, matte, save."""
    print("Example 1: Basic Matting")
    print("-" * 40)
    
    # Load image
    img = io_utils.load_image("kelly2-directors-chair.jpeg", mode="RGB")
    if img is None:
        print("Error: Could not find kelly2-directors-chair.jpeg")
        return
    
    # Prepare 16:9 hero
    hero_rgb = crop_scale.prepare_16_9_hero(img, 7680, 4320)
    print(f"Hero prepared: {hero_rgb.shape}")
    
    # Generate alpha (heuristic)
    alpha = matting.heuristic_matting(hero_rgb)
    print(f"Alpha generated: {alpha.shape}")
    
    # Create soft and tight variants
    alpha_soft = alpha_tools.generate_soft_alpha(alpha, blur_radius=2.0, bias=0.05)
    alpha_tight = alpha_tools.generate_tight_alpha(alpha, blur_radius=1.0, bias=-0.03, erode_size=1)
    
    # Save
    io_utils.ensure_dir("./example_output")
    io_utils.save_image((alpha_soft * 255).astype('uint8'), 
                       "./example_output/alpha_soft.png", mode="L")
    io_utils.save_image((alpha_tight * 255).astype('uint8'), 
                       "./example_output/alpha_tight.png", mode="L")
    
    print("✓ Saved alpha_soft.png and alpha_tight.png")


def example_dark_hero():
    """Example: Create dark-mode composite."""
    print("\nExample 2: Dark Hero Composite")
    print("-" * 40)
    
    # Create sample RGB and alpha
    rgb = np.random.randint(0, 255, (4320, 7680, 3), dtype=np.uint8)
    alpha = np.random.rand(4320, 7680).astype(np.float32)
    
    # Composite over dark gradient
    dark_hero = composite.create_dark_hero(rgb, alpha, "#22262A", "#080808")
    print(f"Dark hero created: {dark_hero.shape}")
    
    io_utils.save_image(dark_hero, "./example_output/dark_hero_demo.png", mode="RGB")
    print("✓ Saved dark_hero_demo.png")


def example_square_sprite():
    """Example: Square sprite with padding."""
    print("\nExample 3: Square Sprite")
    print("-" * 40)
    
    # Create sample portrait
    portrait = np.random.randint(0, 255, (1200, 900, 3), dtype=np.uint8)
    alpha = np.random.rand(1200, 900).astype(np.float32)
    
    # Make alpha more interesting (circular subject)
    y, x = np.ogrid[:1200, :900]
    center_y, center_x = 600, 450
    radius = 400
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    alpha = np.clip(1 - distance / radius, 0, 1).astype(np.float32)
    
    # Create square sprite
    sprite_rgb, sprite_alpha = crop_scale.prepare_square_sprite(
        portrait, alpha, canvas_size=8192, padding_frac=0.10
    )
    print(f"Sprite created: RGB={sprite_rgb.shape}, Alpha={sprite_alpha.shape}")
    
    # Save as RGBA
    sprite_rgba = np.dstack([sprite_rgb, (sprite_alpha * 255).astype('uint8')])
    io_utils.save_image(sprite_rgba, "./example_output/sprite_demo.png", mode="RGBA")
    print("✓ Saved sprite_demo.png")


def example_diffuse_neutral():
    """Example: Diffuse neutralization."""
    print("\nExample 4: Diffuse Neutralization")
    print("-" * 40)
    
    # Create sample image with color cast
    img = np.random.randint(50, 200, (1000, 1000, 3), dtype=np.uint8)
    img[:, :, 0] += 30  # Add red cast
    
    # Neutralize
    neutral = diffuse.neutralize_diffuse(img, contrast_flatten=0.15)
    print(f"Neutralized: {neutral.shape}")
    
    io_utils.save_image(img, "./example_output/before_neutral.png", mode="RGB")
    io_utils.save_image(neutral, "./example_output/after_neutral.png", mode="RGB")
    print("✓ Saved before_neutral.png and after_neutral.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Kelly Asset Pack Generator - Examples")
    print("=" * 60)
    print()
    
    # Run examples
    example_basic()
    example_dark_hero()
    example_square_sprite()
    example_diffuse_neutral()
    
    print("\n" + "=" * 60)
    print("Examples complete! Check ./example_output/")
    print("=" * 60)


