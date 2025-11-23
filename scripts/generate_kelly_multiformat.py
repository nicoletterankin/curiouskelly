#!/usr/bin/env python3
"""
Kelly Multi-Format Generator
Generates Kelly images in multiple aspect ratios (16:9, 1:1, 3:4) from a single preset
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add tools directory to path
tools_dir = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from kelly_asset_generator import load_preset, generate_with_backend, write_manifest, GenerationResult, Preset
from PIL import Image


def adjust_aspect_ratio(img: Image.Image, target_ratio: str) -> Image.Image:
    """
    Adjust image to target aspect ratio by intelligent cropping/padding
    
    Args:
        img: Source PIL Image
        target_ratio: Target ratio as string ("16:9", "1:1", "3:4")
    
    Returns:
        Adjusted PIL Image
    """
    ratio_map = {
        "16:9": (16, 9),
        "1:1": (1, 1),
        "3:4": (3, 4),
        "4:3": (4, 3),
        "9:16": (9, 16)
    }
    
    if target_ratio not in ratio_map:
        return img
    
    target_w, target_h = ratio_map[target_ratio]
    target_aspect = target_w / target_h
    
    src_w, src_h = img.size
    src_aspect = src_w / src_h
    
    # If already correct ratio, return as-is
    if abs(src_aspect - target_aspect) < 0.01:
        return img
    
    # Determine if we need to crop width or height
    if src_aspect > target_aspect:
        # Image is wider than target - crop width
        new_width = int(src_h * target_aspect)
        left = (src_w - new_width) // 2
        img_cropped = img.crop((left, 0, left + new_width, src_h))
    else:
        # Image is taller than target - crop height
        new_height = int(src_w / target_aspect)
        top = (src_h - new_height) // 2
        img_cropped = img.crop((0, top, src_w, top + new_height))
    
    return img_cropped


def generate_multiformat(preset_path: Path, outdir: Path, formats: List[str] = None) -> Dict[str, Any]:
    """
    Generate Kelly image in multiple formats from a single preset
    
    Args:
        preset_path: Path to YAML preset file
        outdir: Output directory for renders and manifests
        formats: List of aspect ratios to generate (default: ["16:9", "1:1", "3:4"])
    
    Returns:
        Dictionary with generation results
    """
    if formats is None:
        formats = ["16:9", "1:1", "3:4"]
    
    print(f"\n{'='*60}")
    print(f"Multi-Format Generation: {preset_path.name}")
    print(f"{'='*60}\n")
    
    # Load preset
    preset = load_preset(preset_path)
    
    # Ensure output directories exist
    (outdir / "renders").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for aspect_ratio in formats:
        print(f"\n--- Generating {aspect_ratio} format ---")
        
        # Temporarily modify preset for this aspect ratio
        original_output = preset.output.copy()
        
        # Calculate dimensions for aspect ratio
        ratio_map = {
            "16:9": (1920, 1080),
            "1:1": (2048, 2048),
            "3:4": (1536, 2048),
        }
        
        if aspect_ratio in ratio_map:
            preset.output["width"], preset.output["height"] = ratio_map[aspect_ratio]
        
        # Generate base image
        result = generate_with_backend(preset, outdir)
        
        # Load the generated image and adjust if needed
        img = Image.open(result.image_path)
        img_adjusted = adjust_aspect_ratio(img, aspect_ratio)
        
        # Save with aspect ratio suffix
        base_name = result.image_path.stem
        # Remove existing version suffix if present
        if "_v" in base_name:
            base_name = base_name.rsplit("_v", 1)[0]
        
        # Add aspect ratio suffix
        ratio_suffix = aspect_ratio.replace(":", "x")
        new_name = f"{base_name}_{ratio_suffix}.png"
        new_path = result.image_path.parent / new_name
        
        # Save adjusted image
        img_adjusted.save(new_path)
        print(f"  [OK] Saved: {new_path.name}")
        
        # If we generated a temp file with wrong name, remove it
        if new_path != result.image_path and result.image_path.exists():
            try:
                result.image_path.unlink()
            except:
                pass
        
        # Update result
        result.image_path = new_path
        result.width = img_adjusted.width
        result.height = img_adjusted.height
        
        # Write manifest for this format
        manifest_base = new_path.stem
        manifest_path = outdir / "manifests" / f"{manifest_base}.json"
        
        manifest = {
            "schema": "kelly.asset.manifest/v1",
            "created_at": result.image_path.stat().st_mtime,
            "preset": {
                "asset_type": preset.asset_type,
                "view": preset.view,
                "lighting": preset.lighting,
                "aspect_ratio": aspect_ratio
            },
            "result": {
                "path": str(result.image_path.as_posix()),
                "width": result.width,
                "height": result.height,
                "aspect_ratio": aspect_ratio,
                "seed": result.seed,
            },
            "lineage": {
                "prompt": preset.prompt[:500],  # Truncate for manifest
                "negative_prompt": preset.negative_prompt,
            }
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        results[aspect_ratio] = {
            "image": str(result.image_path),
            "manifest": str(manifest_path),
            "width": result.width,
            "height": result.height
        }
        
        # Restore original output settings
        preset.output = original_output
    
    print(f"\n[SUCCESS] Generated {len(results)} formats: {', '.join(results.keys())}")
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Kelly images in multiple aspect ratios")
    parser.add_argument("preset", type=Path, help="Path to YAML preset file")
    parser.add_argument("--outdir", type=Path, default=Path("projects/Kelly/assets/age_progressive"))
    parser.add_argument("--formats", nargs="+", default=["16:9", "1:1", "3:4"],
                        help="Aspect ratios to generate (default: 16:9 1:1 3:4)")
    
    args = parser.parse_args()
    
    if not args.preset.exists():
        print(f"Error: Preset file not found: {args.preset}", file=sys.stderr)
        sys.exit(1)
    
    results = generate_multiformat(args.preset, args.outdir, args.formats)
    print(f"\n[SUMMARY]")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

