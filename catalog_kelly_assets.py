#!/usr/bin/env python3
"""
Catalog all Kelly avatar assets for Curious Kelly project
Creates a comprehensive JSON manifest for v0 import
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

# Age mappings based on codebase
AGE_MAPPINGS = {
    "kid": {"approx_age": 7, "label": "Kid Kelly"},
    "teen": {"approx_age": 12, "label": "Tween Kelly"},
    "adult": {"approx_age": 22, "label": "Young Adult Kelly"},
    "mature": {"approx_age": 32, "label": "Adult Kelly"},
    "elder": {"approx_age": 50, "label": "Wise Kelly"},
    "super_elder": {"approx_age": 75, "label": "Elder Kelly"}
}

# Archetypes
ARCHETYPES = [
    "scientist", "explorer", "rebel", "architect", "diplomat",
    "empath", "macgyver", "mystic", "provider", "storyteller",
    "strategist", "survivor", "default"
]

def get_image_dimensions(file_path: Path) -> Optional[Dict[str, int]]:
    """Get image dimensions if possible"""
    try:
        with Image.open(file_path) as img:
            return {"width": img.width, "height": img.height}
    except Exception:
        return None

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes"""
    try:
        return file_path.stat().st_size
    except Exception:
        return 0

def determine_image_type(file_path: Path, file_name: str) -> str:
    """Determine image type based on path and filename"""
    path_str = str(file_path).lower()
    name_lower = file_name.lower()
    
    if "head" in name_lower or "heads" in path_str:
        return "head"
    elif "pose" in name_lower or "poses" in path_str:
        return "pose"
    elif "body" in name_lower or "full" in name_lower:
        return "body"
    elif "hero" in name_lower or "og-image" in name_lower:
        return "hero"
    elif "chair" in name_lower:
        return "chair"
    elif "phase" in path_str or "phases" in path_str:
        return "phase"
    elif "choice" in name_lower:
        return "choice"
    elif "infographic" in name_lower:
        return "infographic"
    elif "social" in path_str:
        return "social"
    else:
        return "other"

def determine_age_from_path(file_path: Path) -> Optional[str]:
    """Determine age category from file path"""
    path_str = str(file_path).lower()
    path_parts = path_str.replace("\\", "/").split("/")
    
    # Check each part of the path
    for part in path_parts:
        for age_key in AGE_MAPPINGS.keys():
            if age_key in part:
                return age_key
    
    # Check for age in filename patterns
    file_name = file_path.name.lower()
    if "kid" in file_name or "kid" in path_str:
        return "kid"
    elif "teen" in file_name or "teen" in path_str:
        return "teen"
    elif "super_elder" in path_str or ("super" in path_str and "elder" in path_str):
        return "super_elder"
    elif "elder" in file_name or "elder" in path_str:
        return "elder"
    elif "mature" in file_name or "mature" in path_str:
        return "mature"
    elif "adult" in file_name or "adult" in path_str:
        return "adult"
    
    # Default for public/kelly/heads (these are adult)
    if "public/kelly/heads" in path_str:
        return "adult"
    
    return None

def determine_archetype_from_path(file_path: Path) -> Optional[str]:
    """Determine archetype from file path"""
    file_name = file_path.name.lower()
    
    for archetype in ARCHETYPES:
        if archetype in file_name:
            return archetype
    
    return None

def catalog_kelly_images(root_dir: Path) -> List[Dict]:
    """Catalog all Kelly images"""
    assets = []
    
    # Directories to skip
    skip_dirs = {
        "node_modules", ".git", "__pycache__", ".next", "dist",
        ".pnpm", ".cache", "build", "out", ".vercel"
    }
    
    # Specific directories to search
    search_dirs = [
        root_dir / "public" / "kelly",
        root_dir / "generated-images" / "kelly-archetypes-head-only",
        root_dir / "public" / "images" / "kelly",
        root_dir / "generated-poses-presenter",
        root_dir / "generated-poses-production",
        root_dir / "generated-poses-pro",
        root_dir / "generated-poses-pulid",
        root_dir / "public" / "images",  # For hero images
    ]
    
    # Add all age subdirectories explicitly
    age_base = root_dir / "generated-images" / "kelly-archetypes-head-only" / "age"
    if age_base.exists():
        for age_dir in age_base.iterdir():
            if age_dir.is_dir():
                search_dirs.append(age_dir)
    
    # Also search root for kelly files
    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    
    def should_skip(path: Path) -> bool:
        """Check if path should be skipped"""
        path_str = str(path).lower()
        return any(skip in path_str for skip in skip_dirs)
    
    # Search specific directories
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        try:
            for file_path in search_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in extensions:
                    continue
                if should_skip(file_path):
                    continue
                # For generated-images, include all files (they're all Kelly)
                # For other dirs, require "kelly" in name
                if "generated-images" not in str(search_dir) and "kelly" not in file_path.name.lower():
                    continue
                
                file_name = file_path.name
                relative_path = str(file_path.relative_to(root_dir)).replace("\\", "/")
            
            # Determine properties
            image_type = determine_image_type(file_path, file_name)
            age_category = determine_age_from_path(file_path)
            archetype = determine_archetype_from_path(file_path)
            dimensions = get_image_dimensions(file_path)
            file_size = get_file_size(file_path)
            
            asset = {
                "file_path": relative_path,
                "file_name": file_name,
                "image_type": image_type,
                "file_size_bytes": file_size,
                "format": file_path.suffix.lower().replace(".", "")
            }
            
            if age_category:
                asset["age_category"] = age_category
                asset["approx_age"] = AGE_MAPPINGS[age_category]["approx_age"]
                asset["age_label"] = AGE_MAPPINGS[age_category]["label"]
            
            if archetype:
                asset["archetype"] = archetype
            
            if dimensions:
                asset["width"] = dimensions["width"]
                asset["height"] = dimensions["height"]
                asset["resolution"] = f"{dimensions['width']}×{dimensions['height']}"
            
                assets.append(asset)
        except (PermissionError, FileNotFoundError, OSError) as e:
            # Skip directories we can't access
            continue
    
    # Also search root public directory
    public_kelly = root_dir / "public" / "kelly"
    if public_kelly.exists():
        try:
            for file_path in public_kelly.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in extensions:
                    continue
                if should_skip(file_path):
                    continue
                
                file_name = file_path.name
                relative_path = str(file_path.relative_to(root_dir)).replace("\\", "/")
                
                # Determine properties
                image_type = determine_image_type(file_path, file_name)
                age_category = determine_age_from_path(file_path)
                archetype = determine_archetype_from_path(file_path)
                dimensions = get_image_dimensions(file_path)
                file_size = get_file_size(file_path)
                
                asset = {
                    "file_path": relative_path,
                    "file_name": file_name,
                    "image_type": image_type,
                    "file_size_bytes": file_size,
                    "format": file_path.suffix.lower().replace(".", "")
                }
                
                if age_category:
                    asset["age_category"] = age_category
                    asset["approx_age"] = AGE_MAPPINGS[age_category]["approx_age"]
                    asset["age_label"] = AGE_MAPPINGS[age_category]["label"]
                
                if archetype:
                    asset["archetype"] = archetype
                
                if dimensions:
                    asset["width"] = dimensions["width"]
                    asset["height"] = dimensions["height"]
                    asset["resolution"] = f"{dimensions['width']}×{dimensions['height']}"
                
                assets.append(asset)
        except (PermissionError, FileNotFoundError, OSError):
            pass
    
    return assets

def create_manifest(assets: List[Dict], output_path: Path):
    """Create organized manifest JSON"""
    
    # Organize by category
    manifest = {
        "metadata": {
            "generated_at": str(Path.cwd()),
            "total_assets": len(assets),
            "age_mappings": AGE_MAPPINGS
        },
        "assets_by_type": {},
        "assets_by_age": {},
        "assets_by_archetype": {},
        "all_assets": assets
    }
    
    # Group by type
    for asset in assets:
        img_type = asset.get("image_type", "other")
        if img_type not in manifest["assets_by_type"]:
            manifest["assets_by_type"][img_type] = []
        manifest["assets_by_type"][img_type].append(asset)
    
    # Group by age
    for asset in assets:
        age = asset.get("age_category", "unknown")
        if age not in manifest["assets_by_age"]:
            manifest["assets_by_age"][age] = []
        manifest["assets_by_age"][age].append(asset)
    
    # Group by archetype
    for asset in assets:
        archetype = asset.get("archetype", "default")
        if archetype not in manifest["assets_by_archetype"]:
            manifest["assets_by_archetype"][archetype] = []
        manifest["assets_by_archetype"][archetype].append(asset)
    
    # Write manifest
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created manifest with {len(assets)} assets")
    print(f"📁 Saved to: {output_path}")
    
    # Print summary
    print("\n📊 Summary by Type:")
    for img_type, items in manifest["assets_by_type"].items():
        print(f"  {img_type}: {len(items)}")
    
    print("\n📊 Summary by Age:")
    for age, items in manifest["assets_by_age"].items():
        print(f"  {age}: {len(items)}")
    
    return manifest

if __name__ == "__main__":
    root_dir = Path(__file__).parent
    output_path = root_dir / "kelly_assets_manifest.json"
    
    print("🔍 Scanning for Kelly images...")
    assets = catalog_kelly_images(root_dir)
    
    print(f"\n📦 Found {len(assets)} Kelly image files")
    print("📝 Creating manifest...")
    
    manifest = create_manifest(assets, output_path)
    
    print("\n✅ Complete!")

