#!/usr/bin/env python3
"""
KELLY LESSON GENERATOR
======================
Generates all assets for a single lesson using Vertex AI Imagen.
Uses canonical Kelly references for consistency.

Usage:
    python generate_lesson.py 001
    python generate_lesson.py 001 --dry-run
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Paths
PROJECT_ROOT = Path(r"C:\Users\user\UI-TARS-desktop")
KELLY_ROOT = PROJECT_ROOT / "public" / "kelly"
REF_PATH = Path(r"C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference")

# Canonical references (in order of priority)
CANONICAL_REFS = [
    REF_PATH / "close up of face.jpeg",
    REF_PATH / "head and shoulders without chair.png",
    REF_PATH / "neutral face with hair.png",
]

# Base Kelly prompt
BASE_PROMPT = """A photorealistic 3D rendered portrait of a young woman in her late twenties,
long wavy brown hair with subtle blonde highlights flowing past her shoulders,
warm brown eyes, fair-medium skin tone with warm undertones,
wearing a blue-teal ribbed crew-neck sweater,
soft professional studio lighting, clean white background,
Character Creator or iClone style, film-grade photorealistic quality,
natural beauty, approachable and friendly demeanor"""

NEGATIVE_PROMPT = """green eyes, hazel eyes, blue eyes, short hair, straight hair, blonde hair,
different sweater, different outfit, cartoon, anime, illustration, painting,
low quality, blurry, distorted, extra limbs, deformed face"""

# Asset specifications
PHASE_ASSETS = {
    "hook": {
        "prompt_suffix": ", warm welcoming smile, seated in director's chair, inviting pose",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "q1": {
        "prompt_suffix": ", curious expression, head tilted slightly, asking a question",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "q2": {
        "prompt_suffix": ", thoughtful expression, hand near chin, contemplating",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "q3": {
        "prompt_suffix": ", engaged expression, leaning forward slightly, interested",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "wisdom": {
        "prompt_suffix": ", serene wise smile, calm composed demeanor, profound moment",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
}

LESSON_ASSETS = {
    "hero": {
        "prompt_suffix": ", heroic confident pose, main presentation stance",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "bg": {
        "prompt_suffix": ", standing in an educational environment, classroom or studio setting",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
    "guide-point": {
        "prompt_suffix": ", pointing gesture, guiding and explaining, vertical portrait",
        "width": 768,
        "height": 1344,
        "aspect": "9:16",
    },
    "reaction": {
        "prompt_suffix": ", excited happy expression, celebrating success, positive reaction",
        "width": 1024,
        "height": 1024,
        "aspect": "1:1",
    },
    "prop": {
        "prompt_suffix": ", holding or presenting something, teaching pose with prop",
        "width": 1344,
        "height": 768,
        "aspect": "16:9",
    },
}


def check_vertex_ai():
    """Check if Vertex AI is available"""
    try:
        from vertexai import init as vertex_init
        from vertexai.preview.vision_models import ImageGenerationModel
        return True
    except ImportError:
        return False


def generate_image(
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    aspect_ratio: str,
    references: list,
    dry_run: bool = False
) -> bool:
    """Generate a single image using Vertex AI"""
    
    if dry_run:
        print(f"  [DRY RUN] Would generate: {output_path.name}")
        print(f"            Prompt: {prompt[:80]}...")
        return True
    
    try:
        from vertexai import init as vertex_init
        from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
        from PIL import Image
        import io
        
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("VERTEX_LOCATION", "us-central1")
        
        if not project:
            print("  [ERROR] GOOGLE_CLOUD_PROJECT not set")
            return False
        
        vertex_init(project=project, location=location)
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Load reference images
        ref_images = []
        for ref_path in references:
            if ref_path.exists():
                try:
                    ref_images.append(VertexImage.load_from_file(str(ref_path)))
                except Exception:
                    pass
        
        params = {
            "prompt": prompt,
            "number_of_images": 1,
            "aspect_ratio": aspect_ratio,
            "negative_prompt": NEGATIVE_PROMPT,
        }
        
        # Try with reference images first
        try:
            if ref_images:
                params["reference_images"] = ref_images
            images = model.generate_images(**params)
        except TypeError:
            # Reference images not supported in this SDK version
            if "reference_images" in params:
                del params["reference_images"]
            images = model.generate_images(**params)
        
        if not images:
            print(f"  [ERROR] No images returned for {output_path.name}")
            return False
        
        # Extract and save image
        img0 = images[0]
        pil_img = None
        
        if hasattr(img0, "image_bytes") and img0.image_bytes:
            pil_img = Image.open(io.BytesIO(img0.image_bytes)).convert("RGBA")
        elif hasattr(img0, "_pil_image"):
            pil_img = img0._pil_image.convert("RGBA")
        
        if pil_img is None:
            print(f"  [ERROR] Could not extract image for {output_path.name}")
            return False
        
        # Resize to exact dimensions
        if pil_img.size != (width, height):
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"  [OK] Generated: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {output_path.name}: {e}")
        return False


def generate_lesson(lesson_num: str, dry_run: bool = False) -> Dict[str, Any]:
    """Generate all assets for a lesson"""
    
    print(f"\n{'='*60}")
    print(f"GENERATING LESSON {lesson_num}")
    print(f"{'='*60}\n")
    
    results = {
        "lesson": lesson_num,
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "phases": {},
        "lessons": {},
        "success_count": 0,
        "error_count": 0,
    }
    
    # Phase assets
    print("Phase Assets:")
    phase_dir = KELLY_ROOT / "phases" / lesson_num
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    for asset_name, spec in PHASE_ASSETS.items():
        output_path = phase_dir / f"{asset_name}.png"
        prompt = BASE_PROMPT + spec["prompt_suffix"]
        
        success = generate_image(
            prompt=prompt,
            output_path=output_path,
            width=spec["width"],
            height=spec["height"],
            aspect_ratio=spec["aspect"],
            references=CANONICAL_REFS,
            dry_run=dry_run,
        )
        
        results["phases"][asset_name] = {
            "path": str(output_path),
            "success": success,
        }
        if success:
            results["success_count"] += 1
        else:
            results["error_count"] += 1
    
    # Lesson assets
    print("\nLesson Assets:")
    lesson_dir = KELLY_ROOT / "lessons" / lesson_num
    lesson_dir.mkdir(parents=True, exist_ok=True)
    lesson_int = int(lesson_num)
    
    for asset_name, spec in LESSON_ASSETS.items():
        output_path = lesson_dir / f"lesson-{lesson_int}-{asset_name}.png"
        prompt = BASE_PROMPT + spec["prompt_suffix"]
        
        success = generate_image(
            prompt=prompt,
            output_path=output_path,
            width=spec["width"],
            height=spec["height"],
            aspect_ratio=spec["aspect"],
            references=CANONICAL_REFS,
            dry_run=dry_run,
        )
        
        results["lessons"][asset_name] = {
            "path": str(output_path),
            "success": success,
        }
        if success:
            results["success_count"] += 1
        else:
            results["error_count"] += 1
    
    # Thumbnail
    print("\nThumbnail:")
    thumb_dir = KELLY_ROOT / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / f"lesson-{lesson_num}.png"
    
    success = generate_image(
        prompt=BASE_PROMPT + ", friendly portrait, thumbnail style, centered face",
        output_path=thumb_path,
        width=512,
        height=512,
        aspect_ratio="1:1",
        references=CANONICAL_REFS,
        dry_run=dry_run,
    )
    
    results["thumbnail"] = {
        "path": str(thumb_path),
        "success": success,
    }
    if success:
        results["success_count"] += 1
    else:
        results["error_count"] += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {results['success_count']} succeeded, {results['error_count']} failed")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Kelly lesson assets")
    parser.add_argument("lesson", help="Lesson number (e.g., 001)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    args = parser.parse_args()
    
    # Validate lesson number
    try:
        lesson_num = f"{int(args.lesson):03d}"
    except ValueError:
        print(f"Invalid lesson number: {args.lesson}")
        return 1
    
    if int(lesson_num) < 1 or int(lesson_num) > 365:
        print(f"Lesson number must be 1-365")
        return 1
    
    # Check Vertex AI
    if not args.dry_run and not check_vertex_ai():
        print("ERROR: Vertex AI not available. Install with:")
        print("  pip install google-cloud-aiplatform")
        return 1
    
    # Generate
    results = generate_lesson(lesson_num, dry_run=args.dry_run)
    
    # Save manifest
    manifest_path = KELLY_ROOT / "lessons" / lesson_num / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Manifest saved: {manifest_path}")
    
    return 0 if results["error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

