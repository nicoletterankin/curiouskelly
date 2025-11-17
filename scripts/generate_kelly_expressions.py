#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Kelly Expression Images in Director's Chair
Creates 4-5 Kelly images with different expressions for lesson phases
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add tools directory to path for imports
tools_dir = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

import io
try:
    from vertexai import init as vertex_init
    from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
    from PIL import Image as PILImage
    VERTEX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vertex AI SDK not available: {e}")
    VERTEX_AVAILABLE = False

# Configuration
PROJECT_ID = "gen-lang-client-0005524332"
LOCATION = "us-central1"
MODEL = "imagen-3.0-generate-002"
ASPECT_RATIO = "16:9"

# Reference images
# Try workspace-relative path first, then absolute path
workspace_ref_dir = Path(__file__).parent.parent / "iLearnStudio" / "projects" / "Kelly" / "Ref"
absolute_ref_dir = Path("C:/iLearnStudio/projects/Kelly/Ref")

# Use whichever exists (prefer absolute path if both exist, as it's more reliable)
if absolute_ref_dir.exists():
    REF_DIR = absolute_ref_dir
elif workspace_ref_dir.exists():
    REF_DIR = workspace_ref_dir
else:
    REF_DIR = absolute_ref_dir  # Default to absolute path

REF_IMAGES = [
    REF_DIR / "kelly_front.png",
    REF_DIR / "kelly_profile.png",
    REF_DIR / "kelly_three_quarter.png"
]

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "lessons" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expression definitions
EXPRESSIONS = {
    "curious": {
        "file": "kelly-directors-chair-curious.png",
        "prompt": """Professional photograph of Kelly seated properly IN a classic director's chair with dark wooden frame and black canvas seat and backrest. She is wearing a light blue ribbed long-sleeved crew-neck sweater, sitting upright with her body properly positioned on the chair seat, shoulders visible, head tilted slightly to one side with a curious inquisitive expression, eyebrows slightly raised, eyes wide with wonder and interest, gentle smile, looking directly at camera with engaging friendly expression. Upper body framing showing Kelly properly seated in the chair with chair frame visible around her. Clean white studio background with soft even lighting, bright natural light streaming from unseen window on the left creating distinct geometric diagonal shadows on the back wall and floor in arrow-like patterns. Photorealistic, high detail, 8K quality, 16:9 aspect ratio, horizontal composition. Kelly is sitting ON the chair seat, not in front of it.""",
        "negative_prompt": "sad, angry, frowning, closed eyes, looking away, dark background, cluttered background, chair floating in front, chair blocking Kelly, Kelly floating above chair, Kelly not sitting on chair"
    },
    "explaining": {
        "file": "kelly-directors-chair-explaining.png",
        "prompt": """Professional photograph of Kelly seated properly IN a classic director's chair with dark wooden frame and black canvas seat and backrest. She is wearing a light blue ribbed long-sleeved crew-neck sweater, sitting upright with her body properly positioned on the chair seat, shoulders visible, right arm extended forward with index finger pointing slightly upward and outward as if explaining or teaching, left hand resting on chair arm, head held confidently, warm engaging smile, eyes looking directly at camera with friendly teaching expression. Upper body framing showing Kelly properly seated in the chair with chair frame visible around her, pointing gesture. Clean white studio background with soft even lighting, bright natural light streaming from unseen window on the left creating distinct geometric diagonal shadows on the back wall and floor in arrow-like patterns. Photorealistic, high detail, 8K quality, 16:9 aspect ratio, horizontal composition. Kelly is sitting ON the chair seat, not in front of it.""",
        "negative_prompt": "sad, angry, confused, looking away, dark background, cluttered background, hands hidden, chair floating in front, chair blocking Kelly, Kelly floating above chair, Kelly not sitting on chair"
    },
    "celebrating": {
        "file": "kelly-directors-chair-celebrating.png",
        "prompt": """Professional photograph of Kelly seated properly IN a classic director's chair with dark wooden frame and black canvas seat and backrest. She is wearing a light blue ribbed long-sleeved crew-neck sweater, sitting upright with her body properly positioned on the chair seat, shoulders visible, both hands raised slightly with palms open in a celebratory gesture, bright genuine smile showing teeth, eyes crinkled with joy, looking directly at camera with enthusiastic happy expression. Upper body framing showing Kelly properly seated in the chair with chair frame visible around her, celebratory gesture. Clean white studio background with soft even lighting, bright natural light streaming from unseen window on the left creating distinct geometric diagonal shadows on the back wall and floor in arrow-like patterns. Photorealistic, high detail, 8K quality, 16:9 aspect ratio, horizontal composition. Kelly is sitting ON the chair seat, not in front of it.""",
        "negative_prompt": "sad, angry, neutral expression, frowning, looking away, dark background, cluttered background, chair floating in front, chair blocking Kelly, Kelly floating above chair, Kelly not sitting on chair"
    },
    "listening": {
        "file": "kelly-directors-chair-listening.png",
        "prompt": """Professional photograph of Kelly seated properly IN a classic director's chair with dark wooden frame and black canvas seat and backrest. She is wearing a light blue ribbed long-sleeved crew-neck sweater, sitting upright with her body properly positioned on the chair seat, shoulders visible, head tilted slightly forward in an attentive listening pose, hands resting gently on chair arms, soft gentle smile, eyes focused and engaged, looking directly at camera with warm attentive expression showing she is actively listening and engaged. Upper body framing showing Kelly properly seated in the chair with chair frame visible around her. Clean white studio background with soft even lighting, bright natural light streaming from unseen window on the left creating distinct geometric diagonal shadows on the back wall and floor in arrow-like patterns. Photorealistic, high detail, 8K quality, 16:9 aspect ratio, horizontal composition. Kelly is sitting ON the chair seat, not in front of it.""",
        "negative_prompt": "distracted, looking away, bored, sad, angry, dark background, cluttered background, chair floating in front, chair blocking Kelly, Kelly floating above chair, Kelly not sitting on chair"
    },
    "wisdom": {
        "file": "kelly-directors-chair-wisdom.png",
        "prompt": """Professional photograph of Kelly seated properly IN a classic director's chair with dark wooden frame and black canvas seat and backrest. She is wearing a light blue ribbed long-sleeved crew-neck sweater, sitting upright with her body properly positioned on the chair seat, shoulders visible, hands resting gently in her lap, head held with quiet confidence, gentle reflective smile, eyes looking directly at camera with warm wise expression showing thoughtfulness and understanding. Upper body framing showing Kelly properly seated in the chair with chair frame visible around her. Clean white studio background with soft even lighting, bright natural light streaming from unseen window on the left creating distinct geometric diagonal shadows on the back wall and floor in arrow-like patterns. Photorealistic, high detail, 8K quality, 16:9 aspect ratio, horizontal composition. Kelly is sitting ON the chair seat, not in front of it.""",
        "negative_prompt": "sad, angry, confused, looking away, dark background, cluttered background, overly excited, chair floating in front, chair blocking Kelly, Kelly floating above chair, Kelly not sitting on chair"
    }
}


def get_reference_images():
    """Get list of existing reference images - scans entire Ref directory and subdirectories"""
    refs = []
    seen_paths = set()
    
    # Check expected references first
    for ref_path in REF_IMAGES:
        if ref_path.exists():
            refs.append(ref_path)
            seen_paths.add(str(ref_path.resolve()))
            print(f"  [OK] Found reference: {ref_path.name}")
        else:
            print(f"  [WARN] Missing reference: {ref_path.name}")
    
    # Scan recursively for ALL reference images (including subdirectories like "Best Character Reference")
    if REF_DIR.exists():
        # Scan for PNG, JPG, JPEG in all subdirectories
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            for img_file in REF_DIR.rglob(ext):
                img_path_str = str(img_file.resolve())
                if img_path_str not in seen_paths:
                    refs.append(img_file)
                    seen_paths.add(img_path_str)
                    rel_path = img_file.relative_to(REF_DIR)
                    print(f"  [OK] Found reference: {rel_path}")
    
    # Sort by priority: chair references first, then face references, then others
    def priority_key(path):
        name_lower = path.name.lower()
        if "chair" in name_lower:
            return (0, path.name)  # Highest priority
        elif any(keyword in name_lower for keyword in ["face", "close", "profile", "head"]):
            return (1, path.name)  # Second priority
        else:
            return (2, path.name)  # Lower priority
    
    refs.sort(key=priority_key)
    
    return refs


def generate_expression_image(expression_name, expression_data):
    """Generate a single expression image - enhanced prompts based on reference images"""
    if not VERTEX_AVAILABLE:
        print(f"  [ERROR] Vertex AI SDK not available. Cannot generate {expression_name}.")
        return False
    
    output_path = OUTPUT_DIR / expression_data["file"]
    
    print(f"\n{'='*60}")
    print(f"Generating: {expression_name}")
    print(f"Output: {output_path.name}")
    print(f"{'='*60}")
    
    # Get reference images - prioritize "Best Character Reference" folder
    ref_images = get_reference_images()
    
    # Filter to use only the BEST reference images (from "Best Character Reference" folder)
    best_refs = [r for r in ref_images if "Best Character Reference" in str(r)]
    if best_refs:
        ref_images = best_refs[:8]  # Use up to 8 best references
        print(f"  ✓ Found {len(ref_images)} reference images from 'Best Character Reference' folder")
        print(f"     Using reference images to enhance prompt descriptions")
        print(f"     Note: Vertex AI SDK doesn't support reference_images parameter")
        print(f"     Instead: Enhancing prompts with detailed character descriptions from references")
    elif ref_images:
        ref_images = ref_images[:5]
        print(f"  ✓ Found {len(ref_images)} reference images")
    else:
        print("  [WARN] No reference images found. Character consistency may be reduced.")
    
    # Enhance prompt with character details from reference images
    # Since we can't use reference images directly, we'll describe what's in them
    enhanced_prompt = expression_data["prompt"]
    
    if ref_images:
        # Add character consistency details based on reference images found
        character_details = []
        
        # Check for specific reference types
        has_chair_ref = any("chair" in str(r).lower() for r in ref_images)
        has_face_ref = any("face" in str(r).lower() or "close" in str(r).lower() for r in ref_images)
        has_profile_ref = any("profile" in str(r).lower() for r in ref_images)
        has_hair_ref = any("hair" in str(r).lower() for r in ref_images)
        
        if has_chair_ref:
            character_details.append("Kelly is seated properly IN the director's chair with her body positioned on the chair seat, not in front of it")
        
        if has_face_ref:
            character_details.append("Kelly has a consistent face: oval face with soft rounded contours, warm brown eyes, fair skin with natural glow")
        
        if has_hair_ref:
            character_details.append("Kelly has long wavy brown hair with subtle caramel highlights extending well past her shoulders")
        
        if character_details:
            enhanced_prompt = f"{expression_data['prompt']}. Character consistency: {', '.join(character_details)}."
            print(f"  ✓ Enhanced prompt with character details from reference images")
    
    try:
        # Initialize Vertex AI
        vertex_init(project=PROJECT_ID, location=LOCATION)
        model = ImageGenerationModel.from_pretrained(MODEL)
        
        # Prepare parameters
        params = {
            "prompt": enhanced_prompt,
            "number_of_images": 1,
            "aspect_ratio": ASPECT_RATIO,
        }
        
        # Add negative prompt if provided
        if expression_data.get("negative_prompt"):
            params["negative_prompt"] = expression_data["negative_prompt"]
        
        # Generate image
        print(f"  → Calling Vertex AI API...")
        response = model.generate_images(**params)
        
        # Extract images from response
        if hasattr(response, 'images'):
            images = response.images
        elif hasattr(response, '__iter__') and not isinstance(response, str):
            try:
                images = list(response)
            except:
                images = [response]
        else:
            images = [response]
        
        if not images:
            print(f"  [ERROR] No images returned from API")
            return False
        
        image0 = images[0] if isinstance(images, list) else images
        
        # Extract image bytes - try multiple methods
        image_bytes = None
        pil_image = None
        
        # Method 1: Check for image_bytes attribute
        if hasattr(image0, "image_bytes") and image0.image_bytes:
            image_bytes = image0.image_bytes
            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
        
        # Method 2: Try _as_bytes()
        if pil_image is None and hasattr(image0, "_as_bytes"):
            try:
                image_bytes = image0._as_bytes()
                pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")
            except Exception:
                pass
        
        # Method 3: Try to_pil() or _pil_image
        if pil_image is None:
            try:
                if hasattr(image0, "to_pil"):
                    pil_image = image0.to_pil()
                elif hasattr(image0, "_pil_image"):
                    pil_image = image0._pil_image
            except Exception:
                pass
        
        # Method 4: Try save() without format parameter
        if pil_image is None:
            try:
                buffer = io.BytesIO()
                if hasattr(image0, "save"):
                    image0.save(buffer)  # Try without format
                    buffer.seek(0)
                    pil_image = PILImage.open(buffer).convert("RGBA")
            except Exception as e:
                print(f"  [WARN] Save method failed: {e}")
        
        # Method 5: Check if it's already a PIL Image
        if pil_image is None and hasattr(image0, 'size'):
            pil_image = image0
        
        if pil_image is None:
            print(f"  [ERROR] Could not extract image from response")
            print(f"     Image object type: {type(image0)}")
            print(f"     Available attributes: {[a for a in dir(image0) if not a.startswith('_')][:10]}")
            return False
        
        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(output_path, "PNG")
        
        print(f"  [SUCCESS] Successfully generated: {output_path.name}")
        print(f"     Image size: {pil_image.size[0]}x{pil_image.size[1]}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error generating {expression_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Generate all expression images"""
    print("="*60)
    print("Kelly Expression Image Generation")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Model: {MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Expressions to generate: {len(EXPRESSIONS)}")
    print("="*60)
    
    # Check for reference images
    ref_images = get_reference_images()
    if not ref_images:
        print("\n⚠ WARNING: No reference images found!")
        print(f"Expected location: {REF_DIR}")
        print("Character consistency may be reduced without reference images.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    # Generate each expression
    results = {}
    for expression_name, expression_data in EXPRESSIONS.items():
        success = generate_expression_image(expression_name, expression_data)
        results[expression_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("Generation Summary")
    print("="*60)
    successful = sum(1 for s in results.values() if s)
    total = len(results)
    
    for expression_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {expression_name}: {EXPRESSIONS[expression_name]['file']}")
    
    print(f"\nTotal: {successful}/{total} images generated successfully")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())

