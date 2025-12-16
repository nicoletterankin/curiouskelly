import os
import json
import requests
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# Configuration
VERTEX_PROJECT = "curious-kelly-proj" # You might need to verify this
VERTEX_LOCATION = "us-central1"
OUTPUT_DIR = "assets/generated_bash"
MANIFEST_FILE = "KELLY_ASSETS_MANIFEST_V2.json"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_best_references(manifest):
    """Selects the highest quality reference images for the bash."""
    refs = []
    # We want the "Best Character Reference" ones specifically
    for name, data in manifest.items():
        if "best-character-reference" in name and "face" in name:
            refs.append(data)
        elif "neutral-face-with-hair" in name:
            refs.append(data)
            
    # Fallback if strict filtering misses too much
    if len(refs) < 3:
        for name, data in manifest.items():
            if "kelly_front" in name or "kelly-ref-front" in name:
                refs.append(data)
                
    return refs[:4] # Limit to 4 to avoid overwhelming the API

def download_image_as_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    return None

def generate_with_bash(references):
    """
    Attempts to generate a new image using the hosted references.
    Since standard Vertex SDK might fail, we will try to construct a payload
    that uses the image bytes directly if we were to use the REST API, 
    or just simulate the "Bash" by creating a detailed prompt.
    """
    
    print(f"🎨 Bashing {len(references)} references for generation...")
    
    # 1. Construct the Prompt
    prompt = """
    Photorealistic portrait of a young woman named Kelly, 25 years old.
    She has long brown wavy hair, warm brown eyes, and a friendly, intelligent expression.
    She is wearing a casual but professional blue ribbed sweater.
    Lighting is soft, studio quality, white background.
    High detail, 8k, sharp focus, cinematic lighting.
    Character consistency is key. Matches the reference images provided.
    """
    
    # 2. (Simulation) In a real scenario with a working Image-to-Image endpoint, 
    # we would send the base64 of the primary reference here.
    # For now, we will save a "Bash Composite" to prove we have the data ready.
    
    # Let's create a composite image locally to show we "bashed" them
    images = []
    for ref in references:
        try:
            resp = requests.get(ref['public_url'], stream=True)
            img = Image.open(resp.raw).convert('RGBA')
            img.thumbnail((512, 512))
            images.append(img)
        except Exception as e:
            print(f"Failed to load {ref['public_url']}: {e}")

    if not images:
        print("❌ No references could be loaded.")
        return

    # Create a grid
    grid_width = 2 * 512
    grid_height = 2 * 512
    composite = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        if i >= 4: break
        x = (i % 2) * 512
        y = (i // 2) * 512
        # Center image in slot
        x_off = (512 - img.width) // 2
        y_off = (512 - img.height) // 2
        composite.paste(img, (x + x_off, y + y_off), img)

    composite_path = os.path.join(OUTPUT_DIR, "kelly_reference_bash_composite.png")
    composite.save(composite_path)
    
    print(f"✅ Generated Reference Bash Composite: {composite_path}")
    print("   (This composite represents the visual data we would feed to a ControlNet or Img2Img pipeline)")
    
    # 3. Attempt Actual Generation (Mockup for now as we need valid credentials/SDK setup confirmed)
    # In a full run, we'd call:
    # vertex_ai.generate_image(prompt=prompt, image=primary_ref_bytes)
    
    print("\n🚀 READY FOR GENERATION PIPELINE")
    print("To run the actual generation, we need to confirm:")
    print("1. Google Cloud Project ID is active")
    print("2. Vertex AI API is enabled")
    print("3. We have quota for 'imagen-3.0-generate-001'")
    
if __name__ == "__main__":
    manifest = load_manifest()
    best_refs = get_best_references(manifest)
    
    if best_refs:
        print(f"Found {len(best_refs)} best references:")
        for r in best_refs:
            print(f" - {r['public_url']}")
        generate_with_bash(best_refs)
    else:
        print("⚠️ Could not identify 'Best' references in manifest. Check naming.")






























