import os
import json
import requests
import base64
from pathlib import Path
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from PIL import Image
from io import BytesIO

# Configuration
GOOGLE_PROJECT_ID = "gen-lang-client-0005524332"
GOOGLE_LOCATION = "us-central1"
IMAGEN_MODEL = "imagen-3.0-generate-001"
OUTPUT_DIR = Path("assets/generated_production")
REFERENCE_COMPOSITE = Path("assets/generated_bash/kelly_reference_bash_composite.png")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def initialize_vertex():
    """Initialize Vertex AI with the project credentials."""
    print(f"🔧 Initializing Vertex AI...")
    print(f"   Project: {GOOGLE_PROJECT_ID}")
    print(f"   Location: {GOOGLE_LOCATION}")
    
    aiplatform.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
    print("✅ Vertex AI initialized.")

def generate_kelly_production_image():
    """Generate the first production-quality Kelly image using reference bash."""
    
    print("\n🎨 GENERATING PRODUCTION KELLY IMAGE")
    print("=" * 60)
    
    # The Ultimate Kelly Prompt (Based on all references we've analyzed)
    prompt = """
    Photorealistic portrait photograph of a young woman named Kelly, age 25.
    
    Physical Appearance:
    - Long, wavy brown hair with subtle caramel highlights
    - Warm brown eyes, expressive and intelligent
    - Fair skin with natural glow
    - Soft, friendly smile
    - Natural makeup, professional but approachable
    
    Clothing:
    - Light blue ribbed long-sleeved crew-neck sweater
    - Casual but professional style
    
    Composition:
    - Head and shoulders framing
    - Direct eye contact with camera
    - Engaging, warm expression
    
    Lighting & Environment:
    - Clean white studio background
    - Soft, even studio lighting
    - Subtle window light creating gentle shadows
    - Professional photography setup
    
    Quality:
    - 8K resolution
    - Sharp focus
    - High detail, photorealistic
    - Cinematic lighting
    - Professional portrait photography
    
    Character consistency is critical. This should match the reference style exactly.
    """
    
    negative_prompt = """
    cartoon, anime, illustration, painting, drawing, 3D render, CGI, 
    unrealistic, low quality, blurry, distorted, deformed, 
    multiple people, text, watermark, logo, artificial looking,
    oversaturated colors, harsh lighting, shadows on face
    """
    
    try:
        # Using the Vertex AI Imagen API
        from vertexai.preview.vision_models import ImageGenerationModel
        
        model = ImageGenerationModel.from_pretrained(IMAGEN_MODEL)
        
        print(f"📸 Generating with model: {IMAGEN_MODEL}")
        print(f"   Aspect Ratio: 16:9 (lesson format)")
        print(f"   Number of variations: 4")
        
        # Generate multiple variations
        response = model.generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=4,
            aspect_ratio="16:9",
            safety_filter_level="block_few",
            person_generation="allow_adult"
        )
        
        print(f"\n✅ Generation complete! Processing {len(response.images)} images...")
        
        # Save all variations
        for i, image in enumerate(response.images):
            output_path = OUTPUT_DIR / f"kelly_production_v{i+1}_16x9.png"
            image._pil_image.save(output_path, format="PNG")
            print(f"   💾 Saved: {output_path.name}")
        
        # Also generate a square version for profile/avatar use
        print(f"\n📸 Generating square version (1:1) for profile use...")
        response_square = model.generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            number_of_images=2,
            aspect_ratio="1:1",
            safety_filter_level="block_few",
            person_generation="allow_adult"
        )
        
        for i, image in enumerate(response_square.images):
            output_path = OUTPUT_DIR / f"kelly_production_profile_v{i+1}_1x1.png"
            image._pil_image.save(output_path, format="PNG")
            print(f"   💾 Saved: {output_path.name}")
        
        print("\n" + "=" * 60)
        print("🎉 PRODUCTION GENERATION COMPLETE!")
        print(f"📂 Output directory: {OUTPUT_DIR}")
        print(f"📊 Total images generated: {len(response.images) + len(response_square.images)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Vertex AI API is enabled in Google Cloud Console")
        print("2. Check that you have quota for imagen-3.0-generate-001")
        print("3. Verify authentication: gcloud auth application-default login")
        return False

def main():
    print("🚀 KELLY PRODUCTION IMAGE GENERATION")
    print("=" * 60)
    
    # Check if reference composite exists
    if REFERENCE_COMPOSITE.exists():
        print(f"✅ Reference composite found: {REFERENCE_COMPOSITE}")
        print("   (This composite informed the prompt design)")
    else:
        print(f"⚠️  Reference composite not found at {REFERENCE_COMPOSITE}")
        print("   Proceeding with text-only generation...")
    
    # Initialize Vertex AI
    initialize_vertex()
    
    # Generate production images
    success = generate_kelly_production_image()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Review the generated images in assets/generated_production/")
        print("2. Select the best variation")
        print("3. Upload to Supabase kelly_v2/production/")
        print("4. Update curiouskelly.com landing page")
    else:
        print("\n⚠️  Generation incomplete. Please check errors above.")

if __name__ == "__main__":
    main()





































