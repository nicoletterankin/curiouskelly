"""
Generate the 3 missing tribe banners using Vertex AI Imagen
"""
import os
import base64
import time
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel

# Initialize Vertex AI
PROJECT_ID = "ui-tars"
LOCATION = "us-central1"
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Prompts from runner_game_asset_prompts.md
BANNERS = {
    "code": {
        "prompt": "Vertical banner, fabric weave texture, bold symbol for the Code Tribe (code brackets '<>'), base color #0BB39C (teal), subtle gold trim, orthographic, transparent background, 128x256px.",
        "negative_prompt": "photo, realistic, 3d, shadows, perspective, text, words"
    },
    "fire": {
        "prompt": "Vertical banner, fabric weave texture, bold symbol for the Fire Tribe (flame), base color #F25F5C (red-orange), subtle gold trim, orthographic, transparent background, 128x256px.",
        "negative_prompt": "photo, realistic, 3d, shadows, perspective, text, words"
    },
    "metal": {
        "prompt": "Vertical banner, fabric weave texture, bold symbol for the Metal Tribe (gear), base color #adb5bd (gray), subtle gold trim, orthographic, transparent background, 128x256px.",
        "negative_prompt": "photo, realistic, 3d, shadows, perspective, text, words"
    }
}

def generate_banner(tribe_name, prompt_data):
    """Generate a single banner"""
    print(f"\n🎨 Generating {tribe_name} banner...")
    
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        response = model.generate_images(
            prompt=prompt_data["prompt"],
            negative_prompt=prompt_data["negative_prompt"],
            number_of_images=1,
            aspect_ratio="9:16",  # Closest to 128:256
            safety_filter_level="block_few",
            person_generation="allow_adult"
        )
        
        if response.images:
            # Save the image
            image = response.images[0]
            output_path = f"assets/banners/banner_{tribe_name}.png"
            
            # Save the image bytes
            image._pil_image.save(output_path)
            print(f"✅ Saved {output_path}")
            return True
        else:
            print(f"❌ No image generated for {tribe_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating {tribe_name}: {e}")
        return False

def main():
    print("🎨 Generating 3 missing tribe banners...")
    print("=" * 50)
    
    success_count = 0
    
    for tribe, prompt_data in BANNERS.items():
        if generate_banner(tribe, prompt_data):
            success_count += 1
        time.sleep(2)  # Rate limiting
    
    print("\n" + "=" * 50)
    print(f"✅ Generated {success_count}/3 banners successfully!")
    
    if success_count == 3:
        print("\n🎉 ALL ASSETS COMPLETE! Ready to build the game.")

if __name__ == "__main__":
    main()







