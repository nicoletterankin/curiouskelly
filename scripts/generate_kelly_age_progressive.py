#!/usr/bin/env python3
"""
Generate Kelly Age-Progressive Images
Creates Kelly character images across 6 age groups and 4 poses for lesson age slider
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import yaml

# Age group definitions from lesson-dna-schema.json
AGE_GROUPS = {
    "2-5": {
        "kelly_age": 3,
        "persona": "playful-toddler",
        "description": "toddler with chubby cheeks, rounded features, larger eyes relative to face, small button nose, fine soft hair, baby teeth visible in smile, innocent wide-eyed expression"
    },
    "6-12": {
        "kelly_age": 9,
        "persona": "curious-kid",
        "description": "child with elongating face proportions, emerging adult teeth, slimmer cheeks than toddler, active bright expression, energetic presence, natural youthful skin"
    },
    "13-17": {
        "kelly_age": 15,
        "persona": "enthusiastic-teen",
        "description": "teenager with maturing facial bone structure, clear youthful skin, developing adult features while maintaining youth, contemporary styling, confident expression"
    },
    "18-35": {
        "kelly_age": 27,
        "persona": "knowledgeable-adult",
        "description": "adult woman in late twenties with fully developed features, mature bone structure, confident presence, professional appearance, warm engaging expression"
    },
    "36-60": {
        "kelly_age": 48,
        "persona": "wise-mentor",
        "description": "mature woman with subtle crow's feet around eyes, slight forehead lines, natural aging, richer depth in expression, confident experienced presence, graceful maturity"
    },
    "61-102": {
        "kelly_age": 82,
        "persona": "reflective-elder",
        "description": "elder woman with silver-white hair, gentle laugh lines, natural weathering, warm wisdom in eyes, softer facial features, graceful aging, dignified presence"
    }
}

# Pose definitions based on reference images
POSES = {
    "pose1": {
        "name": "full_body_seated",
        "description": "Full body shot of Kelly seated in a classic director's chair with dark wooden frame and black canvas seat and backrest, hands resting gently on her lap, feet visible wearing white sneakers, sitting upright with good posture, looking directly at camera with warm engaging smile",
        "framing": "full body, head to feet visible",
        "clothing": "light blue ribbed crew-neck sweater, blue jeans with rolled cuffs, white sneakers",
        "negative": "hands hidden, feet cut off, chair not visible, awkward posture"
    },
    "pose2": {
        "name": "upper_body_seated",
        "description": "Upper body framing of Kelly seated in a classic director's chair with dark wooden frame and black canvas seat and backrest, hands resting on the wooden armrests, chair frame visible around her, sitting upright with shoulders relaxed, looking directly at camera with warm smile",
        "framing": "upper body, waist up, chair visible",
        "clothing": "light blue ribbed crew-neck sweater",
        "negative": "hands hidden, chair not visible, torso cut off awkwardly"
    },
    "pose3": {
        "name": "close_up_portrait",
        "description": "Close-up portrait of Kelly's face and shoulders, looking directly at camera with engaging warm smile, face centered in frame, natural friendly expression",
        "framing": "close-up, face and upper shoulders",
        "clothing": "light blue ribbed crew-neck sweater visible at shoulders",
        "negative": "face cut off, looking away, unfocused eyes, cropped awkwardly"
    },
    "pose4": {
        "name": "front_facing_lean",
        "description": "Medium shot of Kelly seated in director's chair, leaning slightly forward toward camera with engaged interested posture, hands clasped or resting naturally, warm approachable smile, looking directly at camera",
        "framing": "medium shot, chest up",
        "clothing": "light blue ribbed crew-neck sweater",
        "negative": "leaning too far, awkward angle, stiff posture, hands in unnatural position"
    }
}

# Kelly's core identity features (consistent across all ages)
KELLY_CORE_FEATURES = """
Maintaining core identity: oval face shape, warm brown eyes, medium brown hair with subtle caramel/honey-blonde highlights (adjust color for age appropriateness), warm genuine smile, approachable professional demeanor.
"""

# Background and lighting (consistent across all)
BACKGROUND_LIGHTING = """
Clean white studio background with soft even professional lighting, bright natural light streaming from unseen window on the left creating subtle geometric diagonal shadows on the back wall in arrow-like patterns, photorealistic quality, 8K detail.
"""


def build_age_specific_prompt(age_group: str, pose: str) -> str:
    """Build complete prompt for specific age and pose combination"""
    age_info = AGE_GROUPS[age_group]
    pose_info = POSES[pose]
    
    kelly_age = age_info["kelly_age"]
    age_desc = age_info["description"]
    pose_desc = pose_info["description"]
    clothing = pose_info["clothing"]
    framing = pose_info["framing"]
    
    prompt = f"""Professional photograph of Kelly at age {kelly_age}, a {age_desc}. 

{pose_desc}

Framing: {framing}. Clothing: {clothing}.

{KELLY_CORE_FEATURES}

{BACKGROUND_LIGHTING}

Photorealistic, high detail, 8K quality, natural professional photography."""
    
    return prompt.strip()


def build_negative_prompt(age_group: str, pose: str) -> str:
    """Build negative prompt for specific age"""
    pose_info = POSES[pose]
    pose_negative = pose_info["negative"]
    kelly_age = AGE_GROUPS[age_group]["kelly_age"]
    
    base_negative = "distorted face, extra limbs, deformed, blur, low quality, artifacts, bad anatomy, bad proportions, unrealistic, cartoon, anime, illustration, painting, sketch, CGI, 3D render, doll-like, plastic skin, heavy makeup"
    
    # Age-specific negatives
    if kelly_age <= 15:
        age_negative = "aged, mature adult features, wrinkles, crow's feet, professional business attire, heavy makeup, sophisticated styling"
    elif kelly_age == 27:
        age_negative = "child-like, baby features, elderly, wrinkles, gray hair, white hair"
    else:  # 48 and 82
        if kelly_age == 48:
            age_negative = "young child, teenager, baby face, elderly white hair, deep wrinkles (only subtle aging appropriate)"
        else:  # 82
            age_negative = "young, child-like, smooth unaged skin, no natural aging, artificially young, baby face, teenager"
    
    return f"{base_negative}, {age_negative}, {pose_negative}"


def generate_preset_yaml(age_group: str, pose: str, output_dir: Path) -> Path:
    """Generate YAML preset file for specific age/pose combination"""
    age_info = AGE_GROUPS[age_group]
    kelly_age = age_info["kelly_age"]
    pose_name = POSES[pose]["name"]
    
    filename = f"kelly_age{kelly_age}_{pose}_v001.yaml"
    output_path = output_dir / filename
    
    # Build preset structure (only fields accepted by Preset dataclass)
    preset = {
        "asset_type": f"age_{kelly_age}_{pose_name}",
        "view": "front",
        "lighting": "studio_neutral",
        "prompt": build_age_specific_prompt(age_group, pose),
        "negative_prompt": build_negative_prompt(age_group, pose),
        "seed": None,  # Let it vary naturally
        "output": {
            "width": 2048,
            "height": 1152,  # 16:9 ratio
            "version": 1
        },
        "backend": {
            "provider": "google-vertex",
            "model": "imagen-3.0-generate-002",
            "guidance_scale": 7.5,
            "upscale": {
                "enabled": False  # Keep it fast for initial generation
            }
        }
    }
    
    # Write YAML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(preset, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return output_path


def generate_all_presets(output_dir: Path) -> List[Path]:
    """Generate all 24 preset YAML files (6 ages × 4 poses)"""
    generated = []
    
    print("Generating YAML presets for age-progressive Kelly images...")
    print(f"Output directory: {output_dir}\n")
    
    for age_group in AGE_GROUPS.keys():
        kelly_age = AGE_GROUPS[age_group]["kelly_age"]
        print(f"Age group {age_group} (Kelly age {kelly_age}):")
        
        for pose in POSES.keys():
            preset_path = generate_preset_yaml(age_group, pose, output_dir)
            generated.append(preset_path)
            print(f"  ✓ Generated {preset_path.name}")
    
    print(f"\n✅ Generated {len(generated)} preset files")
    return generated


def main():
    """Main entry point"""
    # Setup output directory
    workspace_root = Path(__file__).parent.parent
    preset_dir = workspace_root / "presets" / "age_progressive"
    
    # Generate all presets
    presets = generate_all_presets(preset_dir)
    
    print(f"\n📋 Next steps:")
    print(f"   1. Review generated presets in: {preset_dir}")
    print(f"   2. Run batch generation: .\\scripts\\generate_kelly_batch_ages.ps1")
    print(f"   3. Review outputs in gallery: projects/Kelly/assets/age_progressive/review.html")


if __name__ == "__main__":
    main()

