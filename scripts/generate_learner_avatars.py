#!/usr/bin/env python3
"""
Generate Learner Avatar Images
==============================
Creates diverse, authentic avatar images for the social learning personas.

Uses AI image generation to create friendly, approachable learner portraits
that represent global diversity (age, ethnicity, style).

Usage:
    python scripts/generate_learner_avatars.py --all
    python scripts/generate_learner_avatars.py --persona emma-us
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OUTPUT_DIR = Path("public/images/learners")

# ═══════════════════════════════════════════════════════════════════
# PERSONA DATA (matches learner-personas.js)
# ═══════════════════════════════════════════════════════════════════

PERSONAS = [
    # Each persona has prompt-friendly description
    {"id": "emma-us", "name": "Emma", "age": 28, "ethnicity": "white American", "style": "professional casual, glasses optional"},
    {"id": "marcus-us", "name": "Marcus", "age": 16, "ethnicity": "Black American", "style": "sporty, confident teen"},
    {"id": "sarah-ca", "name": "Sarah", "age": 45, "ethnicity": "white Canadian", "style": "warm teacher, kind eyes"},
    {"id": "joe-us", "name": "Joe", "age": 72, "ethnicity": "white American senior", "style": "wise grandfather, gentle smile"},
    {"id": "maya-mx", "name": "Maya", "age": 34, "ethnicity": "Mexican", "style": "creative professional"},
    
    {"id": "james-uk", "name": "James", "age": 31, "ethnicity": "British", "style": "neat, professional"},
    {"id": "charlotte-uk", "name": "Charlotte", "age": 8, "ethnicity": "British girl", "style": "curious, bright eyes"},
    {"id": "marie-fr", "name": "Marie", "age": 52, "ethnicity": "French woman", "style": "elegant, sophisticated"},
    {"id": "lucas-fr", "name": "Lucas", "age": 19, "ethnicity": "French young man", "style": "intellectual, thoughtful"},
    {"id": "hans-de", "name": "Hans", "age": 67, "ethnicity": "German senior man", "style": "scholarly, kind"},
    {"id": "lena-de", "name": "Lena", "age": 24, "ethnicity": "German young woman", "style": "focused, dedicated"},
    {"id": "isabella-it", "name": "Isabella", "age": 38, "ethnicity": "Italian woman", "style": "passionate, expressive"},
    {"id": "sven-se", "name": "Sven", "age": 29, "ethnicity": "Swedish man", "style": "modern, design-focused"},
    {"id": "nina-no", "name": "Nina", "age": 41, "ethnicity": "Norwegian woman", "style": "outdoorsy, natural"},
    {"id": "olga-ua", "name": "Olga", "age": 33, "ethnicity": "Ukrainian woman", "style": "determined, smart"},
    
    {"id": "yuki-jp", "name": "Yuki", "age": 26, "ethnicity": "Japanese woman", "style": "artistic, modern"},
    {"id": "haruto-jp", "name": "Haruto", "age": 12, "ethnicity": "Japanese boy", "style": "curious student"},
    {"id": "sakura-jp", "name": "Sakura", "age": 58, "ethnicity": "Japanese woman", "style": "serene, graceful"},
    {"id": "priya-in", "name": "Priya", "age": 22, "ethnicity": "Indian woman", "style": "ambitious student"},
    {"id": "arjun-in", "name": "Arjun", "age": 35, "ethnicity": "Indian man", "style": "professional, warm father"},
    {"id": "ananya-in", "name": "Ananya", "age": 9, "ethnicity": "Indian girl", "style": "imaginative, playful"},
    {"id": "wei-cn", "name": "Wei", "age": 44, "ethnicity": "Chinese man", "style": "business professional"},
    {"id": "mei-cn", "name": "Mei", "age": 17, "ethnicity": "Chinese teen girl", "style": "studious, focused"},
    {"id": "jin-kr", "name": "Jin", "age": 27, "ethnicity": "Korean man", "style": "creative, modern"},
    {"id": "soo-yeon-kr", "name": "Soo-yeon", "age": 63, "ethnicity": "Korean grandmother", "style": "warm, wise"},
    
    {"id": "ahmed-eg", "name": "Ahmed", "age": 30, "ethnicity": "Egyptian man", "style": "thoughtful teacher"},
    {"id": "fatima-eg", "name": "Fatima", "age": 21, "ethnicity": "Egyptian woman", "style": "curious journalist"},
    {"id": "omar-ae", "name": "Omar", "age": 39, "ethnicity": "Emirati man", "style": "professional, modern"},
    {"id": "layla-ae", "name": "Layla", "age": 14, "ethnicity": "Emirati teen girl", "style": "bright, scientific"},
    
    {"id": "kofi-gh", "name": "Kofi", "age": 25, "ethnicity": "Ghanaian man", "style": "practical, community-focused"},
    {"id": "ama-gh", "name": "Ama", "age": 48, "ethnicity": "Ghanaian woman", "style": "experienced educator"},
    {"id": "aisha-ke", "name": "Aisha", "age": 20, "ethnicity": "Kenyan woman", "style": "passionate, environmental"},
    {"id": "thabo-za", "name": "Thabo", "age": 36, "ethnicity": "South African man", "style": "creative, musical"},
    {"id": "naledi-za", "name": "Naledi", "age": 11, "ethnicity": "South African girl", "style": "playful, energetic"},
    {"id": "adebayo-ng", "name": "Adebayo", "age": 42, "ethnicity": "Nigerian man", "style": "tech entrepreneur"},
    
    {"id": "maria-br", "name": "Maria", "age": 28, "ethnicity": "Brazilian woman", "style": "compassionate nurse"},
    {"id": "pedro-br", "name": "Pedro", "age": 55, "ethnicity": "Brazilian man", "style": "weathered, experienced"},
    {"id": "carlos-ar", "name": "Carlos", "age": 32, "ethnicity": "Argentinian man", "style": "thoughtful professional"},
    {"id": "diego-cl", "name": "Diego", "age": 18, "ethnicity": "Chilean young man", "style": "dreamy, stargazer"},
    {"id": "valentina-co", "name": "Valentina", "age": 7, "ethnicity": "Colombian girl", "style": "joyful, colorful"},
    
    {"id": "lisa-au", "name": "Lisa", "age": 37, "ethnicity": "Australian woman", "style": "outdoorsy, natural"},
    {"id": "jack-nz", "name": "Jack", "age": 23, "ethnicity": "New Zealand man", "style": "adventurous, friendly"},
    {"id": "linh-vn", "name": "Linh", "age": 29, "ethnicity": "Vietnamese woman", "style": "calm, steady"},
    {"id": "ling-sg", "name": "Ling", "age": 45, "ethnicity": "Singaporean Chinese woman", "style": "professional, structured"},
    {"id": "kai-th", "name": "Kai", "age": 19, "ethnicity": "Thai man", "style": "open, curious traveler"},
    {"id": "putri-id", "name": "Putri", "age": 31, "ethnicity": "Indonesian woman", "style": "nurturing teacher"},
    
    {"id": "zara-pk", "name": "Zara", "age": 26, "ethnicity": "Pakistani woman", "style": "empathetic, community worker"},
    {"id": "elena-ru", "name": "Elena", "age": 40, "ethnicity": "Russian woman", "style": "elegant, disciplined"},
    {"id": "tomasz-pl", "name": "Tomasz", "age": 50, "ethnicity": "Polish man", "style": "practical craftsman"},
    {"id": "anna-gr", "name": "Anna", "age": 65, "ethnicity": "Greek woman", "style": "warm grandmother"},
    {"id": "chen-tw", "name": "Chen", "age": 34, "ethnicity": "Taiwanese man", "style": "tech-focused"},
    {"id": "fatou-sn", "name": "Fatou", "age": 22, "ethnicity": "Senegalese woman", "style": "dedicated student"},
    {"id": "miguel-es", "name": "Miguel", "age": 47, "ethnicity": "Spanish man", "style": "warm, expressive chef"},
    {"id": "ana-pt", "name": "Ana", "age": 15, "ethnicity": "Portuguese teen girl", "style": "active, athletic"},
]

# ═══════════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_avatar_prompt(persona: dict) -> str:
    """Generate a prompt for creating a learner avatar."""
    
    base_prompt = f"""Portrait photo of {persona['name']}, a {persona['age']}-year-old {persona['ethnicity']} person.
Style: {persona['style']}.
Expression: Friendly, approachable, natural smile, engaged learner.
Setting: Simple, clean background with soft lighting.
Quality: Professional headshot, high quality, realistic, natural skin texture.
Mood: Warm, welcoming, curious, ready to learn.
Format: Square crop, centered face, shoulders visible."""
    
    # Age-specific adjustments
    if persona['age'] < 13:
        base_prompt += "\nNote: Child-appropriate, innocent expression, school-age appearance."
    elif persona['age'] > 60:
        base_prompt += "\nNote: Distinguished, wise appearance, natural aging, kind eyes."
    
    return base_prompt


def generate_with_replicate(prompt: str, output_path: Path) -> bool:
    """Generate image using Replicate API (FLUX or similar)."""
    if not REPLICATE_API_TOKEN:
        print("⚠️ REPLICATE_API_TOKEN not set, skipping generation")
        return False
    
    try:
        import replicate
        
        # Using FLUX for high quality portraits
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "jpg",
                "output_quality": 90,
            }
        )
        
        # Download the image
        if output:
            img_url = output[0] if isinstance(output, list) else output
            response = requests.get(img_url)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            return True
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    return False


def create_placeholder(output_path: Path, persona: dict):
    """Create a placeholder SVG avatar."""
    # Simple colored circle with initials
    colors = {
        'US': '#3b82f6', 'CA': '#ef4444', 'MX': '#22c55e', 'GB': '#6366f1',
        'FR': '#f59e0b', 'DE': '#eab308', 'IT': '#14b8a6', 'SE': '#0ea5e9',
        'NO': '#ec4899', 'UA': '#fbbf24', 'JP': '#f43f5e', 'IN': '#8b5cf6',
        'CN': '#dc2626', 'KR': '#2563eb', 'EG': '#d97706', 'AE': '#059669',
        'GH': '#16a34a', 'KE': '#b91c1c', 'ZA': '#7c3aed', 'NG': '#15803d',
        'BR': '#facc15', 'AR': '#60a5fa', 'CL': '#f87171', 'CO': '#fcd34d',
        'AU': '#10b981', 'NZ': '#000000', 'VN': '#dc2626', 'SG': '#ef4444',
        'TH': '#1d4ed8', 'ID': '#dc2626', 'PK': '#15803d', 'RU': '#1e40af',
        'PL': '#dc2626', 'GR': '#2563eb', 'TW': '#dc2626', 'SN': '#15803d',
        'ES': '#fbbf24', 'PT': '#15803d',
    }
    
    color = colors.get(persona['country_code'] if 'country_code' in persona else persona.get('ethnicity', '')[:2].upper(), '#6366f1')
    initials = persona['name'][0].upper()
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
  <circle cx="64" cy="64" r="64" fill="{color}"/>
  <text x="64" y="80" font-family="Arial, sans-serif" font-size="48" font-weight="bold" fill="white" text-anchor="middle">{initials}</text>
</svg>'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_path.with_suffix('.svg')
    svg_path.write_text(svg)
    print(f"   📝 Created placeholder: {svg_path.name}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def generate_avatar(persona: dict, force: bool = False):
    """Generate avatar for a single persona."""
    output_path = OUTPUT_DIR / f"{persona['id']}.jpg"
    
    if output_path.exists() and not force:
        print(f"   ✓ {persona['name']} already exists")
        return
    
    print(f"   🎨 Generating {persona['name']}...")
    
    prompt = generate_avatar_prompt(persona)
    
    if REPLICATE_API_TOKEN:
        success = generate_with_replicate(prompt, output_path)
        if success:
            print(f"   ✅ Created: {output_path.name}")
            return
    
    # Fallback to placeholder
    create_placeholder(output_path, persona)


def main():
    parser = argparse.ArgumentParser(description="Generate learner avatar images")
    parser.add_argument("--all", action="store_true", help="Generate all avatars")
    parser.add_argument("--persona", type=str, help="Generate specific persona (id)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing")
    parser.add_argument("--placeholders", action="store_true", help="Only create placeholders")
    
    args = parser.parse_args()
    
    print("🖼️ Learner Avatar Generator")
    print("=" * 50)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.persona:
        persona = next((p for p in PERSONAS if p['id'] == args.persona), None)
        if persona:
            if args.placeholders:
                create_placeholder(OUTPUT_DIR / f"{persona['id']}.jpg", persona)
            else:
                generate_avatar(persona, args.force)
        else:
            print(f"❌ Persona '{args.persona}' not found")
    
    elif args.all or args.placeholders:
        print(f"Generating {len(PERSONAS)} avatars...")
        for persona in PERSONAS:
            if args.placeholders:
                create_placeholder(OUTPUT_DIR / f"{persona['id']}.jpg", persona)
            else:
                generate_avatar(persona, args.force)
        print(f"\n✅ Complete! Avatars in {OUTPUT_DIR}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

