import json
import os
import subprocess
import sys
from pathlib import Path

# Configuration
CALENDAR_PATH = Path("lessons/365_day_calendar.json")
OUTPUT_DIR = Path("generated_thumbnails")
TOOL_SCRIPT = Path("tools/generate_via_rest.py")

# Full Character Description from Knowledge Base
CHARACTER_DESC = """Kelly Rein, photorealistic digital human, modern timeless "Apple Genius" aesthetic. Oval face, clear smooth complexion with natural glow, warm light-medium skin tone, healthy radiant skin. Warm brown almond-shaped eyes, bright and engaging, well-defined dark brown eyebrows with natural arch, long dark eyelashes. Medium brown hair with subtle caramel/honey-blonde highlights, soft wavy to slightly curly texture, parted slightly off-center or down the middle, cascades over shoulders, rich and voluminous. Full lips with natural rosy-pink color, genuine warm smile showing straight white teeth, natural smile lines (nasolabial folds), slight crinkles at outer corners of eyes when smiling. Late 20s to early 30s, athletic build, strong capable presence, approachable and professional demeanor."""

# Wardrobe Description
WARDROBE_DESC = """Wearing light blue ribbed knit sweater, crew neck, sitting in classic director's chair with dark brown wooden frame and black canvas."""

# Negative Prompts
NEGATIVE_PROMPTS = """cartoon, stylized, anime, illustration, drawing, sketch, fantasy, medieval, Roman, ancient, historical, exaggerated features, unrealistic proportions, bright colors, red, yellow, orange, light browns, tan, beige, leather straps, Roman armor, ornate decorations, jewelry, low quality, blurry, pixelated, compression artifacts, oversaturated colors, unrealistic lighting, watermark, text overlay, logo, CGI, 3D render, game asset, sprite"""

def generate_prompt(lesson):
    topic = lesson.get("title", "Lesson")
    
    # Construct the scene description based on topic
    scene_desc = f"Kelly sitting in a director's chair discussing '{topic}'"
    
    full_prompt = f"""{scene_desc}, featuring {CHARACTER_DESC} {WARDROBE_DESC} Clean bright white or very light gray background with subtle visual elements related to {topic}. Soft even studio lighting, professional photography quality.

Maintain exact facial features, hair color (medium brown with caramel/honey-blonde highlights), skin tone (warm light-medium), eye color (warm brown almond-shaped), and overall appearance.

Negative prompt: {NEGATIVE_PROMPTS}"""

    return full_prompt

def generate_thumbnail(lesson):
    output_path = OUTPUT_DIR / f"thumbnail_day_{lesson['day']}_{lesson['lesson_id']}.png"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Force regeneration for this test
    if output_path.exists():
        os.remove(output_path)

    prompt = generate_prompt(lesson)
    
    cmd = [
        sys.executable, str(TOOL_SCRIPT),
        "--prompt", prompt,
        "--output", str(output_path),
        "--project", "gen-lang-client-0005524332" # Hardcoded from previous success
    ]
    
    print(f"\n🚀 Generating thumbnail for Day {lesson['day']}: {lesson['title']}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Generated: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed for Day {lesson['day']}: {e}")

def main():
    if not CALENDAR_PATH.exists():
        print(f"❌ Calendar file not found: {CALENDAR_PATH}")
        return

    with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        lessons = data.get("lessons", [])

    print(f"Found {len(lessons)} lessons. Generating samples for first 3...")
    
    for lesson in lessons[:3]:
        generate_thumbnail(lesson)

if __name__ == "__main__":
    main()
