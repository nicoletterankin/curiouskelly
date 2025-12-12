import os
import glob
from moviepy import *
from moviepy.video.fx import Resize
import time

# Configuration
DAY_NUMBER = 1
INPUT_VIDEO_DIR = "generated-videos/production-dec17"
INPUT_IMAGE_DIR = f"public/kelly/phases/{DAY_NUMBER:03d}"
OUTPUT_DIR = f"generated-videos/compiled/day_{DAY_NUMBER:03d}"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mappings
PHASES = {
    "Hook": {"img": "hook.png", "type": "full"},  # Hook is usually full screen Kelly
    "Fact1": {"img": "q1.png", "type": "pip"},    # Facts are PiP
    "Fact2": {"img": "q2.png", "type": "pip"},
    "Fact3": {"img": "q3.png", "type": "pip"},
    "Wisdom": {"img": "wisdom.png", "type": "pip"}
}

ARCHETYPES = {
    "architect": "the_architect",
    "empath": "the_empath",
    "explorer": "the_explorer",
    "macgyver": "the_macgyver",
    "mystic": "the_mystic",
    "rebel": "the_rebel",
    "scientist": "the_scientist",
    "storyteller": "the_storyteller",
    "survivor": "the_survivor",
    "provider": "the_provider",
    "strategist": "the_strategist",
    "neutral": "the_diplomat"  # 'neutral' is likely The Diplomat
}

def find_video(archetype_key, phase):
    # Try different naming conventions observed
    archetype_filename = ARCHETYPES.get(archetype_key, archetype_key)
    
    patterns = [
        f"day_{DAY_NUMBER:03d}_{phase}_{archetype_filename}.mp4",          # day_001_Fact1_the_architect.mp4
        f"day_{DAY_NUMBER:03d}_{phase.lower()}_{archetype_filename}.mp4",  # day_001_fact1_the_architect.mp4
        f"{archetype_filename}_{phase.lower()}.mp4",                       # the_architect_hook.mp4
        # Fallback for without 'the_' just in case
        f"day_{DAY_NUMBER:03d}_{phase}_{archetype_key}.mp4",
        # NEW DASH FORMAT: day-001-architect-Hook-main-en.mp4
        f"day-{DAY_NUMBER:03d}-{archetype_key}-{phase}-main-en.mp4",
        f"day-{DAY_NUMBER:03d}-{archetype_key}-{phase}-main-en.mp4".replace("Fact", "Fact"), # Case sensitive check
    ]
    
    for pat in patterns:
        path_attempt = os.path.join(INPUT_VIDEO_DIR, pat)
        if os.path.exists(path_attempt):
            return path_attempt
    return None

def compile_video(archetype, phase):
    print(f"🎬 Processing {archetype} - {phase}...")
    
    video_path = find_video(archetype, phase)
    image_path = os.path.join(INPUT_IMAGE_DIR, PHASES[phase]["img"])
    
    if not video_path:
        print(f"  ❌ Video not found for {archetype} {phase}")
        return False
        
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        # Fallback to full screen video if image missing?
        # return False
    
    try:
        # Load Video
        video = VideoFileClip(video_path)
        
        # If Hook, just return the video (maybe trim slightly if needed)
        if PHASES[phase]["type"] == "full" or not os.path.exists(image_path):
            final_clip = video
            # Optional: Add a subtle zoom or fade if desired for "dynamic" feel
            
        else:
            # PiP Mode
            # Load Image as background, set duration to match video
            bg_image = ImageClip(image_path).with_duration(video.duration)
            
            # Resize image to match video aspect ratio/size if needed?
            # Assuming 1080x1080 output
            bg_image = bg_image.resized(width=1080, height=1080) # Force square 1080
            
            # Resize Kelly video for PiP
            # 35% scale
            scale_factor = 0.35
            kelly_pip = video.resized(scale_factor)
            
            # Calculate position: Bottom Right with 30px padding
            # Canvas is 1080x1080
            # Pip size is roughly 378x378
            padding = 30
            # Position is (x, y) of top-left corner
            pos_x = 1080 - (1080 * scale_factor) - padding
            pos_y = 1080 - (1080 * scale_factor) - padding
            
            kelly_pip = kelly_pip.with_position((pos_x, pos_y))
            
            # Create Composite
            final_clip = CompositeVideoClip([bg_image, kelly_pip], size=(1080, 1080))
        
        # Output Path
        output_filename = f"day_{DAY_NUMBER:03d}_{phase.lower()}_{archetype}_dynamic.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Write File
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            fps=24,
            preset="fast",
            logger=None # Silence verbose logs
        )
        
        print(f"  ✅ Saved: {output_filename}")
        
        # Close clips to free resources
        video.close()
        if 'bg_image' in locals(): bg_image.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("🚀 Starting Dynamic Lesson Compilation...")
    count = 0
    # Process all archetypes
    test_archetypes = ARCHETYPES
    
    for arch in test_archetypes:
        for phase in PHASES:
            if compile_video(arch, phase):
                count += 1
    
    print(f"\n✨ Completed {count} videos.")

if __name__ == "__main__":
    main()

