"""
Create a cinematic 55-second ziggurat showcase video with:
- Subtle zoom effects
- Varied transitions  
- Title overlays
- Fade in/out from black
- Professional pacing
"""
import os
import subprocess

precision_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets\precision'
output_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets'

# Define sequence with metadata - shorter, punchier cuts
# (image_file, title, duration_seconds, transition_type)
scenes = [
    # Opening - dramatic
    ('before-1080p.jpg', 'THE ZIGGURAT', 3.5, 'fade'),
    
    # Warm sequence
    ('warm-twilight-1080p.jpg', 'TWILIGHT', 1.8, 'slideleft'),
    ('warm-dusk-1080p.jpg', '', 1.5, 'fade'),
    ('warm-night-1080p.jpg', '', 1.5, 'dissolve'),
    ('warm-late-night-1080p.jpg', '', 1.5, 'fadeblack'),
    
    # Gold sequence  
    ('gold-twilight-1080p.jpg', 'GOLD', 1.8, 'wipeleft'),
    ('gold-dusk-1080p.jpg', '', 1.5, 'fade'),
    ('gold-night-1080p.jpg', '', 1.5, 'smoothleft'),
    ('gold-late-night-1080p.jpg', '', 1.5, 'fade'),
    
    # Rainbow - faster pace
    ('rainbow-twilight-1080p.jpg', 'SPECTRUM', 1.6, 'circlecrop'),
    ('rainbow-dusk-1080p.jpg', '', 1.3, 'fade'),
    ('rainbow-night-1080p.jpg', '', 1.3, 'radial'),
    ('rainbow-late-night-1080p.jpg', '', 1.3, 'fade'),
    
    # Cyan
    ('cyan-twilight-1080p.jpg', 'CYAN', 1.6, 'slideright'),
    ('cyan-dusk-1080p.jpg', '', 1.3, 'fade'),
    ('cyan-night-1080p.jpg', '', 1.3, 'wiperight'),
    ('cyan-late-night-1080p.jpg', '', 1.3, 'fade'),
    
    # Cool - slow down
    ('cool-twilight-1080p.jpg', 'SERENE', 1.8, 'fadeblack'),
    ('cool-dusk-1080p.jpg', '', 1.5, 'fade'),
    ('cool-night-1080p.jpg', '', 1.5, 'smoothright'),
    ('cool-late-night-1080p.jpg', '', 1.5, 'fade'),
    
    # White - pure
    ('white-twilight-1080p.jpg', 'LIGHT', 1.8, 'vertopen'),
    ('white-dusk-1080p.jpg', '', 1.5, 'fade'),
    ('white-night-1080p.jpg', '', 1.5, 'dissolve'),
    ('white-late-night-1080p.jpg', '', 1.5, 'fade'),
    
    # USA - grand finale
    ('usa-twilight-1080p.jpg', 'AMERICANA', 2.2, 'wipeleft'),
    ('usa-dusk-1080p.jpg', '', 1.8, 'fade'),
    ('usa-night-1080p.jpg', '', 1.8, 'radial'),
    ('usa-late-night-1080p.jpg', 'THE ZIGGURAT', 3.5, 'fadeblack'),
]

# Parameters
crossfade_duration = 0.5
target_duration = 55.0
fps = 30

# Calculate current duration
raw_duration = sum(s[2] for s in scenes)
overlap_time = (len(scenes) - 1) * crossfade_duration
current_duration = raw_duration - overlap_time
print(f"Raw scene time: {raw_duration:.1f}s")
print(f"Overlap time: {overlap_time:.1f}s")
print(f"Current duration: {current_duration:.1f}s")

# Adjust all durations proportionally
scale_factor = (target_duration + overlap_time) / raw_duration
scenes = [(s[0], s[1], s[2] * scale_factor, s[3]) for s in scenes]

# Verify
final_duration = sum(s[2] for s in scenes) - overlap_time
print(f"Final duration: {final_duration:.1f}s")

# Build inputs with scale to even dimensions
inputs = []
for i, scene in enumerate(scenes):
    img_path = os.path.join(precision_dir, scene[0])
    inputs.extend(['-loop', '1', '-t', f'{scene[2]:.3f}', '-i', img_path])

# Build filter complex
filters = []

# Process each input with scale and optional text overlay
for i, scene in enumerate(scenes):
    img, title, duration, trans = scene
    
    # Scale to 1920x1080 (standard HD, even dimensions)
    base_filter = f"[{i}]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1"
    
    if title:
        # Add styled text overlay with shadow and fade
        text_filter = (
            f",drawtext=text='{title}':"
            f"fontfile='C\\:/Windows/Fonts/ariblk.ttf':"
            f"fontsize=80:fontcolor=white:"
            f"x=(w-text_w)/2:y=h-150:"
            f"shadowcolor=black@0.7:shadowx=3:shadowy=3:"
            f"alpha='if(lt(t,0.3),t/0.3,if(lt(t,{duration-0.5}),1,(({duration}-t)/0.5)))'"
        )
        filters.append(f"{base_filter}{text_filter}[s{i}]")
    else:
        filters.append(f"{base_filter}[s{i}]")

# Build xfade chain
prev_label = 's0'
offset = scenes[0][2] - crossfade_duration

for i in range(1, len(scenes)):
    trans_type = scenes[i][3]
    next_label = f'v{i:02d}' if i < len(scenes) - 1 else 'prefinal'
    filters.append(
        f'[{prev_label}][s{i}]xfade=transition={trans_type}:duration={crossfade_duration}:offset={offset:.3f}[{next_label}]'
    )
    prev_label = next_label
    offset += scenes[i][2] - crossfade_duration

# Add fade in at start and fade out at end
filters.append(f"[prefinal]fade=t=in:st=0:d=1.2,fade=t=out:st={final_duration-1.2}:d=1.2[outv]")

filter_str = ';'.join(filters)

# Build ffmpeg command
output_file = os.path.join(output_dir, 'ziggurat-cinematic.mp4')
cmd = [
    'ffmpeg', '-y',
    *inputs,
    '-filter_complex', filter_str,
    '-map', '[outv]',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'slow',
    '-crf', '17',
    '-r', str(fps),
    '-movflags', '+faststart',
    output_file
]

print(f"\nGenerating cinematic video...")
print(f"Output: {output_file}")

# Execute
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("\nVideo created successfully!")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    
    # Verify duration
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', output_file
    ], capture_output=True, text=True)
    print(f"Actual duration: {float(probe.stdout.strip()):.1f}s")
else:
    print(f"\nError:")
    print(result.stderr[-2000:])
