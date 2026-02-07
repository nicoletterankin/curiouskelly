"""
Create a PREMIUM 55-second ziggurat showcase video with:
- Smooth zoom animation (scale-based, not zoompan)
- Varied cinematic transitions  
- Elegant title overlays
- Fade in/out from black
- Vignette effect
- Color grading for cinematic look
- 24fps film feel
"""
import os
import subprocess

precision_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets\precision'
output_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets'

scenes = [
    # (image, title, base_duration, transition)
    ('before-1080p.jpg', 'THE ZIGGURAT', 4.0, 'fade'),
    ('warm-twilight-1080p.jpg', 'WARM', 2.0, 'slideleft'),
    ('warm-dusk-1080p.jpg', '', 1.7, 'fade'),
    ('warm-night-1080p.jpg', '', 1.7, 'dissolve'),
    ('warm-late-night-1080p.jpg', '', 1.7, 'fadeblack'),
    ('gold-twilight-1080p.jpg', 'GOLD', 2.0, 'wipeleft'),
    ('gold-dusk-1080p.jpg', '', 1.7, 'fade'),
    ('gold-night-1080p.jpg', '', 1.7, 'smoothleft'),
    ('gold-late-night-1080p.jpg', '', 1.7, 'fade'),
    ('rainbow-twilight-1080p.jpg', 'SPECTRUM', 1.8, 'circlecrop'),
    ('rainbow-dusk-1080p.jpg', '', 1.4, 'fade'),
    ('rainbow-night-1080p.jpg', '', 1.4, 'radial'),
    ('rainbow-late-night-1080p.jpg', '', 1.4, 'fade'),
    ('cyan-twilight-1080p.jpg', 'CYAN', 1.8, 'slideright'),
    ('cyan-dusk-1080p.jpg', '', 1.4, 'fade'),
    ('cyan-night-1080p.jpg', '', 1.4, 'wiperight'),
    ('cyan-late-night-1080p.jpg', '', 1.4, 'fade'),
    ('cool-twilight-1080p.jpg', 'SERENE', 2.0, 'fadeblack'),
    ('cool-dusk-1080p.jpg', '', 1.7, 'fade'),
    ('cool-night-1080p.jpg', '', 1.7, 'smoothright'),
    ('cool-late-night-1080p.jpg', '', 1.7, 'fade'),
    ('white-twilight-1080p.jpg', 'PURE', 2.0, 'vertopen'),
    ('white-dusk-1080p.jpg', '', 1.7, 'fade'),
    ('white-night-1080p.jpg', '', 1.7, 'dissolve'),
    ('white-late-night-1080p.jpg', '', 1.7, 'fade'),
    ('usa-twilight-1080p.jpg', 'AMERICANA', 2.5, 'wipeleft'),
    ('usa-dusk-1080p.jpg', '', 2.0, 'fade'),
    ('usa-night-1080p.jpg', '', 2.0, 'radial'),
    ('usa-late-night-1080p.jpg', 'THE ZIGGURAT', 4.0, 'fadeblack'),
]

# Parameters
crossfade_duration = 0.5
target_duration = 55.0
fps = 24

# Calculate and scale durations
raw_duration = sum(s[2] for s in scenes)
overlap_time = (len(scenes) - 1) * crossfade_duration
scale_factor = (target_duration + overlap_time) / raw_duration
scenes = [(s[0], s[1], s[2] * scale_factor, s[3]) for s in scenes]
final_duration = sum(s[2] for s in scenes) - overlap_time
print(f"Final duration: {final_duration:.1f}s")

# Build inputs
inputs = []
for scene in scenes:
    img_path = os.path.join(precision_dir, scene[0])
    inputs.extend(['-loop', '1', '-t', f'{scene[2]:.3f}', '-i', img_path])

filters = []

# Alternating zoom direction for variety
zoom_in = True

for i, scene in enumerate(scenes):
    img, title, duration, trans = scene
    
    # Animated zoom using scale with time expression (eval=frame for time vars)
    # Zoom from 100% to 104% (or reverse) over the clip duration
    if zoom_in:
        scale_expr = f"scale=w='1920*(1+0.04*t/{duration})':h='1080*(1+0.04*t/{duration})':eval=frame"
    else:
        scale_expr = f"scale=w='1920*(1.04-0.04*t/{duration})':h='1080*(1.04-0.04*t/{duration})':eval=frame"
    zoom_in = not zoom_in
    
    # Chain: scale input -> animated zoom -> crop to 1920x1080 -> vignette -> color -> format
    filter_chain = (
        f"[{i}]"
        f"scale=2000:1130,"  # Scale up slightly first for zoom headroom
        f"{scale_expr},"
        f"crop=1920:1080:(iw-1920)/2:(ih-1080)/2,"  # Crop to output size
        f"vignette=PI/5,"  # Subtle vignette
        f"eq=contrast=1.04:brightness=0.01:saturation=0.97,"  # Cinematic grade
        f"format=yuv420p"
    )
    
    if title:
        filter_chain += (
            f",drawtext=text='{title}':"
            f"fontfile='C\\:/Windows/Fonts/ariblk.ttf':"
            f"fontsize=70:fontcolor=white:"
            f"x=(w-text_w)/2:y=h-130:"
            f"borderw=3:bordercolor=black@0.6:"
            f"alpha='if(lt(t,0.4),t/0.4,if(lt(t,{duration-0.5}),1,(({duration}-t)/0.5)))'"
        )
    
    filters.append(f"{filter_chain}[s{i}]")

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

# Fade in/out
filters.append(f"[prefinal]fade=t=in:st=0:d=1.5,fade=t=out:st={final_duration-1.5}:d=1.5[outv]")

filter_str = ';'.join(filters)

output_file = os.path.join(output_dir, 'ziggurat-premium.mp4')
cmd = [
    'ffmpeg', '-y',
    *inputs,
    '-filter_complex', filter_str,
    '-map', '[outv]',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'slow',
    '-crf', '16',
    '-r', str(fps),
    '-movflags', '+faststart',
    output_file
]

print(f"\nGenerating premium video with zoom, vignette, color grading...")
print(f"Output: {output_file}")

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("\nVideo created successfully!")
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', output_file
    ], capture_output=True, text=True)
    print(f"Duration: {float(probe.stdout.strip()):.1f}s")
else:
    print(f"\nError:")
    print(result.stderr[-2000:])
