"""
Create INVESTOR-CLASS 55-second coalition video
Old Money Aesthetic: Black, White, Kelly Blue (#3B82F6) only
NO ORANGE/WARM TONES
Includes Kelly's face as the teacher for 8 billion
"""
import os
import subprocess
import shutil

# Paths
precision_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets\precision'
kelly_img = r'c:\Users\user\UI-TARS-desktop\generated-images\kelly-archetypes-clean\default_kelly.png'
output_dir = r'c:\Users\user\UI-TARS-desktop\public\coalition'

# Copy Kelly image to working directory
kelly_work = os.path.join(output_dir, 'kelly-face.png')
shutil.copy(kelly_img, kelly_work)

# OLD MONEY PALETTE - Blue tones only (cyan, cool, white)
# NO warm, gold, rainbow, usa
scenes = [
    # (image, title, subtitle, duration, transition)
    
    # Opening - dramatic black with Kelly
    ('KELLY', 'KELLY', 'The Teacher for 8 Billion Humans', 5.0, 'fade'),
    
    # Building reveal - cool tones only
    ('white-twilight-1080p.jpg', '', '', 2.5, 'fadeblack'),
    ('white-dusk-1080p.jpg', 'THE ZIGGURAT', '1,003,041 Square Feet', 3.0, 'fade'),
    ('white-night-1080p.jpg', '', '', 2.0, 'dissolve'),
    ('white-late-night-1080p.jpg', '', '', 2.0, 'fade'),
    
    # Cool sequence
    ('cool-twilight-1080p.jpg', '', '', 2.0, 'fadeblack'),
    ('cool-dusk-1080p.jpg', 'LAGUNA NIGUEL', 'California', 2.5, 'fade'),
    ('cool-night-1080p.jpg', '', '', 2.0, 'wipeleft'),
    ('cool-late-night-1080p.jpg', '', '', 2.0, 'fade'),
    
    # Cyan - tech future
    ('cyan-twilight-1080p.jpg', '', '', 2.0, 'fadeblack'),
    ('cyan-dusk-1080p.jpg', 'COALITION MODEL', '$309,000,000', 3.0, 'fade'),
    ('cyan-night-1080p.jpg', '', '', 2.0, 'dissolve'),
    ('cyan-late-night-1080p.jpg', '', '', 2.0, 'fade'),
    
    # Return to white - purity
    ('white-twilight-1080p.jpg', '', '', 2.0, 'fadeblack'),
    ('white-dusk-1080p.jpg', '8 BILLION', 'Learners', 3.0, 'fade'),
    ('white-night-1080p.jpg', '', '', 2.0, 'wiperight'),
    
    # Finale with Kelly
    ('KELLY', 'VICTORY', 'of the People', 4.0, 'fadeblack'),
    
    # Final building shot
    ('white-late-night-1080p.jpg', 'LESSON OF THE DAY', 'thedailylesson.com', 4.0, 'fade'),
]

# Parameters
crossfade_duration = 0.6
target_duration = 55.0
fps = 24

# Calculate and scale durations
raw_duration = sum(s[3] for s in scenes)
overlap_time = (len(scenes) - 1) * crossfade_duration
scale_factor = (target_duration + overlap_time) / raw_duration
scenes = [(s[0], s[1], s[2], s[3] * scale_factor, s[4]) for s in scenes]
final_duration = sum(s[3] for s in scenes) - overlap_time
print(f"Final duration: {final_duration:.1f}s")

# Build inputs - handling Kelly specially
inputs = []
for scene in scenes:
    img_file, title, subtitle, duration, trans = scene
    if img_file == 'KELLY':
        img_path = kelly_work
    else:
        img_path = os.path.join(precision_dir, img_file)
    inputs.extend(['-loop', '1', '-t', f'{duration:.3f}', '-i', img_path])

filters = []
zoom_in = True

for i, scene in enumerate(scenes):
    img_file, title, subtitle, duration, trans = scene
    is_kelly = (img_file == 'KELLY')
    
    # Animated zoom
    if zoom_in:
        scale_expr = f"scale=w='1920*(1+0.03*t/{duration})':h='1080*(1+0.03*t/{duration})':eval=frame"
    else:
        scale_expr = f"scale=w='1920*(1.03-0.03*t/{duration})':h='1080*(1.03-0.03*t/{duration})':eval=frame"
    zoom_in = not zoom_in
    
    if is_kelly:
        # Kelly: centered on dark background with blue accent glow
        filter_chain = (
            f"[{i}]"
            f"scale=800:-1,"  # Scale Kelly to reasonable size
            f"pad=1920:1080:(1920-iw)/2:(1080-ih)/2:color=0x0a0a0a,"  # Center on dark bg
            f"format=yuv420p"
        )
    else:
        # Building shots: scale, zoom, crop, desaturate slightly for old money look
        filter_chain = (
            f"[{i}]"
            f"scale=2000:1130,"
            f"{scale_expr},"
            f"crop=1920:1080:(iw-1920)/2:(ih-1080)/2,"
            # Desaturate and add blue tint for old money
            f"eq=contrast=1.06:brightness=-0.02:saturation=0.7,"
            f"colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rm=-0.1:gm=-0.05:bm=0.1,"
            f"vignette=PI/4,"
            f"format=yuv420p"
        )
    
    # Add title and subtitle if present
    if title:
        # Old money typography - centered, elegant
        filter_chain += (
            f",drawtext=text='{title}':"
            f"fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"fontsize=90:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-40:"
            f"alpha='if(lt(t,0.5),t/0.5,if(lt(t,{duration-0.6}),1,(({duration}-t)/0.6)))'"
        )
    if subtitle:
        # Subtitle in Kelly blue
        filter_chain += (
            f",drawtext=text='{subtitle}':"
            f"fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"fontsize=36:fontcolor=0x3B82F6:"
            f"x=(w-text_w)/2:y=(h/2)+50:"
            f"alpha='if(lt(t,0.6),t/0.6,if(lt(t,{duration-0.5}),1,(({duration}-t)/0.5)))'"
        )
    
    filters.append(f"{filter_chain}[s{i}]")

# Build xfade chain
prev_label = 's0'
offset = scenes[0][3] - crossfade_duration

for i in range(1, len(scenes)):
    trans_type = scenes[i][4]
    next_label = f'v{i:02d}' if i < len(scenes) - 1 else 'prefinal'
    filters.append(
        f'[{prev_label}][s{i}]xfade=transition={trans_type}:duration={crossfade_duration}:offset={offset:.3f}[{next_label}]'
    )
    prev_label = next_label
    offset += scenes[i][3] - crossfade_duration

# Fade in/out
filters.append(f"[prefinal]fade=t=in:st=0:d=2.0,fade=t=out:st={final_duration-2.0}:d=2.0[outv]")

filter_str = ';'.join(filters)

# Generate sophisticated ambient audio - lower, more gravitas
print("Generating investor-class ambient audio...")
audio_file = os.path.join(output_dir, 'investor-ambient.aac')
audio_cmd = [
    'ffmpeg', '-y',
    '-f', 'lavfi', '-i', f'sine=frequency=55:duration={int(final_duration)+1}',  # Deep bass A1
    '-f', 'lavfi', '-i', f'sine=frequency=82.5:duration={int(final_duration)+1}',  # E2 fifth
    '-f', 'lavfi', '-i', f'sine=frequency=110:duration={int(final_duration)+1}',  # A2 octave
    '-f', 'lavfi', '-i', f'anoisesrc=d={int(final_duration)+1}:c=pink:a=0.008',  # Subtle texture
    '-filter_complex',
    '[0]volume=0.12[a];[1]volume=0.08[b];[2]volume=0.06[c];[3]volume=0.4[d];'
    '[a][b][c][d]amix=inputs=4:duration=longest,'
    f'afade=t=in:st=0:d=4,afade=t=out:st={final_duration-4}:d=4,'
    'lowpass=f=400,highpass=f=30',  # More filtered, subtle
    '-c:a', 'aac', '-b:a', '128k',
    audio_file
]
subprocess.run(audio_cmd, capture_output=True)
print("Audio generated.")

# Build video
output_video = os.path.join(output_dir, 'ziggurat-investor.mp4')
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
    output_video
]

print(f"\nGenerating investor-class video...")
print(f"Output: {output_video}")

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("\nVideo generated successfully!")
    
    # Combine with audio
    final_output = os.path.join(output_dir, 'ziggurat-coalition.mp4')
    combine_cmd = [
        'ffmpeg', '-y',
        '-i', output_video,
        '-i', audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        '-movflags', '+faststart',
        final_output
    ]
    subprocess.run(combine_cmd, capture_output=True)
    
    size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"Final file: {final_output}")
    print(f"File size: {size_mb:.1f} MB")
    
    # Verify duration
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', final_output
    ], capture_output=True, text=True)
    print(f"Duration: {float(probe.stdout.strip()):.1f}s")
    
    # Clean up intermediate files
    os.remove(output_video)
    os.remove(audio_file)
    print("\nInvestor-class video complete.")
else:
    print(f"\nError:")
    print(result.stderr[-2500:])
