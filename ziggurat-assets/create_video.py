"""
Generate a 55-second ziggurat showcase video with crossfade transitions.
"""
import os
import subprocess

# Define the sequence: color schemes with time-of-day progression
color_schemes = ['warm', 'gold', 'rainbow', 'cyan', 'cool', 'white', 'usa']
times_of_day = ['twilight', 'dusk', 'night', 'late-night']

precision_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets\precision'
output_dir = r'c:\Users\user\UI-TARS-desktop\ziggurat-assets'

# Build ordered image list
images = []

# Start with before image
images.append(os.path.join(precision_dir, 'before-1080p.jpg'))

# Add each color scheme cycling through times of day
for color in color_schemes:
    for time in times_of_day:
        img_path = os.path.join(precision_dir, f'{color}-{time}-1080p.jpg')
        if os.path.exists(img_path):
            images.append(img_path)

print(f"Total images: {len(images)}")

# Calculate timing: 55 seconds total, with crossfades
total_duration = 55
num_images = len(images)
crossfade_duration = 0.8  # seconds of crossfade between images

# With crossfades, effective duration per image
duration_per_image = (total_duration + (num_images - 1) * crossfade_duration) / num_images
print(f"Duration per image: {duration_per_image:.2f}s")
print(f"Crossfade duration: {crossfade_duration}s")

# Build ffmpeg filter for crossfades with scale to even dimensions
inputs = []
for i, img in enumerate(images):
    inputs.extend(['-loop', '1', '-t', str(duration_per_image), '-i', img])

# Build xfade filter chain with scale filters
# First scale each input to 1920x1056 (even height), then xfade
filter_parts = []

# Scale all inputs first
for i in range(len(images)):
    filter_parts.append(f'[{i}]scale=1920:1056:force_original_aspect_ratio=decrease,pad=1920:1056:(ow-iw)/2:(oh-ih)/2[s{i}]')

# Build xfade chain
prev_label = 's0'
offset = duration_per_image - crossfade_duration

for i in range(1, len(images)):
    next_label = f'v{i:02d}' if i < len(images) - 1 else 'outv'
    filter_parts.append(
        f'[{prev_label}][s{i}]xfade=transition=fade:duration={crossfade_duration}:offset={offset:.3f}[{next_label}]'
    )
    prev_label = next_label
    offset += duration_per_image - crossfade_duration

filter_str = ';'.join(filter_parts)

# Build ffmpeg command
output_file = os.path.join(output_dir, 'ziggurat-showcase.mp4')
cmd = [
    'ffmpeg', '-y',
    *inputs,
    '-filter_complex', filter_str,
    '-map', '[outv]',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'medium',
    '-crf', '18',
    '-r', '30',
    output_file
]

print(f"\nGenerating video...")
print(f"Output: {output_file}")

# Execute
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print("Video created successfully!")
    # Check file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
else:
    print(f"Error: {result.stderr}")
