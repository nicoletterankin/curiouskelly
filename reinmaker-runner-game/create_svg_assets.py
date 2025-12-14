import os

def create_svg(filename, content, width=64, height=64):
    svg_template = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">{content}</svg>'
    with open(filename, 'w') as f:
        f.write(svg_template)
    print(f"Created {filename}")

# Ensure directory exists
output_dir = "reinmaker-runner-game/public"
os.makedirs(output_dir, exist_ok=True)

# 1. Player (Kelly Avatar) - Stylized
player_svg = '''
<circle cx="32" cy="32" r="28" fill="#d97757" stroke="#fff" stroke-width="2"/>
<circle cx="32" cy="25" r="12" fill="#202023"/>
<path d="M12 52 Q32 64 52 52 L52 64 L12 64 Z" fill="#202023"/>
<!-- Hair accent -->
<path d="M10 30 Q32 10 54 30" fill="none" stroke="#fff" stroke-width="3"/>
'''
create_svg(f"{output_dir}/player.svg", player_svg)

# 2. Atom (Collectibles) - Replacing Stones
atom_colors = {
    'light': '#FFE066', 'stone': '#8E9AAF', 'metal': '#adb5bd', 
    'code': '#0BB39C', 'air': '#84C0C6', 'water': '#4dabf7', 'fire': '#F25F5C'
}

for name, color in atom_colors.items():
    atom_svg = f'''
    <circle cx="32" cy="32" r="8" fill="{color}"/>
    <ellipse cx="32" cy="32" rx="28" ry="10" fill="none" stroke="{color}" stroke-width="2" transform="rotate(0, 32, 32)"/>
    <ellipse cx="32" cy="32" rx="28" ry="10" fill="none" stroke="{color}" stroke-width="2" transform="rotate(60, 32, 32)"/>
    <ellipse cx="32" cy="32" rx="28" ry="10" fill="none" stroke="{color}" stroke-width="2" transform="rotate(120, 32, 32)"/>
    '''
    # Ensure stones directory exists
    os.makedirs(f"{output_dir}/stones", exist_ok=True)
    create_svg(f"{output_dir}/stones/atom_{name}.svg", atom_svg)

# 3. Obstacle (Glitch Block)
obstacle_svg = '''
<defs>
<linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#ff4d4d;stop-opacity:1" />
<stop offset="100%" style="stop-color:#990000;stop-opacity:1" />
</linearGradient>
</defs>
<rect x="4" y="4" width="56" height="56" fill="url(#grad1)" stroke="#fff" stroke-width="2"/>
<path d="M10 10 L54 54 M54 10 L10 54" stroke="#202023" stroke-width="4"/>
'''
create_svg(f"{output_dir}/obstacle.svg", obstacle_svg)

# 4. Background (Simple Gradient) - 800x600 logic but simple tile
bg_svg = '''
<rect width="800" height="600" fill="#0B1020"/>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
<path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="1"/>
</pattern>
<rect width="800" height="600" fill="url(#grid)" />
<circle cx="100" cy="100" r="2" fill="#fff" opacity="0.5"/>
<circle cx="300" cy="50" r="1" fill="#fff" opacity="0.8"/>
<circle cx="600" cy="200" r="3" fill="#fff" opacity="0.3"/>
<circle cx="700" cy="500" r="2" fill="#fff" opacity="0.6"/>
'''
create_svg(f"{output_dir}/bg.svg", bg_svg, width=800, height=600)

# 5. Ground Texture
ground_svg = '''
<rect width="64" height="64" fill="#18181b"/>
<path d="M0 0 L64 0" stroke="#d97757" stroke-width="4"/>
<line x1="32" y1="0" x2="32" y2="64" stroke="#27272a" stroke-width="2"/>
'''
create_svg(f"{output_dir}/ground_tex.svg", ground_svg)





























