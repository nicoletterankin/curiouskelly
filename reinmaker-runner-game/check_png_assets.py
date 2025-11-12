"""
Analyze all PNG files in the runner game to check:
1. Dimensions (should be reasonable for game)
2. Transparency/alpha channel (should have transparent backgrounds)
3. File sizes (should be optimized)

Then generate prompts for Kelly's running animation frames.
"""

from PIL import Image
import os
from pathlib import Path

def analyze_png(filepath):
    """Analyze a PNG file and return info."""
    try:
        img = Image.open(filepath)
        width, height = img.size
        
        # Check if has alpha channel
        has_alpha = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
        
        # Check file size
        file_size = os.path.getsize(filepath)
        
        # Analyze transparency
        transparent_pixels = 0
        total_pixels = width * height
        if has_alpha:
            if img.mode == 'RGBA':
                alpha_channel = img.split()[3]
                transparent_pixels = sum(1 for p in alpha_channel.getdata() if p < 128)
            elif img.mode == 'LA':
                alpha_channel = img.split()[1]
                transparent_pixels = sum(1 for p in alpha_channel.getdata() if p < 128)
        
        transparency_percent = (transparent_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        return {
            'width': width,
            'height': height,
            'has_alpha': has_alpha,
            'transparency_percent': transparency_percent,
            'file_size_kb': file_size / 1024,
            'mode': img.mode,
            'issues': []
        }
    except Exception as e:
        return {'error': str(e)}

def check_all_pngs():
    """Check all PNG files in public folder."""
    public_dir = Path('public')
    results = {}
    
    # Files to check
    files_to_check = [
        'player.png',
        'obstacle.png',
        'bg.png',
        'ground_tex.png',
        'ground_stripe.png',
        'favicon.png',
        'gameover_bg.png',
    ]
    
    # Check stones
    stones_dir = public_dir / 'stones'
    if stones_dir.exists():
        for stone_file in stones_dir.glob('*.png'):
            files_to_check.append(f'stones/{stone_file.name}')
    
    # Check banners
    banners_dir = public_dir / 'banners'
    if banners_dir.exists():
        for banner_file in banners_dir.glob('*.png'):
            files_to_check.append(f'banners/{banner_file.name}')
    
    print("=" * 70)
    print("PNG FILE ANALYSIS - Runner Game Assets")
    print("=" * 70)
    print()
    
    issues_found = []
    
    for filepath in files_to_check:
        full_path = public_dir / filepath
        if not full_path.exists():
            print(f"⚠️  MISSING: {filepath}")
            issues_found.append(f"Missing: {filepath}")
            continue
        
        info = analyze_png(full_path)
        
        if 'error' in info:
            print(f"❌ ERROR: {filepath} - {info['error']}")
            issues_found.append(f"Error analyzing {filepath}: {info['error']}")
            continue
        
        # Check for issues
        status = "✅"
        if not info['has_alpha']:
            # Some files don't need alpha (bg, ground_tex)
            if 'bg' not in filepath and 'ground_tex' not in filepath and 'ground_stripe' not in filepath:
                status = "⚠️"
                info['issues'].append("No alpha channel - may have opaque background")
        
        if info['file_size_kb'] > 500:
            status = "⚠️"
            info['issues'].append(f"Large file size: {info['file_size_kb']:.1f} KB")
        
        if info['transparency_percent'] > 50 and 'player' in filepath:
            info['issues'].append(f"High transparency: {info['transparency_percent']:.1f}% - may need tighter bounds")
        
        print(f"{status} {filepath}")
        print(f"   Size: {info['width']}x{info['height']}px | Mode: {info['mode']} | Alpha: {info['has_alpha']}")
        print(f"   File: {info['file_size_kb']:.1f} KB | Transparency: {info['transparency_percent']:.1f}%")
        
        if info['issues']:
            for issue in info['issues']:
                print(f"   ⚠️  {issue}")
                issues_found.append(f"{filepath}: {issue}")
        
        results[filepath] = info
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print("\n✅ All files look good!")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Player sprite: Should have transparent background, reasonable size (ideally 500-2000px tall)
2. Obstacles: Should have transparent background for clean edges
3. Stones: Should have transparent background, consistent size
4. Background: Can be opaque (no alpha needed)
5. Ground textures: Can be opaque (tiled seamlessly)
6. File sizes: Should be < 500KB each for web performance

If any files have opaque backgrounds where they shouldn't, they need to be:
- Re-exported with transparency
- Or processed to remove background (magic wand + delete)
""")
    
    return results

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    check_all_pngs()







