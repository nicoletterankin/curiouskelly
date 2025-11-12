"""
Quick asset generator for simple graphics that don't need AI
"""
from PIL import Image, ImageDraw, ImageFilter
import os

def create_ground_stripe():
    """Create 60x6px rounded rectangle stripe"""
    # Create image with transparency
    img = Image.new('RGBA', (60, 6), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle in off-white #F2F7FA
    draw.rounded_rectangle(
        [(0, 0), (59, 5)],
        radius=3,
        fill=(242, 247, 250, 255)
    )
    
    img.save('assets/ground_stripe.png')
    print("✅ Created ground_stripe.png (60x6px)")

def create_ground_texture():
    """Create 512x64px seamless dark steel-stone texture"""
    import random
    
    # Base dark steel color
    img = Image.new('RGBA', (512, 64), (27, 30, 34, 255))
    pixels = img.load()
    
    # Add subtle noise/specks
    for x in range(512):
        for y in range(64):
            if random.random() < 0.03:  # 3% chance of speck
                brightness = random.randint(35, 50)
                pixels[x, y] = (brightness, brightness, brightness, 255)
    
    # Slight blur to make it less harsh
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    img.save('assets/ground_tex.png')
    print("✅ Created ground_tex.png (512x64px seamless)")

def create_logo_square():
    """Resize existing 1280x720 logo to 600x600 square"""
    if not os.path.exists('marketing/cover-1280x720.png'):
        print("⚠️  Source logo not found")
        return
    
    # Load existing logo
    img = Image.open('marketing/cover-1280x720.png')
    
    # Create 600x600 canvas
    square = Image.new('RGBA', (600, 600), (0, 0, 0, 0))
    
    # Resize to fit in 600x600 while maintaining aspect ratio
    img.thumbnail((600, 600), Image.Resampling.LANCZOS)
    
    # Center it
    x = (600 - img.width) // 2
    y = (600 - img.height) // 2
    square.paste(img, (x, y), img if img.mode == 'RGBA' else None)
    
    square.save('marketing/square-600.png')
    print("✅ Created square-600.png (600x600px)")

if __name__ == '__main__':
    print("🎨 Generating simple assets...\n")
    
    create_ground_stripe()
    create_ground_texture()
    create_logo_square()
    
    print("\n✅ Done! 3 simple assets created.")
    print("\n⏭️  Still need: 3 tribe banners (code, fire, metal)")







