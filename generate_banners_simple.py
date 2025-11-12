"""
Generate simple placeholder tribe banners using PIL
We can replace with AI-generated versions later, but need SOMETHING to ship
"""
from PIL import Image, ImageDraw, ImageFont
import os

BANNERS = {
    "code": {
        "color": (11, 179, 156),  # #0BB39C teal
        "symbol": "</>",
        "symbol_emoji": "⟨⟩"
    },
    "fire": {
        "color": (242, 95, 92),  # #F25F5C red-orange
        "symbol": "🔥",
        "symbol_emoji": "▲"
    },
    "metal": {
        "color": (173, 181, 189),  # #adb5bd gray
        "symbol": "⚙",
        "symbol_emoji": "●"
    }
}

def create_banner(tribe_name, data):
    """Create a simple vertical banner"""
    width, height = 128, 256
    
    # Create image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw banner background with slight transparency
    r, g, b = data["color"]
    banner_color = (r, g, b, 230)
    
    # Draw main banner rectangle
    draw.rectangle([(10, 0), (118, 256)], fill=banner_color)
    
    # Draw gold trim on edges
    gold = (216, 162, 74, 255)  # #D8A24A
    draw.rectangle([(10, 0), (12, 256)], fill=gold)
    draw.rectangle([(116, 0), (118, 256)], fill=gold)
    draw.rectangle([(10, 0), (118, 8)], fill=gold)
    draw.rectangle([(10, 248), (118, 256)], fill=gold)
    
    # Draw symbol in center (using text)
    try:
        # Try to use a nice font
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Draw symbol text
    symbol = data["symbol_emoji"]
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw shadow
    draw.text((x+2, y+2), symbol, fill=(0, 0, 0, 100), font=font)
    # Draw symbol
    draw.text((x, y), symbol, fill=(255, 255, 255, 255), font=font)
    
    # Add subtle texture (noise)
    import random
    pixels = img.load()
    for i in range(width):
        for j in range(height):
            if pixels[i, j][3] > 0:  # Only on non-transparent pixels
                if random.random() < 0.05:
                    noise = random.randint(-10, 10)
                    r, g, b, a = pixels[i, j]
                    pixels[i, j] = (
                        max(0, min(255, r + noise)),
                        max(0, min(255, g + noise)),
                        max(0, min(255, b + noise)),
                        a
                    )
    
    # Save
    output_path = f"assets/banners/banner_{tribe_name}.png"
    img.save(output_path)
    print(f"✅ Created {tribe_name} banner: {output_path}")

def main():
    print("🎨 Generating simple placeholder tribe banners...\n")
    
    for tribe, data in BANNERS.items():
        create_banner(tribe, data)
    
    print("\n✅ All banners created!")
    print("💡 These are simple placeholders. Replace with AI-generated later.")
    print("\n🎉 ALL ASSETS NOW COMPLETE - READY TO BUILD THE GAME!")

if __name__ == "__main__":
    main()







