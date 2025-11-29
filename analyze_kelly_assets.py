import json
import os
import re
from pathlib import Path

# Load the manifest
with open('KELLY_ASSETS_MANIFEST.json', 'r') as f:
    manifest = json.load(f)

# OCD Organization Schema
# category -> subcategory -> naming_convention
STRUCTURE = {
    "core": {
        "chair": [],      # The main lesson avatar state
        "portrait": [],   # Headshots/Zoom levels
    },
    "reference": {
        "identity": [],   # The 'source of truth' images (front, side, 3/4)
        "texture": [],    # Hair noise, physics maps
    },
    "marketing": {
        "age_variants": [],
        "seasonal": [],
    },
    "junk_drawer": []     # Files that look messy or duplicate
}

def clean_name(name):
    # Remove extension
    base = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1].lower()
    
    # Lowercase, replace spaces/underscores with hyphens
    clean = base.lower().replace('_', '-').replace(' ', '-')
    
    # Remove common redundant prefixes if we are reorganizing
    clean = clean.replace('kelly-', '')
    
    # Remove version numbers like v001 if they aren't final
    clean = re.sub(r'-v\d{3}$', '', clean)
    
    return f"kelly-{clean}{ext}"

renaming_plan = []

print("🧐 ANALYZING KELLY ASSETS WITH EXTREME PREJUDICE...\n")

for original_name, data in manifest.items():
    path = data['local_path'].replace('\\', '/')
    
    new_category = "junk_drawer"
    new_name = clean_name(original_name)
    
    # CLASSIFICATION LOGIC
    
    # 1. Core Lesson Assets (Director's Chair)
    if "directors-chair" in original_name:
        new_category = "core/chair"
        # simplify name: kelly-directors-chair-wisdom.png -> kelly-chair-wisdom.png
        new_name = new_name.replace('directors-chair', 'chair')
        
    # 2. Reference / Identity
    elif "Ref" in path or "identity" in original_name or original_name in ['kelly_front.png', 'kelly_profile.png', 'kelly_three_quarter.png']:
        new_category = "reference/identity"
        if "contact_sheet" in original_name:
            new_name = "kelly-ref-contact-sheet.png"
        elif original_name == "kelly_front.png":
            new_name = "kelly-ref-front-standard.png"
            
    # 3. Textures / Technical
    elif "Hair" in path or "Noise" in original_name:
        new_category = "reference/texture"
        
    # 4. Age Variants (Marketing/Dev)
    elif "age_" in original_name:
        new_category = "marketing/age_variants"
        # Clean up the massive verbose names
        # e.g. kelly_age_15_close_up_portrait_front_studio_neutral_16x9.png
        # -> kelly-age15-portrait-16x9.png
        
        match = re.search(r'age_(\d+)', original_name)
        age = match.group(1) if match else "X"
        
        if "close_up" in original_name:
            view = "closeup"
        elif "full_body" in original_name:
            view = "fullbody"
        elif "upper_body" in original_name:
            view = "upperbody"
        elif "front_facing" in original_name:
            view = "front-lean"
        else:
            view = "variant"
            
        dims = "1x1"
        if "16x9" in original_name: dims = "16x9"
        if "3x4" in original_name: dims = "3x4"
        
        new_name = f"kelly-age{age}-{view}-{dims}.png"

    # 5. Seasonal / Specific Marketing
    elif "christmas" in original_name:
        new_category = "marketing/seasonal"
        
    # 6. The "Messy" ones
    elif "UI elements" in original_name:
        new_category = "junk_drawer"
        new_name = "REVIEW-kelly-chair-ui-mockup.png"
    elif original_name == "curious kelly.PNG":
        new_category = "junk_drawer" # Likely duplicate
        new_name = "REVIEW-kelly-portrait-legacy.png"
        
    renaming_plan.append({
        "original_path": path,
        "original_name": original_name,
        "new_category": new_category,
        "new_name": new_name,
        "public_url": data['public_url']
    })

# Sort by category
renaming_plan.sort(key=lambda x: x['new_category'])

# Output the Plan
current_cat = ""
for item in renaming_plan:
    if item['new_category'] != current_cat:
        print(f"\n📂 {item['new_category'].upper()}")
        current_cat = item['new_category']
    
    if item['new_category'] == "junk_drawer":
        print(f"   ⚠️  {item['original_name']} -> {item['new_name']}")
    else:
        print(f"   ✅ {item['original_name']} -> {item['new_name']}")

# Generate a script to physically move these if approved
with open('organize_kelly.py', 'w') as f:
    f.write('import os\nimport shutil\nfrom pathlib import Path\n\n')
    f.write('BASE_DIR = "assets/kelly_canonical"\n')
    f.write('print("Creating canonical directory structure...")\n')
    f.write(f'PLAN = {json.dumps(renaming_plan, indent=2)}\n\n')
    f.write('''
for item in PLAN:
    dest_dir = os.path.join(BASE_DIR, item['new_category'])
    os.makedirs(dest_dir, exist_ok=True)
    
    src = item['original_path']
    dst = os.path.join(dest_dir, item['new_name'])
    
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {item['new_name']}")
        else:
            print(f"MISSING: {src}")
    except Exception as e:
        print(f"Error copying {src}: {e}")
''')













