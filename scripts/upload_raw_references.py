import os
import json
import mimetypes
import requests
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SUPABASE_PROJECT_REF = 'tvjalxxsyryjphkforjv'
SUPABASE_URL = f"https://{SUPABASE_PROJECT_REF}.supabase.co"
STORAGE_BUCKET = "images"
STORAGE_FOLDER = "kelly_v2/reference/raw"

# The specific raw/messy locations we found
RAW_LOCATIONS = [
    "projects/Kelly/Ref",
    "lesson-player"
]

# Extensions to look for
EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

def get_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"

def upload_to_supabase(local_path, remote_filename, supabase_key):
    url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{STORAGE_FOLDER}/{remote_filename}"
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
        "Content-Type": get_mime_type(local_path)
    }
    
    with open(local_path, 'rb') as f:
        content = f.read()
        
    response = requests.post(url, headers=headers, data=content)
    if response.status_code == 200:
        return True
    elif response.status_code == 409:
        # Overwrite
        requests.put(url, headers=headers, data=content)
        return True
    return False

def main():
    print("🕵️  HUNTING FOR RAW REFERENCES...")
    
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_key:
        print("⚠️  MISSING KEY. Please set SUPABASE_KEY.")
        return

    files_to_upload = []
    
    # Scan specific directories
    for loc in RAW_LOCATIONS:
        if os.path.exists(loc):
            for root, _, files in os.walk(loc):
                for file in files:
                    if any(file.endswith(ext) for ext in EXTENSIONS):
                        # Filter out the ones we already standardized if they are duplicates
                        # But user said "look at your fake images", so we include lesson-player/*.png
                        
                        # Specific filter for lesson-player to only get the numbered ones or interesting ones
                        if "lesson-player" in loc and not (file[0].isdigit() or "kelly" in file.lower()):
                             continue

                        files_to_upload.append(os.path.join(root, file))

    print(f"Found {len(files_to_upload)} potential raw assets.")
    
    manifest_update = {}
    
    pbar = tqdm(files_to_upload)
    for local_path in pbar:
        filename = os.path.basename(local_path)
        # Clean up filename slightly but keep original identifier
        clean_name = filename.lower().replace(' ', '-')
        
        pbar.set_description(f"Uploading {filename}")
        
        if upload_to_supabase(local_path, clean_name, supabase_key):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{STORAGE_FOLDER}/{clean_name}"
            manifest_update[clean_name] = {
                "category": "reference/raw",
                "public_url": public_url,
                "original_path": local_path
            }

    # Merge with existing V2 manifest if exists
    if os.path.exists('KELLY_ASSETS_MANIFEST_V2.json'):
        with open('KELLY_ASSETS_MANIFEST_V2.json', 'r') as f:
            main_manifest = json.load(f)
    else:
        main_manifest = {}
        
    main_manifest.update(manifest_update)
    
    with open('KELLY_ASSETS_MANIFEST_V2.json', 'w') as f:
        json.dump(main_manifest, f, indent=2, sort_keys=True)

    print("\n✅ Raw references uploaded.")
    print(f"👉 Added {len(manifest_update)} files to 'reference/raw'")

if __name__ == "__main__":
    main()


































