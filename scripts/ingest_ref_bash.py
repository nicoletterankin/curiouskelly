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
STORAGE_FOLDER = "kelly_v2/reference/bash_source" # Dedicated folder for the bash/mix

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
        requests.put(url, headers=headers, data=content)
        return True
    return False

def main():
    print("🧬 INGESTING REFERENCE BASH MATERIAL...")
    
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_key:
        print("⚠️  MISSING KEY.")
        return

    # Target the dropped folder
    TARGET_DIR = "assets/Ref"
    
    if not os.path.exists(TARGET_DIR):
        print(f"❌ Directory {TARGET_DIR} not found. Did you drag and drop it?")
        return

    files_to_upload = []
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                files_to_upload.append(os.path.join(root, file))

    print(f"Found {len(files_to_upload)} new reference candidates.")
    
    manifest_update = {}
    
    pbar = tqdm(files_to_upload)
    for local_path in pbar:
        # Keep folder structure in filename to avoid collisions
        # e.g. "Best Character Reference/1.png" -> "best-character-reference-1.png"
        rel_path = os.path.relpath(local_path, TARGET_DIR)
        clean_name = rel_path.replace('\\', '-').replace('/', '-').replace(' ', '-').lower()
        
        pbar.set_description(f"Uploading {clean_name}")
        
        if upload_to_supabase(local_path, clean_name, supabase_key):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{STORAGE_FOLDER}/{clean_name}"
            manifest_update[clean_name] = {
                "category": "reference/bash_source",
                "public_url": public_url,
                "original_path": local_path
            }

    # Merge with V2 Manifest
    manifest_path = 'KELLY_ASSETS_MANIFEST_V2.json'
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            main_manifest = json.load(f)
    else:
        main_manifest = {}
        
    main_manifest.update(manifest_update)
    
    with open(manifest_path, 'w') as f:
        json.dump(main_manifest, f, indent=2, sort_keys=True)

    print("\n✅ BASH MATERIAL INGESTED.")
    print(f"👉 Added {len(manifest_update)} high-value references to 'reference/bash_source'")

if __name__ == "__main__":
    main()





























