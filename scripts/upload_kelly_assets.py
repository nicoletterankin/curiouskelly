#!/usr/bin/env python3
"""
Upload Kelly images to Supabase Storage.
"""
import os
import json
import mimetypes
import requests
from pathlib import Path
from tqdm import tqdm
import getpass

# Configuration
SUPABASE_PROJECT_REF = 'tvjalxxsyryjphkforjv'
SUPABASE_URL = f"https://{SUPABASE_PROJECT_REF}.supabase.co"
STORAGE_BUCKET = "images"
STORAGE_FOLDER = "kelly"

# Directories to scan (from scan_kelly_images.py)
SCAN_DIRS = [
    'lessons',
    'iLearnStudio/projects/Kelly',
    'digital-kelly/assets/images',
    'lesson-player',
    'projects/Kelly',
    'synthetic_tts',
    'public/images/kelly' # Added this one explicitly
]

EXCLUSIONS = [
    'digital-kelly/engines',
    'Library',
    'PackageCache',
    'node_modules',
    '.git',
    'backup',
    'test_comparison',
    'dist',
]

def is_kelly_image(path_str):
    """Check if image is related to Kelly"""
    path_lower = path_str.lower()
    return any(keyword in path_lower for keyword in ['kelly', 'curious'])

def should_exclude(path_str):
    """Check if path should be excluded"""
    return any(exclusion in path_str for exclusion in EXCLUSIONS)

def scan_images():
    """Scan for all Kelly images"""
    images = []
    root = Path('.')
    
    seen_paths = set()

    for scan_dir in SCAN_DIRS:
        scan_path = root / scan_dir
        if not scan_path.exists():
            continue
            
        for ext in ['png', 'jpg', 'jpeg', 'webp', 'PNG', 'JPG', 'JPEG', 'WEBP']:
            for img_path in scan_path.rglob(f'*.{ext}'):
                path_str = str(img_path)
                
                # Avoid duplicates if paths overlap or are symlinked/copied
                abs_path = img_path.resolve()
                if abs_path in seen_paths:
                    continue
                seen_paths.add(abs_path)

                if should_exclude(path_str):
                    continue
                    
                if is_kelly_image(path_str) or 'public/images/kelly' in path_str.replace('\\', '/'):
                     images.append(img_path)
    
    return images

def upload_file(file_path, supabase_key):
    """Upload a single file to Supabase Storage"""
    filename = file_path.name
    # Create a unique name if needed, but for now try to keep original name
    # If we have duplicates with same name but different content, we might need to handle it.
    # For now, let's use the filename.
    
    # Construct the target path in the bucket
    target_path = f"{STORAGE_FOLDER}/{filename}"
    
    url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{target_path}"
    
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
    }
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    
    headers["Content-Type"] = mime_type

    with open(file_path, 'rb') as f:
        file_content = f.read()

    # Upload (POST to create, PUT to update/overwrite is usually better if we want to update)
    # Supabase Storage standard upload is POST. Update is PUT.
    # We'll try POST first, if it exists (409?), we might just skip or PUT.
    # Actually, let's use POST (create).
    
    response = requests.post(url, headers=headers, data=file_content)
    
    if response.status_code == 200:
        return True, response.json()
    elif response.status_code == 409: # Already exists?
        return True, {"message": "Already exists", "Key": target_path} # Treat as success
    elif response.status_code == 404:
        error_json = {}
        try:
            error_json = response.json()
        except:
            pass
        if error_json.get('error') == 'Bucket not found':
             return False, "BUCKET_NOT_FOUND"
        return False, response.text
    else:
        return False, response.text

def get_public_url(filename):
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{STORAGE_FOLDER}/{filename}"

def main():
    print("🔍 Scanning for Kelly images...")
    images = scan_images()
    print(f"found {len(images)} images.")

    if not images:
        print("No images found to upload.")
        return

    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_key:
        print("\n🔑 Supabase Key not found in environment.")
        print("Please paste your Supabase 'anon' key below (hidden input):")
        try:
            supabase_key = getpass.getpass("Supabase Key: ").strip()
        except Exception:
            # Fallback for environments where getpass might fail (though unlikely in terminal)
            supabase_key = input("Supabase Key: ").strip()
    
    if not supabase_key:
        print("❌ No key provided. Exiting.")
        return

    print(f"\n🚀 Starting upload to bucket '{STORAGE_BUCKET}/{STORAGE_FOLDER}'...")
    
    manifest = {}
    success_count = 0
    
    pbar = tqdm(images, unit="img")
    bucket_error_shown = False

    for img_path in pbar:
        pbar.set_description(f"Uploading {img_path.name}")
        
        success, result = upload_file(img_path, supabase_key)
        
        if success:
            success_count += 1
            public_url = get_public_url(img_path.name)
            manifest[img_path.name] = {
                "local_path": str(img_path),
                "public_url": public_url,
                "supabase_key": result.get("Key", "")
            }
        else:
            if result == "BUCKET_NOT_FOUND":
                if not bucket_error_shown:
                    tqdm.write(f"\n❌ CRITICAL ERROR: The bucket '{STORAGE_BUCKET}' does not exist in Supabase.")
                    tqdm.write(f"👉 Please go to your Supabase Dashboard -> Storage -> New Bucket.")
                    tqdm.write(f"   Name it '{STORAGE_BUCKET}' and make sure it is PUBLIC.")
                    bucket_error_shown = True
                continue
            # Log error but continue
            tqdm.write(f"❌ Failed to upload {img_path.name}: {result}")

    print(f"\n✨ Upload complete!")
    print(f"✅ Successfully uploaded: {success_count}/{len(images)}")
    
    manifest_path = "KELLY_ASSETS_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"📄 Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()

