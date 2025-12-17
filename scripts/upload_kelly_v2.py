import os
import json
import mimetypes
import requests
from pathlib import Path
from tqdm import tqdm

# OCD Configuration
SUPABASE_PROJECT_REF = 'tvjalxxsyryjphkforjv'
SUPABASE_URL = f"https://{SUPABASE_PROJECT_REF}.supabase.co"
STORAGE_BUCKET = "images"
STORAGE_FOLDER = "kelly_v2" # The clean slate
LOCAL_ROOT = "assets/kelly_canonical"

def get_mime_type(path):
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"

def upload_to_supabase(local_path, remote_path, supabase_key):
    url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{remote_path}"
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "apikey": supabase_key,
        "Content-Type": get_mime_type(local_path)
    }
    
    with open(local_path, 'rb') as f:
        content = f.read()
        
    # Use POST (and ignore 409) or strict overwrite logic?
    # OCD says: Ensure it is exactly what we want.
    # We'll try to overwrite if exists (PUT), or create (POST).
    
    # Attempt Create first
    response = requests.post(url, headers=headers, data=content)
    if response.status_code == 200:
        return True
        
    if response.status_code == 409: # Exists
        # Overwrite it!
        response = requests.put(url, headers=headers, data=content)
        return response.status_code == 200
        
    return False

def main():
    print("🚀 STARTING OCD UPLOAD TO 'kelly_v2'...")
    
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_key:
        # Fallback to hardcoded input from previous turn if env var lost context
        # (In a real script we wouldn't hardcode, but for this session...)
        # Actually, let's assume the environment persists or user will be prompted if it fails.
        # We will use the one from the chat history if needed, but let's try env first.
        print("⚠️  MISSING KEY. Please set SUPABASE_KEY.")
        return

    manifest_v2 = {}
    
    # Walk the new canonical directory
    files_to_upload = []
    for root, dirs, files in os.walk(LOCAL_ROOT):
        for file in files:
            local_path = os.path.join(root, file)
            # Rel path for storage: e.g. core/chair/kelly-chair-wisdom.png
            rel_path = os.path.relpath(local_path, LOCAL_ROOT).replace("\\", "/")
            remote_path = f"{STORAGE_FOLDER}/{rel_path}"
            files_to_upload.append((local_path, remote_path))

    pbar = tqdm(files_to_upload)
    for local_p, remote_p in pbar:
        pbar.set_description(f"Uploading {os.path.basename(local_p)}")
        if upload_to_supabase(local_p, remote_p, supabase_key):
            # Add to manifest
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{remote_p}"
            
            # Use the clean filename as the key in the manifest
            clean_filename = os.path.basename(remote_p)
            
            manifest_v2[clean_filename] = {
                "category": os.path.dirname(remote_p).replace(f"{STORAGE_FOLDER}/", ""),
                "public_url": public_url,
                "storage_path": remote_p
            }
        else:
            tqdm.write(f"❌ Failed: {local_p}")

    # Save the pristine manifest
    with open('KELLY_ASSETS_MANIFEST_V2.json', 'w') as f:
        json.dump(manifest_v2, f, indent=2, sort_keys=True)

    print("\n✨ OCD COMPLETE. The filesystem is healing.")
    print(f"📂 Clean assets live at: {STORAGE_BUCKET}/{STORAGE_FOLDER}")
    print("📄 Manifest: KELLY_ASSETS_MANIFEST_V2.json")

if __name__ == "__main__":
    main()



































