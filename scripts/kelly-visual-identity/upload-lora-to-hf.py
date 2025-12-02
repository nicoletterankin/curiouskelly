"""
Upload Kelly LoRA to Hugging Face for use with Replicate

This uploads your trained Civitai LoRA to HuggingFace so Replicate can access it.
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file

# Your LoRA file
LORA_PATH = r"C:\Users\user\UI-TARS-desktop\models\lora\Curious_Kelly.safetensors"
REPO_NAME = "curious-kelly-lora"

def main():
    print("🚀 Uploading Kelly LoRA to Hugging Face")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(LORA_PATH):
        print(f"❌ LoRA file not found: {LORA_PATH}")
        return
    
    file_size = os.path.getsize(LORA_PATH) / (1024 * 1024)
    print(f"📦 LoRA file: {LORA_PATH}")
    print(f"📏 Size: {file_size:.2f} MB")
    
    # Initialize API
    api = HfApi()
    
    # Check if logged in
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face!")
        print("\n📋 TO FIX THIS:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' access")
        print("3. Run: huggingface-cli login")
        print("4. Paste your token when prompted")
        print("\nThen run this script again.")
        return
    
    repo_id = f"{username}/{REPO_NAME}"
    
    # Create repo if it doesn't exist
    try:
        print(f"\n📁 Creating repo: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Repo creation: {e}")
    
    # Upload the LoRA file
    try:
        print(f"\n📤 Uploading LoRA (this may take a minute)...")
        upload_file(
            path_or_fileobj=LORA_PATH,
            path_in_repo="curious_kelly.safetensors",
            repo_id=repo_id,
            repo_type="model"
        )
        
        lora_url = f"https://huggingface.co/{repo_id}/resolve/main/curious_kelly.safetensors"
        
        print(f"\n✅ UPLOAD COMPLETE!")
        print(f"\n🔗 Your LoRA URL (use this with Replicate):")
        print(f"   {lora_url}")
        
        # Save URL to file for the generation script
        with open("lora_url.txt", "w") as f:
            f.write(lora_url)
        print(f"\n💾 URL saved to: lora_url.txt")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()

