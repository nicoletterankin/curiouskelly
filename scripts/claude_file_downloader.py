#!/usr/bin/env python3
"""
Claude.ai File Downloader
Automatically downloads lesson files created by Claude and saves them to the codebase.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_BASE = "https://api.anthropic.com/v1"
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", "lesson-player"))
STATE_FILE = Path(".claude_download_state.json")

# File patterns to download
LESSON_FILE_PATTERNS = [
    "*-dna.json",
    "*-visual-prompts.json",
    "*-knowledge-base.md",
    "*-asset-manifest.json",
    "*-teaching-moments.json",
    "*-interactive-specs.json",
    "*-animation-sequences.json",
    "*-export-package.md",
]


def load_state() -> Dict:
    """Load download state to track which files have been downloaded."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"downloaded_files": [], "last_check": None}


def save_state(state: Dict):
    """Save download state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_project_files(project_id: str) -> List[Dict]:
    """
    Get list of files from Claude.ai project.
    
    Note: This is a placeholder - Claude API may require different endpoints.
    You may need to use Claude's Messages API with project context instead.
    """
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    # TODO: Replace with actual Claude API endpoint for project files
    # This may require using the Messages API with project context
    # or a different API endpoint if available
    
    # For now, this is a template - you'll need to adapt based on Claude's actual API
    try:
        # Example API call (adjust based on Claude's actual API)
        response = requests.get(
            f"{CLAUDE_API_BASE}/projects/{project_id}/files",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("files", [])
        else:
            print(f"⚠️ API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"❌ Error fetching files: {e}")
        return []


def download_file(file_id: str, file_name: str, output_path: Path) -> bool:
    """Download a file from Claude.ai."""
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    
    try:
        # TODO: Replace with actual Claude API endpoint for file download
        response = requests.get(
            f"{CLAUDE_API_BASE}/files/{file_id}/content",
            headers=headers,
            stream=True,
            timeout=60
        )
        
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✅ Downloaded: {file_name}")
            return True
        else:
            print(f"  ❌ Download error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Download exception: {e}")
        return False


def is_lesson_file(filename: str) -> bool:
    """Check if file matches lesson file patterns."""
    import fnmatch
    for pattern in LESSON_FILE_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


def main():
    """Main download function."""
    if not CLAUDE_API_KEY:
        print("❌ Error: CLAUDE_API_KEY environment variable not set")
        print("   Set it with: export CLAUDE_API_KEY=your_key_here")
        return
    
    project_id = os.getenv("CLAUDE_PROJECT_ID", "the-daily-lesson")
    
    print(f"🔍 Checking Claude.ai project: {project_id}")
    print(f"📁 Output directory: {OUTPUT_DIRECTORY.absolute()}")
    
    # Load state
    state = load_state()
    downloaded_files = set(state.get("downloaded_files", []))
    
    # Get files from Claude project
    files = get_project_files(project_id)
    
    if not files:
        print("⚠️ No files found or API error. Check your API key and project ID.")
        print("\n💡 Note: Claude API may require different endpoints.")
        print("   You may need to:")
        print("   1. Use Claude's Messages API to list files")
        print("   2. Use Claude's project-specific endpoints")
        print("   3. Check Claude API documentation for file access")
        return
    
    new_files = []
    for file_info in files:
        file_name = file_info.get("name", "")
        file_id = file_info.get("id", "")
        
        # Skip if not a lesson file
        if not is_lesson_file(file_name):
            continue
        
        # Skip if already downloaded
        if file_id in downloaded_files:
            continue
        
        new_files.append((file_id, file_name))
    
    if not new_files:
        print("✅ No new lesson files to download")
        return
    
    print(f"📥 Found {len(new_files)} new lesson file(s) to download:")
    
    # Download new files
    for file_id, file_name in new_files:
        output_path = OUTPUT_DIRECTORY / file_name
        print(f"\n📥 Downloading: {file_name}")
        
        if download_file(file_id, file_name, output_path):
            downloaded_files.add(file_id)
            state["downloaded_files"] = list(downloaded_files)
            state["last_check"] = datetime.now().isoformat()
            save_state(state)
        else:
            print(f"  ⚠️ Failed to download {file_name}")
    
    print(f"\n✅ Download complete! Files saved to: {OUTPUT_DIRECTORY.absolute()}")


if __name__ == "__main__":
    main()

