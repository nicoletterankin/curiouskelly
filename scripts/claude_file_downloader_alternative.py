#!/usr/bin/env python3
"""
Alternative Claude.ai File Downloader
Uses Claude Messages API to extract file content from conversations.
This approach works by asking Claude to list and provide file contents.
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_BASE = "https://api.anthropic.com/v1"
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", "lesson-player"))
STATE_FILE = Path(".claude_download_state.json")
PROJECT_ID = os.getenv("CLAUDE_PROJECT_ID", "the-daily-lesson")

# File patterns to download
LESSON_FILE_PATTERNS = [
    r".*-dna\.json",
    r".*-visual-prompts\.json",
    r".*-knowledge-base\.md",
    r".*-asset-manifest\.json",
    r".*-teaching-moments\.json",
    r".*-interactive-specs\.json",
    r".*-animation-sequences\.json",
    r".*-export-package\.md",
]


def load_state() -> Dict:
    """Load download state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"downloaded_files": [], "last_check": None}


def save_state(state: Dict):
    """Save download state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def ask_claude_for_files() -> str:
    """
    Ask Claude to list all lesson files in the project.
    Uses Claude Messages API to query the project.
    """
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    # Ask Claude to list files
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": "List all lesson files in this project. For each file, provide:\n"
                          "1. File name\n"
                          "2. File type (dna.json, visual-prompts.json, etc.)\n"
                          "3. Lesson ID (extracted from filename)\n\n"
                          "Format as JSON array with keys: name, type, lesson_id"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{CLAUDE_API_BASE}/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("content", [{}])[0].get("text", "")
        else:
            print(f"⚠️ API Error: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"❌ Error querying Claude: {e}")
        return ""


def ask_claude_for_file_content(file_name: str) -> Optional[str]:
    """
    Ask Claude to provide the content of a specific file.
    """
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": f"Provide the complete content of the file '{file_name}' from this project. "
                          f"Include the entire file content exactly as it exists."
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{CLAUDE_API_BASE}/messages",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("content", [{}])[0].get("text", "")
            
            # Try to extract JSON or markdown content
            # Look for code blocks
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                return json_match.group(1)
            
            # If no code block, return raw content
            return content
        else:
            print(f"  ⚠️ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ Error fetching file content: {e}")
        return None


def extract_files_from_response(response_text: str) -> List[Dict]:
    """Extract file information from Claude's response."""
    files = []
    
    # Try to parse as JSON
    try:
        # Look for JSON array in response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            files = json.loads(json_match.group(0))
            return files
    except:
        pass
    
    # Fallback: extract filenames from text
    for pattern in LESSON_FILE_PATTERNS:
        matches = re.findall(pattern, response_text)
        for match in matches:
            files.append({
                "name": match,
                "type": "unknown",
                "lesson_id": match.split("-")[0] if "-" in match else "unknown"
            })
    
    return files


def save_file_content(file_name: str, content: str):
    """Save file content to disk."""
    output_path = OUTPUT_DIRECTORY / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to format JSON if it's a JSON file
    if file_name.endswith(".json"):
        try:
            json_obj = json.loads(content)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
        except:
            # If not valid JSON, save as-is
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
    else:
        # Save markdown or other text files
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    print(f"  ✅ Saved: {file_name}")


def main():
    """Main download function."""
    if not CLAUDE_API_KEY:
        print("❌ Error: CLAUDE_API_KEY environment variable not set")
        print("   Set it with: export CLAUDE_API_KEY=your_key_here")
        return
    
    print(f"🔍 Checking Claude.ai project: {PROJECT_ID}")
    print(f"📁 Output directory: {OUTPUT_DIRECTORY.absolute()}")
    
    # Load state
    state = load_state()
    downloaded_files = set(state.get("downloaded_files", []))
    
    # Ask Claude for file list
    print("\n📋 Requesting file list from Claude...")
    response = ask_claude_for_files()
    
    if not response:
        print("⚠️ Could not get file list from Claude")
        return
    
    # Extract files from response
    files = extract_files_from_response(response)
    
    if not files:
        print("⚠️ No lesson files found in response")
        print(f"\n💡 Claude's response:\n{response[:500]}...")
        return
    
    # Filter new files
    new_files = [f for f in files if f["name"] not in downloaded_files]
    
    if not new_files:
        print("✅ No new lesson files to download")
        return
    
    print(f"📥 Found {len(new_files)} new lesson file(s) to download:\n")
    
    # Download new files
    for file_info in new_files:
        file_name = file_info["name"]
        print(f"📥 Downloading: {file_name}")
        
        content = ask_claude_for_file_content(file_name)
        
        if content:
            save_file_content(file_name, content)
            downloaded_files.add(file_name)
            state["downloaded_files"] = list(downloaded_files)
            state["last_check"] = datetime.now().isoformat()
            save_state(state)
        else:
            print(f"  ⚠️ Failed to get content for {file_name}")
    
    print(f"\n✅ Download complete! Files saved to: {OUTPUT_DIRECTORY.absolute()}")


if __name__ == "__main__":
    main()

