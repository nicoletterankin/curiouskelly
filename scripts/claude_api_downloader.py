#!/usr/bin/env python3
"""
Claude API File Downloader
Uses Anthropic Claude API to download lesson files from Claude.ai projects.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    from anthropic import Anthropic
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("⚠️ Anthropic SDK not installed. Install with: pip install anthropic")
    print("   Falling back to direct HTTP requests...")

import requests

# Configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
API_BASE = "https://api.anthropic.com/v1"
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", "lesson-player"))
STATE_FILE = Path(".claude_download_state.json")
PROJECT_ID = os.getenv("CLAUDE_PROJECT_ID", "the-daily-lesson")
MODEL = "claude-3-5-sonnet-20241022"

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


def ask_claude_sdk(prompt: str) -> Optional[str]:
    """Ask Claude using official SDK."""
    if not HAS_SDK:
        return None
    
    try:
        client = Anthropic(api_key=API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return message.content[0].text
    except Exception as e:
        print(f"❌ SDK Error: {e}")
        return None


def ask_claude_http(prompt: str) -> Optional[str]:
    """Ask Claude using direct HTTP requests."""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("content", [{}])[0].get("text", "")
        else:
            print(f"⚠️ API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ HTTP Error: {e}")
        return None


def ask_claude(prompt: str) -> Optional[str]:
    """Ask Claude (tries SDK first, falls back to HTTP)."""
    if HAS_SDK:
        result = ask_claude_sdk(prompt)
        if result:
            return result
    
    return ask_claude_http(prompt)


def get_file_list() -> List[Dict]:
    """Get list of lesson files from Claude project."""
    prompt = (
        f"List all lesson files in this project '{PROJECT_ID}'. "
        f"For each file matching these patterns: "
        f"*-dna.json, *-visual-prompts.json, *-knowledge-base.md, "
        f"*-asset-manifest.json, *-teaching-moments.json, "
        f"*-interactive-specs.json, *-animation-sequences.json, "
        f"*-export-package.md\n\n"
        f"Provide a JSON array with format:\n"
        f'[{{"name": "filename", "type": "dna.json|visual-prompts.json|etc", "lesson_id": "extracted-id"}}]'
    )
    
    response = ask_claude(prompt)
    if not response:
        return []
    
    # Try to extract JSON from response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Fallback: extract filenames manually
    files = []
    for pattern in LESSON_FILE_PATTERNS:
        matches = re.findall(pattern, response)
        for match in matches:
            files.append({
                "name": match,
                "type": "unknown",
                "lesson_id": match.split("-")[0] if "-" in match else "unknown"
            })
    
    return files


def get_file_content(file_name: str) -> Optional[str]:
    """Get content of a specific file from Claude."""
    prompt = (
        f"Provide the complete content of the file '{file_name}' from this project. "
        f"Include the entire file content exactly as it exists. "
        f"If it's JSON, provide valid JSON. If it's Markdown, provide the markdown content."
    )
    
    response = ask_claude(prompt)
    if not response:
        return None
    
    # Try to extract JSON code block
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to extract markdown code block
    md_match = re.search(r'```(?:markdown)?\s*\n(.*?)\n```', response, re.DOTALL)
    if md_match:
        return md_match.group(1)
    
    # Return raw content if no code blocks
    return response.strip()


def save_file(file_name: str, content: str):
    """Save file content to disk."""
    output_path = OUTPUT_DIRECTORY / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to format JSON if it's a JSON file
    if file_name.endswith(".json"):
        try:
            json_obj = json.loads(content)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved JSON: {file_name}")
        except json.JSONDecodeError:
            # If not valid JSON, save as-is
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ Saved (raw): {file_name}")
    else:
        # Save markdown or other text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ Saved: {file_name}")


def main():
    """Main download function."""
    if not API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY or CLAUDE_API_KEY not set")
        print("   Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print("   Or install SDK: pip install anthropic")
        return
    
    print(f"🔍 Claude API File Downloader")
    print(f"📁 Project: {PROJECT_ID}")
    print(f"📂 Output: {OUTPUT_DIRECTORY.absolute()}")
    print(f"🔧 Method: {'SDK' if HAS_SDK else 'HTTP'}\n")
    
    # Load state
    state = load_state()
    downloaded_files = set(state.get("downloaded_files", []))
    
    # Get file list
    print("📋 Requesting file list from Claude...")
    files = get_file_list()
    
    if not files:
        print("⚠️ No lesson files found")
        print("\n💡 This might mean:")
        print("   1. No files exist in the project yet")
        print("   2. Project ID is incorrect")
        print("   3. API key doesn't have access")
        return
    
    # Filter new files
    new_files = [f for f in files if f["name"] not in downloaded_files]
    
    if not new_files:
        print("✅ No new lesson files to download")
        return
    
    print(f"📥 Found {len(new_files)} new file(s) to download:\n")
    
    # Download new files
    for file_info in new_files:
        file_name = file_info["name"]
        print(f"📥 Downloading: {file_name}")
        
        content = get_file_content(file_name)
        
        if content:
            save_file(file_name, content)
            downloaded_files.add(file_name)
            state["downloaded_files"] = list(downloaded_files)
            state["last_check"] = datetime.now().isoformat()
            save_state(state)
        else:
            print(f"  ⚠️ Failed to get content for {file_name}")
    
    print(f"\n✅ Download complete! Files saved to: {OUTPUT_DIRECTORY.absolute()}")


if __name__ == "__main__":
    main()

