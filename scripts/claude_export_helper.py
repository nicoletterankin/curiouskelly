#!/usr/bin/env python3
"""
Claude Export Helper
Helps automate the process of extracting files from Claude.ai conversations.
This script can be used with Claude's export feature or manual copy-paste.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

OUTPUT_DIRECTORY = Path("lesson-player")
STATE_FILE = Path(".claude_download_state.json")

# File patterns to extract
LESSON_FILE_PATTERNS = [
    r"```json:([\w-]+-dna\.json)",
    r"```([\w-]+-dna\.json)",
    r"```json\n([\w-]+-dna\.json)",
    r"File: ([\w-]+-dna\.json)",
]


def load_state() -> Dict:
    """Load download state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"processed_files": [], "last_check": None}


def save_state(state: Dict):
    """Save download state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def extract_json_blocks(text: str) -> List[Dict]:
    """Extract JSON code blocks from text."""
    files = []
    
    # Pattern 1: ```json:filename.json\n{...}\n```
    pattern1 = r'```json:([^\n]+)\n(.*?)\n```'
    matches = re.finditer(pattern1, text, re.DOTALL)
    for match in matches:
        filename = match.group(1)
        content = match.group(2)
        files.append({"name": filename, "content": content, "type": "json"})
    
    # Pattern 2: ```json\n{...}\n``` (with filename in comment or header)
    pattern2 = r'```json\n(.*?)\n```'
    matches = re.finditer(pattern2, text, re.DOTALL)
    for match in matches:
        content = match.group(1)
        # Try to extract filename from content or context
        filename_match = re.search(r'([\w-]+-(?:dna|visual-prompts|asset-manifest|teaching-moments|interactive-specs|animation-sequences|export-package)\.(?:json|md))', text[:match.start()] + text[match.end():])
        if filename_match:
            filename = filename_match.group(1)
        else:
            # Default filename based on content structure
            if '"id"' in content:
                try:
                    data = json.loads(content)
                    lesson_id = data.get("id", "unknown")
                    filename = f"{lesson_id}-dna.json"
                except:
                    filename = "unknown-dna.json"
            else:
                filename = "unknown.json"
        files.append({"name": filename, "content": content, "type": "json"})
    
    # Pattern 3: Markdown files
    pattern3 = r'```markdown:([^\n]+)\n(.*?)\n```'
    matches = re.finditer(pattern3, text, re.DOTALL)
    for match in matches:
        filename = match.group(1)
        content = match.group(2)
        files.append({"name": filename, "content": content, "type": "markdown"})
    
    return files


def save_file(file_info: Dict):
    """Save extracted file to disk."""
    filename = file_info["name"]
    content = file_info["content"]
    file_type = file_info["type"]
    
    output_path = OUTPUT_DIRECTORY / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_type == "json":
        try:
            # Validate and format JSON
            json_obj = json.loads(content)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
            print(f"  ✅ Saved JSON: {filename}")
        except json.JSONDecodeError as e:
            print(f"  ⚠️ Invalid JSON in {filename}: {e}")
            # Save as-is anyway
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
    else:
        # Save markdown or other text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ Saved: {filename}")


def process_claude_export(export_file: Path):
    """Process a Claude conversation export file."""
    print(f"📥 Processing export file: {export_file}")
    
    with open(export_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract files from export
    files = extract_json_blocks(content)
    
    if not files:
        print("⚠️ No lesson files found in export")
        return
    
    print(f"📋 Found {len(files)} file(s) to extract:\n")
    
    # Load state
    state = load_state()
    processed_files = set(state.get("processed_files", []))
    
    # Save new files
    for file_info in files:
        filename = file_info["name"]
        
        if filename in processed_files:
            print(f"  ⏭️ Skipping (already processed): {filename}")
            continue
        
        save_file(file_info)
        processed_files.add(filename)
    
    # Update state
    state["processed_files"] = list(processed_files)
    state["last_check"] = datetime.now().isoformat()
    save_state(state)
    
    print(f"\n✅ Processing complete! Files saved to: {OUTPUT_DIRECTORY.absolute()}")


def process_text_input(text: str):
    """Process text input (e.g., from clipboard or manual paste)."""
    print("📥 Processing text input...")
    
    # Extract files from text
    files = extract_json_blocks(text)
    
    if not files:
        print("⚠️ No lesson files found in text")
        return
    
    print(f"📋 Found {len(files)} file(s) to extract:\n")
    
    # Save files
    for file_info in files:
        save_file(file_info)
    
    print(f"\n✅ Processing complete! Files saved to: {OUTPUT_DIRECTORY.absolute()}")


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        # Process file
        export_file = Path(sys.argv[1])
        if export_file.exists():
            process_claude_export(export_file)
        else:
            print(f"❌ File not found: {export_file}")
    else:
        # Interactive mode: process clipboard or stdin
        print("📋 Claude Export Helper")
        print("=" * 50)
        print("\nUsage:")
        print("  1. Process export file: python claude_export_helper.py <export_file.txt>")
        print("  2. Paste text: python claude_export_helper.py < <pasted_text.txt>")
        print("  3. Interactive: Paste text below, then press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows)")
        print("\nPaste Claude conversation text below:\n")
        
        try:
            text = sys.stdin.read()
            if text.strip():
                process_text_input(text)
            else:
                print("⚠️ No text provided")
        except KeyboardInterrupt:
            print("\n\n⚠️ Cancelled")
        except EOFError:
            print("\n\n✅ Processing complete")


if __name__ == "__main__":
    main()

