"""
Legacy DNA Files Cleanup Script
================================
Archives all DNA files and cleans up references to establish
Supabase as the single source of truth for lesson content.

Run: python scripts/cleanup_legacy_dna.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
LESSONS_DIR = PROJECT_ROOT / "lessons"
ARCHIVE_DIR = PROJECT_ROOT / "_archive" / "dna-legacy"
CALENDAR_FILE = LESSONS_DIR / "365_day_calendar.json"

# Patterns to identify DNA files
DNA_PATTERNS = ["*-dna.json", "*_dna.json", "*dna*.json"]

def find_all_dna_files():
    """Find all DNA files across the project."""
    dna_files = []
    
    # Search in lessons directory
    for pattern in DNA_PATTERNS:
        dna_files.extend(LESSONS_DIR.glob(pattern))
    
    # Search in lessons subdirectories
    for pattern in DNA_PATTERNS:
        dna_files.extend(LESSONS_DIR.glob(f"**/{pattern}"))
    
    # Search in daily-lesson-marketing
    marketing_lessons = PROJECT_ROOT / "daily-lesson-marketing" / "public" / "lessons"
    if marketing_lessons.exists():
        for pattern in DNA_PATTERNS:
            dna_files.extend(marketing_lessons.glob(pattern))
            dna_files.extend(marketing_lessons.glob(f"**/{pattern}"))
    
    # Search in public/lessons
    public_lessons = PROJECT_ROOT / "public" / "lessons"
    if public_lessons.exists():
        for pattern in DNA_PATTERNS:
            dna_files.extend(public_lessons.glob(pattern))
    
    # Deduplicate
    unique_files = list(set(dna_files))
    
    # Exclude already archived files
    unique_files = [f for f in unique_files if "_archive" not in str(f)]
    
    return sorted(unique_files)

def archive_dna_files(dna_files):
    """Move DNA files to archive directory."""
    print(f"\n📦 Archiving {len(dna_files)} DNA files...")
    
    # Create archive directory
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create README in archive
    readme_content = f"""# Legacy DNA Files Archive

**Archived:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Reason:** Supabase is now the single source of truth for lesson content

## Why These Were Archived

DNA files were an earlier content format containing:
- Age-variant lesson scripts
- Multilingual translations
- Voice profiles
- Interactive choices

However, this content was OUT OF SYNC with production Supabase data:
- DNA files: "The Sun", "Habit Stacking", "Planet Earth"
- Supabase: "Starting Fresh", "Three Lives of Water", "Where Clouds Come From"

## Current Architecture

```
SUPABASE (Single Source of Truth)
├── core_lessons (365 topics)
├── lesson_atoms (21,915 content pieces)
└── lesson_shards (38,700 demographic variants)
```

## If You Need This Content

The rich DNA structure (age variants, translations, voice profiles) can be 
regenerated FROM Supabase data if needed. The schema is preserved here for reference.

## Files Archived

"""
    for f in dna_files:
        readme_content += f"- {f.name}\n"
    
    (ARCHIVE_DIR / "README.md").write_text(readme_content, encoding="utf-8")
    
    # Move files
    moved = 0
    for dna_file in dna_files:
        try:
            dest = ARCHIVE_DIR / dna_file.name
            if dest.exists():
                # Add parent dir name to avoid conflicts
                dest = ARCHIVE_DIR / f"{dna_file.parent.name}_{dna_file.name}"
            shutil.move(str(dna_file), str(dest))
            print(f"   ✅ Archived: {dna_file.name}")
            moved += 1
        except Exception as e:
            print(f"   ❌ Failed to archive {dna_file.name}: {e}")
    
    return moved

def clean_calendar_json():
    """Remove DNA references from calendar JSON."""
    print(f"\n🧹 Cleaning calendar JSON...")
    
    if not CALENDAR_FILE.exists():
        print("   ⚠️ Calendar file not found")
        return 0
    
    with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
        calendar = json.load(f)
    
    cleaned = 0
    for lesson in calendar.get("lessons", []):
        # Remove DNA-related fields
        if "has_dna" in lesson:
            del lesson["has_dna"]
            cleaned += 1
        if "dna_file" in lesson:
            del lesson["dna_file"]
            cleaned += 1
    
    # Update metadata
    calendar["description"] = "365-day curriculum from Supabase (Single Source of Truth)"
    calendar["version"] = "3.0.0"
    calendar["cleanedAt"] = datetime.utcnow().isoformat() + "Z"
    calendar["note"] = "DNA files archived. Supabase is the sole content source."
    
    with open(CALENDAR_FILE, "w", encoding="utf-8") as f:
        json.dump(calendar, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Removed {cleaned} DNA references from calendar")
    return cleaned

def clean_duplicate_calendars():
    """Remove duplicate calendar files, keep only lessons/365_day_calendar.json."""
    print(f"\n🗑️ Cleaning duplicate calendar files...")
    
    # Files to potentially remove (we keep the primary one)
    duplicates = [
        PROJECT_ROOT / "public" / "data" / "365_day_calendar.json",
        PROJECT_ROOT / "public" / "data" / "calendar.json",
        PROJECT_ROOT / "daily-lesson-marketing" / "public" / "lessons" / "365_day_calendar.json",
    ]
    
    removed = 0
    for dup in duplicates:
        if dup.exists():
            try:
                dup.unlink()
                print(f"   ✅ Removed: {dup.relative_to(PROJECT_ROOT)}")
                removed += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {dup.name}: {e}")
    
    return removed

def clean_dna_documentation():
    """Archive DNA-related documentation files."""
    print(f"\n📄 Archiving DNA documentation...")
    
    docs_to_archive = [
        LESSONS_DIR / "DNA_FILES_LOCATION.md",
        LESSONS_DIR / "DNA_CONSOLIDATION_SUMMARY.md",
        LESSONS_DIR / "365_day_dna_metadata.json",
    ]
    
    archived = 0
    for doc in docs_to_archive:
        if doc.exists():
            dest = ARCHIVE_DIR / doc.name
            shutil.move(str(doc), str(dest))
            print(f"   ✅ Archived: {doc.name}")
            archived += 1
    
    return archived

def main():
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   🧹 LEGACY DNA FILES CLEANUP                                     ║")
    print("║   Establishing Supabase as Single Source of Truth                 ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    
    # Step 1: Find all DNA files
    print("🔍 Finding DNA files...")
    dna_files = find_all_dna_files()
    print(f"   Found {len(dna_files)} DNA files")
    
    if dna_files:
        for f in dna_files[:10]:
            print(f"   - {f.relative_to(PROJECT_ROOT)}")
        if len(dna_files) > 10:
            print(f"   ... and {len(dna_files) - 10} more")
    
    # Step 2: Archive DNA files
    archived = archive_dna_files(dna_files)
    
    # Step 3: Clean calendar JSON
    cleaned = clean_calendar_json()
    
    # Step 4: Remove duplicate calendars
    removed = clean_duplicate_calendars()
    
    # Step 5: Archive DNA documentation
    docs_archived = clean_dna_documentation()
    
    # Summary
    print("")
    print("=" * 60)
    print("📊 CLEANUP SUMMARY")
    print("=" * 60)
    print(f"   DNA files archived:     {archived}")
    print(f"   Calendar refs cleaned:  {cleaned}")
    print(f"   Duplicate files removed: {removed}")
    print(f"   Docs archived:          {docs_archived}")
    print(f"   Archive location:       _archive/dna-legacy/")
    print("=" * 60)
    print("")
    print("✅ CLEANUP COMPLETE!")
    print("")
    print("📌 NEXT: Supabase is now the SINGLE SOURCE OF TRUTH")
    print("   - lessons/365_day_calendar.json mirrors Supabase")
    print("   - Use scripts/sync_supabase_to_calendar.py to re-sync")
    print("")

if __name__ == "__main__":
    main()





