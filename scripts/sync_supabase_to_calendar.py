"""
Sync Supabase core_lessons to lessons/365_day_calendar.json

This script:
1. Pulls ALL core_lessons from Supabase (source of truth)
2. Transforms to the calendar JSON format
3. Updates 365_day_calendar.json to match

Run: python scripts/sync_supabase_to_calendar.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("daily-lesson-marketing/.env")

# Import Supabase after loading env
try:
    from supabase import create_client, Client
except ImportError:
    print("❌ Error: supabase-py not installed. Run: pip install supabase")
    exit(1)

# Configuration
url = os.environ.get("PUBLIC_SUPABASE_URL")
key = os.environ.get("PUBLIC_SUPABASE_ANON_KEY")

if not url or not key:
    # Fallback to hardcoded values from existing scripts
    url = "https://tvjalxxsyryjphkforjv.supabase.co"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI"
    print("⚠️ Using hardcoded Supabase credentials from existing scripts")

supabase: Client = create_client(url, key)

# Calendar date mapping (non-leap year)
MONTH_DAYS = [
    ("January", 31), ("February", 28), ("March", 31), ("April", 30),
    ("May", 31), ("June", 30), ("July", 31), ("August", 31),
    ("September", 30), ("October", 31), ("November", 30), ("December", 31)
]

def get_date_for_day(day_number: int) -> str:
    """Convert day number (1-365) to 'Month Day' format."""
    current_day = 0
    for month_name, days_in_month in MONTH_DAYS:
        if current_day + days_in_month >= day_number:
            day_of_month = day_number - current_day
            return f"{month_name} {day_of_month}"
        current_day += days_in_month
    return f"December 31"  # Fallback

def generate_lesson_id(title: str) -> str:
    """Generate a URL-friendly lesson ID from title."""
    import re
    # Convert to lowercase, replace spaces with hyphens
    lesson_id = title.lower()
    lesson_id = re.sub(r'[^a-z0-9\s-]', '', lesson_id)  # Remove special chars
    lesson_id = re.sub(r'\s+', '-', lesson_id)  # Replace spaces with hyphens
    lesson_id = re.sub(r'-+', '-', lesson_id)  # Remove duplicate hyphens
    lesson_id = lesson_id.strip('-')
    # Truncate if too long
    if len(lesson_id) > 50:
        lesson_id = lesson_id[:50].rstrip('-')
    return lesson_id

def transform_supabase_to_calendar(db_lesson: dict) -> dict:
    """Transform a Supabase core_lesson record to calendar JSON format."""
    day = db_lesson.get("day_number", 0)
    topic = db_lesson.get("topic", "Unknown Topic")
    universal_truth = db_lesson.get("universal_truth", "")
    
    # Build the calendar entry
    calendar_entry = {
        "day": day,
        "date": get_date_for_day(day),
        "title": topic,
        "lesson_id": generate_lesson_id(topic),
        "learning_objective": universal_truth,
        "source": "supabase",
        "has_dna": False,  # Will be updated if DNA file exists
        "dna_file": None,
        "category": db_lesson.get("category", "general"),
        "tags": db_lesson.get("tags", []) if isinstance(db_lesson.get("tags"), list) else [],
    }
    
    # Add optional fields if present in Supabase
    if db_lesson.get("icon_emoji"):
        calendar_entry["icon"] = db_lesson["icon_emoji"]
    
    if db_lesson.get("difficulty_level"):
        calendar_entry["difficulty"] = db_lesson["difficulty_level"].lower()
    
    if db_lesson.get("estimated_duration"):
        calendar_entry["duration"] = {
            "min": max(5, db_lesson["estimated_duration"] - 3),
            "max": db_lesson["estimated_duration"] + 5
        }
    
    if db_lesson.get("learning_objectives"):
        calendar_entry["learning_objectives"] = db_lesson["learning_objectives"]
    
    if db_lesson.get("marketing_headline"):
        calendar_entry["marketing_headline"] = db_lesson["marketing_headline"]
    
    if db_lesson.get("marketing_tagline"):
        calendar_entry["marketing_tagline"] = db_lesson["marketing_tagline"]
    
    return calendar_entry

def check_dna_files(lessons: list, dna_directory: Path) -> list:
    """Update lessons with DNA file information if files exist."""
    # Get list of existing DNA files
    dna_files = set()
    if dna_directory.exists():
        for f in dna_directory.glob("*-dna.json"):
            # Extract the base name (e.g., "the-sun" from "the-sun-dna.json")
            base_name = f.stem.replace("-dna", "")
            dna_files.add(base_name)
    
    for lesson in lessons:
        lesson_id = lesson.get("lesson_id", "")
        # Check various possible DNA file names
        potential_names = [
            lesson_id,
            lesson_id.replace("--", "-"),
            lesson.get("title", "").lower().replace(" ", "-").replace(":", "").replace("--", "-")
        ]
        
        for name in potential_names:
            if name in dna_files:
                lesson["has_dna"] = True
                lesson["dna_file"] = name
                break
    
    return lessons

def main():
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   🔄 SYNC SUPABASE → 365_day_calendar.json                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    
    # Step 1: Fetch all core_lessons from Supabase
    print("📥 Fetching core_lessons from Supabase...")
    
    try:
        # Use pagination to get all records
        all_lessons = []
        page_size = 1000
        page = 0
        
        while True:
            response = supabase.table("core_lessons")\
                .select("*")\
                .range(page * page_size, (page + 1) * page_size - 1)\
                .order("day_number")\
                .execute()
            
            if not response.data:
                break
            
            all_lessons.extend(response.data)
            print(f"   Fetched page {page + 1}: {len(response.data)} records (total: {len(all_lessons)})")
            
            if len(response.data) < page_size:
                break
            page += 1
        
        print(f"✅ Total records fetched: {len(all_lessons)}")
        
    except Exception as e:
        print(f"❌ Error fetching from Supabase: {e}")
        return
    
    if not all_lessons:
        print("❌ No lessons found in Supabase!")
        return
    
    # Step 2: Transform to calendar format
    print("\n🔄 Transforming to calendar format...")
    
    # Sort by day_number
    all_lessons.sort(key=lambda x: x.get("day_number", 0))
    
    calendar_lessons = []
    for db_lesson in all_lessons:
        calendar_entry = transform_supabase_to_calendar(db_lesson)
        calendar_lessons.append(calendar_entry)
    
    print(f"✅ Transformed {len(calendar_lessons)} lessons")
    
    # Step 3: Check for existing DNA files
    print("\n🔍 Checking for DNA files...")
    dna_directory = Path("lessons")
    calendar_lessons = check_dna_files(calendar_lessons, dna_directory)
    dna_count = sum(1 for l in calendar_lessons if l.get("has_dna"))
    print(f"✅ Found DNA files for {dna_count} lessons")
    
    # Step 4: Build the full calendar JSON
    calendar_json = {
        "version": "2.0.0",
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "description": "365-day calendar synced from Supabase core_lessons (Source of Truth)",
        "syncedFrom": "supabase:core_lessons",
        "syncedAt": datetime.utcnow().isoformat() + "Z",
        "totalDays": len(calendar_lessons),
        "lessons": calendar_lessons
    }
    
    # Step 5: Backup existing calendar file
    calendar_path = Path("lessons/365_day_calendar.json")
    if calendar_path.exists():
        backup_path = Path(f"lessons/365_day_calendar.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"\n📦 Backing up existing calendar to: {backup_path.name}")
        import shutil
        shutil.copy(calendar_path, backup_path)
    
    # Step 6: Write the new calendar
    print(f"\n💾 Writing to {calendar_path}...")
    with open(calendar_path, "w", encoding="utf-8") as f:
        json.dump(calendar_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Calendar updated successfully!")
    
    # Step 7: Show sample of what was synced
    print("\n" + "=" * 60)
    print("📋 SAMPLE: First 5 lessons")
    print("=" * 60)
    for lesson in calendar_lessons[:5]:
        print(f"Day {lesson['day']:3d} | {lesson['title'][:45]:<45}")
    
    print("\n" + "=" * 60)
    print("📋 SAMPLE: Last 5 lessons")
    print("=" * 60)
    for lesson in calendar_lessons[-5:]:
        print(f"Day {lesson['day']:3d} | {lesson['title'][:45]:<45}")
    
    print("\n" + "=" * 60)
    print("📊 SYNC SUMMARY")
    print("=" * 60)
    print(f"   Total lessons:     {len(calendar_lessons)}")
    print(f"   With DNA files:    {dna_count}")
    print(f"   Source:            Supabase core_lessons")
    print(f"   Target:            lessons/365_day_calendar.json")
    print("=" * 60)
    print("")
    print("✅ SYNC COMPLETE! Run verify_calendar_alignment.py to confirm.")

if __name__ == "__main__":
    main()

