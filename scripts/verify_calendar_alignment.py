import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()
load_dotenv("daily-lesson-marketing/.env")

url = os.environ.get("PUBLIC_SUPABASE_URL")
key = os.environ.get("PUBLIC_SUPABASE_ANON_KEY")

if not url or not key:
    print("❌ Error: PUBLIC_SUPABASE_URL or PUBLIC_SUPABASE_ANON_KEY not found in environment.")
    print("Please ensure .env exists and is loaded.")
    exit(1)

supabase: Client = create_client(url, key)

def check_alignment():
    print("🔄 Loading 365-day calendar...")
    calendar_path = Path("lessons/365_day_calendar.json")
    if not calendar_path.exists():
        print(f"❌ Error: {calendar_path} not found.")
        return

    with open(calendar_path, "r", encoding="utf-8") as f:
        calendar_data = json.load(f)
        calendar_lessons = {l["day"]: l for l in calendar_data.get("lessons", [])}

    print(f"✅ Loaded {len(calendar_lessons)} lessons from calendar file.")

    print("🔄 Fetching lessons from Supabase...")
    try:
        # Fetch all lessons from core_lessons
        # Using a large limit to get all 365+
        response = supabase.table("core_lessons").select("*").execute()
        db_lessons = {l["day_number"]: l for l in response.data}
        print(f"✅ Fetched {len(db_lessons)} lessons from Supabase.")
    except Exception as e:
        print(f"❌ Error fetching from Supabase: {e}")
        return

    print("\n📊 Alignment Report:")
    print("=" * 40)

    mismatches = []
    missing_in_db = []
    missing_in_file = []

    # Check File vs DB
    for day, file_lesson in calendar_lessons.items():
        if day not in db_lessons:
            missing_in_db.append(day)
            continue
        
        db_lesson = db_lessons[day]
        
        # Compare Titles (ignoring case/trim)
        file_title = file_lesson.get("title", "").strip()
        db_title = db_lesson.get("topic", "").strip() # Assuming 'topic' is the title in DB based on previous file

        if file_title.lower() != db_title.lower():
             mismatches.append({
                 "day": day,
                 "issue": "Title Mismatch",
                 "file": file_title,
                 "db": db_title
             })

    # Check DB vs File
    for day in db_lessons:
        if day not in calendar_lessons:
            missing_in_file.append(day)

    # Report
    if missing_in_db:
        print(f"❌ Missing in Supabase (Total: {len(missing_in_db)}):")
        print(f"   Days: {missing_in_db[:10]}{'...' if len(missing_in_db)>10 else ''}")
    else:
        print("✅ All calendar days present in Supabase.")

    if missing_in_file:
        print(f"⚠️ Found in Supabase but not in Calendar (Total: {len(missing_in_file)}):")
        print(f"   Days: {missing_in_file[:10]}{'...' if len(missing_in_file)>10 else ''}")

    if mismatches:
        print(f"❌ Content Mismatches (Total: {len(mismatches)}):")
        for m in mismatches[:10]:
            print(f"   Day {m['day']}: File='{m['file']}' vs DB='{m['db']}'")
        if len(mismatches) > 10:
            print(f"   ...and {len(mismatches) - 10} more.")
    else:
        print("✅ Content titles match perfectly.")

    if not missing_in_db and not mismatches:
        print("\n✨ ALIGNMENT VERIFIED: 100% MATCH")
    else:
        print("\n⚠️ ALIGNMENT FAILED: See errors above.")

if __name__ == "__main__":
    check_alignment()




