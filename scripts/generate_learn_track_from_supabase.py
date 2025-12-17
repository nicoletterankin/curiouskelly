"""
Generate 12 monthly Learn track curriculum files from 365_day_calendar.json (Supabase data)

This ensures the Learn track curriculum browser matches Supabase exactly.
"""

import json
from pathlib import Path

# Month configuration
MONTHS = [
    ("January", 31, "Beginnings", "Starting fresh with wonder and curiosity"),
    ("February", 28, "Earth & Sky", "Exploring our planet and the cosmos above"),
    ("March", 31, "Life Science", "Understanding living things and the human body"),
    ("April", 30, "Mind & Skills", "Building thinking skills and capabilities"),
    ("May", 31, "Cultures", "Exploring human achievement and diversity"),
    ("June", 30, "Innovation", "Technology, invention, and problem-solving"),
    ("July", 31, "Adventure", "Exploration, discovery, and the natural world"),
    ("August", 31, "Creativity", "Art, music, and creative expression"),
    ("September", 30, "Society", "Communities, systems, and how we live together"),
    ("October", 31, "Mysteries", "The unknown, the ancient, and the fascinating"),
    ("November", 30, "Gratitude", "Appreciation, reflection, and thankfulness"),
    ("December", 31, "Celebration", "Traditions, joy, and looking forward"),
]

def load_supabase_calendar():
    """Load the 365-day calendar synced from Supabase."""
    calendar_path = Path(__file__).parent.parent / "lessons" / "365_day_calendar.json"
    
    with open(calendar_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Create a lookup by day number
    lessons_by_day = {lesson["day"]: lesson for lesson in data.get("lessons", [])}
    return lessons_by_day

def generate_monthly_curriculum(lessons_by_day):
    """Generate 12 monthly curriculum JSON files."""
    
    output_dirs = [
        Path(__file__).parent.parent / "lessons" / "year1-foundations",
        Path(__file__).parent.parent / "public" / "data" / "curriculum" / "year1-foundations",
    ]
    
    # Ensure directories exist
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    current_day = 1
    results = []
    
    for month_name, days_in_month, theme, theme_desc in MONTHS:
        month_data = {
            "year": 1,
            "track": "learn",
            "program": "Learn Track",
            "month": month_name,
            "theme": theme,
            "themeDescription": theme_desc,
            "days": []
        }
        
        for day_of_month in range(1, days_in_month + 1):
            lesson = lessons_by_day.get(current_day)
            
            if lesson:
                day_entry = {
                    "day": current_day,
                    "date": f"{month_name} {day_of_month}",
                    "title": lesson.get("title", f"Day {current_day}"),
                    "learning_objective": lesson.get("learning_objective", ""),
                    "icon": lesson.get("icon", "🌟"),
                }
                
                # Add optional fields if present
                if lesson.get("learning_objectives"):
                    day_entry["learning_objectives"] = lesson["learning_objectives"]
                if lesson.get("difficulty"):
                    day_entry["difficulty"] = lesson["difficulty"]
                if lesson.get("category"):
                    day_entry["category"] = lesson["category"]
            else:
                # Fallback for missing days
                day_entry = {
                    "day": current_day,
                    "date": f"{month_name} {day_of_month}",
                    "title": f"Day {current_day} Topic",
                    "learning_objective": "Learning objective to be defined.",
                    "icon": "🌟"
                }
            
            month_data["days"].append(day_entry)
            current_day += 1
        
        # Write to both locations
        filename = f"{month_name.lower()}_curriculum.json"
        
        for output_dir in output_dirs:
            output_path = output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(month_data, f, indent=2, ensure_ascii=False)
        
        results.append({
            "month": month_name,
            "days": days_in_month,
            "first_topic": month_data["days"][0]["title"],
            "last_topic": month_data["days"][-1]["title"],
        })
        
        print(f"[OK] {month_name}: Days {month_data['days'][0]['day']}-{month_data['days'][-1]['day']} ({days_in_month} topics)")
    
    return results

def main():
    print("")
    print("=" * 67)
    print("   GENERATE LEARN TRACK CURRICULUM FROM SUPABASE")
    print("=" * 67)
    print("")
    
    # Load Supabase data
    print("Loading 365-day calendar from Supabase...")
    lessons_by_day = load_supabase_calendar()
    print(f"Loaded {len(lessons_by_day)} lessons")
    print("")
    
    # Generate monthly files
    print("Generating 12 monthly curriculum files...")
    print("-" * 60)
    results = generate_monthly_curriculum(lessons_by_day)
    print("-" * 60)
    print("")
    
    # Summary
    total_days = sum(r["days"] for r in results)
    print(f"COMPLETE: Generated {len(results)} monthly files ({total_days} days)")
    print("")
    print("Files written to:")
    print("  - lessons/year1-foundations/")
    print("  - public/data/curriculum/year1-foundations/")
    print("")

if __name__ == "__main__":
    main()
