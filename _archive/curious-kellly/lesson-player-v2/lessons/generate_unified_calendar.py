#!/usr/bin/env python3
"""
Generate unified 365-day calendar from all curriculum and DNA lesson files.
Merges month curriculum files with DNA lesson files (sun, moon, puppies, leaves, molecular biology, etc.)
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# DNA lesson mappings (lesson_id -> day_number)
DNA_LESSON_MAPPINGS = {
    "the-sun": 1,  # January 1
    "the-moon": 10,  # January 10
    "leaves-change-color": None,  # Will assign to appropriate fall day
    "puppies": None,  # Will assign to appropriate day
    "molecular-biology": 189,  # July 8
    "aging-process": 210,  # July 29
    "stem-cells": None,
    "disruptive-innovation": 207,  # July 26
    "plasma-physics": 206,  # July 25
    "parasitology": 209,  # July 28
}

# Month names
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

def load_curriculum_file(month_name):
    """Load a month curriculum JSON file."""
    filepath = Path(__file__).parent / f"{month_name}_curriculum.json"
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: JSON error in {filepath.name}: {e}")
            return None
    return None

def load_dna_lesson(lesson_id):
    """Load a DNA lesson file - checks lessons/ directory first, then other locations."""
    # Convert lesson_id to different naming conventions
    kebab_case = lesson_id.replace("_", "-")
    snake_case = lesson_id.replace("-", "_")
    
    # Try multiple possible locations and naming conventions
    # Priority: lessons/ directory first (where we consolidated all DNA files)
    possible_paths = [
        # Lessons directory (primary location) - kebab-case
        Path(__file__).parent / f"{kebab_case}-dna.json",
        Path(__file__).parent / f"{kebab_case}.json",
        # Lessons directory - snake_case
        Path(__file__).parent / f"{snake_case}_dna.json",
        Path(__file__).parent / f"{snake_case}.json",
        # Original lesson_id as-is
        Path(__file__).parent / f"{lesson_id}-dna.json",
        Path(__file__).parent / f"{lesson_id}_dna.json",
        Path(__file__).parent / f"{lesson_id}.json",
        # Fallback: curious-kellly backend (for any we might have missed)
        Path(__file__).parent.parent / "curious-kellly" / "backend" / "config" / "lessons" / f"{kebab_case}-dna.json",
        Path(__file__).parent.parent / "curious-kellly" / "backend" / "config" / "lessons" / f"{kebab_case}.json",
        Path(__file__).parent.parent / "curious-kellly" / "backend" / "config" / "lessons" / f"{lesson_id}.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: JSON error in {path.name}: {e}")
                continue
    return None

def create_lesson_entry(day_data, day_number, source="curriculum"):
    """Create a standardized lesson entry."""
    lesson_id = day_data.get("title", "").lower().replace(" ", "-").replace("--", "-")
    # Clean up lesson_id
    lesson_id = "".join(c for c in lesson_id if c.isalnum() or c == "-")[:50]
    
    entry = {
        "day": day_number,
        "date": day_data.get("date", ""),
        "title": day_data.get("title", ""),
        "lesson_id": lesson_id,
        "learning_objective": day_data.get("learning_objective", ""),
        "source": source,
        "has_dna": False,
        "dna_file": None,
        "category": day_data.get("category", "general"),
        "tags": day_data.get("tags", []),
    }
    
    return entry

def merge_dna_lesson(lesson_entry, dna_data):
    """Merge DNA lesson data into calendar entry."""
    if not dna_data:
        return lesson_entry
    
    lesson_entry["has_dna"] = True
    lesson_entry["dna_file"] = dna_data.get("id", "")
    lesson_entry["universal_concept"] = dna_data.get("universal_concept", "")
    lesson_entry["core_principle"] = dna_data.get("core_principle", "")
    lesson_entry["learning_essence"] = dna_data.get("learning_essence", "")
    lesson_entry["age_variants"] = list(dna_data.get("ageVariants", {}).keys()) if "ageVariants" in dna_data else []
    lesson_entry["languages"] = ["en", "es", "fr"]  # All DNA lessons have multilingual support
    
    # Add metadata if available
    if "metadata" in dna_data:
        lesson_entry["category"] = dna_data["metadata"].get("category", lesson_entry.get("category", "general"))
        lesson_entry["tags"] = dna_data["metadata"].get("tags", lesson_entry.get("tags", []))
        lesson_entry["difficulty"] = dna_data["metadata"].get("difficulty", "beginner")
        lesson_entry["duration"] = dna_data["metadata"].get("duration", {"min": 5, "max": 10})
    
    return lesson_entry

def generate_calendar():
    """Generate the unified 365-day calendar."""
    calendar = {
        "version": "1.0.0",
        "createdAt": datetime.now().isoformat() + "Z",
        "description": "Unified 365-day calendar merging all curriculum files and DNA lesson files",
        "totalDays": 365,
        "lessons": []
    }
    
    day_number = 1
    
    # Load all month curriculum files
    for month in MONTHS:
        curriculum = load_curriculum_file(month)
        if curriculum:
            for day_data in curriculum.get("days", []):
                lesson_entry = create_lesson_entry(day_data, day_number, source="curriculum")
                
                # Check if this day has a DNA lesson mapping
                lesson_id_slug = lesson_entry["lesson_id"]
                title_lower = lesson_entry["title"].lower()
                dna_lesson_id = None
                
                # Check DNA mappings by day number first
                for dna_id, mapped_day in DNA_LESSON_MAPPINGS.items():
                    if mapped_day == day_number:
                        dna_lesson_id = dna_id
                        break
                
                # If no day mapping, check title matches
                if not dna_lesson_id:
                    title_keywords = {
                        "the-sun": ["sun", "solar", "star"],
                        "the-moon": ["moon", "lunar", "phases"],
                        "leaves-change-color": ["leaves", "autumn", "fall", "color"],
                        "puppies": ["puppy", "puppies", "dog", "dogs"],
                        "molecular-biology": ["molecular", "biochemistry", "molecules"],
                        "the-ocean": ["ocean", "marine", "sea"],
                        "aging-process": ["aging", "age", "elder"],
                    }
                    
                    for dna_id, keywords in title_keywords.items():
                        if any(keyword in title_lower for keyword in keywords):
                            dna_lesson_id = dna_id
                            break
                
                # Load and merge DNA lesson if found
                if dna_lesson_id:
                    dna_data = load_dna_lesson(dna_lesson_id)
                    if dna_data:
                        lesson_entry = merge_dna_lesson(lesson_entry, dna_data)
                
                calendar["lessons"].append(lesson_entry)
                day_number += 1
    
    # Ensure we have exactly 365 days
    while len(calendar["lessons"]) < 365:
        # Fill remaining days with placeholder
        calendar["lessons"].append({
            "day": len(calendar["lessons"]) + 1,
            "date": f"Day {len(calendar['lessons']) + 1}",
            "title": "Lesson TBD",
            "lesson_id": f"lesson-{len(calendar['lessons']) + 1}",
            "learning_objective": "To be determined",
            "source": "placeholder",
            "has_dna": False,
        })
    
    # Statistics
    calendar["statistics"] = {
        "total_lessons": len(calendar["lessons"]),
        "dna_lessons": sum(1 for l in calendar["lessons"] if l.get("has_dna", False)),
        "curriculum_lessons": sum(1 for l in calendar["lessons"] if l.get("source") == "curriculum"),
        "categories": {}
    }
    
    # Count by category
    for lesson in calendar["lessons"]:
        cat = lesson.get("category", "general")
        calendar["statistics"]["categories"][cat] = calendar["statistics"]["categories"].get(cat, 0) + 1
    
    return calendar

if __name__ == "__main__":
    print("Generating unified 365-day calendar...")
    calendar = generate_calendar()
    
    output_path = Path(__file__).parent / "365_day_calendar.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calendar, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated calendar with {len(calendar['lessons'])} lessons")
    print(f"   - DNA lessons: {calendar['statistics']['dna_lessons']}")
    print(f"   - Curriculum lessons: {calendar['statistics']['curriculum_lessons']}")
    print(f"   - Saved to: {output_path}")

