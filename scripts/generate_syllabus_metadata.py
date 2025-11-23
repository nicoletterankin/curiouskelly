import json
import os
import sys
from typing import List, Dict, Any

# This script defines the "Batch Content Job" to generate the metadata for the 365-day syllabus.
# It prepares the prompts for an LLM to generate Age, Tone, and Language variations for each topic.

# REFERENCE PROMPT: See CLAUDE_DAILY_LESSON_GENERATION_PROMPT.md for the full zero-shot instruction set.

INPUT_FILE = 'lessons/365_day_calendar.json'
OUTPUT_FILE = 'lessons/365_day_dna_metadata.json'

# The Schema we want to populate in Supabase (lesson_dna table)
# {
#   "lesson_id": "string",
#   "day": int,
#   "variants": {
#     "seedling": { "title": "...", "desc": "..." },  # Ages 2-5
#     "explorer": { "title": "...", "desc": "..." },  # Ages 6-12
#     "scholar":  { "title": "...", "desc": "..." },  # Ages 13-18
#     "sage":     { "title": "...", "desc": "..." }   # Adults
#   },
#   "translations": {
#     "es": { "title": "...", "desc": "..." },
#     "fr": { "title": "...", "desc": "..." }
#   }
# }

SYSTEM_PROMPT = """
You are an expert curriculum designer for "Curious Kelly", an AI education platform.
Your task is to take a standard lesson topic and "adapt" it for 4 distinct personas and 3 languages.

Personas:
1. Seedling (2-5): Magical, playful, simple words. Focus on "What can I see/touch?"
2. Explorer (6-12): Action-oriented, "How it works", cool facts. Focus on mechanics and discovery.
3. Scholar (13-18): Academic but engaging, "Why it matters", systems thinking. Focus on critical analysis.
4. Sage (Adult): Deep, philosophical, historical context. Focus on "Wisdom and connection".

Languages:
- English (Original)
- Spanish (Es)
- French (Fr)

Output Format: JSON Only.
"""

def load_calendar(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('lessons', [])

def generate_mock_data(lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates placeholder data to demonstrate the structure without needing an API key immediately.
    In production, this function would be replaced by an LLM call (e.g., Claude/OpenAI).
    """
    output_data = []
    
    print(f"Processing {len(lessons)} lessons...")
    
    for lesson in lessons:
        title = lesson.get('title', 'Unknown Topic')
        day = lesson.get('day', 0)
        
        # Simulated LLM Output
        entry = {
            "lesson_id": lesson.get('lesson_id', f"day-{day}"),
            "day": day,
            "original_title": title,
            "variants": {
                "seedling": {
                    "title": f"Magic {title.split(':')[0]}",
                    "desc": f"Let's play with {title.split(':')[0]}! It is fun and amazing."
                },
                "explorer": {
                    "title": f"{title.split(':')[0]} Explained",
                    "desc": f"Discover the secrets of {title.split(':')[0]} and how it works."
                },
                "scholar": {
                    "title": title,
                    "desc": f"A comprehensive analysis of {title.split(':')[0]} and its scientific principles."
                },
                "sage": {
                    "title": f"The Philosophy of {title.split(':')[0]}",
                    "desc": f"Reflecting on the deeper meaning of {title.split(':')[0]} in human history."
                }
            },
            "translations": {
                "es": {
                    "title": f"El {title.split(':')[0]}",
                    "desc": "(Spanish description placeholder)"
                },
                "fr": {
                    "title": f"Le {title.split(':')[0]}",
                    "desc": "(French description placeholder)"
                }
            }
        }
        output_data.append(entry)
        
    return output_data

def main():
    print("--- Curious Kelly Content Generation Pipeline ---")
    lessons = load_calendar(INPUT_FILE)
    
    if not lessons:
        return

    # Limit to first 50 for the demo file, or remove slice for full generation
    # lessons = lessons[:50] 
    
    generated_content = generate_mock_data(lessons)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(generated_content, f, indent=2)
        
    print(f"Successfully generated metadata for {len(generated_content)} lessons.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("NEXT STEP: Run 'node scripts/upload_to_supabase.js' to populate the database.")

if __name__ == "__main__":
    main()

