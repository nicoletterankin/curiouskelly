#!/usr/bin/env python3
"""
Generate missing PhaseDNA lesson files for Days 1-30 expansion
Uses the 365-day calendar as source of truth for learning objectives
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Define lessons that need DNA creation
LESSONS_TO_CREATE = [
    {
        "day": 2,
        "id": "habit-stacking-for-productivity",
        "title": "Habit Stacking for Productivity - Building Your Success Architecture",
        "category": "life-skills",
        "difficulty": "beginner"
    },
    {
        "day": 3,
        "id": "our-amazing-planet-earth",
        "title": "Our Amazing Planet Earth - A Spinning, Dancing World",
        "category": "science",
        "difficulty": "beginner"
    },
    {
        "day": 4,
        "id": "simple-machines",
        "title": "Simple Machines - The Amazing Tools That Make Work Easier",
        "category": "science",
        "difficulty": "beginner"
    },
    {
        "day": 5,
        "id": "emotional-regulation",
        "title": "Emotional Regulation - The Superpower of Managing Your Inner World",
        "category": "life-skills",
        "difficulty": "beginner"
    },
    {
        "day": 6,
        "id": "ancient-civilizations",
        "title": "Ancient Civilizations - The Dawn of Human Society",
        "category": "history",
        "difficulty": "intermediate"
    },
    {
        "day": 7,
        "id": "critical-thinking",
        "title": "Critical Thinking",
        "category": "life-skills",
        "difficulty": "intermediate"
    },
    # Day 8 Water Cycle already exists
    # Day 9 already has DNA (leaves-change-color maps to it)
    # Day 10 already has DNA (the-moon)
    {
        "day": 11,
        "id": "industrial-revolution",
        "title": "Industrial Revolution - When Machines Changed Everything",
        "category": "history",
        "difficulty": "intermediate"
    },
    {
        "day": 12,
        "id": "exercise-physiology",
        "title": "Exercise Physiology - How Movement Affects Health",
        "category": "health",
        "difficulty": "beginner"
    },
    {
        "day": 13,
        "id": "world-religions",
        "title": "World Religions - Humanity's Spiritual Journey",
        "category": "culture",
        "difficulty": "intermediate"
    },
    {
        "day": 14,
        "id": "renaissance",
        "title": "Renaissance - The Rebirth of Learning",
        "category": "history",
        "difficulty": "intermediate"
    },
    # Day 15 already has DNA (the-sun maps to photosynthesis)
    {
        "day": 16,
        "id": "weather-and-climate",
        "title": "Weather and Climate - Earth's Amazing Atmospheric Theater",
        "category": "science",
        "difficulty": "beginner"
    },
    {
        "day": 17,
        "id": "gravity",
        "title": "Gravity - The Invisible Force Shaping Our Universe",
        "category": "science",
        "difficulty": "intermediate"
    },
    {
        "day": 18,
        "id": "social-media-revolution",
        "title": "Social Media Revolution - How Digital Connection Changed Society",
        "category": "technology",
        "difficulty": "intermediate"
    },
    # Day 19 already has DNA (the-ocean maps to medical breakthroughs)
    {
        "day": 20,
        "id": "language-development",
        "title": "Language Development - How Humans Communicate",
        "category": "language",
        "difficulty": "intermediate"
    },
    {
        "day": 21,
        "id": "age-of-enlightenment",
        "title": "Age of Enlightenment - Reason and Progress",
        "category": "history",
        "difficulty": "advanced"
    },
    {
        "day": 22,
        "id": "your-amazing-brain",
        "title": "Your Amazing Brain - The Universe's Most Incredible Computer",
        "category": "science",
        "difficulty": "intermediate"
    },
    # Day 23 and 24 already have DNA (the-ocean)
    {
        "day": 25,
        "id": "communication-history",
        "title": "Communication History - From Smoke Signals to Internet",
        "category": "history",
        "difficulty": "intermediate"
    },
    {
        "day": 26,
        "id": "public-health",
        "title": "Public Health - Keeping Communities Healthy",
        "category": "health",
        "difficulty": "intermediate"
    },
    {
        "day": 27,
        "id": "exploration-and-discovery",
        "title": "Exploration and Discovery - Expanding Human Horizons",
        "category": "history",
        "difficulty": "intermediate"
    },
    # Day 28 may have DNA
    {
        "day": 29,
        "id": "the-mysterious-kingdom-of-fungi",
        "title": "The Mysterious Kingdom of Fungi - Nature's Hidden Recyclers and Life-Savers",
        "category": "science",
        "difficulty": "intermediate"
    },
    {
        "day": 30,
        "id": "art-history",
        "title": "Art History - Human Expression Through Time",
        "category": "arts",
        "difficulty": "intermediate"
    },
]

def load_365_calendar():
    """Load the 365-day calendar for learning objectives"""
    with open('lessons/365_day_calendar.json', 'r') as f:
        return json.load(f)

def create_age_variant(lesson_id, title, category, age_range, kelly_age, kelly_persona, learning_objective):
    """Create a single age variant with basic structure"""

    # Age-specific content templates
    templates = {
        "2-5": {
            "welcome": f"Hi friend! Today we're going to learn about {title.split('-')[0].strip().lower()}!",
            "content_style": "Simple, playful, concrete examples",
            "interaction": "Can you show me...? Let's pretend we're..."
        },
        "6-12": {
            "welcome": f"Hey! Ready to discover something cool about {title.split('-')[0].strip().lower()}?",
            "content_style": "Engaging with why/how questions",
            "interaction": "What do you think happens when...? Have you ever noticed...?"
        },
        "13-17": {
            "welcome": f"What's up? Let's talk about {title.split('-')[0].strip().lower()}...",
            "content_style": "Relevant to teen life, acknowledges complexity",
            "interaction": "How does this relate to...? What's your take on...?"
        },
        "18-35": {
            "welcome": f"Today's topic is fascinating: {title.split('-')[0].strip()}.",
            "content_style": "Sophisticated, practical applications",
            "interaction": "In your experience, how might...? Consider the implications..."
        },
        "36-60": {
            "welcome": f"Let's explore {title.split('-')[0].strip()} together.",
            "content_style": "Nuanced, life-experience connections",
            "interaction": "Reflecting on your experience... How has this evolved..."
        },
        "61-102": {
            "welcome": f"Today we're exploring {title.split('-')[0].strip()}.",
            "content_style": "Wisdom-oriented, legacy perspective",
            "interaction": "Looking back... What patterns do you see..."
        }
    }

    template = templates.get(age_range, templates["18-35"])

    return {
        "kellyAge": kelly_age,
        "kellyPersona": kelly_persona,
        "language": {
            "en": {
                "welcome": template["welcome"],
                "mainContent": f"[CONTENT TO BE WRITTEN: {template['content_style']}]\n\nLearning Objective: {learning_objective}",
                "keyPoints": [
                    "[Key point 1 - to be written]",
                    "[Key point 2 - to be written]",
                    "[Key point 3 - to be written]"
                ],
                "interactionPrompts": [
                    template["interaction"].split('?')[0] + "?",
                    "[Additional interaction prompt]"
                ],
                "wisdomMoment": "[Wisdom insight - to be written]",
                "cta": "Keep exploring!",
                "summary": "Great learning today!",
                "title": title
            }
        },
        "teachingMoments": [
            {
                "id": f"tm1-{age_range}",
                "timestamp": 60,
                "type": "inquiry",
                "content": "[Teaching moment - to be written]"
            }
        ],
        "expressionCues": [
            {
                "id": f"ec1-{age_range}",
                "momentRef": f"tm1-{age_range}",
                "type": "micro-smile",
                "offset": 0,
                "duration": 2,
                "intensity": "medium",
                "gazeTarget": "camera"
            }
        ]
    }

def generate_lesson_dna(lesson_info, calendar_data):
    """Generate a complete PhaseDNA lesson file"""

    # Find the lesson in the calendar
    calendar_lesson = next((l for l in calendar_data['lessons'] if l['day'] == lesson_info['day']), None)

    if not calendar_lesson:
        print(f"Warning: Day {lesson_info['day']} not found in calendar")
        learning_objective = "Universal learning objective"
    else:
        learning_objective = calendar_lesson.get('learning_objective', 'Universal learning objective')

    # Create the DNA structure
    dna = {
        "id": lesson_info['id'],
        "title": lesson_info['title'],
        "version": "1.0.0",
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "updatedAt": datetime.utcnow().isoformat() + "Z",
        "author": "Claude AI Content Generator",
        "description": f"Learn about {lesson_info['title'].split('-')[0].strip()}",
        "metadata": {
            "category": lesson_info['category'],
            "difficulty": lesson_info['difficulty'],
            "duration": {
                "min": 5,
                "max": 13
            },
            "learningOutcomes": [
                learning_objective
            ],
            "keywords": [
                lesson_info['id'].replace('-', ' ')
            ],
            "relatedTopics": [],
            "tags": [lesson_info['category']]
        },
        "ageVariants": {}
    }

    # Create all 6 age variants
    age_variants = [
        ("2-5", 3, "playful-toddler"),
        ("6-12", 9, "curious-kid"),
        ("13-17", 15, "teen-mentor"),
        ("18-35", 27, "knowledgeable-adult"),
        ("36-60", 48, "experienced-guide"),
        ("61-102", 82, "wise-elder")
    ]

    for age_range, kelly_age, persona in age_variants:
        dna["ageVariants"][age_range] = create_age_variant(
            lesson_info['id'],
            lesson_info['title'],
            lesson_info['category'],
            age_range,
            kelly_age,
            persona,
            learning_objective
        )

    return dna

def main():
    """Generate all missing lesson DNA files"""

    print("🚀 Generating Missing Lesson DNA Files for Days 1-30 Expansion")
    print("=" * 80)
    print()

    # Load calendar
    calendar = load_365_calendar()
    print(f"✅ Loaded 365-day calendar ({len(calendar['lessons'])} lessons)")
    print()

    # Output directory
    output_dir = Path("curious-kellly/backend/config/lessons")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each lesson
    created = 0
    for lesson_info in LESSONS_TO_CREATE:
        output_file = output_dir / f"{lesson_info['id']}.json"

        # Skip if already exists
        if output_file.exists():
            print(f"⏭️  Day {lesson_info['day']:2d}: {lesson_info['title']} (already exists)")
            continue

        # Generate DNA
        dna = generate_lesson_dna(lesson_info, calendar)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(dna, f, indent=2)

        created += 1
        print(f"✅ Day {lesson_info['day']:2d}: Created {lesson_info['id']}.json")

    print()
    print("=" * 80)
    print(f"📊 SUMMARY: Created {created} new lesson DNA files")
    print(f"📁 Location: {output_dir}")
    print()
    print("⚠️  NOTE: These are TEMPLATE files with placeholders.")
    print("   Next steps:")
    print("   1. Review each file and fill in detailed content")
    print("   2. Add ES and FR translations")
    print("   3. Validate against schema")
    print("   4. Generate audio files")
    print()

if __name__ == "__main__":
    main()
