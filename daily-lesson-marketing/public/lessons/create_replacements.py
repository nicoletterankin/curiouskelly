#!/usr/bin/env python3
"""
Create high-quality replacement lessons for approved topics.
Replaces innovation lessons and low-value tech lessons with substantive universal topics.
"""

import json
from datetime import datetime

# Load calendar
with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
    calendar = json.load(f)

# Approved replacements from APPROVED_REPLACEMENTS.md
replacements = {
    # Innovation lessons to replace
    249: {
        'title': 'Exchange - Giving and Receiving Value',
        'learning_objective': 'Understand how exchange creates value for everyone while exploring how giving and receiving builds relationships, communities, and economies through mutual benefit and fairness.'
    },
    256: {
        'title': 'Governance - How Groups Make Decisions',
        'learning_objective': 'Explore how groups make decisions together while understanding how governance systems enable fairness, consensus, and collective action in families, communities, and societies.'
    },
    263: {
        'title': 'Connection - How We Share Ideas and Feelings',
        'learning_objective': 'Understand how we connect with others through sharing ideas and feelings while exploring how communication, empathy, and understanding build meaningful relationships across all ages.'
    },
    265: {
        'title': 'Community - How Groups Create Together',
        'learning_objective': 'Explore how communities form and create together while understanding how belonging, collaboration, and collective action enable groups to achieve more than individuals alone.'
    },
    270: {
        'title': 'Persistence - Continuing When It\'s Hard',
        'learning_objective': 'Understand the value of persistence while exploring how continuing through challenges builds resilience, character, and achievement across all life stages and endeavors.'
    },
    277: {
        'title': 'Legacy - What We Leave Behind',
        'learning_objective': 'Explore what legacy means while understanding how our actions, values, and contributions create lasting impact that shapes future generations and communities.'
    },
    303: {
        'title': 'Measurement - Understanding What Matters',
        'learning_objective': 'Understand how we measure what matters while exploring how evaluation, standards, and quantitative thinking help us make sense of the world and make better decisions.'
    },
    306: {
        'title': 'Cooperation - How We Work Together',
        'learning_objective': 'Explore how cooperation enables achievement while understanding how working together, sharing resources, and mutual support create better outcomes than individual effort alone.'
    },
    307: {
        'title': 'Curiosity - The Engine of Discovery',
        'learning_objective': 'Understand how curiosity drives learning and discovery while exploring how asking questions, exploring possibilities, and maintaining wonder enable scientific thinking and lifelong learning.'
    },
    308: {
        'title': 'Observation - Learning by Paying Attention',
        'learning_objective': 'Explore how careful observation teaches us about the world while understanding how paying attention, noticing details, and mindful awareness enable scientific discovery and deeper understanding.'
    },
    309: {
        'title': 'Reflection - Understanding What We\'ve Learned',
        'learning_objective': 'Understand how reflection deepens learning while exploring how thinking about our experiences, understanding our growth, and learning from mistakes enable wisdom and personal development.'
    },
    
    # Low-value tech lessons to replace
    77: {
        'title': 'Interdependence - How All Life Is Connected',
        'learning_objective': 'Explore how all living things depend on each other while understanding how relationships, ecosystems, and connections create the web of life that sustains our planet.'
    },
    91: {
        'title': 'Journey - How Ideas and Goods Travel',
        'learning_objective': 'Understand how ideas and goods move through the world while exploring how journeys, exchange, and cultural diffusion connect people, places, and communities across distances and time.'
    },
    92: {
        'title': 'Balance - How Nature Maintains Harmony',
        'learning_objective': 'Explore how nature maintains balance while understanding how equilibrium, cycles, and adaptation enable ecosystems, relationships, and systems to function sustainably.'
    },
    125: {
        'title': 'Compassion - Understanding and Helping Others',
        'learning_objective': 'Understand how compassion connects us while exploring how empathy, care, and helping others create stronger communities and enable us to support each other through challenges.'
    },
}

# Additional high-quality natural phenomena to add
natural_phenomena = {
    # These will replace additional tech lessons - we'll identify which ones
    'clouds': {
        'title': 'Clouds - The Sky\'s Changing Shapes',
        'learning_objective': 'Explore how clouds form and change while understanding how water, air, and temperature create the beautiful and ever-changing shapes we see in the sky.'
    },
    'rain': {
        'title': 'Rain - Water\'s Journey from Sky to Earth',
        'learning_objective': 'Understand how rain forms and falls while exploring how water cycles through the atmosphere, nourishing plants, filling rivers, and sustaining all life on Earth.'
    },
    'rainbow': {
        'title': 'Rainbow - Light\'s Beautiful Display',
        'learning_objective': 'Explore how rainbows appear while understanding how sunlight, water droplets, and light refraction create the beautiful spectrum of colors that inspire wonder across all ages.'
    },
    'wind': {
        'title': 'Wind - The Invisible Force We Feel',
        'learning_objective': 'Understand how wind moves while exploring how air pressure, temperature differences, and Earth\'s rotation create the invisible force we feel and see in action.'
    },
    'trees': {
        'title': 'Trees - Nature\'s Living Pillars',
        'learning_objective': 'Explore how trees grow and thrive while understanding how these living pillars provide oxygen, shelter, beauty, and stability to ecosystems and communities.'
    },
    'flowers': {
        'title': 'Flowers - Nature\'s Beautiful Invitations',
        'learning_objective': 'Understand how flowers bloom while exploring how these beautiful invitations attract pollinators, create seeds, and bring color, fragrance, and joy to the world.'
    },
    'shadows': {
        'title': 'Shadows - Light\'s Absence Creates Shape',
        'learning_objective': 'Explore how shadows form while understanding how light, objects, and the absence of light create the shapes and patterns that change throughout the day.'
    },
    'reflections': {
        'title': 'Reflections - When Surfaces Mirror the World',
        'learning_objective': 'Understand how reflections work while exploring how smooth surfaces like water and mirrors create images that help us see ourselves and the world in new ways.'
    },
}

def create_lesson_entry(day, title, learning_objective, source='replacement'):
    """Create a lesson entry matching the calendar structure."""
    lesson_id = title.lower().replace(' ', '-').replace('--', '-')
    lesson_id = ''.join(c for c in lesson_id if c.isalnum() or c == '-')[:50]
    
    # Extract date from existing lesson
    existing_lesson = next((l for l in calendar['lessons'] if l['day'] == day), None)
    date = existing_lesson['date'] if existing_lesson else f"Day {day}"
    
    return {
        'day': day,
        'date': date,
        'title': title,
        'lesson_id': lesson_id,
        'learning_objective': learning_objective,
        'source': source,
        'has_dna': False,
        'dna_file': None,
        'category': 'general',  # Will be auto-categorized
        'tags': []
    }

def apply_replacements():
    """Apply all approved replacements to the calendar."""
    updated = 0
    
    for day, replacement in replacements.items():
        # Find the lesson
        lesson_index = None
        for i, lesson in enumerate(calendar['lessons']):
            if lesson['day'] == day:
                lesson_index = i
                break
        
        if lesson_index is not None:
            # Create new lesson entry
            new_lesson = create_lesson_entry(
                day,
                replacement['title'],
                replacement['learning_objective']
            )
            
            # Preserve some metadata from original
            original = calendar['lessons'][lesson_index]
            new_lesson['date'] = original['date']
            
            # Replace
            calendar['lessons'][lesson_index] = new_lesson
            updated += 1
            print(f"✅ Day {day}: Replaced '{original['title']}' with '{replacement['title']}'")
        else:
            print(f"⚠️  Day {day}: Lesson not found")
    
    # Update metadata
    calendar['updatedAt'] = datetime.now().isoformat() + 'Z'
    calendar['description'] = 'Unified 365-day calendar with high-quality universal topic replacements'
    
    # Save
    with open('365_day_calendar.json', 'w', encoding='utf-8') as f:
        json.dump(calendar, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated {updated} lessons")
    print(f"✅ Saved to 365_day_calendar.json")

if __name__ == "__main__":
    print("Creating high-quality replacement lessons...")
    print("=" * 80)
    apply_replacements()
    print("\n" + "=" * 80)
    print("Replacements complete!")

