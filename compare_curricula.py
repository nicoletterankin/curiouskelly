#!/usr/bin/env python3
"""Compare 30-day curriculum with 365-day calendar"""

import json
from pathlib import Path

# 30-day curriculum topics
topics_30 = {
    1: "Leaves",
    2: "Water",
    3: "Clouds",
    4: "Light",
    5: "Sound",
    6: "Seeds",
    7: "Stars",
    8: "Friendship",
    9: "Kindness",
    10: "Listening",
    11: "Patience",
    12: "Gratitude",
    13: "Courage",
    14: "Curiosity",
    15: "Balance",
    16: "Breathing",
    17: "Movement",
    18: "Rest",
    19: "Energy",
    20: "Senses",
    21: "Growth",
    22: "Colors",
    23: "Patterns",
    24: "Stories",
    25: "Music",
    26: "Questions",
    27: "Imagination",
    28: "Memory",
    29: "Time",
    30: "Change"
}

# Load 365-day calendar
calendar_path = Path("lessons/365_day_calendar.json")
with open(calendar_path, 'r', encoding='utf-8') as f:
    calendar_365 = json.load(f)

# Search for matches
matches = {}
no_matches = []
multiple_matches = {}

for day_30, topic in topics_30.items():
    topic_lower = topic.lower()
    found = []
    
    for lesson in calendar_365['lessons']:
        title_lower = lesson['title'].lower()
        # Check if topic keyword appears in title
        if topic_lower in title_lower or any(word in title_lower for word in topic_lower.split()):
            found.append({
                'day': lesson['day'],
                'date': lesson['date'],
                'title': lesson['title'],
                'has_dna': lesson.get('has_dna', False),
                'category': lesson.get('category', 'unknown')
            })
    
    if len(found) == 0:
        no_matches.append(topic)
    elif len(found) == 1:
        matches[topic] = found[0]
    else:
        multiple_matches[topic] = found

# Print results
print("="*80)
print("30-DAY CURRICULUM vs 365-DAY CALENDAR COMPARISON")
print("="*80)
print()

print("✅ EXACT OR CLOSE MATCHES:")
print("-"*80)
for topic, match in sorted(matches.items()):
    print(f"Day {list(topics_30.keys())[list(topics_30.values()).index(topic)]:2d}: {topic:15s} → Day {match['day']:3d} ({match['date']:15s}) | {match['title'][:50]:50s} | DNA: {match['has_dna']}")
print()

if multiple_matches:
    print("⚠️  MULTIPLE MATCHES (needs review):")
    print("-"*80)
    for topic, matches_list in multiple_matches.items():
        print(f"\n{topic}:")
        for m in matches_list:
            print(f"  - Day {m['day']:3d} ({m['date']:15s}): {m['title']}")
    print()

if no_matches:
    print("❌ NO MATCHES FOUND:")
    print("-"*80)
    for topic in no_matches:
        day_num = list(topics_30.keys())[list(topics_30.values()).index(topic)]
        print(f"Day {day_num:2d}: {topic}")
    print()

print("="*80)
print(f"Summary: {len(matches)} matches, {len(no_matches)} not found, {len(multiple_matches)} multiple matches")
print("="*80)




