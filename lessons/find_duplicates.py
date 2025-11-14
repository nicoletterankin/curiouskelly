#!/usr/bin/env python3
"""Find duplicate and similar lessons."""
import json

with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("DUPLICATE BIOTECHNOLOGY LESSONS:")
print("-" * 80)
biotech = [l for l in data['lessons'] if 'biotechnology' in l['title'].lower() and 'using life' in l['title'].lower()]
for l in biotech:
    print(f"Day {l['day']}: {l['title']}")
    print(f"  Date: {l['date']}")
    print(f"  Learning Objective: {l['learning_objective'][:80]}...")
    print()

