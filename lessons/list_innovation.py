#!/usr/bin/env python3
"""List all innovation lessons."""
import json

with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

innovation = [l for l in data['lessons'] if 'innovation' in l['title'].lower()]

print(f'Total innovation lessons: {len(innovation)}\n')
print('Innovation lessons by day:')
print('-' * 80)
for lesson in sorted(innovation, key=lambda x: x['day']):
    print(f"Day {lesson['day']:3d}: {lesson['title']}")

