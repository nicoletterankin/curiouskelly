#!/usr/bin/env python3
"""Verify replacements were applied correctly."""
import json

with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

replaced = [l for l in data['lessons'] if l.get('source') == 'replacement']

print(f'✅ Replacement lessons found: {len(replaced)}\n')
print('Replaced lessons:')
print('-' * 80)
for lesson in sorted(replaced, key=lambda x: x['day']):
    print(f"Day {lesson['day']:3d}: {lesson['title']}")
    print(f"         Objective: {lesson['learning_objective'][:70]}...")
    print()

