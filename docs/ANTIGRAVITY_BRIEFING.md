# ANTIGRAVITY AGENT BRIEFING
**Date:** February 3, 2026  
**From:** Cursor Agent  
**Priority:** HIGH

---

## YOUR ROLE

You are responsible for **content generation** - specifically scripts, translations, and quality evaluation. You do NOT generate videos or deploy code.

---

## CURRENT STATE

| Days | Video Status | Script Status |
|------|-------------|---------------|
| 1-30 | 91 videos exist | Scripts exist |
| 31-33 | 15 videos generating NOW | Cursor used placeholder scripts |
| 34 | 5 videos READY | Scripts exist |
| 35-60 | NOT STARTED | **NEED YOUR SCRIPTS** |

---

## WHAT WE NEED FROM YOU

### 1. Scripts for Days 35-60 (URGENT)

For each day, provide scripts for:
- **hook** (~30 seconds) - Curiosity opener
- **story** (~60 seconds) - Historical/narrative content
- **wonder** (~30 seconds) - Open-ended question
- **action** (~30 seconds) - Hands-on activity
- **wisdom** (~30 seconds) - Life lesson/takeaway

Format:
```json
{
  "day": 35,
  "topic": "Topic Name",
  "hook": "Script text...",
  "story": "Script text...",
  "wonder": "Script text...",
  "action": "Script text...",
  "wisdom": "Script text..."
}
```

### 2. Variations by Age (NEXT PRIORITY)

Each script should have versions for:
- **kid** (ages 5-10) - Simple vocabulary, fun examples
- **adult** (ages 18-60) - Standard vocabulary
- **elder** (ages 60+) - Nostalgic references, life experience connections

### 3. Variations by Archetype (FUTURE)

12 archetypes with different teaching styles:
- scientist, explorer, rebel, architect, diplomat, empath
- macgyver, mystic, provider, storyteller, strategist, survivor

---

## DATABASE SCHEMA

Store your scripts in `lesson_perspectives`:

```sql
INSERT INTO lesson_perspectives (
  id, day_number, age_group, archetype, language,
  title, subtitle, topic, theme,
  hook_script, story_script, wonder_script, action_script, wisdom_script,
  created_at, updated_at
) VALUES (
  gen_random_uuid(), 35, 'adult', 'storyteller', 'en',
  'The Science of Sound', 'How vibrations become music', 'Sound', 'Physics',
  'Have you ever wondered...', 'Long ago...', 'What if...', 'Try this...', 'Remember...',
  NOW(), NOW()
);
```

---

## TOPICS NEEDED (Days 35-60)

| Day | Suggested Topic |
|-----|-----------------|
| 35 | The Science of Sound |
| 36 | Why the Sky is Blue |
| 37 | How Vaccines Work |
| 38 | The Power of Compound Interest |
| 39 | Why We Dream |
| 40 | How Batteries Work |
| 41 | The History of Writing |
| 42 | Why Leaves Change Color |
| 43 | How GPS Works |
| 44 | The Science of Cooking |
| 45 | Why Music Affects Emotions |
| 46 | How Memory Works |
| 47 | The Water We Drink |
| 48 | Why We Need Sleep |
| 49 | How Planes Fly |
| 50 | The Science of Smell |
| 51 | Why Ice Floats |
| 52 | How the Internet Works |
| 53 | Why We Get Hiccups |
| 54 | The History of Time |
| 55 | Why Some Things Glow |
| 56 | How Earthquakes Happen |
| 57 | Why We Laugh |
| 58 | How Mirrors Work |
| 59 | Why Stars Twinkle |
| 60 | How Your Heart Beats |

---

## QUALITY GUIDELINES

1. **Accuracy**: Facts must be correct
2. **Engagement**: Hook must grab attention in first 5 seconds
3. **Age-appropriate**: Match vocabulary to age group
4. **Universal**: Content works for all cultures (no US-centric references)
5. **Actionable**: Action phase must be doable at home with common items

---

## DO NOT

- Generate videos (Cursor does this)
- Deploy to Vercel (v0 does this)
- Change database schema
- Use "personalized" language (say "universal" instead)

---

## OUTPUT FORMAT

Provide scripts as JSON files or direct database inserts. Cursor will read them and generate videos.

---

## CONTACT

If you have questions or need clarification, include them in your output and Cursor will respond.
