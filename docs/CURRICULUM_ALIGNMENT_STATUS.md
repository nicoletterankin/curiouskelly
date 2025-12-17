# 🔒 Curriculum Alignment Status — VERIFIED

**Status:** ✅ ALIGNED  
**Verified:** December 16, 2025  
**Authority:** This document confirms alignment between all curriculum data sources.

---

## Source of Truth

### Learn Track (Knowledge)
- **Source:** Supabase `core_lessons` table
- **Synced to:** `lessons/365_day_calendar.json`
- **Curriculum files:** `public/data/curriculum/year1-foundations/*.json`
- **Last sync:** December 1, 2025

### Grow Track (AI Fluency)
- **Source:** Generated curriculum (`lessons/year2-ai-fluency/*.json`)
- **Curriculum files:** `public/data/curriculum/year2-ai-fluency/*.json`
- **Created:** December 16, 2025

---

## File Structure

Both tracks have identical structure with 12 monthly curriculum files:

```
public/data/curriculum/
├── year1-foundations/          # Learn Track (🌟)
│   ├── january_curriculum.json
│   ├── february_curriculum.json
│   ├── march_curriculum.json
│   ├── april_curriculum.json
│   ├── may_curriculum.json
│   ├── june_curriculum.json
│   ├── july_curriculum.json
│   ├── august_curriculum.json
│   ├── september_curriculum.json
│   ├── october_curriculum.json
│   ├── november_curriculum.json
│   └── december_curriculum.json
│
└── year2-ai-fluency/           # Grow Track (🧠)
    ├── january_curriculum.json
    ├── february_curriculum.json
    ├── march_curriculum.json
    ├── april_curriculum.json
    ├── may_curriculum.json
    ├── june_curriculum.json
    ├── july_curriculum.json
    ├── august_curriculum.json
    ├── september_curriculum.json
    ├── october_curriculum.json
    ├── november_curriculum.json
    └── december_curriculum.json
```

---

## Day Mapping

| Month | Days | Learn Day Range | Grow Day Range |
|-------|------|-----------------|----------------|
| January | 31 | 1-31 | 1-31 |
| February | 28 | 32-59 | 32-59 |
| March | 31 | 60-90 | 60-90 |
| April | 30 | 91-120 | 91-120 |
| May | 31 | 121-151 | 121-151 |
| June | 30 | 152-181 | 152-181 |
| July | 31 | 182-212 | 182-212 |
| August | 31 | 213-243 | 213-243 |
| September | 30 | 244-273 | 244-273 |
| October | 31 | 274-304 | 274-304 |
| November | 30 | 305-334 | 305-334 |
| December | 31 | 335-365 | 335-365 |

**Total: 365 days per track = 730 daily topics**

---

## Sample Day Alignment (Day 1)

### Learn Track (🌟)
```json
{
  "day": 1,
  "date": "January 1",
  "title": "Starting Fresh",
  "learning_objective": "Nature renews itself through cycles of growth and decay."
}
```

### Grow Track (🧠)
```json
{
  "day": 1,
  "date": "January 1",
  "title": "I'm an AI - Understanding Your Digital Learning Partner",
  "learning_objective": "Develop foundational AI literacy..."
}
```

---

## Maintaining Alignment

### When updating Learn track topics:
1. Update in Supabase `core_lessons` table first
2. Run `python scripts/sync_supabase_to_calendar.py`
3. Run `python scripts/generate_learn_track_from_supabase.py`
4. Verify with `python scripts/verify_calendar_alignment.py`

### When updating Grow track topics:
1. Edit files in `lessons/year2-ai-fluency/`
2. Copy to `public/data/curriculum/year2-ai-fluency/`

---

## Scripts

| Script | Purpose |
|--------|---------|
| `sync_supabase_to_calendar.py` | Pull Supabase → `365_day_calendar.json` |
| `generate_learn_track_from_supabase.py` | Generate Learn track curriculum files |
| `verify_calendar_alignment.py` | Verify Supabase matches local files |

---

## Notes

1. **Supabase is the source of truth** for Learn track topics
2. The Grow track was generated manually and lives in the repository
3. Both tracks must always have 365 days
4. Day numbers must match between tracks (Day 1 = January 1 for both)

---

*This alignment was verified and locked on December 16, 2025.*
