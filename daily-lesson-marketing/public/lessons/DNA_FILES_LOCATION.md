# DNA Lesson Files Location Guide

## Overview

All DNA lesson files have been consolidated into the `lessons/` directory for easier access and calendar generation.

## Current DNA Files in `lessons/`

### Kebab-case naming (newer format):
- `applied-mathematics-math-in-the-real-world-dna.json`
- `creative-writing-dna.json`
- `dance-expression-dna.json`
- `genetic-engineering-editing-the-code-of-life-dna.json`
- `molecular-biology-dna.json` ✅
- `negotiation-skills-dna.json`
- `nutrition-science-dna.json`
- `poetry-dna.json`
- `the-sun-dna.json`

### Snake_case naming (older format):
- `aging_process_dna.json`
- `disruptive_innovation_dna.json`
- `parasitology_dna.json`
- `plasma_physics_dna.json`
- `stem_cells_dna.json`

## Total: 14 DNA Files

## Archive Location

Older versions of some DNA files are archived in:
- `lessons/archive/` - Contains older versions with snake_case naming

## Source Locations

DNA files were originally scattered across:
1. **Primary**: `curious-kellly/backend/config/lessons/` (active development)
2. **Secondary**: `lessons/` (some files)
3. **Archive**: `lessons/archive/` (old versions)

## Consolidation

All DNA files from `curious-kellly/backend/config/lessons/` have been copied to `lessons/` to:
- Centralize all lesson files in one location
- Make calendar generation easier
- Ensure all DNA lessons are discoverable

## Calendar Integration

The `generate_unified_calendar.py` script now:
1. Checks `lessons/` directory first (primary location)
2. Handles both kebab-case and snake_case naming
3. Falls back to `curious-kellly/backend/config/lessons/` if needed

## DNA Files Not Yet in Calendar

Some DNA files may not be automatically mapped to calendar days. To add them:

1. Edit `generate_unified_calendar.py`
2. Add to `DNA_LESSON_MAPPINGS` dictionary with day number
3. Or add title keywords to `title_keywords` dictionary

Example:
```python
DNA_LESSON_MAPPINGS = {
    "molecular-biology": 189,  # Explicit day mapping
    "puppies": None,  # Will be detected by title keywords
}
```

