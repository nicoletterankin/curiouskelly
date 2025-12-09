# Curriculum Enhancement Roadmap

**Created:** December 1, 2025  
**Status:** 📋 RECOMMENDATIONS

---

## Current State Assessment

### ✅ What's Working
- **365 lessons** in Supabase `core_lessons`
- **21,915 lesson atoms** with interactive content
- **38,700 lesson shards** for demographic targeting
- **Single source of truth** architecture (Supabase)
- **Sync tooling** to maintain local mirrors

### ⚠️ What Needs Attention
- Some topics may be thin on content (need review)
- Archetype coverage may be uneven
- Age-variant content quality varies
- No audio/video assets generated yet
- Missing DNA-style rich metadata (translations, voice profiles)

---

## Recommended Enhancements

### Phase 1: Content Audit (Week 1)

**Goal:** Understand what we have and identify gaps

```bash
# Run these to audit current state
node daily-lesson-marketing/backup_full.js  # Fresh backup
python scripts/audit_lessons.py              # Audit script
```

**Deliverables:**
1. **Content Inventory Report**
   - Lessons with < 3 atoms
   - Lessons missing key phases (Fact1, Fact2, Fact3, Wisdom)
   - Lessons missing archetypes

2. **Quality Assessment**
   - Script length analysis
   - Response quality check
   - Age-appropriateness review

3. **Gap Analysis**
   - Topics needing more depth
   - Missing subject areas
   - Seasonal/calendar alignment issues

---

### Phase 2: Content Enrichment (Weeks 2-3)

**Goal:** Fill gaps and improve quality

#### 2.1 Generate Missing Atoms

For lessons with incomplete phases:

```python
# Example: Generate missing Fact3 for Day 15
prompt = f"""
Topic: {lesson.topic}
Universal Truth: {lesson.universal_truth}
Archetype: The Explorer

Generate a Fact3 phase that:
- Reveals a surprising/delightful fact
- Connects to real-world applications
- Ends with natural transition to Wisdom

Format:
{{
  "script": "...",
  "options": ["A: ...", "B: ...", "C: ..."],
  "responses": {{ "A": "...", "B": "...", "C": "..." }}
}}
"""
```

#### 2.2 Add Missing Archetypes

Current archetypes in database:
- The Survivor
- The Explorer  
- The Scientist
- The Storyteller
- (need to verify full list)

**Action:** Ensure each lesson has atoms for at least 3 archetypes.

#### 2.3 Enhance Learning Objectives

Many lessons have generic objectives. Improve with:
- Bloom's Taxonomy alignment
- Measurable outcomes
- Age-specific expectations

---

### Phase 3: Multilingual Expansion (Weeks 4-5)

**Goal:** Add Spanish and French content

#### 3.1 Create Translation Pipeline

```python
def translate_lesson_atoms(lesson_id, target_language):
    """
    Translate all atoms for a lesson to target language.
    Uses professional translation + AI review.
    """
    atoms = supabase.table("lesson_atoms")\
        .select("*")\
        .eq("core_lesson_id", lesson_id)\
        .execute()
    
    for atom in atoms.data:
        translated = translate_content(atom.content, target_language)
        # Store in lesson_shards with region=es/fr
```

#### 3.2 Translation Priority
1. First 30 days (January)
2. High-engagement lessons (based on analytics)
3. Complete calendar

---

### Phase 4: Age-Variant Content (Weeks 6-8)

**Goal:** Create truly age-appropriate variants

#### 4.1 Age Buckets
| Bucket | Age Range | Voice Character | Complexity |
|--------|-----------|-----------------|------------|
| Early Childhood | 2-5 | Playful, simple | Concrete, sensory |
| Elementary | 6-12 | Curious, encouraging | Cause-effect |
| Teen | 13-17 | Peer-like, challenging | Abstract intro |
| Young Adult | 18-35 | Professional, engaging | Full complexity |
| Midlife | 36-60 | Respectful, practical | Application focus |
| Wisdom Years | 61-102 | Warm, dignified | Legacy & meaning |

#### 4.2 Implementation

Store in `lesson_shards`:
```json
{
  "core_lesson_id": "uuid",
  "age": 7,
  "region": "en",
  "tone": "curious",
  "script_content": {
    "welcome": "Hey there, explorer! Ready to discover...",
    "fact1": "Did you know that...",
    // Age-appropriate language throughout
  }
}
```

---

### Phase 5: Audio Generation (Weeks 9-10)

**Goal:** Generate Kelly voice audio for all content

#### 5.1 Prerequisites
- ElevenLabs API key configured
- Kelly voice ID: `wAdymQH5YucAkXwmrdL0`
- Storage configured (Supabase Storage or CDN)

#### 5.2 Generation Pipeline

```python
# For each lesson atom
for atom in lesson_atoms:
    audio = elevenlabs.generate(
        text=atom.content.script,
        voice_id=KELLY_VOICE_ID,
        model="eleven_multilingual_v2"
    )
    
    # Store in Supabase Storage
    path = f"audio/{atom.core_lesson_id}/{atom.phase}.mp3"
    supabase.storage.upload(path, audio)
    
    # Update atom with audio URL
    atom.audio_url = get_public_url(path)
```

#### 5.3 Audio Manifest
Create `audio_manifest.json` tracking all generated audio:
```json
{
  "generated_at": "2025-12-XX",
  "total_files": 21915,
  "total_duration_hours": 45.3,
  "storage_size_gb": 12.4
}
```

---

### Phase 6: Visual Assets (Weeks 11-12)

**Goal:** Generate lesson thumbnails and illustrations

#### 6.1 Thumbnail Generation
Using existing Kelly image generation pipeline:
```python
for lesson in core_lessons:
    prompt = f"Kelly teaching about {lesson.topic}, {lesson.icon_emoji}"
    thumbnail = generate_kelly_image(prompt)
    upload_to_storage(f"thumbnails/{lesson.day_number}.webp", thumbnail)
```

#### 6.2 Lesson Illustrations
- Scene illustrations for each phase
- Interactive choice visualizations
- Age-appropriate art styles

---

## Quick Wins (Do This Week)

### 1. Topic Name Review
Some topics could be more engaging:

| Current | Suggested |
|---------|-----------|
| "Starting Fresh" | "New Beginnings: The Power of Starting Fresh" |
| "The Three Lives of Water" | Keep (great title!) |
| "Where Clouds Come From" | "Cloud Factories: Where Clouds Are Born" |

### 2. Add Universal Truths
Ensure every lesson has a compelling `universal_truth` that:
- Is timeless and cross-cultural
- Connects to human experience
- Inspires curiosity

### 3. Marketing Copy Audit
Review `marketing_headline`, `marketing_tagline`, `marketing_pitch`:
- Remove generic language
- Add hooks and curiosity gaps
- Ensure age-appropriate tone

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Lessons with 4+ atoms | ? | 365 (100%) |
| Lessons with 3+ archetypes | ? | 365 (100%) |
| Translations (ES/FR) | 0 | 365 each |
| Age variants per lesson | 1? | 6 |
| Audio files generated | 0 | 21,915 |
| Thumbnails generated | ? | 365 |

---

## Budget Estimates

| Item | Cost | Notes |
|------|------|-------|
| ElevenLabs audio | ~$500-1000 | 21,915 atoms × avg 30 sec |
| Translation services | ~$2000-5000 | Professional review |
| Image generation | ~$100-200 | Vertex AI / DALL-E |
| Storage (annual) | ~$50-100 | Supabase Storage |

---

## Next Steps

1. **Run content audit** to establish baseline
2. **Prioritize gaps** based on user journey (Day 1-30 first)
3. **Generate missing atoms** for incomplete lessons
4. **Start audio pipeline** for January lessons
5. **Create translation workflow** for Spanish first

---

## Related Documents

- `docs/DATA_ARCHITECTURE.md` - Data sources and sync
- `docs/backend/SUPABASE_SCHEMA.md` - Database schema
- `CLAUDE.md` - Operating rules for AI contributions
- `scripts/sync_supabase_to_calendar.py` - Sync tooling






