# MVP GENERATION SPEC
## Unified Asset Requirements for 365 Daily Lessons

**Created:** Dec 19, 2024  
**Purpose:** Define exactly what assets the website needs to play a complete lesson end-to-end.

---

## 🎯 MVP SCOPE (LOCKED)

### Per-Day Requirements
Each day requires assets for **3 archetypes × 7 phases = 21 lesson atoms**.

**Archetypes (MVP Set):**
1. The Explorer
2. The Rebel  
3. The Scientist

**Phases (Canonical 7):**
1. Hook - Surprising question to spark curiosity
2. Cliff - Choice point (A/B path selection)
3. Fact1 - Foundation insight
4. Fact2 - Deeper evidence
5. Fact3 - Real-world application
6. Wisdom - Essential takeaway
7. Outro - Celebration + tomorrow preview

---

## 📦 ASSET REQUIREMENTS PER ATOM

### Required (MUST have for lesson to function):
| Asset | Storage Location | Notes |
|-------|-----------------|-------|
| `script` | `lesson_atoms.content.script` | Kelly's spoken text |
| `options` | `lesson_atoms.content.options` | EXACTLY 2 options (A/B) |
| `simulatedComments` | `lesson_atoms.content.simulatedComments` | 2-3 comments per phase |

### Video Assets (SHOULD have for polished experience):
| Phase | Video Source | Fallback |
|-------|-------------|----------|
| Hook | HD Pipeline (ElevenLabs + Flux + MiniMax + SyncLabs) | Static Kelly + TTS |
| Cliff | **FALLBACK ONLY** - Static image + audio | Static Kelly + TTS |
| Fact1 | HD Pipeline | Static Kelly + TTS |
| Fact2 | HD Pipeline | Static Kelly + TTS |
| Fact3 | HD Pipeline | Static Kelly + TTS |
| Wisdom | HD Pipeline | Static Kelly + TTS |
| Outro | **FALLBACK ONLY** - Static image + audio | Static Kelly + TTS |

**Key Insight:** HD Pipeline covers 5/7 phases. Cliff and Outro use audio-only fallback.

---

## 🔧 GENERATION PIPELINES

### 1. Content Generation (AI-Assisted)
**Input:** Day number, topic from `core_lessons`  
**Output:** `lesson_atoms` with scripts, options, comments  
**Script:** `scripts/generate-lesson-content.ts` (to be created if needed)

```
Required per atom:
- script: 50-200 words
- options: array of exactly 2 objects [{id, text, feedback}]
- simulatedComments: array of 2-3 objects [{username, text, reactions}]
```

### 2. HD Video Pipeline (Golden Standard)
**Script:** `scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts`  
**Phases Covered:** Hook, Fact1, Fact2, Fact3, Wisdom (5 phases)  
**Output:** `lesson_atoms.hd_video_url`

```bash
# Generate for one day, all MVP archetypes
npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day=353
```

**Pipeline Steps:**
1. ElevenLabs → Audio (Kelly voice)
2. Replicate Flux + LoRA → Source image (Kelly with persona styling)
3. MiniMax Video-01 → Motion video
4. Sync Labs → Lip-synced final video
5. Upload to Supabase → `kelly-videos/day-{N}/{archetype}/{phase}.mp4`
6. Update `lesson_atoms.hd_video_url`

### 3. Fallback Audio Pipeline (Cliff + Outro)
**For phases not covered by HD Pipeline:**
- Frontend automatically uses TTS with static Kelly image
- No additional generation needed
- Consider future HeyGen generation for polish

---

## 📊 GENERATION CHECKLIST PER DAY

Before a day is "complete", verify:

```sql
-- Run this query to check day N completeness
SELECT 
  la.archetype,
  COUNT(*) as total_phases,
  SUM(CASE WHEN la.content->>'script' IS NOT NULL THEN 1 ELSE 0 END) as has_script,
  SUM(CASE WHEN jsonb_array_length(la.content->'options') >= 2 THEN 1 ELSE 0 END) as has_options,
  SUM(CASE WHEN la.hd_video_url IS NOT NULL THEN 1 ELSE 0 END) as has_video
FROM lesson_atoms la 
JOIN core_lessons cl ON la.core_lesson_id = cl.id 
WHERE cl.day_number = 353  -- Change day number
  AND la.archetype IN ('The Explorer', 'The Rebel', 'The Scientist')
GROUP BY la.archetype;
```

**Expected Results:**
| archetype | total_phases | has_script | has_options | has_video |
|-----------|-------------|------------|-------------|-----------|
| The Explorer | 7 | 7 | 7 | 5 |
| The Rebel | 7 | 7 | 7 | 5 |
| The Scientist | 7 | 7 | 7 | 5 |

---

## 🚀 BATCH GENERATION COMMANDS

### Generate All Assets for Day N:
```bash
# Step 1: Ensure content exists (scripts, options, comments)
# (Manual verification or content generation script)

# Step 2: Generate HD videos for 5 phases
npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day=N

# Step 3: Verify completion
# (Run SQL query above)
```

### Generate Full Year (Future):
```bash
# Generate days 1-365 (run sequentially to manage API costs)
for day in {1..365}; do
  npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day=$day
  sleep 60  # Rate limit protection
done
```

---

## ⚠️ KNOWN LIMITATIONS

1. **Cliff/Outro phases have no HD video** - By design, uses audio fallback
2. **HD Pipeline requires ~10 min per archetype** - Plan for batch processing
3. **API costs**: ElevenLabs, Replicate, Sync Labs all have per-use costs
4. **Only 3 archetypes for MVP** - Other 9 archetypes deferred

---

## 📋 DATA VALIDATION RULES

### lesson_atoms table constraints:
- `content.script`: NOT NULL, min 10 characters
- `content.options`: Array with EXACTLY 2 elements
- `content.simulatedComments`: Array with 2-3 elements
- `hd_video_url`: Can be NULL (fallback to audio)

### Frontend fallback behavior:
1. If `hd_video_url` exists → Play HD video
2. If no video → Show static Kelly image + TTS audio
3. Always show options after phase completes

---

## 🔄 MAINTENANCE WORKFLOW

### Daily Tasks:
1. Generate tomorrow's content (if not pre-generated)
2. Run HD pipeline for tomorrow's videos
3. Verify completion with SQL query

### Weekly Tasks:
1. Audit past week's lessons for completeness
2. Review any failed generations
3. Update this spec if patterns change

---

## 📁 FILE STRUCTURE

```
/scripts/
  kelly-video-factory/
    hd-golden-lesson-pipeline.ts    # Main HD video pipeline
  generate-day-videos-heygen.ts     # Alternative HeyGen pipeline (backup)
  generate-all-phase-visuals.ts     # Static image generation

/generated-videos/
  golden-lesson-hd/
    day_{N}_{phase}_{archetype}/
      final_hd.mp4

/public/kelly/phases/{day}/         # Static phase visuals (not video)
```

---

## ✅ SUCCESS CRITERIA

A day is "MVP Complete" when:
- [ ] All 21 atoms exist (3 archetypes × 7 phases)
- [ ] All atoms have scripts (50-200 words)
- [ ] All atoms have exactly 2 options
- [ ] All atoms have 2-3 simulated comments
- [ ] 15 atoms have HD videos (5 phases × 3 archetypes)
- [ ] 6 atoms use fallback (Cliff + Outro × 3 archetypes)
- [ ] Lesson plays end-to-end without errors
