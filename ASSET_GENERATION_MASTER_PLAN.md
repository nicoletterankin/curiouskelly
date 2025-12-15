# 🎬 ASSET GENERATION MASTER PLAN
## 365 Days × 5 Phases = 1,825 Educational Videos

---

## 📊 CURRENT STATUS

| Asset Type | Total Needed | Currently Have | Gap |
|------------|-------------|----------------|-----|
| **Core Lessons** (text) | 365 | ✅ 365 | Complete |
| **Lesson Atoms** (dialog) | 21,900 | ✅ 20,351 | ~93% |
| **Phase Visuals** (images) | 1,825 | ❌ 0 | 1,825 |
| **Thumbnails** | 365 | 31 | 334 |
| **Motion Videos** (generic) | 336 | ✅ 336 | Complete |
| **Per-Lesson Videos** | 1,825 | ❌ 0 | 1,825 |
| **Audio (TTS)** | 1,825 | ~100 | 1,725 |

---

## 🎯 GENERATION PIPELINE

### Phase 1: Phase Visuals (Images)
**Script:** `scripts/generate-all-phase-visuals.ts`
**Provider:** Replicate Flux + Kelly LoRA
**Cost:** ~$0.04/image × 1,825 = **~$73**
**Time:** ~2 seconds/image = ~1 hour

```bash
# Generate all 365 days, 5 phases each
npx ts-node scripts/generate-all-phase-visuals.ts --all

# Or by range:
npx ts-node scripts/generate-all-phase-visuals.ts --range=1-50
npx ts-node scripts/generate-all-phase-visuals.ts --range=51-100
...
```

### Phase 2: Thumbnails
**Script:** `scripts/kelly-visual-identity/generate-all-365-thumbnails.ts`
**Provider:** Replicate Flux
**Cost:** ~$0.04/image × 334 = **~$14**

```bash
# Generate missing thumbnails
npx ts-node scripts/kelly-visual-identity/generate-all-365-thumbnails.ts --missing
```

### Phase 3: Audio (ElevenLabs TTS)
**Script:** `scripts/generate-day-audio-elevenlabs.ts`
**Provider:** ElevenLabs
**Cost:** ~$0.24/minute × 1,825 phases × 1 min = **~$438**

```bash
# Generate audio for all phases
npx ts-node scripts/generate-day-audio-elevenlabs.ts --all
```

### Phase 4: Upload to Supabase Storage
**Script:** `scripts/upload-phase-assets-to-supabase.ts`
**Storage:** Supabase Storage buckets
**Cost:** Free (included in plan)

### Phase 5: Register in Database
**Script:** `scripts/populate-kelly-video-assets.ts`
**Tables:** `kelly_video_assets`, `core_lessons`

---

## 📁 STORAGE STRUCTURE

```
supabase/storage/
├── kelly-videos/
│   ├── motion/                    # Generic motion clips (336 exists)
│   │   ├── scientist/adult/hook.mp4
│   │   └── ...
│   └── production/                # Per-lesson videos (future)
│       ├── day_001/
│       │   ├── hook.mp4
│       │   ├── q1.mp4
│       │   └── ...
│       └── ...
├── kelly-templates/
│   └── heygen/archetypes-head-only/
│       └── kelly_scientist_head.png   # 60 head images (exists)
├── lesson-visuals/                # Phase visuals (TO GENERATE)
│   ├── day_001/
│   │   ├── hook.png
│   │   ├── q1.png
│   │   ├── q2.png
│   │   ├── q3.png
│   │   └── wisdom.png
│   └── ...
└── lesson-thumbnails/            # Thumbnails (334 to generate)
    ├── day_001.webp
    └── ...
```

---

## 🗄️ DATABASE SCHEMA

### kelly_video_assets (for phase visuals/videos/audio)
```sql
id, day_number, phase, template, asset_type, age_bucket, language,
storage_bucket, storage_path, public_url, resolution, status,
quality_tier, created_at, updated_at
```

### core_lessons (update with URLs)
```sql
-- Add/update these columns:
hero_image_url    -- Phase hook image URL
thumbnail_url     -- Card thumbnail URL  
demo_video_url    -- Optional demo video
```

---

## 🚀 EXECUTION ORDER

### TODAY (Immediate)
1. ✅ Populate `lesson_atoms` visual URLs from existing assets
2. 🔄 Generate Phase Visuals for Days 1-50
3. 🔄 Upload to Supabase storage
4. 🔄 Register in `kelly_video_assets`

### THIS WEEK
5. Generate Phase Visuals for Days 51-365
6. Generate missing Thumbnails (334)
7. Update `core_lessons.thumbnail_url`

### NEXT WEEK
8. Generate Audio (ElevenLabs) for all phases
9. Generate HeyGen videos (if budget allows)

---

## 💰 COST SUMMARY

| Item | Count | Unit Cost | Total |
|------|-------|-----------|-------|
| Phase Visuals | 1,825 | $0.04 | $73 |
| Thumbnails | 334 | $0.04 | $14 |
| Audio (ElevenLabs) | 1,825 min | $0.24/min | $438 |
| **TOTAL** | | | **$525** |

Optional:
- HeyGen Videos: 1,825 × ~$0.50 = **$912** (use motion library instead)

---

## 🔧 ENVIRONMENT SETUP

```bash
# Required environment variables (.env)
REPLICATE_API_TOKEN=r8_xxx
SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJxxx
GOOGLE_AI_API_KEY=AIzaSyxxx  # For Gemini
ELEVENLABS_API_KEY=sk_xxx    # For audio
```

---

## 📝 IMMEDIATE ACTION ITEMS

1. **Run phase visual generator for first batch:**
   ```bash
   npx ts-node scripts/generate-all-phase-visuals.ts --range=1-10
   ```

2. **Upload generated images to Supabase:**
   ```bash
   npx ts-node scripts/upload-phase-assets-to-supabase.ts --range=1-10
   ```

3. **Register in database:**
   ```bash
   npx ts-node scripts/register-phase-assets.ts --range=1-10
   ```

4. **Verify in UI:**
   - Open https://www.curiouskelly.com/learn.html?day=1
   - Check that phase visuals appear

---

*Generated: December 14, 2025*
*Author: Asset Generation Pipeline*
