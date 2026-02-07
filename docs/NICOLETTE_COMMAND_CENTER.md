# NICOLETTE COMMAND CENTER
**Date:** February 3, 2026  
**Status:** ALL SYSTEMS GO  
**Goal:** Use ALL resources TODAY to maximize video output

---

## 🚨 IMMEDIATE COMMANDS TO RUN

Open PowerShell in `C:\Users\user\UI-TARS-desktop` and run:

### Command 1: Check All Pipeline Status
```powershell
npx tsx scripts/monitor-all-pipelines.ts
```

### Command 2: Use HeyGen Credits (600+ available)
```powershell
npx tsx scripts/heygen-use-all-credits.ts --status
npx tsx scripts/heygen-use-all-credits.ts --days=1-7
```

### Command 3: Check Sync Labs Progress
```powershell
npx tsx scripts/batch-lipsync-pipeline.ts --status
```

---

## CURRENT SYSTEMS STATUS

| System | Status | What It's Doing |
|--------|--------|-----------------|
| **Sync Labs** | 🟢 RUNNING | 40/1000 videos done (~3 min each) |
| **Antigravity** | 🟢 RUNNING | Days 13-33, 6 languages |
| **v0.app** | ✅ DEPLOYED | Fixed "Universal" copy |
| **HeyGen** | ⚠️ 600+ CREDITS | MUST USE TODAY |
| **fal.ai** | 🟡 STANDBY | Ready for MuseTalk/LatentSync |

---

## PRIORITY 1: USE HEYGEN CREDITS NOW (600+ credits)

HeyGen produces the HIGHEST QUALITY lip-sync. Use credits for flagship content.

### What to Generate with HeyGen:

**Target:** Days 1-7 (first week) × All 5 phases × Adult = **35 videos**
- These are the videos new users see first
- They MUST be perfect quality
- Cost: ~35 credits (we have 600+)

### Click-by-Click Instructions:

**Step 1: Open HeyGen Dashboard**
1. Go to https://app.heygen.com
2. Click "Create" → "Avatar Video"
3. Select the Kelly avatar (super-elder or adult version you see in "Recent creations")

**Step 2: For EACH video, use this process:**
1. **Avatar:** Select Kelly adult (the one with glasses you've been using)
2. **Script:** Copy from Antigravity output OR from database
3. **Voice:** Use ElevenLabs voice ID for Kelly
4. **Duration:** Should match the audio file duration
5. **Click "Generate"**

**Step 3: Batch Queue Strategy**
Since you have 600+ credits:
- Generate Day 1 all 5 phases: hook, story, wonder, action, wisdom
- Generate Day 2 all 5 phases
- Continue through Day 7
- Then Days 30, 100, 365 (milestone days)

### Videos to Generate (Priority Order):

| Priority | Day | Phases | Archetype | Credits |
|----------|-----|--------|-----------|---------|
| 1 | Day 1 | All 5 | scientist | 5 |
| 2 | Day 2 | All 5 | explorer | 5 |
| 3 | Day 3 | All 5 | diplomat | 5 |
| 4 | Day 4 | All 5 | architect | 5 |
| 5 | Day 5 | All 5 | storyteller | 5 |
| 6 | Day 6 | All 5 | mystic | 5 |
| 7 | Day 7 | All 5 | rebel | 5 |
| 8 | Day 30 | All 5 | strategist | 5 |
| 9 | Day 100 | All 5 | provider | 5 |
| 10 | Day 365 | All 5 | survivor | 5 |
| **TOTAL** | | | | **50 credits** |

That leaves 550+ credits for:
- Kid and Elder age variants
- Other languages (ES, FR, PT)
- Additional days

---

## PRIORITY 2: LET SYNC LABS CONTINUE

The Sync Labs pipeline is running and producing good results.

**Current Status:**
- 40/1000 complete
- Uploading to: `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-videos/`
- Rate: ~20 videos/hour

**DO NOT STOP IT.** Let it run overnight. By morning you'll have ~500 more videos.

---

## PRIORITY 3: ANTIGRAVITY CONTENT GENERATION

Antigravity is generating scripts for Days 13-33 in 6 languages.

**What It Produces:**
- Scripts for each phase (hook, story, wonder, action, wisdom)
- Multiple language versions (en, es, fr, de, pt, zh)
- These scripts feed into the video pipeline

**Action:** Let it finish. Check back in 30 minutes.

---

## SCREENSHOTS I NEED FROM YOU

### Screenshot 1: HeyGen Projects Page
**Why:** I need to see ALL existing videos and their names
**How:**
1. Go to https://app.heygen.com/projects
2. Take screenshot of all projects/folders

### Screenshot 2: HeyGen Video List
**Why:** I need the exact video IDs to map to database
**How:**
1. Click on any project that has Kelly videos
2. Take screenshot showing video names and thumbnails

### Screenshot 3: Cloudflare R2 Avatars Bucket
**Why:** I need to see what's stored in the 1.45GB avatars bucket
**How:**
1. Go to https://dash.cloudflare.com
2. Click R2 → avatars bucket
3. Take screenshot of file listing

### Screenshot 4: Antigravity Output
**Why:** I need to see the generated scripts to feed into HeyGen
**How:**
1. In Antigravity, look at the `outputs/` folder
2. Show me a sample Day 1 or Day 21 output JSON

### Screenshot 5: Sync Labs Terminal (Final State)
**Why:** I need to see the current progress and any errors
**How:**
1. Take screenshot of the Sync Labs terminal window
2. Scroll up to see any error messages if present

---

## PARALLEL EXECUTION PLAN

**RIGHT NOW (You):**
1. Start HeyGen batch for Day 1 (5 videos)
2. Take the 5 screenshots I requested
3. Let Sync Labs and Antigravity continue running

**RIGHT NOW (Cursor):**
1. Create script to download HeyGen videos as they complete
2. Create script to monitor all pipelines
3. Update database as videos complete

**NEXT HOUR:**
1. HeyGen: Days 1-3 generating
2. Sync Labs: Progress to 60/1000
3. Antigravity: Days 13-33 complete

**TONIGHT:**
1. HeyGen: 50+ flagship videos done
2. Sync Labs: 500+ videos done
3. fal.ai: Start bulk generation for remaining

---

## FAL.AI BACKUP PLAN

If HeyGen or Sync Labs hits limits, switch to fal.ai:

**MuseTalk on fal.ai:**
- Cost: ~$0.05/video
- Quality: B+ (good enough for non-flagship)
- Speed: Fast (1-2 min/video)

**Command:**
```bash
npx tsx scripts/fal-lipsync-batch.ts --day 1 --provider musetalk
```

---

## DATABASE UPDATE FLOW

All systems should update the SAME database:

**Neon PostgreSQL** (NOT Supabase):
```
kelly_lesson_assets table:
- video_url → final video URL
- video_source → 'heygen' | 'sync_labs' | 'fal_musetalk'
- status → 'completed'
```

**Supabase Storage** (for file storage):
```
kelly-videos bucket:
- lipsync/2026/en/day-XXX/phase-ageXX.mp4
```

---

## END OF DAY GOAL

By midnight tonight:

| Metric | Target |
|--------|--------|
| HeyGen videos | 50+ (flagship quality) |
| Sync Labs videos | 500+ (good quality) |
| Total lip-synced | 550+ videos |
| Days covered | 1-7 fully, 8-100 partially |
| Languages | EN complete, ES/FR started |

---

## EMERGENCY CONTACTS

If something breaks:

1. **Sync Labs fails:** Check API key, rate limits
2. **HeyGen fails:** Check credits, avatar status
3. **Database fails:** Check Neon connection string
4. **Storage fails:** Check Blob/Supabase tokens

---

**START NOW: Generate Day 1 on HeyGen while taking screenshots.**
