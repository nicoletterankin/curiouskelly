# MASTER VIDEO COORDINATION DOCUMENT
## February 3, 2026 - All Systems Aligned

---

## 🎯 SHARED GOAL
Get Kelly lip-synced videos playing on thedailylesson.com for all 365 days.

---

## 📊 CURRENT STATE

| System | Status | Notes |
|--------|--------|-------|
| HeyGen Credits | 668.5 available | 3,279 videos in history |
| HeyGen Webhook | ✅ Verified | Points to thedailylesson.com/api/webhooks/heygen |
| Kelly on Site | ⚠️ PARTIAL | Static image + audio (no lip-sync video) |
| Database | Neon PostgreSQL | Multiple video tables need sync |

---

## 🔴 CRITICAL GAP IDENTIFIED

```
HeyGen generates video → Webhook fires → Updates `video_jobs` table
                                              ↓
                         BUT playback reads from `heygen_videos` table
                                              ↓
                              VIDEOS DON'T APPEAR ON SITE
```

**SOLUTION:** Webhook must also update `heygen_videos` table, OR playback must check `video_jobs`.

---

## 👥 ROLE ASSIGNMENTS

### 🤖 V0.APP - "The Deployer"
**Responsibilities:**
1. Deploy latest code to Vercel production
2. Ensure `/api/video/url` route checks ALL video tables:
   - `heygen_videos` (primary)
   - `video_jobs` (webhook writes here)
   - `kelly_lesson_assets` (legacy)
3. Update webhook handler to write to `heygen_videos` table too

**Immediate Action:**
```typescript
// In /api/webhooks/heygen - ADD THIS after updating video_jobs:
await sql`
  INSERT INTO heygen_videos (
    day_of_year, phase, age_category, archetype, 
    heygen_video_id, video_url, status
  ) VALUES (
    ${day}, ${phase}, ${age}, ${archetype},
    ${video_id}, ${video_url}, 'completed'
  )
  ON CONFLICT (day_of_year, phase, age_category, archetype, language)
  DO UPDATE SET video_url = ${video_url}, status = 'completed'
`;
```

---

### 🌌 ANTIGRAVITY - "The Content Creator"
**Responsibilities:**
1. Generate scripts for all 365 days × 5 phases × 3 ages × 6 languages
2. Store in `lesson_perspectives` table
3. Ensure scripts match Kelly's voice and tone

**Current Status:** 
- Day 1-10 scripts generated ✅
- Need: Days 11-365

**Output Format:**
```json
{
  "day": 1,
  "phase": "hook",
  "age_group": "adult",
  "language": "en",
  "script": "Welcome to today's lesson..."
}
```

---

### 👩‍💻 NICOLETTE - "The Commander"
**Responsibilities:**
1. Monitor all pipelines from correct directory
2. Verify videos play on production site
3. Approve video quality

**IMPORTANT - Always run commands from:**
```powershell
cd C:\Users\user\UI-TARS-desktop
```

**Monitor Commands:**
```powershell
# Check HeyGen batch progress
Get-Content heygen-batch-output.log -Tail 50

# Check credits
node -e "import('dotenv').then(d=>d.config());fetch('https://api.heygen.com/v2/user/remaining_quota',{headers:{'X-Api-Key':process.env.HEYGEN_API_KEY}}).then(r=>r.json()).then(d=>console.log('Credits:',d.data?.remaining_quota))"

# Verify site is working
Start-Process "https://thedailylesson.com"
```

---

### 🔧 CURSOR - "The Integrator"
**Responsibilities:**
1. Create scripts that poll HeyGen for completed videos
2. Download videos to Vercel Blob
3. Update ALL relevant database tables
4. Ensure data flows correctly between systems

---

## 📋 UNIFIED DATABASE STRATEGY

### Tables That Need Video URLs:

| Table | Purpose | Who Writes | Who Reads |
|-------|---------|------------|-----------|
| `heygen_videos` | Main queue | HeyGen batch script | Video API |
| `video_jobs` | Job tracking | Webhook | Admin dashboard |
| `kelly_lesson_assets` | Legacy storage | Migration script | Video API fallback |

### Sync Strategy:
When a video completes, update ALL THREE tables with the same URL.

---

## 🚀 ACTION PLAN

### Phase 1: Fix Data Flow (NOW)
1. [CURSOR] Create sync script to copy video_jobs → heygen_videos
2. [V0] Update webhook to write to both tables
3. [NICOLETTE] Verify videos appear after sync

### Phase 2: Continue Generation (PARALLEL)
1. [CURSOR] Keep HeyGen batch running for Days 1-30
2. [ANTIGRAVITY] Generate scripts for Days 31-365
3. [V0] Monitor production for errors

### Phase 3: Scale Up (AFTER PHASE 1 WORKS)
1. Add more ages (kid, senior)
2. Add more languages (es, fr, de, pt, zh)
3. Use remaining credits strategically

---

## 📞 COMMUNICATION PROTOCOL

When updating status, use this format:
```
[SYSTEM_NAME] [TIMESTAMP]
Status: [WORKING|BLOCKED|COMPLETE]
Progress: X/Y items done
Blockers: [list any issues]
Next: [what you're doing next]
```

---

## ✅ SUCCESS CRITERIA

Kelly plays lip-synced videos on thedailylesson.com:
- [ ] Day 1 all phases work
- [ ] Days 1-7 all phases work  
- [ ] Days 1-30 hook phase works
- [ ] Debug HUD shows "VIDEO: heygen_videos" not "NOTHING"

---

*Last Updated: February 3, 2026 10:00 AM*
*Owner: All Systems*
