# UNIFIED AGENT COORDINATION
**Date:** February 3, 2026  
**Status:** CRITICAL - ALL AGENTS MUST ALIGN

---

## AGENT LANES (STRICT BOUNDARIES)

| Agent | Primary Responsibility | Secondary | DOES NOT |
|-------|----------------------|-----------|----------|
| **Cursor** | Generate HeyGen videos, sync to DB, local scripts | Monitor pipelines | Deploy to Vercel |
| **v0** | Deploy to Vercel, fix API routes, frontend code | GitHub commits | Generate videos |
| **Antigravity** | Generate scripts/content, translations | Quality evaluation | Video generation, deployment |
| **Local Terminal** | Run batch jobs, long-running processes | File management | Code changes |

---

## CURRENT STATE (Feb 3, 2026 @ 12:35 PM PST)

### Database (CONFIRMED WORKING)
- **heygen_videos**: 49,550+ records
- **Days 1, 18, 31-34**: ALL have HeyGen video URLs (files2.heygen.ai)
- **Day 34 (today)**: 9 completed videos with REAL Curious Kelly
- **Days 31-33**: 15 videos completed and synced

### HeyGen
- **Credits**: ~600 minutes remaining
- **Correct API Key**: `sk_V2_hgu_kk1eiqopBWJ_cjFfMO48xHNTkgEtVvAJIhriXe73rQ6E`
- **Custom Avatar IDs**: In `heygen_talking_photo_ids.json`

### Deployment - CRITICAL ISSUE
- **GitHub Repo**: `nicoletterankin/v0-the-dl` - has the fix!
- **Commit**: `7de2f15` - "fix: infinite recursion in jsonResponse causing 500 errors"
- **GitHub shows**: "Production 2 minutes ago" with my commit
- **BUT API STILL FAILS**: Returns `tts_with_base_video` instead of `heygen_videos`

### The Problem
The database has the right data. The query works locally. But production API is not serving HeyGen videos.

**Root Cause Options:**
1. v0 didn't pull from GitHub before deploying
2. Vercel's GitHub integration deployed OLD code
3. There's edge caching we haven't cleared

### What v0 MUST Do
1. **Pull latest from GitHub** (click GitHub logo → Pull changes)
2. **Publish** (click Publish button)
3. **Wait 2 minutes** for edge cache to clear
4. **Test**: `curl "https://thedailylesson.com/api/video/url?day=34&phase=hook&age=30"`
5. **Verify**: Response should have `"source": "heygen_videos"` and URL starting with `https://files2.heygen.ai/`

---

## CUSTOM KELLY AVATAR IDS (USE THESE ONLY)

### Adult Archetypes
```json
{
  "architect": "afc54d3abfc04947bec026b9ec917ce8",
  "diplomat": "433ad96bf5d647d9964cecf784d008f6",
  "empath": "aa8b5eb1d711468a9a6e2085a4f8469c",
  "explorer": "45e5ef8b651846e0b62b7477e552e87b",
  "macgyver": "b9032c922c6e4e35b58a98abd499d060",
  "mystic": "a2b31ed0b5f84b0fa02d15d411735d3a",
  "provider": "06b78109ad22489ea2165ebbf180f77b",
  "rebel": "e614671b193c40f99772f7de5d1c51f7",
  "scientist": "7bb18cddacd44333813cc90ffa44f766",
  "storyteller": "9ffd06bd986a4e3086612921f3ac87ea",
  "strategist": "2411df8bdb0d40b088aa453d4c2a2d20",
  "survivor": "3f44bd33bfd1494d916d2746808a1a39"
}
```

### Elder Archetypes
```json
{
  "scientist": "d2a5133b931541e986912a37139a9398",
  "explorer": "5af13b2e9db14211a227f7e244b68e87",
  "rebel": "62e4ea7a26524e60b04b35a190dbc023",
  "architect": "b07df83db1bd4baaa7420ae792a6d35f",
  "diplomat": "0abbbc925b144ade83a41d650d23ee10",
  "empath": "6ed09093347d41f38f4d6638abd0a2c4",
  "macgyver": "f24e88a269a54c17a2dffc19eec13123",
  "mystic": "c76d77ffecc3461b87ea2fa0e21d719f",
  "provider": "380af536b170462a907f7692a74367cc",
  "storyteller": "817df044fe1c4f84a0de3aa00a296993",
  "strategist": "e12c985879b94ef3955ee1fc95f30810",
  "survivor": "a027f555728848a088324324c8f189e3"
}
```

### Voice ID
```
Kelly2: BbuMXx40WT4ZuAgRXvNx
```

---

## IMMEDIATE ACTIONS

### Cursor (DONE)
1. ✅ Generated Day 34 with REAL Curious Kelly (9 videos)
2. ✅ Generated Days 31-33 (15 videos synced)
3. ✅ Pushed jsonResponse bug fix to GitHub
4. ✅ Verified database has HeyGen URLs

### v0 (CRITICAL BLOCKER)
1. 🔴 **PULL FROM GITHUB** - Click GitHub logo → Pull changes
2. 🔴 **CLICK PUBLISH** - Deploy the fixed code
3. 🔴 **VERIFY** - Test that API returns HeyGen URLs

### Nicolette (URGENT ACTION NEEDED)
**Tell v0 to:**
1. Click the GitHub logo in v0's left sidebar
2. Click "Pull changes" or "Sync from GitHub"
3. Click the Publish button (top right corner)
4. Wait 2 minutes for deployment
5. Test: Visit https://thedailylesson.com and check if Kelly plays a HeyGen video

### Antigravity
1. ⏳ Generate scripts for Days 35-60
2. ⏳ Quality evaluation of generated videos

---

## ANTIGRAVITY BRIEFING

**To: Antigravity Agent**

### Your Current State
- You have been generating scripts and content
- Some of your output is in `lesson_perspectives` table
- We need you to focus on Days 35-60

### What We Need From You
1. **Scripts for Days 35-60** (5 phases each × 3 ages × 12 archetypes)
2. **Quality evaluation** of existing Day 1-34 content
3. **Translations** for ES, FR, DE, PT, ZH languages

### Data Format
Store in `lesson_perspectives` table:
```sql
INSERT INTO lesson_perspectives (
  day_number, age_group, archetype, language,
  hook_script, story_script, wonder_script, action_script, wisdom_script
) VALUES (...);
```

### Do NOT
- Generate videos (Cursor does this)
- Deploy to Vercel (v0 does this)
- Change database schema

---

## VIDEO GENERATION WORKFLOW

```
1. Antigravity generates scripts → lesson_perspectives table
2. Cursor reads scripts, generates HeyGen videos → heygen_videos table
3. v0 deploys API that reads heygen_videos → thedailylesson.com
4. Users see Kelly teaching with lip-synced video
```

---

## VERIFICATION CHECKLIST

After v0 deploys:
```bash
# Should return HeyGen URL, not Vercel Blob
curl "https://thedailylesson.com/api/video/url?day=34&phase=hook&age=30"

# Expected: {"url": "https://files2.heygen.ai/..."}
```

---

## COMMUNICATION PROTOCOL

1. **Cursor → v0**: Via this document and chat
2. **Cursor → Antigravity**: Via this document
3. **v0 → Cursor**: Via GitHub commits and chat responses
4. **All → Nicolette**: Status updates and blockers

---

## NEXT MILESTONE

**Goal:** Kelly plays lip-synced video on thedailylesson.com for Day 34

**Blockers:**
1. v0 needs to Publish
2. Vercel partial outage (Sandboxes) - may affect deployment

**ETA:** Within 1 hour if Publish happens now
