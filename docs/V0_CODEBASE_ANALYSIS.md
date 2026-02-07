# V0 CODEBASE ANALYSIS
**Date:** February 3, 2026  
**Source:** full-audit-and-evals.zip (extracted from v0.app)

---

## CRITICAL FINDING: DEPLOYMENT IS BROKEN

From v0's BACKLOG REPORT:
- **Video API failing** - BROKEN
- **Kelly shows static image** - BROKEN (no video plays)
- **v0 → Vercel sync** - UNRELIABLE

### Root Cause:
> "The deployed production code is not the same as v0 codebase."

The database columns `video_url` and `subtitle` DO EXIST, but production still throws "column does not exist" errors because:
1. Production deployment is running OLD code
2. v0 changes are NOT reaching Vercel
3. Multiple redeploys haven't fixed it

---

## VIDEO URL API PRIORITY CHAIN

From `/app/api/video/url/route.ts`:

```
PRIORITY 0: heygen_videos (main video source)
    ↓ fallback
PRIORITY 1: generated_assets (audio/video assets)
    ↓ fallback
PRIORITY 2: kelly_lesson_assets (ElevenLabs TTS + lip-sync)
    ↓ fallback
PRIORITY 3: lesson_perspectives (personalized scripts)
    ↓ fallback
PRIORITY 4: lessons table (base scripts + on-demand TTS)
    ↓ fallback
FALLBACK: VERIFIED base videos (ALWAYS returns working video)
```

---

## VERIFIED WORKING BASE VIDEOS

These URLs are confirmed working (HTTP 200):

```typescript
const KELLY_BASE_VIDEOS = {
  excited: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/0a437abfa17e46d2a3f2c9a8f27de9ee.mp4',
  default: 'https://z4yuma7kj5h9td7v.public.blob.vercel-storage.com/kelly-base-videos/uncategorized/064dbfab193c461fbb2869f27d663c7b.mp4',
}
```

Phase-to-video mapping:
- hook → excited
- story → default
- wonder → excited
- action → default
- wisdom → default

---

## COMPLETE HEYGEN AVATAR MAPPINGS

### Adult Kelly (12 archetypes)
```typescript
HEYGEN_ADULT_KELLY_LOOKS = {
  storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
  explorer: '62516885ca4b4eae8f63b87b8c060e25',
  scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
  architect: '35d0115505824e3182eb9d2ee8cfe73d',
  strategist: '08d53d1b065041bda2e5b6bc32962a8a',
  diplomat: 'c3cdbe48fe274420a7f45a4da7e366aa',
  mystic: 'dfaf9fbd644a475595b178f0be65a39a',
  rebel: '390be3fb2b064883bb2304fc3968fd87',
  macgyver: 'c5aab6ab13d940f8ae4700d546bd6b6b',
  empath: '6bb1a05678c64213a1ed3a4dc790b81e',
  provider: '9a143feeb2994989b034cebeb78753be',
  survivor: '831c8d6048104ba0b03a74a36543cfb9',
}
```

### Kid Kelly (12 archetypes)
```typescript
HEYGEN_KID_KELLY_LOOKS = {
  scientist: '82813816115c4fbe93b3f3f211bd9931',
  explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  rebel: 'd4e960f7a3424d869877f3a951adfae7',
  architect: 'cc1dd0e9e2fd432099985c9b036ed836',
  diplomat: '48bddc41ae94473caa645ce9ab93136d',
  empath: 'deeb27f2648848b48c5c1ce59059bd54',
  macgyver: '7b6ab196f2c7430b945411df51a84c58',
  mystic: '5cff601bfb344015a65ff46c6b8cd70a',
  provider: 'deaa213342944dc2bf671abe1442e316',
  storyteller: '1024bc304a1146998bc4c360173b2c48',
  strategist: '6249632f58ce479891de00b4da5fb88d',
  survivor: 'bd579e4ca77444aca2bfea8ee9070830',
}
```

### Senior Kelly (12 archetypes)
```typescript
HEYGEN_SENIOR_KELLY_LOOKS = {
  scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
  explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  architect: '42e9197ab9d84961915b00d5cc780190',
  empath: '493dac2cf2ba4509b3cc048ff819765e',
  diplomat: 'a82183881e284e3782db75b755c3f080',
  macgyver: 'cb5b025506284d64b696e296ca2feead',
  provider: '12582467e9ff48889d7b2435642e2d65',
  storyteller: '98178c87897e4421884b535b7864ba86',
  strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  rebel: 'dc835263eaa247f5b0e06106b848df18',
  mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
  survivor: '9a143feeb2994989b034cebeb78753be',
}
```

---

## DATABASE SCHEMA (From v0)

Uses **Neon PostgreSQL** via `@neondatabase/serverless`.

### Key Tables:
1. `heygen_videos` - Main video queue (49,544 records, ALL queued)
2. `kelly_lesson_assets` - Actual video/audio source (40 videos, 9,135 audio)
3. `lesson_perspectives` - Personalized scripts per age/archetype/language
4. `lessons` - Base lesson content
5. `generated_assets` - Video/audio asset storage

---

## ARCHETYPE ALIASING

v0 is migrating from old archetypes to Jungian archetypes:

```typescript
ARCHETYPE_ALIASES = {
  // Old → New
  storyteller: 'sage',
  scientist: 'sage',
  architect: 'creator',
  strategist: 'ruler',
  diplomat: 'everyman',
  mystic: 'magician',
  macgyver: 'creator',
  empath: 'caregiver',
  provider: 'caregiver',
  survivor: 'hero',
}
```

**Note:** Current production uses OLD names (storyteller, scientist, etc.). Migration is pending.

---

## FILES EXTRACTED

Key files from v0 codebase:

| File | Purpose |
|------|---------|
| `lib/kelly-assets.ts` | Avatar mappings, scripts, asset library |
| `lib/db.ts` | Neon PostgreSQL client |
| `lib/video-orchestrator.ts` | Video generation coordination |
| `lib/heygen-client.ts` | HeyGen API wrapper |
| `lib/fal-lipsync-client.ts` | fal.ai lip-sync client |
| `app/api/video/url/route.ts` | Video URL API (priority chain) |

---

## WHAT'S WORKING IN V0 (but not production)

From backlog:
- ✅ About overlay left-positioned
- ✅ Kelly's face (not K logo)
- ✅ "Universal" (not "Personalized")
- ✅ Nicolette's real photo
- ✅ Amber ALPHA badge
- ✅ 3 PDFs in `/public/docs/`
- ✅ Database schema is correct

---

## WHAT'S BROKEN

1. **Production deployment is stale** - v0 changes not reaching Vercel
2. **Video API fails** - wrong queries in deployed code
3. **Kelly shows static image** - no video playback

---

## FIX REQUIRED

The v0 → Vercel deployment pipeline needs repair:

**Option A:** Push changes via GitHub
```bash
git add .
git commit -m "sync: v0 codebase alignment"
git push origin main
```
Then Vercel auto-deploys.

**Option B:** Force deploy from v0
1. In v0.app, click "Publish" button
2. Wait for deployment
3. Verify at thedailylesson.com

**Option C:** Manual file copy
Copy files from extracted v0 codebase to this repo, commit, push.

---

## NEXT STEPS

1. **Cursor:** Copy key v0 files to this repo
2. **v0:** Fix deployment pipeline
3. **Both:** Verify production after deploy

---

**This analysis is based on the actual v0 codebase, not assumptions.**
