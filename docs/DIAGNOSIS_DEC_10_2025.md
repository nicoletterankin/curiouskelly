# 🔴 FULL DIAGNOSIS: Day 1 Lesson Player Issues
**Date:** December 10, 2025  
**Status:** VIDEO GENERATION HALTED  
**Priority:** CRITICAL — Must fix before any more video creation

---

## EXECUTIVE SUMMARY

**Three interconnected problems are breaking the lesson experience:**

1. **STATIC IMAGES OVERRIDING VIDEOS** — The `kelly-2d-avatar.js` loads phase images AFTER the video starts playing, visually fighting with the HD video
2. **KELLY IDENTITY DRIFT** — The static images in `/public/kelly/phases/` are mockups (a different person) that don't match the LoRA-generated Kelly in videos
3. **MISSING INFOGRAPHICS** — The `visual_url` field is NULL, so no educational diagrams appear alongside Kelly

---

## PROBLEM 1: Static Images Interfering with Videos

### What I Found

When the lesson plays:
1. ✅ Golden HD video DOES start playing (confirmed in console)
2. ❌ BUT `kelly-2d-avatar.js` also loads a static image
3. ❌ The `loadPhaseVisual()` function runs and overwrites the Kelly container

```javascript
// kelly-2d-avatar.js lines 129-158
loadPhaseVisual(dayNumber, phase) {
  const phasePath = `/kelly/phases/${paddedDay}/${phaseFile}.png`;
  testImg.onload = () => {
    if (this.img) {
      this.img.src = phasePath;  // ← THIS OVERWRITES THE VIDEO!
    }
  };
}
```

### The Race Condition

```
Timeline:
0ms   → Video element created
50ms  → Video starts loading
100ms → loadPhaseVisual() called
150ms → Static image loads → OVERWRITES Kelly container
200ms → Video finally visible... but competing with image
```

### Evidence from Console

```
[Golden] 🎬 Playing HD video: https://...day_001_Fact1_The_Scientist_golden.mp4
[Kelly] Phase visual loaded: day 1, q1  ← THIS IS THE PROBLEM
```

---

## PROBLEM 2: Kelly Identity Drift

### The Two "Kellys" Don't Match

| Asset Source | Kelly Description | Match? |
|--------------|-------------------|--------|
| `/public/kelly/phases/001/*.png` | Stock photo woman, brown hair, blue sweater, director's chair | ❌ WRONG |
| `/public/kelly/poses/kelly_hint.png` | Same stock photo woman | ❌ WRONG |
| Golden Videos (LoRA-generated) | Consistent LoRA Kelly per spec | ✅ CORRECT |

### Visual Evidence

**Static Images (phases/001/hook.png, q1.png):**
- Stock photo of a woman
- Different face shape than LoRA Kelly
- Same pose, just static

**Golden Videos (in Supabase):**
- Generated with Kelly LoRA (scale 0.90)
- Consistent across all phases
- Has lip-sync and motion

### The Identity Spec (from KELLY_PRODUCTION_FACTORY.md)

| Attribute | Specification |
|-----------|---------------|
| **Eyes** | Brown, warm, expressive |
| **Hair** | Long wavy brown with subtle blonde highlights |
| **Outfit** | Light blue/teal ribbed crew-neck sweater |
| **Style** | Photorealistic 3D render (iClone/CC quality) |

The static images in `/public/kelly/` are from a DIFFERENT source than the LoRA model. They're mockups, not production assets.

---

## PROBLEM 3: Missing Infographics (Chief Academic Officer is Missing)

### Database State

```sql
SELECT phase, visual_url, hd_video_url 
FROM lesson_atoms WHERE day_number = 1 AND archetype = 'The Scientist';
```

| phase | visual_url | hd_video_url |
|-------|------------|--------------|
| Hook | **NULL** | ✅ Has URL |
| Fact1 | **NULL** | ✅ Has URL |
| Fact2 | **NULL** | ✅ Has URL |
| Fact3 | **NULL** | ✅ Has URL |
| Wisdom | **NULL** | ✅ Has URL |

### What `visual_url` Should Contain

Per the UNIFIED_LESSON_FACTORY architecture:
- **Infographics** (1920×1080) — Educational diagrams for each phase
- **Option Cards** (512×512) — Visual representations of choices

Without these, Kelly is talking about concepts without showing them.

### Asset Inventory (kelly_video_assets for Day 1)

| asset_type | status | count |
|------------|--------|-------|
| animation | generated | 5 |
| audio | generated | 75 |
| image | generated | 5 |
| video | generated | 75 |
| video_4k | validated | 15 |

Note: "image" assets exist (5) but they're not infographics — they're Kelly source images for video generation.

---

## THE FIX PLAN

### PHASE 1: Stop the Interference (Immediate)

**Fix `learn.html` to not load static images when video is playing:**

```javascript
// In setPhase(), BEFORE loading phase visual:
if (phase.videoUrl) {
  // DON'T load static image when we have HD video
  console.log('[Learn] HD video present, skipping static image');
} else {
  // Only load static image as fallback
  kellyAvatar.loadPhaseVisual(state.dayNumber, phaseFile);
}
```

### PHASE 2: Delete the Mockup Images

The `/public/kelly/phases/` images are mockups. They should be:
1. Moved to `_archive/mockup-kelly-phases/`
2. Replaced with LoRA-generated Kelly images (same prompt as videos)
3. OR just removed entirely since videos should be the only Kelly asset

### PHASE 3: Generate Infographics

For each day/phase, generate:
1. **Infographic** — Educational diagram explaining the concept
2. **Option Cards** — Visual choices for questions

Pipeline: `scripts/kelly-phase-visuals/batch-infographics-from-db.ts`

### PHASE 4: Populate `visual_url` in Database

```sql
UPDATE lesson_atoms 
SET visual_url = 'https://supabase.../infographics/day_001_hook.png'
WHERE ...
```

---

## RECOMMENDED ORDER OF OPERATIONS

1. **TODAY: Fix learn.html** — Stop static images from overriding videos
2. **TODAY: Verify Day 1 videos work** — Test with fixed player
3. **HOLD: Video generation** — Don't generate more until #1-2 confirmed
4. **NEXT: Archive mockup images** — Move `/public/kelly/phases/` to archive
5. **NEXT: Generate infographics** — Run the infographic pipeline for Day 1
6. **THEN: Scale** — Once Day 1 is perfect, generate Days 2-365

---

## ASSETS TO DELETE/ARCHIVE

### Mockup Kelly Images (Not LoRA-generated)

```
public/kelly/phases/001/hook.png    → ARCHIVE (mockup)
public/kelly/phases/001/q1.png      → ARCHIVE (mockup)
public/kelly/phases/001/q2.png      → ARCHIVE (mockup)
public/kelly/phases/001/q3.png      → ARCHIVE (mockup)
public/kelly/phases/001/wisdom.png  → ARCHIVE (mockup)
public/kelly/phases/002/*           → ARCHIVE (mockup)
```

### Static Poses (Need Audit)

```
public/kelly/poses/kelly_hint.png       → CHECK if LoRA or mockup
public/kelly/poses/kelly_welcome.png    → CHECK if LoRA or mockup
public/kelly/poses/kelly_idle.png       → CHECK if LoRA or mockup
```

---

## CONCLUSION

**The core issue:** We have TWO competing Kelly systems:
1. **Static mockup images** — Wrong person, interfering with playback
2. **LoRA-generated videos** — Correct Kelly, working correctly

**The fix:** Make videos the ONLY Kelly asset during playback. Remove/archive the mockups.

**Video generation status:** PAUSED until player is fixed.

---

## ✅ FIX APPLIED: December 10, 2025

### Changes Made to `public/learn.html`

**Lines 4688-4702:** Added `phase.videoUrl` check before loading static phase images
```javascript
if (phase.videoUrl) {
  console.log('[Learn] HD video present, skipping static phase image');
} else if (kellyAvatar && kellyAvatar.loadPhaseVisual && state.dayNumber) {
  // Only load static image as fallback
  ...
}
```

**Lines 4660-4682:** Added `!phase.videoUrl` guard to kellyVisualSystem.setPhase()

**Lines 4704-4741:** Wrapped expression mapping in `if (!phase.videoUrl)` block

### Verification

Console now shows:
```
[Learn] HD video present, skipping static phase image  ← FIX WORKING
[Golden] 🎬 Playing HD video: https://...
[Golden] 🎬 Video playing
```

No more `[Kelly] Phase visual loaded:` message conflicting with videos.

### Status
- ✅ Static images no longer override HD videos
- ⏸️ Video quality issues (separate concern) - needs investigation
- ⏸️ Infographics still missing - needs generation pipeline

