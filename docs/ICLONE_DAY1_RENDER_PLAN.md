# 🎬 iCLONE DAY 1 RENDER PLAN
**Created:** December 10, 2025  
**Goal:** Replace AI-generated robot videos with professional iClone renders  
**Timeline:** TODAY

---

## ✅ DONE (by Claude)

1. ✅ Cleared bad AI videos from database — player now falls back to audio
2. ✅ Rewrote Day 1 scripts to be conversational (no more "shall we")
3. ✅ Located canonical Kelly reference: `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\close up of face.jpeg`

---

## 🎯 YOUR ACTION PLAN (Do This NOW)

### STEP 1: Generate Audio Files (5 minutes)

The new scripts need new audio. Run this command:

```powershell
cd C:\Users\user\UI-TARS-desktop
npx tsx scripts/kelly-video-factory/generate-audio-only.ts --day 1 --archetype "The Scientist"
```

If that script doesn't exist, I'll create it. Or you can generate manually:

**New Day 1 Scripts to speak:**

| Phase | Script |
|-------|--------|
| Hook | "Hey! Ever notice how New Year's Day or even just Monday morning makes you feel like anything is possible? That's not just a feeling—it's science. Your brain literally resets at these moments. Today, let's explore how you can use this to your advantage." |
| Fact1 | "Here's what's wild: researchers found that people who start a goal on a 'fresh start' day—like New Year's or a birthday—are way more likely to stick with it. It's called the Fresh Start Effect. Your brain treats these moments as a clean slate, like yesterday's failures don't count anymore." |
| Fact2 | "But here's the thing—you don't have to wait for January 1st. You can create your own fresh starts. The first day of the month, a Monday, even just tomorrow morning. The key is making it feel significant to YOU." |
| Fact3 | "And get this—when you start fresh, your brain's reward system lights up. You literally feel more optimistic. It's like your mind gives you permission to be a different person than you were yesterday." |
| Wisdom | "So here's what I want you to take away: you have the power to create a fresh start whenever you need one. Tomorrow could be day one. Next week could be your reset. The calendar doesn't decide—you do." |

**Option A: Use ElevenLabs directly**
1. Go to https://elevenlabs.io
2. Select Kelly voice (ID: wAdymQH5YucAkXwmrdL0)
3. Generate each script as MP3
4. Save to: `C:\iLearnStudio\projects\Kelly\Audio\Day001\`

---

### STEP 2: Open Character Creator 5 (10 minutes)

1. Launch **Character Creator 5**
2. Open: `C:\iLearnStudio\projects\Kelly\CC5\Kelly_HD_Base.ccProject`
   - If doesn't exist, follow Kelly_HD_Pipeline.md to create from scratch using Headshot 2

3. Verify Kelly looks correct against reference:
   - Brown eyes ✓
   - Long wavy brown hair with blonde highlights ✓
   - Fair/medium skin with warm undertones ✓
   - Blue/teal sweater ✓

---

### STEP 3: Send to iClone (5 minutes)

1. In CC5: `File → Send Character to iClone`
2. In iClone: Set up Director's Chair scene
   - Camera: 85mm focal length
   - Lighting: Key (soft white 45° right), Fill (−45° half), Rim (back warm)
   - Background: Clean white or soft gradient

---

### STEP 4: Apply AccuLips for Each Phase (30 minutes)

For each audio file:

1. Drag audio into iClone timeline
2. Select Audio Track → Right-Click → `AccuLips → Generate Text`
3. Verify transcription
4. Click **Apply to Viseme Track**
5. Preview mouth motion

Do this for:
- [ ] hook.wav
- [ ] fact1.wav
- [ ] fact2.wav
- [ ] fact3.wav
- [ ] wisdom.wav

---

### STEP 5: Render Videos (20 minutes)

For each phase:

1. Menu → `Render → Render Video`
2. Settings:
   - Format: H.264 MP4
   - Resolution: 1920×1080 (HD) or 3840×2160 (4K)
   - Bitrate: 20 Mbps
   - Frame Rate: 30 fps

3. Output paths:
   ```
   C:\iLearnStudio\projects\Kelly\Renders\Day001\
   ├── day_001_hook.mp4
   ├── day_001_fact1.mp4
   ├── day_001_fact2.mp4
   ├── day_001_fact3.mp4
   └── day_001_wisdom.mp4
   ```

4. Click **Render** for each

---

### STEP 6: Upload to Supabase (5 minutes)

Once renders are complete, tell me and I'll:
1. Upload videos to Supabase kelly-videos bucket
2. Update lesson_atoms.hd_video_url with new URLs
3. Test the player

---

## 📁 KEY FILE LOCATIONS

| What | Where |
|------|-------|
| Kelly Reference | `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\` |
| CC5 Projects | `C:\iLearnStudio\projects\Kelly\CC5\` |
| iClone Projects | `C:\iLearnStudio\projects\Kelly\iClone\` |
| Audio Files | `C:\iLearnStudio\projects\Kelly\Audio\Day001\` |
| Render Output | `C:\iLearnStudio\projects\Kelly\Renders\Day001\` |
| Pipeline Docs | `C:\iLearnStudio\projects\Kelly\Ref\Kelly_HD_Pipeline.md` |

---

## ⏱️ ESTIMATED TIME

| Step | Time |
|------|------|
| Generate Audio | 5 min |
| CC5 Setup | 10 min |
| Send to iClone | 5 min |
| AccuLips (5 phases) | 30 min |
| Render (5 videos) | 20 min |
| Upload + Test | 5 min |
| **TOTAL** | **~75 minutes** |

---

## 🚨 IF YOU DON'T HAVE THE CC5 PROJECT

If `Kelly_HD_Base.ccProject` doesn't exist, follow these steps from scratch:

1. Open CC5
2. Plugins → Headshot 2 → Photo to 3D (Pro)
3. Load: `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\close up of face.jpeg`
4. Settings: Ultra High (8K), Maximum detail
5. Generate (~10 min)
6. Apply to Character
7. Save as `Kelly_HD_Base.ccProject`

---

**START NOW. Tell me when you have renders ready to upload.**

