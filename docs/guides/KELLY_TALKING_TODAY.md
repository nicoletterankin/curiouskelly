# 🎬 GET KELLY TALKING TODAY

**Goal:** Create a video of Kelly speaking with perfect lipsync  
**Time:** ~45 minutes active work + 20-60 min render  
**Date:** October 13, 2025

---

## ✅ What We Have Ready

- ✅ Kelly headshot: `C:\iLearnStudio\projects\Kelly\Ref\headshot2-kelly-base169 101225.png`
- ✅ CC5 projects: `Kelly_8K_Production.ccProject` (working, "kinda")
- ✅ Audio file: `kelly25_audio.wav`
- ✅ ElevenLabs Kelly voice: `wAdymQH5YucAkXwmrdL0`
- ✅ API key ready and working

---

## 🚀 5-Step Process

### STEP 1: Generate Kelly's Voice (5 min)

**Run this command:**
```powershell
cd C:\Users\user\UI-TARS-desktop\synthetic_tts
python generate_kelly_lipsync.py
```

**What to do:**
- Select option **4** (short_greeting) for quick test, OR
- Select option **2** (leaves_intro) for full lesson
- Audio saves automatically to Kelly's Audio folder

**Output:** `kelly_[name]_lipsync.wav` ready for iClone

---

### STEP 2: Open Kelly in CC5 (2 min)

1. Launch **Character Creator 5**
2. **File → Open Project**
3. Open: `C:\iLearnStudio\projects\Kelly\CC5\Kelly_8K_Production.ccProject`
4. Review character looks good

---

### STEP 3: Send to iClone (5 min)

**In CC5:**
1. **File → Send Character to iClone**
2. Check:
   - ✅ Export with Facial Profile
   - ✅ Export with Expression Wrinkle
   - Quality: **Ultra High**
3. Click **Send to iClone**
4. Wait for iClone 8 to open (2-3 min)

---

### STEP 4: Apply Lipsync (10 min)

**In iClone 8:**

1. **Import audio:**
   - Right-click timeline audio track
   - Import Audio File
   - Select your WAV file

2. **Run AccuLips:**
   - Select Kelly character
   - **Animation → Facial Animation → AccuLips**
   - Audio: select your track
   - Language: English
   - Quality: High
   - Click **Apply** (wait 1-3 min)

3. **Preview:**
   - Press SPACEBAR to play
   - Check mouth syncs with words

---

### STEP 5: Render Video (20-60 min)

**In iClone 8:**
1. **File → Export → Video**
2. Settings:
   - Format: MP4 (H.264)
   - Resolution: 1920×1080 (for quick test)
   - FPS: 30
   - Quality: High
   - Save to: `C:\iLearnStudio\renders\Kelly\kelly_talking_test_v1.mp4`
3. Click **Export**
4. Wait for render to complete

---

## ✅ Success = Kelly Talking!

**Check the video:**
- Mouth syncs with words?
- Audio is clear?
- Looks natural?

**If yes: 🎉 DONE! Kelly is talking!**

**If no:** See troubleshooting in `UI-Tars_CC5_Runbook.md` Section 19

---

## 📚 Reference Files

- **Full workflow:** `UI-Tars_CC5_Runbook.md` (Section 18)
- **Simple steps:** `C:\iLearnStudio\projects\Kelly\Ref\Kelly_Simple_Steps.txt`
- **Audio generator:** `synthetic_tts\generate_kelly_lipsync.py`
- **Lesson player:** `lesson-player\index.html`

---

## 🆘 Quick Troubleshooting

**Audio generator fails?**
- Check internet (needs ElevenLabs API)
- Try existing audio: `kelly25_audio.wav`

**CC5 project won't open?**
- Try `Kelly_G3Plus_Base.ccProject` instead
- Or follow Section 6 in runbook to rebuild

**AccuLips not working?**
- Verify WAV format (22,050 Hz, Mono)
- Re-import audio
- Select character before running AccuLips

**Render too slow?**
- Reduce to 10 seconds for test
- Lower to 720p resolution
- Set quality to Medium

---

## 🎯 TODAY'S CHECKLIST

- [ ] Run `generate_kelly_lipsync.py` → get WAV file
- [ ] Open `Kelly_8K_Production.ccProject` in CC5
- [ ] Send to iClone 8
- [ ] Import audio in iClone
- [ ] Apply AccuLips
- [ ] Preview lipsync
- [ ] Render video
- [ ] Watch result

**Time estimate:** 45 min active + render time

**🎬 Let's get Kelly talking!**
















