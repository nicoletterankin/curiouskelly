# Working Prototype - Build Plan

## 🎯 Goal
Build a working prototype that demonstrates Kelly teaching ONE complete lesson with:
- Age slider adaptation (ages 2-102)
- PhaseDNA teaching moments
- Interactive choices
- All 6 age variants working

## 📊 Current State Assessment

### ✅ What Works
- Lesson player HTML/JS framework
- Age slider UI (2-102)
- Sample lesson DNA (`leaves-change-color.json`)
- PhaseDNA schema defined
- Unity rendering architecture ready
- Flutter + Unity integration code exists

### ⚠️ What's Missing
- Kelly avatar assets (needs CC5/iClone work)
- Real video files for each age variant
- Audio with lipsync
- Complete lesson DNA with all details
- Flutter Unity export/build

### 🚀 Strategy
**Build the prototype in 3 phases**, each resulting in a working demo:

## Phase 1: Web Prototype (Minimum Viable - 2 hours)
**Goal:** Get age-adaptive lesson working WITHOUT videos

### Tasks:
1. ✅ Update lesson player to handle missing videos gracefully
2. ✅ Add "Coming Soon" placeholders where video would be
3. ✅ Make all text content age-adaptive
4. ✅ Test interactive choices working
5. ✅ Demonstrate teaching moments timestamps

**Deliverable:** Working web demo at `lesson-player/index.html`
- Age slider changes content appropriately
- All 6 age variants display correctly
- Interactions work
- Shows lesson structure

---

## Phase 2: Audio-Only Prototype (4-6 hours)
**Goal:** Add real Kelly voice with ElevenLabs, still no video

### Tasks:
1. Generate 6 age-appropriate audio tracks for "Leaves Change Color"
2. Integrate audio player in lesson player
3. Test audio plays with correct age variant
4. Add audio loading states

### Implementation:
```python
# Quick script to generate audio for each age
# Use existing ElevenLabs integration
# For each age bucket (2-5, 6-12, etc.):
#   - Use appropriate script text from lesson DNA
#   - Generate audio with ElevenLabs
#   - Save as kelly_leaves_{age-bucket}.mp3
```

**Deliverable:** Audio-enabled lesson with Kelly speaking

---

## Phase 3: Full Kelly Avatar Prototype (Depends on CC5 work)
**Goal:** Complete with video rendering

### Blocker Removed:
- Kelly avatar created in CC5
- Video files rendered for each age variant
- Lipsync applied

### Tasks:
1. Replace "Coming Soon" with real videos
2. Ensure video plays for correct age
3. Test full end-to-end lesson

**Deliverable:** Complete prototype with video avatar

---

## 🏃‍♂️ START WITH PHASE 1 RIGHT NOW

### Step 1: Update Lesson Player (30 min)
1. Modify `script.js` to handle missing videos
2. Add placeholder UI for video area
3. Test age slider switching content
4. Verify choices work

### Step 2: Enhance Lesson DNA (1 hour)
1. Complete all teaching moments timestamps
2. Add more interaction points
3. Fill in all age variant details
4. Validate against schema

### Step 3: Test & Polish (30 min)
1. Test each age bucket (2-5 through 61-102)
2. Verify teaching moments trigger
3. Check interaction flow
4. Polish UI/UX

**Result:** Working demo you can show TODAY

---

## 📁 Files to Edit

### Priority 1 (Critical):
- `lesson-player/script.js` - Add video placeholder handling
- `lessons/leaves-change-color.json` - Complete all details
- `lesson-player/index.html` - May need small adjustments

### Priority 2 (Enhancement):
- Add CSS improvements for missing video state
- Add loading indicators
- Add success/error messages

### Priority 3 (Future):
- Generate audio files
- Create video assets
- Build Unity export

---

## 🎬 Immediate Next Action

**You:** Open `lesson-player/index.html` in browser
**Me:** I'll update the JavaScript to handle missing videos and make it work

This gets us to a working demo in ~1 hour!

