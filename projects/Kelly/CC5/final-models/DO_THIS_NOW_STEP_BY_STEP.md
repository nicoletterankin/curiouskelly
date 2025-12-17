# IMPLEMENT KELLY EXPRESSIONS — STEP BY STEP

**Follow this exactly. One step at a time.**

---

## STEP 1: ADD BREATHING (2 minutes)

1. **Select Kelly** in Scene panel (click on `CC3_Base_Plus`)
2. Go to menu: **Animation → Motion → Idle Motion**
3. In the popup, find: **Breath** category
4. Select: **Breath Medium** or **Breathing_Subtle**
5. Click **Apply** or drag to timeline
6. Adjust intensity slider to **0.5** (50%)

**✅ Check:** Play timeline. You should see subtle chest/shoulder movement.

---

## STEP 2: ENABLE AUTO-BLINK (1 minute)

1. With Kelly selected, go to: **Modify → Face**
2. Look for **Expression Plus** or **Facial Animation** panel
3. Find **Blink** settings
4. Enable: **Auto Blink**
5. Set interval: **3-5 seconds**
6. Set randomness: **ON** or **30%**

**✅ Check:** Play timeline. Watch her eyes blink naturally.

---

## STEP 3: SET UP LOOK-AT (1 minute)

1. Go to: **Animation → Face → Look At**
2. Target: **Camera** or create a null at camera position
3. Set weight: **80%** (allows some natural eye wandering)

**✅ Check:** Her eyes should follow camera/viewer.

---

## STEP 4: ADD HEAD MOTION (1 minute)

1. Go to: **Animation → Face → Head Motion**
2. Select: **Subtle Random** or **Natural Talk**
3. Set intensity: **0.3** (30%)

**✅ Check:** Play. Head has micro-movements, not frozen.

---

## STEP 5: KEYFRAME EXPRESSIONS (10-15 minutes)

This is the main work. Use the table below.

### HOW TO KEYFRAME:

1. **Scrub to frame** (use Frame box at bottom, type the number)
2. **Open Face Key:** Modify → Face → Face Key
3. **Apply expression preset** or use sliders
4. **Press K** to set keyframe (or right-click → Add Key)

### YOUR KEYFRAMES:

| Frame | Go Here | Apply This | Notes |
|-------|---------|------------|-------|
| **0** | Start | Curious 30% | Base state |
| **30** | | Surprise 40% + Joy 30% | "three hearts" reveal |
| **90** | | Joy 50% | "I just learned" |
| **150** | | Joy 40% + slight brow up | "Three." emphasis |
| **270** | | Thinking 40% | Teaching mode |
| **390** | | Curious 50% + head tilt | "when they swim?" |
| **420** | | Surprise 40% + subtle concern | "stops" |
| **450** | | Awe 50% | "Just shuts down" |
| **570** | | Thinking 40% | "I keep thinking" |
| **690** | | Empathy 40% + Awe 30% | "stop your own heart" |
| **750** | | **Warm Smile 60%** | "I'm Kelly" — KEY MOMENT |
| **870** | | Playful 40% + Joy 30% | "can't let go" |
| **1050** | | Confident/Clear | "Five minutes" |
| **1200** | | Curious + head tilt | "A year of that?" |
| **1380** | | **FULL SMILE 80%** | "And so do you" — MONEY SHOT |
| **1560** | | Warm 50% | "Today's lesson" |
| **1620** | | Playful 40% + Warm 50% | "jump in?" |
| **1799** | | Gentle smile holds | End |

### THE TWO BIG ONES:

**Frame 750 ("I'm Kelly"):**
- This is her introduction
- Full eye contact
- Warm, genuine smile
- Don't overdo it — 60% is enough

**Frame 1380 ("And so do you"):**
- This is THE payoff
- Full Duchenne smile (eyes + mouth)
- Add eye squint (CheekSquint in Expression Plus)
- Small head nod
- This should feel like connection

---

## STEP 6: ADD MANUAL HEAD NODS (2 minutes)

At these frames, add a small head nod (X rotation down then up):

1. **Frame 150** — after "Three"
2. **Frame 1110** — on "Every day"
3. **Frame 1380** — on "And so do you"
4. **Frame 1440** — on "together"

**How:**
1. Scrub to frame
2. Select head bone or use head controls
3. Rotate slightly down (5-10 degrees)
4. Press K
5. Go forward 10 frames
6. Rotate back to neutral
7. Press K

---

## STEP 7: PREVIEW FULL CLIP (3 minutes)

1. Press **Home** to go to start
2. Press **Space** to play
3. Watch the entire 30 seconds
4. Note any "dead spots" (frozen face, no movement)

### FIX DEAD SPOTS:

If you see frozen moments:
- Add a subtle expression shift
- Add a blink
- Add micro head movement

---

## STEP 8: FINE-TUNE (5 minutes)

Watch again and adjust:

| Issue | Fix |
|-------|-----|
| Expression too strong | Lower intensity (slider) |
| Expression too weak | Increase intensity |
| Transition jarring | Select keyframes, apply smoothing |
| Head motion too robotic | Lower intensity, add randomness |
| Smile looks fake | Add CheekSquint (eye squint) |

---

## STEP 9: SAVE TEMPLATE (1 minute)

**Before exporting:**
```
File → Save As → Kelly_Intro_Template.iProject
```

This is your forever template. Never delete it.

---

## STEP 10: EXPORT (2 minutes)

Follow `EXPORT_NOW.md` exactly:
1. File → Export → FBX
2. Target: Unity 3D
3. Range: 0 to 1799
4. All checkboxes per the doc
5. Save as: `kelly_intro_full.fbx`
6. Location: `projects\Kelly\CC5\final-models\`

---

## TOTAL TIME: ~25-30 minutes

---

## IF YOU GET STUCK

| Problem | Solution |
|---------|----------|
| Can't find Face Key | Modify → Face → Face Key |
| Can't find Expression Plus | Modify → Face → Expression Plus |
| Keyframe not setting | Make sure Kelly is selected, press K |
| Expression not showing | Check you're on the right frame |
| Can't see head controls | View → Show → Controllers |

---

## READY?

**Start at Step 1. Do them in order. Tell me when you're done or if you hit a wall.**
