# ICLONE FULL POWER — Push Kelly to the Limit

**This is it. Test everything. Lock it forever.**

---

## PART 1: BREATHING & IDLE MOTION

### Add Breathing (Body Layer)
```
Animation → Motion → Idle Motion
```
Or:
```
Animation → Perform → Body Idle
```

**Test these presets:**
| Idle Type | Best For |
|-----------|----------|
| Breath Slow | Calm Kelly, wisdom moments |
| Breath Medium | Normal speaking |
| Breath Deep | After emotional statement |
| Subtle Shift | Natural weight shifting |

**How to apply:**
1. Select Kelly in Scene
2. Timeline → Add Motion Layer if needed
3. Drag idle motion to body track
4. Adjust intensity slider (0.3-0.7 looks natural)

---

## PART 2: FACE KEY EXPRESSIONS

### Access Face Key
```
Modify → Face → Face Key
```
Or click Kelly's face → Face Key panel opens

### Expression Categories to Test:

**EMOTIONS (try each one!):**
| Expression | When Kelly Uses It |
|------------|-------------------|
| Joy | Discovering something amazing |
| Surprise | "Wait, what?!" moments |
| Curious | Her default state |
| Thinking | Processing an idea |
| Empathy | Connecting with learner |
| Awe | Mind-blown by facts |
| Mischief | Playful teasing |
| Gentle Smile | Wisdom moments |

**To Test:**
1. Scrub to any frame
2. Click expression preset
3. Adjust intensity (0-100%)
4. See it live on Kelly

### Custom Expression Mixing:
You can COMBINE expressions:
- 50% Curious + 30% Joy = Delighted Discovery
- 40% Thinking + 20% Surprise = Realization
- 60% Empathy + 40% Gentle Smile = Warm Welcome

---

## PART 3: EXPRESSION PLUS (ARKit Blendshapes)

### Enable Expression Plus
```
Modify → Face → Expression Plus
```

This unlocks **52 ARKit blendshapes** for ultra-precise control:

**Eyes:**
- EyeBlinkLeft/Right
- EyeLookDown/Up/In/Out
- EyeSquintLeft/Right
- EyeWideLeft/Right

**Brows:**
- BrowDownLeft/Right
- BrowInnerUp
- BrowOuterUpLeft/Right

**Mouth:**
- JawOpen/Forward/Left/Right
- MouthClose
- MouthFunnel
- MouthPucker
- MouthLeft/Right
- MouthSmileLeft/Right
- MouthFrownLeft/Right
- MouthDimpleLeft/Right
- MouthStretchLeft/Right
- MouthRollLower/Upper
- MouthShrugLower/Upper
- MouthPressLeft/Right

**Cheeks/Nose:**
- CheekPuff
- CheekSquintLeft/Right
- NoseSneerLeft/Right

**TEST COMBINATIONS:**
| Combo | Effect |
|-------|--------|
| BrowInnerUp + EyeWideLeft/Right | Surprise/Interest |
| MouthSmileLeft/Right + CheekSquintLeft/Right | Genuine smile (Duchenne) |
| BrowDownLeft/Right + MouthFrownLeft/Right | Concern |
| MouthPucker + BrowInnerUp | Thinking/Pondering |
| NoseSneerLeft + MouthSmileRight | Playful skepticism |

---

## PART 4: EYE & HEAD MOVEMENT

### Look At Target
```
Animation → Face → Look At
```

**Options:**
- Look at Camera (direct eye contact)
- Look at Object (place a null target)
- Random Look (natural wandering)

### Head Movement
```
Animation → Face → Head Motion
```

**Test these:**
| Motion | When to Use |
|--------|-------------|
| Nod | Agreeing, emphasizing |
| Shake | "No" or disbelief |
| Tilt | Curiosity, question |
| Random Subtle | Natural micro-movements |

### Keyframe Head Rotation Manually:
1. Timeline → Face track
2. Scrub to frame
3. Rotate head in viewport (X/Y/Z)
4. Press K to set keyframe

---

## PART 5: HAND GESTURES

### Access Hand Poses
```
Modify → Hand → Hand Pose
```

**Kelly's Natural Gestures:**
| Gesture | Meaning |
|---------|---------|
| Open Palm | Welcoming, explaining |
| Point | Emphasizing key fact |
| Counting | Listing items |
| Thinking (chin touch) | Pondering |
| Heart Touch | Emotional moment |
| Both Hands Open | Big reveal |

### Animate Hand Gestures:
1. Go to keyframe
2. Apply hand pose
3. Move to next keyframe
4. Apply next pose
5. iClone interpolates between

---

## PART 6: MOTION LAYERS (The Pro Move)

### Layer Multiple Animations
```
Timeline → Right-click → Add Motion Layer
```

**Stack these (bottom to top):**
1. **Base Layer:** Idle breathing
2. **Body Layer:** Subtle weight shifts
3. **Face Layer:** Expressions
4. **Lip Sync Layer:** AccuLips visemes
5. **Head Layer:** Look-at and nods

Each layer can have its own intensity (0-100%)

---

## PART 7: PERFORM MODE (AI-Assisted)

### Try Auto Perform
```
Animation → Perform → Face Puppet
```

This uses your webcam/audio to drive expressions in real-time. Good for testing natural movement.

### Audio-Driven Expression
```
Animation → Perform → Voice to Expression
```

iClone analyzes audio energy and applies expressions automatically.

---

## PART 8: LIGHTING THE MOUTH FIX

For that "light in mouth" issue:

1. **Select Kelly's head mesh**
2. **Modify → Material**
3. **Find mouth interior material**
4. **Reduce:**
   - Subsurface Scattering → 0 or very low
   - Self-Illumination → 0
5. **Increase Opacity if translucent**

Or in Character Creator:
```
CC5 → Skin Settings → Mouth → SSS Intensity → Lower
```

---

## PART 9: TESTING CHECKLIST

Run through this with Kelly's intro audio:

### BEFORE EXPORT, verify these work:

- [ ] **Breathing:** Subtle chest/shoulder movement
- [ ] **Blink:** Natural random blinks (not robotic)
- [ ] **Lip Sync:** Visemes match audio
- [ ] **Expressions:** Changes during emotional beats
- [ ] **Eye Contact:** Looks at camera mostly, occasional glances
- [ ] **Head Motion:** Subtle tilts and nods with speech rhythm
- [ ] **Brows:** Move with emphasis
- [ ] **Smile:** Genuine (eyes + mouth) on warm moments
- [ ] **Hands:** At least one gesture during intro

### EXPRESSION BEATS FOR INTRO SCRIPT:

| Timestamp | Line | Expression |
|-----------|------|------------|
| 0:00 | "Octopuses have three hearts" | Curious + Slight Surprise |
| 0:05 | "I just learned that. Three." | Joy + Eyes Wide |
| 0:10 | "Two for gills, one for body" | Thinking + Explaining gesture |
| 0:15 | "that third one stops" | Surprise + Slight Concern |
| 0:20 | "stop your own heart just to get somewhere" | Awe + Deeper |
| 0:25 | "I'm Kelly" | Warm Smile + Eye Contact |
| 0:28 | "This is the kind of thing I can't let go of" | Playful + Slight head tilt |
| 0:32 | "The world gets a lot more interesting" | Building Joy |
| 0:35 | "And so do you" | Full genuine smile + slight nod |
| 0:37 | "I'd love it if you joined me" | Warm + Inviting + Open hands |

---

## PART 10: THE FOREVER TEMPLATE

Once you've tested everything and it looks perfect:

### Save as Template:
```
File → Save As → Kelly_Intro_Template.iProject
```

### Export Settings to Lock:
```
File → Export → Export Settings → Save Preset → Kelly_Unity_Perfect
```

### Document Your Settings:

**LOCKED FOREVER SETTINGS:**

| Category | Setting | Value |
|----------|---------|-------|
| **Idle** | Breath Type | Medium |
| **Idle** | Intensity | 0.5 |
| **Eyes** | Blink Interval | 3-5 sec |
| **Eyes** | Look At | Camera 80% |
| **Head** | Random Motion | Subtle, 0.3 |
| **Expressions** | Base State | Curious 30% |
| **Lip Sync** | AccuLips Mode | CC4 Extended |
| **Hands** | Default Pose | Relaxed Open |

---

## DO THIS NOW:

1. **Add breathing** (Animation → Motion → Idle Motion)
2. **Add random blinks** (Face → Expression Plus → Blink settings)
3. **Set expressions at key frames** (use table above)
4. **Add subtle head motion**
5. **Test playback** — watch the whole 29 seconds
6. **Fix anything that looks robotic**
7. **Export when perfect**

---

**This is the last time. Make it count. Lock it forever.**
