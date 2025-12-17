# KELLY iCLONE QUICK REFERENCE CARD

**Print this. Keep it next to your monitor.**

---

## KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| **Space** | Play/Pause |
| **K** | Set Keyframe |
| **,** / **.** | Previous/Next Frame |
| **Home** | Go to Start |
| **End** | Go to End |
| **Ctrl+Z** | Undo |
| **F** | Focus on Selected |

---

## EXPRESSION QUICK-ACCESS

### Face Key (Modify → Face → Face Key)

| Emotion | Slider Mix | When |
|---------|------------|------|
| **Curious** | 30% base | Default state |
| **Joy** | 50-70% | Discoveries |
| **Surprise** | 40-60% | "Wait, what?" |
| **Thinking** | 40% | Explaining |
| **Empathy** | 50% | Connecting |
| **Warm Smile** | 60% | Inviting |

---

## EXPRESSION PLUS COMBOS

### Quick Authentic Expressions:

**Genuine Smile:**
- MouthSmileLeft: 70%
- MouthSmileRight: 70%
- CheekSquintLeft: 50%
- CheekSquintRight: 50%

**Curiosity:**
- BrowInnerUp: 40%
- EyeWideLeft: 20%
- EyeWideRight: 20%

**Thinking:**
- BrowDownLeft: 20%
- BrowDownRight: 20%
- MouthPucker: 15%
- JawForward: 10%

**Surprise:**
- BrowInnerUp: 60%
- EyeWideLeft: 50%
- EyeWideRight: 50%
- JawOpen: 20%

---

## ANIMATION WORKFLOW

```
1. LOAD AUDIO → Timeline → Audio Track → Import

2. ADD BREATHING → Animation → Motion → Idle Motion
   → Breath Medium → Intensity 0.5

3. GENERATE LIP SYNC → Select audio → AccuLips → Process

4. ADD EXPRESSIONS → Scrub to emotional beat → Face Key
   → Apply expression → Set keyframe (K)

5. ADD HEAD MOTION → Animation → Face → Head Motion
   → Subtle Random → Intensity 0.3

6. ADD BLINKS → Expression Plus → Blink Auto
   → Interval 3-5 sec

7. PREVIEW → Play full clip → Fix any dead spots

8. EXPORT → File → Export → FBX → Unity 3D preset
```

---

## EMOTIONAL BEATS — INTRO SCRIPT

| Frame | Time | Line | Expression |
|-------|------|------|------------|
| 0 | 0:00 | "Octopuses have three hearts" | Curious + Surprise |
| 150 | 0:05 | "I just learned that" | Joy |
| 300 | 0:10 | "Two for gills..." | Explaining |
| 450 | 0:15 | "that third one stops" | Concern + Wonder |
| 600 | 0:20 | "stop your own heart" | Awe |
| 750 | 0:25 | "I'm Kelly" | Warm Smile |
| 840 | 0:28 | "can't let go" | Playful |
| 960 | 0:32 | "more interesting" | Building Joy |
| 1050 | 0:35 | "And so do you" | Full smile + nod |
| 1110 | 0:37 | "joined me" | Inviting |

*(Frames at 30fps)*

---

## COMMON FIXES

| Problem | Solution |
|---------|----------|
| Face looks frozen | Add random blinks + micro head motion |
| Smile looks fake | Add CheekSquint (Duchenne smile) |
| Eyes look dead | Enable Look At Camera |
| Motion too smooth | Reduce smoothing, add variation |
| Robotic head | Lower Random Motion intensity |
| Mouth glowing | Reduce SSS on mouth material |

---

## EXPORT CHECKLIST

Before clicking Export:

- [ ] Timeline range set correctly (0-1799)
- [ ] Target: Unity 3D
- [ ] Blend Shape: ON
- [ ] Motion: ON
- [ ] Embed Texture: ON
- [ ] File name: kelly_intro_full.fbx

---

## FILE LOCATIONS

```
Project:  projects\Kelly\CC5\final-models\
Audio:    kelly_intro_audio.mp3
Export:   kelly_intro_full.fbx
Template: Kelly_Intro_Template.iProject
```

---

**GO MAKE KELLY COME ALIVE.**
