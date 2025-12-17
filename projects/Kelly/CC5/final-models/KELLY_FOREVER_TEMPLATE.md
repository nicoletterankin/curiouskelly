# KELLY FOREVER TEMPLATE — LOCKED SETTINGS

**Version:** 1.0
**Date Locked:** December 16, 2025
**Status:** CANONICAL — DO NOT MODIFY WITHOUT APPROVAL

---

## PURPOSE

This document defines the exact iClone settings for all Kelly animations, forever. Any new Kelly content must use these settings for consistency.

---

## CHARACTER SETTINGS

### Model
| Setting | Value |
|---------|-------|
| Character | Kelly CC5 Adult |
| Rig | CC4+ Extended (52 blendshapes) |
| Quality | Production |

### Materials
| Setting | Value |
|---------|-------|
| Skin SSS | Standard (no mouth glow) |
| Eye Shader | PBR with wetness |
| Hair | Transparency enabled |

---

## ANIMATION DEFAULTS

### Breathing / Idle
| Setting | Value |
|---------|-------|
| Type | Breath Medium |
| Intensity | 0.5 |
| Layer | Base (always on) |

### Blinking
| Setting | Value |
|---------|-------|
| Mode | Random Automatic |
| Interval | 3-5 seconds |
| Duration | 150ms |
| Variation | ±30% |

### Eye Movement
| Setting | Value |
|---------|-------|
| Primary Target | Camera (80% of time) |
| Secondary | Random subtle glances (20%) |
| Look Speed | Natural (0.3) |

### Head Motion
| Setting | Value |
|---------|-------|
| Random Motion | Enabled |
| Intensity | 0.3 (subtle) |
| Nod on Emphasis | Yes |
| Tilt on Questions | Yes |

### Hands
| Setting | Value |
|---------|-------|
| Default Pose | Relaxed Natural |
| Gesture Frequency | 1-2 per 10 seconds |
| Style | Open, welcoming |

---

## LIP SYNC SETTINGS

### AccuLips
| Setting | Value |
|---------|-------|
| Mode | CC4 Extended Visemes |
| Strength | 100% |
| Smoothing | Medium |

### Viseme Track
| Setting | Value |
|---------|-------|
| Source | Audio file (ElevenLabs Kelly voice) |
| Process | Auto-generate from audio |

---

## EXPRESSION PRESETS

### Kelly's Core Expressions
These are the expressions Kelly uses, mapped to emotional states:

| State | Primary | Secondary | Intensity |
|-------|---------|-----------|-----------|
| **Neutral** | Curious | — | 30% |
| **Discovering** | Surprise | Joy | 50% / 30% |
| **Explaining** | Thinking | Curious | 40% / 20% |
| **Connecting** | Empathy | Gentle Smile | 50% / 40% |
| **Amazed** | Awe | Joy | 60% / 30% |
| **Playful** | Mischief | Smile | 40% / 40% |
| **Inviting** | Warm Smile | Open | 60% / — |
| **Wisdom** | Gentle | Knowing | 50% / 30% |

### Expression Transitions
| From → To | Transition Time |
|-----------|-----------------|
| Any → Any | 0.5 seconds |
| Neutral → Emotion | 0.3 seconds |
| Emotion → Neutral | 0.8 seconds (longer fade) |

---

## TIMELINE STRUCTURE

### Standard Layer Stack (Bottom to Top)
```
Layer 1: Idle Breathing (Base)
Layer 2: Body Motion (if any)
Layer 3: Face Expressions
Layer 4: Lip Sync (AccuLips)
Layer 5: Head Motion
Layer 6: Eye Motion
```

### Keyframe Density
| Track | Keyframes Per Second |
|-------|---------------------|
| Breathing | Continuous (auto) |
| Expressions | 1-2 (on beats) |
| Head | 0.5-1 |
| Eyes | 0.3-0.5 |
| Lip Sync | Auto (per viseme) |

---

## EXPORT SETTINGS (Unity FBX)

### FBX Options
| Setting | Value |
|---------|-------|
| Target | Unity 3D |
| Format | Binary |
| Frame Range | Custom (match audio) |

### Include
| Setting | Value |
|---------|-------|
| Mesh | ✅ Yes |
| Motion | ✅ Yes |
| Embed Texture | ✅ Yes |
| Delete Hidden Faces | ❌ No |

### Mesh
| Setting | Value |
|---------|-------|
| Merge Material Subdivisions | ✅ Yes |
| Merge Mesh as One | ❌ No |

### Bone
| Setting | Value |
|---------|-------|
| Human IK | ✅ Yes |
| Remove Unimportant Bones | ❌ No |

### Motion
| Setting | Value |
|---------|-------|
| Blend Shape | ✅ Yes |
| Body Motion | ✅ Yes |
| Facial (Head & Eye) | ✅ Yes |
| Smooth Rotation | ❌ No |

---

## QUALITY CHECKLIST

Before exporting ANY Kelly animation, verify:

### Natural Human Behavior
- [ ] Breathing visible
- [ ] Blinks are random, not rhythmic
- [ ] Eyes track camera naturally
- [ ] Head has micro-movements
- [ ] No "dead" moments (frozen face)

### Expression Authenticity
- [ ] Smiles include eye squint (Duchenne)
- [ ] Brows move with emotion
- [ ] Expressions transition smoothly
- [ ] Never holds expression too long (>3 sec)

### Lip Sync Quality
- [ ] Mouth shapes match phonemes
- [ ] Jaw opens appropriately
- [ ] Lips close on M/B/P sounds
- [ ] No over-exaggeration

### Technical
- [ ] No mesh clipping
- [ ] No texture popping
- [ ] Smooth motion (no jitter)
- [ ] Correct frame range

---

## AUDIO SPECIFICATIONS

### ElevenLabs Voice
| Setting | Value |
|---------|-------|
| Voice | Kelly2 |
| Stability | 0.50 |
| Similarity | 0.75 |
| Style | 0.94 |
| Speaker Boost | Enabled |

### Audio Format
| Setting | Value |
|---------|-------|
| Format | MP3 or WAV |
| Sample Rate | 44.1kHz |
| Channels | Mono |

---

## FILE NAMING

### iClone Projects
```
Kelly_[ContentType]_[Identifier].iProject
Example: Kelly_Intro_v1.iProject
Example: Kelly_Day001_Adult.iProject
```

### Exported FBX
```
kelly_[content]_[variant].fbx
Example: kelly_intro_full.fbx
Example: kelly_day001_adult.fbx
```

### Audio Files
```
kelly_[content]_audio.mp3
Example: kelly_intro_audio.mp3
```

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-16 | Initial template locked |

---

**THIS IS CANON. ALL KELLY ANIMATIONS MUST FOLLOW THIS TEMPLATE.**
