# Kelly Animation Pipeline — Audio to Unity

**Goal:** Programmatically generate all animation data needed for Kelly to speak in Unity/WebGL.

---

## What Unity Needs

A single JSON file per clip:

```json
{
  "clipName": "kelly_intro",
  "duration": 39.0,
  "fps": 25.0,
  "visemes": [...],      // Lip sync data
  "expressions": [...]   // Facial expressions + head movement
}
```

---

## The Pipeline

```
┌─────────────────┐
│  Script Text    │ ──────────────────────────────────────┐
└─────────────────┘                                        │
                                                           ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ElevenLabs     │ ──▶ │  Audio File     │ ──▶ │  Rhubarb        │
│  (TTS)          │     │  (.mp3/.wav)    │     │  (Lip Sync)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                                                ┌─────────────────┐
                                                │  Viseme JSON    │
                                                └─────────────────┘
                                                           │
                               ┌───────────────────────────┤
                               ▼                           ▼
                    ┌─────────────────┐         ┌─────────────────┐
                    │  AI Expression  │         │  Merge Script   │
                    │  Generator      │         │                 │
                    └─────────────────┘         └─────────────────┘
                               │                           │
                               └───────────┬───────────────┘
                                           ▼
                                ┌─────────────────┐
                                │  Unity JSON     │
                                │  (*_unity.json) │
                                └─────────────────┘
                                           │
                                           ▼
                                ┌─────────────────┐
                                │  Unity WebGL    │
                                │  Kelly speaks!  │
                                └─────────────────┘
```

---

## Step 1: Generate Audio (ElevenLabs)

```bash
# Using ElevenLabs API or web interface
# Output: kelly_intro.mp3
```

**Settings for Kelly:**
- Voice: kelly2
- Stability: 50%
- Similarity: 94%
- Style: 0 (none)

---

## Step 2: Generate Visemes (Rhubarb)

**Install Rhubarb Lip Sync:**
- Download from: https://github.com/DanielSWolf/rhubarb-lip-sync
- Available for Windows/Mac/Linux

```bash
# Generate visemes from audio
rhubarb kelly_intro.mp3 -o kelly_intro_visemes.json --machineReadable
```

**Output format:**
```json
{
  "mouthCues": [
    { "start": 0.00, "end": 0.32, "value": "X" },
    { "start": 0.32, "end": 0.40, "value": "B" },
    { "start": 0.40, "end": 0.68, "value": "F" }
  ]
}
```

**Rhubarb → Unity viseme mapping:**
| Rhubarb | Unity Viseme | Description |
|---------|--------------|-------------|
| X | viseme_sil | Silence |
| A | viseme_aa | Open vowel |
| B | viseme_PP | P, B, M |
| C | viseme_CH | CH, J, SH |
| D | viseme_DD | D, T, TH |
| E | viseme_E | E sounds |
| F | viseme_FF | F, V |
| G | viseme_I | I sounds |
| H | viseme_O | O sounds |

---

## Step 3: Generate Expression Data

**Option A: Script-based (current)**
```bash
npx ts-node scripts/generate-kelly-animation.ts kelly_intro 39 projects/Kelly/CC5/final-models/kelly_intro_script.txt
```

**Option B: AI-enhanced (future)**
Send script to GPT-4 with prompt:
```
Analyze this script and generate expression keyframes.
For each emotional beat, output:
- timestamp
- emotion (curious/wonder/reflective/warm/inviting)
- intensity (0-1)
- head_direction (slight turn/tilt)
```

---

## Step 4: Merge & Output

The `generate-kelly-animation.ts` script combines:
1. Viseme data (from Rhubarb)
2. Expression data (from script analysis)
3. Duration/FPS metadata

**Output:** `kelly_intro_unity.json`

---

## Step 5: Deploy to Unity

1. Copy `kelly_intro_unity.json` to `StreamingAssets/kelly-motion/`
2. Copy `kelly_intro.mp3` (or .wav) to same folder
3. In Unity, set paths on `KellyAnimationPlayer` component

---

## From iClone (Alternative Path)

If you want to export directly from iClone with hand-crafted animation:

### Export FBX with Animation
```
File → Export → FBX
- Target: Unity 3D
- ✅ Mesh and Motion
- ✅ Embed Textures
```

### Extract Blendshape Curves
Use the FBX SDK or Blender Python to extract:
- Blendshape weights per frame
- Head bone rotation per frame

### Convert to Unity JSON
Map iClone blendshapes to our format:
| iClone | Unity JSON |
|--------|------------|
| Mouth_Open | mouthOpen |
| Mouth_Smile_L/R | smile |
| Eye_Blink_L | leftEyeOpen (inverted) |
| Brow_Raise_Inner_L | leftBrowRaise |

---

## Automation Script (Full Pipeline)

```bash
#!/bin/bash
# generate-kelly-clip.sh

CLIP_NAME=$1
SCRIPT_FILE=$2
AUDIO_FILE=$3

# Step 1: Generate visemes with Rhubarb
rhubarb "$AUDIO_FILE" -o "${CLIP_NAME}_visemes.json" --machineReadable

# Step 2: Generate expressions from script
npx ts-node scripts/generate-kelly-animation.ts "$CLIP_NAME" \
  --script "$SCRIPT_FILE" \
  --visemes "${CLIP_NAME}_visemes.json" \
  --output "${CLIP_NAME}_unity.json"

# Step 3: Copy to StreamingAssets
cp "${CLIP_NAME}_unity.json" unity-project/Assets/StreamingAssets/kelly-motion/
cp "$AUDIO_FILE" unity-project/Assets/StreamingAssets/kelly-motion/

echo "Done! Kelly clip ready: $CLIP_NAME"
```

---

## What We Need to Build

| Component | Status | Notes |
|-----------|--------|-------|
| ElevenLabs TTS | ✅ Working | kelly2 voice |
| Rhubarb Lip Sync | 🔲 Install | Download from GitHub |
| Expression Generator | ✅ Created | `generate-kelly-animation.ts` |
| Viseme Converter | 🔲 TODO | Rhubarb → Unity format |
| Full Pipeline Script | 🔲 TODO | Bash/PowerShell wrapper |

---

## Quick Start (Manual)

1. Generate audio in ElevenLabs ✅
2. Download Rhubarb, run on audio
3. Run `generate-kelly-animation.ts`
4. Copy JSON + audio to Unity StreamingAssets
5. Play!

---

*This pipeline enables 365 lessons to be generated programmatically.*
