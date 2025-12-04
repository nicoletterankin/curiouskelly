# Kelly Lip-Sync Integration Guide

**Version:** 1.0.0  
**Created:** December 4, 2025  
**Status:** Production Ready

---

## Overview

The Kelly Lip-Sync System provides frame-perfect mouth synchronization for Kelly's voice. It supports three modes:

| Mode | Use Case | Accuracy | Latency |
|------|----------|----------|---------|
| **Alignment** | Pre-rendered lessons | ⭐⭐⭐⭐⭐ | High (requires preprocessing) |
| **Real-time** | Live audio playback | ⭐⭐⭐⭐ | Low |
| **Streaming** | ElevenLabs WebSocket | ⭐⭐⭐ | Ultra-low |

---

## Quick Start

### 1. For Pre-Rendered Lessons (Best Quality)

```javascript
import { setupLessonLipSync, generateAlignment } from './app/lipsync/index.js';

// Get alignment from API (or pre-computed)
const alignment = await generateAlignment(
  'https://storage.curiouskelly.com/lessons/day1/audio.wav',
  'Hello everyone! Today we are learning about the amazing sun!'
);

// Set up lip-sync with audio element
const audioElement = document.getElementById('lesson-audio');
const lipSync = setupLessonLipSync(audioElement, alignment);

// Start playback
audioElement.play();
```

### 2. For Real-Time Conversation

```javascript
import { setupConversationLipSync } from './app/lipsync/index.js';

// Audio element playing Kelly's voice
const kellyAudio = document.getElementById('kelly-voice');

// Start real-time lip-sync
const lipSync = setupConversationLipSync(kellyAudio);

// Blendshapes will update automatically during playback
lipSync.onBlendshapesUpdate = (blendshapes) => {
  console.log('Current mouth shape:', blendshapes.jawOpen);
};
```

### 3. For ElevenLabs Streaming

```javascript
import { setupStreamingLipSync } from './app/lipsync/index.js';

// Start streaming lip-sync
const lipSync = setupStreamingLipSync();

// When receiving audio chunks from WebSocket
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.audio) {
    const audioBytes = atob(data.audio);
    lipSync.addStreamingAudioChunk(audioBytes);
  }
};
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     KELLY LIP-SYNC SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT SOURCES                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│  │ ElevenLabs    │  │ Pre-rendered  │  │ Live Audio    │                    │
│  │ Streaming     │  │ Audio Files   │  │ Playback      │                    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                    │
│          │                  │                  │                             │
│          ▼                  ▼                  ▼                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    KELLY LIPSYNC ORCHESTRATOR                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │ Streaming    │  │ Alignment    │  │ Real-time    │                  │  │
│  │  │ LipSync      │  │ Timeline     │  │ Analysis     │                  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │  │
│  │         │                 │                 │                           │  │
│  │         └─────────────────┼─────────────────┘                           │  │
│  │                           ▼                                              │  │
│  │              ┌──────────────────────┐                                   │  │
│  │              │  Blendshape Merger   │ ◄── Expression Generator         │  │
│  │              └──────────┬───────────┘     (eyes, brows, emotions)      │  │
│  │                         │                                               │  │
│  └─────────────────────────┼───────────────────────────────────────────────┘  │
│                            ▼                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      OUTPUT (53 ARKit Blendshapes)                     │  │
│  │  jawOpen, mouthFunnel, mouthPucker, mouthStretch, mouthSmile, ...     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                            │                                                 │
│            ┌───────────────┼───────────────┐                                │
│            ▼               ▼               ▼                                │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                       │
│    │ Unity WebGL  │ │ 2D Avatar    │ │ iClone/CC5   │                       │
│    │ (Real-time)  │ │ (Canvas)     │ │ (Pre-render) │                       │
│    └──────────────┘ └──────────────┘ └──────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phoneme → Viseme → Blendshape Pipeline

### 1. Phoneme Alignment (External)

```bash
# Using Montreal Forced Aligner
python scripts/forced-alignment/align_audio.py \
  --audio kelly_lesson.wav \
  --text "Hello everyone!" \
  --output alignment.json
```

Output:
```json
{
  "phones": [
    { "phone": "HH", "start": 0.0, "end": 0.05, "viseme": "A" },
    { "phone": "AH", "start": 0.05, "end": 0.15, "viseme": "A" },
    { "phone": "L", "start": 0.15, "end": 0.22, "viseme": "L" },
    { "phone": "OW", "start": 0.22, "end": 0.35, "viseme": "O" }
  ]
}
```

### 2. Viseme Categories

| Viseme | Phonemes | Mouth Shape |
|--------|----------|-------------|
| **A** | AA, AE, AH, HH | Wide open, jaw dropped |
| **E** | EH, EY | Teeth visible, slight smile |
| **I** | IH, IY, Y | Narrow smile |
| **O** | AO, OW, OY | Rounded, medium open |
| **U** | UH, UW, W | Pursed lips |
| **M** | P, B, M | Lips pressed together |
| **F** | F, V | Lower lip under teeth |
| **C** | T, D, N, S, Z, TH | Teeth close together |
| **L** | L | Tongue up |
| **R** | R, ER | Slightly rounded |
| **SH** | SH, CH, JH, ZH | Lips pushed forward |
| **REST** | SIL, SP | Neutral closed |

### 3. Blendshape Values

Each phoneme maps to ARKit-compatible blendshape values:

```javascript
'AA': {  // "father" - wide open
  jawOpen: 85,
  mouthOpen: 80,
  mouthFunnel: 0,
  mouthStretchLeft: 10,
  mouthStretchRight: 10,
}

'M': {  // "mom" - lips pressed
  jawOpen: 0,
  mouthClose: 100,
  mouthPressLeft: 40,
  mouthPressRight: 40,
}

'UW': {  // "boot" - pursed
  jawOpen: 20,
  mouthFunnel: 60,
  mouthPucker: 70,
}
```

---

## Unity Integration

### Kelly Avatar Setup

1. Ensure Kelly's character has an `ARKitBlendshapeController` component
2. The blendshape mesh should have all 52 ARKit face blendshapes

### Receiving Blendshapes from Web

```csharp
// KellyLipSyncReceiver.cs
using UnityEngine;
using System.Collections.Generic;

public class KellyLipSyncReceiver : MonoBehaviour
{
    private SkinnedMeshRenderer faceRenderer;
    
    void Start()
    {
        faceRenderer = GetComponent<SkinnedMeshRenderer>();
    }
    
    // Called from JavaScript via SendMessage
    public void SetBlendshapes(string json)
    {
        var blendshapes = JsonUtility.FromJson<BlendshapeData>(json);
        
        foreach (var kvp in blendshapes.values)
        {
            int index = faceRenderer.sharedMesh.GetBlendShapeIndex(kvp.Key);
            if (index >= 0)
            {
                faceRenderer.SetBlendShapeWeight(index, kvp.Value);
            }
        }
    }
}
```

### JavaScript Bridge

```javascript
// Send blendshapes to Unity
lipSync.onBlendshapesUpdate = (blendshapes) => {
  if (window.unityInstance) {
    window.unityInstance.SendMessage(
      'kelly_fbx_v4',
      'SetBlendshapes',
      JSON.stringify(blendshapes)
    );
  }
};
```

---

## API Reference

### `KellyLipSyncOrchestrator`

```javascript
const orchestrator = new KellyLipSyncOrchestrator({
  preferredMethod: 'alignment',  // 'alignment' | 'realtime' | 'streaming'
  enableExpressions: true,       // Merge with expression system
  fps: 30,                       // Blendshape update rate
  lipSyncWeight: 0.9,            // Lip-sync vs expression blend
  smoothTransitions: true,       // Smooth between phonemes
  alignmentApiUrl: '/api/align', // Alignment API endpoint
});

// Methods
orchestrator.playFromAlignment(alignment, audioElement);
orchestrator.startRealtimeFromAudio(audioElement);
orchestrator.startStreamingLipSync();
orchestrator.addStreamingAudioChunk(audioBuffer);
orchestrator.setExpression(expressionBlendshapes);
orchestrator.setUnityBridge(unityInstance);
orchestrator.stop();
orchestrator.dispose();

// Callbacks
orchestrator.onBlendshapesUpdate = (blendshapes) => {};
orchestrator.onPlaybackComplete = () => {};
orchestrator.onError = (error) => {};
```

### `/api/align` Endpoint

```
POST /api/align
Content-Type: application/json

{
  "audio_url": "https://...",  // OR audio_base64
  "transcript": "Hello everyone!"
}

Response:
{
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.35, "confidence": 0.95 }
  ],
  "phones": [
    { "phone": "HH", "start": 0.0, "end": 0.05, "viseme": "A" }
  ],
  "duration": 1.2,
  "method": "mfa",
  "confidence": 0.95
}
```

---

## Performance Tips

### Pre-Rendering (Lessons)

1. Generate alignments during content creation, not at runtime
2. Store alignment JSON alongside audio files in Supabase
3. Pre-compute blendshape timelines for 30fps playback

### Real-Time (Conversations)

1. Use `smoothing: 0.5-0.7` for natural movement
2. Set `updateRate: 30` (higher uses more CPU)
3. Enable `useFrequencyAnalysis: true` for better viseme hints

### Streaming (ElevenLabs)

1. Buffer at least 2-3 audio chunks before playing
2. Use `StreamingLipSync` class for automatic queue management
3. Call `clearQueue()` when user interrupts

---

## Troubleshooting

### Mouth Not Moving

1. Check audio is playing (not paused/muted)
2. Verify blendshape names match Unity mesh
3. Check `onBlendshapesUpdate` callback is firing

### Out of Sync

1. For alignment mode: verify alignment file matches audio
2. For real-time: reduce smoothing value
3. Check audio latency settings

### Unnatural Movement

1. Increase smoothing for less jitter
2. Enable coarticulation processing
3. Reduce sensitivity for quieter audio

---

## Files Reference

| File | Purpose |
|------|---------|
| `app/lipsync/index.js` | Main entry point, exports all modules |
| `app/lipsync/phoneme-viseme-map.js` | ARPAbet → Blendshape mapping |
| `app/lipsync/realtime-lipsync.js` | Audio analysis classes |
| `app/lipsync/kelly-lipsync-orchestrator.js` | Main orchestrator |
| `api/align.ts` | Alignment API endpoint |
| `scripts/forced-alignment/align_audio.py` | MFA alignment script |
| `scripts/test-kelly-lipsync.js` | Test suite |

---

## Next Steps

1. **Montreal Forced Aligner Setup**: Run alignment offline for all 365 lessons
2. **Expression Integration**: Connect to existing expression-generator.js
3. **Unity Testing**: Validate blendshape application on Kelly model
4. **ElevenLabs Feature Request**: Push for native alignment data in API

---

## Contact

Questions? Email hello@curiouskelly.com or check `/docs/ELEVENLABS_OPTIMAL_SETUP.md`.

