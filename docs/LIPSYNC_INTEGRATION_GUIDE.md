# Kelly Lip-Sync Integration Guide

Complete guide for Kelly's real-time lip-sync system.

## Overview

Kelly's lip-sync system provides real-time mouth animation synchronized with:
- **ElevenLabs Streaming Audio** - Live conversational AI
- **Pre-rendered Audio** - Lesson content playback
- **Expression System** - Emotional expressions (eyes, brows, gestures)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Kelly Animation Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌───────────────────┐                      │
│  │  ElevenLabs      │    │  Pre-rendered     │                      │
│  │  Streaming       │    │  Audio Files      │                      │
│  └────────┬─────────┘    └─────────┬─────────┘                      │
│           │                        │                                 │
│           ▼                        ▼                                 │
│  ┌─────────────────────────────────────────────┐                    │
│  │           KellyLipSync.js                    │                    │
│  │  • Real-time audio analysis                  │                    │
│  │  • Amplitude → mouth opening                 │                    │
│  │  • Frequency → mouth shape (vowels, etc.)   │                    │
│  └────────────────────┬────────────────────────┘                    │
│                       │                                              │
│                       ▼                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │       KellyExpressionBridge.js              │                    │
│  │  • Merges lip-sync with expressions         │                    │
│  │  • Eyes, brows, gestures from archetype     │                    │
│  │  • Phase-based expression transitions       │                    │
│  └────────────────────┬────────────────────────┘                    │
│                       │                                              │
│           ┌───────────┴───────────┐                                  │
│           ▼                       ▼                                  │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │  Unity Bridge   │    │  2D Avatar      │                         │
│  │  (3D Kelly)     │    │  (Fallback)     │                         │
│  └─────────────────┘    └─────────────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Include the Scripts

Add these scripts to your HTML (order matters):

```html
<!-- Core lip-sync system -->
<script src="/js/kelly-lipsync.js"></script>

<!-- Expression bridge (optional, for full facial animation) -->
<script src="/js/kelly-expression-bridge.js"></script>
```

### 2. Basic Usage

```javascript
// Initialize (auto-initializes on DOM ready, but can be called manually)
KellyLipSync.init();

// For ElevenLabs streaming audio
KellyLipSync.startStreaming();

// Feed audio chunks from WebSocket
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.audio) {
    KellyLipSync.addAudioChunk(data.audio);  // base64 audio
  }
};

// For pre-rendered audio
const audio = new Audio('/audio/lesson.mp3');
KellyLipSync.startFromAudioElement(audio);
audio.play();
```

### 3. With Expression Bridge

```javascript
// Initialize both systems
KellyExpressionBridge.init();  // Also initializes KellyLipSync

// Set Kelly's archetype for appropriate expressions
KellyExpressionBridge.setArchetype('The Explorer');

// Set expression based on lesson phase
KellyExpressionBridge.setPhaseExpression('welcome');  // excited
KellyExpressionBridge.setPhaseExpression('q1');       // curious
KellyExpressionBridge.setPhaseExpression('wisdom');   // wisdom

// React to user actions
KellyExpressionBridge.reactToUser('correct');   // celebrating
KellyExpressionBridge.reactToUser('incorrect'); // encouraging (curious)

// Manual expression control
KellyExpressionBridge.setExpression('thinking', { duration: 400 });
```

## Integration with Existing Systems

### KellyConversation (Streaming)

The lip-sync is already wired into `kelly-conversation.js`:

```javascript
// Automatically starts streaming lip-sync when conversation begins
KellyConversation.startConversation();

// Audio chunks are automatically fed to lip-sync
// See: queueAudio() method in kelly-conversation.js
```

### KellyAudio (Pre-rendered)

The lip-sync is already wired into `kelly-audio.js`:

```javascript
const kellyAudio = new KellyAudio({
  lipSyncEnabled: true  // default
});

// Lip-sync automatically connects when audio plays
kellyAudio.speak("Hello, learner!");
```

### Unity Bridge

The Unity bridge has been updated to receive blendshape data:

```javascript
// Direct blendshape control
unityBridge.setBlendshapes({
  jawOpen: 50,
  mouthSmileLeft: 30,
  mouthSmileRight: 30,
});

// Single blendshape
unityBridge.setSingleBlendshape('jawOpen', 60);

// Reset to neutral
unityBridge.resetBlendshapes();
```

## Blendshape Reference

### Mouth Shapes (Lip-Sync)

| Blendshape | Description | Range |
|------------|-------------|-------|
| `jawOpen` | Jaw opening | 0-100 |
| `mouthOpen` | Mouth opening | 0-100 |
| `mouthFunnel` | Rounded mouth (O) | 0-100 |
| `mouthPucker` | Pursed lips | 0-100 |
| `mouthStretchLeft/Right` | Wide mouth (E, I) | 0-100 |
| `mouthSmileLeft/Right` | Smile | 0-100 |
| `mouthPressLeft/Right` | Pressed lips | 0-100 |
| `mouthUpperUpLeft/Right` | Upper lip raise | 0-100 |
| `mouthLowerDownLeft/Right` | Lower lip drop | 0-100 |
| `mouthClose` | Closed mouth | 0-100 |

### Expression Shapes (Eyes/Brows)

| Blendshape | Description | Range |
|------------|-------------|-------|
| `browInnerUp` | Inner brow raise | 0-100 |
| `browOuterUpLeft/Right` | Outer brow raise | 0-100 |
| `eyeWideLeft/Right` | Wide eyes | 0-100 |
| `eyeSquintLeft/Right` | Squinting | 0-100 |
| `eyeBlinkLeft/Right` | Blink/closed eyes | 0-100 |
| `cheekSquintLeft/Right` | Cheek raise | 0-100 |

## Configuration

### KellyLipSync Configuration

```javascript
KellyLipSync.updateConfig({
  // Smoothing (0-1, higher = smoother but less responsive)
  smoothing: 0.55,
  
  // Minimum amplitude to trigger mouth movement
  minAmplitude: 0.02,
  
  // Maximum jaw opening
  maxJawOpen: 85,
  
  // Sensitivity multiplier
  sensitivity: 1.6,
  
  // Frame rate
  updateRate: 30,
  
  // Decay rate when audio stops
  decayRate: 0.18,
  
  // Send to Unity 3D avatar
  sendToUnity: true,
  
  // Send to 2D avatar fallback
  sendTo2D: true,
  
  // Debug logging
  debug: false,
});
```

### Expression Transition Duration

```javascript
// Faster transitions for responsive feedback
KellyExpressionBridge.setExpression('excited', { duration: 200 });

// Slower transitions for subtle changes
KellyExpressionBridge.setExpression('wisdom', { duration: 500 });
```

## Expression Presets

| Expression | Use Case |
|------------|----------|
| `neutral` | Default state |
| `thinking` | Processing, considering |
| `curious` | Questions, exploration |
| `explaining` | Teaching content |
| `excited` | Welcome, discoveries |
| `surprised` | Unexpected moments |
| `listening` | Waiting for user input |
| `celebrating` | Correct answers, achievements |
| `wisdom` | Philosophical moments |

### Archetype-Specific Expressions

When an archetype is set, these expressions have enhanced personality:

- `scientist_analyzing` - Analytical focus
- `explorer_discovery` - High energy discovery
- `storyteller_dramatic` - Theatrical flair
- `empath_caring` - Warm connection
- `coach_encouraging` - Motivational energy
- `artist_creative` - Creative inspiration

## Troubleshooting

### Lip-sync not working

1. Check if AudioContext is resumed (requires user interaction):
```javascript
await KellyLipSync.resume();
```

2. Verify audio is playing:
```javascript
console.log(KellyLipSync.getIsSpeaking());
```

3. Check blendshape output:
```javascript
console.log(KellyLipSync.getBlendshapes());
```

### Unity not receiving blendshapes

1. Verify Unity is ready:
```javascript
console.log(unityBridge.ready);
```

2. Check if `SetBlendshapes` method exists in Unity script

3. Verify JSON parsing in Unity side

### Expressions not transitioning

1. Check if expression bridge is initialized:
```javascript
console.log(KellyExpressionBridge.isInitialized);
```

2. Verify expression name:
```javascript
console.log(KellyExpressionBridge.getCurrentExpression());
```

## Unity C# Integration

Add this method to your `KellyWebGLBridge.cs`:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class KellyWebGLBridge : MonoBehaviour
{
    public SkinnedMeshRenderer faceRenderer;
    
    // Map of web blendshape names to Unity blendshape indices
    private Dictionary<string, int> blendshapeMap;
    
    void Start()
    {
        BuildBlendshapeMap();
    }
    
    void BuildBlendshapeMap()
    {
        blendshapeMap = new Dictionary<string, int>();
        
        if (faceRenderer == null) return;
        
        var mesh = faceRenderer.sharedMesh;
        for (int i = 0; i < mesh.blendShapeCount; i++)
        {
            string name = mesh.GetBlendShapeName(i);
            blendshapeMap[name] = i;
        }
    }
    
    // Called from JavaScript via SendMessage
    public void SetBlendshapes(string json)
    {
        var data = JsonUtility.FromJson<BlendshapeData>(json);
        
        foreach (var kvp in data.shapes)
        {
            if (blendshapeMap.TryGetValue(kvp.Key, out int index))
            {
                faceRenderer.SetBlendShapeWeight(index, kvp.Value);
            }
        }
    }
    
    public void ResetBlendshapes(string _)
    {
        if (faceRenderer == null) return;
        
        for (int i = 0; i < faceRenderer.sharedMesh.blendShapeCount; i++)
        {
            faceRenderer.SetBlendShapeWeight(i, 0);
        }
    }
}

[System.Serializable]
public class BlendshapeData
{
    public Dictionary<string, float> shapes;
}
```

## Performance Notes

- Lip-sync runs at 30fps by default (configurable)
- Blendshape updates to Unity are throttled to ~30fps
- Expression transitions use requestAnimationFrame
- Audio analysis uses Web Audio API analyser node

## Files Reference

| File | Purpose |
|------|---------|
| `/public/js/kelly-lipsync.js` | Core lip-sync engine |
| `/public/js/kelly-expression-bridge.js` | Expression + lip-sync merger |
| `/public/js/kelly-conversation.js` | ElevenLabs streaming integration |
| `/public/js/kelly-audio.js` | Pre-rendered audio integration |
| `/public/js/unity-bridge.js` | Unity communication |
| `/app/lipsync/` | Server-side lip-sync components |

---

Built for Curious Kelly © 2024 Lesson of the Day PBC


