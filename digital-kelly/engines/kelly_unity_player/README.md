# Kelly Unity Player

Unity URP project for rendering Kelly's 3D avatar with frame-accurate audio-lip-sync.

## Project Structure

```
Assets/
├── Kelly/
│   ├── Scripts/      # BlendshapeDriver, AutoBlink, KellyBridge, etc.
│   ├── Models/       # Kelly_Head.fbx (placeholder: sphere)
│   ├── Audio/        # Test audio files
│   ├── Data/         # A2F JSON data
│   ├── Materials/    # URP materials
│   └── Scenes/       # Main.unity
```

## Scene Setup

### Main.unity Scene

1. **Camera**
   - FOV: 38
   - Position: Tight head close-up
   - Background: Black

2. **Lighting**
   - Directional Light (Soft shadows)

3. **Kelly_Head GameObject**
   - Currently: Placeholder sphere with test blendshape "jawOpen"
   - Future: Replace with Kelly_Head.fbx

### KellyController

Empty GameObject with:
- `KellyBridge` component (Flutter communication)
- `BlendshapeDriver` component (A2F animation)
- `AudioSource` component (audio playback)

Optional components:
- `AutoBlink` - Automatic blinking
- `BreathingLayer` - Micro-expressions for breathing

## Scripts

### BlendshapeDriver.cs
- Loads A2F JSON frames
- Synchronizes audio playback with frames using `AudioSettings.dspTime`
- Applies blendshape weights in real-time

### KellyBridge.cs
- Receives messages from Flutter
- Loads audio clips and A2F data
- Triggers synchronized playback

### AutoBlink.cs
- Generates natural blinking every 3-6 seconds
- Blendshape-based (does not interfere with speech)

### BreathingLayer.cs
- Adds subtle breathing micro-expression
- Sinusoidal animation
- Low amplitude (4 units)

## Build Settings

### Unity Version
- 2022.3 LTS or 2023.x
- URP enabled
- iOS/Android platform modules

### Export for Flutter

1. Open `Main.unity`
2. File → Build Settings
3. Select iOS or Android
4. Export to corresponding directory in Flutter project

See `docs/EMBED.md` for detailed integration steps.

## Testing

1. Open Main.unity in Unity Editor
2. Press Play
3. In Console, manually trigger:
   ```csharp
   var bridge = FindObjectOfType<KellyBridge>();
   bridge.LoadAndPlay("path/to/json|path/to/wav");
   ```
4. Verify blendshape weights animate in real-time

## Next Steps

- Replace placeholder mesh with real FBX
- Add jaw, smile, eyebrows blendshapes
- Calibrate sync timing (±60ms slider)
- Priority system for blink/breathing layers


















