# Kelly Unity Player (Clean Slate)

**Status:** Ready for Final Assets (Incoming)
**Unity Version:** 6000.2.10f1 (Recommended) or 2022.3 LTS

This project has been cleaned to prepare for the final Kelly model and scene integration.

## Project Structure

```
Assets/
├── Scripts/      # Core Logic (Preserved)
│   ├── KellyBridge.cs          # WebGL Messaging Bridge
│   ├── BlendshapeDriver.cs     # Audio2Face Animation Driver
│   ├── KellyAvatarController.cs # Main Controller
│   └── ...
├── Editor/       # Build Tools
│   └── WebGLBuild.cs
├── Models/       # [EMPTY] Place final FBX here
├── Scenes/       # [EMPTY] Place final Scene here
├── Materials/    # [EMPTY] Place textures/materials here
└── Prefabs/      # [EMPTY] Place prefabs here
```

## How to Integrate the New Model

1.  **Open Project:** Open this folder (`digital-kelly/engines/kelly_unity_player`) in Unity Hub.
2.  **Import Assets:** Drag the new `.fbx`, textures, and scene files into the respective folders.
3.  **Setup Scene:**
    *   Open the imported scene (or create a new one in `Scenes/`).
    *   Ensure the Kelly GameObject has the `KellyAvatarController` script attached.
    *   Ensure the `KellyBridge` script is in the scene (usually on a Controller object).
4.  **Build:**
    *   Run `scripts/build_unity_webgl.ps1` from the repo root.

## Core Scripts Overview

*   **KellyBridge.cs**: Handles communication with the web browser (Lesson Player).
*   **BlendshapeDriver.cs**: Syncs audio with facial blendshapes (A2F).
*   **KellyAvatarController.cs**: Manages age variants and overall state.
