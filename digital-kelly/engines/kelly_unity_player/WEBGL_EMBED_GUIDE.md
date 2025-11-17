# Kelly WebGL Embed Guide

End-to-end checklist for shipping the Kelly avatar WebGL build and surfacing it inside the `curiouskelly.com` iframe.

## 1. Build Requirements

- Unity 2022.3 LTS (with WebGL module installed)
- Project path: `digital-kelly/engines/kelly_unity_player`
- Scenes: enable `Assets/Kelly/Scenes/Main.unity` (and any additional demo scenes) in **Build Settings**

## 2. One-Command WebGL Build

The new `Assets/Editor/WebGLBuild.cs` script defines a deterministic build pipeline. Use whichever entry point you prefer:

- **PowerShell helper** (recommended):  
  `scripts\build_unity_webgl.ps1 -UnityPath "C:\Program Files\Unity\Hub\Editor\6000.2.1f1\Editor\Unity.exe" -Version kelly-v1`
- **Raw CLI**:

  ```powershell
  $env:KELLY_WEBGL_VERSION="kelly-v1"
  "C:\Program Files\Unity\Hub\Editor\2022.3.xx\Editor\Unity.exe" `
    -quit -batchmode `
    -projectPath "$PWD\digital-kelly\engines\kelly_unity_player" `
    -executeMethod Kelly.Editor.WebGLBuild.Build `
    -logFile Builds\kelly-webgl.log
  ```

Outputs land in `digital-kelly/engines/kelly_unity_player/Builds/WebGL/<version>/`. The PowerShell wrapper automatically prints the final folder size and log path.

Player settings enforced by the script:

- Compression: **Brotli** + `DecompressionFallback`
- Linker target: **Wasm**
- Exceptions: full stack trace (useful for iframe debugging)
- Files named as hashes for safe CDN caching

## 3. Browser Messaging Layer

`Assets/Plugins/WebGL/KellyBrowserBridge.jslib` bridges `window.postMessage` ↔ Unity.

- Parent page `postMessage({ destination: 'kelly-webgl', type: 'kelly-load', payload: { jsonUrl, audioUrl } })`
- Unity responds with `kelly-ready`, `kelly-loading`, `kelly-playing`, `kelly-stopped`, `kelly-error`
- No additional template hacks required—bridge registers itself during `KellyBridge.Awake()`

## 4. Remote Lesson Loader

`KellyBridge` now supports:

- `LoadLessonFromUrls(string jsonPayload)` for manual testing
- `HandleBrowserMessage(string payloadJson)` automatically invoked in WebGL builds
- Remote viseme JSON, optional expression cues, MP3/WAV audio
- Graceful stop + state reset when a new lesson arrives or when the parent sends `kelly-stop`

Payload schema:

```json
{
  "destination": "kelly-webgl",
  "type": "kelly-load",
  "payload": {
    "lessonId": "water-cycle-18-35",
    "jsonUrl": "https://cdn.curiouskelly.com/unity/kelly-v1/content/water-cycle.a2f.json",
    "audioUrl": "https://cdn.curiouskelly.com/unity/kelly-v1/audio/water-cycle-18-35.mp3",
    "expressionsUrl": "https://cdn.curiouskelly.com/unity/kelly-v1/content/water-cycle.expressions.json",
    "offsetMs": 50
  }
}
```

## 5. Deploying to the Marketing Site

1. Run the build command above.
2. Copy the entire `Builds/WebGL/<version>/` directory into `public/unity/<version>/` inside the `curiouskelly.com` repo **or** upload to `https://assets.curiouskelly.com/unity/<version>/` (preferred for CDN caching).
3. Upload the lesson assets referenced by the iframe controls (viseme JSON, optional expression cues, MP3/WAV). Keep them under `/unity/<version>/content/` and `/unity/<version>/audio/` so the URLs stay tidy.
4. Ensure `index.html` references `./kbridge.js` (already placed in `public/unity/kelly-v1/`) *after* the Unity loader script so the parent iframe can talk to the player.
5. Update `curiouskelly-marketing-site/.env` (or deployment secrets):

   ```
   PUBLIC_UNITY_IFRAME_SRC=https://assets.curiouskelly.com/unity/kelly-v1/index.html
   PUBLIC_UNITY_SAMPLE_JSON=https://assets.curiouskelly.com/unity/kelly-v1/content/water-cycle.a2f.json
   PUBLIC_UNITY_SAMPLE_AUDIO=https://assets.curiouskelly.com/unity/kelly-v1/audio/water-cycle-18-35.mp3
   PUBLIC_UNITY_SAMPLE_EXPRESSIONS=https://assets.curiouskelly.com/unity/kelly-v1/content/water-cycle.expressions.json
   ```

6. Run `pnpm dev` → visit `http://localhost:4321/demo/avatar/` → click “Play sample lesson” to confirm audio + visemes load. `Play` stays disabled if the env vars above are missing, so this is an instant sanity check.

## 6. Caching & Headers

Serve the Unity files from a CDN with:

- `Content-Encoding: br` for `.data`, `.wasm`, `.js`
- `Cross-Origin-Embedder-Policy: credentialless`
- `Cross-Origin-Opener-Policy: same-origin`
- `Cache-Control: public, max-age=31536000, immutable`

These match the requirements for the marketing iframe and avoid COOP/COEP violations.

## 7. Future Work

- Swap the placeholder audio/viseme URLs in the marketing site for production CDN locations. Env vars make this a one-line redeploy.
- Extend `KellyBridge` to stream live visemes via WebSockets (the `HandleBrowserMessage` pipeline is ready for additional message types).
- Automate build + upload via CI so every Unity commit emits a versioned WebGL bundle + manifest.

## Appendix – Smoke Test Checklist

1. `serve Builds/WebGL/<version>` → open `http://localhost:3000`.
2. Confirm the Unity loading bar completes.
3. Press F12 → Network tab → verify `.data.br` and `.wasm.br` use Brotli.
4. In `/demo/avatar/`, watch the status chip transitions (`Waiting → Ready → Loading → Playing`).
5. GA4 Realtime should show events when Play/Stop are triggered (event name: `unity_demo_event`).

