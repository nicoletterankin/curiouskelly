# Unity WebGL Build & Deploy Guide

## Quick Start

### Option 1: Build via Unity Editor (Recommended for first time)

1. **Open Unity Project:**
   - Open Unity Hub
   - Open project at: `digital-kelly/engines/kelly_unity_player`
   - Wait for Unity to finish importing assets

2. **Configure Build Settings:**
   - Go to `File > Build Settings`
   - Select `WebGL` platform
   - Click `Switch Platform` if needed
   - Add scenes to build:
     - Click `Add Open Scenes` or manually add `Assets/Kelly/Scenes/Main.unity` (if it exists)
     - Or add `Assets/Scenes/SampleScene.unity` for testing
   - Ensure at least one scene is checked/enabled

3. **Build via Menu:**
   - Go to `Kelly > Build > WebGL (iframe bundle)` in the Unity menu
   - Or use `File > Build Settings > Build`
   - Output will be in `Builds/WebGL/kelly-v1/`

4. **Deploy:**
   ```powershell
   scripts\deploy_unity_webgl.ps1 -SkipBuild
   ```

### Option 2: Build via Command Line

1. **Ensure scenes are configured:**
   - Open Unity once to configure Build Settings
   - Add at least one scene to the build

2. **Run build script:**
   ```powershell
   scripts\deploy_unity_webgl.ps1
   ```

   Or manually:
   ```powershell
   scripts\build_unity_webgl.ps1 -UnityPath "C:\Program Files\Unity\Hub\Editor\6000.2.10f1\Editor\Unity.exe"
   ```

3. **Deploy build:**
   ```powershell
   scripts\deploy_unity_webgl.ps1 -SkipBuild
   ```

## Troubleshooting

### Build fails with "No enabled scenes found"

**Solution:** Open Unity and configure Build Settings:
1. Open Unity project
2. Go to `File > Build Settings`
3. Add scenes to build (click "Add Open Scenes" or drag scenes from Project window)
4. Ensure scenes are checked/enabled
5. Save project
6. Try building again

### Unity version mismatch

The build script looks for Unity 6.x or 2022.3.x. If you have a different version:
```powershell
scripts\deploy_unity_webgl.ps1 -UnityPath "C:\Path\To\Your\Unity\Editor\Unity.exe"
```

### Build succeeds but files aren't in public folder

Run the deployment step separately:
```powershell
scripts\deploy_unity_webgl.ps1 -SkipBuild
```

This will copy the build from `Builds/WebGL/kelly-v1/` to `public/unity/kelly-v1/`

## Verification

After deployment, verify:

1. **Files exist:**
   - `public/unity/kelly-v1/index.html` exists
   - `public/unity/kelly-v1/kbridge.js` exists
   - Unity build files (`.wasm`, `.data`, etc.) are present

2. **Test locally:**
   ```powershell
   # Install serve if needed
   npm install -g serve
   
   # Serve the public folder
   serve public
   
   # Visit http://localhost:3000/unity/kelly-v1/
   ```

3. **Test in lesson player:**
   - Open `lesson-player/index.html` in a browser
   - Check browser console for Unity messages
   - Unity iframe should load and show "Kelly is ready"

## File Structure After Deployment

```
public/unity/kelly-v1/
├── index.html          (Unity build HTML)
├── kbridge.js          (Messaging bridge)
├── Build/              (Unity build files)
│   ├── *.wasm
│   ├── *.data
│   └── *.js
└── TemplateData/       (Unity template assets)
```

## Next Steps

Once deployed:
1. Test the Unity iframe loads correctly
2. Verify messaging bridge works (check browser console)
3. Test lesson loading in the lesson player
4. Deploy to production hosting




