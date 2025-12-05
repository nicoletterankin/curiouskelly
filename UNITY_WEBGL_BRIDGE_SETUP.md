# 🎮 UNITY WEBGL BRIDGE - SETUP COMPLETE

## ✅ FILES UPDATED

### Unity C# Scripts (Digital Kelly Project)

1. **ARKitBlendshapeController.cs** ✅ UPDATED
   - Path: `digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/Scripts/ARKitBlendshapeController.cs`
   - Changes:
     - Added `Start()` method with auto-initialization
     - Added `InitializeBlendshapeMap()` to auto-populate blendshape dictionary
     - Added partial-match fallback in `SetBlendshape()` for flexibility
     - Updated test blendshape name to `V_Open`

2. **KellyWebGLBridge.cs** ✅ CREATED
   - Path: `digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/Scripts/KellyWebGLBridge.cs`
   - Features:
     - Public methods for JavaScript SendMessage calls
     - Expression system with 7 expressions (neutral, happy, curious, explaining, listening, wisdom, celebrating)
     - Lip sync simulation
     - Phase-to-expression mapping
     - Idle blinking behavior
     - Smooth transitions between expressions

### Web JavaScript

3. **unity-kelly-loader.js** ✅ UPDATED
   - Path: `public/js/unity-kelly-loader.js`
   - Changes:
     - Updated `getKellyObjectName()` to return `kelly_fbx_v4` (was `Kelly_Live_v2`)
     - Added documentation comment

---

## 🎯 NEXT STEPS - UNITY EDITOR

### Step 1: Open Unity Project

```
Open: C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\
Scene: KellyMain
```

### Step 2: Attach KellyWebGLBridge Script

1. In Hierarchy, find GameObject: **kelly_fbx_v4**
2. In Inspector, click **Add Component**
3. Search for: **KellyWebGLBridge**
4. Click to attach

### Step 3: Assign References (Auto-Finds, But Verify)

The script auto-finds these, but you can manually assign for certainty:

**KellyWebGLBridge Component:**
- **Blendshapes:** Drag `ARKitBlendshapeController` component
- **Animator:** Drag Animator component (if you have animations)

**ARKitBlendshapeController Component:**
- **Head Renderer:** Drag the SkinnedMeshRenderer for Kelly's head (usually `CC_Base_Body`)

### Step 4: Verify GameObject Name

**CRITICAL:** The GameObject MUST be named exactly: `kelly_fbx_v4`

If it's named differently:
- **Option A:** Rename it to `kelly_fbx_v4` in Unity
- **Option B:** Update `unity-kelly-loader.js` line 228 to match the actual name

### Step 5: Build WebGL

1. **File → Build Settings**
2. **Platform:** WebGL
3. **Build folder:** Choose a temporary location first (e.g., `Desktop/KellyBuild`)
4. Click **Build**
5. Wait for build to complete (~5-15 minutes)

### Step 6: Copy Build to Web Project

After build completes, copy these files:

**FROM:** `[Build folder]/Build/`
**TO:** `C:\Users\user\UI-TARS-desktop\public\unity\kelly\Build\`

Files to copy:
- `Kelly_Web_Build.data.unityweb`
- `Kelly_Web_Build.framework.js.unityweb`
- `Kelly_Web_Build.loader.js`
- `Kelly_Web_Build.wasm.unityweb`

**Note:** The build name must match `buildName` in `unity-kelly-loader.js` (line 18):
```javascript
buildName: options.buildName || 'Kelly_Web_Build',
```

If your build has a different name, either:
- Rename the files to `Kelly_Web_Build.*`
- OR update the `buildName` in the loader

---

## 🧪 TESTING

### Test in Unity Editor (Before Building)

1. Enter Play Mode
2. Check Console for: `[KellyWebGLBridge] Ready for JavaScript commands`
3. Check Console for: `[ARKitBlendshapeController] Initialized X blendshapes`
4. Verify no errors

### Test WebGL Build Locally

1. Start local server:
   ```bash
   cd C:\Users\user\UI-TARS-desktop
   python -m http.server 8080
   ```

2. Open browser: `http://localhost:8080/learn.html?day=1`

3. Open DevTools Console (F12)

4. Wait for Unity to load (look for progress bar)

5. Test commands manually in console:
   ```javascript
   // Get Unity instance (should be available after load)
   window.unityKellyInstance
   
   // Test expression
   window.unityKellyInstance.SendMessage("kelly_fbx_v4", "SetExpression", "happy");
   
   // Test lip sync
   window.unityKellyInstance.SendMessage("kelly_fbx_v4", "StartLipSync", "Hello world");
   
   // Test phase
   window.unityKellyInstance.SendMessage("kelly_fbx_v4", "SetPhase", "welcome");
   
   // Stop lip sync
   window.unityKellyInstance.SendMessage("kelly_fbx_v4", "StopLipSync");
   ```

6. Check for errors in console:
   - ❌ "SendMessage: object kelly_fbx_v4 does not have receiver" → Script not attached
   - ❌ "SendMessage: object kelly_fbx_v4 does not have receiver for function SetExpression" → Method missing
   - ✅ "[KellyWebGLBridge] SetExpression: happy" → Working!

---

## 📋 EXPRESSION REFERENCE

### Available Expressions

| Expression | Use Case | Blendshapes Used |
|------------|----------|------------------|
| `neutral` | Default, idle | All zeros |
| `happy` | Positive feedback, celebrating | Smile, cheek raise, eye squint |
| `curious` | Questions, wondering | Brow raise inner, eyes wide |
| `explaining` | Teaching, answering | Brow raise outer, mouth shrug |
| `listening` | Waiting for response | Slight smile, brow raise |
| `wisdom` | Wisdom phase, profound | Warm smile, soft eyes |
| `celebrating` | Correct answer! | Big smile, cheeks, brows |

### JavaScript Usage

```javascript
// Via loader instance
const loader = new UnityKellyLoader();
await loader.load();
loader.setExpression('happy');
loader.startLipSync('Hello, I am Kelly!');
loader.stopLipSync();

// Direct SendMessage
loader.sendMessage('kelly_fbx_v4', 'SetExpression', 'curious');
loader.sendMessage('kelly_fbx_v4', 'SetPhase', 'welcome');
```

### Phase-to-Expression Mapping

The `SetPhase()` method automatically maps lesson phases to expressions:

| Phase | Expression |
|-------|------------|
| `welcome` | curious |
| `question`, `q1`, `q2`, `q3` | explaining |
| `wisdom` | wisdom |
| `celebrating` | celebrating |
| (other) | neutral |

---

## 🐛 TROUBLESHOOTING

### Issue: "SendMessage: object kelly_fbx_v4 does not have receiver"

**Cause:** GameObject name mismatch

**Fix:**
1. Check GameObject name in Unity Hierarchy
2. Verify it matches `getKellyObjectName()` in `unity-kelly-loader.js`
3. If different, either rename GameObject or update JS

### Issue: "SendMessage: object kelly_fbx_v4 does not have receiver for function SetExpression"

**Cause:** KellyWebGLBridge script not attached or not in build

**Fix:**
1. Verify script is attached to `kelly_fbx_v4` in Unity
2. Rebuild WebGL
3. Copy new build to `public/unity/kelly/Build/`

### Issue: Expressions not visible / no face movement

**Cause:** ARKitBlendshapeController not finding SkinnedMeshRenderer

**Fix:**
1. Check Unity Console for: `[ARKitBlendshapeController] Initialized X blendshapes`
2. If X = 0, manually assign `headRenderer` in Inspector
3. The SkinnedMeshRenderer should be on the mesh with blendshapes (usually `CC_Base_Body`)

### Issue: Blendshape names not found

**Cause:** Blendshape names in CC4 model don't match code

**Fix:**
1. In Unity, select the SkinnedMeshRenderer
2. Expand "BlendShapes" in Inspector
3. Note the exact names (e.g., `Mouth_Smile_L` vs `mouthSmile_L`)
4. Update `GetExpressionWeights()` in `KellyWebGLBridge.cs` to match
5. OR rely on partial-match fallback (already implemented)

### Issue: Unity build is huge (>100MB)

**Cause:** Default compression settings

**Fix:**
1. **Edit → Project Settings → Player → WebGL**
2. **Publishing Settings:**
   - Compression Format: **Gzip** or **Brotli**
   - Code Optimization: **Disk Size with LTO**
3. Rebuild

### Issue: Unity takes forever to load on web

**Cause:** Large build, slow network, or low-end device

**Fix:**
1. Enable compression (see above)
2. Implement fallback to 2D (already in `kelly-avatar-controller.js`)
3. Show progress bar during load (already in `unity-kelly-loader.js`)

---

## 🎨 CUSTOMIZING EXPRESSIONS

To add new expressions, edit `KellyWebGLBridge.cs`:

```csharp
case "surprised":
    weights["Brow_Raise_Outer_L"] = 80f;
    weights["Brow_Raise_Outer_R"] = 80f;
    weights["Eye_Wide_L"] = 60f;
    weights["Eye_Wide_R"] = 60f;
    weights["Mouth_Open"] = 40f;
    break;
```

Blendshape weight range: **0-100** (0% to 100%)

---

## 📊 BLENDSHAPE REFERENCE

### Available Blendshapes (CC4 ARKit)

**Visemes (Lip Sync):**
- V_Open, V_Explosive, V_Dental_Lip, V_Tight_O, V_Tight, V_Wide, V_Affricate, V_Lip_Open

**Brows:**
- Brow_Raise_Inner_L/R, Brow_Raise_Outer_L/R, Brow_Drop_L/R, Brow_Compress_L/R

**Eyes:**
- Eye_Blink_L/R, Eye_Wide_L/R, Eye_Squint_L/R

**Nose:**
- Nose_Sneer_L/R, Nose_Nostril_Raise_L/R

**Cheeks:**
- Cheek_Raise_L/R, Cheek_Suck_L/R, Cheek_Puff_L/R

**Mouth:**
- Mouth_L/R, Mouth_Up/Down, Mouth_Smile_L/R, Mouth_Frown_L/R, Mouth_Shrug_Upper/Lower, Mouth_Drop_Upper/Lower, Mouth_Up_Upper_L/R, Mouth_Down_Lower_L/R, Mouth_Chin_Up, Mouth_Close, Mouth_Contract

---

## ✅ CHECKLIST

Before rebuilding Unity:
- [ ] KellyWebGLBridge.cs exists in Assets/Scripts/
- [ ] ARKitBlendshapeController.cs updated with auto-init
- [ ] KellyWebGLBridge attached to kelly_fbx_v4 GameObject
- [ ] GameObject name is exactly: `kelly_fbx_v4`
- [ ] References assigned (or auto-find verified)
- [ ] Tested in Play Mode (no errors in console)

After rebuilding Unity:
- [ ] Build completed without errors
- [ ] Build files copied to public/unity/kelly/Build/
- [ ] File names match buildName in unity-kelly-loader.js
- [ ] Local server running
- [ ] Browser test: learn.html loads
- [ ] Unity loads (progress bar completes)
- [ ] Console shows: "[KellyWebGLBridge] Ready for JavaScript commands"
- [ ] Manual SendMessage test works

After web integration:
- [ ] Expressions change during lesson phases
- [ ] Mouth moves during speech
- [ ] No console errors
- [ ] Fallback to 2D works if Unity fails
- [ ] Performance acceptable (30+ FPS)

---

## 🚀 DEPLOYMENT

After successful local testing:

1. Commit Unity scripts:
   ```bash
   cd C:\Users\user\UI-TARS-desktop
   git add digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/Scripts/
   git add public/js/unity-kelly-loader.js
   git commit -m "feat: Unity WebGL bridge for Kelly expressions and lip sync"
   ```

2. Commit Unity build (if not too large):
   ```bash
   git add public/unity/kelly/Build/
   git commit -m "build: Unity WebGL build with KellyWebGLBridge"
   ```

3. Push to deploy:
   ```bash
   git push origin main
   ```

4. Verify on production:
   - https://curiouskelly.com/learn.html?day=1
   - Check DevTools console for Unity load
   - Test expressions during lesson

---

## 📞 SUPPORT

If you encounter issues:

1. **Check Unity Console** for initialization messages
2. **Check Browser Console** for JavaScript errors
3. **Verify GameObject name** matches exactly
4. **Verify script is attached** in Unity Inspector
5. **Rebuild and re-copy** Unity build files

**Common Success Indicators:**
- Unity Console: `[KellyWebGLBridge] Ready for JavaScript commands`
- Unity Console: `[ARKitBlendshapeController] Initialized 52 blendshapes` (or similar count)
- Browser Console: `[UnityLoader] Unity loaded successfully`
- Browser Console: `[KellyWebGLBridge] SetExpression: happy` (when expressions change)

---

**SETUP COMPLETE!** 🎉

The Unity WebGL bridge is now ready. Follow the steps above to attach the script in Unity, rebuild, and test.










