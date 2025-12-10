# 🎭 Kelly WebGL Quick Fix

## The Problem
Kelly appears **gray/clay colored** in the browser due to incompatible shaders.

## The Solution
One-click fix that takes ~5 minutes.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Open Unity Project
```
Open: digital-kelly/engines/Kelly_Engine_V2/onlykelly
Unity Version: 6000.2.10f1 (or compatible)
```

### Step 2: Run Quick Fix
```
Menu: Window > Kelly WebGL > ⚡ DO EVERYTHING (Recommended)
```

This automatically:
- ✅ Switches to Mobile URP (Forward Rendering)
- ✅ Converts all Reallusion shaders to URP/Lit
- ✅ Builds optimized WebGL

### Step 3: Copy Build to Web
After build completes, copy files:

**From:** `Builds/Kelly_Web_Build/Build/*`

**To:** `public/unity/kelly-live/Build/`

Files to copy:
- `Kelly_Web_Build.data` (or `.data.br` if compressed)
- `Kelly_Web_Build.framework.js` (or `.framework.js.br`)
- `Kelly_Web_Build.loader.js`
- `Kelly_Web_Build.wasm` (or `.wasm.br`)

---

## ✅ Verify It Works

1. Start local server:
   ```powershell
   cd public
   npx http-server -p 3000 -c-1 --cors
   ```

2. Open browser:
   ```
   http://localhost:3000/unity-test.html
   ```

3. Kelly should appear with **proper skin colors** (not gray!)

---

## 📋 Manual Steps (If Automated Fix Fails)

### 1. Fix Graphics Settings
```
Edit > Project Settings > Graphics
Set "Scriptable Render Pipeline Settings" to: Mobile_RPAsset
```

### 2. Fix Materials Manually
For each material in Kelly's model:
1. Select material
2. Change Shader dropdown from `Reallusion/...` to `Universal Render Pipeline/Lit`
3. Re-assign textures (Albedo, Normal, etc.)

### 3. Build WebGL
```
File > Build Settings
Platform: WebGL
Click: Build
Output: Builds/Kelly_Web_Build
```

---

## 🔧 Troubleshooting

### "Mobile_RPAsset not found"
- Check `Assets/Settings/Mobile_RPAsset.asset` exists
- If missing, create new URP Asset via:
  `Assets > Create > Rendering > URP Asset (with Forward Renderer)`

### Build Fails with Shader Errors
- Some Reallusion shaders may need manual conversion
- Check Console for specific shader errors
- Replace problematic shaders with `Universal Render Pipeline/Lit`

### Kelly Still Gray After Build
- Ensure Graphics Settings uses Mobile_RPAsset (not PC_RPAsset)
- Check Quality Settings: `Edit > Project Settings > Quality`
- Set WebGL quality level to use Mobile_RPAsset

---

## 📁 File Locations

| File | Location |
|------|----------|
| Unity Project | `digital-kelly/engines/Kelly_Engine_V2/onlykelly` |
| Quick Fix Script | `Assets/Editor/WebGLFixer/WebGLQuickFix.cs` |
| Mobile URP | `Assets/Settings/Mobile_RPAsset.asset` |
| Build Output | `Builds/Kelly_Web_Build/` |
| Web Location | `public/unity/kelly-live/Build/` |

---

## 🎯 Expected Result

Before fix: Kelly appears **gray/clay** colored
After fix: Kelly has **proper skin tones, hair color, and textures**

The fix changes the render pipeline from Deferred (not WebGL compatible) to Forward (WebGL compatible) and replaces complex Shader Graph shaders with standard URP/Lit shaders.




