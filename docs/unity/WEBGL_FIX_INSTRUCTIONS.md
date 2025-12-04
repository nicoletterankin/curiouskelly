# Fix Kelly Gray/White Appearance in WebGL

## Problem
Kelly appears gray/clay-colored in WebGL because:
1. The build uses Deferred Rendering (not supported in WebGL 2.0)
2. Reallusion's custom shaders don't compile for WebGL
3. GraphicsSettings.asset uses PC_RPAsset instead of Mobile_RPAsset

## Solution - Step by Step

### Step 1: Open Unity Project
Open `digital-kelly/engines/Kelly_Engine_V2/onlykelly` in Unity Editor (2023.x or later)

### Step 2: Fix Graphics Settings
1. Go to **Edit > Project Settings > Graphics**
2. Under "Scriptable Render Pipeline Settings", change the Render Pipeline Asset from `PC_RPAsset` to `Mobile_RPAsset`
3. This ensures Forward Rendering is used (WebGL compatible)

### Step 3: Verify Quality Settings
1. Go to **Edit > Project Settings > Quality**
2. Ensure "WebGL" platform uses the "Mobile" quality level (index 0)
3. Verify Mobile quality uses `Mobile_RPAsset` (Forward rendering)

### Step 4: Fix Reallusion Shaders (if needed)
If Kelly is still gray after steps 1-3:

**Option A: Use Standard URP Lit Shader**
1. Select Kelly's materials in the Project window
2. For each material, change the shader from Reallusion's custom shader to `Universal Render Pipeline/Lit`
3. Re-assign textures if needed

**Option B: Create WebGL-specific Material Variants**
1. Duplicate Kelly's materials
2. Name them with `_WebGL` suffix
3. Use standard URP shaders
4. Create a script to swap materials at runtime for WebGL

### Step 5: Build WebGL
1. Go to **File > Build Settings**
2. Select **WebGL** platform
3. Click **Switch Platform** (if not already on WebGL)
4. Set Compression to **Brotli**
5. Enable **Development Build** for debugging (disable for production)
6. Click **Build**

### Step 6: Deploy
1. Upload the new build files to `public/unity/kelly-live/Build/`
2. OR upload to Cloudflare R2 and update worker CDN

## Quick Reference - File Locations

| Setting | File | Change |
|---------|------|--------|
| Default Render Pipeline | `ProjectSettings/GraphicsSettings.asset` | Line 40: Change GUID to `5e6cbd92db86f4b18aec3ed561671858` (Mobile_RPAsset) |
| WebGL Quality | `ProjectSettings/QualitySettings.asset` | Already correct (WebGL: 0 = Mobile) |
| Forward Rendering | `Assets/Settings/Mobile_Renderer.asset` | Already correct (`m_RenderingMode: 0`) |

## Shader Compatibility Notes

### WebGL 2.0 Limitations
- No Compute Shaders
- No Deferred Rendering
- Limited texture units
- No tessellation
- Limited render texture formats

### Reallusion Shader Issues
The following Reallusion shaders may have issues in WebGL:
- `RL_HairShader_URP` - Complex alpha and transmission
- `RL_SkinShader_URP` - SSS (Subsurface Scattering) not supported
- `RL_EyeShader_URP` - Refraction effects may fail

### Recommended Fallback Shaders
- Skin → `Universal Render Pipeline/Lit` with smoothness ~0.4
- Hair → `Universal Render Pipeline/Lit` (Alpha Blend mode)
- Eyes → `Universal Render Pipeline/Lit` with smoothness ~0.9

## Verification

After rebuilding, check the browser console for:
- ✅ No `shader is not supported on this GPU` errors
- ✅ No `WebGL: INVALID_ENUM` errors during startup
- ✅ Kelly should have proper skin tones and textures

## "Trial Version" Watermark

The watermark indicates Reallusion Character Creator tools are in trial mode.
To remove it:
1. Purchase a license for CC/iC Unity Tools
2. Or export the character without the trial watermark from Character Creator

---
*Created: December 3, 2025*
*Issue: Kelly 3D Gray/White Appearance*

