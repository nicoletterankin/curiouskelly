# CC/iC Unity Pipeline Tools 2.1.0 - Complete Setup Guide

## Overview

This guide follows the official tutorial by Victor Soupday for setting up the CC/iC Unity Pipeline Tools.

**Video Tutorial:** https://www.youtube.com/watch?v=hyX8MG5ZIpk (8 minutes)

**Why This Matters:**
- Removes "Trial Version" watermark
- One-button export from iClone to Unity
- Auto-converts materials to URP
- Proper blendshape mapping
- Professional workflow

---

## Prerequisites

| Requirement | Status |
|-------------|--------|
| Character Creator 5 | ✅ Installed & Licensed |
| iClone 8 | ✅ Installed & Licensed |
| Unity 6 (6000.2.10f1) | ✅ Installed |
| GitHub Account | Needed for downloads |
| CC/iC Unity Tools License | ❌ **NEEDS PURCHASE ($199)** |

---

## Step 1: Download CC/iC Unity Pipeline Tools

**Video Timestamp:** 0:30 - 1:45

### GitHub Repositories:

1. **CC/iClone Plugin** (for Character Creator & iClone):
   - https://github.com/soupday/cc_unity_tools_HDRP
   - Or: https://github.com/soupday/cc_unity_tools_URP (for URP projects)

2. **Unity Package**:
   - Same repository, look in Releases section

### Download Steps:

1. Go to the GitHub repository
2. Click **"Releases"** on the right side
3. Download the latest release:
   - `CC_Unity_Tools_URP_x.x.x.unitypackage` (for URP projects like Kelly)
   - Plugin installer `.exe` (for CC5/iClone)
4. Save both files to your Downloads folder

---

## Step 2: Install CC/iClone Plugin

**Video Timestamp:** 1:45 - 2:14

### Steps:

1. **Close** Character Creator 5 and iClone 8 if they're open

2. **Run the plugin installer**:
   - Double-click the downloaded `.exe` file
   - Accept default installation path
   - Click Install
   - Wait for completion

3. **Restart** Character Creator 5 and iClone 8

4. **Verify installation**:
   - Open Character Creator 5
   - Check menu: **Plugins** → You should see "CC/iC Unity Tools"
   - Open iClone 8
   - Check menu: **Plugins** → You should see "CC/iC Unity Tools"

---

## Step 3: Activate Plugin License

**Video Timestamp:** 2:35

### ⚠️ THIS IS THE KEY STEP THAT REMOVES THE WATERMARK

### Steps:

1. **Open Character Creator 5**

2. **Go to**: Plugins → CC/iC Unity Tools → **License Manager**

3. **Enter your license key**:
   - If you purchased CC/iC Unity Tools, you received a license key via email
   - Paste the key into the License Manager
   - Click **Activate**

4. **Restart** Character Creator 5

5. **Verify activation**:
   - Plugins → CC/iC Unity Tools → License Manager
   - Should show: "License: Active" or similar

### If You Don't Have a License Key:

**Purchase from:** https://www.reallusion.com/auto-setup/unity/default.html
**Price:** ~$199 USD (one-time)

**What the license provides:**
- Removes "Trial Version" watermark from all exports
- Full commercial use rights
- Ongoing updates

---

## Step 4: Install Unity Package

**Video Timestamp:** 2:53 - 4:43

### Steps:

1. **Open Unity** with the Kelly project:
   ```
   digital-kelly/engines/Kelly_Engine_V2/onlykelly
   ```

2. **Import the package**:
   - Menu: **Assets → Import Package → Custom Package**
   - Browse to downloaded `CC_Unity_Tools_URP_x.x.x.unitypackage`
   - Click **Open**

3. **Import dialog appears**:
   - Click **Import All** (keep everything selected)
   - Wait for import to complete (may take 1-2 minutes)

4. **Verify installation**:
   - New menu should appear: **Reallusion** or **CC/iC Tools**
   - Check: Window → Reallusion → should show options

### Package Contents:

After import, you'll have:
```
Assets/
├── Reallusion/
│   ├── Editor/           ← Editor scripts
│   ├── Shaders/          ← URP-compatible shaders
│   ├── Scripts/          ← Runtime scripts
│   └── Prefabs/          ← Pre-configured prefabs
```

---

## Step 5: Connect Pipeline (iClone ↔ Unity)

**Video Timestamp:** 4:43 - 5:13

### In Unity:

1. Menu: **Reallusion → Link to iClone** (or similar)
2. Unity will show: "Listening for connection..."
3. Keep Unity open and running

### In iClone 8:

1. Open your Kelly project
2. Menu: **Plugins → CC/iC Unity Tools → Send to Unity**
3. A dialog appears with export options

### Export Options:

| Option | Recommended Setting |
|--------|---------------------|
| Character | ✅ Selected |
| Include Motion | ❌ Off (unless exporting animation) |
| Blendshapes | ✅ On |
| Textures | ✅ Embedded |
| Target | Unity URP |

4. Click **Send**

5. **In Unity**: Kelly should appear automatically!

---

## Step 6: One-Button Export Workflow

**Video Timestamp:** 5:13 - 7:09

### After Initial Setup, Future Exports Are Simple:

1. **Make changes** in iClone 8 (animations, expressions, etc.)

2. **Click**: Plugins → CC/iC Unity Tools → **Send to Unity**

3. **Done!** Kelly updates in Unity automatically

### What Gets Exported:

- ✅ Mesh with all LODs
- ✅ All 50+ facial blendshapes
- ✅ URP-compatible materials (auto-converted!)
- ✅ Textures (embedded or referenced)
- ✅ Animations (if selected)
- ✅ **NO WATERMARK** (if license is active)

---

## Step 7: Verify No Watermark

### Test Export:

1. In iClone 8: Send Kelly to Unity
2. In Unity: Add Kelly to scene
3. Enter Play mode
4. Check: No "Trial Version" text should appear

### If Watermark Still Appears:

1. License not activated properly
   - Check: Plugins → License Manager in CC5
   
2. Using old export (before license)
   - Delete old Kelly from Unity
   - Re-export from iClone

3. Unity package is trial version
   - Re-download from GitHub
   - Re-import package

---

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE-TIME SETUP                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Download CC/iC Unity Tools from GitHub                   │
│ 2. Install plugin in CC5 + iClone 8                         │
│ 3. Activate license (removes watermark)                     │
│ 4. Import Unity package into project                        │
│ 5. Connect pipeline (Unity listens, iClone sends)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   DAILY WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Edit Kelly in iClone 8                                   │
│ 2. Click "Send to Unity"                                    │
│ 3. Kelly appears in Unity (no watermark!)                   │
│ 4. Build WebGL                                              │
│ 5. Deploy to Netlify                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Timeline for Kelly Project

### Today (November 26):
- [x] Fix hair + camera in current build
- [x] Deploy to Netlify (with watermark - temporary)

### This Week:
- [ ] Purchase CC/iC Unity Tools license ($199)
- [ ] Follow this guide to set up pipeline
- [ ] Re-export Kelly without watermark
- [ ] Rebuild and deploy

### Before December 17:
- [ ] Add animations via pipeline
- [ ] Final QA testing
- [ ] LAUNCH!

---

## Resources

- **Video Tutorial:** https://www.youtube.com/watch?v=hyX8MG5ZIpk
- **GitHub (URP):** https://github.com/soupday/cc_unity_tools_URP
- **GitHub (HDRP):** https://github.com/soupday/cc_unity_tools_HDRP
- **Purchase License:** https://www.reallusion.com/auto-setup/unity/default.html
- **Reallusion Support:** https://www.reallusion.com/support/

---

*Last Updated: November 26, 2025*
*Based on CC/iC Unity Pipeline Tools 2.1.0*

