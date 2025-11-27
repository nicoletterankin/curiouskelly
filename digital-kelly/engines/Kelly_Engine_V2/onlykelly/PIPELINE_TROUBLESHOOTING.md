# CC/iC Unity Pipeline - Troubleshooting Guide

## Quick Diagnostics

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "Trial Version" watermark | License not activated | Activate in CC5 Plugins menu |
| Unity doesn't receive export | Pipeline not connected | Run "Link to iClone" in Unity |
| Materials look wrong | URP conversion failed | Re-import Unity package |
| Blendshapes missing | Export settings wrong | Enable in iClone export dialog |
| Plugin menu missing | Plugin not installed | Re-run installer |

---

## Issue 1: "Trial Version" Watermark Still Appears

### Cause:
The CC/iC Unity Tools license is not activated.

### Solution:

**Step 1: Check License Status**
1. Open Character Creator 5
2. Go to: Plugins → CC/iC Unity Tools → License Manager
3. Check if license shows as "Active"

**Step 2: If Not Active, Activate License**
1. Enter your license key (from purchase email)
2. Click Activate
3. Restart CC5

**Step 3: If You Don't Have a License**
- Purchase from: https://www.reallusion.com/auto-setup/unity/default.html
- Price: ~$199 USD

**Step 4: Re-Export Kelly**
1. In iClone 8: Plugins → CC/iC Unity Tools → Send to Unity
2. In Unity: Delete old Kelly, use new export
3. Rebuild WebGL

---

## Issue 2: Unity Doesn't Receive iClone Export

### Cause:
Pipeline connection not established.

### Solution:

**Step 1: Start Unity Listener**
1. Open Unity project
2. Menu: Reallusion → Link to iClone
3. You should see "Listening..." or connection status

**Step 2: Send from iClone**
1. Open Kelly in iClone 8
2. Plugins → CC/iC Unity Tools → Send to Unity
3. Wait for transfer to complete

**Step 3: Check Firewall**
- Windows Firewall may block the connection
- Allow Unity and iClone through firewall
- Or temporarily disable firewall for test

**Step 4: Same Network**
- Both apps must be on same computer
- Or same local network if using remote setup

---

## Issue 3: Materials Look Wrong in Unity

### Symptoms:
- Pink/magenta materials (missing shaders)
- Black materials
- Wrong colors
- Missing textures

### Cause:
URP material conversion didn't work properly.

### Solution:

**Step 1: Verify URP is Active**
1. Edit → Project Settings → Graphics
2. Scriptable Render Pipeline Settings should show URP Asset
3. If empty, assign your URP Asset

**Step 2: Re-import Unity Package**
1. Assets → Import Package → Custom Package
2. Select CC_Unity_Tools_URP package
3. Import All (overwrite existing)

**Step 3: Re-run Material Conversion**
1. Select Kelly in Project panel
2. Reallusion → Convert Materials to URP
3. Or right-click → Reallusion → Convert Materials

**Step 4: Manual Material Fix**
If auto-conversion fails:
1. Select the broken material
2. Change shader to: Universal Render Pipeline/Lit
3. Re-assign textures manually

---

## Issue 4: Blendshapes Missing

### Symptoms:
- Kelly's face doesn't animate
- SkinnedMeshRenderer shows 0 blendshapes
- Lip sync doesn't work

### Cause:
Blendshapes not included in export.

### Solution:

**Step 1: Check Export Settings in iClone**
1. Plugins → CC/iC Unity Tools → Send to Unity
2. In export dialog, ensure "Blendshapes" is checked
3. Re-export

**Step 2: Check Import Settings in Unity**
1. Select Kelly FBX in Project panel
2. Inspector → Model tab
3. Ensure "Import BlendShapes" is checked
4. Click Apply

**Step 3: Verify in Scene**
1. Add Kelly to scene
2. Select the body mesh (CC_Base_Body)
3. Expand SkinnedMeshRenderer → BlendShapes
4. Should show 50+ blendshapes

---

## Issue 5: Plugin Menu Missing in CC5/iClone

### Symptoms:
- No "CC/iC Unity Tools" in Plugins menu
- Can't find License Manager
- Can't find Send to Unity

### Cause:
Plugin not installed correctly.

### Solution:

**Step 1: Verify Installation**
1. Close CC5 and iClone 8
2. Check installation folder:
   - Default: `C:\Program Files\Reallusion\Plugins\`
   - Look for CC Unity Tools folder

**Step 2: Re-install Plugin**
1. Download latest from GitHub releases
2. Run installer as Administrator
3. Complete installation
4. Restart CC5 and iClone 8

**Step 3: Check Plugin is Enabled**
1. In CC5: Edit → Preferences → Plugins
2. Ensure CC/iC Unity Tools is enabled
3. Restart if you made changes

---

## Issue 6: Export Takes Forever or Freezes

### Symptoms:
- Progress bar stuck
- iClone becomes unresponsive
- Export never completes

### Cause:
Large file size or system resources.

### Solution:

**Step 1: Reduce Export Size**
- Uncheck "Include Motion" if not needed
- Use lower texture resolution
- Export character only (not full scene)

**Step 2: Free System Resources**
- Close other applications
- Ensure 8GB+ RAM available
- Check disk space (need 2GB+ free)

**Step 3: Export to Local Drive**
- Don't export to network drive
- Use SSD if available
- Avoid OneDrive/Dropbox synced folders

---

## Issue 7: Kelly Appears in Wrong Position/Scale

### Symptoms:
- Kelly floating in air
- Kelly underground
- Kelly too big or too small

### Cause:
Transform not reset after import.

### Solution:

**Step 1: Reset Transform**
1. Select Kelly in Hierarchy
2. Inspector → Transform
3. Set Position: (0, 0, 0)
4. Set Rotation: (0, 0, 0)
5. Set Scale: (1, 1, 1)

**Step 2: Check Import Scale**
1. Select Kelly FBX in Project
2. Inspector → Model tab
3. Scale Factor should be 1
4. Click Apply if changed

---

## Issue 8: Hair Appears Transparent

### Cause:
Hair material using wrong Surface Type.

### Solution:

1. Find hair material in Project panel
2. Select it
3. In Inspector:
   - Surface Type: Opaque (not Transparent)
   - Alpha Clipping: Enabled
   - Threshold: 0.5
4. Save

---

## Issue 9: Animations Don't Play

### Symptoms:
- Kelly stuck in T-pose
- Animator shows no animation
- Play mode shows no movement

### Cause:
No Animator Controller assigned or no default animation.

### Solution:

**Step 1: Check Animator Component**
1. Select Kelly in Hierarchy
2. Look for Animator component
3. Ensure Controller field is assigned

**Step 2: Create Animator Controller**
1. Right-click in Project → Create → Animator Controller
2. Name it "KellyAnimator"
3. Assign to Kelly's Animator component

**Step 3: Add Animation**
1. Double-click Animator Controller to open
2. Drag animation clip into Animator window
3. Right-click animation → Set as Layer Default State

---

## Issue 10: WebGL Build Fails

### Symptoms:
- Build errors in Console
- Build never completes
- Missing files in output

### Cause:
Various - check error messages.

### Common Solutions:

**Shader Errors:**
- Ensure URP package is installed
- Re-import CC/iC Unity package

**Scene Not Found:**
- Add KellyMain.unity to Build Settings
- File → Build Settings → Add Open Scenes

**Out of Memory:**
- Close other applications
- Reduce texture sizes
- Enable texture compression

---

## Getting Help

### Reallusion Support:
- https://www.reallusion.com/support/
- support@reallusion.com

### Community Forums:
- https://forum.reallusion.com/

### GitHub Issues:
- https://github.com/soupday/cc_unity_tools_URP/issues

### Video Tutorials:
- Victor Soupday's channel
- Reallusion official tutorials

---

*Last Updated: November 26, 2025*

