# Kelly Eyelid Cleanup Guide - Blender Workflow

**Purpose:** Step-by-step guide for removing eyelash shadow artifacts from Kelly's eyelids using Blender  
**Prerequisites:** Blender installed, Kelly FBX exported from CC5  
**Estimated Time:** 30-60 minutes  
**Difficulty:** Beginner-friendly

---

## Overview

This guide walks through cleaning up eyelash shadows left over from the headshot import. The workflow is:
1. Export Kelly FBX from CC5
2. Import into Blender
3. Identify and remove eyelash shadow artifacts
4. Verify clean eyelids match reference
5. Export cleaned mesh back to FBX
6. Re-import into CC5

**Why Blender?**
- Free and open source
- Easier to learn than ZBrush
- Excellent mesh editing tools
- FBX round-trip workflow works seamlessly

---

## Step 1: Export Kelly FBX from CC5

### 1.1 Open Kelly Project in CC5
1. Open Character Creator 5
2. Load your Kelly project file (`.ccProject`)
3. Ensure Kelly character is selected

### 1.2 Export Settings
1. Go to **File → Export → FBX**
2. Configure export settings:
   - **Format:** FBX 2014 or newer
   - **Include:** 
     - ✓ Mesh (geometry)
     - ✓ UV Maps
     - ✓ Materials (optional, we'll focus on geometry)
   - **Scale:** 1.0 (default)
   - **Coordinate System:** Z-up (default)
3. Click **Export**
4. Save as: `kelly_character_cc3_before_cleanup.fbx`
5. **Note the file location** for next step

**Expected Output:** FBX file with Kelly mesh geometry

---

## Step 2: Install Blender (If Not Already Installed)

### 2.1 Download Blender
1. Go to https://www.blender.org/download/
2. Download Blender 3.6 or newer (LTS recommended)
3. Install following standard installation wizard

### 2.2 Launch Blender
1. Open Blender
2. You'll see default cube scene
3. Delete default cube (Select → X → Delete)

---

## Step 3: Import Kelly FBX into Blender

### 3.1 Import FBX
1. Go to **File → Import → FBX (.fbx)**
2. Navigate to your exported FBX file
3. Select `kelly_character_cc3_before_cleanup.fbx`
4. Click **Import FBX**

### 3.2 Initial Setup
1. **Select the imported mesh:**
   - Click on Kelly mesh in viewport
   - Or select from outliner (top-right panel)
2. **View the model:**
   - Press **NumPad 7** (top view) or **NumPad 1** (front view)
   - Use mouse wheel to zoom
   - Middle mouse button to rotate view
   - Shift + middle mouse to pan

### 3.3 Enter Edit Mode
1. With mesh selected, press **Tab** key (enters Edit Mode)
2. You should see orange/white edges indicating edit mode
3. If not in Edit Mode, click **Edit Mode** dropdown at top (next to Object Mode)

---

## Step 4: Identify Eyelash Shadow Artifacts

### 4.1 Navigate to Eye Area
1. **Zoom in on face:**
   - Mouse wheel to zoom
   - Or use **View → Frame Selected** (Numpad .)
2. **Focus on eyelids:**
   - Rotate view to see upper and lower eyelids
   - Use **NumPad 1** (front), **NumPad 3** (side), **NumPad 7** (top)

### 4.2 Identify Problem Areas
Look for:
- **Dark lines or shadows** along eyelid edges
- **Discolored vertices** that don't match surrounding skin
- **Irregular geometry** that suggests artifacts
- **Vertices that seem out of place** compared to reference images

**Reference Check:**
- Open reference image: `projects/Kelly/Ref/headshot2-kelly-base169 101225.png`
- Compare eyelid area visually
- Note what clean eyelids should look like

### 4.3 Select Problem Vertices
1. **Selection Mode:**
   - Press **1** (Vertex Select mode) - or click vertex icon at bottom
   - You should see dots on vertices
2. **Select vertices:**
   - **Right-click** individual vertices to select
   - **Shift + Right-click** to add to selection
   - **B** key to box select (drag to select area)
   - **C** key to circle select (scroll to adjust size)

**Focus Areas:**
- Upper eyelid crease line
- Lower eyelid edge
- Area between eyelids and eyebrows
- Any visible dark lines or shadows

---

## Step 5: Clean Up Eyelash Shadows

### 5.1 Method 1: Smooth Vertices (Recommended First Step)

**Purpose:** Smooth out irregularities without changing overall shape

1. **Select affected vertices:**
   - Select vertices around eyelash shadow area
   - Don't select too many at once (work in small areas)

2. **Apply Smooth:**
   - Press **W** key (special menu)
   - Select **Smooth**
   - Or: **Mesh → Vertices → Smooth**
   - Repeat 2-3 times if needed

3. **Check Result:**
   - Rotate view to inspect
   - Compare to reference image
   - Undo if needed: **Ctrl + Z**

### 5.2 Method 2: Manual Vertex Adjustment

**Purpose:** Fine-tune individual vertices for precise control

1. **Select problematic vertices:**
   - Select vertices that form shadow lines

2. **Move vertices:**
   - Press **G** (Grab/Move)
   - Move mouse to position (constrained to view plane)
   - Or press **X**, **Y**, or **Z** to constrain to axis
   - **Left-click** to confirm, **Right-click** to cancel

3. **Align with surrounding geometry:**
   - Move vertices to match smooth skin surface
   - Avoid creating bumps or depressions
   - Use **Shift + S → Cursor to Selected** to check alignment

**Tips:**
- Work slowly, small adjustments
- Frequently compare to reference image
- Use **Tab** to exit Edit Mode and see results
- Rotate view to check from multiple angles

### 5.3 Method 3: Remove Doubled Vertices (If Applicable)

**Purpose:** Remove duplicate vertices that might cause artifacts

1. **Select all vertices:**
   - Press **A** (Select All)

2. **Remove doubles:**
   - Press **M** key
   - Select **By Distance** (or **Merge → By Distance**)
   - Or: **Mesh → Clean Up → Merge by Distance**
   - Adjust distance threshold if needed (default usually fine)

3. **Check result:**
   - Look for any improvement in problematic areas

### 5.4 Method 4: Sculpt Mode (Advanced - Optional)

**Purpose:** Use sculpting tools for smooth, natural cleanup

1. **Enter Sculpt Mode:**
   - Click dropdown at top: **Sculpt Mode** (instead of Edit Mode)
   - Or: **Tab** to exit Edit Mode, then select **Sculpt Mode**

2. **Select Smooth Brush:**
   - Toolbar on left: Select **Smooth** brush
   - Or press **Shift** while sculpting (temporary smooth)

3. **Smooth problematic areas:**
   - Left-click and drag over shadow areas
   - Adjust brush size: **F** key, drag mouse
   - Adjust strength: **Shift + F**, drag mouse

4. **Return to Edit Mode:**
   - Press **Tab** or select **Edit Mode**

**Note:** Sculpt mode works best if mesh has enough geometry. If mesh is low-poly, subdivide first (see Method 5).

### 5.5 Method 5: Subdivide (If Mesh Too Low-Poly)

**Purpose:** Add geometry for smoother editing

1. **Select affected area:**
   - Select faces/vertices in eyelid area

2. **Subdivide:**
   - Press **W** → **Subdivide**
   - Or: **Right-click → Subdivide**
   - Repeat 1-2 times if needed

3. **Then apply Smooth or Sculpt:**
   - Use Method 1 or 4 on subdivided mesh

**Warning:** Don't subdivide entire mesh unnecessarily (increases file size)

---

## Step 6: Verify Clean Eyelids

### 6.1 Visual Inspection
1. **Exit Edit Mode:**
   - Press **Tab** (Object Mode)

2. **Set up lighting:**
   - **Viewport Shading:** Click **Material Preview** or **Rendered** view (top-right icons)
   - Or add light: **Shift + A → Light → Sun**

3. **Compare to reference:**
   - Open reference image side-by-side
   - Check eyelid area matches reference
   - Look for:
     - ✓ No dark lines or shadows
     - ✓ Smooth skin texture
     - ✓ Clean transition from eyelid to surrounding skin
     - ✓ No artifacts or damage visible

### 6.2 Take Screenshots
1. **Set up view:**
   - Position camera to show eyelid area clearly
   - Match reference image angle if possible

2. **Render viewport:**
   - Press **F12** (Render Image)
   - Or: **View → Viewport Render Image**

3. **Save screenshot:**
   - **Image → Save As**
   - Save as: `kelly_eyelids_after_cleanup.png`

4. **Compare:**
   - Open reference image
   - Open your screenshot
   - Compare side-by-side

### 6.3 Check at 200% Zoom
1. **Zoom in viewport:**
   - Mouse wheel to zoom
   - Focus on eyelid area

2. **Inspect closely:**
   - Look for any remaining artifacts
   - Verify clean appearance

3. **If issues remain:**
   - Return to Step 5
   - Continue cleanup

---

## Step 7: Export Cleaned Mesh

### 7.1 Prepare for Export
1. **Ensure in Object Mode:**
   - Press **Tab** if in Edit Mode
   - Should show "Object Mode" at top

2. **Select Kelly mesh:**
   - Click on mesh in viewport
   - Or select from outliner

### 7.2 Export Settings
1. Go to **File → Export → FBX (.fbx)**
2. Configure settings:
   - **Path Mode:** Copy (if textures)
   - **Batch Mode:** Single Object
   - **Selected Objects:** ✓ Checked (only export selected)
   - **Scale:** 1.0
   - **Apply Transform:** ✓ Checked (recommended)
   - **Forward:** -Z Forward (FBX default)
   - **Up:** Y Up (FBX default)
3. **File name:** `kelly_character_cc3_cleaned.fbx`
4. Click **Export FBX**

**Expected Output:** Cleaned FBX file ready for CC5 import

---

## Step 8: Re-import into CC5

### 8.1 Import FBX
1. Open Character Creator 5
2. Open your Kelly project
3. Go to **File → Import → FBX**
4. Select `kelly_character_cc3_cleaned.fbx`
5. Click **Import**

### 8.2 Verify Import
1. **Check mesh:**
   - Inspect eyelid area in CC5 viewport
   - Verify clean appearance

2. **Compare to reference:**
   - Open reference image
   - Compare visually
   - Check at 200% zoom if possible

3. **If issues remain:**
   - Return to Blender
   - Make additional adjustments
   - Re-export and re-import

### 8.3 Update Project
1. **Save project:**
   - **File → Save** (or **Save As** with new name)
   - Save as: `kelly_cleaned.ccProject`

2. **Document changes:**
   - Note cleanup completion date
   - Update quality assessment document

---

## Troubleshooting

### Issue: Can't see the mesh clearly
**Solution:**
- Enable **Wireframe** view: Press **Z** → **Wireframe**
- Or enable **X-Ray**: Click X-ray icon at top (shows through mesh)
- Adjust viewport shading (Material Preview/Rendered)

### Issue: Too many vertices selected
**Solution:**
- Deselect all: Press **A** (toggles selection)
- Select smaller area: Use **B** (box select) with smaller box
- Use **Shift + Right-click** to remove from selection

### Issue: Mesh looks distorted after cleanup
**Solution:**
- Undo: **Ctrl + Z**
- Work in smaller areas
- Use Smooth brush more gently
- Check reference image frequently

### Issue: FBX import fails in CC5
**Solution:**
- Check FBX version compatibility
- Try exporting with different settings
- Ensure mesh is in Object Mode before export
- Check for non-manifold geometry: **Mesh → Clean Up → Make Manifold**

### Issue: Cleanup didn't fix the issue
**Solution:**
- Verify it's a geometry issue (not texture issue)
- Check if artifacts are in texture map instead
- Try more aggressive smoothing (more iterations)
- Consider texture editing if geometry cleanup insufficient

---

## Quality Checklist

After cleanup, verify:
- [ ] No visible eyelash shadows
- [ ] No artifacts or damage visible
- [ ] Smooth skin texture around eyelids
- [ ] Matches reference images visually
- [ ] Clean appearance at 200% zoom
- [ ] No geometry distortion
- [ ] Eyelids look natural and realistic

---

## Next Steps

After successful cleanup:
1. Update `KELLY_CURRENT_STATE_ASSESSMENT.md` with new score
2. Create before/after comparison screenshots
3. Proceed to hair integration (see `KELLY_HAIR_SELECTION_GUIDE.md`)
4. Continue with texture upgrade if needed

---

## Reference Images

Use these for comparison:
- `projects/Kelly/Ref/headshot2-kelly-base169 101225.png` (primary reference)
- `projects/Kelly/Ref/kelly_directors_chair_8k_light (2).png` (quality reference)
- Any close-up reference images showing clean eyelids

---

## Additional Resources

- **Blender Tutorials:** https://www.blender.org/support/tutorials/
- **Blender Manual:** https://docs.blender.org/manual/en/latest/
- **Blender Keyboard Shortcuts:** Press **Space** in Blender, type "shortcuts"

---

**Document Status:** Active - Ready for use  
**Last Updated:** December 2024  
**Estimated Completion Time:** 30-60 minutes  
**Difficulty:** Beginner-friendly









