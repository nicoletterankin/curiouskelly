# Kelly Blender Visual Upgrade Guide - Click-by-Click Instructions

**Current State:** You're in Blender 4.5.3 LTS, Shading workspace, Edit Mode, with Kelly head model visible  
**Goal:** Upgrade skin, hair, eyebrows, and lip textures/materials  
**Estimated Time:** 45-60 minutes  
**Difficulty:** Beginner-friendly (follow exact clicks)

---

## 🎯 Quick Overview

You'll be upgrading:
1. **Skin Material** - Enhance base color, subsurface scattering, roughness
2. **Hair Material** - Make hair visible with proper color (medium brown + caramel highlights)
3. **Eyebrow Enhancement** - Adjust color and texture to match dark brown specification
4. **Lip Color** - Set natural rosy-pink color per Kelly specification

---

## 🖼️ STEP 0: Load Reference Image into Blender (CRITICAL - Use Your Reference Photo!)

**This step lets you see your reference photo (`kelly41303.png`) side-by-side with the Blender model for accurate color matching!**

### Step 0.1: Exit Edit Mode First
1. **Press `Tab` key** (exits Edit Mode → enters Object Mode)
2. **Verify:** Top of viewport says "Object Mode"

### Step 0.2: Add Reference Image as Background
1. **Press `NumPad 1`** (front orthographic view - aligns with reference photo)
2. **Press `N`** (opens right-side panel)
3. **In the right panel:** Find **"View"** tab (should be at top)
4. **Scroll down** to **"Background Images"** section
5. **Click the small arrow** next to "Background Images" to expand it
6. **Click "Add Image"** button
7. **Click "Open"** button (or browse icon)
8. **Navigate to your reference image:** `kelly41303.png`
9. **Select the file** → **Click "Open"**

### Step 0.3: Position Reference Image
1. **In Background Images section:** You should see the image listed
2. **Set "Display As":** **"Background"** (should be default)
3. **Set "Size":** Adjust slider or type value until image matches your model size
   - Start with `1.0` and adjust up/down until reference head matches Blender head size
4. **Set "Opacity":** `1.0` (fully visible)
5. **Set "Offset X":** `0` (centered horizontally)
6. **Set "Offset Y":** `0` (centered vertically)
7. **Set "Depth":** **"Front"** (shows in front of model)

### Step 0.4: Alternative - Reference Image as Side Panel (Even Better!)
**This method lets you see reference while working:**
1. **At bottom of Blender:** Click **"+"** icon (adds new editor)
2. **Select "Image Editor"** from dropdown
3. **In Image Editor:** Click **"Open"** button (folder icon)
4. **Navigate to:** `kelly41303.png`
5. **Select and open** the reference image
6. **You now have:** Reference image in bottom panel, Blender viewport above it
7. **To resize panels:** Drag the divider between panels up/down

### Step 0.5: Use Reference for Color Picking
**When setting colors in materials, you can sample directly from reference:**
1. **In Shader Editor:** When setting Base Color, click the color swatch
2. **In color picker:** You'll see an eyedropper icon
3. **Click eyedropper** → **Click on reference image** (picks color from reference!)
4. **Or manually match:** Look at reference → Adjust RGB values to match

**Reference Image Color Targets (from your `kelly41303.png`):**
- **Skin Tone:** Warm light-medium (sample from cheek/forehead area)
- **Eyebrows:** Dark brown (sample from eyebrow hairs)
- **Lips:** Natural rosy-pink (sample from lip area)
- **Skin Texture:** Notice the subtle subsurface scattering glow

---

## 📋 PREPARATION: Exit Edit Mode First

**Important:** You're currently in Edit Mode. We need Object Mode for material work.

### Step 1: Switch to Object Mode
1. **Press `Tab` key** (this exits Edit Mode and enters Object Mode)
2. **Verify:** Top of viewport should say "Object Mode" (not "Edit Mode")

### Step 2: Select Kelly Head Mesh
1. **Click on the Kelly head mesh** in the viewport (the head model)
2. **Verify:** The head should have orange outline indicating selection
3. **If unsure:** Look in top-right Outliner panel → find the main head object → **click on it**

---

## 🎨 PART 1: Upgrade Skin Material (15 minutes)

### Step 1.1: Open Material Properties
1. **Click the "Material Properties" icon** in the right panel (looks like a sphere with checker pattern)
   - Located in the right-side Properties panel
   - If not visible: **Press `N`** to toggle side panel, or look for tabs at top of right panel

### Step 1.2: Check Current Material
1. **Look for "Material" dropdown** at top of Material Properties panel
2. **Click the dropdown** to see current material name
3. **Note the name** (likely "Std_Skin_Head" or similar)

### Step 1.3: Enable Material Preview Mode (See Changes Live)
1. **Look at top-right of 3D viewport** (above the viewport)
2. **Find viewport shading icons** (4 colored spheres: Wireframe, Solid, Material Preview, Rendered)
3. **Click "Material Preview" icon** (third icon - shows materials with lighting)
   - Or press **`Z`** key → select **"Material Preview"** from menu

### Step 1.4: Access Shader Editor
1. **Look at bottom of Blender window** - find tabs
2. **Click tab labeled "Shader Editor"** (shows node-based material editor)
   - If not visible: **Go to top menu → Window → Shading → Shader Editor**
   - Or: **Click workspace icon at top → switch to "Shading" workspace** (you're already here!)

### Step 1.5: Add Skin Texture Maps
You should see nodes in the Shader Editor. We'll connect texture maps:

**A. Add Base Color Texture:**
1. **Press `Shift + A`** (opens Add menu)
2. **Navigate:** Texture → Image Texture
3. **Click "Image Texture"** (adds new node)
4. **Click the new Image Texture node**
5. **Click "Open" button** in the node (or click folder icon)
6. **Navigate to:** `digital-kelly/engines/kelly_unity_player/My project/Assets/Kelly/Models/textures/kelly_character_cc3/kelly_character_cc3/CC_Base_Body/Std_Skin_Head/`
7. **Select:** `Std_Skin_Head_BCBMap_v1.png` (or `Std_Skin_Head_BCBMap.png` if v1 doesn't exist)
8. **Click "Open"**

**B. Connect Base Color:**
1. **Look for "Principled BSDF" node** (should be visible - main shader node)
2. **From Image Texture node:** Click and drag from **"Color" output** (right side of Image Texture node)
3. **Drag to:** **"Base Color" input** on Principled BSDF node (left side)
4. **Release mouse** - connection line should appear

**C. Add Normal Map:**
1. **Press `Shift + A`** → **Texture → Image Texture** (adds second Image Texture)
2. **Click "Open"** → Navigate to same folder
3. **Select:** `Std_Skin_Head_NBMap.png`
4. **Press `Shift + A`** → **Vector → Normal Map** (adds Normal Map node)
5. **Connect:**
   - Image Texture Color → Normal Map Color input
   - Normal Map Normal output → Principled BSDF Normal input

**D. Add Subsurface Scattering (SSS):**
1. **Press `Shift + A`** → **Texture → Image Texture** (third Image Texture)
2. **Click "Open"** → Select: `Std_Skin_Head_SSSMap.png`
3. **Connect:** Image Texture Color → Principled BSDF Subsurface Color input
4. **On Principled BSDF node:** Find "Subsurface" slider → **Set to `0.28`** (drag slider or type number)

**E. Add Roughness:**
1. **Press `Shift + A`** → **Texture → Image Texture** (fourth Image Texture)
2. **Click "Open"** → Select: `Std_Skin_Head_SpecMask.png` (or create roughness from this)
3. **Connect:** Image Texture Color → Principled BSDF Roughness input
4. **On Principled BSDF:** Find "Roughness" → **Set base value to `0.40`** (adjusts skin shine)

### Step 1.6: Adjust Skin Color Tone (Match Reference!)
**Use your reference image (`kelly41303.png`) to match skin tone exactly:**
1. **Click Principled BSDF node**
2. **Find "Base Color" input** (should show connected texture)
3. **Add Color Mix node for fine-tuning:**
   - **Press `Shift + A`** → **Color → MixRGB**
   - **Connect:** Image Texture Color → MixRGB Color1
   - **To match reference:** Click MixRGB Color2 swatch → **Use eyedropper tool** → **Click on reference image cheek/forehead** (picks exact skin tone!)
   - **Or manually:** Set Color2 to warm light-medium: **RGB (0.92, 0.82, 0.75)**
   - **Set Fac** to `0.1` (10% mix - subtle adjustment)
   - **Connect:** MixRGB Color → Principled BSDF Base Color
4. **Compare with reference:** Look at your reference image → Adjust Fac slider until skin matches reference tone

### Step 1.7: Verify Skin Appearance
1. **Look at 3D viewport** - skin should look more realistic
2. **Compare with reference:** Look at your reference image (`kelly41303.png`) → Notice the subtle glow on forehead/nose? That's subsurface scattering!
3. **Adjust Subsurface:** If skin doesn't have that soft glow like reference, increase Subsurface slider to `0.3` or `0.35`
4. **Rotate view:** **Middle mouse button drag** to rotate
5. **If too dark/bright:** Adjust Principled BSDF → **"Specular"** slider (try `0.5`)
6. **Match reference lighting:** Your reference has soft, even lighting - adjust until skin matches that quality

---

## 💇 PART 2: Make Hair Visible & Upgrade Material (20 minutes)

### Step 2.1: Find Hair Object
1. **Look at top-right Outliner panel**
2. **Find "Kelly_Hair_Base" objects** (you saw Kelly_Hair_Base.001 and .003 selected)
3. **Click on "Kelly_Hair_Base"** (or any Kelly_Hair_Base object) in Outliner

### Step 2.2: Switch to Particle System View
1. **With hair object selected:** Click **"Particle Properties" icon** in right panel (looks like sparkles/particles)
2. **Look for "Particle System" dropdown** → should show "Kelly_Hair_Settings" or similar
3. **If particle system exists:** Great! Continue to Step 2.3
4. **If no particle system:** Skip to Step 2.8 (setup hair system)

### Step 2.3: Make Hair Visible in Viewport
1. **In Particle Properties panel:** Find "Viewport Display" section
2. **Set "Display As"** dropdown to **"Rendered"** (shows actual hair strands)
3. **Set "Count"** to `1000` or higher (for preview - final can be 50,000)
4. **Check "Emitter" checkbox** (shows hair base mesh)

### Step 2.4: Access Hair Material
1. **With hair object selected:** Click **"Material Properties" icon** (sphere icon)
2. **Check if material exists:**
   - If "Material" dropdown shows a material → Click dropdown → Note name
   - If dropdown shows "New" → No material exists → Continue to Step 2.5

### Step 2.5: Create Hair Material
1. **Click "+ New" button** in Material Properties (creates new material)
2. **Rename material:** Click name field → Type: **"Kelly_Hair_Material"** → Press Enter

### Step 2.6: Setup Hair Shader Nodes
1. **Go to Shader Editor** (bottom panel or Shading workspace)
2. **Make sure hair object is selected** (check Outliner)
3. **You should see hair material nodes** (if not, select hair object again)

**A. Switch to Hair Shader:**
1. **Find "Principled BSDF" node** (default)
2. **Press `Shift + A`** → **Shader → Hair BSDF** (or **Shader → Principled Hair BSDF**)
3. **Delete Principled BSDF:** Click it → **Press `X`** → **Press `Enter`** (confirm delete)
4. **Connect:** Principled Hair BSDF → Material Output Surface

**B. Set Hair Base Color:**
1. **Click Principled Hair BSDF node**
2. **Find "Color" input**
3. **Set color to medium brown:** Click color swatch → **RGB (93, 64, 55)** or **Hex #5D4037**
   - Or use color picker: R=0.365, G=0.251, B=0.216

**C. Set Hair Highlights:**
1. **On Principled Hair BSDF:** Find **"Melanin"** slider → **Set to `0.6`** (brown hair)
2. **Find "Melanin Redness"** → **Set to `0.2`** (warm undertones)
3. **Find "Roughness"** → **Set to `0.15`** (smooth, polished texture)

**D. Add Highlight Color Mix:**
1. **Press `Shift + A`** → **Color → MixRGB**
2. **Connect:** MixRGB Color → Principled Hair BSDF Color input
3. **Set MixRGB Color1** to base brown: **RGB (93, 64, 55)**
4. **Set MixRGB Color2** to caramel highlight: **RGB (212, 165, 116)** or **Hex #D4A574**
5. **Set Fac** to `0.15` (15% highlights - subtle)

### Step 2.7: Verify Hair Visibility
1. **In Particle Properties:** Make sure "Display As" is set to **"Rendered"**
2. **Switch viewport to Rendered:** **Press `Z`** → **"Rendered"** (final icon)
3. **Hair should appear** on head model
4. **If not visible:** Continue to Step 2.8

### Step 2.8: Setup Hair Particle System (If Missing)
**Note:** You have Python scripts for this. Quick manual setup:

1. **Select Kelly head mesh** (not hair base)
2. **Go to Particle Properties** → **Click "+ New"** button
3. **Set "Type"** to **"Hair"**
4. **Set "Count"** to `50000` (high density)
5. **Set "Length"** to `0.45` (shoulder length)
6. **Go to "Hair Shape" section:**
   - **Set "Diameter Root"** to `0.002`
   - **Set "Diameter Tip"** to `0.001`
7. **In "Render" section:**
   - **Set "Render As"** to **"Path"**
   - **Set "Steps"** to `6` (smooth curves)

**For advanced setup:** Use the Python script `kelly_hair_system.py`:
1. **Switch to Scripting workspace** (top tabs)
2. **Open:** `kelly_hair_system.py` from texture folder
3. **Press `Run Script`** button (play icon)

---

## 👁️ PART 3: Enhance Eyebrow Color & Texture (10 minutes)

### Step 3.1: Select Eyebrow Geometry
1. **Look at Outliner** → Find eyebrow objects (might be part of head mesh)
2. **If eyebrows are separate:** Click eyebrow object in Outliner
3. **If eyebrows are part of head:** Select head mesh → Continue to Step 3.2

### Step 3.2: Create Eyebrow Material Slot
1. **Select head mesh** (if eyebrows are part of head)
2. **Go to Material Properties**
3. **Scroll down** → Find **"Material Slots"** section
4. **Click "+" button** (adds new material slot)
5. **Click "New"** button next to new slot → Creates material
6. **Rename:** Click name → Type **"Kelly_Eyebrows"** → Enter

### Step 3.3: Assign Eyebrow Material to Face Selection
**If eyebrows are separate geometry:**
1. **Select eyebrow object** → Material Properties → Assign "Kelly_Eyebrows" material

**If eyebrows are part of head mesh:**
1. **Select head mesh** → **Press `Tab`** (Enter Edit Mode)
2. **Press `3`** (Face Select mode)
3. **Select eyebrow faces:** Use **Right-click** on eyebrow faces, or **B** (box select) around eyebrows
4. **Go to Material Properties**
5. **Select "Kelly_Eyebrows" material slot**
6. **Click "Assign" button** (assigns material to selected faces)
7. **Press `Tab`** (Exit Edit Mode)

### Step 3.4: Setup Eyebrow Material (Match Reference!)
**Use your reference image to match eyebrow color exactly:**
1. **Make sure "Kelly_Eyebrows" material is selected** in Material Properties
2. **Go to Shader Editor**
3. **Find Principled BSDF node**

**A. Set Eyebrow Color:**
1. **Click Principled BSDF node**
2. **Click Base Color swatch** → **Click eyedropper icon** → **Click on eyebrow area in reference image** (picks exact dark brown!)
3. **Or manually set:** Dark brown **RGB (45, 30, 25)** or **Hex #2D1E19**
   - Use color picker: R=0.176, G=0.118, B=0.098
4. **Compare:** Look at reference eyebrows → Adjust until color matches

**B. Set Eyebrow Roughness:**
1. **Set Roughness** to `0.7` (hair-like texture, not shiny)

**C. Add Hair-like Shader (Optional - Better Results):**
1. **Press `Shift + A`** → **Shader → Principled Hair BSDF**
2. **Connect:** Principled Hair BSDF → Material Output Surface
3. **Delete old Principled BSDF:** Click → **X** → **Enter**
4. **Set Color:** Dark brown **RGB (45, 30, 25)**
5. **Set Melanin:** `0.8` (very dark brown)
6. **Set Roughness:** `0.7`

### Step 3.5: Verify Eyebrows
1. **Look at viewport** → Eyebrows should appear dark brown
2. **Rotate view:** Middle mouse drag to check from different angles

---

## 💋 PART 4: Upgrade Lip Color & Texture (10 minutes)

### Step 4.1: Select Lip Area
1. **Select head mesh**
2. **Press `Tab`** (Enter Edit Mode)
3. **Press `3`** (Face Select mode)
4. **Select lip faces:**
   - **Zoom in:** Mouse wheel
   - **Rotate to front view:** **NumPad 1** (front orthographic)
   - **Right-click** on lip faces, or **B** (box select) around lips
   - **Shift + Right-click** to add more faces if needed

### Step 4.2: Create Lip Material Slot
1. **Go to Material Properties**
2. **Scroll to "Material Slots"** → **Click "+"** (add slot)
3. **Click "New"** → Rename to **"Kelly_Lips"**

### Step 4.3: Assign Lip Material
1. **With lip faces selected** (still in Edit Mode)
2. **In Material Properties:** Select "Kelly_Lips" material slot
3. **Click "Assign" button**
4. **Press `Tab`** (Exit Edit Mode)

### Step 4.4: Setup Lip Material (Match Reference!)
**Use your reference image to match lip color exactly:**
1. **Select "Kelly_Lips" material** in Material Properties
2. **Go to Shader Editor**

**A. Set Lip Base Color:**
1. **Click Principled BSDF node**
2. **Click Base Color swatch** → **Click eyedropper icon** → **Click on lip area in reference image** (picks exact rosy-pink!)
3. **Or manually set:** Natural rosy-pink **RGB (200, 120, 130)** or **Hex #C87882**
   - Use color picker: R=0.784, G=0.471, B=0.510
4. **Compare:** Look at reference lips → Adjust until color matches reference perfectly

**B. Add Subsurface Scattering (Makes Lips Realistic):**
1. **Set Subsurface** slider to `0.3`
2. **Set Subsurface Radius** to **RGB (1.0, 0.5, 0.3)** (reddish SSS)
3. **Set Subsurface Color** to slightly redder: **RGB (220, 140, 150)**
4. **Compare with reference:** Look at reference image - notice the subtle glow on lips? Adjust Subsurface slider until your lips have that same soft, realistic glow

**C. Adjust Roughness:**
1. **Set Roughness** to `0.2` (lips are slightly glossy/shiny)

**D. Add Subtle Specular:**
1. **Set Specular** to `0.6` (gives lips natural shine)

### Step 4.5: Fine-Tune Lip Color
1. **Look at viewport** → Lips should appear natural rosy-pink
2. **If too bright:** Adjust Base Color → Reduce R value slightly
3. **If too dark:** Increase R and G values
4. **If needs more "lip" look:** Increase Subsurface to `0.4`

---

## ✅ FINAL VERIFICATION STEPS

### Step 5.1: Switch to Rendered View
1. **Press `Z`** key → Select **"Rendered"** (final shading mode)
2. **Wait for render** (may take a few seconds)
3. **All materials should now be visible**

### Step 5.2: Check All Elements
- ✅ **Skin:** Warm light-medium tone, realistic texture
- ✅ **Hair:** Medium brown with caramel highlights, visible on head
- ✅ **Eyebrows:** Dark brown, well-defined
- ✅ **Lips:** Natural rosy-pink color

### Step 5.3: Take Screenshot
1. **Position view:** Rotate to front view (**NumPad 1**)
2. **Press `F12`** (Render Image) OR **View → Viewport Render Image**
3. **Save:** **Image → Save As** → Save as `kelly_upgraded_preview.png`

### Step 5.4: Save Blender File
1. **Press `Ctrl + S`** (Save)
2. **Or:** **File → Save As** → Save as `kelly_character_upgraded.blend`

---

## 🎯 Quick Reference: Hotkeys Used

- **`Tab`** - Toggle Edit Mode / Object Mode
- **`Shift + A`** - Add menu (for nodes, objects, etc.)
- **`Z`** - Viewport shading menu
- **`N`** - Toggle side panel
- **`X`** - Delete selected
- **`B`** - Box select
- **`C`** - Circle select
- **`3`** - Face select mode
- **`NumPad 1`** - Front view
- **`NumPad 7`** - Top view
- **`F12`** - Render image
- **`Ctrl + S`** - Save file
- **`Ctrl + Z`** - Undo

---

## 🐛 Troubleshooting

### Issue: Can't see changes in viewport
**Solution:**
- Make sure viewport is set to **"Material Preview"** or **"Rendered"** (not Wireframe/Solid)
- Press `Z` → Select "Material Preview"

### Issue: Hair not visible
**Solution:**
- Check Particle Properties → "Display As" → Set to "Rendered"
- Check hair object has material assigned
- Switch viewport to "Rendered" mode (`Z` → "Rendered")

### Issue: Can't find texture files
**Solution:**
- Navigate to: `digital-kelly/engines/kelly_unity_player/My project/Assets/Kelly/Models/textures/kelly_character_cc3/kelly_character_cc3/CC_Base_Body/Std_Skin_Head/`
- Use `Std_Skin_Head_BCBMap.png` if `_v1.png` doesn't exist

### Issue: Material not applying to selected faces
**Solution:**
- Make sure you're in Edit Mode (`Tab`)
- Make sure faces are selected (orange highlight)
- Make sure correct material slot is selected
- Click "Assign" button after selecting material slot

### Issue: Nodes are confusing
**Solution:**
- Focus on connecting: Image Texture → Principled BSDF inputs
- Base Color → Base Color input
- Normal Map → Normal input
- SSS Map → Subsurface Color input

---

## 📝 Next Steps After Upgrades

1. **Test in Unity:** Export FBX and test in Unity player
2. **Fine-tune colors:** Adjust RGB values based on reference images
3. **Hair density:** Increase particle count to 50,000 for final render
4. **Add eyelashes:** Similar process to eyebrows (dark brown material)

---

**Document Status:** Ready for use  
**Last Updated:** December 2024  
**Difficulty:** Beginner-friendly with exact click instructions

