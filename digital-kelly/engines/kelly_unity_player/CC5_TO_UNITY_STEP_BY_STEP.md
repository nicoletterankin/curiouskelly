# Complete Step-by-Step Guide: CC5 → Unity (Click-by-Click)

**You don't open CC5 in Unity. Instead:**
1. **Export** Kelly from CC5 as an FBX file
2. **Import** that FBX file into Unity

---

## 📤 **PART 1: Export Kelly from Character Creator 5**

### **Step 1: Open Character Creator 5**

1. **On your Windows desktop**, look for the **Character Creator 5** icon
2. **Double-click** to open it
3. **Wait** for CC5 to fully load (may take 30-60 seconds)

### **Step 2: Load Your Kelly Character**

1. **In CC5**, look at the top menu bar
2. **Click** on **"File"**
3. **Click** on **"Open Project"** (or **"Open"**)
4. **Navigate** to where your Kelly project is saved
5. **Click** on your Kelly project file (usually ends in `.ccProject`)
6. **Click** "Open"
7. **Wait** for Kelly to load in the 3D viewport

### **Step 3: Select Kelly**

1. **In the 3D viewport** (the big window showing Kelly), **LEFT-CLICK** directly on Kelly's body or head
2. **Verify** Kelly is selected (you should see selection indicators)

### **Step 4: Export as FBX**

1. **At the top menu bar**, click **"File"**
2. **Hover** over **"Export"**
3. **Click** on **"Export Character as FBX"** (or **"Export FBX"**)
4. **A dialog box** will appear titled "Export FBX"

### **Step 5: Configure Export Settings**

**In the "Export FBX" dialog box:**

1. **Target Tool Preset:**
   - **Click** the dropdown menu next to "Target Tool Preset"
   - **Scroll** and click on **"Unity 3D"**

2. **Embed Textures:**
   - **Look for** a checkbox labeled **"Embed Textures"**
   - **Click** the checkbox to **CHECK** it (should have a checkmark ✓)

3. **Convert Skinned Expressions to Morphs:**
   - **Look for** a checkbox labeled **"Convert Skinned Expressions to Morphs"**
   - **If it's grayed out**, try checking "HD Character" first
   - **Click** the checkbox to **CHECK** it (✓) - **THIS IS CRITICAL FOR BLENDSHAPES**

4. **Max Texture Size:**
   - **Click** the dropdown next to "Max Texture Size"
   - **Select** "2048" (or "4096" if you want higher quality)

5. **Convert Image Format to:**
   - **Click** the dropdown next to "Convert Image Format to"
   - **Select** "PNG" (best quality)

6. **Delete Unused Morphs:**
   - **Make sure** this checkbox is **UNCHECKED** (no checkmark)

7. **Use Smooth Mesh:**
   - **Click** to **CHECK** this box (if available)

### **Step 6: Save the FBX File**

1. **Click** the **"Export"** button at the bottom of the dialog
2. **A "Save As" window** will appear
3. **Navigate** to: `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\My project\Assets\Kelly\Models\`
   - If the `Models` folder doesn't exist, create it:
     - Click "New folder" in the Save As window
     - Type "Models" and press Enter
     - Double-click into the Models folder
4. **File name:** Type `Kelly_Character.fbx`
5. **Click** "Save"
6. **Wait** for export to complete (may take 1-2 minutes)

---

## 📥 **PART 2: Import FBX into Unity**

### **Step 1: Open Unity**

1. **If Unity is not open:**
   - Look for **Unity Hub** icon on your desktop
   - **Double-click** Unity Hub
   - **Click** "Projects" tab (left sidebar)
   - **Click** on "My project" (or your project name)
   - **Wait** for Unity to open (30-60 seconds)

### **Step 2: Navigate to Models Folder in Unity**

1. **At the bottom** of Unity Editor, find the **"Project"** window (looks like a file browser)
2. **Click** on **"Assets"** folder (left side of Project window)
3. **Click** on **"Kelly"** folder (inside Assets)
4. **Click** on **"Models"** folder (inside Kelly)
5. **You should see** `iclone_unity_kelly_export1.fbx` and/or `Kelly_Character.fbx`

### **Step 3: Import the FBX File**

1. **If you just exported from CC5:**
   - Unity should **automatically detect** the new file
   - If not, **right-click** in the Models folder → **"Refresh"** or press **F5**

2. **Click** on `Kelly_Character.fbx` (or your exported FBX) to select it

### **Step 4: Configure FBX Import Settings**

1. **Look at the Inspector** (right side of Unity)
2. **Click** the **"Model"** tab (if not already selected)
3. **Scroll down** and find these settings:

   **Scale Factor:**
   - Should be `0.01` (Unity uses meters, CC5 uses cm)
   - If it's `1`, change it to `0.01`

   **Import Blendshapes:**
   - **Click** the checkbox to **CHECK** it (✓) - **CRITICAL!**

   **Import Materials:**
   - **Click** the checkbox to **CHECK** it (✓)

4. **Click** **"Apply"** button (bottom of Inspector)

### **Step 5: Verify Blendshapes**

1. **In Project window**, **expand** `Kelly_Character.fbx` (click the small triangle ▶ next to it)
2. **Look for** a mesh file (usually named "Kelly" or similar, with a cube icon)
3. **Click** on that mesh file
4. **In Inspector**, **scroll down** to find **"Blend Shapes"** section
5. **You should see** a list of blendshape names with sliders

**What do you see?**
- ✅ **List of names** (jawOpen, mouthSmile, etc.) = **SUCCESS!** Blendshapes are there!
- ❌ **Empty or "No Blend Shapes"** = Need to re-export from CC5 with "Convert Skinned Expressions to Morphs" checked

---

## 🎯 **PART 3: Add Kelly to Scene**

### **Step 1: Drag Kelly into Scene**

1. **In Project window**, make sure `Kelly_Character.fbx` is expanded
2. **Find** the mesh file (the one with blendshapes)
3. **LEFT-CLICK and DRAG** it from Project window into the **Hierarchy** window (left side)
4. **Release** the mouse button
5. **Kelly should appear** in the Scene view (center window)

### **Step 2: Position Kelly**

1. **In Hierarchy**, **click** on Kelly (the object you just dragged)
2. **Look at Inspector** (right side)
3. **Find** the "Transform" section:
   - **Position:** X=0, Y=0, Z=0
   - **Rotation:** X=0, Y=0, Z=0
   - **Scale:** X=1, Y=1, Z=1

4. **If Kelly is too big/small:**
   - Adjust **Scale** (try 0.1, 0.01, or 10 to see what looks right)

---

## ✅ **Next Steps**

**Once Kelly is in the scene with blendshapes verified:**

1. Tell me if you see blendshapes in the Inspector
2. Then we'll add the scripts and configure lip-sync!

**Ready to start?** Open CC5 and follow Part 1, Step 1!











