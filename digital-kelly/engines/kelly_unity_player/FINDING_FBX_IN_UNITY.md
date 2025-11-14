# Finding Your FBX File in Unity - Step-by-Step

**Key Point: Unity doesn't "open" files. Files in the Assets folder are automatically available.**

---

## 🎯 **The Unity Interface - Where Everything Is**

Unity has **5 main windows**:

1. **Hierarchy** (top-left) - Shows objects in your scene
2. **Scene View** (center) - Shows the 3D world
3. **Inspector** (right) - Shows details of selected items
4. **Project** (bottom-left) - Shows your files (THIS IS WHERE YOUR FBX IS)
5. **Console** (bottom-right) - Shows errors/messages

---

## 📁 **STEP 1: Find the Project Window**

**Look at the bottom of Unity Editor:**

1. **At the very bottom**, you'll see tabs like:
   - **"Project"** (this is what you need!)
   - **"Console"**
   - **"Animator"**
   - etc.

2. **Click** on the **"Project"** tab

3. **You should see** a file browser-like view with folders:
   ```
   Assets
   ├── Scenes
   ├── Settings
   ├── TutorialInfo
   └── ... (other folders)
   ```

---

## 🔍 **STEP 2: Navigate to Your FBX File**

**In the Project window:**

1. **Look for** a folder called **"Assets"**
   - If you don't see it, look for a small triangle ▶ next to "Assets"
   - **Click** the triangle to expand it

2. **Inside Assets**, look for a folder called **"Kelly"**
   - **Click** on **"Kelly"** folder
   - If you don't see it, the FBX might be in Assets directly

3. **Inside Kelly**, look for a folder called **"Models"**
   - **Click** on **"Models"** folder

4. **You should see** `iclone_unity_kelly_export1.fbx` listed

**If you DON'T see the Kelly/Models folder:**

The FBX file might not be in the right location. Let's fix that:

### **Option A: Copy File to Unity Assets Folder**

1. **Open Windows File Explorer**
2. **Navigate to:** `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\My project\Assets\`
3. **Create** a folder called **"Kelly"** (if it doesn't exist)
4. **Inside Kelly**, create a folder called **"Models"**
5. **Copy** `iclone_unity_kelly_export1.fbx` from wherever it is
6. **Paste** it into `Assets\Kelly\Models\`
7. **Go back to Unity**
8. **In Unity**, click on **"Assets"** folder in Project window
9. **Press F5** or **right-click** → **"Refresh"**
10. **Now** you should see Kelly → Models → your FBX file

---

## ✅ **STEP 3: Select the FBX File**

**Once you see the FBX file in the Project window:**

1. **LEFT-CLICK once** on `iclone_unity_kelly_export1.fbx`
2. **The file will highlight** (usually blue)
3. **Look at the Inspector** (right side of Unity)
4. **You should see** details about the FBX file

---

## 🔧 **STEP 4: Check if Unity Imported It**

**If the FBX is selected:**

1. **In Inspector** (right side), you should see:
   - **"Model"** tab (click it)
   - Import settings like "Scale Factor", "Import Blendshapes", etc.

2. **If you see import settings:**
   - ✅ **GOOD!** Unity has imported the file
   - **Scroll down** and make sure **"Import Blendshapes"** is checked (✓)

3. **If you DON'T see import settings:**
   - Unity hasn't imported it yet
   - **Right-click** the FBX file → **"Reimport"**
   - **Wait** for Unity to finish importing

---

## 🎨 **STEP 5: Expand FBX to See Mesh**

**To check for blendshapes:**

1. **In Project window**, look at `iclone_unity_kelly_export1.fbx`
2. **Next to it**, you should see a small **triangle ▶** or **arrow**
3. **Click** that triangle to expand the FBX
4. **You'll see** sub-items like:
   - A mesh file (cube icon, might be named "Kelly" or similar)
   - Materials (sphere icons)
   - Textures (image icons)

5. **Click** on the **mesh file** (the one with cube icon)
6. **Look at Inspector** (right side)
7. **Scroll down** to find **"Blend Shapes"** section

---

## 🚨 **TROUBLESHOOTING**

### **Problem: I don't see a Project window at all**

**Solution:**
1. **Top menu bar** → **"Window"**
2. **Click** **"General"**
3. **Click** **"Project"**
4. Project window should appear at bottom

### **Problem: Project window is too small**

**Solution:**
1. **Hover** mouse over the **edge** of Project window
2. **Drag** to make it bigger
3. Or **drag** the Project tab to make it a separate window

### **Problem: I still can't find the FBX file**

**Solution:**
1. **In Project window**, look at the **top**
2. **Find** a **search bar** (says "Search" or has a magnifying glass icon)
3. **Type:** `iclone_unity_kelly_export1`
4. **Press Enter**
5. **The file should appear** in search results

### **Problem: Unity shows "SampleScene" instead of my files**

**Solution:**
- The **Scene** window shows the current scene (SampleScene is normal)
- The **Project** window is separate - look at the **bottom tabs**
- **Click** the **"Project"** tab to see files

---

## 📸 **Visual Guide**

**Unity Layout:**
```
┌─────────────────────────────────────────────────────┐
│  File  Edit  Assets  GameObject  ... (Menu Bar)    │
├──────────┬──────────────────────────┬───────────────┤
│          │                          │               │
│ Hierarchy│     Scene View          │  Inspector    │
│ (Objects)│     (3D World)           │  (Details)    │
│          │                          │               │
├──────────┴──────────────────────────┴───────────────┤
│  Project Tab | Console Tab                          │
│  ┌────────────────────────────────────────────────┐ │
│  │ Assets                                          │ │
│  │   ├── Scenes                                    │ │
│  │   ├── Kelly  ← Click here!                     │ │
│  │   │   └── Models                                │ │
│  │   │       └── iclone_unity_kelly_export1.fbx  │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Look at the BOTTOM for the Project window!**

---

## ✅ **Quick Checklist**

- [ ] Found Project window (bottom of Unity)
- [ ] Clicked on Assets folder
- [ ] Found Kelly → Models folder
- [ ] Saw iclone_unity_kelly_export1.fbx listed
- [ ] Clicked on the FBX file
- [ ] Saw Inspector show FBX details
- [ ] Expanded FBX (clicked triangle ▶)
- [ ] Clicked on mesh file inside
- [ ] Found "Blend Shapes" section in Inspector

**Once you find the FBX file and select it, tell me what you see in the Inspector!**












