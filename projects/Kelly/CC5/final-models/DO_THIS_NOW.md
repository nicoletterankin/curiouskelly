# 🎯 GET KELLY LIVE — Click-by-Click Guide

**Time Required:** ~45 minutes  
**Difficulty:** Easy (just follow the clicks)  
**Goal:** Kelly in her chair, teaching on your website

---

## 🔴 PHASE 1: Open Kelly in CC5 (5 minutes)

### Step 1.1: Launch Character Creator 5
- Double-click the **Character Creator 5** icon on your desktop
- Wait for it to fully load

### Step 1.2: Open the Project
1. Click **File** (top menu)
2. Click **Open Project**
3. Navigate to:
   ```
   C:\Users\user\UI-TARS-desktop\projects\Kelly\CC5\final-models\
   ```
4. Select: **CC5 Cloth update 8.1.ccProject**
5. Click **Open**
6. ⏳ Wait 30-60 seconds for the 400MB file to load

### ✅ Checkpoint 1
You should see Kelly sitting in a director's chair wearing:
- Blue sweater
- Jeans
- White sneakers

**Screenshot this moment!** 📸

---

## 🟠 PHASE 2: Send to iClone 8 (2 minutes)

### Step 2.1: Export to iClone
1. Click **File** (top menu)
2. Click **Export**
3. Click **Send Character to iClone**
4. ⏳ Wait for iClone 8 to launch automatically

### ✅ Checkpoint 2
iClone 8 should open with Kelly visible in the viewport.

---

## 🟡 PHASE 3: Set Up Animation (10 minutes)

### Step 3.1: Initialize AccuLips (CRITICAL!)
1. In iClone, click **Animation** (top menu)
2. Click **AccuLips**
3. A panel will open — **this is all you need to do!**
   - Just opening this panel activates the lip-sync blendshapes
4. You can close the panel now

### Step 3.2: Activate Expression Plus
1. Click **Modify** (top menu)
2. Click **Face Key**
3. Click **Expression Plus**
4. In the panel, click the **Default** button
5. This activates all 63+ facial blendshapes

### Step 3.3: Add Breathing (Optional but recommended)
1. Click **Motion** (left side or top menu)
2. Navigate to: **Idle** → **Female** → **Breathing**
3. Double-click a breathing animation to apply
4. In the timeline at the bottom, set:
   - Start: **0**
   - End: **600** (10 seconds)

### ✅ Checkpoint 3
Kelly should now have subtle breathing movement when you hit play.

---

## 🟢 PHASE 4: Export FBX (5 minutes)

### Step 4.1: Open Export Dialog
1. Click **File** (top menu)
2. Click **Export**
3. Click **FBX**

### Step 4.2: Configure Settings (CRITICAL!)

In the export dialog, set **EXACTLY** these options:

| Setting | Value |
|---------|-------|
| **Target Tool Preset** | `Unity 3D` ← SELECT THIS! |
| **FBX Options > Mesh** | ✅ Checked |
| **FBX Options > Motion** | ✅ Checked |
| **Embed Textures** | ✅ Checked |
| **Delete Hidden Faces** | ❌ Unchecked |

### Step 4.3: Save the File
1. Click **Export** button
2. Navigate to:
   ```
   C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\Assets\
   ```
3. Filename: **Kelly_Chair_Final.fbx**
4. Click **Save**
5. ⏳ Wait for export to complete (may take 1-2 minutes)

### ✅ Checkpoint 4
You should see `Kelly_Chair_Final.fbx` in the Assets folder.

---

## 🔵 PHASE 5: Import to Unity (10 minutes)

### Step 5.1: Open Unity Hub
1. Launch **Unity Hub** from your desktop
2. Find the project: **onlykelly**
3. Click to open it
4. ⏳ Wait for Unity to load (may take 2-3 minutes)

### Step 5.2: Auto-Import Will Trigger
1. Unity will detect the new FBX file
2. A popup should appear: **"CCIC Auto Setup"**
3. Select: **High Quality (URP)**
4. Click **OK** or **Apply**
5. ⏳ Wait for processing (textures, materials, rig)

### Step 5.3: Create the Scene
1. **File** → **New Scene** → **URP Basic**
2. **File** → **Save As**
3. Navigate to: `Assets/Scenes/`
4. Name: **Kelly_Chair_Main**
5. Click **Save**

### Step 5.4: Add Kelly to Scene
1. In the **Project** window (bottom), find **Kelly_Chair_Final** prefab
2. **Drag** it into the **Hierarchy** window (left side)
3. In **Inspector** (right side), set Position to: **0, 0, 0**

### Step 5.5: Set Up Camera
1. Click **Main Camera** in Hierarchy
2. In Inspector, set:
   - **Position:** X=0, Y=1.1, Z=2.5
   - **Field of View:** 35
3. You should see Kelly framed nicely in the Game view

### ✅ Checkpoint 5
Press **Play** button (top center). Kelly should be visible and breathing!

---

## 🟣 PHASE 6: Build WebGL (10 minutes)

### Step 6.1: Configure Build
1. **File** → **Build Settings**
2. Select **WebGL** platform (left side)
3. If not active, click **Switch Platform** (wait for it)
4. Make sure **Kelly_Chair_Main** scene is checked in the list
5. Click **Player Settings** (bottom left)

### Step 6.2: Critical Player Settings
In Player Settings panel:
1. Expand **Publishing Settings**
2. Find **Compression Format**
3. Set to: **Disabled** ← CRITICAL!

### Step 6.3: Build!
1. Close Player Settings
2. Click **Build**
3. Create a new folder:
   ```
   Builds/WebGL/kelly-chair
   ```
4. Click **Select Folder**
5. ⏳ Wait for build (10-15 minutes)

### ✅ Checkpoint 6
Build completes without errors. Folder contains `index.html` and `Build/` folder.

---

## ⬛ PHASE 7: Deploy to Website (2 minutes)

### Step 7.1: Run the Deploy Script
I've created a script for you. Open PowerShell and run:

```powershell
cd C:\Users\user\UI-TARS-desktop
.\deploy-kelly-chair.ps1
```

### ✅ DONE! 🎉

Kelly in her chair is now live at:
```
/unity/kelly-chair/index.html
```

---

## 🆘 TROUBLESHOOTING

### "I don't see the CCIC popup in Unity"
- Right-click on `Kelly_Chair_Final.fbx` in Project window
- Click **Reimport**
- The popup should appear

### "Kelly looks weird/black"
- The materials didn't import correctly
- In Project window, find the Materials folder
- Make sure URP materials are applied

### "Build failed"
- Check the Console window (Window → General → Console)
- Look for red error messages
- Most common: compression format not disabled

### "Can't find the deploy script"
- Run these commands manually in PowerShell:
```powershell
robocopy "C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\Builds\WebGL\kelly-chair" "C:\Users\user\UI-TARS-desktop\public\unity\kelly-chair" /MIR
```

---

## 📞 QUICK REFERENCE

| Phase | What You're Doing | Time |
|-------|-------------------|------|
| 1 | Open Kelly in CC5 | 5 min |
| 2 | Send to iClone | 2 min |
| 3 | Set up animation | 10 min |
| 4 | Export FBX | 5 min |
| 5 | Import to Unity | 10 min |
| 6 | Build WebGL | 10 min |
| 7 | Deploy | 2 min |
| **Total** | | **~45 min** |

---

**You've got this! One step at a time.** 🪑✨
