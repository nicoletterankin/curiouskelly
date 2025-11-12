# Kelly Workflow — Which Path to Take?

## 🤔 The Confusion (Multiple Paths Exist)

Character Creator 5 + Headshot 2 offers **3 different workflows**:
1. **Photo to 3D (Direct)** ← **USE THIS ONE**
2. Mesh to 3D (for 3D scans)
3. ActorBUILD (manual photo matching)

---

## ✅ YOUR RECOMMENDED PATH: Photo to 3D (Simple & Fast)

### What You Have:
- ✓ 2D reference photo: `headshot2-kelly-base169 101225.png`
- ✓ No 3D mesh/scan
- ✓ Goal: Fast, high-quality result

### The Right Workflow:

```
START
  ↓
Step 1: Pick ANY CC5 HD base character (just for body proportions)
  ↓
Step 2: Headshot 2 → Photo to 3D → REPLACES the face completely
  ↓
Step 3: Send to iClone
  ↓
DONE
```

---

## 📋 EXACT CLICK-BY-CLICK INSTRUCTIONS

### **STEP 1: Choose Base Character (Body Only)**

**Why:** You need a body. Headshot 2 will replace the face anyway, so the base face doesn't matter.

**Action:**
1. Open Character Creator 5
2. `File → New Project`
3. Content panel (left side) → `Actor → Character → HD Human Anatomy Set`
4. Click **Female Athletic (CC5 HD)** 
5. Click **Apply** → wait for load
6. Save as: `C:\iLearnStudio\projects\Kelly\CC5\Kelly_Base.ccProject`

**⚠️ Important:** Don't waste time adjusting the face with ActorMIXER. Headshot 2 will overwrite it.

---

### **STEP 2: Headshot 2 → Photo to 3D (MAIN STEP)**

**This is where the magic happens - your photo becomes a 3D head.**

**Action:**
1. Top menu → `Plugins → Headshot 2`
2. You'll see TWO options:
   - **Photo to 3D** ← **CLICK THIS ONE**
   - ~~Mesh to 3D~~ ← Don't use (that's for 3D scans)

3. In the Headshot 2 window:
   - Click **Load Photo**
   - Navigate to: `C:\iLearnStudio\projects\Kelly\Ref\`
   - Select: `headshot2-kelly-base169 101225.png`
   - Click **Open**

4. **Configure Settings** (V2 optimized):
   ```
   Resolution: Ultra High (8K)
   Mesh Density: Maximum
   Detail Level: Maximum
   Processing Quality: Ultra High
   Character Gender: Female
   Age Range: 25-35
   Lighting Mode: Balanced
   ```

5. Click **Generate**
   - Progress bar appears
   - Takes ~5-8 minutes (your RTX 5090)
   - ☕ Get coffee

6. When done, you'll see the result preview
   - The face now looks like Kelly!
   - Click **Apply to Character**
   - Wait 2-3 minutes (applying)
   - Click **Accept**

7. Save as: `C:\iLearnStudio\projects\Kelly\CC5\Kelly_HS2_Applied.ccProject`

---

### **STEP 3: Polish (Optional but Recommended)**

**Minor tweaks using Headshot 2's sliders:**

1. Go to: `Modify → Morph → Headshot 2`
2. Adjust if needed:
   - Jaw Roundness: -5 (if too wide)
   - Lip Thickness: +3 (lower lip)
   - Eye Outer Tilt: +1
3. Save as: `Kelly_HS2_Polished.ccProject`

---

### **STEP 4: Materials & Details**

1. `Modify → Material` → Verify **Digital Human Shader** active
2. Skin settings:
   - Roughness: 0.38-0.42
   - SSS: 0.25-0.30
3. Apply HD Eyes & Lashes from Content panel
4. Choose hair style from Content panel
5. Save as: `Kelly_Complete.ccProject`

---

### **STEP 5: Send to iClone**

1. `File → Send Character to iClone`
2. Confirm: "Digital Human (CC5 HD)"
3. iClone opens with Kelly
4. Save iClone project as: `C:\iLearnStudio\projects\Kelly\iClone\Kelly_Scene.iProject`

---

## 🚫 WHAT NOT TO DO

### ❌ Don't Use "Mesh to 3D"
- That's only if you have a `.obj` or `.fbx` 3D scan
- You have a 2D photo, so use "Photo to 3D"

### ❌ Don't Use ActorBUILD
- That's the OLD manual way (pre-Headshot 2)
- Takes hours vs. minutes

### ❌ Don't Spend Time on Base Face
- ActorMIXER adjustments on Step 1 are wasted
- Headshot 2 replaces everything anyway

---

## 🎯 Quick Reference

| You Have | Use This |
|----------|----------|
| 2D Photo (JPG/PNG) | **Photo to 3D** ← You are here |
| 3D Scan (OBJ/FBX) | Mesh to 3D |
| Multiple angle photos | Photo to 3D (front view), then Mesh to 3D (wrap) |

---

## 📸 Your Current Status

**Your files:**
- ✓ Photo ready: `headshot2-kelly-base169 101225.png`
- ✓ CC5 installed and running
- ✓ RTX 5090 ready

**Next immediate action:**
1. Open CC5 → File → New Project
2. Pick Female Athletic HD base
3. Plugins → Headshot 2 → **Photo to 3D**
4. Load your PNG file
5. Click Generate

That's it! No mesh, no confusion.

























