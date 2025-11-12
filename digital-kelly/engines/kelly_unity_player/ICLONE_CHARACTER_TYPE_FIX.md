# Fix Grayed Out "Convert Skinned Expressions to Morphs"

**This option is grayed out because your character needs to be a CC (Character Creator) character type.**

---

## 🔍 **Step 1: Check Character Type in iClone**

### **1.1 Verify Character Type**

1. **Select Kelly** in the viewport (left-click on her)
2. **Look at Modify panel** (right side)
3. **Check the character information:**
   - Does it say **"CC Character"** or **"Character Creator"** anywhere?
   - Or does it say **"G5 Character"**, **"G6 Character"**, or **"Actor"**?

### **1.2 If It's NOT a CC Character:**

You have two options:

---

## 📤 **Option A: Export from Character Creator 5 (Recommended)**

**If Kelly was created in CC5 or you have CC5:**

1. **Open Character Creator 5**
2. **Load your Kelly character** (File → Open Project)
3. **File → Export → Export Character as FBX**
4. **In the export dialog:**
   ```
   ✓ Include Facial Profile (blendshapes) ✓✓✓ CRITICAL
   ✓ Include Materials
   ✓ Include Textures
   Scale: 100
   ```

5. **Export directly to:**
   `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\My project\Assets\Kelly\Models\`

6. **This exports with blendshapes automatically!**

---

## 🔄 **Option B: Convert Character in iClone (If CC5 Not Available)**

### **Step 1: Convert to CC Character**

1. **Select Kelly** in iClone viewport
2. **Right-click** on Kelly in the viewport or in the Scene panel
3. **Look for:**
   - **"Convert to CC Character"**
   - **"Bake Character"**
   - **"Character → Convert"**

4. **If found, click it and wait for conversion**

### **Step 2: Send Character from CC5 to iClone (Proper Method)**

**If Kelly exists in Character Creator 5:**

1. **Open Character Creator 5**
2. **Load Kelly project**
3. **File → Send Character to iClone**
4. **In the send dialog:**
   ```
   ✓ Include Facial Profile (blendshapes) ✓✓✓ CRITICAL
   ✓ Include Expression Wrinkle Maps
   ✓ Include Hair Physics Setup
   Quality: Ultra High
   ```

5. **Click "Send"** - iClone will launch/open and Kelly will import

6. **Now in iClone:** The "Convert Skinned Expressions to Morphs" option should be available!

---

## 🎯 **Option C: Use Existing FBX and Check Blendshapes**

**You already exported `iclone_unity_kelly_export1.fbx` - let's check if it has blendshapes:**

### **Test in Unity:**

1. **Go to Unity** (already open)
2. **Project window** → Navigate to `Assets/Kelly/Models/`
3. **Select** `iclone_unity_kelly_export1.fbx`
4. **In Inspector**, expand the FBX:
   - Click on the mesh inside (usually named like "Kelly" or similar)
   - Scroll down to **"Blend Shapes"** section
   - **Do you see a list of blendshapes?** (jawOpen, mouthSmile, etc.)

### **If Blendshapes Are Present:**

✅ **You're good!** Even without checking that box, the export might have included them.

### **If Blendshapes Are Missing:**

❌ **You'll need to export from CC5 or convert the character first.**

---

## 📋 **Quick Decision Tree**

**Answer these questions:**

1. **Do you have Character Creator 5 installed?**
   - ✅ **Yes** → Use **Option A** (Export from CC5)
   - ❌ **No** → Continue

2. **Is Kelly already in Character Creator 5?**
   - ✅ **Yes** → Use **Option B Step 2** (Send from CC5 to iClone)
   - ❌ **No** → Continue

3. **Can you right-click Kelly in iClone and see "Convert to CC Character"?**
   - ✅ **Yes** → Use **Option B Step 1** (Convert in iClone)
   - ❌ **No** → Continue

4. **Check the exported FBX in Unity - does it have blendshapes?**
   - ✅ **Yes** → You're done! Skip re-export
   - ❌ **No** → You need CC5 or conversion

---

## 🚀 **Recommended Next Step**

**Tell me:**
1. Do you have Character Creator 5 installed?
2. Is Kelly saved as a CC5 project?
3. Can you check the exported FBX in Unity for blendshapes?

**Based on your answers, I'll give you the exact next steps!**











