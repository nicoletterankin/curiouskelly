# iClone FBX Export - Troubleshooting Guide

**If "Export FBX" is grayed out, follow these steps:**

---

## 🔍 **Step 1: Select Kelly Character Properly**

### **1.1 Click on Kelly in the Viewport**

1. **In the 3D viewport** (center), **LEFT-CLICK** directly on Kelly's body/head
   - You should see the character highlighted or selected
   - Look for selection indicators (outline, wireframe, or bounding box)

2. **Verify selection:**
   - Look at the **right panel (Modify panel)**
   - The tab should change from "Camera" to **"Character"** or **"Modify Character"**
   - If it still says "Camera", click Kelly again

### **1.2 Alternative: Select from Hierarchy/Scene List**

1. **Look for a Scene/Content panel** (usually on the left side)
2. **Find** "Kelly" or your character name in the list
3. **Click** on it to select
4. **Verify** the Modify panel updates to show Character options

---

## 🔧 **Step 2: Check if Character is Exportable**

### **2.1 Verify Character Type**

**In Modify panel (right side), after selecting Kelly:**

1. **Look at the top** of the Modify panel
2. **You should see tabs like:**
   - ✅ **"Character"** - Good! This means it's exportable
   - ✅ **"Modify Character"** - Good!
   - ❌ **"Prop"** - Wrong, this won't export as a character
   - ❌ **"Camera"** - Wrong, you're not selecting the character

3. **If you see "Camera" or "Prop":**
   - Go back to Step 1 and properly select the character

### **2.2 Check for Locks or Restrictions**

1. **In Modify panel**, look for:
   - 🔒 **Lock icon** - If present, click to unlock
   - ⚠️ **Warning messages** - Read and resolve

2. **Check Timeline:**
   - Make sure you're not in a motion capture session
   - Make sure character isn't "locked" for animation

---

## 📤 **Step 3: Alternative Export Methods**

### **Method A: Content Manager Export**

If File → Export FBX is still grayed out, try:

1. **Find "Content Manager" or "Asset Browser"** panel (usually top right or bottom)
2. **Navigate** to your character
3. **Right-click** on Kelly character
4. **Look for** "Export" or "Export FBX" in the context menu
5. **Click** it

### **Method B: Right-Click in Viewport**

1. **Right-click** directly on Kelly in the 3D viewport
2. **Context menu** should appear
3. **Look for** "Export" or "Export FBX"
4. **Click** it

### **Method C: Use "Send to..." if Character Came from CC5**

**If Kelly came from Character Creator 5:**

1. **In iClone**, you might need to:
   - **Content Manager** → Right-click character → **"Send to..."** → **"FBX Export"**
   OR
   - Character might need to be "converted" to exportable format first

2. **Alternative:** Go back to Character Creator 5:
   - **File → Export → Export Character as FBX**
   - This exports directly from CC5 (may be easier)

---

## 🎯 **Step 4: Verify Character Setup**

### **4.1 Check if Character Needs to Be Sent from CC5 First**

**If you created Kelly in Character Creator 5:**

1. **You might need to send character to iClone properly:**
   - In CC5: **File → Send Character to iClone**
   - Wait for iClone to launch and import
   - **Then** you can export from iClone

2. **If character is already in iClone but grayed out:**
   - Character might be in wrong format
   - Try re-importing from CC5

### **4.2 Verify Facial Profile (Blendshapes)**

**Once character is selected and Modify panel shows "Character":**

1. **In Modify panel**, click on **"Facial Profile"** or **"Face"** tab
2. **You should see:**
   - List of blendshapes (50+ items)
   - Sliders for expressions
3. **If blendshapes are missing:**
   - Character wasn't exported with facial data
   - You'll need to re-export from CC5 with "Include Facial Profile" checked

---

## ✅ **Step 5: Proper Export Process (Once Selected)**

**Once Kelly is properly selected and Modify panel shows "Character":**

1. **File → Export → Export FBX** (should now be enabled)
2. **Export Settings Dialog:**
   ```
   ✓ Target: Selected Object (or Character)
   ✓ Include: Mesh
   ✓ Include: Blendshapes ✓✓✓ CRITICAL
   ✓ Include: Materials
   ✓ Include: Textures
   ✓ Include: Animation (if you have any)
   
   Scale: 100
   Coordinate System: Y-Up
   ```

3. **Click "Export"**
4. **Save location:** `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\My project\Assets\Kelly\Models\`

---

## 🚨 **Common Issues & Solutions**

### **Issue: "Export FBX" Still Grayed Out After Selecting Character**

**Try these solutions:**

1. **Restart iClone** (sometimes export options need a refresh)
2. **Save your iClone project first** (File → Save)
3. **Check if you're in a "locked" animation mode** (try creating a new scene)
4. **Verify iClone license** allows FBX export (some trial versions have limitations)

### **Issue: Character Exported but No Blendshapes**

**Solution:**
1. Re-export with **"Include Blendshapes"** explicitly checked
2. If coming from CC5, ensure **"Include Facial Profile"** was checked when sending to iClone

### **Issue: Can't Find Character in Content Manager**

**Solution:**
1. Character might be in scene but not saved to library
2. **Save character to library:** Right-click character → **"Add to Content Library"**
3. Then try exporting from Content Manager

---

## 🎯 **Quick Checklist**

- [ ] Kelly character is **selected** in viewport (left-click on character)
- [ ] Modify panel shows **"Character"** (not "Camera" or "Prop")
- [ ] No locks or restrictions visible
- [ ] Facial Profile shows blendshapes
- [ ] File → Export → Export FBX is **enabled** (not grayed out)
- [ ] Export settings include **Blendshapes** checked

---

## 📞 **If Still Not Working**

**Tell me:**
1. What does the Modify panel show when Kelly is selected? (Character/Camera/Prop?)
2. Do you see a Facial Profile tab?
3. Did Kelly come from Character Creator 5, or was it created in iClone?
4. Any error messages or warnings?

**Then I can provide more specific guidance!**












