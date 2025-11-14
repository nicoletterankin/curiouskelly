# iClone Kelly Character → Unity Setup Guide

**Complete step-by-step instructions to import your iClone Kelly character into Unity and set up viseme lip-sync.**

---

## 📋 **Prerequisites**

✅ Unity 6.2 installed and running  
✅ iClone 8 open with Kelly character loaded  
✅ Unity project created (Universal 3D template)  
✅ Scripts copied to `Assets/Kelly/Scripts/`  

---

## **STEP 1: Export Kelly from iClone to FBX** (10 minutes)

### **1.1 Prepare Character in iClone**

1. In iClone, select your Kelly character in the viewport
2. Go to **Modify** tab (right panel)
3. Verify blendshapes are present:
   - Click on **Facial Profile**
   - You should see 50+ blendshapes listed (jawOpen, mouthSmile, etc.)

### **1.2 Export to FBX**

1. **File** → **Export** → **Export FBX**
2. **Export Settings:**
   ```
   ✓ Include: Mesh
   ✓ Include: Animation (if you have any)
   ✓ Include: Blendshapes ✓✓✓ CRITICAL
   ✓ Include: Materials
   ✓ Include: Textures
   
   Scale: 100 (Unity uses meters, iClone uses cm)
   Coordinate System: Y-Up (Unity default)
   ```

3. **Click "Export"**
4. **Save as:** `Kelly_Character.fbx`
5. **Location:** `C:\Users\user\UI-TARS-desktop\digital-kelly\engines\kelly_unity_player\My project\Assets\Kelly\Models\`

---

## **STEP 2: Import FBX into Unity** (5 minutes)

### **2.1 Import the FBX File**

1. In Unity, go to **Project** window (bottom)
2. Navigate to `Assets/Kelly/Models/`
3. **Right-click** in the folder → **Import New Asset**
4. Select `Kelly_Character.fbx`
5. Click **Import**

### **2.2 Configure FBX Import Settings**

1. **Select** `Kelly_Character.fbx` in Project window
2. In **Inspector** (right panel), expand **Model** tab:
   ```
   Scale Factor: 0.01 (or 100 if you set scale to 1 in export)
   ✅ Import Blendshapes ✓✓✓ CRITICAL
   ✅ Import Materials
   ✅ Import Textures
   ```

3. Expand **Materials** tab:
   ```
   Material Creation Mode: Standard
   Location: Use External Materials (Legacy)
   ```

4. Click **Apply** (bottom of Inspector)

### **2.3 Verify Blendshapes**

1. In **Project** window, expand `Kelly_Character.fbx`
2. Click on the mesh (usually named `Kelly_Character` or similar)
3. In **Inspector**, scroll down to **Blend Shapes** section
4. **You should see a list of blendshapes** (jawOpen, mouthSmile, etc.)
5. ✅ **If blendshapes are missing:** Re-export from iClone with "Include Blendshapes" checked

---

## **STEP 3: Create Unity Scene Setup** (15 minutes)

### **3.1 Add Kelly to Scene**

1. **In Project window:**
   - Navigate to `Assets/Kelly/Models/`
   - **Drag** `Kelly_Character.fbx` into the **Hierarchy** window

2. **Rename** the GameObject to `Kelly_Avatar` (select it, press F2)

3. **Position Kelly:**
   - Select `Kelly_Avatar` in Hierarchy
   - In **Inspector**, set **Transform:**
     ```
     Position: (0, 0, 0)
     Rotation: (0, 0, 0)
     Scale: (1, 1, 1)
     ```

### **3.2 Find the Head Mesh**

1. **Expand** `Kelly_Avatar` in Hierarchy
2. **Look for** the head/face mesh (usually named like `Head`, `Face`, `Mesh`, or `Kelly_Head`)
3. **Select** the head GameObject
4. In **Inspector**, verify it has a **Skinned Mesh Renderer** component
5. ✅ **Note the name** (we'll need it for the script setup)

---

## **STEP 4: Add Scripts to Kelly** (10 minutes)

### **4.1 Add Main Controller Scripts**

1. **Select** `Kelly_Avatar` in Hierarchy (or the head GameObject if you found it)
2. In **Inspector**, click **Add Component**
3. Search and add:
   - `Kelly Avatar Controller`
   - `Blendshape Driver 60fps`
   - `Unity Message Manager`

### **4.2 Configure BlendshapeDriver60fps**

1. **Select** `Kelly_Avatar` (or head GameObject)
2. In **Inspector**, find **Blendshape Driver 60fps** component
3. **Drag and drop** the head mesh GameObject into:
   - **Head Renderer** field
   - (The component should auto-find the SkinnedMeshRenderer)

4. **If Head Renderer is empty:**
   - Click the **target icon** (circle) next to the field
   - Select the head GameObject from the scene

5. **Add Audio Source:**
   - Click **Add Component** → **Audio Source**
   - In **Blendshape Driver 60fps**, drag the AudioSource component into the **Audio Source** field

### **4.3 Configure KellyAvatarController**

1. **Select** `Kelly_Avatar` (main GameObject)
2. In **Inspector**, find **Kelly Avatar Controller** component
3. **Drag and drop:**
   - **Blendshape Driver:** Drag `Blendshape Driver 60fps` component into this field
   - (Or select from dropdown if it's on the same GameObject)

---

## **STEP 5: Verify Blendshape Mapping** (10 minutes)

### **5.1 Check Blendshape Names**

1. **Select** the head GameObject (with SkinnedMeshRenderer)
2. In **Inspector**, expand **Skinned Mesh Renderer**
3. Scroll to **Blend Shapes** section
4. **Make a list** of blendshape names (write them down or screenshot)

### **5.2 Update Viseme Mapping (If Needed)**

1. Open `BlendshapeDriver60fps.cs` in your code editor
2. Find the `VisemeToBlendshapeMap` dictionary (around line 128)
3. **Compare** your iClone blendshape names with the mapping:
   - Common iClone names: `jawOpen`, `mouthSmile`, `mouthO`, `tongueUp`
   - If your names differ, update the mapping

**Example mapping:**
```csharp
{ "aa", new[] { "jawOpen", "mouthOpen" } },  // Your blendshapes must match these names
```

4. **Save** the script and return to Unity (Unity will auto-compile)

---

## **STEP 6: Test the Setup** (5 minutes)

### **6.1 Enter Play Mode**

1. Click **Play** ▶️ button in Unity
2. In **Console** window (bottom), you should see:
   ```
   [Kelly60fps] Indexed X blendshapes
   [KellyController] Avatar initialized and ready
   ```

### **6.2 Test Blendshapes (Debug)**

1. **Select** `Kelly_Avatar` in Hierarchy (while Play Mode is active)
2. In **Inspector**, find **Blendshape Driver 60fps**
3. Expand **Blend Shapes** in the SkinnedMeshRenderer
4. **Manually adjust** blendshape sliders to verify:
   - Mouth opens (`jawOpen`)
   - Lips move (`mouthSmile`, `mouthPucker`)
   - Eyes blink (`eyeBlink_L`, `eyeBlink_R`)

✅ **If blendshapes work:** Setup is successful!

### **6.3 Test Viseme Input (Optional)**

1. In **Play Mode**, open the **Console**
2. You can manually test visemes by calling:
   ```
   Kelly_Avatar.GetComponent<BlendshapeDriver60fps>().UpdateVisemes(
       new Dictionary<string, float> { 
           { "aa", 0.8f }, 
           { "ee", 0.5f } 
       }
   );
   ```

---

## **STEP 7: Connect Flutter Integration** (When Ready)

Once the basic setup works, you can connect the Flutter app:

1. **Add** `UnityMessageManager` to a GameObject (it should auto-create)
2. **Verify** `KellyAvatarController` is subscribed to messages
3. **Test** from Flutter by sending:
   ```json
   {
     "type": "updateVisemes",
     "visemes": {
       "aa": 0.8,
       "ee": 0.5
     }
   }
   ```

---

## 🔧 **Troubleshooting**

### **Issue: Blendshapes don't appear in Unity**

**Solutions:**
1. Re-export from iClone with "Include Blendshapes" checked
2. In Unity FBX Import Settings, enable "Import Blendshapes"
3. Check that the mesh has a SkinnedMeshRenderer (not MeshRenderer)

### **Issue: Blendshapes don't move when testing**

**Solutions:**
1. Verify Head Renderer is assigned in BlendshapeDriver60fps
2. Check Console for errors
3. Verify blendshape names match the mapping in `VisemeToBlendshapeMap`

### **Issue: Scripts show compilation errors**

**Solutions:**
1. Check Console for specific error messages
2. Verify all scripts are in `Assets/Kelly/Scripts/` folder
3. Make sure Unity is using the correct C# version (Unity 6.2 should be fine)

### **Issue: Character appears in wrong scale**

**Solutions:**
1. Adjust FBX Import Scale Factor (0.01 or 100)
2. Or scale the GameObject Transform in the scene

---

## ✅ **Success Checklist**

- [ ] Kelly character imported from iClone
- [ ] Blendshapes visible in Unity (50+ shapes)
- [ ] Kelly_Avatar GameObject in scene
- [ ] BlendshapeDriver60fps component added and configured
- [ ] Head Renderer assigned
- [ ] Audio Source added
- [ ] KellyAvatarController component added
- [ ] Scripts compile without errors
- [ ] Play Mode runs without errors
- [ ] Manual blendshape testing works
- [ ] Viseme mapping matches your blendshape names

---

## 🎉 **Next Steps**

Once everything is working:

1. **Test with real viseme data** from OpenAI Realtime API
2. **Fine-tune viseme mapping** if mouth shapes don't look right
3. **Add audio playback** for complete lip-sync testing
4. **Connect to Flutter** for end-to-end testing

**Ready to start?** Follow Step 1 (Export from iClone) and tell me when you've exported the FBX file!












