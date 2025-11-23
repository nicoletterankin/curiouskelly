# Kelly Avatar - Quick Start (5 Minutes)

## 🚀 Get Kelly Running at 60fps in Unity

### **Step 1: Open Unity Project** (1 min)

```
1. Open Unity Hub
2. Click "Add" → Browse to: digital-kelly/engines/kelly_unity_player
3. Unity Version: 2021.3+ LTS (or 2022 LTS)
4. Click "Open"
```

### **Step 2: Import Scripts** (1 min)

All scripts are already in `Assets/Kelly/Scripts/`:
- ✅ `BlendshapeDriver60fps.cs`
- ✅ `AvatarPerformanceMonitor.cs`
- ✅ `KellyAvatarController.cs`
- ✅ `UnityMessageManager.cs`

Unity will auto-compile them.

### **Step 3: Create Test Scene** (2 min)

#### Option A: Use Existing Scene (if you have Kelly model)

1. Open your scene with Kelly avatar
2. Select Kelly root GameObject
3. In Inspector, click "Add Component"
4. Add these 3 components:
   - `BlendshapeDriver60fps`
   - `AvatarPerformanceMonitor`
   - `KellyAvatarController`
5. Wire up in Inspector:
   - `BlendshapeDriver60fps`:
     - `headRenderer` → Drag SkinnedMeshRenderer (Kelly's head)
     - `enableInterpolation` → ✅ Check
     - `enableGaze` → ✅ Check
     - `enableMicroExpressions` → ✅ Check
   - `AvatarPerformanceMonitor`:
     - `enableMonitoring` → ✅ Check
     - `logToConsole` → ✅ Check
   - `KellyAvatarController`:
     - `blendshapeDriver` → Auto-assigned
     - `performanceMonitor` → Auto-assigned

#### Option B: Create New Test Scene

1. Create new scene: File → New Scene
2. Add empty GameObject: GameObject → Create Empty → Name it "Kelly"
3. Add the 4 scripts to it (as above)
4. Add Camera and Light
5. Save scene as `KellyAvatar.unity`

### **Step 4: Set Target Frame Rate** (1 min)

**In Unity Editor:**
```
Edit → Project Settings → Quality
  VSync Count: Don't Sync
  
Edit → Project Settings → Time
  Fixed Timestep: 0.01666667 (60fps)
```

**In Code (already set in scripts):**
```csharp
Application.targetFrameRate = 60;
QualitySettings.vSyncCount = 0;
```

### **Step 5: Test in Play Mode** (1 min)

1. Click **Play** ▶️
2. Check top-left overlay: Should show "FPS: 60"
3. Check console: No errors
4. Click test buttons: "Test Age 5", "Test Age 35", "Test Age 102"
5. Watch console for age updates

**Expected Output:**
```
[Kelly60fps] Indexed 45 blendshapes
[KellyController] Avatar initialized and ready
[KellyController] Set learner age to 35, Kelly is now 27
[Performance] FPS: 60.2 (min: 58.7, max: 62.1), Frame Time: 16.39ms, Status: Excellent
```

---

## ✅ **Success Checklist**

- [ ] Unity project opens without errors
- [ ] Scripts compiled successfully
- [ ] Components added to Kelly GameObject
- [ ] Play Mode shows 60fps consistently
- [ ] Age test buttons work
- [ ] Console shows expected logs
- [ ] No red errors in console

---

## 🐛 **Quick Fixes**

### Problem: "Can't find SkinnedMeshRenderer"
**Fix:** Assign manually in Inspector → `headRenderer` field

### Problem: FPS shows 30 instead of 60
**Fix:** 
```csharp
// Add to Awake() in any script:
Application.targetFrameRate = 60;
QualitySettings.vSyncCount = 0;
```

### Problem: "NullReferenceException"
**Fix:** Make sure all required components are assigned in Inspector

### Problem: Scripts won't compile
**Fix:** 
- Unity version must be 2021.3+ LTS
- Check for missing using statements
- Restart Unity if needed

---

## 📱 **Test on Device (Optional)**

### iOS:
```
1. File → Build Settings → iOS
2. Click "Build"
3. Open in Xcode
4. Connect iPhone
5. Run (⌘R)
6. Should see 60fps on device
```

### Android:
```
1. File → Build Settings → Android
2. Click "Build and Run"
3. Connect Android device (USB debugging on)
4. Wait for install
5. Should see 60fps on device
```

---

## 🎉 **Next Steps**

✅ **Avatar working at 60fps?** Great! Now:

1. **Add Audio2Face data:**
   - Place `.json` files in `Assets/Kelly/A2F/`
   - Assign to `BlendshapeDriver60fps.a2fJsonAsset`
   - Test lip-sync

2. **Create 6 Age Variants:**
   - Import 6 Kelly models (ages 3, 9, 15, 27, 48, 82)
   - Assign to `KellyAvatarController.kellyAgeVariants[]`
   - Test age morphing

3. **Integrate with Flutter:**
   - Build Unity for iOS/Android
   - Copy to Flutter project
   - Test bidirectional communication

**Need detailed instructions?** See `AVATAR_UPGRADE_GUIDE.md`

**60fps not working?** Enable `showDebugInfo = true` and check performance metrics.






















