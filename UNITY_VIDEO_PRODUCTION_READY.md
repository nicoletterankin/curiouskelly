# 🎬 Unity Video Production - Ready for Tomorrow's Art Update

**Generated:** Sunday, December 7, 2025  
**Last Updated:** December 7, 2025 (Added blendshape configuration)  
**Status:** 🟢 READY TO RECEIVE NEW ART FILE  
**Expected Delivery:** Tomorrow (December 8, 2025)

---

## 📋 EXECUTIVE SUMMARY

You're receiving an updated art file tomorrow with:
1. **Final Scene** setup
2. **Starting Pose** for Kelly

This document catalogs everything you need to know for a smooth update — no rebuilding from scratch required.

### NEW: Canonical Configuration Files

All expression, blend shape, and viseme data is now centralized in:

| File | Purpose |
|------|---------|
| `scripts/kelly-video-factory/kelly-blendshape-config.ts` | **Single source of truth** for all Kelly expressions, blend shapes, visemes |
| `scripts/kelly-video-factory/kelly-lipsync-engine.ts` | Lip-sync timing and Unity integration |
| `scripts/kelly-video-factory/sota-video-pipeline.ts` | Video generation (now uses canonical config) |

**These files ensure Unity WebGL and video pipeline stay perfectly in sync.**

---

## 🎯 WHAT NEEDS TO BE UPDATED (Tomorrow's Checklist)

### When New Art File Arrives:

| Step | Action | Location |
|------|--------|----------|
| 1 | Save new file to | `arif-deliveries/milestone-2-phase-1/original/` |
| 2 | Import to CC5 | Character Creator 5 |
| 3 | Export to Unity | `Assets/Kelly/Animations/Lessons/` |
| 4 | Rebuild WebGL | Kelly Menu → Build → 🚀 Build WebGL (Production) |
| 5 | Copy to web | Run `copy-build-to-web.ps1` |

---

## 🗂️ CRUCIAL FILE LOCATIONS

### Unity Project
```
C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly\
```

### Current Model (TO BE REPLACED)
```
Assets/Kelly/Animations/Lessons/kelly_fbx_v4.fbx  (96.77 MB)
```

### Source Projects (CC5/iClone)
```
digital-kelly/Kelly_Unity_Production.ccProject   ← CC5 main project
digital-kelly/kelly_directors_chair.iProject     ← iClone animation project
digital-kelly/CC5 Cloth update 1.1.ccProject     ← Cloth update
```

### WebGL Build Output (Local)
```
digital-kelly/engines/Kelly_Engine_V2/onlykelly/Kelly_Web_Build/Build/
```

### WebGL Build Output (Production Deploy)
```
public/unity/kelly-live/Build/
```

---

## 🎮 UNITY PROJECT SPECIFICATIONS

| Property | Value |
|----------|-------|
| **Unity Version** | **6000.2.10f1** (Unity 6) |
| **Render Pipeline** | **URP** (Universal Render Pipeline) |
| **Main Scene** | `Assets/KellyMain.unity` |
| **Target GameObject** | `kelly_fbx_v4` |
| **WebGL Memory** | 2048 MB |
| **Compression** | Brotli (.br files) |
| **Current Build Size** | ~313 MB (uncompressed), ~235 MB (Brotli) |

---

## 🎭 EXPRESSION SYSTEM REQUIREMENTS

> **📁 CANONICAL SOURCE:** `scripts/kelly-video-factory/kelly-blendshape-config.ts`
> 
> All expression data below is derived from this single source of truth.
> If you need to modify expressions, **edit that file** and both Unity and video pipeline will stay in sync.

### Blend Shapes Expected by Scripts

The new art file MUST include these blend shapes for expressions to work:

**Eyes:**
- `Eye_Blink_L`, `Eye_Blink_R`
- `Eye_Wide_L`, `Eye_Wide_R`
- `Eye_Squint_L`, `Eye_Squint_R`

**Brows:**
- `Brow_Raise_Inner_L`, `Brow_Raise_Inner_R`
- `Brow_Raise_Outer_L`, `Brow_Raise_Outer_R`

**Mouth:**
- `Mouth_Smile_L`, `Mouth_Smile_R`
- `Mouth_Open`
- `Mouth_Shrug_Upper`
- `V_Open` (critical for lip sync)

**Cheeks:**
- `Cheek_Raise_L`, `Cheek_Raise_R`

### Visemes for Lip-Sync (Critical)

| Viseme | Used For | Example Phonemes |
|--------|----------|------------------|
| `V_Open` | Open mouth | AA, AH |
| `V_Wide` | Wide spread | IY, EH, AE |
| `V_Tight_O` | Rounded | OO, OH, W |
| `V_Explosive` | Closed/Plosive | P, B, M |
| `V_Dental_Lip` | Teeth visible | F, V, TH |

### Expressions Mapped in Code

| Expression | Blend Shapes Used | Video Prompt Keywords | Use Case |
|------------|------------------|----------------------|----------|
| **happy** | Smile (70%), Cheek (40%), Eye Squint (20%) | "eyes sparkling, genuine warmth, warm smile showing teeth" | Positive reactions |
| **curious** | Brow Inner (50%), Eye Wide (25%) | "head tilted, eyebrow raised, warm inviting expression" | Welcome phase |
| **explaining** | Brow Outer (35%), Mouth Shrug (15%) | "animated expression, eyebrows raised with emphasis" | Teaching/questions |
| **listening** | Brow Inner (25%), Smile (20%) | "attentive expression, soft encouraging smile" | Receiving answers |
| **wisdom** | Smile (45%), Eye Squint (25%), Brow Inner (20%) | "contemplative, soft knowing smile, sincere gaze" | Wisdom phase |
| **celebrating** | Smile (90%), Cheek (60%), Eye Squint (35%), Brow (40%) | "proud delighted, eyes crinkled, radiant smile" | Achievement |
| **neutral** | All zeros | "calm neutral, relaxed features" | Default state |
| **excited** | Smile (80%), Eye Wide (30%), Brow Inner (45%), Cheek (50%) | "eyes sparkling with excitement, teeth showing" | Hook phase |
| **heartfelt** | Smile (35%), Brow Inner (35%), Eye Squint (10%) | "hand over heart, eyes filled with warmth" | Emotional moments |
| **welcome** | Brow Inner (40%), Eye Wide (20%), Smile (50%), Cheek (25%) | "arms open, genuine warm smile, bright eyes" | Greeting |

### Phase-to-Expression Mapping

```typescript
// From kelly-blendshape-config.ts
const PHASE_TO_EXPRESSION = {
  welcome: 'curious',
  question: 'explaining',
  q1: 'explaining',
  q2: 'explaining',
  q3: 'explaining',
  hook: 'excited',
  fact1: 'curious',
  fact2: 'explaining',
  fact3: 'thoughtful',
  wisdom: 'wisdom',
  celebrating: 'celebrating',
};
```

---

## 📜 CORE SCRIPTS (DO NOT MODIFY)

### 1. KellyWebGLBridge.cs
**Location:** `Assets/Scripts/KellyWebGLBridge.cs`  
**Purpose:** Main JavaScript ↔ Unity communication  
**Target:** Must be attached to `kelly_fbx_v4` GameObject

**JavaScript API:**
```javascript
// Expression Control (see kelly-blendshape-config.ts for all expressions)
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'happy');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'curious');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'explaining');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'wisdom');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'celebrating');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'excited');
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'heartfelt');

// Lip Sync
unityInstance.SendMessage('kelly_fbx_v4', 'StartLipSync', 'Hello world');
unityInstance.SendMessage('kelly_fbx_v4', 'StopLipSync');

// Phase Control (auto-maps to expressions)
unityInstance.SendMessage('kelly_fbx_v4', 'SetPhase', 'welcome');  // → curious
unityInstance.SendMessage('kelly_fbx_v4', 'SetPhase', 'hook');     // → excited
unityInstance.SendMessage('kelly_fbx_v4', 'SetPhase', 'question'); // → explaining
unityInstance.SendMessage('kelly_fbx_v4', 'SetPhase', 'wisdom');   // → wisdom

// Speaking State
unityInstance.SendMessage('kelly_fbx_v4', 'SetSpeaking', 'true');

// Viseme Control (for advanced lip-sync)
unityInstance.SendMessage('kelly_fbx_v4', 'ProcessViseme', 'V_Open:0.8');
unityInstance.SendMessage('kelly_fbx_v4', 'ProcessViseme', 'V_Wide:0.5');

// Animations
unityInstance.SendMessage('kelly_fbx_v4', 'PlayAnimation', 'wave');
```

### 2. ARKitBlendshapeController.cs
**Location:** `Assets/Scripts/ARKitBlendshapeController.cs`  
**Purpose:** Maps and controls blend shapes  
**Note:** Auto-discovers all blend shapes at runtime

### 3. KellyAvatarController.cs
**Location:** `Assets/KellyAvatarController.cs`  
**Purpose:** Alternative expression controller  
**Note:** Uses different blend shape naming (CC4/iClone conventions)

### 4. LipSyncController.cs
**Location:** `Assets/Scripts/LipSyncController.cs`  
**Purpose:** Handles lip animation timing

### 5. ElevenLabsAudioManager.cs
**Location:** `Assets/Scripts/ElevenLabsAudioManager.cs`  
**Purpose:** Audio management for voice synthesis

---

## 🔧 BUILD PROCESS

### Quick Build (In Unity)
```
1. Open Unity project: Kelly_Engine_V2/onlykelly/
2. Go to: Kelly Menu → Build → 🚀 Build WebGL (Production)
3. Wait 5-15 minutes
4. Output: Builds/WebGL/
```

### Copy to Web Deployment
```powershell
cd C:\Users\user\UI-TARS-desktop\digital-kelly\engines\Kelly_Engine_V2\onlykelly
.\copy-build-to-web.ps1
```

### Build Output Files
| File | Purpose | Size (Compressed) |
|------|---------|-------------------|
| `Kelly_Web_Build.loader.js` | Unity WebGL loader | ~30 KB |
| `Kelly_Web_Build.framework.js.br` | Unity framework | ~70 KB |
| `Kelly_Web_Build.wasm.br` | WebAssembly | ~7.6 MB |
| `Kelly_Web_Build.data.br` | Asset data | ~227 MB |

---

## 🔄 UPDATE WORKFLOW (Tomorrow)

### Step 1: Receive New Art File
- Expect: `.ccCharacter` or `.fbx` file
- Save to: `arif-deliveries/milestone-2-phase-1/original/`

### Step 2: Verify Requirements
Before importing, confirm:
- [ ] 52+ facial morphs present
- [ ] Separate L/R eye bones (for gaze)
- [ ] Starting pose is correct (director's chair)
- [ ] Hair materials render correctly
- [ ] Proper UV mapping

### Step 3: Import to Unity
```
1. Open Unity: digital-kelly/engines/Kelly_Engine_V2/onlykelly/
2. Delete OLD kelly_fbx_v4.fbx from Assets/Kelly/Animations/Lessons/
3. Import NEW .fbx file to same location
4. RENAME to kelly_fbx_v4.fbx (critical!)
5. Drag into KellyMain.unity scene
6. Verify scripts are attached to new model
```

### Step 4: Verify Script Attachment
**On kelly_fbx_v4 GameObject, ensure these are attached:**
- [ ] KellyWebGLBridge (Script)
- [ ] ARKitBlendshapeController (headRenderer assigned)
- [ ] Animator (if using animations)

### Step 5: Test in Editor
```
1. Press Play
2. Use Inspector to test expressions
3. Check Console for errors
4. Verify blend shapes work
5. Test lip sync simulation
```

### Step 6: Build & Deploy
```
1. Kelly Menu → Build → 🚀 Build WebGL (Production)
2. Run .\copy-build-to-web.ps1
3. Test locally: python -m http.server 8000
4. Navigate to: http://localhost:8000/unity-test.html
```

---

## 📊 CURRENT BUILD STATUS

### Kelly_Web_Build (Local)
| File | Size | Modified |
|------|------|----------|
| Kelly_Web_Build.data.unityweb | 313.12 MB | Nov 28, 2025 |
| Kelly_Web_Build.framework.js.unityweb | 0.07 MB | Nov 28, 2025 |
| Kelly_Web_Build.loader.js | 0.11 MB | Nov 28, 2025 |
| Kelly_Web_Build.wasm.unityweb | 8.76 MB | Nov 28, 2025 |

### Production Build (kelly-live)
| File | Size |
|------|------|
| Kelly_Web_Build.data.br | 227.07 MB |
| Kelly_Web_Build.framework.js.br | 0.07 MB |
| Kelly_Web_Build.wasm.br | 7.58 MB |

---

## 🖼️ REFERENCE ASSETS

### Canonical Character References
**Location:** `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\`

| File | Purpose |
|------|---------|
| `Curious Kelly in final pose in Chair...` | Welcome phase reference |
| `facing to the left.png` | Question phase reference |
| `head and shoulders without chair.png` | Close-up reference |
| `neutral face with hair.png` | Compositing reference |
| `profile of kelly.png` | Side view reference |
| `close up of face.jpeg` | Detail reference |
| `close up of kellys eyes.png` | Eye detail reference |

---

## 🔧 CANONICAL CONFIGURATION SYSTEM

### Single Source of Truth

The `kelly-blendshape-config.ts` file is now the canonical source for:
- All expression definitions with exact blend shape weights
- Phase-to-expression mappings
- Phoneme-to-viseme mappings for lip-sync
- Video prompt keywords for each expression
- Voice settings for ElevenLabs (stability, style per expression)
- Kelly's visual identity (prompts, negative prompts)
- Background descriptions per phase

### How It Works

```
┌────────────────────────────────────────────────────────────────────┐
│  kelly-blendshape-config.ts (Single Source of Truth)              │
│  ────────────────────────────────────────────────────────────────  │
│  • EXPRESSIONS: blend shape weights                                │
│  • PHASE_TO_EXPRESSION: lesson phase mapping                       │
│  • PHONEME_TO_VISEME: lip-sync timing                             │
│  • KELLY_VISUAL_IDENTITY: video generation prompts                 │
└───────────────────────────┬────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Unity WebGL   │   │ Video Pipeline│   │ Lip-Sync      │
│ (via JS API)  │   │ (Sync Labs)   │   │ Engine        │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Editing Expressions

To modify an expression, edit `kelly-blendshape-config.ts`:

```typescript
// Example: Making "happy" even happier
happy: {
  weights: {
    'Mouth_Smile_L': 80,  // was 70
    'Mouth_Smile_R': 80,  // was 70
    'Cheek_Raise_L': 50,  // was 40
    'Cheek_Raise_R': 50,  // was 40
    'Eye_Squint_L': 25,   // was 20
    'Eye_Squint_R': 25,   // was 20
  },
  videoPrompt: 'radiant smile with pure joy, eyes sparkling...',
  voiceTone: 'warm enthusiastic friendly',
  headMotion: 'slight_nod',
  transitionDuration: 0.3,
},
```

Changes automatically propagate to both Unity and video pipeline.

## ⚠️ CRITICAL NOTES

### DO NOT CHANGE:
1. GameObject name `kelly_fbx_v4` (JavaScript targets this)
2. Script method signatures (breaks JS integration)
3. Build output path conventions
4. Blend shape naming if already working
5. **The structure of `kelly-blendshape-config.ts`** (other systems depend on it)

### MUST VERIFY:
1. New FBX has same/compatible blend shape names
2. Materials render correctly in URP
3. Hair transparency works in WebGL
4. Camera framing matches current setup

### IF BLEND SHAPES DON'T MATCH:
- Check `ARKitBlendshapeController` - it has fuzzy matching
- Verify using Inspector → SkinnedMeshRenderer → BlendShapes
- Update `kelly-blendshape-config.ts` with correct names (not the Unity scripts!)
- The config file supports both CC4 naming and alternative names

---

## 🧪 TESTING CHECKLIST (After Update)

### Local Editor Tests
- [ ] Press Play - no console errors
- [ ] Kelly renders correctly (not pink/magenta)
- [ ] Hair is visible (not transparent)
- [ ] Expression changes work via Inspector
- [ ] Idle blink animation runs
- [ ] Mouth opens for simulated lip sync

### WebGL Build Tests
- [ ] Build completes without errors
- [ ] Files appear in Kelly_Web_Build folder
- [ ] Load in browser via http-server
- [ ] Kelly renders in canvas
- [ ] Console shows "Ready for JavaScript commands"
- [ ] Test SendMessage from browser console

### Browser Console Tests
```javascript
// Test after Unity loads
unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', 'happy');
unityInstance.SendMessage('kelly_fbx_v4', 'StartLipSync', 'Testing');
unityInstance.SendMessage('kelly_fbx_v4', 'StopLipSync');
```

---

## 📞 QUICK TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Pink/magenta materials | URP shaders not applied - check material shader |
| Hair invisible | Transparency issue - check shader queue |
| No response to SetExpression | Script not attached or wrong GameObject name |
| Blend shape not found | Check exact name match in SkinnedMeshRenderer |
| Build fails | Check console for specific error |
| WebGL won't load | Check browser console, CORS, MIME types |

---

## 📁 FILE INVENTORY

### Files That Will Change
```
Assets/Kelly/Animations/Lessons/kelly_fbx_v4.fbx  ← REPLACE
Assets/Kelly/Animations/Lessons/kelly_fbx_v4.fbx.meta  ← UPDATE
```

### Files That Should NOT Change
```
Assets/Scripts/KellyWebGLBridge.cs
Assets/Scripts/ARKitBlendshapeController.cs
Assets/Scripts/LipSyncController.cs
Assets/Scripts/ElevenLabsAudioManager.cs
Assets/KellyAvatarController.cs
Assets/KellyMain.unity  ← Only scene references change
```

### Files That Get Regenerated
```
Kelly_Web_Build/Build/*  ← All rebuild
public/unity/kelly-live/Build/*  ← Copy from build
```

---

## ✅ READY STATUS

| Component | Status |
|-----------|--------|
| Unity Project | ✅ Ready |
| Unity Scripts | ✅ Ready |
| Build Pipeline | ✅ Ready |
| Deploy Scripts | ✅ Ready |
| Test Folders | ✅ Ready |
| Documentation | ✅ Ready |
| **Blendshape Config** | ✅ Canonical source created |
| **Lip-Sync Engine** | ✅ Phoneme timing ready |
| **Video Pipeline** | ✅ Wired to canonical config |
| **Intelligent Director** | ✅ **DEPLOYED** - Live on curiouskelly.com |
| **Performance Engine** | ✅ **DEPLOYED** - Orchestrating expressions |
| **Lesson Integration** | ✅ **DEPLOYED** - Auto-directs during lessons |

### New Files Created (December 7, 2025)

| File | Description |
|------|-------------|
| `scripts/kelly-video-factory/kelly-blendshape-config.ts` | Single source of truth for all Kelly expressions, visemes, and prompts |
| `scripts/kelly-video-factory/kelly-lipsync-engine.ts` | Lip-sync timing engine with Unity and video pipeline integration |
| `scripts/kelly-video-factory/kelly-intelligent-director.ts` | Intelligent Director for automatic performance scripting |
| `public/js/kelly-intelligent-director.js` | **LIVE** Browser version of Intelligent Director |
| `public/js/kelly-performance-engine.js` | **LIVE** Real-time performance orchestration |
| `public/js/lesson-director-integration.js` | **LIVE** Wires director into lesson player |

### 🎬 INTELLIGENT DIRECTOR (NEW!)

The Intelligent Director system is now **LIVE** on curiouskelly.com/learn.html:

**What it does:**
1. **Analyzes text in real-time** for emotional content
2. **Automatically switches Kelly's expressions** based on what she's saying
3. **Reacts to user interactions** (correct = celebrating, incorrect = encouraging)
4. **Shows expression badge** in bottom-left corner

**Emotion Detection Patterns:**
- Questions (?) → Curious
- Exclamations (!) → Excited  
- "Because..." → Explaining
- "Remember..." → Wisdom
- "You can do it!" → Encouraging
- "Congratulations!" → Celebrating

**Console Commands:**
```javascript
// Test the director directly
KellyDirector.analyzeAndDirect("Wow, this is amazing!");
// → Sets expression to 'excited' (confidence: 90%)

KellyDirector.reactToUser('correct');
// → Sets expression to 'celebrating'

KellyDirector.directPhase('wisdom', 'Remember this important lesson...');
// → Sets expression based on phase + text analysis
```

### Video Pipeline Integration

The `sota-video-pipeline.ts` now uses the canonical configuration:
- Expression prompts automatically include blend shape info
- Voice settings match expression emotional tone
- Phase backgrounds are centralized
- All expression names stay in sync with Unity

**🟢 YOU ARE READY TO RECEIVE THE NEW ART FILE TOMORROW**

When the new art file arrives:
1. Import to Unity
2. Verify blend shapes match the canonical config
3. Test expressions using browser console
4. Build and deploy

The video pipeline will automatically use the same expressions!

---

*This document auto-generated from Unity project analysis.*  
*Last update: December 7, 2025*  
*Canonical config: `scripts/kelly-video-factory/kelly-blendshape-config.ts`*

