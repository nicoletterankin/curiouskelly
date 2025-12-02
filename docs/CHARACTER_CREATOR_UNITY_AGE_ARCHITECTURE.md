# Character Creator → Unity WebGL Age Morphing Architecture

**Document Owner:** Technical Architecture  
**Created:** November 25, 2025  
**Status:** Architectural Guide for CC5 Expert

---

## EXECUTIVE SUMMARY

This document provides architectural guidance for exporting Kelly from Character Creator 5 (CC5) to Unity WebGL, with a phased approach:

- **LAUNCH (Dec 17):** Single Kelly model at 27 years old, no visible age morphing
- **POST-LAUNCH:** Full age morphing system (2-102 years) using blend shapes

---

## 1. LAUNCH SCOPE (December 17, 2025)

### 1.1 Single Model Strategy

For launch, export **ONE optimized Kelly model at 27 years old**.

#### Export Format: **GLB (Recommended)**

| Format | Pros | Cons | Verdict |
|--------|------|------|---------|
| **GLB** | Single file, binary, compressed, web-native | Less Unity tooling | ✅ **BEST for WebGL** |
| **glTF** | JSON-based, human readable | Multiple files | Good alternative |
| **FBX** | Unity native support | Larger files, not web-optimized | ❌ For WebGL |

**Recommendation:** Export as GLB for smallest file size and fastest web loading.

#### CC5 Export Settings for Launch

```
Export Path: C:\iLearnStudio\exports\Kelly\kelly-27-launch.glb

Settings:
├── Format: glTF/GLB
├── Mesh Quality: Medium (for WebGL performance)
├── Texture Resolution: 2K (4096 is too large for WebGL)
├── Include:
│   ├── ✅ Skeleton/Armature
│   ├── ✅ Blend Shapes (expressions only)
│   ├── ✅ Materials (PBR)
│   ├── ✅ UV Maps
│   └── ❌ LODs (manual optimization later)
└── Compression:
    ├── Mesh: Draco compression enabled
    └── Textures: WebP/JPEG (no PNG for diffuse)
```

### 1.2 Required Expression Blend Shapes (Launch)

Export Kelly with these 15 essential expression blend shapes for Unity:

| Category | Blend Shape Name | Purpose |
|----------|------------------|---------|
| **Smile** | `smile_left` | Left corner up |
| **Smile** | `smile_right` | Right corner up |
| **Smile** | `smile_open` | Open mouth smile |
| **Eyebrows** | `eyebrow_raise_left` | Left brow up |
| **Eyebrows** | `eyebrow_raise_right` | Right brow up |
| **Eyebrows** | `eyebrow_furrow` | Brows together (concerned) |
| **Eyes** | `eyes_wide` | Surprise/interest |
| **Eyes** | `eyes_squint` | Thinking/skeptical |
| **Eyes** | `blink_left` | Left eye blink |
| **Eyes** | `blink_right` | Right eye blink |
| **Mouth** | `mouth_open` | Jaw drop for speech |
| **Mouth** | `lips_pursed` | Thinking expression |
| **Mouth** | `mouth_frown` | Sad/concerned |
| **Head** | `head_tilt_left` | Head tilt left |
| **Head** | `head_tilt_right` | Head tilt right |

**ARKit Compatibility:** If possible, use ARKit-compatible naming (52 blend shapes) for future iOS/face tracking integration.

### 1.3 Age Simulation for Launch (Voice + Expression Only)

Since no visible age morphing at launch, simulate age perception through:

| Age Bucket | Voice Pitch | Expression Style | Animation Speed |
|------------|-------------|------------------|-----------------|
| 2-5 | 1.2x higher | +40% intensity, bouncy | 1.1x faster |
| 6-12 | 1.1x higher | +25% intensity, curious | 1.05x faster |
| 13-17 | Normal | -10% intensity, restrained | Normal |
| 18-35 | 0.95x | Balanced, natural | 0.98x |
| 36-60 | 0.9x | -15% intensity, confident | 0.95x |
| 61-102 | 0.85x lower | -25% intensity, gentle | 0.9x slower |

**Unity Implementation:**
```javascript
// Web-side pitch/expression adjustment
const AGE_PARAMS = {
  '18-35': { pitch: 1.0, expressionMultiplier: 1.0, animationSpeed: 1.0 }
};

// At launch, only one bucket is "real" (27 = 18-35)
// Other buckets apply pitch + expression modifiers to same model
```

### 1.4 File Size Budget (Launch)

**Target:** < 10MB total GLB file

| Component | Budget |
|-----------|--------|
| Mesh (Kelly body) | 2-3 MB |
| Skeleton | 100 KB |
| Blend shapes (15) | 500 KB |
| Textures (2K, compressed) | 4-5 MB |
| Materials | 100 KB |
| **Total** | **7-9 MB** |

---

## 2. POST-LAUNCH: AGE MORPHING ARCHITECTURE

### 2.1 Architectural Decision: Blend Shapes vs. Separate Models

**RECOMMENDATION: Hybrid Approach**

| Approach | When to Use | Why |
|----------|-------------|-----|
| **Blend Shapes** | Adult ages (18-102) | Smooth interpolation, small file delta |
| **Separate Models** | Child ages (2-12) | Body proportions too different for morphs |

#### Rationale:

**Body Proportion Changes by Age:**

```
Ages 2-5:   Head/body ratio ~1:4, short limbs, round features
Ages 6-12:  Head/body ratio ~1:5, still proportionally different
Ages 13-17: Nearly adult proportions, mostly face changes
Ages 18+:   Standard adult proportions, face aging only
```

**Blend shapes can't handle:**
- Skeletal length changes (arms, legs, torso)
- Head-to-body ratio changes
- Major topology changes

**Solution:**
- **3 base models:** Child (2-12), Teen/Adult (13-60), Elder (61-102)
- **Blend shapes within each model** for age progression within that range

### 2.2 Character Creator Export Strategy (Post-Launch)

#### Option A: Create 3 Base Characters + Age Blend Shapes (RECOMMENDED)

```
Export Structure:
├── kelly-child-base.glb      (ages 2-12)
│   ├── Blend shapes: age_2, age_6, age_10, age_12
│   └── Body: child proportions
├── kelly-adult-base.glb      (ages 13-60)
│   ├── Blend shapes: age_18, age_27, age_40, age_55
│   └── Body: adult proportions
└── kelly-elder-base.glb      (ages 61-102)
    ├── Blend shapes: age_65, age_80, age_95
    └── Body: elder proportions (slight shrinkage, posture)
```

**File Size Impact:** ~25-30 MB total (3 models)

#### Option B: 6 Separate Character Files (Simpler, Larger Files)

```
Export Structure:
├── kelly-age-2-5.glb        (~8 MB)
├── kelly-age-6-12.glb       (~8 MB)
├── kelly-age-13-17.glb      (~9 MB)
├── kelly-age-18-35.glb      (~9 MB) ← Launch model
├── kelly-age-36-60.glb      (~9 MB)
└── kelly-age-61-102.glb     (~9 MB)
```

**File Size Impact:** ~52 MB total
**Pro:** No blend shape complexity
**Con:** Larger total size, no smooth age interpolation within buckets

### 2.3 ActorMixer Strategy

**Can ActorMixer morphs export to Unity?** 

**Answer:** Not directly. Here's the workflow:

```
ActorMixer Morphs (CC5)
       │
       ▼
  Bake to Character (CC5)
       │
       ▼
  Export as Blend Shapes (GLB/FBX)
       │
       ▼
  Unity imports blend shapes
```

#### CC5 → Unity ActorMixer Workflow:

1. **In CC5 with ActorMixer:**
   - Create age variants using ActorMixer sliders
   - **SAVE each variant as a morph target**, not a separate character
   
2. **Bake to Blend Shapes:**
   - ActorMixer changes become blend shape targets
   - Export the base character with all age morphs as blend shapes

3. **Unity Side:**
   - Import GLB with blend shapes
   - Drive `age_young` (0-100) and `age_old` (0-100) values from age slider

#### Creating Age Blend Shapes Manually (if ActorMixer export fails):

1. **Create Kelly at 27 (base)**
2. **Duplicate project, age to ~12 (young morph)**
   - Adjust: jawline, nose size, eye size, lip fullness, skin smoothness
3. **Duplicate project, age to ~75 (old morph)**
   - Adjust: wrinkles, skin sag, lip thinning, jowls, eye bags
4. **Export all 3 as separate meshes**
5. **Use Unity's "Extract Blend Shapes from mesh" tool**

### 2.4 Unity WebGL Import Pipeline

```
Unity Project Structure:
Assets/
├── Characters/
│   ├── Kelly/
│   │   ├── Models/
│   │   │   ├── kelly-adult-base.glb
│   │   │   ├── kelly-child-base.glb      (post-launch)
│   │   │   └── kelly-elder-base.glb      (post-launch)
│   │   ├── Materials/
│   │   │   ├── KellySkin.mat
│   │   │   ├── KellyEyes.mat
│   │   │   └── KellyHair.mat
│   │   ├── Animations/
│   │   │   ├── Idle.anim
│   │   │   ├── Talking.anim
│   │   │   └── Gestures/
│   │   └── Scripts/
│   │       ├── KellyBlendShapeController.cs
│   │       └── KellyAgeManager.cs
│   └── Shared/
│       └── Shaders/
│           └── WebGLOptimized/
```

### 2.5 Unity Blend Shape Controller

```csharp
// KellyBlendShapeController.cs
using UnityEngine;

public class KellyBlendShapeController : MonoBehaviour
{
    [SerializeField] private SkinnedMeshRenderer headRenderer;
    
    // Expression blend shapes (launch)
    private int smileIndex, eyebrowRaiseIndex, eyesWideIndex;
    
    // Age blend shapes (post-launch)
    private int ageYoungIndex, ageOldIndex;
    
    void Start()
    {
        // Cache blend shape indices for performance
        smileIndex = headRenderer.sharedMesh.GetBlendShapeIndex("smile");
        eyebrowRaiseIndex = headRenderer.sharedMesh.GetBlendShapeIndex("eyebrow_raise");
        ageYoungIndex = headRenderer.sharedMesh.GetBlendShapeIndex("age_young");
        ageOldIndex = headRenderer.sharedMesh.GetBlendShapeIndex("age_old");
    }
    
    // Called from JavaScript bridge
    public void SetExpression(string expressionName, float value)
    {
        int index = headRenderer.sharedMesh.GetBlendShapeIndex(expressionName);
        if (index >= 0)
        {
            headRenderer.SetBlendShapeWeight(index, value);
        }
    }
    
    // Age morphing (post-launch)
    public void SetAge(int age)
    {
        // Linear interpolation: 27 = baseline
        // < 27 drives age_young, > 27 drives age_old
        if (age < 27)
        {
            float youngAmount = Mathf.InverseLerp(27, 2, age) * 100f;
            headRenderer.SetBlendShapeWeight(ageYoungIndex, youngAmount);
            headRenderer.SetBlendShapeWeight(ageOldIndex, 0);
        }
        else
        {
            float oldAmount = Mathf.InverseLerp(27, 102, age) * 100f;
            headRenderer.SetBlendShapeWeight(ageYoungIndex, 0);
            headRenderer.SetBlendShapeWeight(ageOldIndex, oldAmount);
        }
    }
}
```

---

## 3. DETAILED CC5 EXPORT WORKFLOW

### 3.1 Launch Export Checklist (Dec 17)

```
PRE-EXPORT CHECKLIST:
□ Kelly loaded in CC5 at age 27 appearance
□ Hair applied and styled
□ Eyes and lashes configured
□ Skin materials set (SSS, roughness, etc.)
□ All 15 expression blend shapes created/verified

EXPORT STEPS:
1. File → Export → FBX/glTF Options
2. Select: GLB format (binary glTF)
3. Configure:
   □ Mesh: Include → Check all relevant meshes
   □ Skeleton: Include → ARKit-compatible if available
   □ Blend Shapes: Include → All facial expressions
   □ Textures: Export embedded (into GLB)
   □ Resolution: 2048x2048 (2K)
   □ Compression: Enable Draco mesh compression
4. Export to: C:\iLearnStudio\exports\Kelly\kelly-27-launch.glb
5. Verify file size < 10MB

POST-EXPORT:
□ Open GLB in Blender or gltf-viewer to verify
□ Check blend shapes work
□ Verify texture quality acceptable
□ Test in Unity WebGL build
```

### 3.2 Texture Optimization for WebGL

| Texture Type | Source Resolution | Export Resolution | Format |
|--------------|-------------------|-------------------|--------|
| Diffuse/Albedo | 8K | **2K** | JPEG (quality 85) |
| Normal Map | 8K | **2K** | PNG (lossless) |
| Roughness | 4K | **1K** | JPEG (quality 90) |
| Ambient Occlusion | 4K | **1K** | JPEG (quality 90) |
| Specular | 4K | **1K** | JPEG (quality 90) |

**WebGL Texture Limits:**
- Max texture size: 4096x4096 (many devices)
- Recommended: 2048x2048 for character textures
- VRAM budget: ~50MB for entire scene

### 3.3 Mesh Optimization

**Polygon Budget (WebGL):**

| Component | Max Triangles | Notes |
|-----------|---------------|-------|
| Head/Face | 15,000-20,000 | Highest detail for expressions |
| Body | 10,000-15,000 | Medium detail |
| Hair | 5,000-10,000 | Cards/planes, not strands |
| Eyes | 2,000 | Separate mesh recommended |
| **Total** | **35,000-50,000** | Target for WebGL |

**Optimization Steps in CC5:**
1. Use LOD settings if available
2. Reduce subdivision level before export
3. Merge materials where possible
4. Use hair cards instead of strand hair

---

## 4. POST-LAUNCH ROADMAP

### Phase 1: Adult Age Range (Q1 2026)

```
Goal: Smooth morphing within 18-60 range

Tasks:
1. Create Kelly at ages 18, 40, 55 in CC5
2. Export as blend shape targets
3. Implement age slider interpolation in Unity
4. Test smooth age transitions

Deliverables:
- kelly-adult-base.glb with age blend shapes
- Unity age morphing controller
- QA validation across age range
```

### Phase 2: Elder Range (Q1 2026)

```
Goal: Kelly 61-102 with graceful aging

Tasks:
1. Create elder Kelly variants (65, 80, 95)
2. Add wrinkle maps and skin adjustments
3. Posture adjustments (slight stoop, etc.)
4. Export and integrate

Considerations:
- May need separate model due to posture changes
- Hair color/style changes (gray, shorter)
- Facial sagging requires careful blend shapes
```

### Phase 3: Child/Teen Range (Q2 2026)

```
Goal: Kelly 2-17 (most complex due to body proportions)

Tasks:
1. Create child Kelly (body proportions 2-5, 6-12)
2. Create teen Kelly (13-17)
3. Model swap logic in Unity
4. Transition animations between models

Technical Challenges:
- Different skeleton proportions
- Clothing/outfit adjustments
- Voice and personality matching
- Age-appropriate expressions
```

---

## 5. UNITY WEBGL PERFORMANCE TIPS

### 5.1 Model Loading Strategy

```javascript
// From unity-asset-manager.js (existing in codebase)

const MODEL_PRELOAD_STRATEGY = {
  // Launch: single model, always loaded
  'launch': {
    preload: ['kelly-27-launch.glb'],
    onDemand: []
  },
  
  // Post-launch: preload likely ages, lazy-load extremes
  'post-launch': {
    preload: ['kelly-adult-base.glb'],  // 18-60, most common
    onDemand: ['kelly-child-base.glb', 'kelly-elder-base.glb']
  }
};
```

### 5.2 Blend Shape Performance

**WebGL Blend Shape Limits:**
- Active blend shapes per mesh: ~8-12 for smooth performance
- Total blend shapes: 50+ is fine, just don't animate all simultaneously

**Optimization:**
```csharp
// Only update blend shapes when values actually change
private float lastSmileValue = -1;

public void SetSmile(float value)
{
    if (Mathf.Abs(value - lastSmileValue) > 0.01f)
    {
        headRenderer.SetBlendShapeWeight(smileIndex, value);
        lastSmileValue = value;
    }
}
```

### 5.3 Memory Management

```csharp
// Unload unused models when switching age ranges
public void SwitchAgeModel(string newModel)
{
    if (currentModelName != newModel)
    {
        // Destroy old model
        if (currentModel != null)
        {
            Destroy(currentModel);
            Resources.UnloadUnusedAssets();
        }
        
        // Load new model
        currentModel = Instantiate(Resources.Load<GameObject>(newModel));
        currentModelName = newModel;
    }
}
```

---

## 6. INTEGRATION WITH EXISTING SYSTEMS

### 6.1 Unity Bridge Events (from UNITY_INTEGRATION_PLAN.md)

```javascript
// Existing events to extend for age morphing:

// Web → Unity
'age-changed': { age: 35, ageBucket: '18-35', sessionId }
'character-load': { modelUrl: '/unity/character-models/age-18-35.glb', ... }

// Unity → Web
'character-loaded': { modelUrl, ageBucket }
'morph-applied': { age, blendShapeValues }  // NEW for post-launch
```

### 6.2 Expression System Integration (from EXPRESSION_SYSTEM.md)

The existing expression generator outputs blend shape values:

```javascript
// Existing format works with this architecture
{
  expressions: [
    {
      timestamp: 0.0,
      blendShapes: {
        smile: 65,
        eyebrowRaise: 35,
        eyesWide: 25
      }
    }
  ]
}
```

**Post-Launch Addition:**
```javascript
{
  // Add age-adjusted intensities
  ageAdjustedBlendShapes: {
    smile: 65 * 1.25,  // For age 6-12 (+25% intensity)
    eyebrowRaise: 35 * 1.25,
    eyesWide: 25 * 1.25
  }
}
```

---

## 7. CC5 EXPERT INSTRUCTIONS SUMMARY

### For December 17 Launch:

1. **Open** `Kelly_8K_Production.ccProject` in CC5
2. **Set appearance** to 27 years old (current baseline)
3. **Verify** all 15 expression blend shapes exist
4. **Export** as GLB with settings:
   - Format: GLB (binary glTF)
   - Textures: 2K, embedded, compressed
   - Mesh: Draco compression
   - Blend shapes: Include all facial expressions
5. **Target file size:** < 10MB
6. **Output:** `kelly-27-launch.glb`

### Post-Launch Preparation:

1. **Create age variants** using ActorMixer or manual morphing
2. **Save each age** as a morph target in CC5
3. **Export** with age blend shapes embedded
4. **Test** age slider drives blend shapes correctly

---

## 8. QUESTIONS FOR CC5 EXPERT

Please clarify with your CC5 expert:

1. **ActorMixer → Blend Shape Export:**
   - Can ActorMixer morphs be baked to blend shapes during GLB export?
   - Or must we create separate character files and extract blend shapes in Unity/Blender?

2. **Headshot 2 Age Variants:**
   - Can Headshot 2 generate age variants of existing Kelly character?
   - Or must age morphing be done manually with morphs?

3. **Child Body Proportions:**
   - Does CC5 have child body presets that maintain Kelly's identity?
   - Or do we need to create a separate "Kelly as a child" character?

4. **GLB Export Quality:**
   - What's the maximum blend shape count CC5 can export to GLB?
   - Any known issues with GLB export from CC5?

---

## APPENDIX A: File Naming Convention

```
kelly-{age}-{variant}.glb

Examples:
kelly-27-launch.glb         Launch model
kelly-adult-base.glb        Adult with age blend shapes
kelly-child-base.glb        Child with age blend shapes
kelly-elder-base.glb        Elder with age blend shapes
```

## APPENDIX B: Blend Shape Naming Convention

```
Expression Shapes:
- smile_left, smile_right, smile_open
- eyebrow_raise_left, eyebrow_raise_right, eyebrow_furrow
- eyes_wide, eyes_squint, blink_left, blink_right
- mouth_open, lips_pursed, mouth_frown
- head_tilt_left, head_tilt_right

Age Shapes (post-launch):
- age_young (drives toward younger appearance)
- age_old (drives toward older appearance)
- wrinkles_forehead, wrinkles_eyes, wrinkles_mouth (detail)
- skin_sag_cheeks, skin_sag_neck (detail)
```

---

**Document End**

*This document should be reviewed with your Character Creator 5 expert to validate export capabilities and identify any CC5-specific limitations.*










