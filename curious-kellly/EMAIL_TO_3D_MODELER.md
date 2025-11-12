# Email to 3D Modeler - Technical Requirements

**Subject**: Kelly Avatar Technical Requirements - Unity Mobile (60 FPS Target)

---

## Email Draft

**To**: [3D Modeler Name]  
**From**: Curious Kelly Development Team  
**Subject**: Kelly Avatar Technical Requirements - Unity Mobile (60 FPS Target)  
**Priority**: High

---

Hi [Name],

We're building the Kelly avatar system for our Curious Kelly learning app, and I wanted to share the technical requirements for the 3D model. We've just completed our Week 3 avatar upgrade with a focus on 60 FPS mobile performance, natural gaze tracking, and real-time lip-sync.

## 📋 Technical Requirements

### 1. **Blendshapes (Critical)**

We need **ARKit-standard blendshapes** (52 shapes) or **Audio2Face-compatible** naming. Our system uses these for:
- Real-time lip-sync (viseme-driven speech)
- Micro-expressions (teaching moments)
- Natural facial animation

**Required Blendshapes** (minimum set):

**Eyes & Brows:**
- `eyeBlinkLeft`, `eyeBlinkRight`
- `eyeLookUpLeft`, `eyeLookUpRight`, `eyeLookDownLeft`, `eyeLookDownRight`
- `browInnerUp`, `browOuterUpLeft`, `browOuterUpRight`
- `browDownLeft`, `browDownRight`

**Mouth (Lip-Sync):**
- `jawOpen`, `jawForward`, `jawLeft`, `jawRight`
- `mouthFunnel`, `mouthPucker`, `mouthSmile`, `mouthFrown`
- `mouthLeft`, `mouthRight`
- `mouthRollUpper`, `mouthRollLower`
- `mouthShrugUpper`, `mouthShrugLower`
- `mouthClose`, `mouthUpperUpLeft`, `mouthUpperUpRight`
- `mouthLowerDownLeft`, `mouthLowerDownRight`

**Additional (Nice to Have):**
- `tongueOut` (for "th" sounds)
- `breathMicro` (subtle breathing)
- `noseSneerLeft`, `noseSneerRight` (expressions)

**Question 1**: Which blendshape standard are you currently using? (ARKit / Audio2Face / Custom)  
**Question 2**: Can you provide a full list of available blendshapes in your current model?

---

### 2. **Eye Bone Hierarchy (Critical for Gaze Tracking)**

Our gaze controller needs **separate eye bones** for realistic eye movement with micro-saccades (2-4/sec).

**Required Hierarchy:**
```
Kelly_Head (root)
├── LeftEye_Bone
│   └── LeftEye_Mesh (optional child)
└── RightEye_Bone
    └── RightEye_Mesh (optional child)
```

**Eye Bone Requirements:**
- Pivot at center of eyeball
- Forward axis aligned with default gaze direction
- Separate bones for left and right eyes
- Rotation range: ±30° horizontal, ±20° vertical

**Question 3**: Do you currently have separate eye bones in the rig?  
**Question 4**: If not, can you add them? (This is critical for our gaze system)

---

### 3. **Mobile Performance Optimization**

We're targeting **60 FPS on iPhone 12+ and Pixel 6+**. Performance is critical.

**Target Specifications:**
- **Poly Count**: 10,000-15,000 triangles (head only)
- **Vertex Count**: 5,000-7,500 vertices
- **Blendshape Count**: 52 shapes (ARKit standard)
- **Texture Resolution**: 2048x2048 max (diffuse + normal)
- **Material Count**: 1-2 materials max

**Optimization Requirements:**
- Clean topology (quads preferred, triangulated on export)
- No n-gons
- Welded vertices (no doubles)
- UV mapping optimized (minimal seams on face)
- LOD levels NOT needed (we use single close-up view)

**Question 5**: What's the current poly count of Kelly's head model?  
**Question 6**: Can you optimize to our target if it's higher?

---

### 4. **Materials & Textures**

We're using **Unity URP (Universal Render Pipeline)** for mobile.

**Material Requirements:**
- **Shader**: URP/Lit (standard Unity shader)
- **Textures Needed**:
  - Base Color (Albedo) - 2048x2048
  - Normal Map - 2048x2048
  - Metallic/Smoothness - 1024x1024 (optional)
- **Format**: PNG or TGA (uncompressed)
- **Color Space**: sRGB for albedo, Linear for normal maps

**Lighting Setup:**
- We use soft directional lighting
- Simple studio setup (no complex environment)
- Subsurface scattering NOT needed (mobile limitation)

**Question 7**: What material/texture format are you currently working in?  
**Question 8**: Can you provide URP-compatible materials, or should we convert them?

---

### 5. **FBX Export Settings**

**Required Export Settings:**
- **Format**: FBX 2020 (ASCII preferred for debugging)
- **Units**: Centimeters (Unity default)
- **Up Axis**: Y-up (Unity standard)
- **Forward Axis**: Z-forward
- **Bake Animation**: NO (we handle animation in Unity)
- **Include**: Mesh + Bones + Blendshapes
- **Smoothing Groups**: Export

**Embedded vs. External Textures:**
- **Preferred**: External textures (separate files)
- **Fallback**: Embedded OK, but we'll extract them

**Question 9**: Can you provide FBX with these export settings?

---

### 6. **Model Composition**

**What We Need:**
- Head mesh (with neck)
- Eye bones (left + right)
- Blendshapes (52 ARKit or A2F compatible)
- Textures (base color + normal)

**What We DON'T Need:**
- Body (we only show head close-up)
- Hair (if complex, consider as separate mesh)
- Accessories (earrings, etc. - can be separate)
- Tongue mesh (unless you want it visible)
- Teeth mesh (can be part of head or separate)

**Camera View:**
- Tight head close-up (like video call framing)
- FOV: 38° (portrait mode)
- Distance: ~1.5-2 meters equivalent

**Question 10**: Is the current model head-only, or does it include a full body?

---

### 7. **Testing & Validation**

Once you provide the model, we'll test:
1. ✅ Import into Unity (FBX compatibility)
2. ✅ Blendshape mapping (ARKit/A2F naming)
3. ✅ Eye bone setup (gaze tracking)
4. ✅ Performance (60 FPS on target devices)
5. ✅ Visual quality (lighting, materials)
6. ✅ Lip-sync accuracy (<5% error target)

**We'll provide feedback within 24 hours of receiving the model.**

---

## 🎯 Priority Items

**Critical (Must Have):**
1. ✅ ARKit or Audio2Face blendshapes (52 shapes minimum)
2. ✅ Separate eye bones (left + right)
3. ✅ Mobile-optimized poly count (10k-15k tris)
4. ✅ Clean FBX export (Unity-compatible)

**Important (Should Have):**
1. ✅ URP-compatible materials
2. ✅ 2048x2048 textures (base + normal)
3. ✅ Proper UV mapping (minimal seams)

**Nice to Have (Optional):**
1. ⭐ Tongue mesh (for "th" sounds)
2. ⭐ Separate teeth mesh
3. ⭐ Hair as separate mesh (easier to modify)

---

## 📅 Timeline

**When do you need this?**
- **Preferred**: Within 1-2 weeks
- **Critical Path**: We're currently in Week 3 of a 12-week development cycle
- **Testing Phase**: We need to test on 7 devices (iPhone 12-15, Pixel 6-8)

**Question 11**: What's your estimated timeline for delivery?  
**Question 12**: Can you provide a preview/WIP version for early testing?

---

## 📚 Reference Materials

**Example Blendshapes:**
- ARKit Standard: https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
- Audio2Face: NVIDIA documentation

**Technical Specs:**
- Unity Version: 2022.3 LTS
- Render Pipeline: URP (Universal Render Pipeline)
- Target Platforms: iOS 15+ and Android 12+
- Target Devices: iPhone 12-15, Pixel 6-8

**Similar Projects:**
- Ready Player Me (for reference on optimization)
- Live Link Face (ARKit blendshape usage)

---

## 🤝 Collaboration

**How We Can Help:**
- We can provide technical feedback on WIP models
- We can test early versions in Unity
- We can help with blendshape naming/mapping
- We have Audio2Face setup if you need lip-sync testing

**What We Need from You:**
- Current model status (WIP screenshots/videos)
- Blendshape list (which ones are implemented)
- Timeline estimate
- Any technical constraints or questions

---

## 📞 Questions Summary

1. Which blendshape standard are you using? (ARKit / Audio2Face / Custom)
2. Can you provide a full list of available blendshapes?
3. Do you have separate eye bones in the rig?
4. If not, can you add eye bones? (Critical for gaze tracking)
5. What's the current poly count?
6. Can you optimize to 10k-15k triangles if needed?
7. What material/texture format are you working in?
8. Can you provide URP-compatible materials?
9. Can you export FBX with our settings (Y-up, Z-forward, cm)?
10. Is the model head-only or full body?
11. What's your estimated delivery timeline?
12. Can you provide a WIP version for early testing?

---

## 🎯 Next Steps

**Please Reply With:**
1. Answers to the questions above
2. Current model status/screenshots
3. Timeline estimate
4. Any questions or concerns you have

**We'll Then:**
1. Review your responses
2. Provide any additional clarification
3. Set up a testing workflow
4. Establish delivery milestones

---

Looking forward to collaborating with you on this! Kelly is going to be amazing. 🎨

If you have any questions about our technical requirements or need clarification on anything, please don't hesitate to ask.

Best regards,  
**Curious Kelly Development Team**

---

P.S. If you want to see the avatar system in action, we can schedule a quick call to demo our Week 3 gaze tracking and expression system. It might help clarify what we're building!

---

## Attachments (To Include):

1. **`blendshape_reference.png`** - ARKit blendshape chart
2. **`eye_bone_hierarchy.png`** - Example bone setup
3. **`camera_framing.png`** - Our desired head close-up view
4. **`TECHNICAL_ALIGNMENT_MATRIX.md`** - Full technical specs

---

**Email End**


