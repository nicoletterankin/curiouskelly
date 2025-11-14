# Follow-Up Email to Arif - Technical Requirements

**Subject:** Kelly Avatar Requirements - CC4→CC5→iClone→Unity Pipeline

---

Hi Arif,

Great meeting with you this morning! As discussed, I wanted to document the specific technical requirements for Kelly's avatar so we're 100% aligned on what we need for our Unity mobile app.

## 🔄 Our Pipeline

**Your Workflow:**
- Character Creator 4 (CC4) + ZBrush

**My Workflow:**
- Character Creator 5 (CC5) + iClone 8 → Export to Unity

**Critical Path:**
```
CC4/ZBrush (you) → CC5 (me) → iClone (me) → Unity FBX Export → Mobile App
```

---

## ✅ What We Need from You (CC4 Export)

### 1. **Character Creator 4 Export Format**

**Primary Deliverable:**
- **CC4 Character File** (.ccCharacter or .iAvatar format)
- Must include ALL of the following:
  - Base mesh (head + neck, optimized)
  - Blendshapes/Morphs (52 minimum - see list below)
  - Eye bones (separate left/right - CRITICAL)
  - Textures (embedded or linked)
  - Rig/skeleton (CC4 standard)

**Export Settings (CC4):**
- File → Export → Character
- Format: .ccCharacter (preferred) or .iAvatar
- Include: Mesh, Skin, Morph Sliders, Skeleton
- Texture Resolution: 2048x2048 max
- **DO NOT** bake morphs - we need them editable in CC5

**Question 1:** Can CC4 export with all morph sliders intact for CC5 import?

---

### 2. **Blendshapes/Morphs (CRITICAL)**

We need **52 facial morphs** matching the **Facial Profile system** in CC4.

**Required Morph Categories:**

**Eyes (10 morphs):**
- Eye Blink L/R
- Eye Look Up L/R
- Eye Look Down L/R
- Eye Look In L/R (cross-eyed)
- Eye Look Out L/R
- Eye Wide L/R (optional but helpful)

**Brows (8 morphs):**
- Brow Inner Up L/R
- Brow Outer Up L/R
- Brow Down L/R
- Brow Squeeze L/R (optional)

**Mouth/Jaw (20+ morphs for lip-sync):**
- Jaw Open, Jaw Forward, Jaw Left, Jaw Right
- Mouth Funnel (ooh/oo sounds)
- Mouth Pucker (kiss/p/b sounds)
- Mouth Smile L/R
- Mouth Frown L/R
- Mouth Left, Mouth Right
- Mouth Roll Upper, Mouth Roll Lower
- Mouth Shrug Upper, Mouth Shrug Lower
- Mouth Close
- Mouth Upper Up L/R
- Mouth Lower Down L/R
- Mouth Press L/R (optional)
- Mouth Stretch L/R (optional)

**Nose (4 morphs - nice to have):**
- Nose Sneer L/R
- Nose Scrunch (optional)

**Cheeks (4 morphs - nice to have):**
- Cheek Puff L/R
- Cheek Squint L/R

**Tongue (if possible):**
- Tongue Out (for "th" sounds)
- Tongue Up (optional)

**Question 2:** Does CC4 Facial Profile give you all these morphs automatically?  
**Question 3:** Can you export the character with these morphs active/editable?

**CRITICAL:** These morphs must survive the CC4→CC5→iClone→Unity pipeline. We'll test this in Milestone 2.

---

### 3. **Eye Bones (CRITICAL - Most Important)**

**This is absolutely critical for our gaze tracking system.**

In CC4, we need:
- **Separate eye bones** for left and right eyes
- Bones must be **independently controllable** (not linked)
- Pivot point at **center of eyeball**
- Eyes must **rotate** (not translate)

**CC4 Eye Setup:**
- Use CC4's eye bone system (not just eye texture)
- Eyes should be separate meshes or submeshes if possible
- Eye rig must export to iClone properly

**Test This:**
When you export from CC4:
1. Import into iClone
2. Go to Edit → Face Puppet
3. Can you control left/right eyes **independently**?
4. Can eyes rotate ±30° horizontally, ±20° vertically?

**Question 4:** Does CC4 automatically create proper eye bones that work in iClone?  
**Question 5:** Have you tested eye bone export from CC4→iClone before?

**This is so important that if you're unsure, let's test this in Milestone 2 FIRST before continuing.**

---

### 4. **Head Model Optimization (Mobile Performance)**

**Target Specs for Mobile (60 FPS on iPhone 12/Pixel 6):**

**Poly Count:**
- **Head + Neck**: 10,000-15,000 triangles MAX
- **Eyes**: 500-1,000 triangles each
- **Mouth Interior**: 1,000 triangles (if visible)
- **Teeth**: 500 triangles (if separate mesh)
- **Total**: ~12,000-18,000 triangles for entire head

**Question 6:** What's your typical CC4 head poly count?  
**Question 7:** Can CC4 export at different LOD levels, or should we optimize in ZBrush first?

**Topology Requirements:**
- Clean quad topology (CC4 should handle this)
- No n-gons
- Good edge loops around mouth and eyes (for deformation)
- Welded vertices (no doubles)

**Body:**
- We **DO NOT** need a full body (we only show head in close-up)
- **Neck is required** (cut off at shoulders)
- If CC4 requires a body for rigging, that's fine, but we'll hide it in Unity

---

### 5. **Textures & Materials**

**Texture Requirements:**
- **Resolution**: 2048x2048 MAX (mobile limitation)
- **Maps Needed**:
  - Diffuse/Base Color (2048x2048)
  - Normal Map (2048x2048)
  - Roughness/Glossiness (1024x1024)
  - (Optional) Metallic, AO, SSS
- **Format**: PNG or TGA
- **Organization**: Clear naming (kelly_diffuse.png, kelly_normal.png, etc.)

**CC4 Materials:**
- Use CC4's PBR materials (they convert to Unity well)
- Keep material count to 1-2 for head
- Avoid complex material setups (subsurface scattering won't work on mobile)

**Question 8:** Can you export textures separately, or do they embed in the .ccCharacter file?

---

### 6. **Hair**

**This is Milestone 3 in your contract.**

**Hair Requirements:**
- Separate mesh from head (easier to modify)
- Optimized poly count: 5,000-8,000 triangles
- Clean topology (not hair cards if possible, but we can work with it)
- Must work in iClone → Unity pipeline

**Question 9:** What's your usual approach for hair in CC4? (Hair cards? Solid mesh? CC4 hair system?)

---

### 7. **Workflow & Testing (Milestone 2)**

**THIS IS THE MOST IMPORTANT MILESTONE.**

After Milestone 1 (base modeling done ✅), Milestone 2 is where we test the **entire pipeline**:

**Testing Checklist for Milestone 2:**
1. ✅ You export from CC4 (.ccCharacter or .iAvatar)
2. ✅ I import into CC5 (does it work?)
3. ✅ I check all morphs/blendshapes (are they all there?)
4. ✅ I export from CC5 to iClone (any issues?)
5. ✅ I test eye bones in iClone (can I control them independently?)
6. ✅ I export FBX from iClone with blendshapes
7. ✅ I import FBX into Unity
8. ✅ I test blendshapes in Unity (do they work?)
9. ✅ I test eye bones in Unity (gaze tracking functional?)
10. ✅ I test performance (60 FPS?)

**If ANY step fails, we need to fix it before Milestone 3/4.**

**Question 10:** Are you available to do quick iterations if we find issues in Milestone 2 testing?

---

## 🎯 Final Unity Requirements (For Your Reference)

This is what we need in Unity after the full pipeline:

**FBX Export from iClone:**
- Format: FBX 2020
- Units: Centimeters
- Up Axis: Y-up, Forward: Z-forward
- Include: Mesh + Blendshapes + Eye Bones
- Frame Rate: Not needed (static model, we animate in Unity)

**Unity Import:**
- 52 blendshapes accessible
- Eye bones in hierarchy (LeftEye_Bone, RightEye_Bone)
- Materials: URP (Universal Render Pipeline) - we'll convert
- Performance: 60 FPS on mobile

---

## 📋 Pipeline Testing Plan (Milestone 2)

**Step 1: You Export from CC4**
- Send me: .ccCharacter or .iAvatar file
- Send me: Textures (if not embedded)
- Send me: Screenshots of CC4 morph list

**Step 2: I Import to CC5**
- I verify morphs survived
- I check eye bones work
- I test basic animations

**Step 3: I Export to iClone**
- I test Face Puppet (eye control)
- I test morph sliders
- I test basic expressions

**Step 4: I Export to Unity**
- I import FBX
- I test blendshapes
- I test eye bones
- I test performance (FPS)

**Step 5: Feedback Loop**
- I report results to you
- We fix any issues in CC4
- We re-test until perfect

**Timeline for Milestone 2:** Should take 2-3 days of back-and-forth testing.

---

## ❓ Questions Summary

Please answer these so I can prepare for Milestone 2:

1. Can CC4 export .ccCharacter with all morph sliders intact for CC5?
2. Does CC4 Facial Profile give you all 52 morphs automatically?
3. Can you export the character with morphs active/editable?
4. Does CC4 automatically create proper eye bones that work in iClone?
5. Have you tested eye bone export from CC4→iClone before?
6. What's your typical CC4 head poly count?
7. Can CC4 export at different LOD levels?
8. Can you export textures separately from .ccCharacter file?
9. What's your usual approach for hair in CC4?
10. Are you available for quick iterations during Milestone 2 testing?

---

## 🚨 Critical Items (Must Verify in Milestone 2)

**Before we continue to Milestone 3 & 4, we MUST verify:**

1. ✅ **Eye bones work** through CC4→CC5→iClone→Unity pipeline
2. ✅ **All 52 morphs** survive the pipeline and work in Unity
3. ✅ **Poly count** is optimized for 60 FPS mobile
4. ✅ **Textures** import correctly
5. ✅ **File format** (.ccCharacter) works with CC5

**If any of these fail, we need to solve them before proceeding.**

---

## 📁 File Delivery Format

**For Milestone 2, please send:**
```
Kelly_Avatar_v1/
├── Kelly_Base.ccCharacter (or .iAvatar)
├── Textures/
│   ├── Kelly_Diffuse_2048.png
│   ├── Kelly_Normal_2048.png
│   ├── Kelly_Roughness_1024.png
│   └── (any other maps)
├── Screenshots/
│   ├── CC4_Morph_List.png
│   ├── CC4_Eye_Bones.png
│   └── CC4_Topology.png
└── Notes.txt (any special instructions)
```

---

## 🎯 Success Criteria

**We'll know Milestone 2 is successful when:**
- ✅ I can import your CC4 file into CC5 with no errors
- ✅ All morphs are present and functional
- ✅ Eye bones work independently in iClone
- ✅ FBX exports from iClone with blendshapes intact
- ✅ Unity import works with 60 FPS performance
- ✅ Gaze tracking works (eyes move independently)
- ✅ Lip-sync works (mouth morphs respond correctly)

**Once these are verified, we can confidently proceed to Milestones 3 & 4!**

---

## 📞 Next Steps

1. Please review this email and answer the 10 questions above
2. Let me know if anything is unclear or needs clarification
3. Once you're ready, export the base model from CC4 (Milestone 1 complete ✅)
4. Send me the files for Milestone 2 testing
5. I'll test the full pipeline and give you feedback within 24 hours

Looking forward to testing the pipeline with you! The base model looks great from the images you shared.

Best regards,  
[Your Name]

---

**P.S.** - The eye bones are THE most critical part. If you're unsure about anything related to eye bone setup in CC4, please let me know ASAP so we can research it together before Milestone 2. This is the foundation of our gaze tracking system and we can't proceed without it working perfectly.

---

**P.P.S.** - I know this email is long, but Milestone 2 is where we "test the bridge" between our workflows. Better to catch any issues early than discover them in Milestone 4!



