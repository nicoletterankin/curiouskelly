# Email to Arif - Short Version (Post-Meeting)

**Subject:** Kelly Avatar - Phase 1 Requirements (Face-Focused, Ship Fast!)

---

Hi Arif,

Great meeting this morning! Here's our plan to get Kelly teaching quickly:

---

## 🎯 **Phased Rollout Strategy**

We're building Kelly in **4 phases** (start small, add capabilities as we prove the system):

**Phase 1 (NOW - Milestone 2):** Face-focused teaching → Ship THIS MONTH  
**Phase 2 (Milestone 3):** Add arms & gestures → More engaging  
**Phase 3 (Milestone 4):** Full body & poses → Rich variety  
**Phase 4 (Future):** Full animation → Interactive magic

---

## 📋 **Phase 1: What You Need to Build**

### **Scope:**
```
    👤 Kelly
   ┌─────┐
   │ 😊  │  ← Head (full detail)
   ├─────┤
   │ 👚  │  ← Shoulders + upper chest
   └─────┘
═════════  ← Cut here (mid-torso)
```

**Camera:** Tight close-up (like Zoom)  
**Poly Count:** 15,000-20,000 triangles  
**Why:** Get Kelly teaching ASAP, focus on face/eyes

---

### **Critical Requirements:**

**1. Eye Bones (MOST IMPORTANT) 🚨**
- Separate left/right eye bones (independent control)
- Must work in iClone Face Puppet
- **Test this before sending!**

**2. 52 Facial Morphs**
- Use CC4 Facial Profile (should be automatic)
- Jaw, mouth, eyes, brows (for lip-sync + expressions)
- Must NOT be baked (keep editable)

**3. Export Format**
- .ccCharacter (or .iAvatar)
- Include: Mesh + ALL morph sliders + eye bones + textures
- Texture: 2048x2048

---

## ❓ **Key Question: Build Approach**

**Option A (RECOMMENDED):** Build full body now, we'll hide parts we don't need yet  
- Easier to extend later
- Cleaner topology
- ~40-60k tris total (with LOD)

**Option B:** Build bust only (15-20k), extend in phases  
- More optimized per phase
- More rework to extend later

**Which do you prefer?** I recommend Option A for simplicity.

---

## 🧪 **Milestone 2 = Pipeline Testing**

This is where we test CC4 → CC5 → iClone → Unity:

**Testing Checklist:**
1. Does .ccCharacter import to CC5?
2. Are all 52 morphs intact?
3. Do eye bones work in iClone Face Puppet?
4. Does FBX export with blendshapes?
5. Does Unity hit 60 FPS?
6. Does gaze tracking work?
7. Does lip-sync work?

**Timeline:** 2-3 days back-and-forth  
**My feedback:** Within 24 hours of testing

---

## 📁 **What to Send:**

```
Kelly_Phase1_v1/
├── Kelly_Base.ccCharacter
├── Textures/ (if separate)
├── Screenshots/
│   ├── CC4_Morph_List.png
│   ├── CC4_Eye_Bones.png
│   └── Views (front/side/wireframe)
└── Notes.txt
```

---

## ❓ **5 Quick Questions:**

1. Build approach: Option A or B?
2. Have you exported CC4 eye bones to iClone before?
3. Does Facial Profile give all 52 morphs automatically?
4. Any concerns with Milestone 2 timeline?
5. Can you test iClone import before sending?

---

## 📎 **Attached:**

1. **Detailed Rollout Plan** (full 4-phase breakdown)
2. **Quick Reference** (print this for your desk!)
3. **Visual Pipeline** (diagrams)

---

## ✅ **Next Steps:**

1. Answer 5 questions above
2. Choose build approach (A or B)
3. Complete Milestone 1 (base modeling) ✅
4. Export Phase 1 model
5. Send for testing
6. We iterate until all 7 tests pass
7. Milestone 2 approved! → Phase 2 🎉

---

**Bottom Line:** We're starting with face-focused Kelly (bust only) to ship fast. We'll add arms/body in later phases as we prove the system works.

**Critical:** Eye bones must be separate and work through the full pipeline. Test in iClone before sending!

Let me know your build approach preference and any questions!

Best,  
[Your Name]

---

**P.S.** The Quick Reference attachment is a one-pager you can print and keep at your desk while working!


