# Final Email to Arif

**SEND THIS NOW** ✉️

---

**To:** Arif Ahmed  
**Subject:** Kelly Avatar - Phase 1 Requirements (Print the Attachment!)

---

Hi Arif,

Great meeting with you this morning! Here's everything you need to build Kelly's Phase 1 avatar.

---

## 🎯 **Quick Summary**

We're building Kelly in **4 phases** to ship fast and add capabilities as we prove the system:

**Phase 1 (NOW - Milestone 2):** Face-focused teaching → Ship THIS MONTH  
**Phase 2 (Milestone 3):** Add arms/hands + hair → Gestures  
**Phase 3 (Milestone 4):** Full body + poses → Rich variety  
**Phase 4 (Future):** Full animation → Interactive

---

## 📋 **Phase 1: What to Build**

**Model Scope:**
```
    👤 Kelly
   ┌─────┐
   │ 😊  │  ← Head (full detail)
   ├─────┤
   │ 👚  │  ← Shoulders + upper chest
   └─────┘
═══════  ← Cut here (mid-torso)
```

**Camera View:** Tight close-up (like Zoom/FaceTime)  
**Poly Count:** 15,000-20,000 triangles  
**Why:** Get Kelly teaching ASAP, focus on face/eyes

---

## 🚨 **Top 3 Critical Items**

### **1. Eye Bones (MOST IMPORTANT)**
- Separate left/right eye bones (independent control)
- Must work in iClone Face Puppet
- **Test this before sending!**

### **2. 52 Facial Morphs**
- Use CC4 Facial Profile (should be automatic)
- Jaw, mouth, eyes, brows (for lip-sync + expressions)
- **DO NOT bake** (keep editable)

### **3. Export as .ccCharacter**
- Include: Mesh + ALL morph sliders + eye bones + textures
- Texture: 2048x2048
- Test in iClone before sending

---

## ❓ **Key Question: Build Approach**

**Option A (RECOMMENDED):** Build full body now, we'll hide parts we don't need yet
- Easier to extend later
- Cleaner topology
- ~40-60k tris total (we'll use LOD)

**Option B:** Build bust only (15-20k), extend in later phases
- More optimized per phase
- More rework to extend later

**Which do you prefer?** I recommend Option A for simplicity, but it's your call.

---

## 🧪 **Milestone 2 = Pipeline Testing**

This is where we test the entire CC4 → CC5 → iClone → Unity pipeline.

**7 Tests I'll Run:**
1. Does .ccCharacter import to CC5?
2. Are all 52 morphs intact?
3. Do eye bones work in iClone Face Puppet?
4. Does FBX export with blendshapes?
5. Does Unity hit 60 FPS?
6. Does gaze tracking work?
7. Does lip-sync work?

**Timeline:** 2-3 days back-and-forth  
**My feedback:** Within 24 hours

---

## 📁 **What to Send**

```
Kelly_Phase1_v1/
├── Kelly_Base.ccCharacter
├── Textures/ (if not embedded)
├── Screenshots/
│   ├── CC4_Morph_List.png
│   ├── CC4_Eye_Bones.png
│   └── Views (front/side/wireframe)
└── Notes.txt
```

---

## ❓ **5 Quick Questions**

Please answer these so I can prepare:

1. **Build approach:** Option A or B?
2. **Eye bones:** Have you exported CC4 eye bones to iClone before?
3. **Morphs:** Does CC4 Facial Profile give all 52 morphs automatically?
4. **Timeline:** Any concerns with Milestone 2?
5. **Testing:** Can you test iClone import before sending to me?

---

## 📎 **ATTACHMENT: Print This!**

I've attached a **one-page reference guide** with everything you need:
- Pipeline overview
- Critical requirements
- Before-send checklist
- Eye bone setup diagram
- What to send
- Success criteria

**Please print it and keep it at your desk while working!** It'll save us a ton of back-and-forth.

---

## ✅ **Next Steps**

1. Answer the 5 questions above
2. Let me know your build approach preference (A or B)
3. Complete Milestone 1 (base modeling) ✅
4. Export Phase 1 model
5. Test in iClone (especially eye bones!)
6. Send for testing
7. I'll test within 24 hours
8. We iterate until all 7 tests pass
9. Milestone 2 approved! → Payment released 💰

---

## 🎯 **Bottom Line**

**Focus:** Face, eyes, expressions (what matters most)  
**Goal:** Get Kelly teaching THIS MONTH  
**Critical:** Eye bones must be separate and work through the pipeline  
**Timeline:** Fast iteration, we'll solve issues together

The attached reference has all the technical details. Print it and use it!

---

Looking forward to testing the pipeline with you! The base model looks great from the images you shared.

Let me know your build approach preference and when you're ready to send Phase 1.

Best,  
[Your Name]

---

**P.S.** - The eye bones are THE most critical part. If you're unsure about anything related to CC4 eye bone setup, let me know ASAP so we can research it together. This is the foundation of our gaze tracking system.

**P.P.S.** - Print the attachment! Seriously. It'll save you so much time. 📄

---

**ATTACHMENT:**
- Kelly Avatar - Quick Reference for Arif.pdf (1 page, printable)



