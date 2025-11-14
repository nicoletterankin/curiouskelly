# Arif Quick Reference - Kelly Avatar Pipeline

**Print this and keep at your desk!** ⭐

---

## 🔄 The Pipeline (What Happens to Your Work)

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR WORKFLOW (Arif)                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CC4 + ZBrush  →  Export .ccCharacter                       │
│                                                              │
│  ✅ Base mesh                                                │
│  ✅ 52 facial morphs (CC4 Facial Profile)                   │
│  ✅ Eye bones (L/R separate)                                │
│  ✅ Textures (2048x2048)                                     │
│  ✅ Static pose (sitting)                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  MY WORKFLOW (Client)                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Import to CC5  →  Export to iClone  →  Export FBX         │
│                                                              │
│  ✅ Verify morphs work                                       │
│  ✅ Test eye bones (Face Puppet)                            │
│  ✅ Check performance (60 FPS)                              │
│  ✅ Import to Unity                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  PRODUCTION (Mobile App)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Unity → iOS/Android                                        │
│                                                              │
│  ✅ 60 FPS on iPhone 12 / Pixel 6                           │
│  ✅ Real-time lip-sync (mouth morphs)                       │
│  ✅ Gaze tracking (eye bones)                               │
│  ✅ Expressions (teaching moments)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Phase 1 Deliverable (Milestone 2)

### **What to Build:**

**MODEL SCOPE:**
```
        👱 Head (full detail)
        🫱 Neck
        👚 Shoulders + Upper Chest
        ✂️ CUT HERE (mid-torso)
        
Camera shows: Shoulders up (like Zoom)
```

**POLY COUNT:** 15,000-20,000 triangles MAX

---

## ✅ Critical Checklist (Phase 1)

### **Before You Export from CC4:**

- [ ] **52 Facial Morphs** (CC4 Facial Profile enabled)
- [ ] **Eye Bones** (left and right separate - test rotation ±30°)
- [ ] **Breathing Morph** (1 subtle chest movement)
- [ ] **Clean Topology** (quads, no n-gons, good edge loops)
- [ ] **UV Mapped** (minimal seams on face)
- [ ] **Textures** (2048x2048 diffuse + normal)
- [ ] **Static Pose** (sitting, looking forward, neutral expression)

### **Export Settings (CC4):**

- [ ] File → Export → Character
- [ ] Format: **.ccCharacter** (preferred) or .iAvatar
- [ ] Include: Mesh, Skin, **Morph Sliders**, Skeleton
- [ ] **Do NOT bake morphs** (keep editable)
- [ ] Texture resolution: 2048x2048 max
- [ ] Test import in iClone before sending to me

---

## 🚨 Top 3 Critical Items

### **1. EYE BONES (MOST IMPORTANT)**
```
Kelly_Head
├── LeftEye_Bone  ← Must be separate
└── RightEye_Bone ← Must be separate

✅ Pivot at eyeball center
✅ Rotates (not translates)
✅ Independent control (not linked)
```

**TEST THIS:** Import to iClone → Face Puppet → Can you move left/right eyes separately?

---

### **2. FACIAL MORPHS (52 MINIMUM)**

**Must Have for Lip-Sync:**
- Jaw: Open, Forward, Left, Right
- Mouth: Funnel, Pucker, Smile, Frown, Roll Upper/Lower
- Mouth: Shrug Upper/Lower, Close, Upper Up L/R, Lower Down L/R

**Must Have for Eyes:**
- Eye: Blink L/R, Look Up/Down/In/Out L/R
- Brow: Inner Up, Outer Up L/R, Down L/R

**Nice to Have:**
- Tongue Out, Nose Sneer L/R, Cheek Puff L/R

**TEST THIS:** Do morphs show up in CC5 after import?

---

### **3. PERFORMANCE (60 FPS TARGET)**

**Poly Count Budget:**
- Head: 10k tris
- Eyes: 1k tris total
- Neck/Shoulders: 4-5k tris
- **Total: 15-20k tris**

**Texture Budget:**
- 2048x2048 (head + body combined)
- Or 2x 1024x1024 (head + body separate)

---

## 📁 What to Send Me (Milestone 2)

```
Kelly_Phase1_v1/
├── Kelly_Base.ccCharacter      ← Main file
├── Textures/                   ← If not embedded
│   ├── Kelly_Diffuse_2048.png
│   └── Kelly_Normal_2048.png
├── Screenshots/
│   ├── CC4_Morph_List.png     ← Show me all morphs
│   ├── CC4_Eye_Bones.png      ← Show eye bone setup
│   ├── Front_View.png
│   └── Side_View.png
└── Notes.txt                   ← Any special instructions
```

---

## ❓ Quick Answers to Common Questions

**Q: Should I build full body now or just bust?**  
A: **Your choice!** Option A (build full, hide parts) is easier. Option B (build in phases) is more optimized. Let me know your preference.

**Q: What if CC4 morphs don't import to CC5?**  
A: **Flag this immediately!** This is the #1 thing Milestone 2 is testing. We'll troubleshoot together.

**Q: What if eye bones don't work in iClone?**  
A: **Critical blocker.** We need to solve this before continuing. I'll help research.

**Q: Poly count too high?**  
A: **We can optimize together.** Send it anyway and I'll test performance. May need retopology.

**Q: Can I use CC4 hair system?**  
A: **Yes, if it exports cleanly.** Test in iClone first. Hair is Milestone 3 anyway.

---

## 🎯 Success = These 5 Things Work

1. ✅ I import your .ccCharacter into CC5 (no errors)
2. ✅ All 52 morphs show up in CC5
3. ✅ Eye bones work in iClone Face Puppet (independent control)
4. ✅ FBX exports from iClone with morphs intact
5. ✅ Unity runs at 60 FPS on mobile

**If all 5 work → Milestone 2 complete! → Move to Phase 2** 🎉

---

## 📞 When to Contact Me

**Contact ASAP if:**
- ❌ Eye bones don't work in CC4→iClone
- ❌ Morphs disappear after export
- ❌ .ccCharacter format doesn't work
- ❌ Poly count way over budget
- ❌ Any technical blocker

**No need to contact for:**
- ✅ Small topology tweaks
- ✅ Texture resolution adjustments
- ✅ UV mapping fixes
- ✅ Minor visual polish

---

## 🚀 Remember

**Phase 1 Goal:** Get Kelly's face teaching lessons.  
**Focus on:** Face, eyes, expressions.  
**Performance:** 60 FPS or bust.  
**Timeline:** Fast iteration, test early, ship quickly.

---

**Print this page and keep it visible while working!** 📄

Good luck! 🎨



