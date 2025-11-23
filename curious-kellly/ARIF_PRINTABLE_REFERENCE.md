# Kelly Avatar - Quick Reference for Arif
**Print and keep at your desk!** 📌

---

## 🔄 The Pipeline (Your Work → Production)

```
CC4/ZBrush → .ccCharacter → CC5 → iClone → Unity → Mobile (60 FPS)
   (You)      (Export)     (Me)   (Me)    (Me)   (iPhone/Pixel)
```

---

## 🎯 Phase 1: What to Build NOW (Milestone 2)

**Model Scope:**
```
    👤 Kelly
   ┌─────┐
   │ 😊  │  ← Head (full detail)
   ├─────┤
   │ 👚  │  ← Shoulders + upper chest
   └─────┘
═════════════ CUT HERE (mid-torso)
```

**Camera:** Tight close-up (like Zoom)  
**Poly Count:** 15,000-20,000 triangles MAX  
**Goal:** Get Kelly teaching THIS MONTH

---

## 🚨 TOP 3 CRITICAL ITEMS

### **1. EYE BONES (MOST IMPORTANT)**
```
Kelly_Head
├── LeftEye_Bone  ← MUST be separate
└── RightEye_Bone ← MUST be separate

✅ Independent control (not linked)
✅ Pivot at eyeball center
✅ Rotates (not translates)
✅ Range: ±30° horizontal, ±20° vertical
```

**TEST THIS:** Import to iClone → Face Puppet → Can you move left/right eyes separately?

### **2. 52 FACIAL MORPHS**
- **Use CC4 Facial Profile** (should be automatic)
- **Eyes:** Blink L/R, Look Up/Down/In/Out L/R (10 morphs)
- **Brows:** Inner Up, Outer Up L/R, Down L/R (8 morphs)
- **Mouth/Jaw:** Open, Forward, Left, Right, Funnel, Pucker, Smile, Frown, Roll Upper/Lower, Shrug Upper/Lower, Close, Upper Up L/R, Lower Down L/R (20+ morphs)
- **DO NOT BAKE** (must be editable)

### **3. EXPORT AS .ccCharacter**
- **Format:** .ccCharacter (or .iAvatar)
- **Include:** Mesh + ALL morph sliders + eye bones + textures
- **Textures:** 2048x2048 (diffuse + normal)
- **Test in iClone before sending to me**

---

## ✅ Before You Send - Checklist

- [ ] 52 morphs present (CC4 Facial Profile enabled)
- [ ] Eye bones separate (L/R independent)
- [ ] Morphs NOT baked (editable)
- [ ] Tested import to iClone
- [ ] Eye bones work in Face Puppet
- [ ] Clean topology (quads, no n-gons)
- [ ] UV mapped (minimal face seams)
- [ ] Textures 2048x2048
- [ ] Static pose (sitting, neutral)
- [ ] Poly count 15-20k
- [ ] Screenshots included

---

## 📁 What to Send Me

```
Kelly_Phase1_v1/
├── Kelly_Base.ccCharacter      ← Main file
├── Textures/                   ← If not embedded
│   ├── Kelly_Diffuse_2048.png
│   └── Kelly_Normal_2048.png
├── Screenshots/
│   ├── CC4_Morph_List.png     ← Show all morphs
│   ├── CC4_Eye_Bones.png      ← Show eye bone setup
│   ├── Front_View.png
│   ├── Side_View.png
│   └── Wireframe.png
└── Notes.txt                   ← Any special notes
```

---

## 🧪 Testing (I Do This - 24 Hour Turnaround)

**7 Tests I'll Run:**
1. ✅ Import to CC5 (works? morphs there?)
2. ✅ Export to iClone (eye bones work?)
3. ✅ Test Face Puppet (eyes independent?)
4. ✅ Export FBX to Unity (blendshapes export?)
5. ✅ Test performance (60 FPS achieved?)
6. ✅ Test gaze tracking (eyes move naturally?)
7. ✅ Test lip-sync (mouth morphs work?)

**If all 7 pass → Milestone 2 APPROVED! 💰**

---

## 🎯 Build Approach Question

**Option A (RECOMMENDED):** Build full body now, we hide parts we don't need yet
- Easier to extend later
- Cleaner topology
- ~40-60k tris total

**Option B:** Build bust only (15-20k), extend in phases
- More optimized per phase
- More rework later

**Tell me which you prefer!**

---

## 🚀 The 4 Phases (Big Picture)

**Phase 1 (NOW - Milestone 2):** Face-focused → Ship THIS MONTH  
**Phase 2 (Milestone 3):** Add arms/hands + hair → Gestures  
**Phase 3 (Milestone 4):** Full body + poses → Rich variety  
**Phase 4 (Future):** Animation rig → Interactive

---

## ❌ Common Mistakes to Avoid

❌ **Baking morphs** → Keep editable!  
❌ **Linked eye bones** → Must be separate!  
❌ **Skipping iClone test** → Test before sending!  
❌ **High poly count** → Stay under 20k for Phase 1  
❌ **Missing morphs** → Need all 52 from Facial Profile

---

## 💰 Payment

**Milestone 1:** Base modeling - $250 ✅ DONE  
**Milestone 2:** Pipeline testing - $250 (when 7 tests pass)  
**Milestone 3:** Hair + upper body - $250  
**Milestone 4:** Full body + final - $250

---

## 📞 Contact Me If...

🚨 **Eye bones don't work in iClone**  
🚨 **Morphs disappear after export**  
🚨 **CC4→iClone pipeline issues**  
🚨 **Any technical blocker**

**Don't worry about:**
✅ Small topology tweaks  
✅ Texture adjustments  
✅ UV fixes  
✅ Minor polish

---

## ✅ Success = These 7 Things Work

1. CC5 import (no errors)
2. All 52 morphs present
3. Eye bones independent (L/R)
4. FBX exports clean
5. Unity 60 FPS achieved
6. Gaze tracking works
7. Lip-sync functional

**When all 7 pass → Payment released → Phase 2!** 🎉

---

## 🎯 Remember

**Focus:** Face, eyes, expressions (what matters most)  
**Goal:** Kelly teaching lessons THIS MONTH  
**Critical:** Eye bones MUST work independently  
**Timeline:** Fast iteration, test early, fix together

**Pipeline:** CC4 → CC5 → iClone → Unity → Mobile (60 FPS)

---

## 🔧 Quick CC4 Export Settings

**File → Export → Character**
- Format: .ccCharacter
- Include: Mesh, Skin, **Morph Sliders**, Skeleton
- Texture Resolution: 2048x2048 max
- **DO NOT** bake morphs

---

## 📊 Performance Targets

- **Poly Count:** 15-20k tris (Phase 1)
- **FPS:** 60 FPS on iPhone 12 & Pixel 6
- **CPU:** < 30%
- **GPU:** < 50%
- **Memory:** < 500MB
- **Textures:** 2048x2048 max

---

## 🎬 Next Steps

1. Answer build approach question (Option A or B?)
2. Complete Milestone 1 (base modeling) ✅
3. Export Phase 1 model (.ccCharacter)
4. Test in iClone (especially eye bones!)
5. Send files + screenshots
6. I test within 24 hours
7. We iterate if needed
8. Milestone 2 approved!

---

**Questions? Ask anytime!** 💬

**This is your desk reference - keep it visible while working!** 📌

---

## 🔍 Eye Bone Setup (Critical!)

**CORRECT (What We Need):**
```
Kelly_Head
│
├─ LeftEye_Bone ───▶ Independent
│  └─ LeftEye_Mesh
│
└─ RightEye_Bone ──▶ Independent
   └─ RightEye_Mesh
```

**WRONG (Won't Work):**
```
Kelly_Head
│
└─ Eyes_Bone ───▶ Linked together ❌
   ├─ LeftEye_Mesh
   └─ RightEye_Mesh
```

---

**PIN THIS TO YOUR DESK!** 📌










