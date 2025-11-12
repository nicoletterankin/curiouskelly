# Short Follow-Up Email to Arif

**Subject:** Kelly Avatar Pipeline - CC4→CC5→iClone→Unity Requirements

---

Hi Arif,

Thanks for the meeting this morning! Here's what we need to make sure works through our pipeline:

## **Your Workflow → My Workflow**
CC4/ZBrush (you) → CC5 (me) → iClone (me) → Unity FBX → Mobile App

---

## **Critical Requirements**

### 1. **Export Format from CC4**
- **File**: .ccCharacter or .iAvatar format
- **Must include**: Base mesh, ALL morph sliders, eye bones, textures
- **DO NOT** bake morphs - I need them editable in CC5

**Question:** Can CC4 export with all morphs intact for CC5 import?

---

### 2. **Eye Bones (MOST CRITICAL)**
We need **separate, independent eye bones** for left and right eyes that work through the entire pipeline.

**Test this:** After you export from CC4, can you import to iClone and control left/right eyes independently in Face Puppet?

**This is #1 priority** - our entire gaze tracking system depends on it. If you're unsure, let's test this in Milestone 2 FIRST.

---

### 3. **Morphs/Blendshapes (52 minimum)**
We need all CC4 Facial Profile morphs:
- **Eyes**: Blink L/R, Look Up/Down/In/Out L/R (10 morphs)
- **Brows**: Inner Up, Outer Up, Down L/R (8 morphs)
- **Mouth/Jaw**: Open, Forward, Left, Right, Funnel, Pucker, Smile, Frown, Roll Upper/Lower, Shrug Upper/Lower, etc. (20+ morphs for lip-sync)
- **Tongue Out** (if possible)

These must work in Unity after going through CC5 and iClone.

---

### 4. **Optimization (Mobile - 60 FPS target)**
- **Poly count**: 10k-15k triangles for head + neck
- **Textures**: 2048x2048 max (Diffuse + Normal)
- **We don't need**: Full body (just head + neck is fine)

**Question:** What's your typical CC4 head poly count?

---

### 5. **Hair (Milestone 3)**
- Separate mesh from head
- 5k-8k triangles
- Must export through the pipeline

**Question:** What's your usual approach for hair in CC4?

---

## **Milestone 2 Testing (MOST IMPORTANT)**

This is where we test the **entire pipeline** before proceeding:

1. ✅ You export from CC4
2. ✅ I import to CC5 (does it work?)
3. ✅ I export to iClone (morphs intact?)
4. ✅ I test eye bones (independent control?)
5. ✅ I export FBX to Unity (everything works?)
6. ✅ I test performance (60 FPS?)

**We fix any issues HERE before Milestones 3 & 4.**

**Question:** Are you available for quick iterations during this testing phase?

---

## **What I Need from You**

### **Now (Quick Answers):**
1. Can CC4 export to CC5 with morphs intact?
2. Have you tested CC4 eye bones in iClone before?
3. What's your typical poly count?
4. Can you do quick iterations in Milestone 2?

### **For Milestone 2 (After Current Milestone 1):**
```
Kelly_Avatar_v1/
├── Kelly_Base.ccCharacter
├── Textures/ (if not embedded)
└── Screenshots of CC4 morph list
```

---

## **Critical Items to Verify in Milestone 2**

**Must work or we can't proceed:**
1. ✅ Eye bones work independently (CC4→CC5→iClone→Unity)
2. ✅ All 52 morphs survive the pipeline
3. ✅ 60 FPS performance in Unity
4. ✅ File format compatibility

---

## **Next Steps**

1. Answer the 4 questions above
2. Let me know if anything is unclear
3. When Milestone 1 is ready, send files for Milestone 2 testing
4. I'll test and give feedback within 24 hours
5. We iterate until everything works perfectly

Base model looks great! Looking forward to testing the pipeline with you.

Best,  
[Your Name]

---

**P.S.** - Eye bones are #1 priority. If you're unsure about anything with eye bone setup in CC4→iClone, let me know ASAP so we can research together.


