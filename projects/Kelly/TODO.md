# Kelly — Build TODO

## Core Production Pipeline
1) CC5: Create HD head (Headshot 2 or ActorMIXER), save to projects/Kelly/CC5
2) **[NEW]** Apply Hair Physics: Load `CC5/HairPhysics/Kelly_Hair_Physics.json` preset
   - Weight map: `Kelly_Hair_PhysicsMap.png` (roots locked → tips free)
   - Fine detail: `Fine_Strand_Noise.png` for strand-level realism
   - See: `HAIR_PHYSICS_WORKFLOW.md` for full integration steps
3) Send to iClone. Load DirectorsChair_Template.iProject.
4) Voice: drop WAV into projects/Kelly/Audio and run AccuLips.
5) (Optional) AccuFACE VIDEO with mouth disabled.
6) Render test to renders/Kelly/Kelly_test_talk_v1.mp4
7) Run contact sheet + frame metrics scripts for Kelly.

## Hair Physics Assets ✅ COMPLETE
- ✅ `Kelly_Hair_Physics.json` — Natural weighted simulation preset
- ✅ `Kelly_Hair_PhysicsMap.png` — Gradient weight map (black roots → white tips)
- ✅ `Fine_Strand_Noise.png` — Micro-strand detail texture
- ✅ `README.txt` — Quick import reference
- ✅ `HAIR_PHYSICS_WORKFLOW.md` — Complete integration guide

**Location:** `projects/Kelly/CC5/HairPhysics/`

## Next Steps
- [ ] Test hair physics with head turn animations
- [ ] Cache physics simulation for lesson segments
- [ ] Validate physics in iClone with wind environment
- [ ] Document final physics parameters in production notes
