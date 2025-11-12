# Kelly | CC5 • Headshot 2 • iClone 8 — Operator Training Manual (v2)

A practical, step-by-step reference for building a photoreal, talking digital human in Reallusion Character Creator 5 (CC5) and iClone 8 using Headshot 2. Organized for production: menus, dropdowns, and exact workflows.

---

## 1) Version Map — What’s What

- **CC3 + Headshot 1**: legacy photo-to-head pipeline; SkinGen plug-in; CC3 Base/CC3+ topology.
- **CC4 + Headshot 2**: adds **Mesh Mode** (import OBJ/FBX head), alignment points, re-baking, improved masking & reprojection.
- **CC5**: **HD Characters**, subdivision workflow, **8K** textures, **Cavity** maps, **HD Morphs**, **HD/Extended+ facial profiles**.
- **iClone 8**: **AccuLIPS** lipsync, facial puppet/mocap, viseme dictionary management.

---

## 2) Headshot 2 — UI & Dropdowns You’ll Use

### 2.1 Pro Mode (Photo) — Controls
- **Image Matching Tools** (overlay): Grayscale toggle, Contrast slider, Zoom View, **Lens** slider, **Reset to Default Lens**.
- **Sculpt Morph** + morph sliders: push/pull likeness while viewing the overlay.
- **Re-Project Photo**: re-bakes textures after mesh/morph edits (Pro-only).

### 2.2 Mesh Mode — Stages & Options
- **Start Head Generation → Alignment Points**  
  Choose a preset (e.g., 32 pts) and refine landmarks (brows, eyes, nose tip/alar, lip corners, jaw, ears, chin).
- **HEAD GEN → Effective Areas**  
  Entire Head / Face-only / Face+Ears. Unselected zones auto-complete.
- **REFINE MESH (brushes)**  
  **Move (Shift+1)** and **Smooth (Shift+2)** to conform the CC mesh to your source mesh.
- **ATTACH TO BODY → Generate Character (dialog)**  
  - **Texture Bake Options (Diffuse)**: From Source Mesh / From High Poly Mesh / Project from Image / Textureless  
  - **Texture Bake Options (Normal)**: From Source Mesh / From High Poly Mesh  
  - **Texture Size**: 512 → **8192 (8K)**  
  - **Body Type**: choose target CC body shell  
  - **Texture Mask**: No Mask, Clean Soft, Clean Rough, Scalp, etc. (varies by body type)  
  - Note: **Project from Image** disables **Texture Mask** selection by design.

### 2.3 Skin Type / Mask Presets (Quick Swap)
- Use Skin Type buttons for packaged masking: **No Mask**, **Clean Soft**, **Clean Rough**, **Scalp**, plus body-type-specific presets (e.g., Old, Allergic, Contour).

---

## 3) Aligning Mesh to a Reference Image — Step-by-Step

### For Photos (Pro Mode)
1. Load front photo → enable **Image Matching Tools**.  
2. Toggle **Grayscale** and raise **Contrast** for landmarks.  
3. Adjust **Lens** to match the photo’s FOV; **Reset** if needed.  
4. While overlay is active, use **Sculpt Morph** + morph sliders to match the silhouette (eyes → nose → mouth → jaw).  
5. **Re-Project Photo** to re-bake a neutral, aligned base color.

### For 3D Mesh (Mesh Mode)
1. Place/adjust **Alignment Points** (include ears).  
2. Run **HEAD GEN** with **Entire Head** if possible.  
3. Use **REFINE** (Move/Smooth) to conform CC mesh.  
4. **Generate Character** and choose bake options: **From High Poly** (normals/diffuse) or **Project from Image** (texture from still), at **8K** if needed.

---

## 4) CC5 Essentials — Morphs, SkinGen, Materials, Eyes

### 4.1 Morphs Tab (Body Adjusting)
- **Filter** drop-down; **Add** morphs; **Tree** view (Currently Used/Favorites); **Show Sub Items**; **Search**; **Bake/Reset**.

### 4.2 SkinGen (Appearance Editor)
- Enable **Activate Editor**; layer-based skin/makeup.  
- After Subdivision Level 2 work, **Bake normals** to Levels 0/1 for stability.

### 4.3 Materials — 8K & Cavity
- Promote **Base Color**, **Normal**, **Roughness**, **AO**, **Cavity** to **8K** for close-ups.  
- Use **Cavity** to emphasize micro creases/pore valleys (avoid over-reliance on roughness).

### 4.4 Eye Realism — Tear Line & Eye Occlusion
- Add **Eye Occlusion** + **Tear Line** meshes (Enhance Eyes).  
- Tune **Tear Line Depth Offset**, **Detail Amount**, and tiling for a natural wet line under the upper lid.

---

## 5) CC5 HD — What’s New for Photoreal
- **HD Morphs** across Subdivision 0–2.  
- **8K textures**, **AO/Cavity/Normal** bakes for pore crispness.  
- **HD/Extended+ facial profiles** (MetaHuman-compatible; hundreds of facial morph sliders).  
- **Face Control Panel** plugin for testing expressions.

---

## 6) Edit Mesh & GoZ — Local Fixes and ZBrush Loop
- **Edit Mesh**: vertex/face/element selection; Push/Pull/Smooth; Delta Mush; weld/symmetrize.  
  Use for mouth corner cleanup, eyelid–cornea intersections, clothing fit.  
- **GoZ**: choose texture size, subdiv, pose; auto-regenerate Tear Line/Eye Occlusion on return.  
  For skin pores from ZBrush, bake displacement/normal appropriately.

---

## 7) iClone 8 — AccuLIPS, Visemes & Polish
- **Create Script → AccuLIPS**: import WAV/MP3 (or record/TTS), **Generate Text**, correct words, **Align** or **Update Selected**, **Apply** (visemes).  
- **Dictionary**: Export/Import/Reset; add unknown words/visemes; share across projects.  
- **Lip Sync Options**: reduce lip keys, **Smooth** transitions; stylize via **Clip Strength** for **Full Mouth** or per-part (Tongue/Lips/Jaw).  
- **Playback Range**: double-click word(s) to loop while adjusting durations.

---

## 8) End-to-End: Kelly Build (CC5 + Headshot 2 + iClone 8)

1. Select best front/profile photos (or clean OBJ/FBX head with good UVs).  
2. **Headshot 2 Pro**: overlay photo (Grayscale/Contrast), match **Lens**, sculpt morphs, **Re-Project Photo**.  
3. **Headshot 2 Mesh**: alignment points → **HEAD GEN** (Entire Head) → **REFINE** (Move/Smooth) → **Generate** (From High Poly or Project from Image, **8K**).  
4. **CC5 Polish**: Morphs pass (jaw/eyes/nose/mouth); **SkinGen** (redness/veins) → **Bake normals** to L0/L1; upgrade to **8K**; add **Cavity**.  
5. **Eyes**: Enhance Eyes; tune **Tear Line** & **Eye Occlusion**.  
6. **Send to iClone**: **AccuLIPS** → Align/Update → Apply; **Lip Sync Options** smoothing/strength; Facial Puppet/Face Key for blinks and micro-nods.  
7. **QC**: eyelid coverage; lip seal (M/B/P); tooth intersections; specular control (T-zone).  
8. **Export**: stills at 8K; talking clip at 4K.

---

## 9) Reality Check — What You Can / Can’t Do

**Can do**
- Photo→head with Pro overlay; **Re-Project Photo** after morphing.  
- Mesh→head with alignment points + **REFINE** brushes; bake Diffuse/Normal **From High Poly** or **Project from Image**; **8K** textures.  
- Eye realism via **Eye Occlusion** + **Tear Line**.  
- HD character work: **HD morphs**, **8K**, **Cavity**, **HD/Extended+ facial profiles**.  
- Lip-sync via **AccuLIPS** with dictionary and stylization.

**Can’t / Limitations**
- **Image Matching Tools** and **Re-Project Photo** are Pro-only (not Auto).  
- **AccuLIPS** targets CC Standard (G1–G3+) and Game Base (not arbitrary non-CC rigs).  
- In Mesh Mode, **Project from Image** disables **Texture Mask** selection.

---

## 10) Troubleshooting — Fast Fixes

- **Waxy skin**: raise roughness; reduce micro-normal intensity; ensure de-lit base color.  
- **Double lower lip**: lighten AO at lower lip; reduce local normal; refine corner topology; reproject if needed.  
- **Dead eyes**: add Eye Occlusion + Tear Line; adjust Tear Line depth/detail; subtle AO at upper lid.  
- **Viseme clipping**: lower viseme strength; reduce F/V bursts; smooth transitions; add micro head motion.  
- **Hairline seam**: repaint scalp base; add baby-hair/flyaway hair layers.

---

## One-Glance Checklist (Print)

- Headshot aligned & re-projected  
- Mesh refined  
- 8K maps on head  
- SkinGen baked to L0/L1  
- Eye occlusion & tear line tuned  
- HD Facial Profile applied  
- AccuLIPS aligned/applied  
- iClone polish pass complete  
- 8K still + 4K clip exported
