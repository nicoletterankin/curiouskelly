
# 🎬 Kelly HD Pipeline (v2) — ActorMIXER → Headshot 2 → iClone 8.62
**Revision date:** 2025-10-12  
**Reason for update:** Uncanny‑valley optimizations (skin/eyes/teeth), SubD targets, AccuFACE↔AccuLips blending, color management, QA hooks.

---

## 0) Project & Color Management
- Project root: `projects/Kelly/`
- Color space: **sRGB** for textures, **Rec.709 Gamma 2.4** for review videos.
- Naming: `kelly.[stage].[pass].[vNN].ext` (e.g., `kelly.l1.short.v01.mp4`).

---

## 1) Start HD Character (ActorMIXER)
1. Launch **Character Creator 5** → `File → New Project` → save as `projects/Kelly/CC5/Kelly_HD_Base.ccProject`.
2. Content: `Actor → Character → HD Human Anatomy Set` → **Female Athletic (CC5 HD)** → Apply.
3. **ActorMIXER → Head**: coarse likeness pass (jaw width +4, nose bridge −5~−10, mouth corner +3).
4. **SubD targets (important):**
   - Viewport SubD: **1** (use **2** only for close-up sculpting).
   - Render SubD: **2** (use **3** for hero macro close‑ups). Avoid 4 except stills.
5. Enable **Corrective Expressions**; keep **Auto‑Blink: None** in CC (we’ll drive blinks later in iClone).
6. Save checkpoint.

---

## 2) Headshot 2 → Photo to 3D (Pro)
1. `Plugins → Headshot 2 → Photo to 3D (Pro)` → **Load Photo**: `Ref/headshot2-kelly-base169_101225.png`.
2. **HS2 Settings:** Ultra High (8K), Mesh Density: Maximum, Detail: Maximum, Processing: Ultra High.
3. **Generate** → **Apply to Character** → **Accept**.
4. **Sculpt polish (HS2 sliders):** jaw roundness −5, lower‑lip thickness +3, eye outer tilt +1.
5. **Bake Normals** if prompted (keep a copy of pre‑bake morphs for future edits).
6. Save `Kelly_HS2_HD.ccProject`.

**Tip:** If a multi‑angle scan arrives, use **Headshot 2 → Mesh to 3D** to wrap + bake onto a CC5 HD head.

---

## 3) Skin, Eyes, Teeth, Hair (15–30 min)
### Skin (Digital Human Shader)
- Verify **DHS** active. Start from HS2 8K maps. Tuning ranges:
  - Roughness 0.45 → **0.38–0.42** (T‑zone slightly lower).
  - SSS strength **0.25–0.30**; radius default; tint slightly warm.
- Add **micro normal** (pores) on top of HS2 normal for close‑ups.
- Optional: enable **Dynamic/Wrinkle normals** if pack installed (driven by expressions).

### Eyes (critical for realism)
- Use **HD Eyes** + **HD Lashes**. Ensure:
  - Separate **cornea** (IOR ≈ 1.376), bulge enabled.
  - **Tear‑line/wetness** mesh on (thin strip at lid margin).
  - AO/cavity map around caruncle and limbal ring.
- Iris size 11–12 mm equiv; subtle dilation animation later in iClone.

### Teeth/Tongue
- Teeth roughness 0.25–0.35, specular color near white; add AO at gum line.
- Tongue roughness 0.4–0.5, subsurface slight; avoid mirror‑like spec.

### Hair
- Choose **Hair Builder** style close to ref; reduce specular to ~0.25; add **baby‑hair cards** at hairline when available.

Save `Kelly_HS2_HD_SkinEyesHair.ccProject`.

---

## 4) Send to iClone & Scene Lock
1. `File → Send Character to iClone` (Digital Human CC5 HD).
2. Create **Director’s Chair** scene:
   - Camera **85 mm**, DOF focus on eyes.
   - Lights: soft 3‑point or neutral studio HDRI.
   - Idle layer: gentle breathing (no blinks yet).
3. Save `DirectorsChair_Template.iProject`.

---

## 5) Lip‑Sync — AccuLips
1. Import `kelly25_audio.wav` → Right‑click track → `AccuLips`.
2. Let it transcribe; fix words; **Apply to Viseme Track**.
3. Use AccuLips Dictionary for names/rare words; export `.txt` transcript for reuse.
4. Save `Kelly_LipSync_Test.iProject`.

**Metric:** phone alignment drift ≤ **±3 frames** on random spot‑checks.

---

## 6) Facial Nuance — AccuFACE + Motion LIVE
1. `Plugins → AccuFACE → Video Mode` → load HeyGen video → **Calibrate Neutral**.
2. `Plugins → Motion LIVE`:
   - Facial: **AccuFACE (Video)** — **disable Mouth/Jaw** channels.
   - Enable **Brows, Lids, Cheeks, Head**; weights 0.8–1.0 to taste.
3. **Preview** then **Record**. Micro‑edit with **HD Facial Control**.
4. Optional: add **blink generator** clip (12–18/min with variance) if AccuFACE video lacks blinks.
5. Save `Kelly_Hybrid_FacialPass.iProject`.

---

## 7) Render Test
- `Render → Video` → H.264, 1080p/4K, 20 Mbps, 30 fps.  
- Name `kelly.l1.short.v01.mp4`.

---

## Quality Gate (Pass/Fail)
- **Likeness:** 20‑point landmark overlay within 2–3 px at 4K on front & 3/4.
- **Lip‑sync:** ≤ ±3 frames drift on 10 random words.
- **Eyes:** blink rate 12–18/min; eyelid follows eye (no sclera pop‑through); visible tear‑line; catchlight present.
- **Skin:** no pore “swim” under expressions; wrinkle normals engage on strong smiles/brow raise.
- **Teeth/Tongue:** no full‑white frames; occlusion looks natural.
- **Lighting:** no double‑shadows; skin neither plastic nor powdery.

---

## Delivery
- CC5 project(s), iClone scene, and final MP4 in `projects/Kelly/Renders/`.
- Include a **1‑min montage**: neutral → speech segment → strong smile → head turns.
