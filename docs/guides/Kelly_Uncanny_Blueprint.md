
# Kelly Uncanny‑Valley Blueprint — Technical Plan & Requirements

## 1) Goal
Deliver a photoreal, personable digital human whose perceived naturalness scores **≥ 4.2/5** (Likert 'looks/feels real') in blinded tests across 30+ viewers and passes quantitative animation and shading checks.

---

## 2) Pillars & Requirements

### A. Geometry & Likeness
- Source: high‑res front + 3/4 references; optional multi‑angle scan.
- Head topology: **CC5 HD** with HS2 wrap; **Render SubD 2** (3 for hero).
- **FACS‑aligned rig**: 45–60 primary expressions; add **≥ 20 corrective shapes** for extreme vowels (AA, EE) + smiles + squints to prevent mesh collapse.
- Landmark QA: average error ≤ 3 px at 4K on front/3/4.

### B. Materials & Shading
- **Skin (DHS)**: base, normal, micro‑normal; roughness map with T‑zone variation; SSS ~0.28 ±0.05.
- **Wrinkle normals** tied to expression curves (if available).
- **Eyes**: separate cornea with bulge; **tear‑line mesh**; iris parallax; sclera veins at 2–4K.
- **Teeth/Tongue**: dedicated roughness/spec; AO at gum line; translucency for gums.
- **Hair**: multi‑card style + baby hairs; physics simmer ≤ 0.3 cm amplitude.

### C. Animation & Performance
- **Lip‑sync**: AccuLips visemes from WAV; custom dictionary; coarticulation smoothing 0.2–0.4.
- **Facial nuance**: AccuFACE VIDEO driving brows/lids/cheeks/head; mouth/jaw disabled.
- **Eye behavior**: 
  - Blinks: 12–18/min, 120–200 ms closure; occasional longer rest blink.
  - Saccades: small gaze shifts every 1–3 s; micro head‑nod coupling 1–2°.
- **Breathing**: 5–9 cycles/min subtle chest/shoulder motion.
- **Secondary**: slight pupil dilation on excitement (+3–5%).

### D. Lighting & Camera
- Portrait baseline: **85 mm**, eye‑level, soft 3‑point; DOF on irises.
- HDRI neutrality for color checks; clamp key light specular on forehead < 0.9.

### E. Rendering & Color
- Review: **Realtime PBR** (iClone 8.62) with TAA; **Rec.709** export.
- Hero option: **Iray plug‑in** (if licensed) for stills; match color via LUT.
- Motion blur: on for head turns; shutter 180° equiv.

### F. Runtime/Distribution (iLearn)
- Delivery options:
  1) **Pre‑rendered** MP4 for low‑spec devices (target 8–12 Mbps 1080p).
  2) **Interactive** (Windows): keep Viewport SubD 1; baked wrinkle normals; hair physics off; cap draw calls < 300.

---

## 3) QA Scorecard

| Category | Metric | Target |
|---|---|---|
| Likeness | 20‑landmark avg px error (4K) | ≤ 3 px |
| Lip‑sync | Forced alignment drift | ≤ ±3 frames |
| Eye realism | Blink rate variance | 12–18/min + randomness |
| Eye‑lid coupling | % frames with sclera pop‑through | 0% |
| Skin | Pore swimming under expression | None |
| Teeth/Tongue | Over‑white/black frames | None |
| Perception | Mean “looks/feels real” score | ≥ 4.2/5 |

Procedure: 30‑viewer blind A/B (real video vs Kelly) with randomized 15‑sec clips. Record comments for failure analysis.

---

## 4) Pipeline Diagram
Audio (ElevenLabs WAV) → **AccuLips** → visemes  
HeyGen video → **AccuFACE (Motion LIVE)** → brows/lids/cheeks/head  
HS2 head (CC5 HD) → **DHS materials** → iClone scene (85 mm, DOF) → **Render**

---

## 5) Risks & Mitigations
- **Mouth double‑driven** (AccuLips + AccuFACE) → disable mouth/jaw in Motion LIVE.
- **Eye glassiness** → ensure tear‑line on; add catchlight; clamp specular.
- **Plastic skin** → raise roughness in T‑zone; verify SSS radius.
- **Viseme popping** → smooth strength 0.2–0.4; add corrective shapes.

---

## 6) Deliverables
- CC5 project(s), iClone scenes, textures (8K), MP4 tests, QA scorecard CSV, and a 1‑page summary with viewer feedback.
