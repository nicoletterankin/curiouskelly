# Kelly CC5 Character Roadmap

Persistent home for Kelly CC5 build phases, with goals and deliverables you can reference quickly. Source path conventions assume assets live under `C:\Kelly\CC5\projects\Kelly_Character\`.

## Phase 0 — Source & Project Hygiene
- **Goal**: A clean working set you can reuse forever.
- **Deliverables**:
  - `C:\Kelly\CC5\projects\Kelly_Character\00_ref\` — front/profile PNGs (attach color-calibrated baselines for both angles).
  - `C:\Kelly\CC5\projects\Kelly_Character\01_headshot\` — Character Creator 5 projects (headshot-authoring checkpoints).
  - `C:\Kelly\CC5\projects\Kelly_Character\02_mesh\` — FBX/OBJ exports of the working mesh.
  - `C:\Kelly\CC5\projects\Kelly_Character\03_textures\` — 4k/8k texture sets (Albedo/Normal/Roughness/Displacement).
  - `C:\Kelly\CC5\projects\Kelly_Character\04_hair\` — approved grooms/presets.
  - `C:\Kelly\CC5\projects\Kelly_Character\05_actorMixer\` — ActorCore/ActorMixer recipes and presets.
  - Baseline color-calibrated images (front/profile) stored alongside the reference set.

## Phase 1 — Identity Base (Two Valid Starting Points)
- **Goal**: Produce two dependable starting identities inside CC5 before branching into variants.
- **Path A — Headshot Images (recommended)**:
  - Switch Headshot to *Headshot 2 Image* mode.
  - Load the calibrated front headshot (the bald front capture) and keep the profile image visible as a guide.
  - Let Auto Landmarks run to generate the initial head.
  - Use Morph sliders (±1–2%) to refine features to your references—stay within the tolerance to preserve topology.
  - Save the project as `Kelly_HS2_ImageBase.ccProject` under `01_headshot\`.





