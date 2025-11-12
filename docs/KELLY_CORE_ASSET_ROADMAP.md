# Kelly Core Asset Roadmap

## 1. Baseline Alignment
- **Unified vision:** Launch Kelly as the single reusable hero asset for all Lesson of the Day properties (Digital Kelly, Curious Kelly, Reinmaker, The Daily Lesson, iLearn) to eliminate duplicate pipelines.
- **Studio look:** Pure white cyclorama floor and walls, single off-camera window casting a crisscross shadow on the back wall to anchor depth and natural light direction.
- **Wardrobe lock:** Soft blue cashmere crewneck sweater with stitched collar detail, fitted dark denim, white sneakers, and a clipped name tag on the left chest; no alternate costumes for v1.
- **Performance stance:** Replace the original runner sprint pose with a comfortable forward walk cycle; locomotion speed and stride tuned for calm confidence.
- **Collectible metaphor:** Apples replace Knowledge Shards; each apple is equivalent to a lore progression marker and must map 1:1 with the Seven Tribes (Light, Stone, Metal, Code, Air, Water, Fire).
- **Non-negotiable constraints (from CLAUDE.md & CK plans):**
  - Precompute EN + ES/FR text/audio for every interactive experience; never rely on runtime translation.
  - Maintain ≥60 minutes of pristine voice training data per persona; synthesize launch VO through ElevenLabs with logged model versions; fall back to Piper only if pre-approved.
  - Ship gameplay and cinematic loops at 60 FPS; profile Unity/Flutter builds on target devices.
  - Reuse existing lesson player/PhaseDNA schemas; do not author new lesson player surfaces.
  - Respect caching/batching rules to avoid duplicate renders or API overruns.

## 2. Asset & Pipeline Definition

### 2.1 Kelly Character Package
- **Source of truth:** Character Creator 4/Unreal-compatible FBX with full facial rig, ARKit blendshapes, and Audio2Face-compatible mesh (per `KELLY_AVATAR_PRODUCTION_SPECS.md`).
- **Materials:** Neutral studio lighting LUT; shader variants for Unity HDRP, Unreal, and WebGL; ensure gamma consistency by baking a white balance LUT tested in `KELLY_AVATAR_WORKFLOW.md`.
- **Textures:** 8K face and hair textures, 4K clothing maps (albedo, normal, roughness); export compact 2K versions for mobile with matching UVs.
- **Wardrobe details:** Knit displacement pass for the cashmere sweater, stitched name tag geometry with changeable label texture (PNG, 512×256) for localization of the “Kelly” badge.
- **Rig & animation hooks:** Maintain existing skeletal naming; retarget walk cycle from mocap or hand-key to 30 frames per stride loop, plus idle, look-around, apple pickup, and celebratory nod.
- **Facial performance:** Micro-expression library triggered via cues defined in PhaseDNA (smile, empathy, curiosity); reuse Audio2Face pipeline to generate visemes and layer gaze micro-saccades (2–4 per second).

### 2.2 White Room Environment
- **Scene setup:** Infinite white floor with subtle AO at foot contact; rear wall positioned 4m behind Kelly with the window shadow gobo projected (2×2 pane grid, sun angle 35°).
- **Lighting presets:** Key directional light (6500K), soft fill at 30% intensity, and rim kicker to mimic daylight wrap. Bake matching HDRI for 2D composites.
- **Shadow fidelity:** Raytraced or high-resolution shadow maps to preserve the crisp window lattice; include contact shadows beneath Kelly and apples for grounding.

### 2.3 Apple Collectibles (Seven Tribes Mapping)
- Create stylized, semi-translucent apples with engraved tribe symbols and color-coded skin. Each apple requires 3D (animation-ready) and 2D icon variants.
- **Mappings:**
  - **Light → Aurora Apple:** Pale gold skin with luminous core; etched eye glyph; emits gentle bloom when collected.
  - **Stone → Granite Crisp:** Slate-gray speckles, mountain etching, subtle weighty pickup sound.
  - **Metal → Alloy Pippin:** Brushed steel sheen, gear symbol, metallic ring FX.
  - **Code → Logic Apple:** Deep teal skin with glowing bracket symbol; particle bits dissolve upward.
  - **Air → Zephyr Gala:** Sky-blue gradient, feather etching, airy chime.
  - **Water → Tidal Fuji:** Cerulean ripple shader, wave crest glyph, droplet splash FX.
  - **Fire → Ember Spice:** Ember-red glow, flame emblem, warm whoosh pickup.
- **Gameplay hooks:** Each apple keeps legacy Knowledge Stone progression (7-unit cycles). Define drop tables and scoring identical to prior shards; update manifest entries to reference apple IDs instead of stones.

### 2.4 Production Pipeline & Ownership
- **Concept lock:** Finalize wardrobe turnarounds and apple color scripts in Figma; approvals stored in `content/art/kelly-core/`.
- **Modeling & texturing:** Character Creator → ZBrush polish → Substance Painter bake; apples modeled in Maya with procedural materials.
- **Rigging & animation:** Use existing Kelly rig as base; author walk/idle/pickup loops in MotionBuilder, bake to Unity Humanoid; validate in Unreal as well.
- **Simulation:** Optional cloth sim test for sweater drape; ensure stable at 60 FPS on mobile builds.
- **Audio & VO:** ElevenLabs batch generation for English primary lines; cache ES/FR alternates; log metadata per `KELLY_BATCH_RUN_CHEAT_CARD.md`.
- **Review gates:**
  1. Geometry & texture review (QA, look dev).
  2. Animation review (walk + pickup, 60 FPS capture).
  3. Integration review (Unity + Flutter + WebGL test scenes).
  4. Localization & VO sync check (multilingual, lip-sync error <5%).

## 3. Cross-Product Integration Tracks

### 3.1 Daily Lesson
- Bundle Kelly’s white-room hero pose and looping idle for lesson intros; preload PhaseDNA cues to trigger micro-expressions and apple reveals during teaching beats.
- Package assets as optimized MP4 (60 FPS) plus transparent WebM overlays for the web lesson player; ensure precomputed ES/FR subtitles accompany each clip.

### 3.2 Curious Kelly Mobile App
- Import Kelly FBX and animations into Unity `curious-kellly/mobile/` project; wire ElevenLabs realtime fallback per Technical Alignment Matrix.
- Implement apple collectibles as 3D UI elements for daily streak rewards; reuse same textures for reward screen and inventory.
- Cache asset bundles locally; version with semantic tags (e.g., `kelly-core-v1.0`).

### 3.3 Digital Kelly (Realtime Conversational Avatar)
- Use the same white-room lighting LUT for consistency; feed walk and idle states into Audio2Face-driven rig while maintaining microphone-sync gestures.
- Ensure name tag texture swaps correctly when switching between English, Spanish, and French greeting routines.

### 3.4 Reinmaker Runner (Walking Edition)
- Replace runner sprint animation with Kelly walk loop; adjust navmesh speed and collider height accordingly.
- Swap Knowledge Stone assets with apple pickups; reuse existing spawn logic and scoring tables, updating VFX/audio triggers to new apple IDs.
- Update lore UI panels to explain apple-to-tribe mapping; keep quest manifests unchanged aside from collectible references.

### 3.5 I Learn Hardware Experiences
- Export lightweight GLB and Lottie sequences for low-power displays; maintain white-room look with baked lighting.
- Provide kiosk-ready loops (30s) featuring Kelly walking and presenting apple collectibles; ensure offline cache due to limited connectivity.

## 4. Launch & Validation Milestones

| Milestone | Scope | Validation Checklist |
|-----------|-------|----------------------|
| **M1 – Vision Lock (Week 0-1)** | Approve moodboards, wardrobe, apple palette | Signoff from creative brief, confirm CLAUDE.md compliance, document apple-tribe mapping |
| **M2 – Asset Production (Week 2-4)** | Finalize Kelly model updates, apple models/icons, white-room scene | Geometry QA, texture compression tests (8K→2K), shader profiling |
| **M3 – Animation & VO (Week 4-5)** | Walk/idle/pickup loops, facial cue set, multilingual VO batches | 60 FPS capture review, lip-sync drift <5%, audio loudness targets (-16 LUFS) |
| **M4 – Integration Sprint (Week 6-7)** | Plug assets into Daily Lesson (web), Curious Kelly (mobile), Reinmaker (runner), Digital Kelly (realtime) | Build verification on iPhone 12/Pixel 6, WebGL regression tests, asset bundle hash checks |
| **M5 – Cross-Product QA (Week 8)** | Unified playtest including apple progression, localization, analytics hooks | Schema validation for PhaseDNA, telemetry events firing, caching/batching audit |
| **M6 – Launch Prep (Week 9)** | Finalize packaging, release notes, marketing capture | Create hero renders, record gameplay clips, confirm deployment scripts |
| **M7 – Post-Launch Iteration (Week 10+)** | Monitor metrics, schedule updates | Track retention (D1 45%, D30 20%), gather apple-themed feedback, plan variant costumes |

### Success Metrics & Telemetry
- **Engagement:** Runner session length 5–7 minutes; apple collection completion rate ≥80% per run.
- **Consistency:** Visual continuity of Kelly across products; measure brand recall via surveys.
- **Performance:** Maintain 60 FPS on target devices; asset bundle download <80 MB for mobile.
- **Localization:** Zero missing ES/FR strings; VO coverage 100% with correct name tag labels.

### Post-Launch Backlog Seeds
- Seasonal wardrobe overlays (scarf, jacket) that preserve core blue sweater identity.
- Additional apple varieties for future tribes or events without altering core seven mapping.
- Expanded interaction set (hand wave, seated conversation) for cross-product storytelling.








