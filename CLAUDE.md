## CLAUDE.md — Operating Rules for AI Contributions

This document encodes how the assistant must operate in this repository to maximize learner experience, speed, and daily habit formation for lifelong learners, while protecting quality, cost, and safety. It applies across the repo, with subproject specifics called out where relevant.

### How to use this doc
- If you are onboarding: read `START_HERE.md` → then skim this file top-to-bottom.
- When acting: follow this file’s rules; for concrete steps, jump to:
  - `CURIOUS_KELLLY_EXECUTION_PLAN.md` → Executive Summary, “SPRINT 0: Foundation,” “SPRINT 1: Voice & Avatar,” “SPRINT 2: Content Creation.”
  - `TECHNICAL_ALIGNMENT_MATRIX.md` → “Voice & Audio Pipeline,” “Avatar Rendering,” “Lesson System,” “Backend Services.”
  - `BUILD_PLAN.md` → “Phase 1: Generate Audio,” “Phase 2: Update Lesson Player,” “Phase 5: Testing & Validation.”
  - `CURIOUS_KELLLY_INDEX.md` → navigation to all living documents.
  - `docs/reinmaker/API_OVERVIEW.md` → Reinmaker endpoints, OAuth scopes, webhook contracts.
  - `docs/billing/GLOBAL_ROADMAP.md` → unified billing responsibilities and global payment rollout.
  - `docs/web/SITE_MAP.md` → domain roles, cross-link rules, analytics instrumentation.
  - `content/manifests/reinmaker/README.md` → overlay/FX manifest schema and validation pipeline.

### Golden objectives (never trade off without approval)
- Learner experience: consistent, low-friction, clearly guided lessons with zero surprises.
- Speed: fast startup, sub-second UI navigations, GPU-accelerated media where applicable.
- Daily habits: effortless continuity, micro-sessions, streaks, reminders, and clear progress.

### Authority and boundaries
- Allowed: voice model training; precomputing multilingual DNA/content; wiring to existing players; asset preloading/caching; validators/schemas/tests; analytics; CI/CD automation.
- Forbidden: creating new lesson players or pages; using browser TTS; degrading or shrinking training datasets; deleting or moving content/data without explicit approval; interest-driven lesson selection; learner “learning-style” classification.
- Plan adherence: Follow `CURIOUS_KELLLY_*`, `BUILD_PLAN.md`, `TECHNICAL_ALIGNMENT_MATRIX.md`, and deployment guides. If a conflict is found, flag it in the PR and defer to those documents.

### Non‑negotiable invariants
- Languages are precomputed in every DNA/content file (EN + ES/FR). No runtime language generation.
- Minimum 60 minutes of training audio per voice model (e.g., Kelly, Kyle). Never downsample or shrink sets.
- Never use browser TTS. Prefer ElevenLabs; only use local Piper if explicitly configured by the user.
- Respect rate limits and costs; batch requests; cache and reuse assets; avoid duplicate renders.
- Target 60 FPS for applicable media; adhere to existing render/export presets.

### Interaction style and output contract
- Provide step‑by‑step, click‑by‑click instructions for any user-facing procedure.
- Begin work with a brief status note; end with a concise summary of changes/next steps.
- Maintain a small TODO plan for multi‑step changes; update statuses as you proceed.
- Never expose secrets or token values; redact in logs and diffs.

### Ownership model (single-operator)
- There are no separate teams or stakeholders. The assistant is responsible for every software, billing, web, and game deliverable in this repository.
- Assume full-stack ownership: author specs, implement code, run validators/tests, and manage deployments for all products (Reinmaker, The Daily Lesson, Curious Kelly, iLearnHow hardware).
- Escalate only to the user for approvals mandated elsewhere in this file (e.g., schema changes, cost increases). Otherwise, make decisions and proceed.

### Repo map and canonical commands (do not invent new entry points)
- Setup: `setup_local.ps1` → verify with `verify_installation.py`
- Audio/VO: `curious-kellly/backend` scripts; `generate_lesson_audio_for_iclone.py`
- Training: `gpu_optimized_trainer.py` and `synthetic_tts/` tooling per docs
- Deployment: `deployment/setup-cloud.sh`; `deployment/vercel.json` and Cloudflare rules
- Diagnostics: `tests/`, `test_*.py`, and validation tools in `tools/` and `curious-kellly/content-tools`

### Workflows (must follow exactly)
1) Voice training (Kelly/Kyle)
   - Ensure ≥60 minutes curated audio per voice; confirm sample rate/format per docs.
   - Use ElevenLabs for synthesis aligned to trained voices. Log model versions.
   - Never compress or trim datasets; do not relabel without explicit approval.
   - See: `TECHNICAL_ALIGNMENT_MATRIX.md` → “Voice & Audio Pipeline”.

2) Audio generation pipeline
   - Generate VO via ElevenLabs; cache responses; batch API calls.
   - Validate duration, sample rate, and read‑along sync markers.
   - For A2F/iClone, ensure GPU and drivers present before proceeding; else skip with clear messaging.
   - See: `BUILD_PLAN.md` → “Phase 1: Generate Audio Files”; `60FPS_SETUP_GUIDE.md` for A2F specifics.

3) PhaseDNA content authoring
   - Encode phases: welcome → Q phases → wisdom; include cues, timing, and no‑choice hints.
   - Precompute EN + ES/FR in each DNA file. Validate against JSON Schemas.
   - Show variant badges and ensure inspectors (live state/content/DNA) remain functional.
   - See: `CURIOUS_KELLLY_EXECUTION_PLAN.md` → “0.3 Lesson Planner Migration,” “Sprint 2: Content Creation”; `CURIOUS_KELLLY_INDEX.md` → Content Creator path.

4) Asset preloading and caching
   - Preload next‑phase assets; reuse across variants; avoid re‑render loops.
   - Hash and version outputs; store seeds/config for determinism.
   - See: `TECHNICAL_ALIGNMENT_MATRIX.md` → “Lesson System,” “Avatar Rendering.”

5) Deployment and release
   - Cloud deploys only from approved branches per `deployment/` configs.
   - Include change notes about cost implications, new assets, and schema updates.
   - See: `CURIOUS_KELLLY_EXECUTION_PLAN.md` → “SPRINT 7: Store Submission & Launch,” and `deployment/` docs.

### Safety rails and approvals
- Require explicit user approval for: schema changes; DNA structure edits; deletion/moves of assets; production config changes; any action that increases recurring costs.
- Secrets: keep keys in `.env`/secure store; never commit or print. Redact in logs and diffs.

### Testing and quality gates (pre‑merge requirements)
- Run unit tests in `tests/` and subproject suites.
- Lint/type/style checks where configured; fix violations.
- Media validation: duration, sample rate, format, 60 FPS where applicable, sync markers.
- DNA/content validation: pass JSON Schema; verify multilingual completeness (EN/ES/FR present).
- See: `BUILD_PLAN.md` → “Phase 5: Testing & Validation”; `CURIOUS_KELLLY_EXECUTION_PLAN.md` → “SPRINT 5: Analytics & Testing.”

### Performance and 60 FPS expectations
- Favor GPU offload for Audio2Face and media transforms; detect and communicate when unavailable.
- Batch operations and parallelize safe steps; throttle to respect rate limits.
- Keep UI paths sub‑second and file sizes optimized per presets.
- See: `60FPS_SETUP_GUIDE.md`; `TECHNICAL_ALIGNMENT_MATRIX.md` → “Avatar Rendering.”

### Integrations and environment constraints
- ElevenLabs for high‑quality synthesis; match Kelly/Kyle training personas.
- NVIDIA Audio2Face and iClone pipelines require proper GPU/driver setup.
- Cloudflare/Vercel deployment as configured; do not alter without approval.
- Reinmaker integrations follow `docs/reinmaker/API_OVERVIEW.md`; shared manifests live under `content/manifests/reinmaker/`.
- Billing flows must align with `docs/billing/GLOBAL_ROADMAP.md`. Web presence changes must respect `docs/web/SITE_MAP.md`.

### Daily habit reinforcement (experience rules)
- Default to micro‑sessions (5–10 min) with resume‑from‑last‑phase.
- Maintain streaks and progress; offer opt‑in reminders; integrate right‑rail calendar (y/y/t) and search/settings.
- Keep cognitive load low: show current variant badge; keep live state visible to avoid surprises.
- See: `CURIOUS_KELLLY_EXECUTION_PLAN.md` → “Success Metrics,” “Daily Lesson pipeline.”

### Pitfalls to avoid
- Runtime language generation; interest‑driven lesson selection; learner “learning‑style” personalization.
- Creating/replacing lesson players or pages.
- Re‑render loops and duplicate API calls; missing asset cache keys.
- Any dataset degradation (compression, trimming, or filtering down) without explicit approval.

### Change management and precedence
- This document governs assistant behavior. If it conflicts with `CURIOUS_KELLLY_*`, `BUILD_PLAN.md`, or deployment guides, defer to those and flag the discrepancy.
- Propose diffs with clear impact notes (cost, assets, schema). Do not bypass review gates.
- Primary references: `CURIOUS_KELLLY_EXECUTION_PLAN.md` (roadmap authority), `TECHNICAL_ALIGNMENT_MATRIX.md` (asset-to-requirement mapping), `BUILD_PLAN.md` (prototype lineage).

### Quick checklist before you start a task
1. Is there an approved plan reference (file/section) for this change?
2. Are languages precomputed and schemas defined/validated?
3. Will any step increase costs or touch production configs (requires approval)?
4. Are GPU/driver requirements satisfied for A2F/iClone tasks?
5. Do you have caching/batching to avoid duplicates and cost loops?
6. Do tests, linters, and media validators pass locally?
7. Does the outcome help daily use: faster entry, resume state, progress/streak integrity?


