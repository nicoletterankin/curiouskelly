# Performance Validation Playbook

## Goal
Guarantee 60 fps animation, frame-accurate lip-sync, and sub-600 ms voice round-trip for every Kelly persona before shipping daily lessons.

## Required Artifacts
- `analytics/Kelly/perf-baseline.csv` – Unity profiler export for each persona scene
- `analytics/Kelly/voice-latency.csv` – Backend latency log (request start/end, RTT)
- Created automatically when voice metrics first log; includes hashed session ids only.
- `analytics/Kelly/qa/persona-<bucket>.md` – Manual QA checklists per age bucket
- `analytics/Kelly/daily-report.json` – Nightly regression summary
- Audio + viseme caches stored under `curious-kellly/backend/config/audio/<topic>/<age>` and `kelly_audio2face/output/<persona>/`

## Validation Pipeline
1. **Baseline Perf Capture**
   - Follow `60FPS_SETUP_GUIDE.md` to render each persona scene in Unity 2022.3 LTS.
   - Record GPU/CPU frame timings; export metrics to `perf-baseline.csv`.

2. **Voice & Lip-Sync Prep**
   - Generate ElevenLabs reference clips (≥90 s) for welcome, main, and wisdom segments.
   - Run Audio2Face batch scripts to build viseme caches and bind inside Unity.

3. **Realtime Instrumentation**
   - `backend/src/services/voice.js` logs request start/end timestamps, Kelly persona, and calculated RTT for every interaction; extend the same pattern to `session.js` when persistence is added.
   - Append entries to `voice-latency.csv` for every interaction.

4. **Automated Performance Tests**
   - Add Unity playmode test `digital-kelly/engines/tests/Perf60Test.cs` that asserts 58+ fps for all clips.
   - Wire test + backend latency unit tests into CI (GitHub Actions or local `Makefile`).

5. **Manual Persona QA**
   - Drive an end-to-end session (backend + Flutter + Unity) for each age bucket.
   - Capture screen recording, backend logs, and write findings to `qa/persona-<bucket>.md`.

6. **Nightly Regression Sweep**
   - Schedule `scripts/run_perf_suite.ps1` to replay all personas nightly.
   - Emit `daily-report.json` with fps, latency, and safety status; alert on regressions.

## Ship Gate
Release is blocked until:
- All automated tests pass locally and in CI.
- Latest nightly report shows ≥60 fps and RTT ≤600 ms (p50) across personas.
- QA checklists contain sign-off for each age bucket (including safety validation).

Maintain this file as the canonical reference for performance + audio validation work.



