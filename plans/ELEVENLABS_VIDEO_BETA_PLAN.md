## ElevenLabs Video Beta Plan

### Goal
Replace the HeyGen-based clip experiment with a short library of high-quality Kelly clips rendered via ElevenLabs Video Beta, then hook them into the lesson player using the same manifest/viseme approach.

---

### Phase 0 – Access & Environment
1. **Beta enrollment**  
   - Ensure the ElevenLabs account has Video Beta enabled (dashboard → Labs → Video).  
   - Confirm Kelly avatar assets are uploaded (photo or mesh per ElevenLabs instructions).
2. **Credentials**  
   - Add `ELEVEN_API_KEY` and, if needed, `ELEVEN_VIDEO_MODEL_ID` to `.env.local`.  
   - Update `tools/eleven_video/config.json` with avatar reference + style presets.

---

### Phase 1 – Script + Audio Prep (Reuse where possible)
1. Reuse `assets/kelly_clips/v1/scripts.json`.  
2. Regenerate ElevenLabs **audio** per script using the Kelly voice to ensure timing matches the eventual video (Video Beta can ingest external audio).  
3. Export viseme data via existing pipeline for later mapping.

Deliverables: WAV + viseme JSON per clip stored under `assets/kelly_clips/v1/audio/`.

---

### Phase 2 – ElevenLabs Video Rendering
1. **Automation helper**  
   - Create `scripts/generate_eleven_video_clips.py` that reads `scripts.json`, uploads the corresponding WAV, and hits the Video Beta endpoint with:  
     ```
     POST /v1/video/generate
     {
       "model_id": "...video-beta...",
       "audio": "<wav>",
       "avatar_id": "<kelly_avatar>",
       "style_preset": "studio_warm"
     }
     ```
   - Store response IDs in `assets/kelly_clips/v1/generation_log.json`.
2. **Manual fallback**  
   - If automation is blocked, user can upload audio+script via the ElevenLabs UI; we document naming convention for downloaded MP4s.
3. **Download & catalog**  
   - Save MP4 to `assets/kelly_clips/v1/video/<clip_id>.mp4`, capture duration + checksum.

---

### Phase 3 – Manifest + Player Integration
1. Update `assets/kelly_clips/v1/manifest.template.json` with actual metadata (video IDs, durations, phoneme tags).  
2. Build loader in lesson player (`lesson-player/src/utils/kellyClips.ts`) that:
   - Loads manifest at startup.
   - Matches requested viseme profile with best clip.  
   - Handles preloading, looped playback, and cross-fades.

---

### Phase 4 – QA
1. **Visual checks**: confirm lighting/pose continuity, lip-sync accuracy, no artifacts.  
2. **Performance**: measure load time & memory; target <150 MB for 12 clips.  
3. **User flow**: run at least one full lesson using the new clips and log issues.

---

### Risks & Mitigations
- **Beta stability**: keep retry logic + local cache of rendered MP4s.  
- **Lip-sync mismatch**: since we control the audio, match sample rate and trim silence before upload.  
- **Quota/cost**: track minutes rendered in `assets/kelly_clips/v1/quota.json`.

---

### Next Immediate Tasks
1. Confirm ElevenLabs Video Beta access + avatar readiness.  
2. Build/adjust the automation script (or prep manual instructions) to submit the 12 clips.  
3. Once renders succeed, continue with manifest population and player integration.




