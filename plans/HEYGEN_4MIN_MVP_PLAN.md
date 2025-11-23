## HeyGen 4-Minute MVP Plan

### Goal
Ship a reusable 4-minute library of Kelly clips rendered with HeyGen Video Avatar 4 that can be recombined with existing ElevenLabs audio via viseme-aware mapping.

---

### Phase 1 – Pre-Production (This Week)
1. **Script grid**  
   - Author 12 × 20 s scripts that jointly cover every major English phoneme + key emotional beats (welcome, excitement, instruction, reassurance, CTA, curiosity, celebration, reflection).  
   - Include stage directions (gaze, gestures) to keep continuity.

2. **Audio sources**  
   - Generate clean ElevenLabs WAV for each script (`kelly_avatar_v3`, 48 kHz).  
   - Store under `assets/kelly_clips/v1/audio/<clip_id>.wav` with metadata JSON capturing text + phoneme timeline.  
   - If ElevenLabs quota is tight, record scratch VO locally and replace later.

3. **Viseme reference**  
   - Run scripts through the existing viseme extractor (`tools/viseme_export.py`) to capture phoneme–timestamp CSV per clip.  
   - Files live next to audio (`<clip_id>_visemes.json`).

Deliverables: script list, audio files, viseme metadata.

---

### Phase 2 – HeyGen Rendering (Next Week)
1. **API configuration**
   - Set `HEYGEN_API_KEY`, `HEYGEN_AVATAR_ID`, `HEYGEN_VOICE_ID` in `.env.local`.
   - Confirm monthly quota ≥ 4 minutes (12 × 0.02 hr = 4 min).

2. **Automation script**
   - Extend `scripts/generate_heygen_videos.py` with a `--script-manifest` flag that ingests `assets/kelly_clips/v1/scripts.json` and sends custom audio/text clips instead of per-lesson audio.
   - Ensure quota tracker resets once recordings finish (store `last_reset` as YYYY-MM in `.heygen_quota.json`).

3. **Render & download**
   - Submit one job per clip (12 total).  
   - Poll every 2 minutes; auto-download MP4 into `assets/kelly_clips/v1/video/<clip_id>.mp4`.  
   - Capture HeyGen `video_id`, render duration, checksum.

Deliverables: 12 MP4 clips, updated quota log, generation log.

---

### Phase 3 – Storage & Mapping (Same Week)
1. **Manifest schema**
   - Define `assets/kelly_clips/v1/manifest.json`:  
     ```json
     {
       "version": "1.0",
       "clips": [
         {
           "id": "clip01_warm_welcome",
           "duration": 19.6,
           "video_path": "assets/kelly_clips/v1/video/clip01.mp4",
           "audio_path": "assets/kelly_clips/v1/audio/clip01.wav",
           "visemes": "assets/kelly_clips/v1/audio/clip01_visemes.json",
           "dominant_phonemes": ["EH", "AE", "L", "R"],
           "emotion": "warm",
           "pose": "center_frontal"
         }
       ]
     }
     ```
   - Include MD5 hash + HeyGen `video_id` for auditing.

2. **Player hook**
   - Add loader utility in `curious-kellly/frontend` to fetch manifest and cache MP4s.  
   - Map lesson beats → clip IDs using viseme cosine similarity (existing viseme pipeline from `VISEME_INTEGRATION_SETUP_GUIDE.md`).  
   - Add 6-frame cross-fade to smooth clip seams.

Deliverables: manifest JSON, loader module, matching utility tests.

---

### Phase 4 – QA & Validation
1. **Visual review**
   - Watch stitched sequences for three lessons (science, arts, emotional) to catch lighting/framing drift.
   - Log issues in `QA/heygen_clip_review.md`.

2. **Lip-sync check**
   - Compare ElevenLabs waveform vs. clip viseme durations; adjust playback offsets ±30 ms as needed.

3. **Performance**
   - Measure load time and memory use when caching all 12 clips (target <200 MB RAM, streaming friendly).

Deliverables: QA notes, performance stats, fixes applied.

---

### Dependencies & Risks
- Requires valid HeyGen API key and Avatar 4 access.
- ElevenLabs generation must match HeyGen phoneme expectations; mismatched pacing will reduce lip-sync quality.
- Monthly quota (5 min) leaves ~1 min buffer only—avoid rerenders without reset.
- Storage footprint ~150 MB; ensure CDN bucket or local cache has space.

---

### Success Criteria
- 12 high-quality Kelly clips totaling 4 minutes rendered and cataloged.
- Manifest-driven playback successfully swaps clips for at least one full lesson sequence without noticeable seams.
- Documentation and scripts allow regenerating or extending the clip library within a single work session.








