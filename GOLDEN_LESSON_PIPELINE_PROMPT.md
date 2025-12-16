# 🎯 GOLDEN LESSON PIPELINE — Build Specification

> **To:** Fresh Claude Session  
> **From:** Previous Claude (who failed)  
> **Date:** December 9, 2025  
> **Mission:** Build ONE perfect, replicable lesson that works end-to-end, then scale to 365 days.

---

## YOUR FAILURES TO AVOID

Previous Claude made these mistakes:
1. Generated videos that no player uses
2. Built pipelines without verifying output quality
3. Created analysis documents instead of working code
4. Said "Day 1 is perfect" without testing it
5. Gave the user "options" instead of solutions

**DO NOT REPEAT THESE MISTAKES.**

---

## THE GOAL

Create a **single working lesson** where:
1. User opens `curiouskelly.com/learn`
2. Kelly VIDEO appears and speaks the lesson script
3. User clicks through 5 phases (Hook → Fact1 → Fact2 → Fact3 → Wisdom)
4. Each phase plays a different HD lipsync video of Kelly
5. The experience is polished, professional, and delightful

Then replicate this for 365 days × 3 archetypes = 5,475 videos.

---

## PART 1: THE DATA MODEL

### Supabase Tables You Need

```
core_lessons (365 rows)
├── id: uuid
├── day_number: 1-365
├── topic: "Starting Fresh"
└── universal_truth: "Fresh starts provide psychological permission to change"

lesson_atoms (20,341 rows)
├── id: uuid
├── core_lesson_id: fk → core_lessons.id
├── archetype: "The Explorer" | "The Rebel" | "The Scientist" (use only these 3 for now)
├── phase: "Hook" | "Fact1" | "Fact2" | "Fact3" | "Wisdom"
├── content: jsonb { script: "Kelly's spoken words..." }
├── hd_video_url: "https://supabase.../kelly-videos/day-001/explorer/hook.mp4" ← THIS IS THE OUTPUT
└── visual_url: (unused for now)
```

### The Critical Field

**`lesson_atoms.hd_video_url`** — This is where the generated video URL goes. The lesson player reads this field to know what video to play.

### Query to Get Lesson Data

```sql
SELECT 
  cl.day_number,
  cl.topic,
  cl.universal_truth,
  la.archetype,
  la.phase,
  la.content->>'script' as script,
  la.hd_video_url
FROM lesson_atoms la
JOIN core_lessons cl ON la.core_lesson_id = cl.id
WHERE cl.day_number = 1 
  AND la.archetype IN ('The Explorer', 'The Rebel', 'The Scientist')
ORDER BY la.archetype, 
  CASE la.phase 
    WHEN 'Hook' THEN 1 
    WHEN 'Fact1' THEN 2 
    WHEN 'Fact2' THEN 3 
    WHEN 'Fact3' THEN 4 
    WHEN 'Wisdom' THEN 5 
  END;
```

---

## PART 2: THE VIDEO GENERATION PIPELINE

### Pipeline Location
`scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts`

### Pipeline Architecture

```
INPUT: lesson_atoms row (script, archetype, phase)
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: AUDIO GENERATION (ElevenLabs)                            │
│                                                                  │
│ API: https://api.elevenlabs.io/v1/text-to-speech/{voice_id}     │
│ Model: eleven_multilingual_v2                                    │
│ Voice ID: env.ELEVENLABS_KELLY_VOICE_ID                         │
│                                                                  │
│ Voice Settings by Archetype:                                     │
│   Explorer: stability=0.45, similarity=0.85, style=0.25         │
│   Rebel:    stability=0.40, similarity=0.85, style=0.35         │
│   Scientist: stability=0.55, similarity=0.85, style=0.15        │
│                                                                  │
│ Output: MP3 file uploaded to Supabase kelly-templates bucket     │
│ Cost: ~$0.01-0.03 per generation                                 │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: SOURCE IMAGE (Replicate Flux + Kelly LoRA)               │
│                                                                  │
│ Model: lucataco/flux-dev-lora                                    │
│ LoRA: https://huggingface.co/CuriousKellycom/curious-kelly-lora │
│ LoRA Scale: 0.90                                                 │
│                                                                  │
│ KELLY IDENTITY PROMPT (LOCKED - DO NOT CHANGE):                  │
│ "kelly, calm confident female teacher, warm brown wavy           │
│ shoulder-length hair with subtle caramel highlights              │
│ center-parted, hazel-brown eyes with steady direct gaze,         │
│ soft natural features, light natural makeup, wearing soft        │
│ powder blue cashmere crewneck sweater, poised composed posture,  │
│ looking directly at camera"                                      │
│                                                                  │
│ Phase Expressions:                                               │
│   Hook:   "warm confident expression, direct eye contact"        │
│   Fact1:  "calm curious expression, engaged teaching face"       │
│   Fact2:  "warm teaching expression, direct eye contact"         │
│   Fact3:  "knowing confident expression, warm direct gaze"       │
│   Wisdom: "warm sincere expression, soft empathetic gaze"        │
│                                                                  │
│ NEGATIVE PROMPT (LOCKED):                                        │
│ "pink sweater, red sweater, purple sweater, teal sweater,        │
│ green sweater, yellow sweater, beige sweater, auburn hair,       │
│ deformed, blurry, bad anatomy, wandering eyes, looking away"     │
│                                                                  │
│ Output: 1344x768 PNG (16:9 aspect ratio)                         │
│ Cost: ~$0.02 per generation                                      │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: MOTION VIDEO (Replicate MiniMax Video-01)                │
│                                                                  │
│ Model: minimax/video-01                                          │
│ Input: Source image from Step 2                                  │
│                                                                  │
│ Motion Prompt:                                                   │
│ "Professional female teacher speaking to camera. She is TALKING  │
│ and her mouth is moving naturally as she speaks. Steady direct   │
│ eye contact with camera. Natural breathing, soft blinking.       │
│ Smooth cinematic quality. CRITICAL: Mouth must open and move     │
│ naturally while speaking. Eyes stay focused on camera.           │
│ AVOID: closed mouth, frozen face, wandering eyes."               │
│                                                                  │
│ Output: ~6 second MP4 video with movement                        │
│ Processing Time: 2-5 minutes                                     │
│ Cost: ~$0.10-0.15 per generation                                 │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: LIP-SYNC (Sync Labs API)                                 │
│                                                                  │
│ API: https://api.sync.so/v2/generate                            │
│ Model: lipsync-2-pro                                             │
│ Input: Motion video (Step 3) + Audio (Step 1)                    │
│                                                                  │
│ Fallback: Replicate wav2lip if Sync Labs unavailable             │
│                                                                  │
│ Output: Final HD video with synced lip movements                 │
│ Processing Time: 3-5 minutes                                     │
│ Cost: ~$0.15-0.25 per generation                                 │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│ STEP 5: UPLOAD & DATABASE UPDATE                                 │
│                                                                  │
│ Storage: Supabase bucket "kelly-videos"                          │
│ Path: day-{XXX}/{archetype}/{phase}.mp4                         │
│ Example: day-001/explorer/hook.mp4                               │
│                                                                  │
│ Database Update:                                                 │
│   UPDATE lesson_atoms                                            │
│   SET hd_video_url = '{public_url}'                             │
│   WHERE core_lesson_id = '{lesson_id}'                          │
│     AND archetype = '{archetype}'                               │
│     AND phase = '{phase}';                                      │
│                                                                  │
│ Verification: Confirm URL is accessible via HTTP GET             │
└──────────────────────────────────────────────────────────────────┘

OUTPUT: lesson_atoms.hd_video_url populated with working video URL
```

### Required Environment Variables

```bash
REPLICATE_API_TOKEN=r8_...        # For Flux + MiniMax
ELEVENLABS_API_KEY=sk_...         # For Kelly voice
ELEVENLABS_KELLY_VOICE_ID=wAdym...# Kelly's trained voice
SYNC_LABS_API_KEY=...             # For lip-sync (optional, has fallback)
PUBLIC_SUPABASE_URL=https://...   # Supabase project URL
SUPABASE_SERVICE_ROLE_KEY=eyJ...  # Supabase admin key
```

### Cost Per Video
- Audio: $0.02
- Image: $0.02
- Motion: $0.12
- Lipsync: $0.20
- **Total: ~$0.36 per video**

### Cost for Full Generation
- Day 1 only (15 videos): $5.40
- All 365 days (5,475 videos): ~$1,970

---

## PART 3: THE LESSON PLAYER

### Player Requirements

The player MUST:
1. Fetch lesson data from Supabase (topic, universal_truth, atoms with video URLs)
2. Display Kelly video for current phase
3. Play video with audio on user action
4. Allow navigation between phases
5. Handle archetype selection (Explorer, Rebel, Scientist)
6. Work on mobile and desktop

### Player Location
Create: `public/learn.html` (or use existing if it works)

### Player Data Flow

```javascript
// 1. Get day number (from URL or localStorage)
const dayNumber = parseInt(new URLSearchParams(window.location.search).get('day')) || 1;

// 2. Get archetype (from localStorage or default)
const archetype = localStorage.getItem('kelly_archetype') || 'The Explorer';

// 3. Fetch lesson data from Supabase
const { data: lesson } = await supabase
  .from('core_lessons')
  .select('id, topic, universal_truth')
  .eq('day_number', dayNumber)
  .single();

// 4. Fetch atoms for this lesson + archetype
const { data: atoms } = await supabase
  .from('lesson_atoms')
  .select('phase, content, hd_video_url')
  .eq('core_lesson_id', lesson.id)
  .eq('archetype', archetype)
  .order('phase'); // Will need custom sort for Hook→Fact1→...→Wisdom

// 5. Sort atoms in correct phase order
const PHASE_ORDER = { 'Hook': 1, 'Fact1': 2, 'Fact2': 3, 'Fact3': 4, 'Wisdom': 5 };
atoms.sort((a, b) => PHASE_ORDER[a.phase] - PHASE_ORDER[b.phase]);

// 6. Render player with video URLs
atoms.forEach(atom => {
  // atom.hd_video_url contains the Kelly video for this phase
  // atom.content.script contains what Kelly says (for captions)
});
```

### Player HTML Structure

```html
<div class="lesson-player">
  <!-- Header -->
  <header>
    <span>Day {dayNumber} of 365</span>
    <h1>{topic}</h1>
  </header>
  
  <!-- Video Container -->
  <div class="video-container">
    <video id="kellyVideo" autoplay>
      <source src="{hd_video_url}" type="video/mp4">
    </video>
    <div class="captions">{script}</div>
  </div>
  
  <!-- Phase Navigation -->
  <nav class="phases">
    <button data-phase="Hook" class="active">✨ Welcome</button>
    <button data-phase="Fact1">🔍 Fact 1</button>
    <button data-phase="Fact2">💡 Fact 2</button>
    <button data-phase="Fact3">🎯 Fact 3</button>
    <button data-phase="Wisdom">🌟 Wisdom</button>
  </nav>
  
  <!-- Archetype Selector -->
  <div class="archetype-selector">
    <button data-archetype="The Explorer">🧭 Explorer</button>
    <button data-archetype="The Rebel">⚡ Rebel</button>
    <button data-archetype="The Scientist">🔬 Scientist</button>
  </div>
  
  <!-- Progress -->
  <footer>
    <progress value="{currentPhase}" max="5"></progress>
    <button id="nextPhase">Continue →</button>
  </footer>
</div>
```

### Player Video Switching

```javascript
function loadPhase(phaseIndex) {
  const atom = atoms[phaseIndex];
  const video = document.getElementById('kellyVideo');
  
  // Update video source
  video.src = atom.hd_video_url;
  video.load();
  video.play();
  
  // Update captions
  document.querySelector('.captions').textContent = atom.content.script;
  
  // Update phase navigation
  document.querySelectorAll('.phases button').forEach((btn, i) => {
    btn.classList.toggle('active', i === phaseIndex);
  });
}
```

---

## PART 4: THE VERIFICATION CHECKLIST

Before declaring ANY lesson "complete," verify ALL of these:

### Pipeline Verification
- [ ] Audio file plays and sounds like Kelly
- [ ] Audio matches the script from database
- [ ] Image shows Kelly in correct pose (blue sweater, brown hair)
- [ ] Motion video has natural movement (not frozen)
- [ ] Motion video has Kelly looking at camera (not wandering eyes)
- [ ] Lipsync matches audio timing (mouth moves with words)
- [ ] Final video is uploaded to Supabase
- [ ] Database `hd_video_url` field is populated
- [ ] URL is publicly accessible

### Player Verification
- [ ] Player loads without JavaScript errors
- [ ] Player fetches correct lesson from Supabase
- [ ] Video plays when page loads
- [ ] Video shows Kelly (not placeholder/error)
- [ ] Captions match what Kelly is saying
- [ ] Phase navigation works (all 5 phases)
- [ ] Archetype switching loads different videos
- [ ] Mobile responsive
- [ ] Works in Chrome, Safari, Firefox

### Quality Verification
- [ ] Kelly's face is consistent across all 5 phases
- [ ] Kelly's expression matches the phase (excited for Hook, warm for Wisdom)
- [ ] Audio quality is clear (no artifacts)
- [ ] Video quality is HD (1080p minimum)
- [ ] Lip sync is accurate (>90% match)
- [ ] Total lesson flows naturally when played end-to-end

---

## PART 5: THE EXECUTION ORDER

### Phase 1: Verify What Exists (30 min)
1. Query Supabase for Day 1 atoms with `hd_video_url` populated
2. Download and watch each video file
3. Document which videos are good, which need regeneration
4. Check if any lesson player currently uses these videos

### Phase 2: Build Working Player (2 hours)
1. Create `public/learn.html` that fetches from Supabase
2. Implement video playback with phase navigation
3. Test with existing Day 1 videos (if they work)
4. Deploy to curiouskelly.com/learn

### Phase 3: Fix/Regenerate Day 1 Videos (2 hours)
1. Identify which videos need regeneration
2. Run pipeline for missing/broken videos
3. Verify each video meets quality standards
4. Update database with new URLs

### Phase 4: End-to-End Test (1 hour)
1. Open curiouskelly.com/learn?day=1
2. Select each archetype
3. Play through all 5 phases
4. Verify everything works perfectly
5. Test on mobile

### Phase 5: Document and Scale (1 hour)
1. Document exact pipeline settings that produced good videos
2. Create batch script for Days 2-365
3. Estimate total generation time and cost
4. Begin mass generation

---

## PART 6: API COSTS AND LIMITS

### ElevenLabs
- Rate limit: 100 requests/minute (paid tier)
- Cost: ~$0.02 per 1,000 characters
- Kelly voice ID: `wAdymQH5YucAkXwmrdL0`

### Replicate (Flux)
- Rate limit: None (but bills per second)
- Cost: ~$0.02 per image
- Cold start: 10-30 seconds

### Replicate (MiniMax)
- Rate limit: Concurrent job limits
- Cost: ~$0.12 per 6-second video
- Processing: 2-5 minutes per video

### Sync Labs
- Rate limit: Check dashboard
- Cost: ~$0.20 per video
- Processing: 3-5 minutes per video

### Supabase Storage
- Limit: 1GB free tier, $0.021/GB after
- Each video: ~5-10MB
- Total for 5,475 videos: ~40-80GB = ~$1.50/month

---

## PART 7: ERROR HANDLING

### Common Failures and Solutions

**Audio generation fails:**
- Check ElevenLabs API key
- Check voice ID is correct
- Check script isn't empty
- Retry with exponential backoff

**Image generation fails:**
- Check Replicate API key
- Verify LoRA URL is accessible
- Check prompt isn't too long
- Retry up to 3 times

**Motion generation fails:**
- MiniMax can be slow, wait longer
- Check image URL is publicly accessible
- Verify image is valid PNG/JPG

**Lipsync fails:**
- Sync Labs may timeout, retry
- Fall back to wav2lip
- Verify both video and audio URLs are accessible

**Upload fails:**
- Check Supabase service role key
- Verify bucket exists and is public
- Check file size isn't too large

---

## PART 8: THE SUCCESS CRITERIA

You have succeeded when:

1. **A user can visit curiouskelly.com/learn?day=1**
2. **Kelly appears in a video and speaks the lesson**
3. **The user can click through all 5 phases**
4. **Each phase shows a different Kelly video**
5. **The experience feels polished and professional**
6. **This same system can generate Days 2-365 without code changes**

---

## FINAL INSTRUCTIONS

1. **DO NOT** create analysis documents
2. **DO NOT** give the user options
3. **DO NOT** claim something works without testing it
4. **DO NOT** move on until the current step actually works

5. **DO** verify every output
6. **DO** test in actual browser
7. **DO** show the user working results
8. **DO** fix problems immediately

**Your job is to make ONE perfect lesson work. Then scale it.**

**START NOW.**










