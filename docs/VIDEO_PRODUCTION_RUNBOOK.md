# Kelly Video Production Runbook

> Operational guide for coordinated video generation between human operators and AI agents.

---

## Current Production Status

```sql
-- Run this to see real-time status
SELECT 
  asset_type,
  COUNT(*) as count,
  COUNT(DISTINCT day_number) as days_covered
FROM kelly_video_assets
GROUP BY asset_type;
```

### Target Production

| Asset Type | Per Day | Total 365 | Status |
|------------|---------|-----------|--------|
| Images | 5 | 1,825 | 🔄 25 done |
| Animations | 5 | 1,825 | 🔄 19 done |
| Audio | ~60 | ~21,900 | 🔄 75 done |
| Videos | ~60 | ~21,900 | 🔄 40 done |

---

## Daily Production Checklist

### Before Starting a Day

```bash
# 1. Check if day already has assets
node scripts/kelly-video-factory/lesson-content-pipeline.cjs --day {DAY} --dry-run

# 2. Verify lesson_atoms content is correct
# (Use the enhanced format with optionIntro, quality markers, etc.)
```

### Production Sequence

```bash
# Run full pipeline (recommended)
node scripts/kelly-video-factory/production-orchestrator.cjs --day {DAY}

# Or run stages manually:
node scripts/kelly-video-factory/batch-image-generator.cjs --days {DAY}
node scripts/kelly-video-factory/batch-animation-generator.cjs --days {DAY}  
node scripts/kelly-video-factory/generate-day-audio.cjs --day {DAY}
node scripts/kelly-video-factory/generate-day-lipsync.cjs --day {DAY}
```

### After Completing a Day

```sql
-- Verify all assets generated
SELECT phase, asset_type, COUNT(*) 
FROM kelly_video_assets 
WHERE day_number = {DAY}
GROUP BY phase, asset_type
ORDER BY phase, asset_type;

-- Should show:
-- hook    | image     | 1
-- hook    | animation | 1
-- hook    | audio     | 12
-- hook    | video     | 12
-- (repeat for q1, q2, q3, wisdom)
```

---

## Parallel Work Coordination

### Claiming Work
Before starting, announce: "CLAIMING: Days X-Y"

### Work Assignment Strategy
```
Agent A: Days 1-50 (January + February start)
Agent B: Days 51-100 (February + March)
Human: Quality review, exceptions, fixes
```

### Status Updates
Post updates in this format:
```
DAY 23 COMPLETE:
- Images: 5/5 ✅
- Animations: 5/5 ✅
- Audio: 60/60 ✅
- Videos: 60/60 ✅
- Quality check: PASSED
```

---

## Content Quality Requirements

### lesson_atoms Content (REQUIRED)

Each atom must have:
```json
{
  "script": "Non-empty teaching content",
  "options": [
    {
      "letter": "A",
      "text": "Option text",
      "quality": "best|good|redirect",
      "response": "Kelly's response"
    }
  ]
}
```

### Content Validation Query
```sql
-- Find atoms with missing or broken content
SELECT cl.day_number, la.phase, la.archetype, la.content->>'script' as script_preview
FROM lesson_atoms la
JOIN core_lessons cl ON la.core_lesson_id = cl.id
WHERE la.content->>'script' IS NULL 
   OR la.content->>'script' LIKE '%Generation Offline%'
   OR la.content->>'script' LIKE '%Error%'
ORDER BY cl.day_number, la.phase;
```

---

## Error Recovery Procedures

### Image Generation Failed
```bash
# Check Replicate logs
# Re-run with specific day:
node batch-image-generator.cjs --days {DAY} --force-regenerate
```

### Animation Failed  
```bash
# Usually means source image is bad
# First regenerate image, then:
node batch-animation-generator.cjs --days {DAY} --force-regenerate
```

### Audio Failed
```bash
# Check ElevenLabs quota
# Re-run specific archetype:
node generate-day-audio.cjs --day {DAY} --archetype "The Explorer"
```

### Lipsync Failed
```bash
# Check animation URL validity
# Re-run:
node generate-day-lipsync.cjs --day {DAY} --force-regenerate
```

### Full Reset for a Day
```sql
-- CAUTION: Deletes all assets for a day
DELETE FROM kelly_video_assets WHERE day_number = {DAY};
```

---

## Cost Tracking

### Per-Day Estimates
```
Images (5):      $0.015
Animations (5):  $0.25
Audio (60):      $0.60
Videos (60):     $1.20
─────────────────────────
Total:           ~$2.07/day
```

### Running Total Query
```sql
SELECT 
  SUM(generation_cost_usd) as total_cost,
  COUNT(*) as total_assets
FROM kelly_video_assets
WHERE generation_cost_usd IS NOT NULL;
```

---

## Quality Assurance

### Automated Checks
- Face audit score > 0.8
- Sweater color = teal (not purple/pink/blue)
- Duration > 5 seconds
- File size > 100KB

### Manual Review (Sample These)
- [ ] Kelly's face consistent
- [ ] Lip sync aligned with audio
- [ ] No visual artifacts
- [ ] Audio clear and audible
- [ ] Content matches lesson topic

### QA Query
```sql
SELECT day_number, phase, asset_type, 
       face_audit_score, sweater_color_check
FROM kelly_video_assets
WHERE asset_type = 'video'
AND (face_audit_passed = false OR sweater_color_check != 'teal')
ORDER BY day_number;
```

---

## Batch Processing Commands

### Small Batch (Testing)
```bash
node production-orchestrator.cjs --days 1-3
```

### Medium Batch (Daily Work)
```bash
node production-orchestrator.cjs --days 1-10
```

### Large Batch (Weekend Run)
```bash
node production-orchestrator.cjs --days 1-50
```

### Full Year (Overnight)
```bash
# NOT RECOMMENDED - use batches instead
node production-orchestrator.cjs --days 1-365
```

---

## Environment Setup

### Required Variables
```bash
# .env file
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforvv.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
REPLICATE_API_TOKEN=r8_...
ELEVENLABS_API_KEY=...
ELEVENLABS_KELLY_VOICE_ID=...
```

### Verify Setup
```bash
# Test Supabase connection
node -e "require('@supabase/supabase-js').createClient(process.env.PUBLIC_SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY).from('core_lessons').select('count').then(r => console.log('✅ Supabase OK', r.count))"

# Test Replicate
curl -s -H "Authorization: Token $REPLICATE_API_TOKEN" https://api.replicate.com/v1/predictions | head
```

---

## File Structure After Production

```
template-forge/lesson-videos/
├── day_001_hook_The_Explorer.mp4
├── day_001_hook_The_Architect.mp4
├── day_001_hook_The_Diplomat.mp4
├── ... (12 archetypes × 5 phases = 60 files per day)
├── day_001_q1_The_Explorer.mp4
├── day_001_q2_The_Explorer.mp4
├── day_001_q3_The_Explorer.mp4
├── day_001_wisdom_The_Explorer.mp4
└── ...
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "No animations found" | Stage 1-2 incomplete | Run batch-image then batch-animation |
| Face audit failed | LoRA not applied correctly | Regenerate with higher lora_scale |
| Wrong sweater color | Prompt issue | Check template prompts in config.cjs |
| Audio too short | Script too short | Check lesson_atoms content |
| Lipsync mismatch | Audio/animation length mismatch | Regenerate animation |
| Rate limit | Too many API calls | Add delays, reduce batch size |

---

## Escalation

If you encounter:
- Repeated face audit failures → Check LoRA training
- Budget exceeded → Pause and notify human
- Content corruption → Run audit scripts, fix atoms first
- API outages → Wait and retry with exponential backoff

---

*Last updated: December 2024*

