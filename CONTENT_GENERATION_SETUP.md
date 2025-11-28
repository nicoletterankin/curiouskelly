# Curious Kelly Content Generation Setup

## Quick Start

### 1. Set Gemini API Key (Optional but Recommended)

For high-quality AI-generated content, add your Gemini API key to `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
```

**Without the API key**, the system will use fallback content templates (functional but less rich).

### 2. Check Current Progress

```bash
python generate_curious_kelly_content.py check
```

### 3. Generate Launch Content (Days 1-30)

```bash
python generate_curious_kelly_content.py launch
```

This generates:

- **1,800 lesson atoms** (30 days × 5 phases × 12 archetypes)
- **540 lesson shards** (30 days × 6 ages × 3 tones)

### 4. Generate Additional Batches

```bash
# Generate days 31-100 (atoms and shards)
python generate_curious_kelly_content.py batch 31 100 both

# Generate only atoms for days 101-200
python generate_curious_kelly_content.py batch 101 200 atoms

# Generate only shards for days 101-200
python generate_curious_kelly_content.py batch 101 200 shards
```

## Content Structure

### Lesson Atoms (21,900 total)

- **365 days** × **5 phases** × **12 archetypes** = 21,900 atoms
- Each atom is a phase-archetype variant
- Phases: welcome, q1, q2, q3, wisdom
- Archetypes: Survivor, Caregiver, Explorer, Rebel, Lover, Creator, Jester, Sage, Magician, Hero, Everyman, Ruler

### Lesson Shards (6,570 total)

- **365 days** × **6 ages** × **3 tones** = 6,570 shards
- Each shard is a complete lesson for specific age/tone
- Ages: 5, 10, 15, 25, 40, 65
- Tones: curious, playful, serious

## Generation Timeline

### Phase 1: Launch (Days 1-30) - PRIORITY

- Complete atoms and shards for soft launch
- **Estimated time**: 2-3 hours with API, 30 minutes with fallback
- **Target**: December 17, 2025

### Phase 2: Early Content (Days 31-100)

- Build content library for first 100 days
- **Estimated time**: 6-8 hours with API, 1.5 hours with fallback

### Phase 3: Full Library (Days 101-365)

- Complete all 365 days
- **Estimated time**: 15-20 hours with API, 4 hours with fallback
- Can be done incrementally post-launch

## Kelly Constitution

All generated content follows these 5 principles:

1. **Graceful Authority** - Confident without arrogance
2. **Radical Curiosity** - Genuinely excited to explore WITH learner
3. **Warm Neutrality** - Multiple perspectives, no agenda
4. **Concise Poetics** - Beautiful language, no waste
5. **"Yes, And..."** - Build on responses, never shut down

## Database Schema

### core_lessons

- 365 rows (already populated)
- Contains: day_number, topic, universal_truth, headlines

### lesson_atoms

- 21,900 rows (to be generated)
- Contains: core_lesson_id, day_number, archetype, phase, content (JSONB)

### lesson_shards

- 6,570 rows (to be generated)
- Contains: core_lesson_id, day_number, age, region, tone, script_content (JSONB)

## Progress Tracking

The system automatically tracks:

- Total atoms/shards generated
- Days completed
- Percentage progress
- Generation timestamps

## Troubleshooting

### "GEMINI_API_KEY not set"

- This is OK! System will use fallback content
- For better quality, add API key to `.env`

### Database connection errors

- Verify PostgreSQL is running
- Check connection string: `postgresql://antigravity:antigravity123@localhost:5432/antigravity_dev`

### Rate limiting

- System includes 0.5s delays between API calls
- Adjust `time.sleep()` in script if needed

## Next Steps

1. ✅ Run `python generate_curious_kelly_content.py check`
2. ✅ Run `python generate_curious_kelly_content.py launch`
3. ✅ Verify content in database
4. ✅ Test lesson playback in app
5. ✅ Generate remaining content incrementally
