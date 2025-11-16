# Audio Generation Plan for Top 10 Lessons
**Date**: November 16, 2025
**Status**: Ready for Execution

---

## 🎯 Summary

**Content Status**: ✅ 100% Complete
- 10 lessons selected and validated
- 180 language variants complete (10 lessons × 6 ages × 3 languages)
- All EN/ES/FR translations ready

**Audio Status**: 🟡 Partial
- Water-cycle: ✅ 72 audio files generated
- Remaining 9 lessons: ❌ 0 audio files (648 files needed)

---

## 📊 Audio File Structure

### Current Structure (water-cycle example)
```
curious-kellly/backend/config/audio/water-cycle/
├── 2-5-welcome-en.mp3
├── 2-5-welcome-es.mp3
├── 2-5-welcome-fr.mp3
├── 2-5-mainContent-en.mp3
├── 2-5-mainContent-es.mp3
├── 2-5-mainContent-fr.mp3
├── 2-5-wisdomMoment-en.mp3
├── 2-5-wisdomMoment-es.mp3
├── 2-5-wisdomMoment-fr.mp3
... (repeat for all 6 age buckets)
= 72 files total
```

### Content Sections per Age Variant
Each age variant has these sections in each language:
1. **welcome** - Greeting and lesson intro
2. **mainContent** - Core teaching content
3. **wisdomMoment** - Key takeaway/reflection
4. **cta** (call-to-action) - What to do next
5. **summary** - Lesson wrap-up

**Minimum**: 3 sections × 3 languages = 9 files per age bucket
**Full**: 5+ sections × 3 languages = 15 files per age bucket

Water-cycle uses: **12 files per age bucket** (72 total)

---

## 📋 Audio Generation Requirements

### For 9 Remaining Lessons

**Minimum approach** (3 sections):
- 9 lessons × 6 ages × 3 languages × 3 sections = **486 audio files**

**Full approach** (match water-cycle):
- 9 lessons × 72 files each = **648 audio files**

### Recommended: Start with Minimum
Generate **welcome, mainContent, wisdomMoment** first (486 files), then add additional sections if needed.

---

## 🛠️ Generation Tools

### Existing Script
`/home/user/curiouskelly/generate_lesson_audio_for_iclone.py`
- ⚠️ Designed for old lesson structure (single "script" field)
- ⚠️ Hardcoded API key (security issue)
- ⚠️ Only generates EN, no multilingual support
- ⚠️ Outputs to different directory structure

### Required Updates
1. Load from new lesson structure (`ageVariants.{age}.language.{lang}.{section}`)
2. Use `ELEVENLABS_API_KEY` from environment variables
3. Support all 3 languages (en, es, fr)
4. Output to correct directory: `curious-kellly/backend/config/audio/{lesson-id}/`
5. Generate all content sections (welcome, mainContent, wisdomMoment, etc.)
6. Add progress tracking and resume capability
7. Implement rate limiting and error handling

---

## 💰 Cost Estimate

### ElevenLabs Pricing (as of 2025)
- **Free tier**: 10,000 characters/month
- **Starter**: ~$5/month for 30,000 characters
- **Creator**: ~$22/month for 100,000 characters
- **Pro**: ~$99/month for 500,000 characters

### Estimated Character Count
- Average section: ~200 words = ~1,000 characters
- Per file: ~1,000 characters
- **486 files × 1,000 chars = 486,000 characters**
- **648 files × 1,000 chars = 648,000 characters**

**Recommendation**: Pro tier (~$99) for one month to generate all audio, then cancel/downgrade

### API Rate Limits
- ElevenLabs: ~20 requests/minute (free), ~120 requests/minute (pro)
- Generation time: ~5-10 seconds per file
- **Total time estimate**: 3-8 hours for 486 files (with rate limiting)

---

## 🚀 Execution Plan

### Phase 1: Preparation (15 minutes)
1. Get ElevenLabs API key (user must provide)
2. Add `ELEVENLABS_API_KEY` to `.env`
3. Create updated generation script
4. Test with 1 lesson (the-sun, 6 files as proof)

### Phase 2: Batch Generation (3-8 hours)
1. Generate audio for all 9 lessons
2. Monitor progress and handle errors
3. Verify file sizes and quality
4. Store metadata (duration, size, timestamp)

### Phase 3: Validation (30 minutes)
1. Verify all expected files generated
2. Spot-check audio quality across ages/languages
3. Update lesson JSON with audio file references
4. Test in lesson player

---

## 📝 Lessons Queue (Priority Order)

1. **the-sun** (Day 1) - Foundational, engaging
2. **puppies** (Day 2) - High appeal, emotional connection
3. **the-ocean** (Day 3) - Rich content, conservation
4. **the-moon** (Day 4) - Celestial complement
5. **water-cycle** (Day 5) - ✅ ALREADY DONE
6. **molecular-biology** (Day 6) - Complexity, depth
7. **creative-writing** (Day 7) - Creative expression
8. **poetry** (Day 8) - Language arts
9. **dance-expression** (Day 9) - Embodied learning
10. **negotiation-skills** (Day 10) - Practical skills

---

## ⚠️ Important Notes

### API Key Security
- **NEVER** commit API keys to git
- Store in `.env` file (already gitignored)
- Use environment variables in all scripts

### Voice Profile
All lessons use Kelly voice:
- **Voice ID**: `wAdymQH5YucAkXwmrdL0`
- **Model**: `eleven_multilingual_v2`
- **Settings**: stability=0.6, similarity_boost=0.8

### File Naming Convention
```
{age-bucket}-{section}-{language}.mp3

Examples:
- 2-5-welcome-en.mp3
- 6-12-mainContent-es.mp3
- 61-102-wisdomMoment-fr.mp3
```

### Storage
- Directory: `curious-kellly/backend/config/audio/{lesson-id}/`
- Format: MP3 (128 kbps, 44.1 kHz recommended)
- Estimated total size: 300-500 MB for 486 files

---

## ✅ Current Progress

- [x] Select top 10 lessons
- [x] Verify multilingual content (100% complete)
- [x] Analyze audio file structure
- [x] Create generation plan
- [ ] Update generation script
- [ ] Generate audio for 9 lessons (486-648 files)
- [ ] Validate all audio files
- [ ] Test in lesson player

**Next Action**: User must provide `ELEVENLABS_API_KEY`, then run generation script

---

## 🎓 Success Criteria

Audio generation is complete when:
- [ ] All 9 lessons have audio subdirectories
- [ ] Each lesson has 72 audio files (or minimum 54 for 3 sections)
- [ ] Spot-check confirms quality across ages/languages
- [ ] Lesson player successfully loads and plays all audio
- [ ] No generation errors or missing files

---

**Status**: ✅ Plan Complete, Ready for User API Key
**Blocker**: Requires `ELEVENLABS_API_KEY` from user
**Timeline**: 3-8 hours for full generation once key provided
