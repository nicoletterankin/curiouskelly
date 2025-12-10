# 🎤 Kelly Voice Check - Quick Reference

## What It Does

The voice check tool validates Kelly's voice quality before bulk audio generation by:
- Testing ElevenLabs API connectivity
- Verifying Kelly's voice ID
- Testing all expression types (excited, curious, explaining, etc.)
- Testing archetype variations (Explorer, Rebel, Scientist)
- Analyzing audio quality and consistency
- Generating sample audio files for manual review

## When to Use

**Always run before:**
- Bulk generating audio for multiple days
- Starting a new batch of video generation
- After changing voice settings or API keys
- When you notice voice quality issues

**Run periodically:**
- Weekly during active development
- Before major content releases
- After ElevenLabs API updates

## Usage

### Quick Test (1 sample)
```bash
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick
```
**Time:** ~5 seconds  
**Use when:** Quick validation before small batches

### Full Test (13 samples)
```bash
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts
```
**Time:** ~30 seconds  
**Use when:** Comprehensive validation before bulk generation

### Test Specific Expression
```bash
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --expression excited
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --expression wisdom
```
**Time:** ~5 seconds  
**Use when:** Debugging specific expression issues

## Output

### Test Audio Files
Location: `test-output/voice-check/*.mp3`

Listen to these files to verify:
- Voice sounds like Kelly
- Emotional tone matches expression
- Audio quality is clear
- No artifacts or glitches

### Reports
- `voice-check-report.json` - Detailed technical data
- `VOICE_CHECK_SUMMARY.md` - Human-readable summary

## Health Status

### 🟢 EXCELLENT (100% success)
✅ **Ready for bulk generation**
- All tests passed
- Consistent quality
- Fast API responses

### 🟡 GOOD (90-99% success)
⚠️ **Review failed tests first**
- Minor issues detected
- Most tests passed
- Safe to proceed with caution

### 🟠 WARNING (70-89% success)
⚠️ **Fix issues before bulk generation**
- Multiple failures
- Inconsistent quality
- Risk of wasted API calls

### 🔴 CRITICAL (<70% success)
🛑 **DO NOT proceed**
- Major configuration issues
- API problems
- Must resolve before continuing

## What It Checks

### API Configuration
- ✅ ELEVENLABS_API_KEY is set and valid
- ✅ ELEVENLABS_KELLY_VOICE_ID is correct
- ✅ API is accessible and responding

### Voice Quality
- ✅ File sizes are reasonable (5-500 KB)
- ✅ Audio duration matches script length
- ✅ Consistent quality across expressions
- ✅ Fast generation times (<5s per sample)

### Expression Settings
- ✅ Excited: stability=0.45, style=0.35
- ✅ Curious: stability=0.48, style=0.30
- ✅ Explaining: stability=0.55, style=0.15
- ✅ Wisdom: stability=0.60, style=0.20
- ✅ Calm: stability=0.65, style=0.10

## Troubleshooting

### "ELEVENLABS_API_KEY not set"
**Fix:** Add to `.env` file:
```
ELEVENLABS_API_KEY=sk_your_key_here
```

### "Voice ID not found"
**Fix:** Verify voice ID in `.env`:
```
ELEVENLABS_KELLY_VOICE_ID=wAdymQH5YucAkXwmrdL0
```

### "API error 401"
**Fix:** Check API key is valid at https://elevenlabs.io/

### "Slow API response times"
**Possible causes:**
- Network connectivity issues
- ElevenLabs service slowdown
- Rate limiting

**Fix:** Wait and retry, or check ElevenLabs status

### "High variance in file sizes"
**Possible causes:**
- Inconsistent voice settings
- API issues

**Fix:** Review voice settings in `kelly-blendshape-config.ts`

## Integration with Pipeline

### Before Bulk Generation
```bash
# 1. Run voice check
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts

# 2. Listen to test files in test-output/voice-check/

# 3. If EXCELLENT or GOOD, proceed with generation
npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day 1
```

### Daily Workflow
```bash
# Morning: Quick check
npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick

# If good, start daily generation
npx tsx scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts --day 2
```

## Available Expressions

Test any of these with `--expression <name>`:
- `excited` - Energetic, enthusiastic
- `curious` - Inquisitive, engaged
- `explaining` - Clear, instructive
- `thoughtful` - Patient, attentive
- `wisdom` - Wise, reflective
- `calm` - Steady, measured
- `welcoming` - Warm, inviting
- `contemplative` - Reflective, deep
- `sincere` - Empathetic, heartfelt
- `celebrating` - Proud, joyful

## Cost Considerations

### API Costs
- Quick test: ~$0.01 (1 sample)
- Full test: ~$0.13 (13 samples)
- Recommended frequency: Before each bulk batch

### Time Investment
- Quick test: 5 seconds
- Full test: 30 seconds
- Manual review: 2-3 minutes (listen to samples)

**ROI:** Prevents wasted API calls on bulk generation with bad configuration (~$50-100 saved per caught issue)

## Best Practices

1. **Always run before bulk generation** - Catches issues early
2. **Listen to test files** - Automated checks can't verify "sounds like Kelly"
3. **Save reports** - Track voice quality over time
4. **Run after API changes** - Verify new keys or settings
5. **Test specific expressions** - When debugging issues

## Files Created

```
test-output/voice-check/
├── excited.mp3                    # Expression tests
├── curious.mp3
├── explaining.mp3
├── thoughtful.mp3
├── wisdom.mp3
├── calm.mp3
├── welcoming.mp3
├── contemplative.mp3
├── sincere.mp3
├── celebrating.mp3
├── excited_The_Explorer.mp3       # Archetype tests
├── curious_The_Rebel.mp3
├── explaining_The_Scientist.mp3
├── voice-check-report.json        # Technical data
└── VOICE_CHECK_SUMMARY.md         # Human summary
```

## Related Tools

- `test-elevenlabs.js` - Basic connectivity test
- `hd-golden-lesson-pipeline.ts` - Main video generation
- `kelly-blendshape-config.ts` - Voice settings configuration

---

**Tool Location:** `scripts/kelly-video-factory/kelly-voice-check.ts`  
**Last Updated:** December 9, 2025







