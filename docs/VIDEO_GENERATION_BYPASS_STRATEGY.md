# 🎬 VIDEO GENERATION BYPASS STRATEGY

> **LOCKED: December 17, 2025**  
> This document captures the critical HeyGen bypass strategy. DO NOT FORGET.

---

## 🚨 THE PROBLEM

HeyGen's queue is unpredictable. Videos can sit in "processing" or "waiting" for 8+ hours.  
We need 4,380 videos/year (12 archetypes × 365 days).  
Can't depend on a queue we don't control.

---

## 💡 THE KEY INSIGHT

**Sync Labs `lipsync-2` can do VIDEO-TO-VIDEO, not just image-to-video.**

This means:
1. Use a **completed HeyGen video** (with Kling motion baked in) as the base
2. Generate **new audio** with ElevenLabs for any day's script  
3. **Re-dub** the video with Sync Labs → Kelly stays consistent!

---

## 🏗️ THE TWO PIPELINES

### Pipeline 1: Sync Labs Video Re-Dub (PREFERRED)
**Use when:** You have completed HeyGen videos to use as motion reference

```
HeyGen Video (motion base) + ElevenLabs Audio (new script) → Sync Labs → Final Video
```

| Aspect | Value |
|--------|-------|
| Quality | 95% lip-sync |
| Time | ~1-2 min/video |
| Kelly Consistency | ✅ Perfect (uses HeyGen motion) |
| Queue | None |

**Command:**
```powershell
npx tsx scripts/sync-labs-video-redub.ts --day 352 --reference-day 351
npx tsx scripts/sync-labs-video-redub.ts --day 353 --only scientist,explorer
```

### Pipeline 2: Sync Labs Fresh Generation (FALLBACK)
**Use when:** No HeyGen reference video exists for that archetype

```
Replicate LoRA Image + ElevenLabs Audio → Wav2Lip → Sync Labs → Final Video
```

| Aspect | Value |
|--------|-------|
| Quality | 85% lip-sync (less natural motion) |
| Time | ~2 min/video |
| Kelly Consistency | ⚠️ Variable (new image each time) |
| Queue | None |

**Command:**
```powershell
npx tsx scripts/sync-labs-batch-generate.ts --day 351 --only explorer,mystic,provider
```

---

## 📊 WHEN TO USE WHAT

| Scenario | Use This Pipeline |
|----------|-------------------|
| HeyGen completed video exists for archetype | **Video Re-Dub** |
| HeyGen video missing for archetype | **Fresh Generation** (fallback) |
| HeyGen queue is fast (rare) | Let HeyGen run normally |
| Day 352+ after Day 351 is complete | **Video Re-Dub** using Day 351 as reference |

---

## 📁 KEY FILES

| File | Purpose |
|------|---------|
| `scripts/sync-labs-video-redub.ts` | Re-dub HeyGen videos with new audio |
| `scripts/sync-labs-batch-generate.ts` | Fresh generation (fallback) |
| `generated-videos/day-XXX-manifest.json` | HeyGen video manifests with URLs |
| `generated-images/kelly-motion-library.json` | 36 HeyGen talking photo IDs |
| `generated-images/kelly-archetypes-head-only/archetype_head_urls.json` | Kelly head images on Supabase |

---

## 🔑 API KEYS REQUIRED

```env
ELEVENLABS_API_KEY=xxx
SYNC_LABS_API_KEY=xxx
REPLICATE_API_TOKEN=xxx  # Only for fallback pipeline
PUBLIC_SUPABASE_URL=xxx
SUPABASE_SERVICE_ROLE_KEY=xxx
```

---

## 📈 PRODUCTION WORKFLOW

### For Day 352 (after Day 351 is mostly complete):

1. **Check Day 351 manifest** - see which archetypes have completed HeyGen videos:
   ```powershell
   cat generated-videos/day-351-manifest.json | Select-String "completed"
   ```

2. **Re-dub available archetypes** (9 currently available):
   ```powershell
   npx tsx scripts/sync-labs-video-redub.ts --day 352 --reference-day 351
   ```

3. **For missing archetypes** (explorer, mystic, provider) - use fallback:
   ```powershell
   npx tsx scripts/sync-labs-batch-generate.ts --day 352 --only explorer,mystic,provider
   ```

4. **Once HeyGen finishes those 3**, you can re-generate them with proper motion:
   ```powershell
   npx tsx scripts/sync-labs-video-redub.ts --day 352 --only explorer,mystic,provider
   ```

---

## 🎯 LONG-TERM STRATEGY

1. **Build a "motion base library"**: Get all 12 archetypes completed via HeyGen at least once
2. **Use re-dub for all future days**: ~18 min per day (9 archetypes × 2 min)
3. **Never wait for HeyGen queue again**: Only use HeyGen for initial motion bases
4. **Store HeyGen videos permanently**: They're the reusable motion assets

---

## ⚠️ CRITICAL REMINDERS

- **DO NOT** generate fresh LoRA images when HeyGen videos exist → Kelly looks different
- **DO** use completed HeyGen videos as motion reference → Kelly stays consistent  
- **HeyGen URLs expire** → Download/store completed videos in Supabase
- **Sync Labs has no queue** → Videos process in ~1-2 minutes

---

## 🔗 Related Docs

- `docs/DUAL_MODE_KELLY_ARCHITECTURE.md` - 2D/3D mode toggle
- `docs/HEYGEN_MOTION_LIBRARY_GUIDE.md` - Multi-motion scene stitching
- `docs/KELLY_36_MOTION_PROMPTS.md` - Motion prompt templates

---

**REMEMBER:** The goal is to use HeyGen to create high-quality motion bases ONCE per archetype, then re-dub forever with Sync Labs. HeyGen becomes an asset generator, not a production bottleneck.
