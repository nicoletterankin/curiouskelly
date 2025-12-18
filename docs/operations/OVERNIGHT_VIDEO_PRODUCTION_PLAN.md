# 🌙 OVERNIGHT VIDEO PRODUCTION PLAN
> **Created:** December 17, 2025  
> **Target:** Generate videos for Days 352-365 using Sync Labs re-dub pipeline

---

## 📊 CURRENT STATE AUDIT

### Lesson Content Available
| Day Range | Count | Status |
|-----------|-------|--------|
| 1-351 | 351 | ✅ Full content |
| 352-365 | 14 | ✅ Lessons ready |
| **TOTAL** | **365** | Full year covered |

### Day 351 HeyGen Videos (Motion Reference Base)
| Archetype | HeyGen Status | Available as Re-dub Base |
|-----------|---------------|--------------------------|
| scientist | ✅ Complete | ✅ Yes |
| rebel | ✅ Complete | ✅ Yes |
| architect | ✅ Complete | ✅ Yes |
| diplomat | ✅ Complete | ✅ Yes |
| empath | ✅ Complete | ✅ Yes |
| macgyver | ✅ Complete | ✅ Yes |
| storyteller | ✅ Complete | ✅ Yes |
| strategist | ✅ Complete | ✅ Yes |
| survivor | ✅ Complete | ✅ Yes |
| explorer | ⏳ HeyGen queue | ❌ Fresh gen only |
| mystic | ⏳ HeyGen queue | ❌ Fresh gen only |
| provider | ⏳ HeyGen queue | ❌ Fresh gen only |

**Summary:** 9/12 archetypes can use HeyGen-quality re-dub. 3/12 require fresh generation.

---

## 🎯 PRODUCTION TARGETS

### Phase 1: Priority Days (Re-dub Pipeline)
Generate 9 archetypes per day using HeyGen motion base

| Day | Topic | Archetypes | Est. Time |
|-----|-------|------------|-----------|
| 352 | Quieting Your Thoughts | 9 | ~18 min |
| 353 | Being Where You Are | 9 | ~18 min |
| 354 | [TBD] | 9 | ~18 min |
| 355 | [TBD] | 9 | ~18 min |
| 356 | [TBD] | 9 | ~18 min |
| 357 | [TBD] | 9 | ~18 min |
| 358 | [TBD] | 9 | ~18 min |
| 359 | [TBD] | 9 | ~18 min |
| 360 | [TBD] | 9 | ~18 min |
| 361 | [TBD] | 9 | ~18 min |
| 362 | [TBD] | 9 | ~18 min |
| 363 | [TBD] | 9 | ~18 min |
| 364 | [TBD] | 9 | ~18 min |
| 365 | [TBD] | 9 | ~18 min |

**Phase 1 Total:** 14 days × 9 archetypes = **126 videos** (~4.2 hours)

### Phase 2: Missing Archetypes (Fresh Generation)
Generate explorer, mystic, provider for each day

| Day | Archetypes | Est. Time |
|-----|------------|-----------|
| 352-365 | 3 each × 14 days | ~42 videos |

**Phase 2 Total:** 14 days × 3 archetypes = **42 videos** (~1.4 hours)

---

## 💰 COST ESTIMATES

### API Usage Estimates

| Service | Usage | Est. Cost |
|---------|-------|-----------|
| **ElevenLabs** | ~168 audio files (~90 sec each) | ~42 min of audio |
| **Sync Labs** | 168 video generations | Based on plan |
| **Replicate** | 42 image + 42 base video (fresh only) | ~$2-4 |
| **Supabase** | Storage for audio/manifests | Negligible |

### Time Estimates
| Phase | Videos | Time per Video | Total Time |
|-------|--------|----------------|------------|
| Phase 1 (Re-dub) | 126 | ~2 min | ~4.2 hours |
| Phase 2 (Fresh) | 42 | ~3 min | ~2.1 hours |
| **TOTAL** | **168** | - | **~6.3 hours** |

---

## 🔧 EXECUTION COMMANDS

### Phase 1: Re-dub Pipeline (Days 352-365)

```powershell
# Run sequentially (recommended for overnight)
$days = 352..365
foreach ($day in $days) {
    Write-Host "Starting Day $day..." -ForegroundColor Cyan
    npx tsx scripts/sync-labs-video-redub.ts --day $day --reference-day 351
    Write-Host "Completed Day $day" -ForegroundColor Green
}
```

### Phase 2: Fresh Generation (Missing Archetypes)

```powershell
# Run after Phase 1
$days = 352..365
foreach ($day in $days) {
    Write-Host "Fresh gen for Day $day (explorer, mystic, provider)..." -ForegroundColor Yellow
    npx tsx scripts/sync-labs-batch-generate.ts --day $day --only explorer,mystic,provider
}
```

### All-in-One Overnight Script

```powershell
# Save as: scripts/overnight-production.ps1

Write-Host "🌙 OVERNIGHT VIDEO PRODUCTION STARTING" -ForegroundColor Magenta
Write-Host "Target: Days 352-365 (168 videos)" -ForegroundColor Cyan
$startTime = Get-Date

# Phase 1: Re-dub (9 archetypes per day)
Write-Host "`n📹 PHASE 1: Re-dub Pipeline" -ForegroundColor Yellow
352..365 | ForEach-Object {
    Write-Host "  Day $_..." -NoNewline
    npx tsx scripts/sync-labs-video-redub.ts --day $_ --reference-day 351 2>&1 | Out-Null
    Write-Host " ✅" -ForegroundColor Green
}

# Phase 2: Fresh generation (3 archetypes per day)
Write-Host "`n🎨 PHASE 2: Fresh Generation" -ForegroundColor Yellow
352..365 | ForEach-Object {
    Write-Host "  Day $_ (explorer, mystic, provider)..." -NoNewline
    npx tsx scripts/sync-labs-batch-generate.ts --day $_ --only explorer,mystic,provider 2>&1 | Out-Null
    Write-Host " ✅" -ForegroundColor Green
}

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "`n🎉 PRODUCTION COMPLETE" -ForegroundColor Green
Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m" -ForegroundColor Cyan
```

---

## ✅ PRE-FLIGHT CHECKLIST

Before running overnight:

- [ ] **API Keys Verified**
  - [ ] ELEVENLABS_API_KEY set
  - [ ] SYNC_LABS_API_KEY set
  - [ ] REPLICATE_API_TOKEN set
  - [ ] SUPABASE credentials set

- [ ] **Credit Balances Checked**
  - [ ] ElevenLabs: Sufficient characters remaining
  - [ ] Sync Labs: Sufficient credits
  - [ ] Replicate: Billing active

- [ ] **Dry Run Completed**
  ```powershell
  npx tsx scripts/sync-labs-video-redub.ts --day 352 --dry-run
  ```

- [ ] **Single Day Test**
  ```powershell
  npx tsx scripts/sync-labs-video-redub.ts --day 352 --only scientist
  ```

- [ ] **Disk Space Available**
  - Audio files: ~25 KB each × 168 = ~4.2 MB
  - Manifest files: ~5 KB each × 14 = ~70 KB

- [ ] **Network Stable**
  - Wired connection preferred
  - No scheduled updates/restarts

---

## 📁 OUTPUT LOCATIONS

| Output Type | Location |
|-------------|----------|
| Re-dub manifests | `generated-videos/sync-labs-redub/day-XXX-redub-manifest.json` |
| Fresh manifests | `generated-videos/sync-labs-production/day-XXX-sync-labs-manifest.json` |
| Audio files | Supabase: `kelly-templates/sync-labs-redub/` |

---

## 🚨 ERROR RECOVERY

### If Script Fails Mid-Run

1. Check which day failed in console output
2. Resume from that day:
   ```powershell
   # Example: Resume from Day 358
   358..365 | ForEach-Object { npx tsx scripts/sync-labs-video-redub.ts --day $_ --reference-day 351 }
   ```

### If API Rate Limited

1. Add delay between videos in script
2. Or run in smaller batches:
   ```powershell
   352..357 | ForEach-Object { npx tsx scripts/sync-labs-video-redub.ts --day $_ --reference-day 351 }
   Start-Sleep -Seconds 300  # 5 min break
   358..365 | ForEach-Object { npx tsx scripts/sync-labs-video-redub.ts --day $_ --reference-day 351 }
   ```

---

## 📊 SUCCESS METRICS

After overnight run, verify:

1. **Manifest files exist** for each day
2. **Video URLs are valid** (spot check 3-5 random videos)
3. **Kelly is consistent** across archetypes
4. **Audio matches lesson content**

---

## 🎬 POST-PRODUCTION

After videos are generated:

1. **Archive HeyGen videos** to permanent Supabase storage (URLs expire)
2. **Update lesson player** to use new video URLs
3. **QA spot-check** random videos from each day
4. **Generate Day 351 missing archetypes** when HeyGen completes

---

**Ready to run?** Start with the dry-run, then the single-day test, then let it run overnight! 🌙
