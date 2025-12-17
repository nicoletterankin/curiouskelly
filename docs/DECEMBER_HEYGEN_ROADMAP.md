# December HeyGen Production Roadmap

**Date:** December 17, 2025  
**Goal:** Full Kelly videos for Days 351-365 (Dec 17-31)

---

## 📊 THE HARD NUMBERS

### What We Need to Produce

| Metric | Per Day | Total (15 Days) |
|--------|---------|-----------------|
| Days | 1 | 15 (Days 351-365) |
| Archetypes per day | 12 | 180 videos |
| Scenes per archetype | 14 | 2,520 scenes |
| Video duration | ~90s | ~27 hours of video |
| **Estimated credits** | **~18** | **~270 credits** |

### Current HeyGen Processing Time (Observed)

| Phase | Time |
|-------|------|
| Avatar upscaling (Kling) | ~10 minutes |
| Video generation (14 scenes) | ~20-25 minutes |
| Queue wait (if busy) | Variable |

**Throughput:** ~3 videos/hour in parallel queue

---

## 🚨 CRITICAL BLOCKERS

### 1. Lesson Content (SEVERE)

**Status:** Only **1 of 15** lesson JSONs exist

| Day | Date | Lesson JSON | Status |
|-----|------|-------------|--------|
| 351 | Dec 17 | ✅ day-351.json | Ready |
| 352 | Dec 18 | ❌ Missing | BLOCKED |
| 353 | Dec 19 | ❌ Missing | BLOCKED |
| ... | ... | ... | ... |
| 365 | Dec 31 | ❌ Missing | BLOCKED |

**Action Required:** Generate 14 lesson JSON files with scripts

### 2. Motion Library IDs (MODERATE)

**Status:** 6 of 36 avatar IDs are broken/still upscaling

| Archetype | Broken Motion | Status |
|-----------|---------------|--------|
| architect | C | ⏳ Upscaling |
| diplomat | A | ⏳ Upscaling |
| macgyver | A | ⏳ Upscaling |
| strategist | A | ⏳ Upscaling |
| survivor | B, C | ⏳ Upscaling |

**Action Required:** Wait for HeyGen upscaling to complete (~10 min each), then update IDs

### 3. HeyGen Credits (UNKNOWN)

**Status:** Need to verify account balance

**Required:** ~270 credits for full December content

---

## 📅 REALISTIC TIMELINE

### Best Case (Everything Works)

| Date | Task | Videos | Credits |
|------|------|--------|---------|
| Dec 17 (Today) | Day 351 (7/12 done, 5 blocked) | 12 | 18 |
| Dec 17 (Tonight) | Fix IDs, finish Day 351 | +5 | +7 |
| Dec 18 | Generate lessons 352-356, start videos | 12 | 18 |
| Dec 19-23 | Generate 5 days content + videos | 60 | 90 |
| Dec 24-31 | Generate remaining 8 days | 96 | 144 |
| **Total** | | **180** | **~270** |

### Daily Production Capacity

With HeyGen parallel processing:
- **Submit:** 12 videos in ~10 minutes (with rate limiting)
- **Render:** ~4 hours total (parallelized to ~1 hour wall time)
- **Realistic daily output:** 12-24 videos/day

---

## 🎯 PRIORITY ACTION PLAN

### RIGHT NOW (Next 30 min)
1. ⏳ Wait for 6 avatar upscales to complete
2. 📊 Check HeyGen credit balance
3. 🎬 Finish Day 351 once IDs are ready

### TODAY (Dec 17)
4. ✅ Complete all 12 Day 351 archetypes
5. 📝 Generate lesson JSONs for Days 352-354

### THIS WEEK (Dec 18-23)
6. 🏭 Establish daily production rhythm:
   - Morning: Generate that day's lesson JSON
   - Midday: Submit 12 archetype videos
   - Evening: Verify completion, download
7. 🎯 Target: 12 videos/day minimum

### NEXT WEEK (Dec 24-31)
8. 🚀 Continue production
9. 📦 Build delivery/hosting pipeline
10. ✅ Complete all 180 videos by Dec 31

---

## 💰 CREDIT BUDGET

| Plan | Monthly Credits | Cost |
|------|-----------------|------|
| Creator | 60 credits | $48/mo |
| Team | 200 credits | $120/mo |
| Business | 500 credits | $300/mo |

**We need:** ~270 credits  
**Recommendation:** Business plan ($300) or Team plan + overage

---

## ⚠️ RISK FACTORS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Avatar IDs fail permanently | 5 archetypes blocked | Re-upload with new motion prompts |
| Lesson content not ready | No videos to generate | Prioritize lesson JSON creation |
| Credit shortage | Production halts | Upgrade plan or reduce archetypes |
| HeyGen API issues | Delays | Retry logic built into scripts |
| Rate limiting | Slow submission | Already have 5s delays between videos |

---

## 🔄 PRODUCTION WORKFLOW

```
Daily Cycle:
┌─────────────────────────────────────────────────────────┐
│  1. CONTENT: Ensure day-{N}.json exists                 │
│  2. SUBMIT:  npx tsx scripts/heygen-batch-generate.ts   │
│  3. MONITOR: npx tsx scripts/heygen-check-status.ts     │
│  4. VERIFY:  Watch completed videos for quality         │
│  5. STORE:   Download/archive completed videos          │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 DECISION POINT

**Question:** Do we need ALL 12 archetypes per day?

| Option | Videos | Credits | Feasibility |
|--------|--------|---------|-------------|
| All 12 archetypes | 180 | 270 | Ambitious but doable |
| Top 6 archetypes | 90 | 135 | Very achievable |
| Top 3 archetypes | 45 | 68 | Conservative/safe |
| 1 archetype (proof) | 15 | 23 | Minimum viable |

**Recommendation:** Start with 6-12 archetypes, scale based on capacity

---

## 📝 IMMEDIATE NEXT STEPS

1. **Check HeyGen Dashboard** - Are the 6 avatars done upscaling?
2. **Verify credit balance** - Do we have 270+ credits?
3. **Generate lesson content** - Days 352-365 need JSON files
4. **Finish Day 351** - Get all 12 archetypes complete today

---

*Last updated: Dec 17, 2025 17:30*

