# KELLY MASTER VIDEO PLAN: All 394,200 Videos

**Date:** February 3, 2026  
**Status:** PLANNING (awaiting v0 sync)  
**Goal:** Kelly teaches every lesson with perfect lip-sync

---

## THE FULL PICTURE

### Total Videos Needed

| Dimension | Count | Notes |
|-----------|-------|-------|
| Days | 365 | Full year curriculum |
| Phases | 5 | hook, story, wonder, action, wisdom |
| Ages | 3 | Kid, Adult, Elder |
| Languages | 6 | EN, ES, FR, DE, PT, ZH |
| Archetypes | 12 | Matched to lesson tone |
| **TOTAL** | **394,200** | 365 × 5 × 3 × 6 × ~12 |

*Note: Not every lesson needs all 12 archetypes - archetype is matched to topic*

### Simplified MVP Math

If we use 1 archetype per day (matched to topic):
- 365 days × 5 phases × 3 ages × 6 languages = **32,850 videos**

---

## CURRENT ASSET INVENTORY

### What Exists (Confirmed)

| Location | Type | Count | Status |
|----------|------|-------|--------|
| HeyGen Cloud | Final videos | ~100+ visible | Need to download |
| Cloudflare R2 `avatars` | Videos? | 1,180 | Need to verify |
| Local `kelly-pipeline/videos/base/` | Base videos | ~120 | UUID filenames |
| Supabase `kelly_lesson_assets` | Audio | 1,861 | Ready for lip-sync |
| Local JSON mappings | Archetype→UUID | Adult, Elder done | Kid missing |

### What's Missing

| Gap | Impact | Resolution |
|-----|--------|------------|
| Kid talking photo IDs | Can't generate kid videos | Upload to HeyGen |
| UUID→Archetype mapping | Can't use local videos | Build from HeyGen API |
| R2 contents unknown | May have usable assets | List bucket |
| Neon↔Supabase sync | Two sources of truth | Unify schema |

---

## MULTI-FACETED APPROACH

### FACET 1: Asset Recovery (Day 1-2)

**Goal:** Find and catalog everything we already paid for.

**Tasks:**
- [ ] Download all HeyGen videos from cloud
- [ ] List Cloudflare R2 `avatars` bucket contents
- [ ] List Vercel Blob `kelly-base-videos/` contents
- [ ] Map UUIDs to archetypes using HeyGen API
- [ ] Create master asset registry JSON

**Owner:** Cursor + v0 collaboration

### FACET 2: Database Unification (Day 2-3)

**Goal:** Single source of truth for all video assets.

**Tasks:**
- [ ] Get Neon schema from v0
- [ ] Compare with Supabase schema
- [ ] Design unified `kelly_video_assets` table
- [ ] Migrate existing data
- [ ] Update API endpoints

**Owner:** v0 (has Neon access)

### FACET 3: HeyGen Optimization (Day 3-4)

**Goal:** Maximize value from 668 remaining credits.

**Tasks:**
- [ ] Calculate which videos give most ROI
- [ ] Priority: Day 1-7 × all phases × adult × EN = 35 videos
- [ ] Generate Kid and Elder for Day 1 (proof of concept)
- [ ] Download immediately after generation

**Credits Budget:**
- 668 credits ÷ 36 credits/day = ~18 days of content
- Focus on flagship days (1, 7, 30, 100, 365)

### FACET 4: Lip-Sync Pipeline (Day 4-7)

**Goal:** Bulk generation for non-HeyGen content.

**Multiple providers for scale:**

| Provider | Quality | Cost | Speed | Use For |
|----------|---------|------|-------|---------|
| Sync Labs `lipsync-2-pro` | A | $0.20/video | 5 min | Primary |
| MuseTalk (local) | B+ | Free | 2 min | Backup |
| Wav2Lip (Replicate) | B | $0.10/video | 3 min | Fallback |
| fal.ai LatentSync | B | $0.05/video | 1 min | Bulk |

**Pipeline:**
1. Check if HeyGen video exists → use it
2. Else, get base video + audio
3. Lip-sync with Sync Labs
4. Upload to storage
5. Update database

### FACET 5: Quality Assurance (Ongoing)

**Goal:** Every video passes quality gates before serving.

**Gates:**
- [ ] File size > 100KB
- [ ] Duration matches audio ±10%
- [ ] Face detected in every frame
- [ ] Lip-sync score > 0.8 (where measurable)
- [ ] Plays correctly in browser

**Automation:**
- Run QA on upload
- Flag failed videos
- Retry with different provider

### FACET 6: Delivery Infrastructure (Day 5-7)

**Goal:** Fast, reliable video playback worldwide.

**CDN Strategy:**
- Primary: Cloudflare R2 (zero egress)
- Backup: Supabase Storage
- Fallback: Vercel Blob

**API Design:**
```typescript
GET /api/kelly-video
  ?day=34
  &phase=hook
  &age=adult
  &lang=en
  &archetype=scientist  // optional, derived from lesson

Response:
{
  video_url: "https://r2.../day-034-hook-adult-en.mp4",
  audio_url: "https://...",  // for separate playback if needed
  archetype: "scientist",
  quality: "heygen",  // or "sync_labs", "musetalk"
  duration: 45.2
}
```

### FACET 7: Language Expansion (Week 2+)

**Goal:** 6 languages for global reach.

**Priority Order:**
1. EN (English) - Day 1 focus
2. ES (Spanish) - Largest second market
3. PT (Portuguese) - Brazil market
4. FR (French) - Europe + Africa
5. DE (German) - Europe
6. ZH (Chinese) - Largest population

**Approach:**
- Same base video (archetype/age)
- Different audio per language
- Lip-sync creates language-specific video

### FACET 8: Age Variants (Week 2+)

**Goal:** Kelly speaks at the right level for each learner.

**Age Mapping:**
| Age Group | Display Name | Kelly Appearance | Voice Tone |
|-----------|--------------|------------------|------------|
| Kid | Ages 4-12 | Kid Kelly | Playful, simple |
| Adult | Ages 13-64 | Adult Kelly | Conversational |
| Elder | Ages 65+ | Elder Kelly | Warm, patient |

**Current Status:**
- Adult: Talking photo IDs ✅
- Elder: Talking photo IDs ✅
- Kid: NOT UPLOADED - need to create

---

## IMMEDIATE NEXT STEPS

### For Nicolette:

1. **Send the v0 request document** to your v0.app chat
   - File: `docs/V0_REQUEST_MASTER_SYNC.md`
   - Copy-paste the content or share the file

2. **Check HeyGen dashboard:**
   - Can you export a list of all videos?
   - Can you download them in bulk?

3. **Check Cloudflare R2:**
   - What's in the `avatars` bucket?
   - Can you list the file names?

### For v0.app:

1. Respond to the sync request with database schemas
2. Share what's in R2 storage
3. Explain current video playback logic

### For Cursor:

1. Waiting on v0 response before:
   - Building unified database schema
   - Creating download scripts
   - Running lip-sync pipeline

---

## COST ESTIMATES

### Option A: All HeyGen (Highest Quality)
- 32,850 videos × ~1 credit each = 32,850 credits
- Cost: ~$3,285 (at $0.10/credit)
- **Not viable with 668 remaining**

### Option B: HeyGen Flagship + Lip-Sync Bulk
- HeyGen: 600 videos (Days 1-7, all variants) = 600 credits
- Sync Labs: 32,250 videos × $0.20 = $6,450
- **Total: ~$6,500**

### Option C: All Lip-Sync (Lowest Cost)
- Sync Labs: 32,850 videos × $0.20 = $6,570
- Fallback to cheaper providers where needed
- **Total: $5,000 - $7,000**

### Option D: Local MuseTalk (Free but Slower)
- GPU compute time only
- ~2 min/video × 32,850 = 1,095 hours = 45 days continuous
- **Cost: $0 + electricity**

---

## TIMELINE

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Asset Recovery | 2 days | Complete inventory |
| DB Unification | 2 days | Single source of truth |
| Day 1 Complete | 1 day | 90 videos (5 phases × 3 ages × 6 langs) |
| Days 2-7 | 3 days | 630 videos |
| Days 8-30 | 1 week | 2,070 videos |
| Days 31-365 | 3 weeks | 30,150 videos |
| **Total** | ~5 weeks | 32,850 videos |

*With parallelization and multiple providers, could be faster*

---

## SUCCESS METRICS

1. **Day 1:** Kelly teaches Day 1 with all variants
2. **Week 1:** Days 1-7 complete
3. **Month 1:** Days 1-30 complete
4. **Month 2:** All 365 days complete

---

## RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| HeyGen credits run out | Can't generate high-quality | Use lip-sync for bulk |
| Sync Labs rate limited | Slow generation | Multiple providers |
| Storage costs spike | Budget overrun | Use R2 (free egress) |
| Quality inconsistency | User experience | QA gates + retries |
| DB sync issues | Wrong videos play | Single source of truth |

---

**This is the plan. Once v0 responds, we execute.**
