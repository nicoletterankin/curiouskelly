# Video Trailer System Roadmap
**Date:** December 23, 2025  
**Goal:** Turn lesson artifacts into video trailers for each day  
**Status:** 🟡 Planning Phase

---

## Vision

Every day dot in the calendar should show a **10-15 second video trailer** that:
- Previews the lesson content
- Shows Kelly teaching
- Highlights key moments
- Encourages clicks

---

## Current State Analysis

### What We Have
- ✅ HD videos for Day 1 (15 main + 27 responses)
- ✅ Video pipeline ready (`hd-golden-lesson-pipeline.ts`)
- ✅ Assets scattered across Supabase, local files, CDN
- ✅ Lesson metadata (topics, phases, archetypes)

### What We're Missing
- ❌ Unified artifact inventory
- ❌ Trailer generation pipeline
- ❌ Trailer storage structure
- ❌ Calendar video preview system
- ❌ Asset organization system

---

## Proposed Solution

### Phase 1: Artifact Organization (Foundation)

**Goal:** Know what assets exist for each day

**Create:**
```typescript
interface DayArtifactInventory {
  day: number;
  tracks: {
    learn: {
      videos: VideoAsset[];
      visuals: VisualAsset[];
      audio: AudioAsset[];
      phases: PhaseAsset[];
    };
    grow: {
      videos: VideoAsset[];
      visuals: VisualAsset[];
      audio: AudioAsset[];
      phases: PhaseAsset[];
    };
  };
  completeness: {
    learn: number; // 0-100
    grow: number;  // 0-100
    overall: number;
  };
  bestTrailerSource: {
    video: string; // Best video for trailer
    thumbnail: string;
    duration: number;
  };
}
```

**Script:** `scripts/inventory-day-artifacts.ts`
- Scans Supabase for all assets
- Checks local files
- Generates inventory JSON
- Stores in `/data/artifact-inventory/day-XXX.json`

---

### Phase 2: Trailer Generation Pipeline

**Goal:** Generate 10-15s preview videos automatically

**Pipeline:**
```
1. Select best video (Hook phase, Explorer archetype, adult age)
2. Extract first 10 seconds
3. Add fade-out at end
4. Overlay day number + topic
5. Export as /trailers/day-XXX-preview.mp4
6. Generate thumbnail: /trailers/day-XXX-thumb.jpg
```

**Script:** `scripts/generate-day-trailers.ts`
- Uses FFmpeg for video processing
- Batch processes all days
- Stores trailers in unified location

**Requirements:**
- FFmpeg installed
- Video assets accessible
- Storage space (~50MB per trailer = 18GB total)

---

### Phase 3: Calendar Integration

**Goal:** Show trailers in calendar

**Implementation:**
```html
<div class="day-dot" data-day="1">
  <video class="day-preview" 
         preload="metadata" 
         muted
         poster="/trailers/day-001-thumb.jpg">
    <source src="/trailers/day-001-preview.mp4" type="video/mp4">
  </video>
  <div class="day-overlay">
    <span class="day-number">1</span>
    <span class="completeness-badge">95%</span>
  </div>
</div>
```

**Behavior:**
- **Hover:** Play trailer (muted, loop)
- **Click:** Open audit panel
- **Double-click:** Show preview popup
- **No video:** Show static thumbnail + completeness badge

---

## Implementation Plan

### Step 1: Artifact Inventory (Week 1)
- [ ] Create inventory script
- [ ] Scan all 365 days
- [ ] Generate inventory JSON files
- [ ] Store in `/data/artifact-inventory/`

### Step 2: Trailer Generation (Week 2)
- [ ] Create trailer generation script
- [ ] Test on Day 1 (already has videos)
- [ ] Generate trailers for Days 1-10 (proof of concept)
- [ ] Optimize pipeline performance

### Step 3: Calendar Integration (Week 3)
- [ ] Add video elements to calendar
- [ ] Implement hover preview
- [ ] Add loading states
- [ ] Test performance (lazy loading)

### Step 4: Full Rollout (Week 4)
- [ ] Generate trailers for all days with videos
- [ ] Deploy to production
- [ ] Monitor performance
- [ ] Collect feedback

---

## Technical Considerations

### Performance
- **Lazy loading:** Only load trailers on hover/scroll
- **Thumbnail first:** Show static image until hover
- **CDN:** Store trailers on CDN for fast delivery
- **Compression:** Optimize video file sizes

### Storage
- **Location:** `/public/trailers/` or Supabase Storage
- **Naming:** `day-XXX-preview.mp4`, `day-XXX-thumb.jpg`
- **Size:** ~50MB per trailer (18GB total for 365)
- **CDN:** Cloudflare or Vercel Blob Storage

### Browser Support
- **Video codecs:** MP4 (H.264) primary, WebM fallback
- **Poster images:** Always show thumbnail
- **No autoplay:** Respect user preferences
- **Mobile:** Optimize for mobile data usage

---

## Success Metrics

1. **Trailer Coverage:** % of days with trailers
2. **Click-through:** % of calendar clicks after seeing trailer
3. **Performance:** Page load time with trailers
4. **User Engagement:** Time spent on calendar

---

## Next Steps

1. **Review this roadmap** - Does this align with vision?
2. **Start with inventory** - Know what we have first
3. **Test trailer generation** - Proof of concept on Day 1
4. **Iterate** - Refine based on results

---

**Priority:** 🟡 MEDIUM (Calendar layout fixed, trailers are enhancement)  
**Complexity:** 🟠 HIGH (Requires video processing, storage, organization)  
**Timeline:** 4 weeks for full implementation

