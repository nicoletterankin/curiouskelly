# Kelly Segment Inventory & Quality Analysis

**Date:** December 2025  
**Status:** Comprehensive audit of video assets, quality pipeline, and segment architecture

---

## Executive Summary

### Current State
- **Videos Generated:** ~122 HD videos (2.2% of target 5,475)
- **Videos Uploaded:** 0 (blocked by missing storage bucket)
- **Videos in Database:** 10-99 (varies by table)
- **Quality Pipeline:** Automated checker exists but not integrated
- **Segment Architecture:** 5-phase structure with ~8-second natural break points

### Critical Findings
1. **Storage Blocker:** Videos generated but not uploaded due to missing Supabase bucket
2. **Quality Gap:** No automated quality gates in production pipeline
3. **Segment Opportunity:** Current phases can be split into 10-second reusable segments
4. **Inventory Tracking:** Multiple tables track videos but not synchronized

---

## 1. Asset Inventory

### 1.1 Video Storage Locations

#### Supabase Storage
- **Bucket:** `kelly-videos` (production)
- **Bucket:** `kelly-templates` (temporary audio)
- **Path Format:** `videos/day-{dayNumber}/{archetype}/{phase}.mp4`
- **Example:** `videos/day-001/explorer/hook_main.mp4`

#### Database Fields
- **`lesson_atoms.hd_video_url`** - Primary video URL per phase
- **`lesson_atoms.content.script_video_url`** - Main script video
- **`lesson_atoms.content.options[].response_video_url`** - Response videos (A/B/C)
- **`kelly_video_assets.video_public_url`** - Comprehensive asset tracking
- **`lesson_video_generation_status.video_url`** - Generation status tracking

### 1.2 Video Tracking Tables

#### Table: `kelly_video_assets`
**Purpose:** Comprehensive video asset registry with quality metadata

```sql
CREATE TABLE kelly_video_assets (
    id UUID PRIMARY KEY,
    lesson_day INTEGER NOT NULL,
    phase TEXT NOT NULL,  -- 'welcome', 'q1', 'q2', 'q3', 'wisdom'
    age_bucket TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    archetype TEXT,
    
    -- Storage
    video_storage_path TEXT,
    video_public_url TEXT,
    video_duration_ms INTEGER,
    video_file_size_bytes BIGINT,
    video_resolution TEXT,  -- e.g., '1080x1920'
    
    -- Generation metadata
    elevenlabs_generation_id TEXT,
    model_used TEXT DEFAULT 'omnihuman-1.5',
    generation_credits_used INTEGER,
    generation_started_at TIMESTAMPTZ,
    generation_completed_at TIMESTAMPTZ,
    
    -- Quality metadata
    lip_sync_quality_score DECIMAL(4,3),  -- 0.000 to 1.000
    video_quality_score DECIMAL(4,3),
    is_approved BOOLEAN DEFAULT false,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    
    -- Status
    status TEXT DEFAULT 'pending',  -- 'pending', 'generating', 'completed', 'failed'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Analytics
    view_count INTEGER DEFAULT 0,
    last_viewed_at TIMESTAMPTZ
);
```

**Current Count:** ~99 rows (Day 1 only)

#### Table: `lesson_video_generation_status`
**Purpose:** Track generation progress for 5-phase journey

```sql
CREATE TABLE lesson_video_generation_status (
    id UUID PRIMARY KEY,
    core_lesson_id UUID NOT NULL,
    archetype TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom')),
    video_type TEXT NOT NULL CHECK (video_type IN ('main', 'response_A', 'response_B', 'response_C')),
    
    status TEXT NOT NULL DEFAULT 'pending',
    video_url TEXT,
    error_message TEXT,
    
    duration_seconds NUMERIC(6,2),
    file_size_bytes BIGINT,
    resolution TEXT,
    
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);
```

**Current Count:** Unknown (needs query)

#### Table: `lesson_atoms`
**Purpose:** Primary content storage with video URLs

```typescript
interface LessonAtom {
  id: string;
  core_lesson_id: string;
  archetype: string;  // 'The Explorer', 'The Rebel', 'The Scientist'
  phase: string;       // 'Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'
  content: {
    script: string;
    script_video_url?: string;  // Main script video
    options?: Array<{
      letter: string;
      text: string;
      quality: string;  // 'good', 'best', 'redirect'
      response: string;
      response_video_url?: string;  // Response video
    }>;
  };
  hd_video_url?: string;  // Primary video URL
}
```

**Current Count:** 20,341 atoms (all 365 days × 12 archetypes × 5 phases)

### 1.3 Current Inventory Status

#### Generated Videos (Local/Unuploaded)
- **Count:** ~122 videos
- **Days Covered:** Days 1-5 (partial)
- **Location:** Local filesystem (`generated-videos/golden-lesson-hd/`)
- **Status:** ❌ Not uploaded (blocked by missing bucket)

#### Uploaded Videos (Database)
- **`lesson_atoms.hd_video_url`:** 10-60 videos (Day 1 only)
- **`kelly_video_assets`:** 99 rows (Day 1, multi-language)
- **Status:** ✅ Day 1 complete, Days 2-365 missing

#### Target Production
| Asset Type | Per Day | Total 365 Days | Generated | Uploaded | % Complete |
|------------|---------|----------------|-----------|----------|------------|
| **HD Videos (main)** | 15 | 5,475 | ~122 | 0 | 2.2% |
| **Response Videos** | 36 | 13,140 | 0 | 0 | 0% |
| **Total Videos** | 51 | 18,615 | ~122 | 0 | 0.7% |

**Breakdown per Day:**
- 5 phases × 3 archetypes = 15 main videos
- 4 phases × 3 archetypes × 3 options = 36 response videos (Wisdom has no options)
- **Total:** 51 videos per day

### 1.4 Generation Sources

#### Primary Pipeline: HD Golden Lesson Pipeline
**Script:** `scripts/kelly-video-factory/hd-golden-lesson-pipeline.ts`

**Architecture:**
1. **Audio:** ElevenLabs TTS (`eleven_multilingual_v2`)
2. **Image:** Flux + Kelly LoRA (`lucataco/flux-dev-lora`)
3. **Motion:** MiniMax Video-01 (base video with gestures)
4. **Lip-Sync:** Sync Labs `lipsync-2-pro` (95%+ accuracy)
5. **Upload:** Supabase Storage → Database update

**Quality:** Production-grade 1080p HD

#### Alternative Pipelines

**Sync Labs Video Re-Dub (Preferred):**
- Uses completed HeyGen video as motion base
- Re-dubs with new ElevenLabs audio
- **Quality:** 95% lip-sync, perfect Kelly consistency
- **Time:** ~1-2 min/video
- **Queue:** None

**Sync Labs Fresh Generation (Fallback):**
- Replicate LoRA Image + ElevenLabs Audio → Wav2Lip → Sync Labs
- **Quality:** 85% lip-sync, variable consistency
- **Time:** ~2 min/video
- **Queue:** None

**HeyGen (Legacy/Deprecated):**
- Queue unpredictable (8+ hour delays)
- Not recommended for production

### 1.5 File Naming Conventions

#### Current Format
```
day-{dayNumber}/{archetype}/{phase}_{type}.mp4

Examples:
- day-001/explorer/hook_main.mp4
- day-001/explorer/hook_response_a.mp4
- day-001/explorer/hook_response_b.mp4
- day-001/explorer/hook_response_c.mp4
- day-001/explorer/fact1_main.mp4
- day-001/explorer/wisdom_main.mp4 (no responses)
```

#### Alternative Format (Legacy)
```
day_{DAY}_phase_{PHASE}_archetype_{ARCHETYPE}_type_{TYPE}_option_{OPTION}.mp4

Examples:
- day_001_phase_Hook_archetype_Explorer_type_main.mp4
- day_001_phase_Hook_archetype_Explorer_type_response_option_A.mp4
```

**Recommendation:** Use current format (cleaner, more consistent)

---

## 2. Quality Assessment

### 2.1 Known Quality Issues

#### Documented Issues (`VIDEO_QUALITY_ENHANCEMENT_REPORT.md`)

| Issue | Description | Impact | Status |
|-------|-------------|--------|--------|
| **Face Inconsistency** | Kelly's face varied noticeably between images | Uncanny valley effect | ✅ Fixed (enhanced prompts) |
| **Pose Monotony** | Same exact poses repeated (arms up, hand on heart) | Robotic/templated feel | ✅ Fixed (varied motion library) |
| **Background Blandness** | Plain solid colors | Stock photo aesthetic | ✅ Fixed (rich environments) |
| **Expression Limitation** | Only extreme expressions | Lacks nuance | ✅ Fixed (emotional range) |
| **Missing Context** | No environmental storytelling | Disconnected from learning | ✅ Fixed (contextual backgrounds) |

#### Current Quality Problems

**1. Uncanny Valley Triggers**
- **Symptoms:** Too-perfect skin, unnatural eye movement, robotic gestures
- **Detection:** `quality_check.py` uncanny_score < 60
- **Fix:** Enhanced prompts with natural variation, breathing, micro-expressions

**2. Audio Sync Issues**
- **Symptoms:** Lip movements don't match audio
- **Detection:** `quality_check.py` lipsync_score < 85
- **Fix:** Sync Labs `lipsync-2-pro` (95%+ accuracy)

**3. Consistency Problems**
- **Symptoms:** Kelly looks different between clips
- **Detection:** `quality_check.py` identity_score < 75
- **Fix:** Consistent LoRA scale (0.90), same source images

**4. Motion Artifacts**
- **Symptoms:** Glitches, flickers, temporal instability
- **Detection:** `quality_check.py` temporal_score < 60
- **Fix:** MiniMax Video-01 for natural motion

### 2.2 Quality Criteria

#### Automated Quality Metrics (`kelly-sync/scripts/quality_check.py`)

**1. Resolution Score (0-100)**
- **Target:** 4K (3840×2160) = 100
- **Minimum:** 1080p (1920×1080) = 50
- **Threshold:** < 1920×1080 fails

**2. Blur Score (0-100)**
- **Method:** Laplacian variance
- **Good:** Variance > 500 = 100
- **Bad:** Variance < 50 = 20
- **Threshold:** < 50 fails

**3. Temporal Consistency (0-100)**
- **Method:** Inter-frame variance (LPIPS)
- **Good:** Variance ≤ 5 = 100
- **Bad:** Variance ≥ 30 = 30
- **Threshold:** < 60 fails

**4. Face Identity (0-100)**
- **Method:** ArcFace similarity (placeholder: MediaPipe detection rate)
- **Target:** 0.75 similarity = 75
- **Threshold:** < 75 indicates inconsistency

**5. Lip Sync Accuracy (0-100)**
- **Method:** SyncNet confidence (placeholder: neutral 75)
- **Target:** 0.85 confidence = 85
- **Threshold:** < 85 indicates desync

**6. Photorealism / Uncanny Valley (0-100)**
- **Method:** Skin texture analysis, edge quality
- **Good:** Natural skin variance (10-30), 5-15% edges
- **Bad:** Too smooth or too sharp
- **Threshold:** < 60 indicates uncanny valley

#### Overall Quality Score
**Weighted Average:**
- Resolution: 15%
- Blur: 20%
- Temporal: 15%
- Identity: 15%
- Lip Sync: 20%
- Photorealism: 15%

**Pass Threshold:** 70/100 (Grade C)

**Grade Boundaries:**
- A: 90-100
- B: 80-89
- C: 70-79
- D: 60-69
- F: 0-59

### 2.3 Quality Criteria (Manual Review)

#### Visual Quality (`KELLY_TEACHING_EXCELLENCE_EVAL.md`)

| Criteria | Pass | Fail |
|----------|------|------|
| **Resolution** | 1080p minimum, 4K preferred | Below 1080p |
| **Artifacts** | Clean render, no glitches | Glitches, flickers |
| **Sharpness** | Crisp details, no excessive blur | Blurry, soft focus |
| **Color Grading** | Professional, consistent | Inconsistent, washed out |

#### Motion Quality

| Criteria | Pass | Fail |
|----------|------|------|
| **Natural Movement** | Human-like gestures | Robotic/jerky |
| **Eye Contact** | Looks at learner | Wandering gaze |
| **Breathing** | Subtle chest movement | Frozen body |
| **Emotional Range** | Shows appropriate feeling | Flat affect |

#### Lip Sync Accuracy

| Criteria | Pass | Fail |
|----------|------|------|
| **Sync Precision** | Matches audio perfectly | Visible desync |
| **Mouth Shape** | Accurate visemes | Wrong mouth shapes |
| **Timing** | On-beat with speech | Delayed/advanced |

#### Brand Consistency

| Criteria | Pass | Fail |
|----------|------|------|
| **Kelly Appearance** | Consistent face, outfit | Different Kellys |
| **Persona Costume** | Correct archetype outfit | Wrong outfit |
| **Age Variant** | Correct age face | Wrong age |
| **Expression Match** | Matches content emotion | Emotional mismatch |
| **Background** | Appropriate, not distracting | Busy/wrong background |

### 2.4 Existing Quality Process

#### Automated Quality Checker
**Script:** `kelly-sync/scripts/quality_check.py`

**Features:**
- ✅ Resolution verification
- ✅ Blur detection (Laplacian variance)
- ✅ Temporal consistency (inter-frame variance)
- ⚠️ Face identity (placeholder: MediaPipe detection)
- ⚠️ Lip sync (placeholder: neutral score)
- ✅ Uncanny valley score (photorealism)

**Usage:**
```bash
python quality_check.py video.mp4
python quality_check.py --batch output/*.mp4
python quality_check.py --compare video1.mp4 video2.mp4 --json report.json
```

**Status:** ✅ Exists but not integrated into production pipeline

#### Manual Review Process
**Current State:** ❌ No formal review queue

**Proposed Process:**
1. **Automated Check:** Run `quality_check.py` on all generated videos
2. **Review Queue:** Videos scoring 70-85 flagged for manual review
3. **Approval:** Videos scoring >85 auto-approved, <70 auto-rejected
4. **Rejection:** Failed videos logged with error message, retry count incremented

**Database Fields:**
- `kelly_video_assets.is_approved` (boolean)
- `kelly_video_assets.approved_by` (text)
- `kelly_video_assets.approved_at` (timestamp)
- `kelly_video_assets.lip_sync_quality_score` (0-1)
- `kelly_video_assets.video_quality_score` (0-1)

**Approval Rate:** Unknown (no data collected)

**Rejected Videos:** Stored in `kelly_video_assets` with `status='failed'`, can be regenerated

### 2.5 Quality Pipeline Integration

#### Current Gap
**Problem:** Quality checker exists but not called automatically

**Solution:** Integrate into `hd-golden-lesson-pipeline.ts`

```typescript
// After video generation, before upload
const qualityReport = await runQualityCheck(finalVideoPath);
if (qualityReport.overall_score < 70) {
  console.error(`❌ Quality check failed: ${qualityReport.overall_score}`);
  // Retry with different parameters or flag for manual review
  return { success: false, error: 'Quality check failed' };
}

// Update database with quality scores
await supabase.from('kelly_video_assets').update({
  lip_sync_quality_score: qualityReport.lipsync_score / 100,
  video_quality_score: qualityReport.overall_score / 100,
  is_approved: qualityReport.passed,
});
```

---

## 3. Segment Architecture

### 3.1 Current Content Structure

#### Lesson Phases (`lesson_atoms`)

**5 Phases per Lesson:**
1. **Hook** - Surprising reveal (~45 seconds)
2. **Fact1** - First teaching moment (~60 seconds)
3. **Fact2** - Second teaching moment (~60 seconds)
4. **Fact3** - Third teaching moment (~60 seconds)
5. **Wisdom** - Final insight (~45 seconds)

**Total Duration:** ~4-5 minutes per lesson

#### Script Length Analysis

**Average Script Length:**
- Hook: ~100-150 words (~45 seconds @ 2.5 words/second)
- Fact1-3: ~150-200 words each (~60 seconds)
- Wisdom: ~100-150 words (~45 seconds)

**Natural Break Points:**
- Sentence endings (`.`, `!`, `?`)
- Clause boundaries (`,`, `;`, `:`)
- Pause indicators (`...`, `—`)

### 3.2 Segment Mapping

#### Segment Types

```typescript
interface SegmentRequirement {
  type: 'welcome' | 'topic_intro' | 'question' | 'response' | 'explain' | 'transition' | 'idle';
  scope: 'generic' | 'per_lesson' | 'per_atom';
  estimatedCount: number;
  currentCount: number;
  qualityPassCount: number;
}
```

#### Current Segment Inventory

| Segment Type | Scope | Estimated Count | Current Count | Quality Pass | Notes |
|-------------|-------|----------------|----------------|--------------|-------|
| **welcome** | per_lesson | 365 | 0 | 0 | Generic greeting, reusable |
| **topic_intro** | per_lesson | 365 | ~122 | ~10 | Day-specific intro |
| **question** | per_atom | 1,095 | ~122 | ~10 | Fact1-3 question phases |
| **response** | per_atom | 4,380 | 0 | 0 | A/B/C responses (4 phases × 3 options) |
| **explain** | per_atom | 1,095 | ~122 | ~10 | Fact1-3 explanations |
| **transition** | generic | 50 | 0 | 0 | Reusable transitions |
| **idle** | generic | 20 | 0 | 0 | Waiting/thinking moments |

**Total Estimated:** ~7,370 segments  
**Total Current:** ~366 segments (5% complete)

#### Segment Duration Targets

**10-Second Segments:**
- **Words:** ~25 words per segment (2.5 words/second)
- **Natural Breaks:** Sentence/clause boundaries
- **Reusability:** Generic segments can be reused across lessons

**Current Phase Duration:**
- Hook: ~45 seconds = 4-5 segments
- Fact1-3: ~60 seconds = 6 segments each
- Wisdom: ~45 seconds = 4-5 segments

**Total Segments per Lesson:** ~22-26 segments

### 3.3 Script Splitting Algorithm

#### Implementation (`scripts/heygen-smart-scene-generator.ts`)

```typescript
function splitScript(
  script: string, 
  phaseType: 'hook' | 'fact' | 'wisdom' | 'outro',
  maxSeconds: number = 8
): ScriptSegment[] {
  const totalDuration = estimateDuration(script);
  
  // If short enough, single scene
  if (totalDuration <= maxSeconds) {
    return [{ text: script, motion, estimatedDuration: totalDuration }];
  }
  
  // Calculate number of segments needed
  const numSegments = Math.ceil(totalDuration / maxSeconds);
  const segments: ScriptSegment[] = [];
  
  // Find natural break points
  let remaining = script;
  let segmentIndex = 0;
  
  while (remaining.length > 0 && segmentIndex < numSegments) {
    const targetLength = Math.floor(script.length / numSegments);
    const breakPoint = findNaturalBreak(remaining, targetLength);
    
    segments.push({
      text: remaining.slice(0, breakPoint).trim(),
      motion: getMotionForSegment(phaseType, segmentIndex),
      estimatedDuration: estimateDuration(remaining.slice(0, breakPoint)),
    });
    
    remaining = remaining.slice(breakPoint).trim();
    segmentIndex++;
  }
  
  return segments;
}

function findNaturalBreak(text: string, targetPosition: number): number {
  // Prefer sentence endings
  const sentenceEnd = text.lastIndexOf('.', targetPosition);
  const questionEnd = text.lastIndexOf('?', targetPosition);
  const exclamationEnd = text.lastIndexOf('!', targetPosition);
  
  const bestBreak = Math.max(sentenceEnd, questionEnd, exclamationEnd);
  if (bestBreak > targetPosition * 0.7) return bestBreak + 1;
  
  // Fallback to clause boundaries
  const comma = text.lastIndexOf(',', targetPosition);
  const semicolon = text.lastIndexOf(';', targetPosition);
  
  return Math.max(comma, semicolon, targetPosition);
}
```

**Natural Break Priority:**
1. Sentence endings (`.`, `!`, `?`)
2. Clause boundaries (`,`, `;`)
3. Target position (fallback)

### 3.4 Segment Reusability Strategy

#### Generic Segments (Reusable)

**Welcome Segments:**
- "Hey there!" (generic greeting)
- "Ready to learn something amazing?" (engagement hook)
- "Let's dive in!" (transition)

**Transition Segments:**
- "Now, here's the thing..." (fact introduction)
- "But wait, there's more!" (continuation)
- "So what does this mean?" (reflection)

**Idle Segments:**
- Kelly thinking (5 seconds)
- Kelly listening (3 seconds)
- Kelly nodding (2 seconds)

**Estimated Reusable Segments:** ~70 segments

#### Lesson-Specific Segments

**Topic Intro:**
- Day-specific content
- Cannot be reused

**Question Segments:**
- Fact1-3 questions
- Day-specific but similar structure

**Response Segments:**
- A/B/C responses
- Quality-specific (good/best/redirect)
- Day-specific content

**Explain Segments:**
- Fact1-3 explanations
- Day-specific content

**Estimated Lesson-Specific Segments:** ~22 per lesson × 365 = 8,030 segments

### 3.5 Segment Quality Requirements

#### Segment-Level Quality Criteria

**1. Duration**
- **Target:** 8-12 seconds
- **Minimum:** 5 seconds
- **Maximum:** 15 seconds

**2. Natural Breaks**
- **Requirement:** Must end at sentence/clause boundary
- **Prohibition:** No mid-word or mid-phrase cuts

**3. Motion Consistency**
- **Requirement:** Motion matches segment emotion
- **Rotation:** Alternate motions to avoid monotony

**4. Audio Quality**
- **Requirement:** Clear, no artifacts
- **Sync:** Perfect lip-sync (95%+)

**5. Visual Consistency**
- **Requirement:** Same Kelly appearance throughout
- **Background:** Consistent environment

---

## 4. Recommendations

### 4.1 Immediate Actions

#### 1. Fix Storage Blocker
**Problem:** Videos generated but not uploaded  
**Solution:** Create Supabase storage bucket `kelly-videos`  
**ETA:** 15 minutes  
**Impact:** Unblocks 122 existing videos

#### 2. Upload Existing Videos
**Problem:** 122 videos sitting on local filesystem  
**Solution:** Run upload script for Days 1-5  
**ETA:** 30 minutes  
**Impact:** Day 1 complete, Days 2-5 partial

#### 3. Integrate Quality Checker
**Problem:** Quality checker not called automatically  
**Solution:** Add quality check step to `hd-golden-lesson-pipeline.ts`  
**ETA:** 2 hours  
**Impact:** Automated quality gates

### 4.2 Short-Term Improvements

#### 1. Segment Library
**Goal:** Build reusable segment library  
**Action:** Extract generic segments from existing videos  
**Benefit:** Faster generation, consistent quality

#### 2. Quality Dashboard
**Goal:** Visualize quality metrics  
**Action:** Build dashboard showing quality scores, approval rates  
**Benefit:** Data-driven quality improvements

#### 3. Batch Quality Review
**Goal:** Review all existing videos  
**Action:** Run quality checker on all 122 videos, flag failures  
**Benefit:** Identify quality issues early

### 4.3 Long-Term Strategy

#### 1. Segment-Based Generation
**Goal:** Generate reusable segments, compose into lessons  
**Action:** Build segment composer that stitches segments  
**Benefit:** Faster generation, better consistency

#### 2. Quality ML Model
**Goal:** Predict quality before generation  
**Action:** Train model on quality scores vs. generation parameters  
**Benefit:** Avoid generating low-quality videos

#### 3. Automated Regeneration
**Goal:** Auto-regenerate failed videos  
**Action:** Retry logic with different parameters  
**Benefit:** Higher success rate

---

## 5. Database Queries

### 5.1 Inventory Queries

#### Count Videos by Status
```sql
SELECT 
  status,
  COUNT(*) as count,
  COUNT(DISTINCT lesson_day) as days_covered
FROM kelly_video_assets
WHERE asset_type = 'video'
GROUP BY status;
```

#### Count Videos by Day
```sql
SELECT 
  lesson_day,
  COUNT(*) as video_count,
  COUNT(DISTINCT phase) as phases_covered,
  COUNT(DISTINCT archetype) as archetypes_covered
FROM kelly_video_assets
WHERE asset_type = 'video' AND status = 'completed'
GROUP BY lesson_day
ORDER BY lesson_day;
```

#### Quality Score Distribution
```sql
SELECT 
  CASE 
    WHEN video_quality_score >= 0.9 THEN 'A (90-100)'
    WHEN video_quality_score >= 0.8 THEN 'B (80-89)'
    WHEN video_quality_score >= 0.7 THEN 'C (70-79)'
    WHEN video_quality_score >= 0.6 THEN 'D (60-69)'
    ELSE 'F (<60)'
  END as grade,
  COUNT(*) as count
FROM kelly_video_assets
WHERE video_quality_score IS NOT NULL
GROUP BY grade
ORDER BY grade;
```

### 5.2 Quality Analysis Queries

#### Failed Videos Needing Regeneration
```sql
SELECT 
  lesson_day,
  phase,
  archetype,
  error_message,
  retry_count,
  created_at
FROM kelly_video_assets
WHERE status = 'failed'
ORDER BY retry_count ASC, created_at ASC;
```

#### Videos Pending Approval
```sql
SELECT 
  lesson_day,
  phase,
  archetype,
  video_quality_score,
  lip_sync_quality_score,
  is_approved
FROM kelly_video_assets
WHERE status = 'completed' 
  AND is_approved = false
  AND video_quality_score IS NOT NULL
ORDER BY video_quality_score ASC;
```

#### Quality Trends Over Time
```sql
SELECT 
  DATE(created_at) as generation_date,
  AVG(video_quality_score) as avg_quality,
  AVG(lip_sync_quality_score) as avg_lipsync,
  COUNT(*) as videos_generated
FROM kelly_video_assets
WHERE status = 'completed' 
  AND video_quality_score IS NOT NULL
GROUP BY DATE(created_at)
ORDER BY generation_date DESC;
```

---

## 6. Appendices

### 6.1 File Structure

```
scripts/
├── kelly-video-factory/
│   ├── hd-golden-lesson-pipeline.ts    # Main generation pipeline
│   ├── db-prompt-pipeline.ts           # Database-driven generation
│   └── upload-hd-videos-to-supabase.ts # Upload script
├── sync-labs-batch-generate.ts          # Sync Labs fallback
├── sync-labs-video-redub.ts             # Video re-dub pipeline
└── sync-videos-to-database.ts           # Database sync script

kelly-sync/
└── scripts/
    └── quality_check.py                  # Quality checker

supabase/
└── migrations/
    ├── 004_kelly_video_assets.sql       # Video asset table
    └── 20251209_complete_video_schema.sql # Generation status table
```

### 6.2 TypeScript Interfaces

```typescript
interface SegmentRequirement {
  type: 'welcome' | 'topic_intro' | 'question' | 'response' | 'explain' | 'transition' | 'idle';
  scope: 'generic' | 'per_lesson' | 'per_atom';
  estimatedCount: number;
  currentCount: number;
  qualityPassCount: number;
}

interface QualityReport {
  video_path: string;
  resolution: [number, number];
  fps: number;
  duration: number;
  frame_count: number;
  resolution_score: number;
  blur_score: number;
  temporal_score: number;
  identity_score: number;
  lipsync_score: number;
  uncanny_score: number;
  overall_score: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  passed: boolean;
  issues: string[];
}

interface VideoAsset {
  id: string;
  lesson_day: number;
  phase: string;
  age_bucket: string;
  language: string;
  archetype?: string;
  video_public_url?: string;
  video_duration_ms?: number;
  video_resolution?: string;
  lip_sync_quality_score?: number;
  video_quality_score?: number;
  is_approved: boolean;
  status: 'pending' | 'generating' | 'completed' | 'failed';
}
```

---

## 7. Summary

### Current State
- **Videos Generated:** 122 (2.2% of target)
- **Videos Uploaded:** 0 (blocked)
- **Quality Pipeline:** Exists but not integrated
- **Segment Architecture:** 5-phase structure, ~8-second natural breaks

### Critical Blockers
1. Storage bucket missing → Fix in 15 minutes
2. Quality checker not integrated → Fix in 2 hours
3. No segment library → Build over time

### Next Steps
1. ✅ Create storage bucket
2. ✅ Upload existing 122 videos
3. ✅ Integrate quality checker
4. ✅ Build segment inventory
5. ✅ Generate remaining 5,353 videos

**Target:** Complete all 5,475 HD videos for 365 days  
**ETA:** 2-3 weeks with rate limits and retries





