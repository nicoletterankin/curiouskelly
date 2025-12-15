# 🎨 VISUAL SYSTEM - COMPLETE UNDERSTANDING

## WHAT ARE VISUALS?

**Visuals are INFOGRAPHIC POPUP IMAGES** that appear when the learner clicks the 📊 button during a lesson phase.

- **NOT** backgrounds or wallpapers
- **NOT** Kelly's avatar (that's separate)
- **ARE** educational diagrams/illustrations that complement the spoken content

---

## WHEN DO THEY APPEAR?

1. User is in a lesson phase (Hook, Fact1, Fact2, Fact3, or Wisdom)
2. User clicks the **📊 button** (btn-infographic) in the UI
3. An **overlay pops up** (overlay-infographic) showing the visual
4. Visual is specific to the current phase

---

## UI FLOW

```
Lesson Playing
    ↓
User clicks 📊 button
    ↓
Overlay opens (overlay-infographic)
    ↓
Shows: <img src="{visualUrl}" alt="infographic">
    ↓
If visualUrl is NULL → Shows "Infographic Coming Soon"
```

### Code Location
```javascript
// learn.html line ~8419
document.getElementById('btn-infographic').addEventListener('click', () => {
  const overlay = document.getElementById('overlay-infographic');
  const visualUrl = currentAtom?.visualUrl;  // From lesson_atoms
  
  if (visualUrl) {
    // Show the infographic image
    imageEl.innerHTML = `<img src="${visualUrl}" ...>`;
  } else {
    // Fallback message
    imageEl.innerHTML = '📊';
    titleEl.textContent = 'Infographic Coming Soon';
  }
});
```

---

## DATABASE SCHEMA

### lesson_atoms (WHERE VISUALS ARE STORED)
```sql
CREATE TABLE lesson_atoms (
  id UUID PRIMARY KEY,
  core_lesson_id UUID REFERENCES core_lessons(id),
  archetype TEXT,           -- 'The Scientist', 'The Explorer', etc.
  phase TEXT,               -- 'Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'
  content JSONB,            -- { script: "...", text: "..." }
  visual_url TEXT,          -- 🎯 URL to infographic image
  hd_video_url TEXT,        -- Optional: URL to HD video
  created_at TIMESTAMPTZ
);
```

### kelly_video_assets (ASSET REGISTRY)
```sql
CREATE TABLE kelly_video_assets (
  id UUID PRIMARY KEY,
  day_number INT,
  phase TEXT,               -- 'hook', 'q1', 'q2', 'q3', 'wisdom' (lowercase!)
  template TEXT,            -- 'excited', 'curious', 'explain', etc.
  asset_type TEXT,          -- 'image', 'video', 'audio', 'animation'
  storage_bucket TEXT,      -- 'kelly-templates', 'lesson-visuals'
  storage_path TEXT,        -- 'production/images/day_001_hook.png'
  public_url TEXT,          -- Full public URL
  resolution TEXT,          -- '1344x768'
  status TEXT,              -- 'generated', 'pending', 'failed'
  quality_tier TEXT,        -- 'standard', 'production'
  ...
);
```

---

## CURRENT STATE (Dec 14, 2025)

| Metric | Count | Status |
|--------|-------|--------|
| **lesson_atoms.visual_url** populated | 60 | Only Day 1 |
| **lesson_atoms.hd_video_url** populated | 5 | Minimal |
| **kelly_video_assets** (type=image) | 25 | Days 1-5 |
| **kelly_video_assets** (type=video) | 169 | Partial |
| **kelly_video_assets** (type=audio) | 768 | Good coverage |

### Days with Complete Visuals
- **Day 1**: ✅ All 5 phases (hook, q1, q2, q3, wisdom)
- **Days 2-5**: ✅ Images exist in kelly_video_assets
- **Days 6-365**: ❌ No visuals

---

## STORAGE PATTERNS

### Pattern A: Direct in lesson_atoms
```
lesson_atoms.visual_url = 
  "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/
   lesson-visuals/day-001/infographics/hook.png"
```

### Pattern B: Registry in kelly_video_assets
```
kelly_video_assets:
  day_number: 1
  phase: 'hook'
  asset_type: 'image'
  storage_bucket: 'kelly-templates'
  storage_path: 'production/images/day_001_hook.png'
  public_url: 'https://...supabase.co/.../kelly-templates/production/images/day_001_hook.png'
```

**Current implementation uses BOTH:**
- Day 1 uses Pattern A (lesson_atoms.visual_url)
- Days 2-5 use Pattern B (kelly_video_assets registry)

---

## GENERATION SCRIPTS

### 1. generate-lesson-visuals.ts (GEMINI)
- **Purpose**: Generate infographic images using Gemini
- **Generates**: thumbnail + illustration + 2-3 infographics per lesson
- **Uploads to**: `lesson-visuals` bucket
- **Updates**: `lesson_visuals` table (separate table!)

### 2. generate-all-phase-visuals.ts (REPLICATE)
- **Purpose**: Generate phase visuals using Replicate Flux + Kelly LoRA
- **Generates**: 5 images per lesson (hook, q1, q2, q3, wisdom)
- **Saves to**: Local filesystem (`public/kelly/phases/`)
- **Does NOT upload** to Supabase automatically

### 3. populate-kelly-video-assets.ts
- **Purpose**: Register existing assets in kelly_video_assets table
- **Reads from**: Local files or JSON manifests
- **Writes to**: kelly_video_assets table

---

## WHAT NEEDS TO HAPPEN

### Option 1: Use Existing Script (Gemini)
```bash
# Generate visuals for all 365 days
npx tsx scripts/generate-lesson-visuals.ts 1 365

# This will:
# 1. Generate 3-5 images per lesson using Gemini
# 2. Upload to lesson-visuals bucket
# 3. Store URLs in lesson_visuals table
# 4. Need to link to lesson_atoms.visual_url
```

### Option 2: Use Kelly LoRA Script (Replicate)
```bash
# Generate phase visuals with Kelly's face
npx tsx scripts/generate-all-phase-visuals.ts --all

# Then upload:
npx tsx scripts/upload-phase-assets-to-supabase.ts --all

# Then register:
npx tsx scripts/populate-kelly-video-assets.ts --all
```

### Option 3: HYBRID (RECOMMENDED)
Use **Gemini for infographics** (no Kelly face needed) + **Kelly LoRA for thumbnails** (Kelly's face)

---

## THE MISSING LINK

The scripts generate and upload assets, but **lesson_atoms.visual_url** isn't being updated!

### Solution: Create Link Script
```typescript
// scripts/link-visuals-to-atoms.ts
// 
// For each lesson_atom:
// 1. Find matching asset in kelly_video_assets
// 2. Update lesson_atoms.visual_url with public_url
```

---

## IMMEDIATE ACTION PLAN

### Step 1: Link Existing Assets (Days 1-5)
```bash
npx tsx scripts/link-visuals-to-atoms.ts --range=1-5
```

### Step 2: Generate Visuals for Days 6-50
```bash
npx tsx scripts/generate-lesson-visuals.ts 6 50
```

### Step 3: Link New Assets
```bash
npx tsx scripts/link-visuals-to-atoms.ts --range=6-50
```

### Step 4: Verify in UI
- Open https://www.curiouskelly.com/learn.html?day=6
- Click 📊 button
- Should see infographic popup

### Step 5: Scale to All 365
Repeat for remaining days in batches

---

## COST ESTIMATE

| Provider | Images | Cost/Image | Total |
|----------|--------|------------|-------|
| Gemini Imagen | 1,825 | $0.02 | **$37** |
| Replicate Flux | 1,825 | $0.04 | **$73** |

**Recommended: Gemini** (cheaper, faster, no Kelly face needed for infographics)

---

*Understanding completed: December 14, 2025*
