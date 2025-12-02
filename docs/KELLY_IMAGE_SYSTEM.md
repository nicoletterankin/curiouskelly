# Kelly Image System Architecture

**Last Updated:** December 2, 2025  
**Status:** Phase 1 Complete (Placeholder images wired)

---

## 📋 Overview

Kelly's visual state is controlled by a pose system that maps lesson phases and interactions to specific images. This document covers:

1. Current image inventory
2. Phase-to-pose mapping
3. How the code works
4. Future Supabase integration
5. Image creation guidelines

---

## 🖼️ Current Image Inventory

### Base Poses (`/kelly/poses/`)

| File | Visual | Use Case |
|------|--------|----------|
| `kelly_welcome.png` | Standing, hand on arm | Welcome, Celebrating, Proud |
| `kelly_idle.png` | Chair, hand gesturing | Explaining, Presenting, Excited |
| `kelly_hint.png` | Hand on chin | **Thinking**, Questions, Pondering |
| `kelly_clasp.png` | Hands clasped in lap | Encouraging, Attentive, Supportive |
| `kelly_choice_left.png` | Pointing left | Option A highlight |
| `kelly_choice_right.png` | Pointing right | Option B highlight |
| `kelly_listening.png` | Hands clasped, leaning in | Voice conversation, Listening |
| `kelly_hint_flip.png` | Alternate thinking pose | Variety for hints |
| `bot_right_index.png` | Pointing bottom right | Special interactions |
| `cam_right_index.png` | Pointing at camera | Direct engagement |
| `rail_left_thumb.png` | Thumbs up | Positive feedback |

### Per-Lesson Images (`/kelly/lessons/{day}/`)

For lessons 001-021 and 335-339, we have custom images:

| File | Purpose |
|------|---------|
| `lesson-{N}-hero.png` | Main Kelly image for that lesson |
| `lesson-{N}-bg.png` | Background/setting |
| `lesson-{N}-guide-point.png` | Kelly pointing/teaching |
| `lesson-{N}-prop.png` | Prop/visual aid |
| `lesson-{N}-reaction.png` | Kelly's reaction |

---

## 🎯 Phase-to-Pose Mapping

### From CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md

```javascript
LESSON_PHASES = {
  WELCOME:  { kellyPose: 'welcome' },     // kelly_welcome.png
  Q1:       { kellyPose: 'thinking' },    // kelly_hint.png
  Q2:       { kellyPose: 'thinking' },    // kelly_hint.png
  Q3:       { kellyPose: 'thinking' },    // kelly_hint.png
  HOOK:     { kellyPose: 'excited' },     // kelly_idle.png
  COMPLETE: { kellyPose: 'celebrating' }  // kelly_welcome.png
}
```

### Full State Mapping

| Semantic State | Image File | When Used |
|----------------|------------|-----------|
| `welcome` | `kelly_welcome.png` | Lesson start |
| `thinking` | `kelly_hint.png` | During questions |
| `explaining` | `kelly_idle.png` | Presenting content |
| `excited` | `kelly_idle.png` | Hook reveal |
| `celebrating` | `kelly_welcome.png` | Correct answer, completion |
| `encouraging` | `kelly_clasp.png` | Wrong answer, support |
| `listening` | `kelly_listening.png` | Voice conversation active |
| `pointing-left` | `kelly_choice_left.png` | Hover on Option A |
| `pointing-right` | `kelly_choice_right.png` | Hover on Option B |

---

## 💻 Code Architecture

### Files Involved

```
/js/
  kelly-production-assets.js  # KellyAssetManager - main asset system
  kelly-lesson-system.js      # KellyPoseManager - phase-aware poses
  kelly-2d-avatar.js          # Legacy simple pointing system
```

### KellyAssetManager Usage

```javascript
// Initialize
const kellyImg = document.getElementById('kelly-avatar');
const assets = new KellyAssetManager(kellyImg);
assets.preloadEssential();

// Change pose
assets.setState('thinking');
assets.pointLeft();
assets.celebrate();

// Phase-aware
assets.setStateForPhase('q1');  // Uses KELLY_PHASE_MAP
```

### KellyPoseManager Usage

```javascript
// Initialize with container
KellyPoseManager.init('kelly-avatar');

// Direct pose control
KellyPoseManager.setPose('thinking');
KellyPoseManager.think();
KellyPoseManager.celebrate();

// Get pose for phase
const pose = KellyPoseManager.getPoseForPhase('q1'); // 'thinking'
```

---

## 🗄️ Future: Supabase Integration

### Proposed Schema

```sql
CREATE TABLE kelly_poses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pose_name TEXT NOT NULL,           -- 'thinking', 'welcome', etc.
  image_url TEXT NOT NULL,           -- Supabase storage URL
  thumbnail_url TEXT,                -- Small preview
  use_cases TEXT[],                  -- ['question', 'pondering']
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE lesson_kelly_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lesson_day INT NOT NULL,           -- 1-365
  image_type TEXT NOT NULL,          -- 'hero', 'reaction', 'prop', etc.
  phase TEXT,                        -- 'welcome', 'q1', 'hook', etc.
  image_url TEXT NOT NULL,
  prompt_used TEXT,                  -- AI generation prompt
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Loading from Supabase

```javascript
// Future implementation
async function loadKellyImage(lessonDay, phase) {
  // First check for lesson-specific image
  const { data: lessonImg } = await supabase
    .from('lesson_kelly_images')
    .select('image_url')
    .eq('lesson_day', lessonDay)
    .eq('phase', phase)
    .single();
  
  if (lessonImg) return lessonImg.image_url;
  
  // Fall back to default pose
  const poseName = KELLY_PHASE_MAP[phase] || 'thinking';
  const { data: pose } = await supabase
    .from('kelly_poses')
    .select('image_url')
    .eq('pose_name', poseName)
    .single();
  
  return pose?.image_url || `/kelly/poses/kelly_${poseName}.png`;
}
```

---

## 🎨 Image Creation Guidelines

### For AI Generation (Future)

**Prompt Template:**
```
Kelly, a warm female educator in her late 20s with light brown hair, 
wearing a light blue sweater, sitting in a director's chair. 
Expression: [thinking/excited/welcoming]. 
Gesture: [hand on chin/pointing left/hands clasped].
Clean white studio background with soft window light.
High quality, professional photography style.
```

### Technical Requirements

| Property | Value |
|----------|-------|
| Resolution | 1024x1024 minimum |
| Format | PNG (transparent BG) or JPEG |
| Aspect Ratio | 3:4 (portrait) or 1:1 |
| File Size | <500KB optimized |
| Style | Consistent with existing Kelly images |

### Pose Consistency Checklist

- [ ] Same person/face appearance
- [ ] Same blue sweater
- [ ] Same director's chair (for seated poses)
- [ ] Same lighting direction (from right)
- [ ] Same background style (clean white)
- [ ] Expression matches intended emotion

---

## 🔧 Migration Path

### Phase 1: Placeholder Images (CURRENT)
- ✅ Wire `/kelly/poses/` images to all states
- ✅ Map phases to correct poses
- ✅ Document system

### Phase 2: Per-Lesson Images
- Generate custom hero images for each lesson
- Store in `/kelly/lessons/{day}/`
- Load lesson-specific images when available

### Phase 3: Supabase Storage
- Migrate images to Supabase Storage
- Create database tables for metadata
- Update client to load from Supabase
- Enable dynamic image generation

### Phase 4: Real-Time Generation
- AI-generated Kelly images on demand
- Lesson-context-aware poses
- User interaction responses

---

## 📊 Current File Paths

```
public/
├── kelly/
│   ├── poses/                    # Base pose images
│   │   ├── kelly_welcome.png
│   │   ├── kelly_idle.png
│   │   ├── kelly_hint.png
│   │   ├── kelly_clasp.png
│   │   ├── kelly_choice_left.png
│   │   ├── kelly_choice_right.png
│   │   ├── kelly_listening.png
│   │   └── ...
│   ├── lessons/                  # Per-lesson images
│   │   ├── 001/
│   │   ├── 002/
│   │   └── ...
│   ├── choices/                  # Choice cards
│   └── thumbnails/               # Lesson thumbnails
├── assets/kelly/production/      # Legacy (deprecated)
└── images/                       # Other images
```

---

## ⚡ Quick Reference

### Change Kelly's Pose

```javascript
// Method 1: KellyAssetManager
window.kellyAssets?.setState('thinking');

// Method 2: KellyPoseManager  
KellyPoseManager.setPose('thinking');

// Method 3: Direct (not recommended)
document.getElementById('kelly-avatar').src = '/kelly/poses/kelly_hint.png';
```

### Available Poses

```javascript
const POSES = [
  'welcome',      // Standing welcome
  'thinking',     // Hand on chin
  'explaining',   // Gesturing
  'listening',    // Attentive
  'celebrating',  // Happy
  'encouraging',  // Supportive
  'excited',      // Animated
  'pointing-left',
  'pointing-right'
];
```

---

## 📞 Contact

For Kelly image questions:
- Email: hello@curiouskelly.com
- See: `/docs/` for other implementation guides

