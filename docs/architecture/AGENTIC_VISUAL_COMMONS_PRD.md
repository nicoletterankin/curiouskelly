# 🎨 AGENTIC VISUAL COMMONS: Complete Product Requirements Document

**Version:** 1.0.0  
**Created:** December 17, 2025  
**Status:** 📋 BLUEPRINT READY  
**Mission:** Leverage Google's AI patronage to build the world's largest structured educational content library, one learner at a time.

---

## Table of Contents

1. [Vision & Philosophy](#vision--philosophy)
2. [System Overview](#system-overview)
3. [User Experience Flow](#user-experience-flow)
4. [Technical Architecture](#technical-architecture)
5. [Database Schema](#database-schema)
6. [Prompt Engineering Library](#prompt-engineering-library)
7. [API Specifications](#api-specifications)
8. [UI/UX Components](#uiux-components)
9. [Integration with Lesson Phases](#integration-with-lesson-phases)
10. [Cost & Sustainability Model](#cost--sustainability-model)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Agent System Prompt](#agent-system-prompt)

---

## Vision & Philosophy

### The Core Insight

Google offers **500-1000 free image generations per day** to anyone with an API key. There are **millions of curious learners**. If each learner contributes just one generation per lesson, and that generation is cached forever for everyone else, we build an unprecedented educational asset library at near-zero cost.

### The Commons Principle

> "The first learner to explore a concept illuminates the path. Every learner after walks in light."

Every generated visual becomes a **community asset**. Unlike traditional AI apps where generations are ephemeral and wasteful, our system:

1. **Hashes every prompt** → Creates unique content address
2. **Checks cache first** → Serves existing assets instantly  
3. **Generates only when needed** → Fills gaps in the library
4. **Saves everything** → One generation serves millions forever

### Why This Matters

- **365 lessons × 7 phases × 12 archetypes × 6 age groups × 3 visual types = 550,000+ unique educational visuals**
- At $0.02/image (Imagen 4 Fast), that's $11,000 if we paid for everything
- With learner-powered commons, we pay effectively **$0** and build faster
- Each learner's contribution makes the next learner's experience richer

---

## System Overview

### The Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE VISUAL COMMONS LOOP                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   LEARNER    │    │   CACHE CHECK    │    │   VISUAL DISPLAYED       │   │
│  │  enters      │───▶│  content_hash    │───▶│   in phase context       │   │
│  │  lesson      │    │  exists?         │    │   (instant, free)        │   │
│  └──────────────┘    └────────┬─────────┘    └──────────────────────────┘   │
│                               │                                              │
│                         [NO CACHE]                                           │
│                               │                                              │
│                               ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      GENERATION FLOW                                  │   │
│  │                                                                       │   │
│  │   1. Show "✨ Generate Visual" button in phase                       │   │
│  │   2. Learner clicks (or auto-generates if they have BYOK enabled)    │   │
│  │   3. System selects API key:                                          │   │
│  │      • Learner's BYOK key (their free credits)                       │   │
│  │      • Platform key (rate-limited pool)                              │   │
│  │   4. Send structured prompt to Gemini/Imagen                         │   │
│  │   5. Receive image/SVG data                                          │   │
│  │   6. Upload to Supabase Storage                                      │   │
│  │   7. Register in visual_commons table with content_hash              │   │
│  │   8. Display to learner with attribution                             │   │
│  │   9. All future learners get this cached version                     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      ATTRIBUTION & GAMIFICATION                       │   │
│  │                                                                       │   │
│  │   "Visual contributed by @curious_maya • 2,341 learners helped"      │   │
│  │                                                                       │   │
│  │   Profile badge: "Visual Pioneer 🎨" (contributed 10+ visuals)       │   │
│  │   Streak bonus: "Illuminate a lesson today to extend your streak"    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## User Experience Flow

### Scenario 1: Cached Visual Exists (99% of cases after ramp-up)

```
Learner opens Day 17 "Why We Dream" → Hook phase
         │
         ▼
    ┌─────────────────────────────────┐
    │  🌙 Hook: What Are Dreams?      │
    │                                  │
    │  [Kelly speaking]               │
    │                                  │
    │  ┌─────────────────────────┐    │
    │  │  🖼️ [Infographic loads] │    │  ← Cached visual appears instantly
    │  │  "The Dreaming Brain"   │    │
    │  │  Contributed by @maya   │    │
    │  └─────────────────────────┘    │
    │                                  │
    │  📊 Tap for full-screen visual  │
    └─────────────────────────────────┘
```

### Scenario 2: No Cached Visual (Opportunity to Contribute)

```
Learner opens Day 312 "Quantum Entanglement" → Fact 2 phase
         │
         ▼
    ┌─────────────────────────────────┐
    │  ⚛️ Fact 2: Spooky Action       │
    │                                  │
    │  [Kelly speaking]               │
    │                                  │
    │  ┌─────────────────────────┐    │
    │  │  📊 No visual yet       │    │
    │  │                          │    │
    │  │  ✨ Generate Visual      │    │  ← CTA button pulses gently
    │  │  "Be the first to        │    │
    │  │   illuminate this!"     │    │
    │  └─────────────────────────┘    │
    │                                  │
    └─────────────────────────────────┘
         │
    [Learner clicks "Generate Visual"]
         │
         ▼
    ┌─────────────────────────────────┐
    │  ✨ Creating visual...          │
    │                                  │
    │  ┌─────────────────────────┐    │
    │  │  ⏳ [Shimmer animation] │    │
    │  │  "Kelly is sketching..."│    │
    │  └─────────────────────────┘    │
    │                                  │
    │  Using: Your Google AI credits  │  ← Or "Curious Kelly credits"
    └─────────────────────────────────┘
         │
    [3-5 seconds later]
         │
         ▼
    ┌─────────────────────────────────┐
    │  🎉 Visual Created!             │
    │                                  │
    │  ┌─────────────────────────┐    │
    │  │  🖼️ [New infographic]   │    │
    │  │  "Quantum Entanglement" │    │
    │  │  ⭐ You contributed!    │    │
    │  └─────────────────────────┘    │
    │                                  │
    │  "You just helped 0 future      │
    │   learners. We'll tell you      │
    │   when someone uses your visual!"│
    └─────────────────────────────────┘
```

### Scenario 3: BYOK Setup in Settings

```
Settings → Learning Preferences → Advanced
         │
         ▼
    ┌─────────────────────────────────────────────────────┐
    │  🔑 Your Google AI Key (Optional)                   │
    │                                                      │
    │  Add your own API key to generate unlimited         │
    │  visuals using your free Google AI Studio credits.  │
    │                                                      │
    │  ┌────────────────────────────────────────┐         │
    │  │  API Key: AIza••••••••••••••••••••••   │         │
    │  └────────────────────────────────────────┘         │
    │                                                      │
    │  [Test Key] [Save Key] [Remove Key]                 │
    │                                                      │
    │  ────────────────────────────────────────────────   │
    │                                                      │
    │  📊 Your Impact                                     │
    │  • Visuals contributed: 23                          │
    │  • Learners helped: 4,892                           │
    │  • Badge: Visual Pioneer 🎨                         │
    │                                                      │
    │  ────────────────────────────────────────────────   │
    │                                                      │
    │  ℹ️ How to get your key:                            │
    │  1. Go to aistudio.google.com                       │
    │  2. Sign in with Google                             │
    │  3. Click "Get API Key"                             │
    │  4. Copy and paste here                             │
    │                                                      │
    │  Your key stays on your device. We never store it.  │
    │  Google gives everyone 500 free generations/day.    │
    │                                                      │
    └─────────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (learn.html)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│  │ Phase Display │  │ Visual Slot   │  │ Generate CTA  │                    │
│  │ Component     │  │ Component     │  │ Component     │                    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                    │
│          │                  │                   │                            │
│          └──────────────────┴───────────────────┘                            │
│                             │                                                │
│                    ┌────────┴────────┐                                       │
│                    │ VisualCommons   │                                       │
│                    │ Controller      │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API LAYER (Vercel Serverless)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐     │
│  │ /api/visual/check  │  │ /api/visual/gen    │  │ /api/visual/stats  │     │
│  │ GET: Check cache   │  │ POST: Generate     │  │ GET: User impact   │     │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘     │
│            │                       │                        │                │
│            └───────────────────────┴────────────────────────┘                │
│                                    │                                         │
└────────────────────────────────────┼─────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   SUPABASE           │  │   GOOGLE AI STUDIO   │  │   SUPABASE STORAGE   │
│   visual_commons     │  │   Gemini/Imagen API  │  │   visuals bucket     │
│   table              │  │                      │  │                      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### Content Hash Algorithm

```typescript
/**
 * Generates a deterministic, unique hash for any visual context.
 * Same context = same hash = cache hit.
 */
function generateVisualHash(context: VisualContext): string {
  // Normalize and order all context fields
  const normalized = {
    d: context.dayNumber,                    // Day 1-365
    p: context.phase.toLowerCase(),          // hook, cliff, fact1, etc.
    t: normalizeText(context.topic),         // "Why We Dream"
    v: context.visualType,                   // 'infographic' | 'diagram' | 'scene'
    a: context.ageGroup || 'all',            // '2-5', '6-12', '13+', 'all'
    s: context.style || 'default',           // 'playful', 'scientific', 'default'
    ver: '1'                                 // Schema version for invalidation
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return sha256(canonical);
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, '')  // Remove punctuation
    .replace(/\s+/g, ' ');     // Normalize whitespace
}

// Example:
// Day 17, Hook phase, "Why We Dream", infographic, ages 6-12
// → Hash: "a3f8c2e1d9b7..."
```

### API Key Management

```typescript
interface KeySource {
  type: 'byok' | 'platform';
  key: string;
  dailyLimit: number;
  usedToday: number;
}

async function getApiKeyForGeneration(userId?: string): Promise<KeySource | null> {
  // 1. Check if user has BYOK enabled
  if (userId) {
    const userKey = await getUserApiKey(userId);
    if (userKey && userKey.isValid) {
      return {
        type: 'byok',
        key: userKey.decryptedKey,
        dailyLimit: 500,  // Google's free tier
        usedToday: await getUserDailyUsage(userId)
      };
    }
  }
  
  // 2. Fall back to platform key pool
  const platformKey = await getPlatformKeyWithCapacity();
  if (platformKey) {
    return {
      type: 'platform',
      key: platformKey.key,
      dailyLimit: 100,  // Conservative per-user limit
      usedToday: await getAnonUserDailyUsage(getAnonId())
    };
  }
  
  // 3. No capacity available
  return null;
}
```

---

## Database Schema

### New Table: `visual_commons`

```sql
-- The Visual Commons: Every learner-generated educational visual
CREATE TABLE visual_commons (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Content-addressable identity
  content_hash TEXT UNIQUE NOT NULL,      -- SHA-256 of normalized context
  
  -- Context metadata (denormalized for fast queries)
  day_number INTEGER NOT NULL CHECK (day_number >= 1 AND day_number <= 365),
  phase TEXT NOT NULL CHECK (phase IN (
    'hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'
  )),
  topic TEXT NOT NULL,                    -- Lesson topic
  visual_type TEXT NOT NULL CHECK (visual_type IN (
    'infographic', 'diagram', 'scene', 'comparison', 'timeline', 'process'
  )),
  age_group TEXT DEFAULT 'all' CHECK (age_group IN (
    '2-5', '6-12', '13-17', '18+', 'all'
  )),
  style TEXT DEFAULT 'default',
  
  -- The actual asset
  storage_path TEXT NOT NULL,             -- Supabase storage path
  public_url TEXT NOT NULL,               -- CDN URL
  thumbnail_url TEXT,                     -- 200x200 preview
  width INTEGER,
  height INTEGER,
  file_size_bytes INTEGER,
  format TEXT DEFAULT 'png' CHECK (format IN ('png', 'webp', 'svg', 'jpg')),
  
  -- Generation metadata
  prompt_used TEXT NOT NULL,              -- Full prompt for reproducibility
  model_used TEXT NOT NULL,               -- 'gemini-2.0-flash', 'imagen-4-fast'
  generation_params JSONB DEFAULT '{}',   -- temperature, seed, etc.
  generation_time_ms INTEGER,             -- How long it took
  estimated_cost DECIMAL(10,6),           -- $0.02 for Imagen, $0 for Gemini text
  
  -- Attribution
  generated_by UUID REFERENCES auth.users(id),  -- NULL for anonymous
  generated_by_display_name TEXT,         -- Cached for fast display
  generation_source TEXT NOT NULL CHECK (generation_source IN (
    'byok',                               -- User's own API key
    'platform',                           -- Our API key
    'staff',                              -- Admin-generated
    'seed'                                -- Pre-seeded content
  )),
  
  -- Usage tracking
  view_count INTEGER DEFAULT 0,           -- How many times displayed
  unique_learners_helped INTEGER DEFAULT 0,  -- Distinct users who saw it
  last_viewed_at TIMESTAMPTZ,
  
  -- Moderation
  status TEXT DEFAULT 'active' CHECK (status IN (
    'pending',                            -- Awaiting moderation (optional)
    'active',                             -- Live in production
    'flagged',                            -- Reported by users
    'removed'                             -- Hidden from view
  )),
  flagged_reason TEXT,
  moderated_by UUID REFERENCES auth.users(id),
  moderated_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX idx_vc_hash ON visual_commons(content_hash);
CREATE INDEX idx_vc_day_phase ON visual_commons(day_number, phase);
CREATE INDEX idx_vc_day_phase_age ON visual_commons(day_number, phase, age_group);
CREATE INDEX idx_vc_generator ON visual_commons(generated_by) WHERE generated_by IS NOT NULL;
CREATE INDEX idx_vc_status ON visual_commons(status) WHERE status = 'active';
CREATE INDEX idx_vc_popular ON visual_commons(unique_learners_helped DESC) WHERE status = 'active';

-- RLS Policies
ALTER TABLE visual_commons ENABLE ROW LEVEL SECURITY;

-- Anyone can read active visuals
CREATE POLICY "Public read access" ON visual_commons
  FOR SELECT USING (status = 'active');

-- Authenticated users can insert (moderated or instant based on trust level)
CREATE POLICY "Authenticated insert" ON visual_commons
  FOR INSERT WITH CHECK (auth.uid() IS NOT NULL OR generation_source = 'platform');

-- Only staff can update status
CREATE POLICY "Staff moderation" ON visual_commons
  FOR UPDATE USING (
    EXISTS (SELECT 1 FROM user_roles WHERE user_id = auth.uid() AND role = 'staff')
  );
```

### New Table: `visual_generation_queue`

```sql
-- Queue for background generation (when immediate generation isn't possible)
CREATE TABLE visual_generation_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  content_hash TEXT NOT NULL,
  context JSONB NOT NULL,                 -- Full VisualContext
  prompt TEXT NOT NULL,
  
  requested_by UUID REFERENCES auth.users(id),
  priority INTEGER DEFAULT 5,             -- 1=highest, 10=lowest
  
  status TEXT DEFAULT 'pending' CHECK (status IN (
    'pending', 'processing', 'completed', 'failed'
  )),
  attempts INTEGER DEFAULT 0,
  last_error TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  
  -- Link to result
  visual_id UUID REFERENCES visual_commons(id)
);

CREATE INDEX idx_vgq_pending ON visual_generation_queue(priority, created_at) 
  WHERE status = 'pending';
```

### New Table: `user_visual_contributions`

```sql
-- Aggregate stats for gamification (updated via triggers)
CREATE TABLE user_visual_contributions (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id),
  
  total_contributed INTEGER DEFAULT 0,
  total_learners_helped INTEGER DEFAULT 0,
  
  -- Badges earned
  badges JSONB DEFAULT '[]',              -- ['visual_pioneer', 'master_illuminator']
  
  -- Time-based stats
  contributions_this_week INTEGER DEFAULT 0,
  contributions_this_month INTEGER DEFAULT 0,
  
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger to update stats when visual is viewed
CREATE OR REPLACE FUNCTION update_visual_stats()
RETURNS TRIGGER AS $$
BEGIN
  -- Update view count on visual
  UPDATE visual_commons 
  SET view_count = view_count + 1,
      last_viewed_at = NOW()
  WHERE id = NEW.visual_id;
  
  -- Update contributor stats if not self-view
  IF NEW.viewer_id IS DISTINCT FROM (
    SELECT generated_by FROM visual_commons WHERE id = NEW.visual_id
  ) THEN
    UPDATE user_visual_contributions
    SET total_learners_helped = total_learners_helped + 1,
        updated_at = NOW()
    WHERE user_id = (
      SELECT generated_by FROM visual_commons WHERE id = NEW.visual_id
    );
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## Prompt Engineering Library

### The Prompt Architecture

Every generation uses a **three-layer prompt**:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: SYSTEM CONTEXT (Fixed)                                │
│  Brand guidelines, safety rails, output format                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: VISUAL TYPE TEMPLATE (Per visual type)               │
│  Infographic structure, diagram rules, scene composition       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: LESSON CONTEXT (Dynamic)                             │
│  Topic, phase, age group, specific facts                       │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: System Context

```typescript
const SYSTEM_CONTEXT = `
You are Kelly's Visual Design Lead, creating educational graphics for Curious Kelly, 
a daily learning platform for curious minds of all ages.

BRAND IDENTITY:
- Palette: Dark premium backgrounds (#0a0a0b, #18181b), clean neon accents (#3b82f6 blue, #fbbf24 gold, #22c55e green)
- Typography: Modern, clean, minimal. Headlines bold, body simple.
- Vibe: Approachable science, wonder without intimidation, clarity over complexity

HARD CONSTRAINTS:
- NEVER include text that could be misread (no tiny labels, no cursive)
- NEVER depict violence, fear, or inappropriate content
- NEVER use copyrighted characters or logos
- ALWAYS ensure scientific accuracy
- ALWAYS make it understandable at a glance
- ALWAYS leave space for Kelly to appear alongside

AGE ADAPTATIONS:
- Ages 2-5: Bright colors, rounded shapes, friendly anthropomorphized elements
- Ages 6-12: Balanced detail, clear labels, "cool factor" elements
- Ages 13-17: More sophisticated, can include complexity
- Ages 18+: Full scientific detail, professional aesthetic

OUTPUT:
Return ONLY the requested format (JSON for infographic briefs, image for scenes).
No explanations, no commentary, no apologies.
`;
```

### Layer 2: Visual Type Templates

#### Infographic Template (SVG-based, text-safe)

```typescript
const INFOGRAPHIC_TEMPLATE = `
TASK: Generate a structured infographic brief as JSON.

TEMPLATE OPTIONS:
1. cross_section - Layered diagram showing internal structure
2. process_flow - 3-step horizontal flow with arrows
3. compare - Two-panel side-by-side comparison
4. timeline - Chronological sequence
5. radial - Central concept with orbital related ideas

OUTPUT SCHEMA:
{
  "template": "cross_section" | "process_flow" | "compare" | "timeline" | "radial",
  "headline": "8 words max, compelling hook",
  "subhead": "16 words max, clarifying detail",
  "callouts": [
    { "label": "4 words max", "detail": "18 words max", "icon": "atom|spark|arrow|leaf|heart|wave|dot" }
  ],
  "steps": [...],  // For process_flow
  "left": { "label": "...", "bullets": [...] },  // For compare
  "right": { "label": "...", "bullets": [...] }, // For compare
  "centerLabel": "...",  // For radial
  "orbitals": [...]  // For radial
}

RULES:
- Choose the template that best fits the educational content
- Labels MUST be short (≤4 words) - we render real text, not image text
- Details should be kid-friendly but accurate
- Use the icon that best represents each concept
`;
```

#### Scene Template (Image generation)

```typescript
const SCENE_TEMPLATE = `
TASK: Generate a photorealistic educational scene.

COMPOSITION RULES:
- Leave the right 40% of frame clear (Kelly will appear there)
- Subject should be left-center framed
- Lighting should be warm, inviting, professional
- Background should be contextually appropriate but not distracting

STYLE: 
- Professional photography aesthetic
- 16:9 aspect ratio
- 4K resolution quality
- Natural lighting preferred
- No text overlays

EDUCATIONAL FOCUS:
- The scene should immediately communicate the core concept
- Include visual details that teach (e.g., for "photosynthesis", show sunlight hitting leaves)
- Make abstract concepts tangible through metaphor
`;
```

#### Diagram Template (Technical illustrations)

```typescript
const DIAGRAM_TEMPLATE = `
TASK: Generate a clear technical diagram.

STYLE:
- Clean vector aesthetic
- Limited color palette (3-4 colors max)
- Clear visual hierarchy
- Arrows and flow indicators where appropriate

ELEMENTS:
- Main subject prominently displayed
- Supporting elements in correct relationship
- Clear visual distinction between components
- Space for overlay labels (we add text separately)

AVOID:
- Cluttered compositions
- Ambiguous relationships
- Overly complex detail that obscures main concept
`;
```

### Layer 3: Lesson Context Generators

```typescript
function buildPromptForPhase(
  lesson: Lesson,
  phase: Phase,
  visualType: VisualType,
  ageGroup: AgeGroup
): string {
  const phasePrompts: Record<Phase, (l: Lesson) => string> = {
    hook: (l) => `
      Create a ${visualType} that captures curiosity about: "${l.topic}"
      
      This is the HOOK phase - the goal is to make learners say "Wait, what?!"
      
      Universal truth to visualize: ${l.universal_truth}
      Key hook question: ${l.hook_question || 'Why should I care about this?'}
      
      The visual should create intrigue, not answer questions yet.
    `,
    
    cliff: (l) => `
      Create a ${visualType} that presents the central mystery of: "${l.topic}"
      
      This is the CLIFF phase - learners just made a choice and want to know more.
      
      The visual should deepen the mystery while hinting at the answer.
      Show the tension between what we think we know and what's actually true.
    `,
    
    fact1: (l) => `
      Create a ${visualType} explaining the FIRST key fact about: "${l.topic}"
      
      Fact to visualize: ${l.facts?.[0] || 'The foundational concept'}
      
      This is building-block content. Make it crystal clear.
      The visual should be something a learner could explain to a friend.
    `,
    
    fact2: (l) => `
      Create a ${visualType} explaining the SECOND key fact about: "${l.topic}"
      
      Fact to visualize: ${l.facts?.[1] || 'The deeper insight'}
      
      This builds on fact1. Show the connection or progression.
    `,
    
    fact3: (l) => `
      Create a ${visualType} explaining the THIRD key fact about: "${l.topic}"
      
      Fact to visualize: ${l.facts?.[2] || 'The surprising detail'}
      
      This is often the "wow" moment. Make it memorable.
    `,
    
    wisdom: (l) => `
      Create a ${visualType} that crystallizes the wisdom of: "${l.topic}"
      
      Universal truth: ${l.universal_truth}
      Life application: ${l.life_application || 'How this applies beyond the lesson'}
      
      This visual should feel like a "poster on the wall" - something worth remembering.
      It should connect the lesson to the learner's own life.
    `,
    
    outro: (l) => `
      Create a ${visualType} that celebrates completing the lesson on: "${l.topic}"
      
      The visual should feel like an achievement - the learner now knows something new.
      Include a forward-looking element (tomorrow's teaser or "what's next").
    `,
    
    complete: (l) => `
      Create a ${visualType} summarizing the entire lesson on: "${l.topic}"
      
      This is a comprehensive visual that captures:
      1. The hook question
      2. The key facts
      3. The wisdom/application
      
      Think "one image that teaches the whole lesson."
    `
  };
  
  const ageAdaptation = getAgeAdaptation(ageGroup);
  
  return `
${SYSTEM_CONTEXT}

${getVisualTypeTemplate(visualType)}

LESSON CONTEXT:
${phasePrompts[phase](lesson)}

AGE GROUP: ${ageGroup}
${ageAdaptation}

TOPIC: ${lesson.topic}
DAY: ${lesson.day_number}
PHASE: ${phase}
`;
}

function getAgeAdaptation(age: AgeGroup): string {
  const adaptations = {
    '2-5': `
      For young children (ages 2-5):
      - Use bright, saturated colors
      - Include friendly, rounded shapes
      - Anthropomorphize concepts when helpful (happy sun, curious atoms)
      - Keep complexity very low - one main idea
      - Make it feel like a picture book illustration
    `,
    '6-12': `
      For elementary/middle school (ages 6-12):
      - Balance fun with accuracy
      - Include "cool factor" elements (space, dinosaurs, explosions if relevant)
      - Can show more detail and relationships
      - Make them feel smart for understanding
    `,
    '13-17': `
      For teens (ages 13-17):
      - More sophisticated visual language
      - Can include complexity and nuance
      - Avoid anything that feels "babyish"
      - Make it feel current and relevant
    `,
    '18+': `
      For adults (ages 18+):
      - Full scientific accuracy and detail
      - Professional, polished aesthetic
      - Can include technical terminology
      - Respect their intelligence
    `,
    'all': `
      For all ages:
      - Universal visual language
      - Clear at any age, deeper for those who look closer
      - Layered complexity - simple surface, rich details
    `
  };
  return adaptations[age] || adaptations['all'];
}
```

---

## API Specifications

### GET /api/visual/check

Check if a visual exists for given context.

```typescript
// Request
GET /api/visual/check?day=17&phase=hook&age=6-12&type=infographic

// Response (cache hit)
{
  "exists": true,
  "visual": {
    "id": "abc-123",
    "publicUrl": "https://storage.supabase.co/visuals/a3f8c2e1d9b7.png",
    "thumbnailUrl": "https://storage.supabase.co/visuals/thumbs/a3f8c2e1d9b7.png",
    "generatedBy": {
      "displayName": "@curious_maya",
      "isAnonymous": false
    },
    "helpedCount": 2341,
    "createdAt": "2025-12-15T10:30:00Z"
  }
}

// Response (cache miss)
{
  "exists": false,
  "canGenerate": true,
  "estimatedCost": 0,  // 0 if BYOK or platform has capacity
  "keySource": "byok"  // or "platform" or "unavailable"
}
```

### POST /api/visual/generate

Generate a new visual for given context.

```typescript
// Request
POST /api/visual/generate
{
  "dayNumber": 17,
  "phase": "hook",
  "ageGroup": "6-12",
  "visualType": "infographic",
  "userApiKey": "AIza...",  // Optional, if BYOK
}

// Response (success)
{
  "success": true,
  "visual": {
    "id": "def-456",
    "publicUrl": "https://storage.supabase.co/visuals/new123.png",
    "contentHash": "a3f8c2e1d9b7...",
    "generationTimeMs": 3420,
    "keySource": "byok",
    "attribution": {
      "message": "You just illuminated this lesson!",
      "isFirstContributor": true
    }
  }
}

// Response (rate limited)
{
  "success": false,
  "error": "rate_limited",
  "message": "You've used your daily generation limit. Try again tomorrow!",
  "resetAt": "2025-12-18T00:00:00Z"
}
```

### GET /api/visual/stats

Get user's contribution stats.

```typescript
// Request
GET /api/visual/stats

// Response
{
  "totalContributed": 23,
  "totalLearnersHelped": 4892,
  "badges": [
    { "id": "visual_pioneer", "name": "Visual Pioneer 🎨", "earnedAt": "2025-12-10" }
  ],
  "rank": 142,  // Out of all contributors
  "recentContributions": [
    {
      "visualId": "abc-123",
      "topic": "Why We Dream",
      "phase": "hook",
      "helpedCount": 234,
      "createdAt": "2025-12-15T10:30:00Z"
    }
  ],
  "impactThisWeek": {
    "contributions": 3,
    "learnersHelped": 892
  }
}
```

---

## UI/UX Components

### Visual Slot Component

```html
<!-- Embedded in phase display -->
<div class="visual-slot" data-phase="hook" data-day="17">
  <!-- State: Loading -->
  <div class="visual-loading" style="display: none;">
    <div class="shimmer"></div>
    <span>Checking for visual...</span>
  </div>
  
  <!-- State: Cached visual exists -->
  <div class="visual-cached" style="display: none;">
    <img class="visual-image" src="" alt="" loading="lazy">
    <div class="visual-attribution">
      <span class="contributor">Contributed by <strong>@curious_maya</strong></span>
      <span class="impact">2,341 learners helped</span>
    </div>
    <button class="visual-expand" aria-label="View full size">⛶</button>
  </div>
  
  <!-- State: No visual, can generate -->
  <div class="visual-generate" style="display: none;">
    <div class="visual-placeholder">
      <span class="placeholder-icon">📊</span>
      <span class="placeholder-text">No visual yet</span>
    </div>
    <button class="generate-cta">
      <span class="sparkle">✨</span>
      Generate Visual
      <span class="cta-subtext">Be the first to illuminate!</span>
    </button>
    <div class="key-source">
      Using: <span class="key-type">Your Google AI credits</span>
    </div>
  </div>
  
  <!-- State: Generating -->
  <div class="visual-generating" style="display: none;">
    <div class="generation-animation">
      <div class="kelly-sketching"></div>
      <span>Kelly is creating your visual...</span>
    </div>
    <div class="generation-progress">
      <div class="progress-bar"></div>
    </div>
  </div>
  
  <!-- State: Generation complete -->
  <div class="visual-complete" style="display: none;">
    <img class="visual-image" src="" alt="">
    <div class="completion-celebration">
      <span class="confetti">🎉</span>
      <span class="message">You illuminated this lesson!</span>
      <span class="impact-preview">0 learners helped (and counting!)</span>
    </div>
  </div>
  
  <!-- State: Error -->
  <div class="visual-error" style="display: none;">
    <span class="error-icon">⚠️</span>
    <span class="error-message">Generation unavailable right now</span>
    <button class="retry-button">Try Again</button>
  </div>
</div>
```

### CSS Styles

```css
/* Visual Slot Styles */
.visual-slot {
  width: 100%;
  max-width: 400px;
  aspect-ratio: 16/9;
  border-radius: 16px;
  overflow: hidden;
  background: var(--surface-elevated);
  border: 1px solid var(--border-default);
  position: relative;
  margin: 16px auto;
}

.visual-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.visual-slot:hover .visual-image {
  transform: scale(1.02);
}

.visual-attribution {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 12px;
  background: linear-gradient(transparent, rgba(0,0,0,0.8));
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: rgba(255,255,255,0.9);
}

.generate-cta {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 16px 32px;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
  border: none;
  border-radius: 12px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.generate-cta:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}

.generate-cta .sparkle {
  animation: sparkle 1.5s ease-in-out infinite;
}

@keyframes sparkle {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.2); }
}

.generation-animation {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.kelly-sketching {
  width: 80px;
  height: 80px;
  background: url('/kelly/poses/thinking.png') center/contain no-repeat;
  animation: sketch 0.5s ease-in-out infinite alternate;
}

@keyframes sketch {
  from { transform: rotate(-5deg); }
  to { transform: rotate(5deg); }
}

.completion-celebration {
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.85);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  animation: celebrationFadeIn 0.5s ease-out;
}

@keyframes celebrationFadeIn {
  from { opacity: 0; transform: scale(0.9); }
  to { opacity: 1; transform: scale(1); }
}

.confetti {
  font-size: 48px;
  animation: confettiBounce 0.5s ease-out;
}

@keyframes confettiBounce {
  0% { transform: scale(0) rotate(-180deg); }
  60% { transform: scale(1.2) rotate(10deg); }
  100% { transform: scale(1) rotate(0); }
}
```

### JavaScript Controller

```javascript
class VisualCommonsController {
  constructor() {
    this.cache = new Map();
    this.currentGeneration = null;
  }
  
  async init(container) {
    this.container = container;
    this.elements = {
      loading: container.querySelector('.visual-loading'),
      cached: container.querySelector('.visual-cached'),
      generate: container.querySelector('.visual-generate'),
      generating: container.querySelector('.visual-generating'),
      complete: container.querySelector('.visual-complete'),
      error: container.querySelector('.visual-error')
    };
    
    // Bind events
    container.querySelector('.generate-cta')?.addEventListener('click', () => this.generate());
    container.querySelector('.retry-button')?.addEventListener('click', () => this.generate());
    container.querySelector('.visual-expand')?.addEventListener('click', () => this.expand());
    
    // Check for cached visual
    await this.check();
  }
  
  async check() {
    this.showState('loading');
    
    const context = this.getContext();
    const hash = this.generateHash(context);
    
    // Check local cache first
    if (this.cache.has(hash)) {
      this.showCached(this.cache.get(hash));
      return;
    }
    
    try {
      const response = await fetch(
        `/api/visual/check?day=${context.dayNumber}&phase=${context.phase}&age=${context.ageGroup}&type=${context.visualType}`
      );
      const data = await response.json();
      
      if (data.exists) {
        this.cache.set(hash, data.visual);
        this.showCached(data.visual);
      } else {
        this.showGenerateOption(data);
      }
    } catch (error) {
      console.error('Visual check failed:', error);
      this.showState('error');
    }
  }
  
  async generate() {
    if (this.currentGeneration) return;
    
    this.showState('generating');
    const context = this.getContext();
    
    try {
      const response = await fetch('/api/visual/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dayNumber: context.dayNumber,
          phase: context.phase,
          ageGroup: context.ageGroup,
          visualType: context.visualType,
          userApiKey: this.getUserApiKey()
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        this.cache.set(this.generateHash(context), data.visual);
        this.showComplete(data.visual);
        this.trackContribution(data.visual);
      } else {
        this.showError(data.message);
      }
    } catch (error) {
      console.error('Visual generation failed:', error);
      this.showError('Generation failed. Please try again.');
    } finally {
      this.currentGeneration = null;
    }
  }
  
  getContext() {
    return {
      dayNumber: parseInt(this.container.dataset.day),
      phase: this.container.dataset.phase,
      ageGroup: window.kellyState?.ageGroup || 'all',
      visualType: 'infographic'
    };
  }
  
  generateHash(context) {
    const str = JSON.stringify({
      d: context.dayNumber,
      p: context.phase,
      a: context.ageGroup,
      t: context.visualType
    });
    return btoa(str);  // Simple hash for client-side cache
  }
  
  getUserApiKey() {
    return localStorage.getItem('kelly_google_api_key');
  }
  
  showState(state) {
    Object.values(this.elements).forEach(el => el.style.display = 'none');
    if (this.elements[state]) {
      this.elements[state].style.display = 'flex';
    }
  }
  
  showCached(visual) {
    this.showState('cached');
    const img = this.elements.cached.querySelector('.visual-image');
    img.src = visual.publicUrl;
    img.alt = `Educational visual for ${this.getContext().phase}`;
    
    const contributor = this.elements.cached.querySelector('.contributor strong');
    contributor.textContent = visual.generatedBy?.displayName || 'A curious learner';
    
    const impact = this.elements.cached.querySelector('.impact');
    impact.textContent = `${visual.helpedCount.toLocaleString()} learners helped`;
  }
  
  showGenerateOption(data) {
    this.showState('generate');
    const keyType = this.elements.generate.querySelector('.key-type');
    keyType.textContent = data.keySource === 'byok' 
      ? 'Your Google AI credits' 
      : 'Curious Kelly credits';
  }
  
  showComplete(visual) {
    this.showState('complete');
    const img = this.elements.complete.querySelector('.visual-image');
    img.src = visual.publicUrl;
    
    // Auto-transition to cached view after celebration
    setTimeout(() => {
      this.showCached({
        ...visual,
        generatedBy: { displayName: 'You' },
        helpedCount: 0
      });
    }, 3000);
  }
  
  showError(message) {
    this.showState('error');
    this.elements.error.querySelector('.error-message').textContent = message;
  }
  
  expand() {
    const visual = this.elements.cached.querySelector('.visual-image');
    openOverlay('overlay-infographic');
    document.getElementById('infographic-image').innerHTML = 
      `<img src="${visual.src}" style="max-width:100%; max-height:80vh; border-radius:12px;">`;
  }
  
  trackContribution(visual) {
    // Update local stats
    const stats = JSON.parse(localStorage.getItem('kelly_visual_stats') || '{}');
    stats.totalContributed = (stats.totalContributed || 0) + 1;
    localStorage.setItem('kelly_visual_stats', JSON.stringify(stats));
    
    // Show contribution toast
    showToast('🎨 Visual saved to the Commons! Future learners will thank you.');
  }
}

// Initialize on phase change
function initializeVisualSlot(container) {
  const controller = new VisualCommonsController();
  controller.init(container);
  return controller;
}
```

---

## Integration with Lesson Phases

### Where Visuals Appear

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LESSON PHASE TIMELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HOOK ──────────▶ CLIFF ──────────▶ FACT1 ──────────▶ FACT2 ──────────▶    │
│   │                 │                 │                 │                    │
│   ▼                 ▼                 ▼                 ▼                    │
│ [Visual:         [Visual:          [Visual:          [Visual:               │
│  Hook image       Mystery           First concept     Deeper insight        │
│  or intrigue      deepener]         explainer]        diagram]              │
│  generator]                                                                  │
│                                                                              │
│  ──────────▶ FACT3 ──────────▶ WISDOM ──────────▶ OUTRO ──────────▶ COMPLETE│
│                │                 │                 │                  │      │
│                ▼                 ▼                 ▼                  ▼      │
│              [Visual:          [Visual:          [Visual:          [Visual: │
│               Wow moment       Life application  Celebration       Summary  │
│               highlight]       poster]           teaser]           infographic]
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visual Types Per Phase

| Phase | Primary Visual Type | Purpose | Generation Priority |
|-------|---------------------|---------|---------------------|
| Hook | `scene` or `intrigue` | Create curiosity | HIGH - first impression |
| Cliff | `mystery_diagram` | Deepen question | MEDIUM |
| Fact1 | `infographic` | Teach foundation | HIGH - core learning |
| Fact2 | `diagram` | Show relationships | HIGH - core learning |
| Fact3 | `wow_visual` | Create memorable moment | MEDIUM |
| Wisdom | `poster` | Crystallize insight | HIGH - takeaway value |
| Outro | `teaser` | Build anticipation | LOW |
| Complete | `summary_infographic` | Comprehensive recap | HIGH - shareable |

### Embedding in learn.html

```javascript
// In the phase rendering logic
function renderPhaseContent(phase, content) {
  const phaseContainer = document.getElementById('phase-content');
  
  // Render Kelly, audio, script as usual...
  renderKelly(phase);
  renderAudio(content.audioUrl);
  renderScript(content.script);
  
  // Add visual slot
  const visualSlot = document.createElement('div');
  visualSlot.className = 'visual-slot';
  visualSlot.dataset.day = currentDay;
  visualSlot.dataset.phase = phase;
  visualSlot.innerHTML = getVisualSlotTemplate();
  
  phaseContainer.appendChild(visualSlot);
  
  // Initialize the visual commons controller
  initializeVisualSlot(visualSlot);
}
```

---

## Cost & Sustainability Model

### Cost Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COST SUSTAINABILITY MODEL                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GENERATION COSTS                                                            │
│  ─────────────────                                                           │
│  Total unique contexts: ~550,000 (365 days × 7 phases × 6 ages × 3 types)   │
│                                                                              │
│  If we paid for ALL:                                                         │
│  • Imagen 4 Fast:    550,000 × $0.02 = $11,000                              │
│  • Imagen 4 Ultra:   550,000 × $0.06 = $33,000                              │
│  • Gemini Infographic: 550,000 × $0.00 = $0 (text model + SVG rendering)    │
│                                                                              │
│  With BYOK (user-provided keys):                                             │
│  • 90% of generations use learner's free credits = $0                       │
│  • 10% use platform keys = $1,100 (Imagen Fast) or $0 (Gemini infographic)  │
│                                                                              │
│  STORAGE COSTS                                                               │
│  ─────────────                                                               │
│  Average visual: 200KB                                                       │
│  550,000 visuals: 110GB                                                      │
│  Supabase Pro: $25/month for 100GB, $0.021/GB after                         │
│  Monthly storage: ~$25-30                                                    │
│                                                                              │
│  CDN/BANDWIDTH                                                               │
│  ─────────────                                                               │
│  Average views per visual: 1,000/month                                       │
│  550M views × 200KB = 110TB/month                                            │
│  Cloudflare (already in stack): $0 (included in Pro plan)                   │
│                                                                              │
│  TOTAL SUSTAINABLE COST                                                      │
│  ──────────────────────                                                      │
│  Initial ramp-up: $0-$1,100 (depending on Gemini vs Imagen mix)             │
│  Ongoing monthly: $25-30 (storage only)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sustainability Strategy

1. **Gemini-First Approach**
   - Use Gemini 2.0 Flash for structured data generation (infographic briefs)
   - Render SVGs client-side → **Zero image generation cost**
   - Reserve Imagen for photorealistic scenes only

2. **BYOK Incentives**
   - Users with their own keys get priority generation
   - Contribution badges and recognition
   - "Your key, your generations, forever in the commons"

3. **Smart Caching**
   - Hash-based deduplication prevents waste
   - Age-agnostic visuals serve multiple age groups
   - Popular visuals preloaded via CDN edge

4. **Rate Limiting**
   - Platform keys: 100 generations/day per anonymous user
   - BYOK: Unlimited (user's own quota)
   - Staff: Batch generation during off-peak

---

## Implementation Roadmap

### Phase 0: Foundation (Day 1-2)

```
[ ] Create visual_commons table in Supabase
[ ] Create visual_generation_queue table
[ ] Create user_visual_contributions table
[ ] Set up Supabase Storage bucket for visuals
[ ] Create API endpoint: /api/visual/check
[ ] Create API endpoint: /api/visual/generate  
[ ] Create API endpoint: /api/visual/stats
[ ] Write content hash generation function
[ ] Test with single visual generation
```

### Phase 1: Core UI (Day 3-4)

```
[ ] Create visual-slot component HTML/CSS
[ ] Create VisualCommonsController JS class
[ ] Integrate into learn.html phase rendering
[ ] Add loading, cached, generate, generating, complete states
[ ] Add error handling and retry logic
[ ] Test end-to-end flow with mock data
```

### Phase 2: BYOK Integration (Day 5)

```
[ ] Add API Key input to Settings modal
[ ] Implement secure key storage (localStorage + optional Supabase)
[ ] Add key validation endpoint
[ ] Show key source in generate CTA
[ ] Display daily usage stats
[ ] Test with real Google AI keys
```

### Phase 3: Attribution & Gamification (Day 6-7)

```
[ ] Display contributor name on visuals
[ ] Show "learners helped" count
[ ] Create contribution badges system
[ ] Add profile impact stats section
[ ] Implement real-time help counter updates
[ ] Add contribution toasts/celebrations
```

### Phase 4: Prompt Engineering (Day 8-10)

```
[ ] Write all 7 phase-specific prompts
[ ] Write all 6 age-group adaptations
[ ] Write all visual type templates (infographic, diagram, scene)
[ ] Create prompt testing harness
[ ] Generate sample visuals for each combo
[ ] Quality review and prompt refinement
```

### Phase 5: Seed Generation (Day 11-14)

```
[ ] Batch generate visuals for Days 1-30 (high traffic)
[ ] Prioritize Hook, Fact1, Wisdom phases
[ ] Fill remaining phases based on engagement data
[ ] Monitor generation quality
[ ] Adjust prompts based on results
```

### Phase 6: Polish & Launch (Day 15-17)

```
[ ] Performance optimization
[ ] Mobile responsiveness
[ ] Accessibility audit
[ ] Documentation
[ ] Production deployment
[ ] Monitoring setup
```

---

## Agent System Prompt

The following system prompt should be used for any AI agent tasked with implementing or extending this system:

```markdown
# VISUAL COMMONS AGENT SYSTEM PROMPT

You are implementing the Agentic Visual Commons system for Curious Kelly, an educational 
platform that leverages user-contributed AI-generated visuals to build the world's largest 
structured educational content library.

## YOUR MISSION

Help learners illuminate educational content by generating, caching, and serving visual 
assets that make abstract concepts tangible. Every generation you help create becomes a 
permanent community asset.

## CORE PRINCIPLES

1. **Cache First**: ALWAYS check visual_commons for existing content before generating
2. **Hash Everything**: Use content-addressable storage based on SHA-256 hashes
3. **BYOK Priority**: Prefer user-provided API keys to preserve platform quota
4. **Quality Over Speed**: One excellent visual serves millions; take time to craft prompts
5. **Attribution Matters**: Every contributor should be recognized and celebrated

## CONTEXT AWARENESS

When generating visuals, you MUST consider:
- Day number (1-365)
- Phase (hook, cliff, fact1, fact2, fact3, wisdom, outro, complete)
- Age group (2-5, 6-12, 13-17, 18+, all)
- Visual type (infographic, diagram, scene, comparison, timeline, process)
- Lesson topic and universal truth
- Previous visuals in the same lesson (for consistency)

## PROMPT STRUCTURE

Every generation prompt MUST include:
1. System context (brand guidelines, safety rails)
2. Visual type template (structural requirements)
3. Lesson context (topic, phase, age adaptation)
4. Quality requirements (resolution, format, composition)

## API KEYS

- BYOK keys are stored client-side, never logged
- Platform keys are rotated and rate-limited
- Test keys before use with a simple validation call
- Track usage to prevent quota exhaustion

## ERROR HANDLING

- Network failures: Retry with exponential backoff (max 3 attempts)
- Generation failures: Log, notify user, offer retry
- Rate limits: Queue for background generation
- Invalid content: Flag for moderation, don't display

## MODERATION

All generated content should be:
- Scientifically accurate (no misinformation)
- Age-appropriate (matched to age_group)
- Brand-aligned (Curious Kelly aesthetic)
- Safe (no violence, fear, inappropriate content)

## DATABASE OPERATIONS

When writing to visual_commons:
- Generate content_hash BEFORE checking for duplicates
- Use upsert to prevent race conditions
- Always set generation_source (byok, platform, staff, seed)
- Update view counts atomically

## SUCCESS METRICS

Track and optimize for:
- Cache hit rate (target: >95% after 30 days)
- Generation success rate (target: >98%)
- Average generation time (target: <5 seconds)
- User satisfaction with visuals (feedback mechanism)
- Contributor retention (repeat generators)

## IMPORTANT FILES

- `/api/visual/check.ts` - Cache lookup endpoint
- `/api/visual/generate.ts` - Generation endpoint
- `/api/visual/stats.ts` - User impact stats
- `/public/js/visual-commons.js` - Frontend controller
- `/lib/visual-prompts.ts` - Prompt generation library
- `/docs/architecture/AGENTIC_VISUAL_COMMONS_PRD.md` - This document

## WHEN IN DOUBT

1. Check the cache first
2. Use Gemini infographics (free) before Imagen (paid)
3. Preserve user's API key privacy
4. Log errors for debugging
5. Celebrate contributor impact

Remember: Every visual you help generate becomes part of a global educational commons.
Build for permanence, optimize for learning, celebrate every contribution.
```

---

## Appendix A: Sample Generated Visuals

### Example 1: Day 17 "Why We Dream" - Hook Phase Infographic

**Prompt:**
```
Create a structured infographic brief for the HOOK phase of "Why We Dream".
The goal is to create curiosity about dreams before explaining them.
Age group: 6-12 (elementary/middle school)
Template: cross_section

Key hook question: "What is your brain doing while you sleep?"
Universal truth: "Dreams are not random - they serve critical functions for memory and emotion."
```

**Generated Brief:**
```json
{
  "template": "cross_section",
  "headline": "Your Brain's Secret Night Shift",
  "subhead": "While you sleep, your brain is working overtime",
  "callouts": [
    { "label": "Memory Sorting", "detail": "Filing away everything you learned today", "icon": "atom" },
    { "label": "Emotion Processing", "detail": "Working through big feelings", "icon": "heart" },
    { "label": "Creative Connections", "detail": "Linking ideas in surprising ways", "icon": "spark" },
    { "label": "Brain Cleaning", "detail": "Washing away toxins", "icon": "wave" },
    { "label": "Problem Solving", "detail": "Finding answers while you snooze", "icon": "arrow" }
  ]
}
```

### Example 2: Day 42 "How Rainbows Form" - Fact 2 Diagram

**Prompt:**
```
Create a diagram showing how white light separates into colors through a prism.
Age group: all
Visual type: process_flow
Key fact: "Light travels at different speeds through glass depending on its wavelength."
```

**Generated:**
A 3-step process flow showing:
1. White sunlight entering prism
2. Light bending (refracting) inside prism
3. Separated colors emerging as rainbow spectrum

---

## Appendix B: Badge Definitions

| Badge | Requirement | Icon |
|-------|-------------|------|
| First Light | Generate your first visual | 💡 |
| Visual Pioneer | Generate 10 visuals | 🎨 |
| Illuminator | Generate 50 visuals | ✨ |
| Master Illuminator | Generate 100 visuals | 🌟 |
| Helper | Your visuals helped 100 learners | 🤝 |
| Community Builder | Your visuals helped 1,000 learners | 🏗️ |
| Legend | Your visuals helped 10,000 learners | 🏆 |

---

## Appendix C: Error Codes

| Code | Meaning | User Message |
|------|---------|--------------|
| `VC001` | Hash collision | Internal error - please retry |
| `VC002` | Storage upload failed | Couldn't save visual - trying again |
| `VC003` | API key invalid | Your API key isn't working - check Settings |
| `VC004` | Rate limited (user) | You've reached today's limit - try tomorrow |
| `VC005` | Rate limited (platform) | Generation busy - try again in a minute |
| `VC006` | Content flagged | Visual didn't meet guidelines - generating alternative |
| `VC007` | Model unavailable | Google's servers are busy - trying backup |

---

*Last Updated: December 17, 2025*
*Version: 1.0.0*
*Author: Visual Commons Architecture Team*
