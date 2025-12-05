# Kelly Image Generation Architecture
## The Soul of Curious Kelly - A System Built to Last Forever

**Created:** December 2, 2025  
**Status:** Architectural Specification  
**Priority:** CRITICAL - This defines Kelly's visual identity for millions of learners

---

## 🎯 Vision

Every learner, every day, for the rest of their lives, will see Kelly - not as a static avatar, but as a living, responsive teacher who:
- Looks at them with genuine curiosity
- Reacts to their choices with authentic emotion
- Teaches each lesson with topic-appropriate visuals
- Maintains perfect visual consistency across all 365+ lessons

**This is not just an image system. This is Kelly's soul.**

---

## 📋 Table of Contents

1. [The Kelly Character Bible](#1-the-kelly-character-bible)
2. [Image Types Taxonomy](#2-image-types-taxonomy)
3. [Phase 2: Per-Lesson Image Structure](#3-phase-2-per-lesson-image-structure)
4. [Phase 3: Supabase Storage Architecture](#4-phase-3-supabase-storage-architecture)
5. [Phase 4: AI Generation Pipeline](#5-phase-4-ai-generation-pipeline)
6. [Prompt Engineering System](#6-prompt-engineering-system)
7. [Quality Control & Approval Workflow](#7-quality-control--approval-workflow)
8. [Client Integration](#8-client-integration)
9. [Cost Analysis & Optimization](#9-cost-analysis--optimization)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. The Kelly Character Bible

### 1.1 Visual Identity (IMMUTABLE)

```yaml
Kelly:
  age_appearance: "Late 20s to early 30s"
  ethnicity: "Mediterranean/Mixed - warm olive skin tone"
  
  face:
    shape: "Oval with soft features"
    eyes: "Warm brown, expressive, slight smile in eyes"
    eyebrows: "Natural, well-groomed, expressive"
    nose: "Straight, proportional"
    lips: "Natural pink, often in warm smile"
    expression_default: "Curious, warm, approachable"
  
  hair:
    color: "Medium to light brown with subtle caramel highlights"
    style: "Long, soft waves, past shoulders"
    texture: "Healthy, natural movement"
    parting: "Slightly off-center"
  
  body:
    build: "Healthy, average, relatable"
    posture: "Open, confident but not intimidating"
    height: "Average (appears 5'6\" to 5'8\")"
  
  clothing:
    primary: "Light blue crewneck sweater"
    style: "Casual professional, approachable"
    fit: "Comfortable, not tight"
    alternatives:
      - "Light blue button-down (formal lessons)"
      - "Light blue cardigan (cozy lessons)"
    never:
      - "Logos or text on clothing"
      - "Busy patterns"
      - "Dark or harsh colors"
  
  setting:
    primary: "Director's chair in bright studio"
    background: "Clean white/light gray with soft shadows"
    lighting: "Soft natural light from camera-right"
    mood: "Professional yet warm, like a favorite teacher's office"
```

### 1.2 Personality Through Expression

```yaml
Expressions:
  curious:
    eyes: "Wide, engaged, slightly raised eyebrows"
    mouth: "Slight open-mouth smile or pursed thoughtfully"
    head: "Slight tilt (5-10 degrees)"
    use: "Default, questions, exploring ideas"
  
  warm_welcome:
    eyes: "Crinkled at corners, direct eye contact"
    mouth: "Full genuine smile showing teeth"
    head: "Straight, facing camera"
    body: "Open posture, hand gesture welcoming"
    use: "Lesson start, greetings"
  
  thinking:
    eyes: "Focused, looking slightly up or to side"
    mouth: "Closed, perhaps slight pout"
    head: "Tilt with hand near chin"
    body: "Thoughtful pose"
    use: "During questions, pondering"
  
  excited:
    eyes: "Bright, wide, sparkling"
    mouth: "Big smile, enthusiasm evident"
    head: "Slight forward lean"
    body: "Animated, gesturing"
    use: "Hook reveal, discoveries"
  
  proud:
    eyes: "Warm, admiring"
    mouth: "Gentle smile of approval"
    head: "Slight nod position"
    body: "Relaxed, pleased"
    use: "Correct answers, completion"
  
  encouraging:
    eyes: "Soft, understanding"
    mouth: "Gentle reassuring smile"
    head: "Slight empathetic tilt"
    body: "Leaning slightly toward learner"
    use: "Wrong answers, support"
  
  explaining:
    eyes: "Focused, engaging"
    mouth: "Speaking or slight open"
    head: "Dynamic, following gesture"
    body: "Hand gestures illustrating points"
    use: "Teaching moments, wisdom"
```

### 1.3 The Director's Chair

```yaml
Director_Chair:
  material: "Classic wood frame with black canvas"
  style: "Vintage Hollywood director's chair"
  position: "Slightly angled (15-20 degrees) to camera"
  symbolism: "Kelly as director of learning journey"
  
  visibility:
    full_body: "Chair back, armrests, footrest visible"
    seated: "Armrests frame Kelly comfortably"
    standing: "Chair visible in background or to side"
  
  never:
    - "Modern office chair"
    - "Couch or casual seating"
    - "Standing without chair context"
```

---

## 2. Image Types Taxonomy

### 2.1 Base Poses (11 Universal States)

These work for ANY lesson - topic-agnostic:

| ID | Name | Description | File Pattern |
|----|------|-------------|--------------|
| `welcome` | Warm Welcome | Standing or seated, welcoming gesture | `kelly_welcome.png` |
| `thinking` | Deep Thought | Hand on chin, pondering | `kelly_thinking.png` |
| `explaining` | Teaching | Animated gesture while speaking | `kelly_explaining.png` |
| `listening` | Attentive | Leaning in, focused on learner | `kelly_listening.png` |
| `excited` | Discovery | Big smile, animated enthusiasm | `kelly_excited.png` |
| `celebrating` | Success | Proud, celebratory | `kelly_celebrating.png` |
| `encouraging` | Support | Gentle, reassuring | `kelly_encouraging.png` |
| `curious` | Wonder | Raised eyebrow, intrigued | `kelly_curious.png` |
| `pointing_left` | Option A | Pointing to left choice | `kelly_pointing_left.png` |
| `pointing_right` | Option B | Pointing to right choice | `kelly_pointing_right.png` |
| `waving` | Goodbye | Friendly wave, see you tomorrow | `kelly_waving.png` |

### 2.2 Per-Lesson Images (5 Per Day)

Each of 365 lessons gets custom images:

| Type | Purpose | Example for "Rules Everyone Agrees On" |
|------|---------|----------------------------------------|
| `hero` | Main lesson thumbnail | Kelly holding a scroll with rules |
| `intro` | Welcome to today's topic | Kelly in front of a chalkboard with "RULES" |
| `teaching` | Explaining concepts | Kelly gesturing at a visual of social contract |
| `reaction` | Responding to choices | Kelly reacting to learner's perspective |
| `wisdom` | Final insight | Kelly with contemplative, wise expression |

### 2.3 Prop Integration

Per-topic visual elements Kelly interacts with:

```yaml
Prop_Categories:
  science:
    - "Laboratory equipment"
    - "Magnifying glass"
    - "Periodic table"
    - "DNA helix model"
  
  philosophy:
    - "Open book"
    - "Scroll"
    - "Balance scales"
    - "Globe"
  
  creativity:
    - "Paint palette"
    - "Musical notes"
    - "Light bulb"
    - "Colorful shapes"
  
  nature:
    - "Plant or leaf"
    - "Earth globe"
    - "Animal silhouettes"
    - "Water droplet"
  
  emotion:
    - "Heart"
    - "Mirror"
    - "Masks (happy/sad)"
    - "Hands clasped"
```

---

## 3. Phase 2: Per-Lesson Image Structure

### 3.1 File System Organization

```
public/kelly/lessons/
├── 001/
│   ├── hero.png           # Thumbnail for day 1
│   ├── intro.png          # Welcome pose with topic context
│   ├── q1.png             # Question 1 pose
│   ├── q2.png             # Question 2 pose
│   ├── q3.png             # Question 3 pose
│   ├── hook.png           # Hook/reveal pose
│   ├── wisdom.png         # Final wisdom pose
│   ├── reaction_a.png     # Reaction to choice A
│   ├── reaction_b.png     # Reaction to choice B
│   ├── prop.png           # Topic-specific prop
│   └── metadata.json      # Generation prompts & settings
├── 002/
│   └── ...
└── 365/
    └── ...
```

### 3.2 Metadata Schema Per Lesson

```json
{
  "lesson_day": 336,
  "topic": "Rules Everyone Agrees On",
  "category": "philosophy",
  "generated_at": "2025-12-02T10:00:00Z",
  "generator": "flux-1.1-pro",
  "images": {
    "hero": {
      "prompt": "...",
      "seed": 12345,
      "approved": true,
      "approved_by": "system",
      "file_size_kb": 245,
      "dimensions": "1024x1024"
    },
    "intro": { ... },
    "q1": { ... }
  },
  "character_reference_id": "kelly-v2-base",
  "quality_score": 0.95
}
```

### 3.3 Fallback Chain

```javascript
// Client loading logic
async function getKellyImage(lessonDay, imageType) {
  // 1. Try lesson-specific image
  const lessonImage = `/kelly/lessons/${String(lessonDay).padStart(3, '0')}/${imageType}.png`;
  if (await imageExists(lessonImage)) return lessonImage;
  
  // 2. Fall back to base pose
  const basePose = IMAGE_TYPE_TO_POSE[imageType]; // e.g., 'q1' -> 'thinking'
  const baseImage = `/kelly/poses/kelly_${basePose}.png`;
  if (await imageExists(baseImage)) return baseImage;
  
  // 3. Ultimate fallback
  return '/kelly/poses/kelly_welcome.png';
}
```

---

## 4. Phase 3: Supabase Storage Architecture

### 4.1 Database Schema

```sql
-- ═══════════════════════════════════════════════════════════════
-- KELLY IMAGE GENERATION SYSTEM
-- ═══════════════════════════════════════════════════════════════

-- Character reference library (immutable base images for consistency)
CREATE TABLE kelly_character_references (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  version TEXT NOT NULL,                    -- 'v2-base', 'v2.1-refined'
  description TEXT,
  reference_images JSONB NOT NULL,          -- Array of reference image URLs
  face_embedding BYTEA,                     -- Optional: face recognition embedding
  style_prompt TEXT NOT NULL,               -- Base prompt for this version
  negative_prompt TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  is_active BOOLEAN DEFAULT true
);

-- Master image catalog
CREATE TABLE kelly_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Identity
  image_type TEXT NOT NULL,                 -- 'base_pose', 'lesson_specific', 'reaction'
  state TEXT NOT NULL,                      -- 'thinking', 'excited', 'hero', etc.
  
  -- Lesson context (NULL for base poses)
  lesson_day INT,
  lesson_topic TEXT,
  lesson_category TEXT,
  
  -- Storage
  storage_path TEXT NOT NULL,               -- Supabase Storage path
  public_url TEXT NOT NULL,                 -- CDN URL
  thumbnail_url TEXT,                       -- 256px thumbnail
  
  -- Generation metadata
  character_ref_id UUID REFERENCES kelly_character_references(id),
  full_prompt TEXT NOT NULL,
  negative_prompt TEXT,
  seed BIGINT,
  generator TEXT NOT NULL,                  -- 'flux-1.1-pro', 'dall-e-3', etc.
  model_version TEXT,
  generation_params JSONB,                  -- steps, guidance, etc.
  
  -- Quality
  quality_score DECIMAL(3,2),               -- 0.00 to 1.00
  consistency_score DECIMAL(3,2),           -- How well it matches character ref
  is_approved BOOLEAN DEFAULT false,
  approved_by TEXT,
  approved_at TIMESTAMP,
  rejection_reason TEXT,
  
  -- Technical
  width INT NOT NULL,
  height INT NOT NULL,
  file_size_bytes BIGINT,
  format TEXT DEFAULT 'png',
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX idx_kelly_images_lesson ON kelly_images(lesson_day, image_type);
CREATE INDEX idx_kelly_images_state ON kelly_images(state, is_approved);
CREATE INDEX idx_kelly_images_type ON kelly_images(image_type);

-- Generation job queue
CREATE TABLE kelly_generation_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Job specification
  job_type TEXT NOT NULL,                   -- 'lesson_batch', 'single', 'regenerate'
  lesson_day INT,
  image_types TEXT[],                       -- ['hero', 'intro', 'q1', ...]
  priority INT DEFAULT 5,                   -- 1 = highest
  
  -- Prompts
  character_ref_id UUID REFERENCES kelly_character_references(id),
  prompt_template TEXT NOT NULL,
  prompt_variables JSONB,
  
  -- Status
  status TEXT DEFAULT 'pending',            -- pending, processing, completed, failed
  progress DECIMAL(5,2) DEFAULT 0,          -- 0.00 to 100.00
  error_message TEXT,
  
  -- Results
  generated_image_ids UUID[],
  
  -- Timing
  created_at TIMESTAMP DEFAULT NOW(),
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  
  -- Retry logic
  attempt_count INT DEFAULT 0,
  max_attempts INT DEFAULT 3,
  next_retry_at TIMESTAMP
);

-- Prompt template library
CREATE TABLE kelly_prompt_templates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  name TEXT NOT NULL UNIQUE,                -- 'lesson_hero', 'q_phase_thinking'
  description TEXT,
  
  -- Template with {{variables}}
  prompt_template TEXT NOT NULL,
  required_variables TEXT[],                -- ['topic', 'category', 'emotion']
  
  -- Defaults
  default_negative_prompt TEXT,
  default_params JSONB,
  
  -- Versioning
  version INT DEFAULT 1,
  is_active BOOLEAN DEFAULT true,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking for cost analysis
CREATE TABLE kelly_generation_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID REFERENCES kelly_generation_jobs(id),
  
  generator TEXT NOT NULL,
  model TEXT,
  tokens_used INT,
  compute_seconds DECIMAL(10,2),
  estimated_cost_usd DECIMAL(10,4),
  
  created_at TIMESTAMP DEFAULT NOW()
);
```

### 4.2 Supabase Storage Buckets

```javascript
// Storage bucket configuration
const STORAGE_BUCKETS = {
  // Public bucket for serving images
  'kelly-images': {
    public: true,
    allowedMimeTypes: ['image/png', 'image/webp', 'image/jpeg'],
    fileSizeLimit: 5 * 1024 * 1024, // 5MB
  },
  
  // Private bucket for processing/staging
  'kelly-staging': {
    public: false,
    allowedMimeTypes: ['image/png'],
    fileSizeLimit: 10 * 1024 * 1024, // 10MB for high-res
  },
  
  // Character references (private, high value)
  'kelly-references': {
    public: false,
    allowedMimeTypes: ['image/png', 'image/jpeg'],
  }
};

// Path structure in kelly-images bucket
const PATH_STRUCTURE = {
  base_poses: 'poses/{state}.png',                    // poses/thinking.png
  lesson_images: 'lessons/{day}/{type}.png',          // lessons/336/hero.png
  thumbnails: 'thumbnails/lessons/{day}/{type}.webp', // thumbnails/lessons/336/hero.webp
  reactions: 'lessons/{day}/reactions/{choice}.png',  // lessons/336/reactions/a.png
};
```

### 4.3 Storage Functions

```sql
-- Function to get Kelly image with fallback
CREATE OR REPLACE FUNCTION get_kelly_image(
  p_lesson_day INT,
  p_image_type TEXT,
  p_fallback_state TEXT DEFAULT 'welcome'
)
RETURNS TABLE (
  image_url TEXT,
  is_lesson_specific BOOLEAN,
  image_id UUID
) AS $$
BEGIN
  -- Try lesson-specific first
  RETURN QUERY
  SELECT 
    ki.public_url,
    true,
    ki.id
  FROM kelly_images ki
  WHERE ki.lesson_day = p_lesson_day
    AND ki.image_type = 'lesson_specific'
    AND ki.state = p_image_type
    AND ki.is_approved = true
  LIMIT 1;
  
  IF NOT FOUND THEN
    -- Fall back to base pose
    RETURN QUERY
    SELECT 
      ki.public_url,
      false,
      ki.id
    FROM kelly_images ki
    WHERE ki.image_type = 'base_pose'
      AND ki.state = p_fallback_state
      AND ki.is_approved = true
    ORDER BY ki.created_at DESC
    LIMIT 1;
  END IF;
END;
$$ LANGUAGE plpgsql;
```

---

## 5. Phase 4: AI Generation Pipeline

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KELLY IMAGE GENERATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   TRIGGER    │ → │   PROMPT     │ → │   GENERATION          │  │
│  │              │    │   ENGINE     │    │   SERVICE             │  │
│  │ • Batch job  │    │              │    │                       │  │
│  │ • New lesson │    │ • Template   │    │ • Flux API           │  │
│  │ • Manual     │    │ • Variables  │    │ • DALL-E 3           │  │
│  │ • Regen req  │    │ • Character  │    │ • Stable Diffusion   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                    │                 │
│                                                    ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   DELIVERY   │ ← │   APPROVAL   │ ← │   POST-PROCESSING    │  │
│  │              │    │              │    │                       │  │
│  │ • CDN cache  │    │ • Auto QC    │    │ • Resize             │  │
│  │ • Client SDK │    │ • Human rev  │    │ • Optimize           │  │
│  │ • Preload    │    │ • A/B test   │    │ • Thumbnails         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Generation Service Interface

```typescript
// interfaces/kelly-image-generator.ts

interface GenerationRequest {
  // What to generate
  lessonDay: number;
  imageTypes: ImageType[];
  
  // Character consistency
  characterRefId: string;
  
  // Context for prompts
  lessonContext: {
    topic: string;
    category: LessonCategory;
    universalTruth: string;
    keyTerms: string[];
  };
  
  // Quality settings
  quality: 'draft' | 'standard' | 'premium';
  
  // Options
  seed?: number;  // For reproducibility
  variations?: number;  // Generate N variations to choose from
}

interface GenerationResult {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  images: GeneratedImage[];
  usage: {
    tokensUsed: number;
    computeSeconds: number;
    estimatedCostUsd: number;
  };
}

interface GeneratedImage {
  id: string;
  imageType: ImageType;
  url: string;
  thumbnailUrl: string;
  prompt: string;
  seed: number;
  qualityScore: number;
  consistencyScore: number;
}

type ImageType = 
  | 'hero' 
  | 'intro' 
  | 'q1' | 'q2' | 'q3'
  | 'hook' 
  | 'wisdom'
  | 'reaction_correct'
  | 'reaction_incorrect';

type LessonCategory = 
  | 'science' 
  | 'philosophy' 
  | 'creativity'
  | 'nature' 
  | 'emotion' 
  | 'society'
  | 'health'
  | 'technology';
```

### 5.3 Multi-Provider Support

```typescript
// services/image-generators/index.ts

abstract class KellyImageGenerator {
  abstract name: string;
  abstract supportedFeatures: string[];
  
  abstract generate(
    prompt: string, 
    options: GeneratorOptions
  ): Promise<GeneratedImage>;
  
  abstract checkHealth(): Promise<boolean>;
  abstract getEstimatedCost(imageCount: number): number;
}

class FluxGenerator extends KellyImageGenerator {
  name = 'flux-1.1-pro';
  supportedFeatures = ['character-consistency', 'high-quality', 'fast'];
  
  async generate(prompt: string, options: GeneratorOptions) {
    // Flux API implementation
    // Best for: character consistency, high quality
    // Cost: ~$0.04 per image
  }
}

class DallE3Generator extends KellyImageGenerator {
  name = 'dall-e-3';
  supportedFeatures = ['natural-language', 'creative'];
  
  async generate(prompt: string, options: GeneratorOptions) {
    // OpenAI DALL-E 3 implementation
    // Best for: creative interpretations, natural prompts
    // Cost: ~$0.04-0.08 per image
  }
}

class StableDiffusionGenerator extends KellyImageGenerator {
  name = 'stable-diffusion-xl';
  supportedFeatures = ['fine-tunable', 'self-hosted', 'lora'];
  
  async generate(prompt: string, options: GeneratorOptions) {
    // Self-hosted or Replicate implementation
    // Best for: fine-tuned Kelly model, cost control
    // Cost: ~$0.01-0.02 per image (self-hosted)
  }
}

// Provider selection strategy
class GeneratorOrchestrator {
  private generators: Map<string, KellyImageGenerator>;
  
  async generate(request: GenerationRequest): Promise<GeneratedImage[]> {
    // Select best generator based on:
    // 1. Quality requirements
    // 2. Cost budget
    // 3. Provider health
    // 4. Feature needs (e.g., character consistency)
    
    const generator = this.selectOptimalGenerator(request);
    return generator.generate(
      this.buildPrompt(request),
      this.buildOptions(request)
    );
  }
}
```

### 5.4 Character Consistency System

```typescript
// services/character-consistency.ts

class KellyCharacterSystem {
  private baseCharacterRef: CharacterReference;
  
  /**
   * The secret to consistent Kelly:
   * 1. Use reference images with every generation
   * 2. Embed face features for similarity checking
   * 3. Maintain strict prompt structure
   * 4. Reject images that don't meet consistency threshold
   */
  
  async ensureConsistency(
    generatedImage: GeneratedImage
  ): Promise<ConsistencyResult> {
    // 1. Extract face features from generated image
    const generatedFeatures = await this.extractFaceFeatures(generatedImage);
    
    // 2. Compare to reference Kelly
    const similarity = this.compareFaceFeatures(
      generatedFeatures,
      this.baseCharacterRef.faceEmbedding
    );
    
    // 3. Check clothing/style consistency
    const styleMatch = await this.checkStyleConsistency(generatedImage);
    
    // 4. Overall score
    const consistencyScore = (similarity * 0.7) + (styleMatch * 0.3);
    
    return {
      score: consistencyScore,
      isAcceptable: consistencyScore >= 0.85,
      issues: this.identifyIssues(generatedImage),
    };
  }
  
  buildConsistentPrompt(basePrompt: string): string {
    // Always prepend character description
    return `${this.baseCharacterRef.stylePrompt}. ${basePrompt}`;
  }
}

// Character reference management
interface CharacterReference {
  id: string;
  version: string;
  referenceImages: string[];  // URLs to canonical Kelly images
  faceEmbedding: Float32Array;  // Face recognition embedding
  stylePrompt: string;  // Base prompt that defines Kelly
  negativePrompt: string;  // What to avoid
}
```

---

## 6. Prompt Engineering System

### 6.1 The Master Kelly Prompt

```typescript
// prompts/kelly-master.ts

const KELLY_MASTER_PROMPT = {
  // Core character description (NEVER CHANGE without versioning)
  character: `
    A warm, intelligent woman in her late 20s named Kelly.
    
    FACE: Oval face with soft features, warm brown expressive eyes with slight 
    smile lines, natural well-groomed eyebrows, straight proportional nose, 
    natural pink lips often in a genuine warm smile.
    
    HAIR: Medium to light brown with subtle caramel highlights, long soft waves 
    past shoulders, healthy natural movement, slightly off-center parting.
    
    SKIN: Warm olive Mediterranean complexion, healthy natural glow.
    
    BODY: Healthy average build, confident open posture, appears 5'6" to 5'8".
    
    CLOTHING: Wearing a comfortable light blue crewneck sweater, casual 
    professional style, approachable.
    
    SETTING: Seated in a vintage Hollywood director's chair with wood frame and 
    black canvas, in a bright clean studio with white/light gray background, 
    soft natural light from camera-right casting gentle shadows.
    
    STYLE: Professional photography, high quality, warm and inviting atmosphere,
    like a trusted teacher or mentor.
  `,
  
  // Negative prompt (what to avoid)
  negative: `
    cartoon, anime, illustration, painting, drawing, sketch, 
    3D render, CGI, plastic, doll-like, uncanny valley,
    harsh lighting, dark shadows, moody, cold colors,
    busy background, clutter, text, watermarks, logos,
    different clothing, different hair color, different eye color,
    different age, different ethnicity, masculine features,
    uncomfortable expression, forced smile, stiff posture
  `,
  
  // Quality modifiers
  quality: `
    professional photography, 8k resolution, sharp focus, 
    natural skin texture, authentic expression, candid feel,
    studio lighting, clean composition
  `
};
```

### 6.2 Prompt Templates

```typescript
// prompts/templates.ts

const PROMPT_TEMPLATES = {
  // Base pose templates
  base_thinking: {
    template: `
      {{character}}
      
      Kelly has a thoughtful expression, hand resting gently on her chin, 
      eyes looking slightly upward as if pondering a deep question.
      Her posture is relaxed but engaged, curious about what the learner
      will discover.
      
      {{quality}}
    `,
    variables: ['character', 'quality'],
  },
  
  base_excited: {
    template: `
      {{character}}
      
      Kelly's face lights up with genuine excitement and discovery.
      Her eyes are bright and wide, a big authentic smile showing her
      enthusiasm. She leans slightly forward, hands animated as if
      she's about to share something wonderful.
      
      {{quality}}
    `,
    variables: ['character', 'quality'],
  },
  
  // Lesson-specific templates
  lesson_hero: {
    template: `
      {{character}}
      
      Kelly is introducing today's lesson about "{{topic}}".
      She holds or gestures toward {{prop_description}}, which relates
      to the theme of {{category}}.
      
      Her expression is {{emotion}} - inviting the learner to explore
      this fascinating topic together.
      
      The prop is tastefully integrated, not dominating the frame,
      suggesting the topic without being literal.
      
      {{quality}}
    `,
    variables: [
      'character', 'topic', 'prop_description', 
      'category', 'emotion', 'quality'
    ],
  },
  
  lesson_question: {
    template: `
      {{character}}
      
      Kelly is presenting question {{question_number}} about "{{topic}}".
      
      She has a curious, encouraging expression - the kind that says
      "I believe in you, take your time to think about this."
      
      Her posture is open and patient, {{gesture_description}}.
      
      {{quality}}
    `,
    variables: [
      'character', 'question_number', 'topic',
      'gesture_description', 'quality'
    ],
  },
  
  lesson_wisdom: {
    template: `
      {{character}}
      
      Kelly shares the final wisdom for today's lesson on "{{topic}}":
      "{{wisdom_text}}"
      
      Her expression is {{wisdom_emotion}} - she's sharing something
      meaningful that she hopes will stay with the learner.
      
      Her posture suggests she's delivering an important message,
      perhaps hands gently clasped or gesturing for emphasis.
      
      {{quality}}
    `,
    variables: [
      'character', 'topic', 'wisdom_text',
      'wisdom_emotion', 'quality'
    ],
  },
  
  reaction_correct: {
    template: `
      {{character}}
      
      Kelly reacts to the learner choosing correctly.
      
      Her face shows genuine pride and delight - not exaggerated,
      but the authentic joy of a teacher watching a student succeed.
      
      She might be giving a subtle thumbs up, or her posture
      communicates "Yes! You've got it!"
      
      {{quality}}
    `,
    variables: ['character', 'quality'],
  },
  
  reaction_incorrect: {
    template: `
      {{character}}
      
      Kelly responds warmly to a learner who chose a different answer.
      
      Her expression is understanding and encouraging - no judgment,
      just gentle support. The kind of look that says "That's an
      interesting perspective, let's think about it together."
      
      She leans in slightly, maintaining warmth and connection.
      
      {{quality}}
    `,
    variables: ['character', 'quality'],
  },
};

// Prop descriptions by category
const PROP_LIBRARY = {
  science: [
    "a small magnifying glass, held thoughtfully",
    "a molecular model floating nearby (chemistry)",
    "a miniature telescope",
    "a beaker with colorful liquid",
  ],
  philosophy: [
    "an antique book with visible pages",
    "a small balance scale",
    "a glowing light bulb (ideas)",
    "a compass (direction/choices)",
  ],
  nature: [
    "a small potted plant",
    "a globe showing Earth",
    "a butterfly resting nearby",
    "a smooth river stone",
  ],
  // ... more categories
};
```

### 6.3 Prompt Builder

```typescript
// services/prompt-builder.ts

class KellyPromptBuilder {
  private templates: typeof PROMPT_TEMPLATES;
  private masterPrompt: typeof KELLY_MASTER_PROMPT;
  
  buildPrompt(
    templateName: string,
    variables: Record<string, string>
  ): { prompt: string; negativePrompt: string } {
    const template = this.templates[templateName];
    
    // Start with character
    let prompt = template.template
      .replace('{{character}}', this.masterPrompt.character)
      .replace('{{quality}}', this.masterPrompt.quality);
    
    // Fill in variables
    for (const [key, value] of Object.entries(variables)) {
      prompt = prompt.replace(new RegExp(`{{${key}}}`, 'g'), value);
    }
    
    // Validate all variables filled
    const unfilled = prompt.match(/{{[^}]+}}/g);
    if (unfilled) {
      throw new Error(`Unfilled variables: ${unfilled.join(', ')}`);
    }
    
    return {
      prompt: prompt.trim(),
      negativePrompt: this.masterPrompt.negative.trim(),
    };
  }
  
  buildLessonPrompts(lesson: LessonData): Map<ImageType, PromptPair> {
    const prompts = new Map();
    
    // Hero image
    prompts.set('hero', this.buildPrompt('lesson_hero', {
      topic: lesson.topic,
      prop_description: this.selectProp(lesson.category),
      category: lesson.category,
      emotion: 'curious and inviting',
    }));
    
    // Question phases
    for (let i = 1; i <= 3; i++) {
      prompts.set(`q${i}`, this.buildPrompt('lesson_question', {
        question_number: String(i),
        topic: lesson.topic,
        gesture_description: i === 1 
          ? 'leaning forward with interest'
          : i === 2 
            ? 'hand extended as if presenting options'
            : 'nodding encouragingly',
      }));
    }
    
    // Wisdom
    prompts.set('wisdom', this.buildPrompt('lesson_wisdom', {
      topic: lesson.topic,
      wisdom_text: lesson.universalTruth.substring(0, 100),
      wisdom_emotion: 'thoughtful yet warm',
    }));
    
    return prompts;
  }
  
  private selectProp(category: string): string {
    const props = PROP_LIBRARY[category] || PROP_LIBRARY['philosophy'];
    return props[Math.floor(Math.random() * props.length)];
  }
}
```

---

## 7. Quality Control & Approval Workflow

### 7.1 Automated Quality Checks

```typescript
// services/quality-control.ts

class KellyQualityControl {
  // Automated checks before human review
  async autoCheck(image: GeneratedImage): Promise<QualityResult> {
    const checks = await Promise.all([
      this.checkResolution(image),
      this.checkFacePresence(image),
      this.checkCharacterConsistency(image),
      this.checkLighting(image),
      this.checkBackground(image),
      this.checkExpression(image),
    ]);
    
    const overallScore = this.calculateOverallScore(checks);
    const issues = checks.filter(c => !c.passed);
    
    return {
      score: overallScore,
      autoApproved: overallScore >= 0.90 && issues.length === 0,
      requiresHumanReview: overallScore >= 0.70 && overallScore < 0.90,
      autoRejected: overallScore < 0.70,
      issues: issues.map(i => i.reason),
    };
  }
  
  private async checkCharacterConsistency(image: GeneratedImage): Promise<Check> {
    // Compare face embedding to reference
    const similarity = await this.faceMatcher.compare(
      image.url,
      this.referenceEmbedding
    );
    
    return {
      name: 'character_consistency',
      passed: similarity >= 0.85,
      score: similarity,
      reason: similarity < 0.85 
        ? `Face similarity ${(similarity * 100).toFixed(1)}% below threshold`
        : null,
    };
  }
  
  private async checkBackground(image: GeneratedImage): Promise<Check> {
    // Ensure background is clean studio, not busy
    const analysis = await this.imageAnalyzer.analyzeBackground(image.url);
    
    return {
      name: 'background',
      passed: analysis.isClean && analysis.brightness >= 0.7,
      score: analysis.cleanlinessScore,
      reason: !analysis.isClean 
        ? 'Background too busy or dark'
        : null,
    };
  }
}
```

### 7.2 Human Review Interface

```typescript
// components/KellyImageReview.tsx

interface ReviewQueue {
  images: PendingImage[];
  filters: {
    lessonDay: number | null;
    imageType: ImageType | null;
    status: 'pending' | 'flagged';
  };
}

// Review actions
type ReviewAction = 
  | { type: 'approve'; notes?: string }
  | { type: 'reject'; reason: string }
  | { type: 'regenerate'; promptModification: string }
  | { type: 'edit'; cropArea?: Rect; adjustments?: ImageAdjustments };

// Keyboard shortcuts for efficient review
const REVIEW_SHORTCUTS = {
  'a': 'approve',
  'r': 'reject',
  'n': 'next',
  'p': 'previous',
  'g': 'regenerate',
  '1-5': 'setQualityRating',
};
```

### 7.3 A/B Testing Framework

```typescript
// services/ab-testing.ts

class KellyImageABTest {
  /**
   * For important images (hero, intro), generate variations
   * and measure engagement to pick the best one.
   */
  
  async runTest(lessonDay: number, variants: GeneratedImage[]): Promise<void> {
    // Create test
    const test = await this.createTest({
      lessonDay,
      variants: variants.map(v => v.id),
      metric: 'lesson_completion_rate',
      minSampleSize: 1000,
      maxDuration: '7d',
    });
    
    // Client will serve random variant
    await this.activateTest(test.id);
  }
  
  async getWinningVariant(testId: string): Promise<GeneratedImage | null> {
    const results = await this.getTestResults(testId);
    
    if (results.hasSignificantWinner) {
      return results.winner;
    }
    
    return null;  // No clear winner yet
  }
}
```

---

## 8. Client Integration

### 8.1 Kelly Image SDK

```typescript
// sdk/kelly-images.ts

class KellyImageSDK {
  private supabase: SupabaseClient;
  private cache: Map<string, string>;
  private preloadQueue: string[];
  
  /**
   * Get the best available image for a lesson context
   */
  async getImage(
    lessonDay: number,
    imageType: ImageType,
    options: GetImageOptions = {}
  ): Promise<KellyImage> {
    const cacheKey = `${lessonDay}:${imageType}`;
    
    // Check local cache first
    if (this.cache.has(cacheKey)) {
      return { url: this.cache.get(cacheKey)!, source: 'cache' };
    }
    
    // Query Supabase with fallback
    const { data } = await this.supabase
      .rpc('get_kelly_image', {
        p_lesson_day: lessonDay,
        p_image_type: imageType,
        p_fallback_state: IMAGE_TYPE_FALLBACKS[imageType],
      })
      .single();
    
    if (data) {
      this.cache.set(cacheKey, data.image_url);
      return {
        url: data.image_url,
        source: data.is_lesson_specific ? 'lesson' : 'base',
        imageId: data.image_id,
      };
    }
    
    // Ultimate fallback to local files
    return {
      url: `/kelly/poses/kelly_${IMAGE_TYPE_FALLBACKS[imageType]}.png`,
      source: 'local_fallback',
    };
  }
  
  /**
   * Preload images for upcoming phases
   */
  async preloadForLesson(lessonDay: number): Promise<void> {
    const imageTypes: ImageType[] = [
      'hero', 'intro', 'q1', 'q2', 'q3', 'hook', 'wisdom',
      'reaction_correct', 'reaction_incorrect'
    ];
    
    const urls = await Promise.all(
      imageTypes.map(type => this.getImage(lessonDay, type))
    );
    
    // Preload all images into browser cache
    await Promise.all(
      urls.map(({ url }) => {
        const img = new Image();
        img.src = url;
        return new Promise(resolve => {
          img.onload = resolve;
          img.onerror = resolve;
        });
      })
    );
    
    console.log(`[KellyImages] Preloaded ${urls.length} images for day ${lessonDay}`);
  }
  
  /**
   * Change Kelly's current display state
   */
  async setKellyState(
    element: HTMLImageElement,
    lessonDay: number,
    state: ImageType,
    animate: boolean = true
  ): Promise<void> {
    const { url } = await this.getImage(lessonDay, state);
    
    if (animate) {
      element.style.opacity = '0';
      await new Promise(r => setTimeout(r, 150));
    }
    
    element.src = url;
    
    if (animate) {
      element.style.opacity = '1';
    }
  }
}

// Singleton for app
export const kellyImages = new KellyImageSDK();
```

### 8.2 React Hooks

```typescript
// hooks/useKellyImage.ts

function useKellyImage(
  lessonDay: number,
  imageType: ImageType
): { url: string; isLoading: boolean; source: string } {
  const [state, setState] = useState({
    url: '',
    isLoading: true,
    source: 'pending',
  });
  
  useEffect(() => {
    kellyImages.getImage(lessonDay, imageType)
      .then(result => setState({
        url: result.url,
        isLoading: false,
        source: result.source,
      }))
      .catch(() => setState({
        url: '/kelly/poses/kelly_welcome.png',
        isLoading: false,
        source: 'error_fallback',
      }));
  }, [lessonDay, imageType]);
  
  return state;
}

function useKellyPreload(lessonDay: number): void {
  useEffect(() => {
    kellyImages.preloadForLesson(lessonDay);
  }, [lessonDay]);
}
```

---

## 9. Cost Analysis & Optimization

### 9.1 Generation Costs

```yaml
Per_Image_Costs:
  flux_1_1_pro: $0.04
  dall_e_3_standard: $0.04
  dall_e_3_hd: $0.08
  stable_diffusion_api: $0.02
  stable_diffusion_self_hosted: $0.005
  midjourney: $0.10 (manual, but highest quality)

Images_Per_Lesson: 9
  - hero: 1
  - intro: 1
  - questions: 3
  - hook: 1
  - wisdom: 1
  - reactions: 2

Total_Lessons: 365

One_Time_Full_Generation:
  total_images: 3,285 (365 × 9)
  
  at_flux_rates:
    cost: $131.40
    
  at_self_hosted_sd:
    cost: $16.43
    
  with_3_variations_each:
    images: 9,855
    cost_flux: $394.20
    cost_self_hosted: $49.28
```

### 9.2 Optimization Strategies

```yaml
Cost_Optimization:
  1_smart_regeneration:
    description: "Only regenerate failed images, not entire lessons"
    savings: "~30% reduction"
    
  2_variation_on_demand:
    description: "Generate 1 image first, variations only if needed"
    savings: "~60% reduction in variation costs"
    
  3_base_pose_reuse:
    description: "Many lessons can share base thinking/excited poses"
    savings: "~20% for generic phases"
    
  4_batch_processing:
    description: "Generate during off-peak hours for lower API costs"
    savings: "~10-20% on some providers"
    
  5_caching_aggressive:
    description: "Once approved, images never regenerated"
    savings: "100% on repeat requests"
    
  6_hybrid_approach:
    description: "Use expensive generator for hero, cheaper for reactions"
    implementation:
      hero_intro: "flux (quality critical)"
      questions_wisdom: "stable-diffusion (good enough)"
      reactions: "stable-diffusion (simple variations)"
```

---

## 10. Implementation Roadmap

### Phase 2: Per-Lesson Images (Week 1-2)

```yaml
Week_1:
  Day_1_2:
    - Create directory structure for 365 lessons
    - Define metadata.json schema
    - Build image loading fallback system
    
  Day_3_4:
    - Update KellyImageSDK to check lesson dirs
    - Add preloading for lesson images
    - Test with manually created images for days 1-7
    
  Day_5:
    - Create generation manifest for all 365 lessons
    - Define prop library for all categories
    - Document prompt templates

Week_2:
  Day_1_3:
    - Generate test batch (days 1-10) using Flux
    - Evaluate quality, iterate prompts
    - Build quality scoring pipeline
    
  Day_4_5:
    - Run full 365-lesson generation
    - Human review of flagged images
    - Deploy to staging
```

### Phase 3: Supabase Storage (Week 3)

```yaml
Week_3:
  Day_1:
    - Create Supabase Storage buckets
    - Apply security policies
    - Create database tables
    
  Day_2_3:
    - Build upload pipeline (local → Supabase)
    - Generate and store all thumbnails
    - Update CDN URLs in database
    
  Day_4:
    - Update client SDK to use Supabase
    - Test fallback chain
    - Performance testing
    
  Day_5:
    - Deploy to production
    - Monitor performance
    - Document for team
```

### Phase 4: AI Generation Pipeline (Week 4-5)

```yaml
Week_4:
  Day_1_2:
    - Build GeneratorOrchestrator with Flux + SD
    - Implement prompt builder service
    - Create job queue system
    
  Day_3_4:
    - Build quality control pipeline
    - Implement character consistency checking
    - Create human review interface
    
  Day_5:
    - Integration testing
    - Cost tracking implementation

Week_5:
  Day_1_2:
    - A/B testing framework
    - Analytics for image performance
    - Admin dashboard for monitoring
    
  Day_3_4:
    - On-demand generation API
    - Custom lesson support
    - Rate limiting and quotas
    
  Day_5:
    - Production deployment
    - Documentation finalization
    - Training for content team
```

---

## 🎯 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Character consistency | > 90% similarity score | Face embedding comparison |
| Image quality | > 85% auto-approval rate | QC pipeline |
| Load time | < 200ms | P95 latency |
| Lesson completion | +5% vs baseline | A/B test |
| User satisfaction | > 4.5/5 | Survey |
| Generation cost | < $150/month ongoing | Usage tracking |

---

## 🔐 Security Considerations

1. **API Keys**: Never expose generation API keys client-side
2. **Storage**: Supabase Storage with proper RLS
3. **Rate Limiting**: Prevent abuse of on-demand generation
4. **Content Moderation**: All prompts pass through filter
5. **Audit Trail**: All generation jobs logged with prompts

---

*This system, built correctly, will give Kelly her soul. Every learner, every day, will see her react authentically to their journey. This is not just technology - it's the personality that will inspire millions to become lifelong learners.*





