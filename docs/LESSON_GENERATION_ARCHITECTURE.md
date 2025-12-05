# 🌍 CURIOUS KELLY: WORLD-CLASS LESSON GENERATION ARCHITECTURE

**Vision:** Every human on Earth deserves a personal teacher who knows them, adapts to them, and never gives up on them.

**Mission:** Build the infrastructure to generate unlimited, personalized, world-class educational content at any quality tier, for any learner, in any language, forever.

**Status:** ARCHITECTURE SPECIFICATION - Pre-Launch  
**Created:** December 5, 2025  
**Author:** Claude (Chief Academic Officer) + Nicky (Founder)

---

## 🎯 THE THREE TIERS OF LESSON GENERATION

### Tier 1: FREE - "Daily Bread" 
**Cost per lesson:** ~$5-20  
**Use case:** Standard daily lessons for all learners  
**Quality:** Professional, consistent, reliable  

| Asset Type | Generator | Cost | Notes |
|------------|-----------|------|-------|
| Text content | Claude 3.5 Sonnet | ~$0.50 | 60 atoms, all JSONB fields |
| Phase visuals (5) | Flux + Kelly LoRA | ~$0.20 | Topic-specific poses |
| Lesson images (5) | Flux + Kelly LoRA | ~$0.20 | Hero, bg, prop, reaction |
| Kelly voiceover | ElevenLabs | ~$2.00 | 5-10 min of audio |
| Thumbnails | Sharp/ImageMagick | ~$0.01 | Auto-generated |
| **Total** | | **~$3-5** | Per lesson core |

### Tier 2: PREMIUM - "Artisan Crafted"
**Cost per lesson:** ~$50-200  
**Use case:** Flagship lessons, special topics, seasonal content  
**Quality:** Cinematic, emotionally resonant, memorable  

| Asset Type | Generator | Cost | Notes |
|------------|-----------|------|-------|
| All Tier 1 assets | (above) | ~$5 | Base layer |
| Extended visuals (20+) | Midjourney v6 + DALL-E 3 | ~$10 | Multiple angles, expressions |
| Kelly video (lip-sync) | Hedra/HeyGen/OmniHuman | ~$20 | Full phase videos |
| Custom music | Suno AI | ~$5 | Topic-appropriate soundtrack |
| Human QA review | Internal | ~$20 | Expert educator review |
| A/B variants | (multiple gens) | ~$15 | 3 variants per key asset |
| **Total** | | **~$75-150** | Per lesson |

### Tier 3: GONE WILD - "Philanthropic Masterwork"
**Cost per lesson:** $1,000-10,000  
**Use case:** Landmark topics, sponsored lessons, curriculum flagships  
**Quality:** Documentary-quality, award-winning, timeless  

| Asset Type | Generator | Cost | Notes |
|------------|-----------|------|-------|
| All Tier 2 assets | (above) | ~$150 | Base layer |
| 3D Kelly animation | Custom Unreal/Unity | ~$500 | Full 3D character |
| Professional VO | Human voice actor | ~$200 | Studio recording |
| Original score | Human composer | ~$500 | Custom music |
| Documentary footage | Stock + licensed | ~$300 | Real-world visuals |
| Interactive simulation | Custom dev | ~$1,000 | WebGL/Unity |
| Expert review | PhD subject expert | ~$200 | Fact-checking |
| 50+ languages | Professional translation | ~$2,000 | Human translators |
| Accessibility suite | Full compliance | ~$500 | Sign language, audio desc |
| **Total** | | **$5,000-10,000** | Per masterwork |

---

## 🏗️ THE COMPLETE LESSON ANATOMY

A fully-realized lesson contains **137+ distinct assets**:

### Text Assets (60+)
```yaml
Core:
  - topic: "The single most important concept"
  - universal_truth: "The emotional/philosophical anchor"
  - marketing_headline: "The hook that captures attention"
  - marketing_tagline: "The supporting value proposition"
  - marketing_pitch: "The full elevator pitch"
  - extended_explanation: "The deep dive (500+ words)"
  - historical_context: "How we got here"

Learning Structure:
  - learning_objectives: [3-5 specific outcomes]
  - prerequisite_concepts: [what you need to know first]
  - related_topics: [where to go next]
  - bloom_taxonomy_level: "understand|apply|analyze|evaluate|create"

Engagement:
  - fun_facts: [3 surprising truths]
  - common_misconceptions: [3 things people get wrong]
  - real_world_applications: [3 practical uses]
  - discussion_questions: [3 conversation starters]
  - challenge_questions: [3 advanced problems]

Assessment:
  - quick_quiz_questions: [5 multiple choice]
  - hands_on_activities: [3 experiential exercises]
  - creative_prompts: [3 open-ended projects]
  - mastery_criteria: "How you know you've learned it"

Resources:
  - recommended_books: [{isbn, title, author, why}]
  - recommended_videos: [{url, title, duration, why}]
  - interactive_simulations: [{url, title, type}]
  - downloadable_resources: [{url, title, format}]

Content Atoms (60):
  - 12 archetypes × 5 phases
  - Each atom: script, 3 options, 3 responses
  - Totaling ~36,000 words per lesson
```

### Visual Assets (20+)
```yaml
Lesson-Level:
  - hero_image: "Main lesson thumbnail (1920×1080)"
  - thumbnail: "Card thumbnail (640×360)"
  - background: "Environmental context"
  - prop_primary: "Main teaching prop"
  - prop_secondary: "Supporting visual"

Phase-Level (5 phases):
  - hook: "Kelly welcoming, topic context"
  - q1: "Kelly with first teaching prop"
  - q2: "Kelly in contemplation"
  - q3: "Kelly explaining"
  - wisdom: "Kelly celebrating completion"

Reaction Variants (per phase):
  - reaction_a: "Response to first option"
  - reaction_b: "Response to second option"
  - reaction_c: "Response to third option"
  - encouragement: "Supportive reaction"
  - celebration: "Success reaction"
```

### Audio Assets (10+)
```yaml
Voiceover:
  - hook_vo: "Welcome and hook delivery"
  - q1_vo: "First question/fact"
  - q2_vo: "Second question/fact"
  - q3_vo: "Third question/fact"
  - wisdom_vo: "Closing wisdom"

Variants:
  - child_friendly: "Ages 5-8 version"
  - standard: "Ages 9-14 version"
  - adult: "Ages 15+ version"

Ambience:
  - background_music: "Topic-appropriate soundtrack"
  - transition_sounds: "Phase change effects"
```

### Video Assets (5+ for Premium)
```yaml
Kelly Videos:
  - hook_video: "Lip-synced Kelly welcome"
  - teaching_video: "Kelly explaining concept"
  - wisdom_video: "Kelly closing"

B-Roll:
  - topic_footage: "Real-world examples"
  - animation: "Concept visualization"
```

---

## 🔧 THE GENERATION PIPELINE

### Stage 1: IDEATION
```
Input: Topic seed (e.g., "Why do leaves change color?")
Output: Complete lesson DNA

Process:
1. Topic Analysis (Claude)
   - Extract core concept
   - Identify misconceptions
   - Map to curriculum standards
   - Determine age appropriateness
   
2. Universal Truth Discovery (Claude)
   - What's the deeper meaning?
   - Why should anyone care?
   - What's the emotional hook?
   
3. Content Structure (Claude)
   - Generate all 60 atoms
   - Create JSONB fields
   - Validate coherence
   
4. Quality Gate: Human review of DNA
```

### Stage 2: VISUAL GENERATION
```
Input: Lesson DNA + Visual Context Library
Output: All visual assets

Process:
1. Context Matching
   - Map topic to environment
   - Select appropriate props
   - Determine mood/palette
   
2. Prompt Assembly
   - Character consistency (Kelly LoRA)
   - Phase-specific poses
   - Topic integration
   
3. Generation (Parallel)
   - Flux for Kelly shots
   - DALL-E for environments
   - Midjourney for hero images
   
4. Quality Control
   - Character consistency check (face embedding)
   - Background cleanliness
   - Expression appropriateness
   
5. Post-Processing
   - Resize to all required dimensions
   - Generate thumbnails
   - Optimize for web
   
6. Quality Gate: Visual QA review
```

### Stage 3: AUDIO GENERATION
```
Input: Scripts from DNA
Output: All voiceover assets

Process:
1. Script Preparation
   - Add SSML markers
   - Mark emphasis points
   - Set pacing cues
   
2. Voice Generation (ElevenLabs)
   - Kelly voice model
   - Emotional variants
   - Age-appropriate versions
   
3. Quality Control
   - Pronunciation check
   - Timing verification
   - Emotional appropriateness
   
4. Post-Processing
   - Normalize audio levels
   - Add room tone
   - Generate waveforms for UI
   
5. Quality Gate: Audio QA review
```

### Stage 4: VIDEO GENERATION (Premium)
```
Input: Audio + Visuals
Output: Lip-synced Kelly videos

Process:
1. Audio-to-Face
   - Generate facial animation
   - Match Kelly reference
   
2. Video Synthesis
   - OmniHuman/Hedra/HeyGen
   - Character consistency
   
3. Compositing
   - Overlay on backgrounds
   - Add visual effects
   
4. Quality Gate: Video QA review
```

### Stage 5: ASSEMBLY & STORAGE
```
Input: All generated assets
Output: Deployed lesson

Process:
1. Asset Organization
   - Upload to R2/Supabase Storage
   - Generate CDN URLs
   
2. Database Updates
   - Link all URLs to lesson
   - Update atom visual_urls
   - Set generation metadata
   
3. Validation
   - All assets accessible
   - No broken links
   - Performance check
   
4. Quality Gate: Final review before publish
```

---

## 📊 DATABASE SCHEMA FOR LESSON GENERATION

### New Tables Needed:

```sql
-- ═══════════════════════════════════════════════════════════════
-- LESSON GENERATION SYSTEM
-- ═══════════════════════════════════════════════════════════════

-- Track every generation job
CREATE TABLE lesson_generation_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What's being generated
  lesson_id UUID REFERENCES core_lessons(id),
  tier TEXT NOT NULL CHECK (tier IN ('free', 'premium', 'masterwork')),
  requested_assets TEXT[] NOT NULL,
  
  -- Status
  status TEXT DEFAULT 'pending' CHECK (status IN (
    'pending', 'ideation', 'visuals', 'audio', 'video', 
    'assembly', 'review', 'completed', 'failed'
  )),
  progress DECIMAL(5,2) DEFAULT 0,
  current_stage TEXT,
  
  -- Cost tracking
  estimated_cost_usd DECIMAL(10,2),
  actual_cost_usd DECIMAL(10,2),
  cost_breakdown JSONB,
  
  -- Metadata
  requested_by TEXT,
  priority INT DEFAULT 5,
  created_at TIMESTAMP DEFAULT NOW(),
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  
  -- Quality
  qa_status TEXT DEFAULT 'pending',
  qa_reviewer TEXT,
  qa_notes TEXT,
  qa_score DECIMAL(3,2)
);

-- Track individual asset generation
CREATE TABLE lesson_assets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- What it belongs to
  lesson_id UUID REFERENCES core_lessons(id),
  atom_id UUID REFERENCES lesson_atoms(id),
  job_id UUID REFERENCES lesson_generation_jobs(id),
  
  -- Asset identity
  asset_type TEXT NOT NULL, -- 'image', 'audio', 'video', 'text'
  asset_subtype TEXT, -- 'hero', 'phase_hook', 'voiceover_q1', etc.
  variant_name TEXT, -- 'a', 'b', 'child', 'adult', etc.
  
  -- Storage
  storage_provider TEXT, -- 'supabase', 'r2', 'cloudflare'
  storage_path TEXT,
  public_url TEXT,
  cdn_url TEXT,
  
  -- Generation details
  generator TEXT, -- 'flux', 'elevenlabs', 'hedra', 'claude'
  model_version TEXT,
  prompt TEXT,
  prompt_template_id UUID,
  seed BIGINT,
  generation_params JSONB,
  
  -- Quality
  quality_score DECIMAL(3,2),
  is_approved BOOLEAN DEFAULT FALSE,
  approval_method TEXT, -- 'auto', 'human'
  
  -- Technical
  file_size_bytes BIGINT,
  dimensions JSONB, -- {width, height} or {duration_seconds}
  format TEXT,
  
  -- Cost
  generation_cost_usd DECIMAL(10,4),
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT NOW(),
  approved_at TIMESTAMP
);

-- Prompt templates for consistency
CREATE TABLE prompt_templates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Identity
  name TEXT NOT NULL UNIQUE,
  category TEXT, -- 'kelly_pose', 'background', 'voiceover', etc.
  version INT DEFAULT 1,
  
  -- Template
  template TEXT NOT NULL,
  required_variables TEXT[],
  default_values JSONB,
  
  -- Settings
  recommended_generator TEXT,
  recommended_params JSONB,
  
  -- Quality
  success_rate DECIMAL(5,2),
  avg_quality_score DECIMAL(3,2),
  usage_count INT DEFAULT 0,
  
  -- Status
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Visual context library
CREATE TABLE visual_contexts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Matching
  keywords TEXT[] NOT NULL,
  categories TEXT[],
  
  -- Context
  environment TEXT NOT NULL,
  props TEXT[] NOT NULL,
  mood TEXT,
  color_palette TEXT,
  lighting TEXT,
  
  -- Quality
  usage_count INT DEFAULT 0,
  avg_quality_score DECIMAL(3,2),
  
  created_at TIMESTAMP DEFAULT NOW()
);

-- Cost tracking for billing/reporting
CREATE TABLE generation_costs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  job_id UUID REFERENCES lesson_generation_jobs(id),
  asset_id UUID REFERENCES lesson_assets(id),
  
  provider TEXT NOT NULL, -- 'replicate', 'elevenlabs', 'openai', etc.
  service TEXT, -- 'flux-pro', 'claude-sonnet', etc.
  
  tokens_used INT,
  compute_seconds DECIMAL(10,2),
  api_calls INT DEFAULT 1,
  
  cost_usd DECIMAL(10,4) NOT NULL,
  
  created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_jobs_status ON lesson_generation_jobs(status);
CREATE INDEX idx_jobs_lesson ON lesson_generation_jobs(lesson_id);
CREATE INDEX idx_assets_lesson ON lesson_assets(lesson_id);
CREATE INDEX idx_assets_type ON lesson_assets(asset_type, asset_subtype);
CREATE INDEX idx_costs_job ON generation_costs(job_id);
```

---

## 🌐 SCALE REQUIREMENTS

### For 1 Million Daily Active Learners:

```yaml
Daily Traffic:
  - Peak concurrent users: 100,000
  - Lessons viewed per day: 2,000,000
  - Assets served per day: 20,000,000
  - Bandwidth: ~10 TB/day

Storage (365 lessons × full assets):
  - Images: ~20GB (optimized)
  - Audio: ~50GB
  - Video (premium): ~500GB
  - Total: ~600GB + growth

CDN Requirements:
  - Global edge distribution
  - <100ms latency worldwide
  - 99.99% uptime

Database:
  - Read replicas for scale
  - Connection pooling (pgbouncer)
  - Query optimization
```

### Infrastructure Recommendations:

```yaml
Storage: Cloudflare R2
  - $0.015/GB stored
  - $0 egress (huge savings)
  - S3-compatible
  
CDN: Cloudflare
  - Global edge network
  - Free tier handles most traffic
  - Workers for dynamic content
  
Database: Supabase Pro
  - Managed Postgres
  - Built-in auth
  - Edge functions
  
Compute: Vercel + Cloudflare Workers
  - Serverless scale
  - Global distribution
  - No cold starts (Workers)
```

---

## 🚀 IMMEDIATE NEXT STEPS

### Phase 1: Foundation (This Week)
1. [ ] Create `lesson_generation_jobs` table
2. [ ] Create `lesson_assets` table  
3. [ ] Create `prompt_templates` table
4. [ ] Create `visual_contexts` table
5. [ ] Seed visual_contexts with 50+ topic contexts
6. [ ] Migrate existing prompts to prompt_templates

### Phase 2: Pipeline (Next Week)
1. [ ] Build generation orchestrator
2. [ ] Implement quality control checks
3. [ ] Create admin dashboard for monitoring
4. [ ] Add cost tracking

### Phase 3: Scale (Week 3)
1. [ ] Generate all 365 lesson visuals
2. [ ] Generate all 365 lesson audio
3. [ ] Update database with all URLs
4. [ ] Quality review all assets

### Phase 4: Polish (Week 4)
1. [ ] A/B test hero images
2. [ ] Optimize for performance
3. [ ] Final QA pass
4. [ ] Launch readiness review

---

## 💰 BUDGET REQUIREMENTS

### Initial 365 Lessons (Tier 1):
```
Text generation: 365 × $0.50 = $183
Visual generation: 365 × $0.40 = $146  
Audio generation: 365 × $2.00 = $730
Quality control: 365 × $0.10 = $37
═══════════════════════════════════
Total: ~$1,100 for full visual/audio coverage
```

### Ongoing Monthly (at scale):
```
Supabase Pro: $25/mo
Cloudflare Pro: $20/mo
Replicate: ~$50/mo (new content)
ElevenLabs: ~$50/mo (voice)
═══════════════════════════════════
Total: ~$150/mo infrastructure
```

### Premium Masterwork Fund:
```
Available for philanthropic sponsorship:
- 10 masterwork lessons × $5,000 = $50,000
- Would fund development of flagship content
- Sponsor recognition in lesson credits
```

---

## 🎓 THE PROMISE

When this architecture is complete:

1. **Any learner, anywhere** can access world-class education
2. **Any topic** can become a lesson in hours, not months
3. **Any language** is supported from day one
4. **Any device** delivers the full experience
5. **Any educator** can contribute to the library
6. **Every interaction** improves the system

This is not just an app. This is infrastructure for human potential.

---

*"The goal of education is not to increase the amount of knowledge but to create the possibilities for a child to invent and discover."* — Jean Piaget

*"We're building those possibilities, at scale, for everyone."* — Curious Kelly

