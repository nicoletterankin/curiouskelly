# Visual Variants System
## Standing on the Shoulders of Learners

> "Every learner who passes through enriches the path for those who follow."

---

## 🎯 Core Philosophy

Traditional education: One textbook, one illustration, take it or leave it.

**Curious Kelly's Visual Commons**: A living library of visual interpretations, grown by learners, for learners. When you arrive at a lesson phase, you see the visual tapestry created by everyone before you—and you can add your own thread.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    VISUAL VARIANTS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Day 1, Phase: Hook, Topic: "Starting Fresh"               │
│                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│   │ARTISTIC │  │TEXTBOOK │  │DIAGRAM  │  │MINIMAL  │       │
│   │ Scene   │  │ Page    │  │ Labeled │  │ Concept │       │
│   │         │  │         │  │         │  │         │       │
│   │ 847     │  │ 234     │  │ 156     │  │ 89      │       │
│   │ helped  │  │ helped  │  │ helped  │  │ helped  │       │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
│                                                             │
│   [+ Generate Your Own Variant]                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Variant Dimensions

### 1. Visual Style (`style`)

| Style | Description | Best For | Imagen Prompt Modifier |
|-------|-------------|----------|------------------------|
| `artistic` | Photorealistic, cinematic, emotional | Hooks, Wisdom | "Ultra photorealistic, cinematic lighting, emotional" |
| `textbook` | Educational illustration with labels | Facts, Complete | "Educational textbook illustration, labeled diagram, clear annotations" |
| `diagram` | Technical diagram, flowcharts, systems | Process, Comparison | "Technical diagram, clean lines, labeled components, blueprint style" |
| `medical` | Anatomical accuracy, scientific precision | Biology, Health | "Medical illustration, anatomical accuracy, scientific detail, labeled" |
| `minimal` | Simple shapes, one core concept | Young learners, Quick review | "Minimalist design, simple shapes, clean background, single concept" |
| `infographic` | Data visualization, statistics, hierarchy | Facts with numbers | "Infographic style, data visualization, clear hierarchy, bold typography" |
| `illustrated` | Warm, hand-drawn feel, approachable | All ages, friendly | "Warm illustrated style, hand-drawn feel, friendly and approachable" |
| `3d_render` | 3D visualization, depth, models | Science, Engineering | "3D rendered visualization, detailed model, professional lighting" |

### 2. Complexity Level (`complexity`)

| Level | Description | Target Audience |
|-------|-------------|-----------------|
| `simple` | Core concept only, no extras | Ages 2-8, Quick review |
| `standard` | Balanced detail, main points | General audience |
| `detailed` | Rich detail, multiple concepts | Deep learners |
| `expert` | Maximum complexity, professional | Advanced learners, Teachers |

### 3. Text Inclusion (`includes_text`)

| Setting | Description | Use Case |
|---------|-------------|----------|
| `none` | Pure visual, no text | Social sharing, Overlays |
| `labels` | Key terms labeled | Learning, Reference |
| `full` | Explanatory text, captions | Textbook-style, Self-study |
| `bilingual` | EN + ES labels | Language learners |

### 4. Age Adaptation (`age_group`)

Already tracked: `2-5`, `6-12`, `13-17`, `18+`, `all`

---

## 🔑 Content Hash Structure (Updated)

```typescript
interface VisualVariantContext {
  dayNumber: number;        // 1-365
  phase: string;            // hook, fact1, wisdom, etc.
  style: VisualStyle;       // artistic, textbook, diagram, etc.
  complexity: Complexity;   // simple, standard, detailed, expert
  includesText: TextMode;   // none, labels, full, bilingual
  ageGroup: AgeGroup;       // 2-5, 6-12, 13-17, 18+, all
  version: string;          // Schema version for cache invalidation
}

// Hash = SHA256(canonicalized JSON)
// This creates a unique slot for each variant combination
```

**Example hashes for Day 1, Hook phase:**
- `abc123...` = artistic + standard + none + all
- `def456...` = textbook + detailed + labels + 18+
- `ghi789...` = diagram + simple + labels + 6-12

Each is a **separate cache slot** that can be filled by any learner.

---

## 🎨 Prompt Templates by Style

### ARTISTIC (Current Default)
```
Create a stunning, curiosity-sparking scene for: "{topic}"

[Phase-specific context]

STYLE:
- Ultra photorealistic, professional photography aesthetic
- Dramatic lighting, cinematic composition
- 16:9 aspect ratio, 4K quality
- Warm, inviting color palette with a sense of wonder

DO NOT include any text, logos, or watermarks.
```

### TEXTBOOK
```
Create an educational textbook illustration for: "{topic}"

Key concept to illustrate: {fact_or_truth}

STYLE:
- Professional educational illustration
- Clear, well-organized layout
- Include labeled annotations pointing to key elements
- Clean white or light background
- 16:9 aspect ratio, print quality

TEXT TO INCLUDE:
- Title: "{topic}"
- Key labels: {extracted_key_terms}
- Brief caption explaining the main concept

Make this suitable for an educational textbook or classroom poster.
```

### DIAGRAM
```
Create a detailed technical diagram for: "{topic}"

Concept to visualize: {universal_truth}

STYLE:
- Clean technical diagram with precise lines
- Blueprint or schematic aesthetic
- Numbered or lettered components with legend
- Arrows showing relationships or flow
- Professional engineering drawing style
- 16:9 aspect ratio

INCLUDE LABELS FOR:
- Main components
- Relationships between elements
- Key terms: {extracted_terms}

This should look like it belongs in a technical manual or scientific paper.
```

### MEDICAL/SCIENTIFIC
```
Create a detailed medical/scientific illustration for: "{topic}"

Scientific concept: {fact_or_truth}

STYLE:
- Medical illustration accuracy
- Cross-section or cutaway views where appropriate
- Anatomically/scientifically precise
- Professional medical textbook quality
- Labeled with proper terminology
- 16:9 aspect ratio

LABELS:
- Anatomical/scientific terms
- Process steps if applicable
- Scale reference if relevant

Suitable for medical education or scientific publication.
```

### MINIMAL
```
Create a minimalist visual for: "{topic}"

Core concept: {single_key_point}

STYLE:
- Ultra-minimalist design
- Maximum 3 colors
- Single central concept
- Clean negative space
- Modern, elegant simplicity
- 16:9 aspect ratio

OPTIONAL TEXT:
- One key word or phrase if essential: "{key_word}"

Think Apple product design meets educational clarity.
```

### INFOGRAPHIC
```
Create an infographic-style visual for: "{topic}"

Key data/facts to visualize:
{fun_facts_as_bullet_points}

STYLE:
- Bold infographic design
- Clear visual hierarchy
- Icons and visual metaphors
- Statistics displayed prominently
- Eye-catching color scheme
- 16:9 aspect ratio

TEXT ELEMENTS:
- Headline: "{topic}"
- 2-3 key statistics or facts
- Brief explanatory text

Make this shareable and instantly informative.
```

---

## 🗄️ Database Schema Enhancement

### Updated `visual_commons` table

```sql
-- Add variant columns to existing table
ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS 
  style TEXT DEFAULT 'artistic' CHECK (style IN (
    'artistic', 'textbook', 'diagram', 'medical', 
    'minimal', 'infographic', 'illustrated', '3d_render'
  ));

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  complexity TEXT DEFAULT 'standard' CHECK (complexity IN (
    'simple', 'standard', 'detailed', 'expert'
  ));

ALTER TABLE visual_commons ADD COLUMN IF NOT EXISTS
  includes_text TEXT DEFAULT 'none' CHECK (includes_text IN (
    'none', 'labels', 'full', 'bilingual'
  ));

-- Index for variant queries
CREATE INDEX IF NOT EXISTS idx_vc_variants 
  ON visual_commons(day_number, phase, style, complexity, includes_text, age_group);
```

### New: `learner_visual_preferences` table

```sql
CREATE TABLE IF NOT EXISTS learner_visual_preferences (
  user_id UUID PRIMARY KEY REFERENCES auth.users(id),
  
  -- Default preferences
  preferred_style TEXT DEFAULT 'artistic',
  preferred_complexity TEXT DEFAULT 'standard',
  preferred_text_mode TEXT DEFAULT 'none',
  
  -- Learning over time
  style_history JSONB DEFAULT '{}',  -- {artistic: 45, textbook: 12, ...}
  
  -- A/B testing participation
  experiment_cohort TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### New: `visual_selections` table (what learners choose)

```sql
CREATE TABLE IF NOT EXISTS visual_selections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  visual_id UUID NOT NULL REFERENCES visual_commons(id),
  learner_id UUID REFERENCES auth.users(id),
  session_id TEXT,
  
  -- What variants were shown
  variants_shown UUID[] NOT NULL,
  
  -- Selection context
  day_number INTEGER NOT NULL,
  phase TEXT NOT NULL,
  
  -- Timing
  time_to_select_ms INTEGER,
  selected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vs_visual ON visual_selections(visual_id);
CREATE INDEX IF NOT EXISTS idx_vs_day_phase ON visual_selections(day_number, phase);
```

---

## 🔄 The Learner Flow

### Step 1: Arrive at Phase
```
Learner reaches "Hook" phase of Day 1
```

### Step 2: Fetch Available Variants
```sql
SELECT * FROM visual_commons
WHERE day_number = 1 
  AND phase = 'hook'
  AND status = 'active'
ORDER BY 
  -- Prioritize learner's preferred style
  CASE WHEN style = $preferred_style THEN 0 ELSE 1 END,
  -- Then by popularity
  unique_learners_helped DESC
LIMIT 8;
```

### Step 3: Display Variant Grid
```
┌─────────────────────────────────────────────┐
│  Choose your visual style:                  │
│                                             │
│  [Artistic]  [Textbook]  [Diagram]          │
│    ⭐ 847      234          156             │
│                                             │
│  [+ Create your own variant]                │
└─────────────────────────────────────────────┘
```

### Step 4: Selection or Generation
- **If they select**: Record the selection, increment counters
- **If they generate**: New variant added to commons

### Step 5: System Learns
- Update `learner_visual_preferences`
- Track which variants perform best for which phases
- Surface winning variants to more learners

---

## 🧬 Personalization Without Isolation

The magic: **Personalization at the variant level, not the individual level.**

Instead of generating unique images per learner (expensive, isolated), we generate variants that many learners share. Over time, we discover:

- "Learners who struggle with Day 1 prefer `diagram` style"
- "Phase `wisdom` performs best with `minimal` style"
- "Age 6-12 engages more with `illustrated` style"

This creates **emergent personalization** without per-learner generation costs.

---

## 📈 Growth Model

### Day 1 Launch
- 8 artistic variants (current)
- Learners start generating alternatives

### Month 1
- ~50 variants per lesson across styles
- Preference patterns emerge

### Month 6
- ~200 variants per lesson
- Rich diversity of visual approaches
- Strong data on what works

### Year 1
- Comprehensive visual library
- AI-recommended variants based on learner history
- Near-instant personalization from commons

---

## 🎯 Implementation Priority

### Phase 1: Schema & Seeding (Now)
1. ✅ Add variant columns to `visual_commons`
2. ✅ Create preference/selection tables
3. Generate seed variants: artistic + textbook + diagram for key phases

### Phase 2: Variant Selection UI
1. Show variant grid when multiple exist
2. Track selections
3. Simple preference learning

### Phase 3: Generation Options
1. Let learners choose style when generating
2. Prompt templates per style
3. Quality gates for commons inclusion

### Phase 4: Smart Recommendations
1. ML model for variant recommendations
2. A/B testing framework
3. Cohort-based personalization

---

## 💡 Key Insight: Text in Visuals

Imagen 4 can render text! This unlocks:

1. **Labeled diagrams** - Anatomical parts, scientific terms
2. **Textbook pages** - Topic headers, explanatory captions
3. **Infographics** - Statistics, key facts, quotes
4. **Bilingual visuals** - EN/ES labels for language learners

### Text Prompt Guidelines

```
TEXT TO INCLUDE IN IMAGE:
- Keep text minimal (5-10 words max per label)
- Use clear, simple fonts
- High contrast against background
- Proper scientific/medical terminology
- Spell out abbreviations

EXAMPLE:
- Title: "Photosynthesis"
- Labels: "Chloroplast", "Sunlight", "CO₂", "O₂", "Glucose"
- Caption: "How plants convert light to energy"
```

---

## 🔐 Quality Gates

Not every generation makes it to the commons:

1. **Automatic checks**
   - Image renders correctly
   - Meets size/format requirements
   - No safety filter triggers

2. **Community signals**
   - Low selection rate → flag for review
   - Negative feedback → remove from rotation
   - High engagement → boost visibility

3. **Staff review** (optional)
   - Periodic quality audits
   - Featured variant curation

---

## 📝 Summary

| Concept | Old Approach | New Approach |
|---------|--------------|--------------|
| Visuals per phase | 1 | Many variants |
| Personalization | None | Style preferences |
| Generation | Platform only | Platform + Learners |
| Cost model | Fixed | Distributed |
| Quality | Static | Evolutionary |
| Learning | Passive | Every learner contributes |

**The Visual Commons is a living library that grows smarter and richer with every learner who passes through.**
