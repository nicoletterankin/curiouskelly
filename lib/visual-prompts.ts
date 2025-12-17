/**
 * VISUAL COMMONS PROMPT LIBRARY
 * 
 * Structured prompt generation for educational visuals.
 * Every prompt is designed to produce consistent, high-quality,
 * age-appropriate educational content.
 * 
 * @version 1.0.0
 * @created December 17, 2025
 */

import * as crypto from 'crypto';

// =============================================================================
// TYPES
// =============================================================================

export type Phase = 'hook' | 'cliff' | 'fact1' | 'fact2' | 'fact3' | 'wisdom' | 'outro' | 'complete';
export type AgeGroup = '2-5' | '6-12' | '13-17' | '18+' | 'all';
export type VisualType = 'infographic' | 'diagram' | 'scene' | 'comparison' | 'timeline' | 'process';
export type IconType = 'atom' | 'spark' | 'arrow' | 'leaf' | 'heart' | 'wave' | 'dot' | 'star' | 'bulb';

export interface VisualContext {
  dayNumber: number;
  phase: Phase;
  topic: string;
  visualType: VisualType;
  ageGroup: AgeGroup;
  style?: string;
  // Lesson-specific content
  universalTruth?: string;
  hookQuestion?: string;
  facts?: string[];
  lifeApplication?: string;
}

export interface InfographicBrief {
  template: 'cross_section' | 'process_flow' | 'compare' | 'timeline' | 'radial';
  headline: string;
  subhead: string;
  callouts?: Array<{ label: string; detail: string; icon: IconType }>;
  steps?: Array<{ label: string; detail: string; icon: IconType }>;
  left?: { label: string; bullets: string[] };
  right?: { label: string; bullets: string[] };
  centerLabel?: string;
  orbitals?: Array<{ label: string; icon: IconType }>;
}

// =============================================================================
// CONTENT HASH GENERATION
// =============================================================================

/**
 * Generates a deterministic SHA-256 hash for any visual context.
 * Same context = same hash = cache hit.
 */
export function generateVisualHash(context: VisualContext): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    t: normalizeText(context.topic),
    v: context.visualType,
    a: context.ageGroup || 'all',
    s: context.style || 'default',
    ver: '1' // Schema version for cache invalidation
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, '')
    .replace(/\s+/g, ' ');
}

// =============================================================================
// BRAND CONSTANTS
// =============================================================================

export const BRAND = {
  colors: {
    bg: '#0a0a0b',
    card: '#18181b',
    border: '#27272a',
    text: '#f4f4f5',
    muted: '#a1a1aa',
    dim: '#71717a',
    accent: '#3b82f6',
    gold: '#fbbf24',
    ok: '#22c55e',
    warn: '#f59e0b',
  },
  typography: {
    headline: 'bold, modern, clean',
    body: 'simple, minimal, kid-friendly',
  }
};

// =============================================================================
// SYSTEM CONTEXT (Layer 1)
// =============================================================================

export const SYSTEM_CONTEXT = `
You are Kelly's Visual Design Lead, creating educational graphics for Curious Kelly,
a daily learning platform for curious minds of all ages.

BRAND IDENTITY:
- Palette: Dark premium backgrounds (${BRAND.colors.bg}, ${BRAND.colors.card}), clean neon accents (${BRAND.colors.accent} blue, ${BRAND.colors.gold} gold, ${BRAND.colors.ok} green)
- Typography: Modern, clean, minimal. Headlines bold, body simple.
- Vibe: Approachable science, wonder without intimidation, clarity over complexity

HARD CONSTRAINTS:
- NEVER include text that could be misread (no tiny labels, no cursive)
- NEVER depict violence, fear, or inappropriate content
- NEVER use copyrighted characters or logos
- ALWAYS ensure scientific accuracy
- ALWAYS make it understandable at a glance
- ALWAYS leave space for Kelly to appear alongside (right 40% of frame for scenes)

OUTPUT:
Return ONLY the requested format (JSON for infographic briefs, base64 for images).
No explanations, no commentary, no apologies.
`;

// =============================================================================
// AGE ADAPTATIONS (Layer 2a)
// =============================================================================

const AGE_ADAPTATIONS: Record<AgeGroup, string> = {
  '2-5': `
For young children (ages 2-5):
- Use bright, saturated colors
- Include friendly, rounded shapes
- Anthropomorphize concepts when helpful (happy sun, curious atoms)
- Keep complexity very low - one main idea
- Make it feel like a picture book illustration
- Use large, simple shapes
- Avoid any scary or overwhelming imagery
`,

  '6-12': `
For elementary/middle school (ages 6-12):
- Balance fun with accuracy
- Include "cool factor" elements (space, dinosaurs, explosions if relevant)
- Can show more detail and relationships
- Make them feel smart for understanding
- Use analogies to things they know (video games, sports, animals)
- Keep text labels short but more detailed than younger ages
`,

  '13-17': `
For teens (ages 13-17):
- More sophisticated visual language
- Can include complexity and nuance
- Avoid anything that feels "babyish"
- Make it feel current and relevant
- Reference pop culture where appropriate (without copyright)
- Show real scientific accuracy
`,

  '18+': `
For adults (ages 18+):
- Full scientific accuracy and detail
- Professional, polished aesthetic
- Can include technical terminology
- Respect their intelligence
- Clean, minimalist design preferred
- No need for anthropomorphization
`,

  'all': `
For all ages (universal):
- Universal visual language that works at any age
- Clear at surface level, richer for those who look closer
- Layered complexity - simple main idea, detailed background
- Avoid age-specific references
- Focus on the wonder of the concept itself
`
};

// =============================================================================
// VISUAL TYPE TEMPLATES (Layer 2b)
// =============================================================================

const INFOGRAPHIC_TEMPLATE = `
TASK: Generate a structured infographic brief as JSON.

TEMPLATE OPTIONS:
1. cross_section - Layered diagram showing internal structure (great for "how X works")
2. process_flow - 3-step horizontal flow with arrows (great for sequences)
3. compare - Two-panel side-by-side comparison (great for contrasts)
4. timeline - Chronological sequence (great for history/development)
5. radial - Central concept with orbital related ideas (great for ecosystems/relationships)

OUTPUT SCHEMA:
{
  "template": "cross_section" | "process_flow" | "compare" | "timeline" | "radial",
  "headline": "8 words max, compelling hook",
  "subhead": "16 words max, clarifying detail",
  "callouts": [
    { "label": "4 words max", "detail": "18 words max", "icon": "atom|spark|arrow|leaf|heart|wave|dot|star|bulb" }
  ],
  "steps": [...],  // For process_flow only
  "left": { "label": "...", "bullets": [...] },  // For compare only
  "right": { "label": "...", "bullets": [...] }, // For compare only
  "centerLabel": "...",  // For radial only
  "orbitals": [...]  // For radial only
}

RULES:
- Choose the template that BEST fits the educational content
- Labels MUST be short (≤4 words) - we render real text, not image text
- Details should be kid-friendly but scientifically accurate
- Use the icon that best represents each concept:
  • atom - scientific/molecular concepts
  • spark - ideas, creativity, insights
  • arrow - direction, process, movement
  • leaf - nature, growth, environment
  • heart - emotions, body, health
  • wave - sound, light, energy
  • dot - neutral/general point
  • star - important highlights
  • bulb - discoveries, "aha" moments
`;

const DIAGRAM_TEMPLATE = `
TASK: Generate a clear technical diagram.

STYLE:
- Clean vector aesthetic
- Limited color palette (3-4 colors max from brand palette)
- Clear visual hierarchy
- Arrows and flow indicators where appropriate
- Dark background (${BRAND.colors.bg})

COMPOSITION:
- Main subject prominently displayed
- Supporting elements in correct spatial relationship
- Clear visual distinction between components
- Space for overlay labels (we add text separately)
- Leave right 30% clear for Kelly placement

AVOID:
- Cluttered compositions
- Ambiguous relationships between elements
- Overly complex detail that obscures main concept
- Text within the image (we add that separately)
`;

const SCENE_TEMPLATE = `
TASK: Generate a photorealistic educational scene.

COMPOSITION RULES:
- Leave the right 40% of frame clear (Kelly will appear there)
- Subject should be left-center framed
- Lighting should be warm, inviting, professional
- Background should be contextually appropriate but not distracting

STYLE:
- Professional photography aesthetic
- 16:9 aspect ratio (1344×768 pixels)
- 4K resolution quality
- Natural lighting preferred
- No text overlays
- Shot on professional camera (Canon EOS R5 or similar)

EDUCATIONAL FOCUS:
- The scene should immediately communicate the core concept
- Include visual details that teach (e.g., for "photosynthesis", show sunlight hitting leaves)
- Make abstract concepts tangible through metaphor
`;

const VISUAL_TYPE_TEMPLATES: Record<VisualType, string> = {
  infographic: INFOGRAPHIC_TEMPLATE,
  diagram: DIAGRAM_TEMPLATE,
  scene: SCENE_TEMPLATE,
  comparison: INFOGRAPHIC_TEMPLATE, // Uses compare template
  timeline: INFOGRAPHIC_TEMPLATE, // Uses timeline template
  process: INFOGRAPHIC_TEMPLATE // Uses process_flow template
};

// =============================================================================
// PHASE-SPECIFIC PROMPTS (Layer 3)
// =============================================================================

function getPhasePrompt(phase: Phase, context: VisualContext): string {
  const prompts: Record<Phase, string> = {
    hook: `
PHASE: HOOK (Creating Curiosity)

This is the opening moment - we want learners to say "Wait, what?!"

Topic: "${context.topic}"
Universal truth: ${context.universalTruth || 'Not specified'}
Hook question: ${context.hookQuestion || 'Why should I care about this?'}

GOAL: Create intrigue, NOT answers
- Hint at something surprising
- Show the mystery, not the solution
- Make them NEED to know more

VISUAL APPROACH:
- If infographic: Use a question-provoking headline
- If scene: Capture the moment of wonder/discovery
- If diagram: Show the "before" state that begs explanation
`,

    cliff: `
PHASE: CLIFF (Deepening the Mystery)

The learner just made a choice - now we deepen the mystery before revealing answers.

Topic: "${context.topic}"
Universal truth: ${context.universalTruth || 'Not specified'}

GOAL: Create productive tension
- Show the gap between what we think and what's true
- Hint that something surprising is coming
- Build anticipation

VISUAL APPROACH:
- Compare misconception vs. reality (but don't fully reveal)
- Show complexity that needs explanation
- Create a "but wait, there's more" feeling
`,

    fact1: `
PHASE: FACT 1 (Foundation Building)

First key learning point - this is TEACHING content.

Topic: "${context.topic}"
Key fact: ${context.facts?.[0] || 'The foundational concept'}
Universal truth: ${context.universalTruth || 'Not specified'}

GOAL: Crystal clear understanding
- One main idea, well explained
- Building block for what comes next
- Learner should be able to explain this to a friend

VISUAL APPROACH:
- Clarity is everything
- If infographic: Process or cross-section showing "how"
- If diagram: Clear labeled components
- If scene: Concrete example of the concept
`,

    fact2: `
PHASE: FACT 2 (Deeper Insight)

Second key learning point - building on fact1.

Topic: "${context.topic}"
Key fact: ${context.facts?.[1] || 'The deeper insight'}
Previous fact: ${context.facts?.[0] || 'Foundation concept'}

GOAL: Show connections and depth
- Build on what they just learned
- Reveal a new layer
- "Now that you know X, here's Y"

VISUAL APPROACH:
- Show relationship between concepts
- If infographic: Add complexity to earlier model
- If diagram: Zoom in on a component, or show next step
- If scene: Show the concept in action
`,

    fact3: `
PHASE: FACT 3 (The Wow Moment)

Third key learning point - often the most memorable/surprising.

Topic: "${context.topic}"
Key fact: ${context.facts?.[2] || 'The surprising detail'}

GOAL: Create a memorable "wow" moment
- This is often what learners remember most
- Surprising but true
- Makes them want to share

VISUAL APPROACH:
- Maximum impact, minimum complexity
- If infographic: Highlight the surprising statistic or comparison
- If diagram: Show the unexpected connection
- If scene: Capture the dramatic moment
`,

    wisdom: `
PHASE: WISDOM (Life Application)

Crystallizing the lesson into lasting insight.

Topic: "${context.topic}"
Universal truth: ${context.universalTruth || 'Not specified'}
Life application: ${context.lifeApplication || 'How this applies beyond the lesson'}

GOAL: Make it personal and permanent
- Connect lesson to learner's own life
- "Why this matters to YOU"
- Poster-worthy insight

VISUAL APPROACH:
- Think "wall poster" or "quote card"
- If infographic: Summary with life application
- If diagram: Show concept applied to everyday life
- If scene: Person (not Kelly) experiencing the insight
`,

    outro: `
PHASE: OUTRO (Celebration & Teaser)

Closing the lesson and teasing what's next.

Topic: "${context.topic}"

GOAL: Celebrate and build anticipation
- Acknowledge what they learned
- Hint at tomorrow's adventure
- End on positive energy

VISUAL APPROACH:
- Celebratory, forward-looking
- Can show tomorrow's topic preview
- Keep it light and energetic
`,

    complete: `
PHASE: COMPLETE (Summary Infographic)

Comprehensive visual summary of the entire lesson.

Topic: "${context.topic}"
Universal truth: ${context.universalTruth || 'Not specified'}
All facts: ${context.facts?.join(' | ') || 'Key concepts'}

GOAL: One image that teaches the whole lesson
- Shareable on social media
- Reference material for later
- "The complete picture"

VISUAL APPROACH:
- Use radial or process_flow template
- Include hook, key facts, and wisdom
- Optimized for sharing (include topic title)
`
  };

  return prompts[phase];
}

// =============================================================================
// MAIN PROMPT BUILDER
// =============================================================================

/**
 * Builds a complete prompt for visual generation.
 * Combines all three layers: system context, visual type template, and lesson context.
 */
export function buildVisualPrompt(context: VisualContext): string {
  const layers = [
    // Layer 1: System Context
    SYSTEM_CONTEXT,
    
    // Layer 2a: Age Adaptation
    `AGE GROUP: ${context.ageGroup}`,
    AGE_ADAPTATIONS[context.ageGroup],
    
    // Layer 2b: Visual Type Template
    `VISUAL TYPE: ${context.visualType}`,
    VISUAL_TYPE_TEMPLATES[context.visualType],
    
    // Layer 3: Phase-Specific Context
    getPhasePrompt(context.phase, context),
    
    // Final context summary
    `
GENERATION CONTEXT:
- Day: ${context.dayNumber}
- Topic: ${context.topic}
- Phase: ${context.phase}
- Visual Type: ${context.visualType}
- Age Group: ${context.ageGroup}
- Style: ${context.style || 'default'}

Generate the ${context.visualType} now.
`
  ];

  return layers.join('\n\n---\n\n');
}

// =============================================================================
// SPECIALIZED PROMPT BUILDERS
// =============================================================================

/**
 * Builds a prompt specifically for Gemini text model to generate infographic briefs.
 * These briefs are then rendered as SVGs client-side.
 */
export function buildInfographicBriefPrompt(context: VisualContext): string {
  return `
${SYSTEM_CONTEXT}

${AGE_ADAPTATIONS[context.ageGroup]}

${INFOGRAPHIC_TEMPLATE}

${getPhasePrompt(context.phase, context)}

LESSON DETAILS:
- Day: ${context.dayNumber}
- Topic: ${context.topic}
- Phase: ${context.phase}
- Age Group: ${context.ageGroup}

Return ONLY valid JSON matching the schema above. No markdown, no explanation.
`;
}

/**
 * Builds a prompt for Imagen/DALL-E style image generation.
 */
export function buildImageGenerationPrompt(context: VisualContext): string {
  const baseStyle = context.ageGroup === '2-5' 
    ? 'Pixar-style 3D animation, friendly, colorful, child-safe'
    : context.ageGroup === '6-12'
    ? 'Clean digital illustration, educational, engaging, modern'
    : 'Professional photography or clean infographic style';

  return `
${context.visualType === 'scene' ? SCENE_TEMPLATE : DIAGRAM_TEMPLATE}

TOPIC: ${context.topic}
PHASE: ${context.phase}
AGE: ${context.ageGroup}

${getPhasePrompt(context.phase, context)}

STYLE: ${baseStyle}

Generate a single, clear image that teaches the concept at a glance.
No text in the image. Educational, accurate, engaging.
`;
}

// =============================================================================
// VALIDATION
// =============================================================================

/**
 * Validates an infographic brief for quality and completeness.
 */
export function validateInfographicBrief(brief: InfographicBrief): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  
  // Check required fields
  if (!brief.template) errors.push('Missing template');
  if (!brief.headline) errors.push('Missing headline');
  if (!brief.subhead) errors.push('Missing subhead');
  
  // Check headline length
  if (brief.headline && brief.headline.split(' ').length > 8) {
    errors.push('Headline exceeds 8 words');
  }
  
  // Check subhead length
  if (brief.subhead && brief.subhead.split(' ').length > 16) {
    errors.push('Subhead exceeds 16 words');
  }
  
  // Check callout labels
  if (brief.callouts) {
    brief.callouts.forEach((c, i) => {
      if (c.label.split(' ').length > 4) {
        errors.push(`Callout ${i + 1} label exceeds 4 words`);
      }
      if (c.detail.split(' ').length > 18) {
        errors.push(`Callout ${i + 1} detail exceeds 18 words`);
      }
    });
  }
  
  return { valid: errors.length === 0, errors };
}

/**
 * Clamps text to maximum word count.
 */
export function clampWords(text: string, maxWords: number): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text.trim();
  return words.slice(0, maxWords).join(' ');
}

/**
 * Ensures brief quality by enforcing constraints.
 */
export function ensureBriefQuality(brief: InfographicBrief): InfographicBrief {
  return {
    ...brief,
    headline: clampWords(brief.headline, 8),
    subhead: clampWords(brief.subhead, 16),
    callouts: brief.callouts?.slice(0, 5).map(c => ({
      ...c,
      label: clampWords(c.label, 4),
      detail: clampWords(c.detail, 18)
    })),
    steps: brief.steps?.slice(0, 3).map(s => ({
      ...s,
      label: clampWords(s.label, 4),
      detail: clampWords(s.detail, 14)
    }))
  };
}

// =============================================================================
// EXPORTS
// =============================================================================

export default {
  generateVisualHash,
  buildVisualPrompt,
  buildInfographicBriefPrompt,
  buildImageGenerationPrompt,
  validateInfographicBrief,
  ensureBriefQuality,
  BRAND,
  SYSTEM_CONTEXT
};
