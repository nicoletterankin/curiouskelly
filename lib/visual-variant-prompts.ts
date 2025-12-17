/**
 * VISUAL VARIANT PROMPTS
 * 
 * Structured prompt templates for generating diverse visual styles.
 * Each style has unique characteristics optimized for different learning preferences.
 * 
 * @created December 17, 2025
 */

// =============================================================================
// TYPES
// =============================================================================

export type VisualStyle = 
  | 'artistic' 
  | 'textbook' 
  | 'diagram' 
  | 'medical' 
  | 'minimal' 
  | 'infographic' 
  | 'illustrated' 
  | '3d_render';

export type Complexity = 'simple' | 'standard' | 'detailed' | 'expert';
export type TextMode = 'none' | 'labels' | 'full' | 'bilingual';
export type Phase = 'hook' | 'cliff' | 'fact1' | 'fact2' | 'fact3' | 'wisdom' | 'outro' | 'complete';

export interface LessonContext {
  topic: string;
  universalTruth: string;
  funFacts?: string[];
  wowMoment?: string;
  keyTerms?: string[];
  dayNumber: number;
}

export interface VariantRequest {
  lesson: LessonContext;
  phase: Phase;
  style: VisualStyle;
  complexity: Complexity;
  includesText: TextMode;
  ageGroup?: string;
}

// =============================================================================
// STYLE FOUNDATIONS
// =============================================================================

const STYLE_FOUNDATIONS: Record<VisualStyle, string> = {
  artistic: `
VISUAL STYLE: Ultra Photorealistic Artistic
- Professional photography aesthetic, cinematic quality
- Dramatic lighting with natural feel
- Warm, inviting color palette
- Emotional resonance and wonder
- 16:9 aspect ratio, 4K resolution
- Think "National Geographic meets museum exhibit"`,

  textbook: `
VISUAL STYLE: Educational Textbook Illustration
- Professional educational illustration quality
- Clear, organized visual hierarchy
- Well-integrated text labels and annotations
- Clean background (white, cream, or light blue)
- Print-ready quality, suitable for classroom
- 16:9 aspect ratio
- Think "college textbook meets modern design"`,

  diagram: `
VISUAL STYLE: Technical Diagram
- Clean, precise technical drawing
- Blueprint or schematic aesthetic
- Clearly numbered/lettered components
- Arrows showing relationships and flow
- Legend or key for reference
- 16:9 aspect ratio
- Think "engineering manual meets infographic"`,

  medical: `
VISUAL STYLE: Medical/Scientific Illustration
- Anatomical accuracy and scientific precision
- Cross-section or cutaway views where appropriate
- Proper medical/scientific terminology in labels
- Professional medical illustration quality
- Neutral, clinical color palette
- 16:9 aspect ratio
- Think "Gray's Anatomy meets modern medical education"`,

  minimal: `
VISUAL STYLE: Minimalist Concept
- Ultra-clean, modern minimalist design
- Maximum 3-4 colors
- Single central concept, no clutter
- Generous negative space
- Elegant simplicity
- 16:9 aspect ratio
- Think "Apple design principles meets education"`,

  infographic: `
VISUAL STYLE: Bold Infographic
- Eye-catching data visualization
- Clear visual hierarchy with icons
- Statistics displayed prominently
- Modern, bold color scheme
- Information-dense but organized
- 16:9 aspect ratio
- Think "New York Times graphics meets educational content"`,

  illustrated: `
VISUAL STYLE: Warm Illustrated
- Hand-drawn, approachable aesthetic
- Warm, friendly color palette
- Slightly whimsical but educational
- Accessible to all ages
- Storybook quality
- 16:9 aspect ratio
- Think "children's book illustration meets educational clarity"`,

  '3d_render': `
VISUAL STYLE: 3D Rendered Visualization
- Professional 3D modeling and rendering
- Realistic materials and lighting
- Depth and dimensional clarity
- Technical accuracy in proportions
- Studio lighting quality
- 16:9 aspect ratio
- Think "CGI documentary meets educational model"`
};

// =============================================================================
// COMPLEXITY MODIFIERS
// =============================================================================

const COMPLEXITY_MODIFIERS: Record<Complexity, string> = {
  simple: `
COMPLEXITY: Simple
- Focus on ONE core concept only
- Remove all non-essential elements
- Maximum clarity for quick understanding
- Suitable for young learners or quick review`,

  standard: `
COMPLEXITY: Standard
- 2-3 key concepts shown
- Balanced detail level
- Accessible to general audience
- Good for initial learning`,

  detailed: `
COMPLEXITY: Detailed
- Multiple interconnected concepts
- Rich visual detail
- Secondary elements included
- For deeper understanding`,

  expert: `
COMPLEXITY: Expert
- Maximum information density
- Professional/academic level detail
- Assumes foundational knowledge
- Reference-quality depth`
};

// =============================================================================
// TEXT MODE INSTRUCTIONS
// =============================================================================

const TEXT_INSTRUCTIONS: Record<TextMode, (lesson: LessonContext) => string> = {
  none: () => `
TEXT: None
- DO NOT include any text, labels, or annotations
- Pure visual communication only
- Suitable for overlay or social sharing`,

  labels: (lesson) => `
TEXT: Key Labels Only
Include these labels in the image:
${lesson.keyTerms?.slice(0, 5).map(t => `- "${t}"`).join('\n') || '- Key concept labels as appropriate'}
- Use clear, legible sans-serif font
- High contrast against background
- Labels should point to relevant elements`,

  full: (lesson) => `
TEXT: Full Educational Text
Include in the image:
- Title: "${lesson.topic}"
- 2-3 key labels pointing to elements
- Brief explanatory caption (10-15 words)
- Use clear, professional typography
- Hierarchy: Title > Labels > Caption`,

  bilingual: (lesson) => `
TEXT: Bilingual (English + Spanish)
Include labels in BOTH languages:
- Title: "${lesson.topic}" / "[Spanish translation]"
- Labels in format: "English / Español"
- Clear separation between languages
- Equal visual weight for both`
};

// =============================================================================
// PHASE-SPECIFIC CONTENT
// =============================================================================

const PHASE_CONTENT: Record<Phase, (lesson: LessonContext) => string> = {
  hook: (lesson) => `
PHASE: Opening Hook
Purpose: Spark curiosity, create "Wait, what?!" moment

Content focus:
- Topic: "${lesson.topic}"
- Hint at (don't reveal): ${lesson.universalTruth.substring(0, 100)}
- Create visual mystery or intrigue
- Make viewers want to learn more`,

  cliff: (lesson) => `
PHASE: Cliffhanger
Purpose: Deepen the mystery, show tension

Content focus:
- Topic: "${lesson.topic}"
- Show contrast between expectation and reality
- Visual tension that needs resolution
- Hint at surprising truth`,

  fact1: (lesson) => `
PHASE: First Key Concept
Purpose: Clear teaching of foundational idea

Content focus:
- Topic: "${lesson.topic}"
- Core concept: ${lesson.funFacts?.[0] || lesson.universalTruth}
- Maximum clarity - viewer should understand at a glance
- Foundation for deeper learning`,

  fact2: (lesson) => `
PHASE: Deeper Understanding
Purpose: Build on foundation with more detail

Content focus:
- Topic: "${lesson.topic}"
- Deeper concept: ${lesson.funFacts?.[1] || 'Advanced understanding'}
- Show relationships and connections
- Layer complexity appropriately`,

  fact3: (lesson) => `
PHASE: Wow Moment
Purpose: The surprising detail that makes it memorable

Content focus:
- Topic: "${lesson.topic}"
- Wow factor: ${lesson.wowMoment || lesson.funFacts?.[2] || 'Most surprising aspect'}
- Maximum visual impact
- The "mind-blown" revelation`,

  wisdom: (lesson) => `
PHASE: Life Application
Purpose: Universal truth, poster-worthy wisdom

Content focus:
- Topic: "${lesson.topic}"
- Universal truth: ${lesson.universalTruth}
- Connect to everyday life
- Timeless wisdom worth remembering`,

  outro: (lesson) => `
PHASE: Celebration & Closure
Purpose: Mark completion, inspire continuation

Content focus:
- Topic: "${lesson.topic}"
- Sense of achievement and growth
- Forward-looking energy
- "What's next" feeling`,

  complete: (lesson) => `
PHASE: Complete Summary
Purpose: One image that captures everything

Content focus:
- Topic: "${lesson.topic}"
- Universal truth: ${lesson.universalTruth}
- Reference multiple key concepts
- Shareable, memorable, comprehensive`
};

// =============================================================================
// MAIN PROMPT BUILDER
// =============================================================================

export function buildVariantPrompt(request: VariantRequest): string {
  const { lesson, phase, style, complexity, includesText } = request;
  
  const parts = [
    `Create an educational visual for: "${lesson.topic}"`,
    '',
    STYLE_FOUNDATIONS[style],
    '',
    COMPLEXITY_MODIFIERS[complexity],
    '',
    PHASE_CONTENT[phase](lesson),
    '',
    TEXT_INSTRUCTIONS[includesText](lesson),
    '',
    'IMPORTANT GUIDELINES:',
    '- Educational accuracy is paramount',
    '- Avoid stereotypes and bias',
    '- No copyrighted characters or logos',
    '- Safe for all ages',
    '- Culturally inclusive representation'
  ];
  
  return parts.join('\n');
}

// =============================================================================
// QUICK VARIANT GENERATORS
// =============================================================================

/**
 * Generate prompts for all standard variants of a phase
 */
export function generateStandardVariants(
  lesson: LessonContext, 
  phase: Phase
): Record<VisualStyle, string> {
  const styles: VisualStyle[] = ['artistic', 'textbook', 'diagram', 'minimal'];
  const result: Partial<Record<VisualStyle, string>> = {};
  
  for (const style of styles) {
    result[style] = buildVariantPrompt({
      lesson,
      phase,
      style,
      complexity: 'standard',
      includesText: style === 'textbook' || style === 'diagram' ? 'labels' : 'none'
    });
  }
  
  return result as Record<VisualStyle, string>;
}

/**
 * Generate prompt for textbook-style with full educational text
 */
export function buildTextbookPrompt(lesson: LessonContext, phase: Phase): string {
  return buildVariantPrompt({
    lesson,
    phase,
    style: 'textbook',
    complexity: 'detailed',
    includesText: 'full'
  });
}

/**
 * Generate prompt for medical/scientific diagram
 */
export function buildMedicalPrompt(lesson: LessonContext, phase: Phase): string {
  return buildVariantPrompt({
    lesson,
    phase,
    style: 'medical',
    complexity: 'expert',
    includesText: 'labels'
  });
}

/**
 * Generate prompt for simple, child-friendly visual
 */
export function buildSimplePrompt(lesson: LessonContext, phase: Phase): string {
  return buildVariantPrompt({
    lesson,
    phase,
    style: 'illustrated',
    complexity: 'simple',
    includesText: 'none'
  });
}

// =============================================================================
// HASH GENERATION FOR VARIANTS
// =============================================================================

import * as crypto from 'crypto';

export interface VariantHashContext {
  dayNumber: number;
  phase: Phase;
  style: VisualStyle;
  complexity: Complexity;
  includesText: TextMode;
  ageGroup?: string;
}

export function generateVariantHash(context: VariantHashContext): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    s: context.style,
    c: context.complexity,
    t: context.includesText,
    a: context.ageGroup || 'all',
    ver: '2' // Schema version - increment to invalidate old hashes
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// =============================================================================
// EXAMPLE USAGE
// =============================================================================

/*
const lesson: LessonContext = {
  topic: "Photosynthesis",
  universalTruth: "Plants convert light into energy, forming the foundation of most food chains",
  funFacts: [
    "Chlorophyll gives plants their green color",
    "Plants produce oxygen as a byproduct",
    "One large tree can produce enough oxygen for 4 people daily"
  ],
  keyTerms: ["Chloroplast", "Chlorophyll", "Photosynthesis", "CO₂", "Oxygen"],
  dayNumber: 42
};

// Generate artistic hook
const artisticPrompt = buildVariantPrompt({
  lesson,
  phase: 'hook',
  style: 'artistic',
  complexity: 'standard',
  includesText: 'none'
});

// Generate textbook-style with labels
const textbookPrompt = buildVariantPrompt({
  lesson,
  phase: 'fact1',
  style: 'textbook',
  complexity: 'detailed',
  includesText: 'full'
});

// Generate medical diagram
const medicalPrompt = buildVariantPrompt({
  lesson,
  phase: 'fact2',
  style: 'medical',
  complexity: 'expert',
  includesText: 'labels'
});
*/

