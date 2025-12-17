/**
 * Visual Prompts V2 - Curious Kelly Educational Illustration System
 * 
 * Key improvements over V1:
 * - ONE consistent illustrated style (not photorealistic)
 * - NEVER request text in images (AI fails at text)
 * - SPECIFIC visual subjects (not vague "show the concept")
 * - Consistent composition with UI-safe zones
 * - Quality validation built-in
 */

import * as crypto from 'crypto';

// ============================================================================
// TYPES
// ============================================================================

export type Phase = 'hook' | 'cliff' | 'q1' | 'q2' | 'q3' | 'wisdom' | 'outro';

export interface LessonContext {
  dayNumber: number;
  topic: string;
  hookTeaser: string;
  cliffChoice: string;
  q1Content: string;
  q2Content: string;
  q3Content: string;
  wisdomInsight: string;
  funFacts: string[];
  wowMoment: string;
}

export interface VisualSubject {
  phase: Phase;
  subject: string;        // What to actually show
  mood: string;           // Emotional tone
  keyElements: string[];  // Specific elements to include
}

export interface GeneratedPrompt {
  prompt: string;
  phase: Phase;
  contentHash: string;
  expectedDimensions: { width: number; height: number };
}

// ============================================================================
// STYLE CONSTANTS - The Curious Kelly Look
// ============================================================================

const CURIOUS_KELLY_STYLE = `
STYLE: Modern Educational Illustration
- Clean flat illustration with subtle depth and soft shadows
- Warm, friendly color palette: teals (#4ECDC4), corals (#FF6B6B), warm yellows (#FFE66D), soft purples (#A78BFA)
- Clean lines, approachable and friendly aesthetic
- Stylized but not cartoonish, professional educational quality
- Soft, even, welcoming lighting
- Think: Headspace, Duolingo, Khan Academy visual style
`.trim();

const COMPOSITION_RULES = `
COMPOSITION:
- 16:9 aspect ratio exactly
- Main subject takes 50-70% of frame
- LEFT 70% contains primary visual content
- RIGHT 30% is simpler (reserved for UI overlay)
- Clean, uncluttered background
- Generous whitespace and breathing room
- No busy patterns competing with subject
`.trim();

const UNIVERSAL_CONSTRAINTS = `
CRITICAL REQUIREMENTS:
- DO NOT include ANY text, labels, numbers, letters, or writing
- DO NOT include watermarks, signatures, or logos
- DO NOT include realistic photographs of real people
- Keep it appropriate for all ages (family-friendly)
- Avoid culturally specific or religious imagery
`.trim();

// ============================================================================
// PHASE-SPECIFIC TEMPLATES
// ============================================================================

const PHASE_TEMPLATES: Record<Phase, { purpose: string; moodGuidance: string }> = {
  hook: {
    purpose: 'Spark curiosity, create a "wait, what?" moment of intrigue',
    moodGuidance: 'Slightly mysterious, intriguing, wonder-inducing'
  },
  cliff: {
    purpose: 'Show tension, choice, or contrast that creates anticipation',
    moodGuidance: 'Visual tension between two elements, decision moment'
  },
  q1: {
    purpose: 'Clearly illustrate the first key concept',
    moodGuidance: 'Clear, educational, enlightening, approachable'
  },
  q2: {
    purpose: 'Deepen understanding with the second concept',
    moodGuidance: 'Building, layered, showing progression'
  },
  q3: {
    purpose: 'Challenge or surprise with the third concept',
    moodGuidance: 'Surprising, eye-opening, "aha moment"'
  },
  wisdom: {
    purpose: 'Inspire with timeless, universal truth',
    moodGuidance: 'Peaceful, inspiring, golden-hour warmth, aspirational'
  },
  outro: {
    purpose: 'Celebrate completion and create forward momentum',
    moodGuidance: 'Celebratory, energetic, forward-looking, accomplished'
  }
};

// ============================================================================
// SUBJECT EXTRACTION - Convert lesson content to specific visuals
// ============================================================================

/**
 * Extract specific visual subjects from lesson content
 * This is the key improvement - we generate CONCRETE visuals, not abstract concepts
 */
export function extractVisualSubjects(lesson: LessonContext): VisualSubject[] {
  return [
    {
      phase: 'hook',
      subject: generateHookSubject(lesson),
      mood: 'intriguing and wonder-inducing',
      keyElements: extractKeyElements(lesson.hookTeaser)
    },
    {
      phase: 'cliff',
      subject: generateCliffSubject(lesson),
      mood: 'tension between two possibilities',
      keyElements: ['contrast', 'choice', 'decision point']
    },
    {
      phase: 'q1',
      subject: generateFactSubject(lesson.q1Content, lesson.funFacts[0] || ''),
      mood: 'clear and educational',
      keyElements: extractKeyElements(lesson.q1Content)
    },
    {
      phase: 'q2',
      subject: generateFactSubject(lesson.q2Content, lesson.funFacts[1] || ''),
      mood: 'building and layered',
      keyElements: extractKeyElements(lesson.q2Content)
    },
    {
      phase: 'q3',
      subject: generateFactSubject(lesson.q3Content, lesson.funFacts[2] || lesson.wowMoment),
      mood: 'surprising aha moment',
      keyElements: extractKeyElements(lesson.q3Content)
    },
    {
      phase: 'wisdom',
      subject: generateWisdomSubject(lesson),
      mood: 'peaceful and inspiring',
      keyElements: ['universal truth', 'timeless', 'aspirational']
    },
    {
      phase: 'outro',
      subject: generateOutroSubject(lesson),
      mood: 'celebratory and forward-looking',
      keyElements: ['achievement', 'momentum', 'growth']
    }
  ];
}

function generateHookSubject(lesson: LessonContext): string {
  // Convert abstract hook teaser into concrete visual
  const topic = lesson.topic.toLowerCase();
  
  // Generic but effective hook visual patterns
  return `A curious scene related to "${lesson.topic}" that immediately grabs attention - show the surprising or unexpected aspect of ${lesson.hookTeaser.substring(0, 100)}. Use visual metaphor rather than literal interpretation.`;
}

function generateCliffSubject(lesson: LessonContext): string {
  return `A split or contrasting scene showing two perspectives on "${lesson.topic}" - visualize the tension between what people commonly believe versus the surprising reality. Show a clear visual fork in the road or comparison.`;
}

function generateFactSubject(questionContent: string, fact: string): string {
  const combined = questionContent + ' ' + fact;
  return `An educational illustration explaining a key concept: ${combined.substring(0, 150)}. Show the mechanism or relationship clearly through visual metaphor - make the abstract concrete and understandable.`;
}

function generateWisdomSubject(lesson: LessonContext): string {
  return `An inspiring, peaceful scene that embodies the wisdom of "${lesson.topic}" - ${lesson.wisdomInsight.substring(0, 100)}. Show a universal moment of realization, growth, or possibility. Timeless and aspirational.`;
}

function generateOutroSubject(lesson: LessonContext): string {
  return `A celebratory scene showing accomplishment and forward momentum after learning about "${lesson.topic}". A figure or symbol of achievement taking the next step forward with confidence and energy.`;
}

function extractKeyElements(text: string): string[] {
  // Extract nouns and key concepts from text
  const words = text.toLowerCase().split(/\s+/);
  const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these', 'those']);
  
  return words
    .filter(w => w.length > 4 && !stopWords.has(w))
    .slice(0, 5);
}

// ============================================================================
// PROMPT GENERATION
// ============================================================================

/**
 * Build a complete prompt for a visual subject
 */
export function buildPromptV2(subject: VisualSubject, lesson: LessonContext): GeneratedPrompt {
  const template = PHASE_TEMPLATES[subject.phase];
  
  const prompt = `
Create an illustrated educational scene for a lesson about "${lesson.topic}".

PURPOSE: ${template.purpose}

SUBJECT TO ILLUSTRATE:
${subject.subject}

KEY VISUAL ELEMENTS TO INCLUDE:
${subject.keyElements.map(e => `- ${e}`).join('\n')}

MOOD: ${subject.mood}

${CURIOUS_KELLY_STYLE}

${COMPOSITION_RULES}

${UNIVERSAL_CONSTRAINTS}
`.trim();

  const contentHash = generateContentHash({
    dayNumber: lesson.dayNumber,
    phase: subject.phase,
    topic: lesson.topic,
    version: 'v2'
  });

  return {
    prompt,
    phase: subject.phase,
    contentHash,
    expectedDimensions: { width: 1920, height: 1080 }
  };
}

/**
 * Generate all prompts for a lesson
 */
export function generateAllPromptsV2(lesson: LessonContext): GeneratedPrompt[] {
  const subjects = extractVisualSubjects(lesson);
  return subjects.map(subject => buildPromptV2(subject, lesson));
}

// ============================================================================
// CONTENT HASH - For deduplication and caching
// ============================================================================

interface HashInput {
  dayNumber: number;
  phase: Phase;
  topic: string;
  version: string;
}

export function generateContentHash(input: HashInput): string {
  const canonical = JSON.stringify({
    d: input.dayNumber,
    p: input.phase,
    t: input.topic.toLowerCase().trim(),
    v: input.version
  });
  
  return crypto
    .createHash('sha256')
    .update(canonical)
    .digest('hex');
}

// ============================================================================
// QUALITY VALIDATION
// ============================================================================

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateGeneratedImage(
  imageBuffer: Buffer,
  expectedDimensions: { width: number; height: number }
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  // Check file size
  const sizeKB = imageBuffer.length / 1024;
  if (sizeKB < 50) {
    errors.push(`Image too small (${sizeKB.toFixed(1)}KB) - likely failed generation`);
  }
  if (sizeKB > 5000) {
    warnings.push(`Image very large (${sizeKB.toFixed(1)}KB) - consider compression`);
  }
  
  // Check PNG header for dimensions (basic check)
  if (imageBuffer[0] === 0x89 && imageBuffer[1] === 0x50) {
    // PNG file - extract dimensions from IHDR chunk
    const width = imageBuffer.readUInt32BE(16);
    const height = imageBuffer.readUInt32BE(20);
    
    const aspectRatio = width / height;
    const expectedAspectRatio = expectedDimensions.width / expectedDimensions.height;
    
    if (Math.abs(aspectRatio - expectedAspectRatio) > 0.1) {
      errors.push(`Wrong aspect ratio: ${aspectRatio.toFixed(2)} (expected ~${expectedAspectRatio.toFixed(2)})`);
    }
    
    if (width < 1000) {
      warnings.push(`Low resolution width: ${width}px`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export const STYLE_GUIDE = {
  colors: {
    primaryTeal: '#4ECDC4',
    coralAccent: '#FF6B6B',
    warmYellow: '#FFE66D',
    softPurple: '#A78BFA',
    deepNavy: '#1A1A2E',
    creamBackground: '#FFF9F0',
    forestGreen: '#27AE60'
  },
  dimensions: {
    width: 1920,
    height: 1080,
    aspectRatio: '16:9'
  },
  safeZone: {
    leftContent: 0.7,  // Left 70% for main content
    rightOverlay: 0.3  // Right 30% simpler for UI
  }
};
