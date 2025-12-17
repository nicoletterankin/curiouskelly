/**
 * PHASE-ALIGNED VISUAL PROMPTS
 * 
 * Generates prompts that are deeply integrated with lesson content:
 * - Aligned with specific quiz questions and answers
 * - Incorporates misconceptions and corrections
 * - Uses learning objectives as visual goals
 * - Creates answer-illustration visuals
 * 
 * @created December 17, 2025
 */

import * as crypto from 'crypto';

// =============================================================================
// TYPES
// =============================================================================

export interface QuizQuestion {
  question: string;
  options: string[];
  correct: string;
}

export interface Misconception {
  misconception: string;
  correction: string;
}

export interface FullLessonContext {
  // Core
  day_number: number;
  topic: string;
  universal_truth: string;
  wow_moment: string;
  
  // Teaching content
  fun_facts: string[];
  extended_explanation: string;
  learning_objectives: string[];
  
  // Q&A
  quick_quiz_questions: QuizQuestion[];
  discussion_questions: string[];
  
  // Misconceptions
  common_misconceptions: Misconception[];
  
  // Applications
  real_world_applications: string[];
  
  // Marketing (great for hooks)
  marketing_headline: string;
  marketing_pitch: string;
}

export type Phase = 'hook' | 'cliff' | 'fact1' | 'fact2' | 'fact3' | 'wisdom' | 'outro' | 'complete';
export type Style = 'artistic' | 'textbook' | 'diagram' | 'minimal' | 'infographic' | 'comparison';
export type Complexity = 'simple' | 'standard' | 'detailed' | 'expert';

// =============================================================================
// STYLE FOUNDATIONS
// =============================================================================

const STYLE_FOUNDATIONS: Record<Style, string> = {
  artistic: `
VISUAL STYLE: Ultra Photorealistic Cinematic
- Professional photography, dramatic lighting
- Emotional resonance and visual storytelling
- Warm, inviting color palette
- 16:9 aspect ratio, 4K quality
- Leave right 30% simpler for text overlay
DO NOT include any text, logos, or watermarks.`,

  textbook: `
VISUAL STYLE: Educational Textbook Illustration
- Professional educational illustration quality
- Clean, organized visual hierarchy
- Light background (white or cream)
- Include clear text labels and annotations
- 16:9 aspect ratio, print-ready quality

TEXT ELEMENTS:
- Title at top with topic name
- 3-5 clear labels pointing to key elements
- Brief explanatory caption if helpful
- Use clear sans-serif fonts, high contrast`,

  diagram: `
VISUAL STYLE: Technical Educational Diagram
- Clean, precise technical drawing
- Blueprint or schematic aesthetic
- Numbered/lettered components with legend
- Arrows showing flow, relationships, causation
- 16:9 aspect ratio

TEXT ELEMENTS:
- Component labels (1, 2, 3 or A, B, C)
- Key term labels with leader lines
- Directional arrows with brief annotations
- Clean technical typography`,

  minimal: `
VISUAL STYLE: Minimalist Concept
- Ultra-clean modern design
- Maximum 3 colors
- Single central concept
- Generous negative space
- Elegant simplicity
- 16:9 aspect ratio
DO NOT include any text.`,

  infographic: `
VISUAL STYLE: Bold Infographic Design
- Eye-catching data visualization
- Clear visual hierarchy with icons
- Statistics displayed prominently
- Modern color scheme
- Information-dense but organized
- 16:9 aspect ratio

TEXT TO INCLUDE:
- Headline with topic
- 2-3 key statistics or facts
- Visual hierarchy showing most important info first
- Icons representing key concepts`,

  comparison: `
VISUAL STYLE: Split Comparison Visual
- Side-by-side or before/after composition
- Clear visual contrast between two states
- Labels identifying each side
- Visual metaphors for difference
- 16:9 aspect ratio

TEXT ELEMENTS:
- Labels for each side
- "vs" or dividing element
- Key difference callouts`
};

// =============================================================================
// PHASE-SPECIFIC PROMPT BUILDERS
// =============================================================================

interface PhasePromptData {
  purpose: string;
  content: string;
  visualDirective: string;
  criticalElement: string;
  alignedQuestion?: QuizQuestion;
  alignedFact?: string;
  alignedObjective?: string;
}

function extractHookData(lesson: FullLessonContext): PhasePromptData {
  const misconception = lesson.common_misconceptions?.[0];
  
  return {
    purpose: `Create CURIOSITY and cognitive tension.
This is the OPENING HOOK - viewers should think "Wait, what?!"
Make them NEED to learn more.`,

    content: `TOPIC: "${lesson.topic}"

ATTENTION GRABBER:
"${lesson.marketing_headline}"

COMMON MISCONCEPTION (hint that this might be wrong):
"${misconception?.misconception || 'Common assumptions about ' + lesson.topic}"

THE SURPRISE TO HINT AT (don't reveal, just tease):
"${lesson.wow_moment}"`,

    visualDirective: `Create visual MYSTERY. Show what people commonly believe,
but hint that something unexpected is about to be revealed.
The viewer should feel curious, intrigued, questioning.
This is the "before the twist" moment in a story.`,

    criticalElement: 'Must create curiosity WITHOUT giving away the answer',
    alignedFact: lesson.marketing_headline
  };
}

function extractCliffData(lesson: FullLessonContext): PhasePromptData {
  const misconception = lesson.common_misconceptions?.[0];
  
  return {
    purpose: `Deepen the MYSTERY and show CONTRAST.
This is the "But wait..." moment - the plot twist is coming.`,

    content: `TOPIC: "${lesson.topic}"

WHAT PEOPLE THINK (the misconception):
"${misconception?.misconception || 'Common belief about ' + lesson.topic}"

WHAT'S ACTUALLY TRUE (the correction):
"${misconception?.correction || lesson.universal_truth}"

THE TENSION:
Show the gap between expectation and reality.`,

    visualDirective: `Create a SPLIT composition or visual CONTRAST.
LEFT/TOP: What people commonly believe
RIGHT/BOTTOM: Hints at the surprising truth
Show the tension between these two states.
The viewer should be leaning in, ready for the reveal.`,

    criticalElement: 'Must show contrast between misconception and truth'
  };
}

function extractFact1Data(lesson: FullLessonContext): PhasePromptData {
  const q1 = lesson.quick_quiz_questions?.[0];
  const fact1 = lesson.fun_facts?.[0];
  const obj1 = lesson.learning_objectives?.[0];
  
  return {
    purpose: `TEACH the first key concept with crystal clarity.
This is FOUNDATIONAL teaching - understanding is everything.`,

    content: `TOPIC: "${lesson.topic}"

KEY FACT TO ILLUSTRATE:
"${fact1}"

${q1 ? `THIS VISUAL SHOULD ANSWER:
"${q1.question}"

THE CORRECT ANSWER IS:
"${q1.correct}"

WRONG OPTIONS (do NOT illustrate these):
${q1.options.filter(o => o !== q1.correct).map(o => `- "${o}"`).join('\n')}` : ''}

LEARNING OBJECTIVE:
"${obj1}"`,

    visualDirective: `Make the CORRECT ANSWER visually OBVIOUS.
A learner should be able to answer the quiz question
just by studying this image carefully.
The key concept should be unmistakably clear.`,

    criticalElement: 'The correct answer MUST be clearly illustrated',
    alignedQuestion: q1,
    alignedFact: fact1,
    alignedObjective: obj1
  };
}

function extractFact2Data(lesson: FullLessonContext): PhasePromptData {
  const q2 = lesson.quick_quiz_questions?.[1];
  const fact2 = lesson.fun_facts?.[1];
  const obj2 = lesson.learning_objectives?.[1];
  
  return {
    purpose: `Go DEEPER into the concept.
Build on the foundation with more detail and connections.`,

    content: `TOPIC: "${lesson.topic}"

DEEPER FACT TO ILLUSTRATE:
"${fact2}"

${q2 ? `THIS VISUAL SHOULD ANSWER:
"${q2.question}"

THE CORRECT ANSWER IS:
"${q2.correct}"` : ''}

CONTEXT (from extended explanation):
"${lesson.extended_explanation?.substring(0, 300)}..."`,

    visualDirective: `Show RELATIONSHIPS and CONNECTIONS.
This visual should reveal the "why" behind the "what".
More detail than Fact1, showing how concepts connect.
Use visual hierarchy to show cause → effect.`,

    criticalElement: 'Must show relationships between concepts',
    alignedQuestion: q2,
    alignedFact: fact2,
    alignedObjective: obj2
  };
}

function extractFact3Data(lesson: FullLessonContext): PhasePromptData {
  const q3 = lesson.quick_quiz_questions?.[2];
  const fact3 = lesson.fun_facts?.[2] || lesson.fun_facts?.[3];
  
  return {
    purpose: `Create the WOW MOMENT - the surprising revelation.
This is what makes the lesson memorable and shareable.`,

    content: `TOPIC: "${lesson.topic}"

THE WOW MOMENT:
"${lesson.wow_moment}"

SUPPORTING FACT:
"${fact3}"

${q3 ? `THIS ANSWERS:
"${q3.question}"
CORRECT: "${q3.correct}"` : ''}`,

    visualDirective: `Create a MIND-BLOWN moment.
This is the image someone would screenshot and share.
Maximum visual impact. The "I had no idea!" revelation.
Make it dramatic, memorable, shareable.`,

    criticalElement: 'Must create a "wow" reaction',
    alignedQuestion: q3,
    alignedFact: lesson.wow_moment
  };
}

function extractWisdomData(lesson: FullLessonContext): PhasePromptData {
  const application = lesson.real_world_applications?.[0];
  const discussion = lesson.discussion_questions?.[0];
  
  return {
    purpose: `Life APPLICATION - connect knowledge to real life.
This is poster-worthy WISDOM that stays with the learner.`,

    content: `TOPIC: "${lesson.topic}"

UNIVERSAL TRUTH:
"${lesson.universal_truth}"

REAL-WORLD APPLICATION:
"${application}"

REFLECTION QUESTION:
"${discussion}"`,

    visualDirective: `Create something INSPIRATIONAL and TIMELESS.
This should feel like a poster worth hanging on your wall.
Connect the abstract concept to human experience.
Show how this wisdom applies to everyday life.`,

    criticalElement: 'Must connect learning to real life',
    alignedFact: lesson.universal_truth,
    alignedObjective: application
  };
}

function extractOutroData(lesson: FullLessonContext): PhasePromptData {
  return {
    purpose: `CELEBRATE completion and point forward.
This marks achievement with energy for what's next.`,

    content: `TOPIC: "${lesson.topic}"

WHAT WAS LEARNED:
${lesson.learning_objectives?.slice(0, 2).map(o => `- ${o}`).join('\n')}

FORWARD MOMENTUM:
The learner is now ready to apply this knowledge.`,

    visualDirective: `Create a sense of ACHIEVEMENT and FORWARD ENERGY.
Celebratory but not cheesy.
Hint at "what's next" - the journey continues.
The feeling of growth and readiness.`,

    criticalElement: 'Must feel celebratory and forward-looking'
  };
}

function extractCompleteData(lesson: FullLessonContext): PhasePromptData {
  return {
    purpose: `COMPREHENSIVE summary of the entire lesson.
This is the ONE image that captures everything.`,

    content: `TOPIC: "${lesson.topic}"

UNIVERSAL TRUTH:
"${lesson.universal_truth}"

LEARNING OBJECTIVES ACHIEVED:
${lesson.learning_objectives?.map((o, i) => `${i + 1}. ${o}`).join('\n')}

KEY FACTS:
${lesson.fun_facts?.slice(0, 3).map((f, i) => `• ${f}`).join('\n')}

WOW MOMENT:
"${lesson.wow_moment}"`,

    visualDirective: `Create a REFERENCE-QUALITY comprehensive visual.
This should capture the ENTIRE lesson at a glance.
Include visual references to multiple key concepts.
Suitable for printing, sharing, studying from.
The ultimate summary image.`,

    criticalElement: 'Must comprehensively represent the full lesson'
  };
}

// =============================================================================
// MAIN PROMPT BUILDER
// =============================================================================

const PHASE_EXTRACTORS: Record<Phase, (lesson: FullLessonContext) => PhasePromptData> = {
  hook: extractHookData,
  cliff: extractCliffData,
  fact1: extractFact1Data,
  fact2: extractFact2Data,
  fact3: extractFact3Data,
  wisdom: extractWisdomData,
  outro: extractOutroData,
  complete: extractCompleteData
};

export function buildPhaseAlignedPrompt(
  lesson: FullLessonContext,
  phase: Phase,
  style: Style
): { prompt: string; metadata: PhasePromptData } {
  
  const extractor = PHASE_EXTRACTORS[phase];
  const phaseData = extractor(lesson);
  const styleFoundation = STYLE_FOUNDATIONS[style];
  
  const prompt = `Create an educational visual for: "${lesson.topic}"

${styleFoundation}

═══════════════════════════════════════════════════════════
PHASE: ${phase.toUpperCase()}
═══════════════════════════════════════════════════════════

PURPOSE:
${phaseData.purpose}

CONTENT TO VISUALIZE:
${phaseData.content}

VISUAL DIRECTIVE:
${phaseData.visualDirective}

═══════════════════════════════════════════════════════════

CRITICAL REQUIREMENT:
${phaseData.criticalElement}

GUIDELINES:
- Educational accuracy is paramount
- This visual is specifically for the ${phase.toUpperCase()} phase
- It must serve this exact moment in the learning journey
- No copyrighted characters or logos
- Safe for all ages, culturally inclusive`;

  return { prompt, metadata: phaseData };
}

// =============================================================================
// SPECIALIZED PROMPT BUILDERS
// =============================================================================

/**
 * Build a prompt for answer-illustration visual
 * Shows the correct answer to a quiz question visually
 */
export function buildAnswerIllustrationPrompt(
  lesson: FullLessonContext,
  question: QuizQuestion,
  style: Style = 'diagram'
): string {
  const wrongOptions = question.options.filter(o => o !== question.correct);
  
  return `Create an educational visual that ILLUSTRATES THE ANSWER to:

QUESTION: "${question.question}"

CORRECT ANSWER: "${question.correct}"

${STYLE_FOUNDATIONS[style]}

VISUAL REQUIREMENTS:
- The correct answer concept must be VISUALLY OBVIOUS
- A viewer should understand why "${question.correct}" is correct
- The image should TEACH, not just decorate

WRONG ANSWERS (for context, do NOT illustrate):
${wrongOptions.map(w => `- "${w}"`).join('\n')}

The goal: Someone could answer this question correctly
just by studying this image carefully.

Topic: "${lesson.topic}"`;
}

/**
 * Build a prompt for misconception-correction visual
 * Shows the contrast between wrong belief and truth
 */
export function buildMisconceptionPrompt(
  lesson: FullLessonContext,
  misconception: Misconception,
  style: Style = 'comparison'
): string {
  return `Create an educational visual showing MISCONCEPTION vs REALITY:

${STYLE_FOUNDATIONS[style]}

MISCONCEPTION (what people wrongly believe):
"${misconception.misconception}"

CORRECTION (what's actually true):
"${misconception.correction}"

VISUAL APPROACH:
- Split composition: LEFT shows the misconception, RIGHT shows truth
- Use visual metaphors to show why the misconception is wrong
- The correction should be visually more prominent/positive
- Include subtle "X" or "✓" indicators if appropriate

Topic: "${lesson.topic}"
Make the truth visually compelling and memorable.`;
}

/**
 * Build a prompt for data/statistic visualization
 */
export function buildDataVisualizationPrompt(
  lesson: FullLessonContext,
  fact: string,
  style: Style = 'infographic'
): string {
  // Extract any numbers from the fact
  const numbers = fact.match(/\d+%?/g) || [];
  
  return `Create a DATA VISUALIZATION for:

"${fact}"

${STYLE_FOUNDATIONS[style]}

KEY STATISTICS TO HIGHLIGHT:
${numbers.map(n => `- ${n}`).join('\n') || '- The key quantitative insight from the fact'}

VISUALIZATION APPROACH:
- Make numbers visually impactful
- Use comparison, scale, or proportion to show significance
- The data should tell a story
- Include clear labels and context

Topic: "${lesson.topic}"
Make the data memorable and shareable.`;
}

// =============================================================================
// HASH GENERATION
// =============================================================================

export interface PhaseAlignedHashContext {
  dayNumber: number;
  phase: Phase;
  style: Style;
  complexity?: Complexity;
  version?: string;
}

export function generatePhaseAlignedHash(context: PhaseAlignedHashContext): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    s: context.style,
    c: context.complexity || 'standard',
    ver: context.version || '3' // v3 for phase-aligned prompts
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  STYLE_FOUNDATIONS,
  PHASE_EXTRACTORS
};
