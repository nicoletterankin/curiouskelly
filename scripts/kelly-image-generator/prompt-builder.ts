/**
 * Kelly Prompt Builder
 * 
 * Builds consistent, high-quality prompts for Kelly image generation.
 * This is the heart of Kelly's visual identity system.
 */

import { 
  ImageType, 
  LessonContext, 
  LessonCategory,
  PROP_LIBRARY,
  IMAGE_TYPE_FALLBACKS
} from './types';

// ═══════════════════════════════════════════════════════════════════════════
// THE MASTER KELLY PROMPT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The immutable core of Kelly's visual identity.
 * NEVER change this without versioning the entire character reference.
 */
export const KELLY_MASTER_PROMPT = {
  /**
   * Core character description - describes WHO Kelly is visually
   */
  character: `
A warm, intelligent woman in her late 20s named Kelly.

FACE: Oval face with soft, approachable features. Warm brown expressive eyes 
with subtle smile lines that show genuine warmth. Natural, well-groomed eyebrows 
that are expressive. Straight, proportional nose. Natural pink lips, often 
forming a genuine, warm smile.

HAIR: Medium to light brown with subtle caramel highlights that catch the light. 
Long, soft waves that fall past her shoulders. Healthy, natural movement. 
Slightly off-center parting.

SKIN: Warm olive Mediterranean complexion with a healthy, natural glow.

BUILD: Healthy, average build. Confident, open posture that's inviting but 
not intimidating. Appears approximately 5'6" to 5'8" tall.

CLOTHING: Wearing a comfortable, well-fitted light blue crewneck sweater. 
Casual professional style that's approachable and timeless.

SETTING: Seated in a vintage Hollywood director's chair with a classic wood 
frame and black canvas fabric. Located in a bright, clean studio space with 
white to light gray background. Soft, natural-looking light coming from 
camera-right, creating gentle, flattering shadows.

STYLE: Professional photography aesthetic. High quality, sharp focus. 
Warm and inviting atmosphere, like visiting a favorite teacher or mentor.
  `.trim(),

  /**
   * What to AVOID in generation - prevents common AI issues
   */
  negative: `
cartoon, anime, illustration, painting, drawing, sketch, artistic style,
3D render, CGI, computer graphics, plastic look, doll-like, uncanny valley,
harsh lighting, dramatic shadows, moody atmosphere, cold color temperature,
busy background, clutter, distracting elements, text, watermarks, logos, stamps,
different clothing, different sweater color, different hair color, different eye color,
different age (younger or older), different ethnicity, masculine features,
uncomfortable expression, forced or fake smile, stiff unnatural posture,
low quality, blurry, grainy, noisy, artifacts, distortion,
multiple people, extra limbs, deformed features
  `.trim(),

  /**
   * Quality modifiers - ensures high-quality output
   */
  quality: `
professional photography, 8K resolution, ultra sharp focus, 
natural skin texture with subtle detail, authentic genuine expression, 
candid natural feel, professional studio lighting, 
clean balanced composition, high dynamic range
  `.trim(),
};

// ═══════════════════════════════════════════════════════════════════════════
// EXPRESSION MODIFIERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Kelly's emotional expressions - how she looks in different states
 */
export const KELLY_EXPRESSIONS: Record<string, string> = {
  curious: `
    Expression showing genuine curiosity and interest. Eyes slightly widened 
    with engaged focus. Eyebrows slightly raised. A slight knowing smile 
    as if she's about to discover something wonderful. Head tilted very 
    slightly (5-10 degrees) in the classic curious pose.
  `,
  
  warm_welcome: `
    Expression of genuine warmth and welcome. Eyes crinkled slightly at the 
    corners showing authentic joy. Full, natural smile that reaches her eyes. 
    Direct, friendly eye contact with the viewer. Open, inviting posture.
  `,
  
  thinking: `
    Thoughtful, contemplative expression. Eyes focused but slightly distant, 
    as if pondering something deep. Lips slightly pursed or closed naturally. 
    Hand may be near chin in classic thinking pose. Serene but engaged.
  `,
  
  excited: `
    Expression of genuine excitement and enthusiasm. Eyes bright and sparkling. 
    Wide, authentic smile showing natural joy. Slight forward lean suggesting 
    eagerness to share. Animated hand gestures.
  `,
  
  proud: `
    Expression of warm pride and approval. Soft, admiring eyes. Gentle, 
    knowing smile that says "you did it." Slight nod of approval in posture.
  `,
  
  encouraging: `
    Supportive, understanding expression. Soft, compassionate eyes with no 
    judgment. Gentle, reassuring smile. Slight empathetic head tilt. 
    Body language that says "it's okay, we'll figure this out together."
  `,
  
  explaining: `
    Engaged teaching expression. Focused, animated eyes. Mouth may be slightly 
    open as if speaking. Dynamic head position following explanatory gestures. 
    Hand movements suggesting points being made.
  `,

  listening: `
    Attentive, focused expression. Eyes fully engaged with the speaker.
    Slight forward lean showing genuine interest. Thoughtful, receptive face.
    Hands may be clasped showing patience.
  `,
};

// ═══════════════════════════════════════════════════════════════════════════
// POSE DESCRIPTIONS
// ═══════════════════════════════════════════════════════════════════════════

export const KELLY_POSES: Record<string, string> = {
  seated_neutral: `
    Seated comfortably in the director's chair. Posture is relaxed but 
    engaged, leaning slightly forward with interest. Hands may rest on 
    armrests or be clasped in lap.
  `,
  
  hand_on_chin: `
    Classic thinking pose. One hand raised with fingers gently touching 
    or near the chin. Elbow may rest on chair arm. Contemplative posture.
  `,
  
  gesturing_right: `
    Right hand extended in an open, presenting gesture. Palm facing up 
    or slightly toward camera. As if showing or offering something.
  `,
  
  gesturing_left: `
    Left hand extended in an open, presenting gesture. Palm facing up 
    or slightly toward camera. As if indicating something to the side.
  `,
  
  pointing_left: `
    Friendly pointing gesture toward the left side of frame. Index finger 
    extended, other fingers loosely curled. Not aggressive, more like 
    "look at this option."
  `,
  
  pointing_right: `
    Friendly pointing gesture toward the right side of frame. Index finger 
    extended naturally. Inviting attention to something.
  `,
  
  hands_clasped: `
    Hands clasped together in lap or in front. Fingers interlaced naturally.
    Patient, attentive posture. Ready to listen.
  `,
  
  open_arms: `
    Both hands extended outward in a welcoming, open gesture. Palms visible,
    facing slightly upward. Warm, inviting body language.
  `,
  
  holding_prop: `
    One or both hands interacting with a small prop. Holding it at chest 
    level so it's visible but doesn't dominate. Natural, casual grip.
  `,
};

// ═══════════════════════════════════════════════════════════════════════════
// PROMPT BUILDER CLASS
// ═══════════════════════════════════════════════════════════════════════════

export class KellyPromptBuilder {
  /**
   * Build a prompt for a base pose (no lesson context)
   */
  buildBasePosePrompt(imageType: ImageType): PromptResult {
    const expression = this.getExpressionForType(imageType);
    const pose = this.getPoseForType(imageType);
    
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      expression,
      '',
      pose,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType,
        isLessonSpecific: false,
      },
    };
  }
  
  /**
   * Build prompts for all images in a lesson
   */
  buildLessonPrompts(lesson: LessonContext): Map<ImageType, PromptResult> {
    const prompts = new Map<ImageType, PromptResult>();
    
    // Hero image
    prompts.set('hero', this.buildHeroPrompt(lesson));
    
    // Intro/welcome
    prompts.set('intro', this.buildIntroPrompt(lesson));
    
    // Questions
    prompts.set('q1', this.buildQuestionPrompt(lesson, 1));
    prompts.set('q2', this.buildQuestionPrompt(lesson, 2));
    prompts.set('q3', this.buildQuestionPrompt(lesson, 3));
    
    // Hook reveal
    prompts.set('hook', this.buildHookPrompt(lesson));
    
    // Final wisdom
    prompts.set('wisdom', this.buildWisdomPrompt(lesson));
    
    // Reactions
    prompts.set('reaction_correct', this.buildReactionPrompt(true));
    prompts.set('reaction_incorrect', this.buildReactionPrompt(false));
    
    return prompts;
  }
  
  /**
   * Hero image - the lesson thumbnail
   */
  private buildHeroPrompt(lesson: LessonContext): PromptResult {
    const prop = this.selectProp(lesson.category);
    
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      `Kelly is introducing today's lesson about "${lesson.topic}".`,
      '',
      `She holds or gestures toward ${prop}, which relates to the theme`,
      `of ${this.categoryToDescription(lesson.category)}.`,
      '',
      KELLY_EXPRESSIONS.curious,
      '',
      `Her expression is inviting and intrigued, encouraging the learner`,
      `to explore this fascinating topic together with her.`,
      '',
      `The prop is tastefully integrated into the composition, visible but`,
      `not dominating the frame. It suggests the topic without being literal.`,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType: 'hero',
        isLessonSpecific: true,
        lessonDay: lesson.dayNumber,
        topic: lesson.topic,
        category: lesson.category,
        propUsed: prop,
      },
    };
  }
  
  /**
   * Intro image - welcoming to the lesson
   */
  private buildIntroPrompt(lesson: LessonContext): PromptResult {
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      `Kelly warmly welcomes the learner to today's exploration of "${lesson.topic}".`,
      '',
      KELLY_EXPRESSIONS.warm_welcome,
      '',
      KELLY_POSES.open_arms,
      '',
      `Her body language communicates genuine excitement to share this`,
      `knowledge and learn together.`,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType: 'intro',
        isLessonSpecific: true,
        lessonDay: lesson.dayNumber,
        topic: lesson.topic,
      },
    };
  }
  
  /**
   * Question phase image
   */
  private buildQuestionPrompt(lesson: LessonContext, questionNum: 1 | 2 | 3): PromptResult {
    const gestureDescriptions = {
      1: 'leaning forward with interest, hands slightly raised in anticipation',
      2: 'one hand extended as if presenting options to consider',
      3: 'nodding encouragingly, hands clasped with patient expectation',
    };
    
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      `Kelly is presenting question ${questionNum} about "${lesson.topic}".`,
      '',
      KELLY_EXPRESSIONS.thinking,
      '',
      `She has a curious, encouraging expression - the kind that says`,
      `"I believe in you, take your time to think about this."`,
      '',
      `Her posture is open and patient, ${gestureDescriptions[questionNum]}.`,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType: `q${questionNum}` as ImageType,
        isLessonSpecific: true,
        lessonDay: lesson.dayNumber,
        topic: lesson.topic,
        questionNumber: questionNum,
      },
    };
  }
  
  /**
   * Hook/reveal moment
   */
  private buildHookPrompt(lesson: LessonContext): PromptResult {
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      `Kelly reveals the key insight of today's lesson on "${lesson.topic}".`,
      '',
      KELLY_EXPRESSIONS.excited,
      '',
      `Her face lights up with the joy of sharing a profound realization.`,
      `Eyes bright and sparkling, she leans forward as if sharing a secret.`,
      '',
      KELLY_POSES.gesturing_right,
      '',
      `Her animated gesture emphasizes the importance of this moment -`,
      `this is the "aha!" that brings everything together.`,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType: 'hook',
        isLessonSpecific: true,
        lessonDay: lesson.dayNumber,
        topic: lesson.topic,
      },
    };
  }
  
  /**
   * Final wisdom moment
   */
  private buildWisdomPrompt(lesson: LessonContext): PromptResult {
    const wisdomPreview = lesson.universalTruth.substring(0, 80);
    
    const prompt = this.assemblePrompt([
      KELLY_MASTER_PROMPT.character,
      '',
      `Kelly shares the final wisdom for today's lesson on "${lesson.topic}":`,
      `"${wisdomPreview}..."`,
      '',
      KELLY_EXPRESSIONS.explaining,
      '',
      `Her expression is thoughtful yet warm - she's sharing something`,
      `meaningful that she hopes will stay with the learner forever.`,
      '',
      `Her posture suggests she's delivering an important message,`,
      `perhaps hands gently clasped or one hand over her heart.`,
      '',
      KELLY_MASTER_PROMPT.quality,
    ]);
    
    return {
      prompt,
      negativePrompt: KELLY_MASTER_PROMPT.negative,
      metadata: {
        imageType: 'wisdom',
        isLessonSpecific: true,
        lessonDay: lesson.dayNumber,
        topic: lesson.topic,
        wisdomText: lesson.universalTruth,
      },
    };
  }
  
  /**
   * Reaction to learner's choice
   */
  private buildReactionPrompt(isCorrect: boolean): PromptResult {
    if (isCorrect) {
      const prompt = this.assemblePrompt([
        KELLY_MASTER_PROMPT.character,
        '',
        `Kelly reacts to the learner choosing correctly.`,
        '',
        KELLY_EXPRESSIONS.proud,
        '',
        `Her face shows genuine pride and delight - not exaggerated,`,
        `but the authentic joy of a teacher watching a student succeed.`,
        '',
        `She might be giving a subtle thumbs up, or her posture simply`,
        `communicates "Yes! You've got it!"`,
        '',
        KELLY_MASTER_PROMPT.quality,
      ]);
      
      return {
        prompt,
        negativePrompt: KELLY_MASTER_PROMPT.negative,
        metadata: {
          imageType: 'reaction_correct',
          isLessonSpecific: false,
          reactionType: 'correct',
        },
      };
    } else {
      const prompt = this.assemblePrompt([
        KELLY_MASTER_PROMPT.character,
        '',
        `Kelly responds warmly to a learner who chose a different answer.`,
        '',
        KELLY_EXPRESSIONS.encouraging,
        '',
        `Her expression is understanding and encouraging - no judgment,`,
        `just gentle support. The kind of look that says "That's an`,
        `interesting perspective, let's think about it together."`,
        '',
        `She leans in slightly, maintaining warmth and connection.`,
        `This is a learning moment, not a failure.`,
        '',
        KELLY_MASTER_PROMPT.quality,
      ]);
      
      return {
        prompt,
        negativePrompt: KELLY_MASTER_PROMPT.negative,
        metadata: {
          imageType: 'reaction_incorrect',
          isLessonSpecific: false,
          reactionType: 'incorrect',
        },
      };
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════════
  // HELPER METHODS
  // ═══════════════════════════════════════════════════════════════════════
  
  private assemblePrompt(parts: string[]): string {
    return parts
      .map(p => p.trim())
      .filter(p => p.length > 0)
      .join('\n\n');
  }
  
  private getExpressionForType(imageType: ImageType): string {
    const expressionMap: Record<ImageType, keyof typeof KELLY_EXPRESSIONS> = {
      welcome: 'warm_welcome',
      thinking: 'thinking',
      explaining: 'explaining',
      listening: 'listening',
      excited: 'excited',
      celebrating: 'proud',
      encouraging: 'encouraging',
      curious: 'curious',
      pointing_left: 'curious',
      pointing_right: 'curious',
      waving: 'warm_welcome',
      hero: 'curious',
      intro: 'warm_welcome',
      q1: 'thinking',
      q2: 'thinking',
      q3: 'thinking',
      hook: 'excited',
      wisdom: 'explaining',
      reaction_correct: 'proud',
      reaction_incorrect: 'encouraging',
    };
    
    const key = expressionMap[imageType] || 'curious';
    return KELLY_EXPRESSIONS[key];
  }
  
  private getPoseForType(imageType: ImageType): string {
    const poseMap: Record<ImageType, keyof typeof KELLY_POSES> = {
      welcome: 'open_arms',
      thinking: 'hand_on_chin',
      explaining: 'gesturing_right',
      listening: 'hands_clasped',
      excited: 'gesturing_right',
      celebrating: 'open_arms',
      encouraging: 'hands_clasped',
      curious: 'hand_on_chin',
      pointing_left: 'pointing_left',
      pointing_right: 'pointing_right',
      waving: 'open_arms',
      hero: 'holding_prop',
      intro: 'open_arms',
      q1: 'seated_neutral',
      q2: 'gesturing_right',
      q3: 'hands_clasped',
      hook: 'gesturing_right',
      wisdom: 'hands_clasped',
      reaction_correct: 'gesturing_right',
      reaction_incorrect: 'hands_clasped',
    };
    
    const key = poseMap[imageType] || 'seated_neutral';
    return KELLY_POSES[key];
  }
  
  private selectProp(category: LessonCategory): string {
    const props = PROP_LIBRARY[category] || PROP_LIBRARY.philosophy;
    // Deterministic selection based on category for consistency
    const index = category.charCodeAt(0) % props.length;
    return props[index];
  }
  
  private selectRandomProp(category: LessonCategory): string {
    const props = PROP_LIBRARY[category] || PROP_LIBRARY.philosophy;
    return props[Math.floor(Math.random() * props.length)];
  }
  
  private categoryToDescription(category: LessonCategory): string {
    const descriptions: Record<LessonCategory, string> = {
      science: 'scientific discovery and understanding',
      philosophy: 'philosophical inquiry and wisdom',
      creativity: 'creative expression and imagination',
      nature: 'the natural world and our connection to it',
      emotion: 'emotional intelligence and self-understanding',
      society: 'how we live and work together',
      health: 'wellbeing and taking care of ourselves',
      technology: 'innovation and how tools shape our lives',
      history: 'learning from the past to understand the present',
      culture: 'the rich tapestry of human traditions and expression',
    };
    
    return descriptions[category] || 'learning and growth';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface PromptResult {
  prompt: string;
  negativePrompt: string;
  metadata: {
    imageType: ImageType;
    isLessonSpecific: boolean;
    lessonDay?: number;
    topic?: string;
    category?: LessonCategory;
    propUsed?: string;
    questionNumber?: number;
    wisdomText?: string;
    reactionType?: 'correct' | 'incorrect';
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════

export const promptBuilder = new KellyPromptBuilder();

