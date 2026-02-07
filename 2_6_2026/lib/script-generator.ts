// ============================================================================
// KELLY SCRIPT GENERATOR
// Generates personalized scripts for each phase × archetype × age combination
// ============================================================================

import type { Archetype, AgeCategory } from './heygen-client'

// ============================================================================
// TYPES
// ============================================================================

export type Phase = 'hook' | 'story' | 'wonder' | 'action' | 'wisdom'

export interface LessonData {
  dayOfYear: number
  topic: string
  theme: string
  kellyAge: number
  phases: {
    hook: { question: string }
    story: { narrative: string }
    wonder: { exploration: string }
    action: { activity: string }
    wisdom: { quote: string; quoteAuthor: string; reflection: string }
  }
}

export interface GeneratedScript {
  phase: Phase
  archetype: Archetype
  ageCategory: AgeCategory
  script: string
  estimatedDuration: number // seconds
}

// ============================================================================
// ARCHETYPE VOICE PATTERNS
// ============================================================================

const ARCHETYPE_VOICES: Record<Archetype, {
  opening: Record<Phase, string>
  transition: string
  visualRef: string
}> = {
  scientist: {
    opening: {
      hook: "Here's an interesting phenomenon...",
      story: "In my observations one day...",
      wonder: "The data shows us something incredible...",
      action: "Let's run an experiment...",
      wisdom: "The evidence teaches us...",
    },
    transition: "Fascinating, isn't it?",
    visualRef: "Observe this carefully - see the pattern?",
  },
  explorer: {
    opening: {
      hook: "I discovered something amazing!",
      story: "On one of my adventures...",
      wonder: "Here's what I mapped out...",
      action: "Time for an adventure!",
      wisdom: "Every journey reveals...",
    },
    transition: "Ready to explore further?",
    visualRef: "Look what I found!",
  },
  rebel: {
    opening: {
      hook: "Bet you never questioned this...",
      story: "Everyone said it was one way, but I found...",
      wonder: "Here's what they don't teach you...",
      action: "Break the rules a little...",
      wisdom: "The real truth is...",
    },
    transition: "Pretty wild, right?",
    visualRef: "Check this out - it's not what you think.",
  },
  architect: {
    opening: {
      hook: "Let's examine the structure here...",
      story: "I was building something when...",
      wonder: "Let's break down the structure...",
      action: "Follow these steps carefully...",
      wisdom: "When we build with this knowledge...",
    },
    transition: "See how it all connects?",
    visualRef: "Notice how this is built?",
  },
  diplomat: {
    opening: {
      hook: "There's a beautiful balance here...",
      story: "I was talking with someone and...",
      wonder: "Both perspectives reveal...",
      action: "Try this with someone...",
      wisdom: "What connects us all is...",
    },
    transition: "Something we can all appreciate.",
    visualRef: "See how these elements connect?",
  },
  empath: {
    opening: {
      hook: "I felt this too, and wondered...",
      story: "I really felt it when...",
      wonder: "What this really means is...",
      action: "Really feel this...",
      wisdom: "In your heart, remember...",
    },
    transition: "Can you feel it too?",
    visualRef: "Can you feel what's happening here?",
  },
  macgyver: {
    opening: {
      hook: "So I was tinkering and noticed...",
      story: "I was fixing something and noticed...",
      wonder: "Here's how it actually works...",
      action: "Grab these simple materials...",
      wisdom: "With this tool, you can...",
    },
    transition: "You can totally try this yourself!",
    visualRef: "Look at how this works!",
  },
  mystic: {
    opening: {
      hook: "There's something deeper here...",
      story: "In a quiet moment, I sensed...",
      wonder: "The deeper truth is...",
      action: "Take a moment to...",
      wisdom: "The universe whispers...",
    },
    transition: "There's more to discover within.",
    visualRef: "Gaze at this for a moment...",
  },
  provider: {
    opening: {
      hook: "I want to share something helpful...",
      story: "I was taking care of someone when...",
      wonder: "What you need to know is...",
      action: "Here's something helpful to try...",
      wisdom: "Take this with you...",
    },
    transition: "This will serve you well.",
    visualRef: "See this? It affects us every day.",
  },
  storyteller: {
    opening: {
      hook: "Let me tell you about something magical...",
      story: "Once upon a time, I...",
      wonder: "The story the science tells us...",
      action: "Become part of the story...",
      wisdom: "And so the story teaches...",
    },
    transition: "What a tale, right?",
    visualRef: "Picture this scene with me...",
  },
  strategist: {
    opening: {
      hook: "Here's what most people miss...",
      story: "I was planning ahead when...",
      wonder: "The key insight is...",
      action: "Here's the optimal approach...",
      wisdom: "The key takeaway is...",
    },
    transition: "See the bigger picture now?",
    visualRef: "Notice the pattern here?",
  },
  survivor: {
    opening: {
      hook: "This knowledge could change everything...",
      story: "I was in a tough spot when...",
      wonder: "This is what really matters...",
      action: "This skill might come in handy...",
      wisdom: "Above all, remember...",
    },
    transition: "Now you're prepared.",
    visualRef: "Look closely - this matters.",
  },
}

// ============================================================================
// AGE STYLE PATTERNS
// ============================================================================

const AGE_STYLES: Record<AgeCategory, {
  energy: 'high' | 'balanced' | 'gentle'
  sentenceLength: 'short' | 'medium' | 'varied'
  exclamations: boolean
  closingStyle: Record<Phase, string>
}> = {
  kid: {
    energy: 'high',
    sentenceLength: 'short',
    exclamations: true,
    closingStyle: {
      hook: "Pretty cool, right? Let's find out!",
      story: "It was SO cool when I figured it out!",
      wonder: "Isn't that AMAZING?!",
      action: "Ready? Let's do it!",
      wisdom: "Pretty cool, huh?",
    },
  },
  adult: {
    energy: 'balanced',
    sentenceLength: 'medium',
    exclamations: false,
    closingStyle: {
      hook: "Let me show you something fascinating.",
      story: "That moment really stuck with me.",
      wonder: "Elegant once you understand it.",
      action: "Give it a try when you get a chance.",
      wisdom: "Something to carry with you.",
    },
  },
  senior: {
    energy: 'gentle',
    sentenceLength: 'varied',
    exclamations: false,
    closingStyle: {
      hook: "Let me share what I've discovered.",
      story: "Some discoveries stay with you for a lifetime.",
      wonder: "Nature's engineering is quite sophisticated.",
      action: "It's quite satisfying once you try it.",
      wisdom: "May this serve you well.",
    },
  },
}

// ============================================================================
// SCRIPT GENERATOR
// ============================================================================

/**
 * Generate a script for a specific phase, archetype, and age
 */
export function generateScript(
  lesson: LessonData,
  phase: Phase,
  archetype: Archetype,
  ageCategory: AgeCategory
): GeneratedScript {
  const voice = ARCHETYPE_VOICES[archetype]
  const style = AGE_STYLES[ageCategory]
  
  let content: string
  
  switch (phase) {
    case 'hook':
      content = buildHookScript(lesson, voice, style)
      break
    case 'story':
      content = buildStoryScript(lesson, voice, style)
      break
    case 'wonder':
      content = buildWonderScript(lesson, voice, style)
      break
    case 'action':
      content = buildActionScript(lesson, voice, style)
      break
    case 'wisdom':
      content = buildWisdomScript(lesson, voice, style)
      break
  }
  
  // Estimate duration: ~150 words per minute
  const wordCount = content.split(/\s+/).length
  const estimatedDuration = Math.ceil((wordCount / 150) * 60)
  
  return {
    phase,
    archetype,
    ageCategory,
    script: content,
    estimatedDuration,
  }
}

function buildHookScript(
  lesson: LessonData,
  voice: typeof ARCHETYPE_VOICES[Archetype],
  style: typeof AGE_STYLES[AgeCategory]
): string {
  const parts = [
    voice.visualRef,
    voice.opening.hook,
    lesson.phases.hook.question,
    style.closingStyle.hook,
  ]
  
  return formatForAge(parts.join(' '), style)
}

function buildStoryScript(
  lesson: LessonData,
  voice: typeof ARCHETYPE_VOICES[Archetype],
  style: typeof AGE_STYLES[AgeCategory]
): string {
  const parts = [
    voice.opening.story,
    lesson.phases.story.narrative,
    style.closingStyle.story,
  ]
  
  return formatForAge(parts.join(' '), style)
}

function buildWonderScript(
  lesson: LessonData,
  voice: typeof ARCHETYPE_VOICES[Archetype],
  style: typeof AGE_STYLES[AgeCategory]
): string {
  const parts = [
    voice.opening.wonder,
    lesson.phases.wonder.exploration,
    voice.transition,
    style.closingStyle.wonder,
  ]
  
  return formatForAge(parts.join(' '), style)
}

function buildActionScript(
  lesson: LessonData,
  voice: typeof ARCHETYPE_VOICES[Archetype],
  style: typeof AGE_STYLES[AgeCategory]
): string {
  const parts = [
    voice.opening.action,
    lesson.phases.action.activity,
    style.closingStyle.action,
  ]
  
  return formatForAge(parts.join(' '), style)
}

function buildWisdomScript(
  lesson: LessonData,
  voice: typeof ARCHETYPE_VOICES[Archetype],
  style: typeof AGE_STYLES[AgeCategory]
): string {
  const { quote, quoteAuthor, reflection } = lesson.phases.wisdom
  
  const parts = [
    voice.opening.wisdom,
    `As ${quoteAuthor} said: "${quote}"`,
    reflection,
    style.closingStyle.wisdom,
  ]
  
  return formatForAge(parts.join(' '), style)
}

function formatForAge(
  text: string,
  style: typeof AGE_STYLES[AgeCategory]
): string {
  let formatted = text
  
  // Add energy markers for kid voice
  if (style.energy === 'high' && style.exclamations) {
    // Convert some periods to exclamation marks
    formatted = formatted.replace(/\.\s+/g, (match, offset) => {
      // Randomly add exclamation for ~30% of sentences
      return Math.random() > 0.7 ? '! ' : match
    })
  }
  
  // Soften for senior voice
  if (style.energy === 'gentle') {
    formatted = formatted
      .replace(/!/g, '.')
      .replace(/AMAZING/gi, 'remarkable')
      .replace(/SO cool/gi, 'quite fascinating')
  }
  
  return formatted.trim()
}

// ============================================================================
// BATCH GENERATION
// ============================================================================

/**
 * Generate all scripts for a lesson (5 phases × 12 archetypes × 3 ages = 180 scripts)
 */
export function generateAllScriptsForLesson(lesson: LessonData): GeneratedScript[] {
  const phases: Phase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
  const archetypes: Archetype[] = [
    'scientist', 'explorer', 'rebel', 'architect',
    'diplomat', 'empath', 'macgyver', 'mystic',
    'provider', 'storyteller', 'strategist', 'survivor',
  ]
  const ages: AgeCategory[] = ['kid', 'adult', 'senior']
  
  const scripts: GeneratedScript[] = []
  
  for (const phase of phases) {
    for (const archetype of archetypes) {
      for (const age of ages) {
        scripts.push(generateScript(lesson, phase, archetype, age))
      }
    }
  }
  
  return scripts
}

/**
 * Generate scripts for a specific user's preferences
 */
export function generateUserScripts(
  lesson: LessonData,
  userAge: number,
  userArchetype: Archetype
): Record<Phase, GeneratedScript> {
  const ageCategory: AgeCategory = 
    userAge <= 12 ? 'kid' : 
    userAge >= 65 ? 'senior' : 
    'adult'
  
  const phases: Phase[] = ['hook', 'story', 'wonder', 'action', 'wisdom']
  const scripts: Partial<Record<Phase, GeneratedScript>> = {}
  
  for (const phase of phases) {
    scripts[phase] = generateScript(lesson, phase, userArchetype, ageCategory)
  }
  
  return scripts as Record<Phase, GeneratedScript>
}

/**
 * Calculate total scripts for the year
 * 365 days × 5 phases × 12 archetypes × 3 ages = 65,700 scripts
 */
export function getTotalScriptCount(): number {
  return 365 * 5 * 12 * 3
}
