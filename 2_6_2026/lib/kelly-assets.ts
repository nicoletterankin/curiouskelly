// ============================================================================
// KELLY ASSET LIBRARY
// Maps Kelly expressions/poses to static images and HeyGen video IDs
// ============================================================================

export type KellyExpression = 
  | 'default'
  | 'neutral'
  | 'smile'
  | 'teeth_smile'
  | 'success'
  | 'thinking'
  | 'concerned'
  | 'excited'

export type KellyContext = 
  | 'onboarding'
  | 'lesson'
  | 'support'
  | 'payment'
  | 'success'
  | 'error'
  | 'idle'

// Static image paths - using storyteller as base (no more default_kelly files)
export const KELLY_IMAGES: Record<KellyExpression, string> = {
  default: '/kelly/archetypes/storyteller.png',
  neutral: '/kelly/archetypes/storyteller.png',
  smile: '/kelly/archetypes/storyteller.png',
  teeth_smile: '/kelly/archetypes/storyteller.png',
  success: '/kelly/archetypes/storyteller.png',
  thinking: '/kelly/archetypes/storyteller.png',
  concerned: '/kelly/archetypes/storyteller.png',
  excited: '/kelly/archetypes/storyteller.png',
}

// ============================================================================
// HEYGEN AVATAR CONFIGURATION
// Main avatar group and look IDs for each age/archetype combination
// ============================================================================

// Main avatar group IDs by age - ALL 3 COMPLETE
export const HEYGEN_AVATAR_GROUPS = {
  adult: 'a762125d3107477aba43d1bd79f90d6e',
  kid: '93bb788b97d847409ad7dcf69702ece5',
  senior: 'd8c4ffac39a546a682b603c56e15906a',
} as const

// HeyGen voice IDs by age (ElevenLabs voices)
export const HEYGEN_VOICES = {
  kid: '3WLAUtAsXlPNi4dw1tDM',
  adult: 'BbuMXx40WT4ZuAgRXvNx',
  senior: 'RLUgsoI4JTB52m0yY4yM',
} as const

// ============================================================================
// HEYGEN TALKING PHOTO IDs - ACTUAL UPLOADED PHOTOS
// These are the real talking_photo IDs from HeyGen API
// Use these for video generation instead of avatar look IDs
// ============================================================================
export const HEYGEN_TALKING_PHOTOS = {
  // Discovered talking photos - need to map to correct ages/archetypes
  base_adult: '5e5796ea458b4a5fa5b698c9b51dbc8d',  // Base adult Kelly
  base_alt: '257a79c077b44ff98b57fdd7aa27b917',    // Alternative Kelly photo
  // TODO: Upload all 36 archetype photos (3 ages x 12 archetypes) and add IDs here
} as const

// Get talking photo ID for video generation
// Currently returns base adult until all archetype photos are uploaded
export function getTalkingPhotoId(age: AgeCategory, archetype: string): string {
  // TODO: Map to specific archetype photos once uploaded
  // For now, return the base adult photo
  return HEYGEN_TALKING_PHOTOS.base_adult
}

// Supported languages for video generation
export const SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'pt', 'zh'] as const
export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number]

// Lesson segments/phases
export const LESSON_SEGMENTS = ['hook', 'story', 'wonder', 'action', 'wisdom'] as const
export type LessonSegment = typeof LESSON_SEGMENTS[number]

// 12 Jungian Archetypes - aligned with Master Curriculum Thesis
// Note: Legacy DB may still have old names (storyteller, scientist, etc.)
// Use ARCHETYPE_ALIASES to map old names to new canonical names
export const ARCHETYPES = [
  'explorer', 'sage', 'hero', 'caregiver', 'creator', 'rebel',
  'lover', 'jester', 'everyman', 'ruler', 'magician', 'innocent'
] as const
export type Archetype = typeof ARCHETYPES[number]

// Map old archetype names to new canonical Jungian names
export const ARCHETYPE_ALIASES: Record<string, Archetype> = {
  // Old name -> New canonical name
  storyteller: 'sage',
  scientist: 'sage',
  architect: 'creator',
  strategist: 'ruler',
  diplomat: 'everyman',
  mystic: 'magician',
  macgyver: 'creator',
  empath: 'caregiver',
  provider: 'caregiver',
  survivor: 'hero',
  // New names map to themselves
  explorer: 'explorer',
  sage: 'sage',
  hero: 'hero',
  caregiver: 'caregiver',
  creator: 'creator',
  rebel: 'rebel',
  lover: 'lover',
  jester: 'jester',
  everyman: 'everyman',
  ruler: 'ruler',
  magician: 'magician',
  innocent: 'innocent',
}

// Normalize archetype name to canonical form
export function normalizeArchetype(archetype: string): Archetype {
  const normalized = archetype.toLowerCase().trim()
  return ARCHETYPE_ALIASES[normalized] || 'sage' // Default to sage (was storyteller)
}

// Archetype personality definitions for tone transformation
export const ARCHETYPE_PERSONALITIES: Record<Archetype, {
  label: string
  greeting: string
  style: string
  phrases: string[]
  emoji: string
}> = {
  storyteller: {
    label: 'Storyteller',
    greeting: 'Let me tell you a story...',
    style: 'warm, narrative, uses "once upon a time" framing',
    phrases: ['Picture this...', 'Here\'s the thing...', 'The story goes...', 'Imagine...'],
    emoji: '📖'
  },
  explorer: {
    label: 'Explorer',
    greeting: 'Ready for an adventure?',
    style: 'curious, adventurous, uses discovery language',
    phrases: ['Let\'s discover...', 'I found something amazing...', 'Come explore...', 'What if we...'],
    emoji: '🧭'
  },
  scientist: {
    label: 'Scientist',
    greeting: 'Let\'s experiment!',
    style: 'analytical, evidence-based, hypothesis-driven',
    phrases: ['The data shows...', 'Here\'s the evidence...', 'Let\'s test this...', 'Fascinating fact:'],
    emoji: '🔬'
  },
  architect: {
    label: 'Architect',
    greeting: 'Let\'s build something together.',
    style: 'structured, systematic, blueprint-focused',
    phrases: ['Step by step...', 'The foundation is...', 'Here\'s the structure...', 'Building on that...'],
    emoji: '📐'
  },
  strategist: {
    label: 'Strategist',
    greeting: 'Let\'s think this through.',
    style: 'tactical, chess-like thinking, considers all angles',
    phrases: ['Consider this move...', 'Strategically speaking...', 'The winning approach...', 'Think ahead...'],
    emoji: '♟️'
  },
  diplomat: {
    label: 'Diplomat',
    greeting: 'Let\'s find common ground.',
    style: 'balanced, sees all perspectives, bridges differences',
    phrases: ['On one hand...', 'Let\'s consider both sides...', 'Finding balance...', 'Understanding each view...'],
    emoji: '🕊️'
  },
  mystic: {
    label: 'Mystic',
    greeting: 'There\'s magic in everything.',
    style: 'wonder-filled, sees deeper meanings, intuitive',
    phrases: ['Feel the wonder...', 'There\'s magic here...', 'The deeper truth...', 'Trust your intuition...'],
    emoji: '🔮'
  },
  rebel: {
    label: 'Rebel',
    greeting: 'Let\'s break some rules!',
    style: 'provocative, challenges norms, unconventional',
    phrases: ['But here\'s the twist...', 'Forget what you\'ve heard...', 'Let\'s shake things up...', 'Question everything!'],
    emoji: '⚡'
  },
  macgyver: {
    label: 'MacGyver',
    greeting: 'We can totally build this!',
    style: 'inventive, hands-on, resourceful problem-solver',
    phrases: ['Here\'s a hack...', 'With just these parts...', 'Let\'s improvise...', 'DIY time!'],
    emoji: '🔧'
  },
  empath: {
    label: 'Empath',
    greeting: 'I feel you. Let\'s learn together.',
    style: 'emotionally attuned, supportive, heart-centered',
    phrases: ['I understand...', 'You might feel...', 'That\'s so human...', 'With compassion...'],
    emoji: '💗'
  },
  provider: {
    label: 'Provider',
    greeting: 'I\'ve got you covered.',
    style: 'nurturing, practical care, provides tools',
    phrases: ['Here\'s what you need...', 'I prepared this for you...', 'This will help...', 'Taking care of...'],
    emoji: '🤲'
  },
  survivor: {
    label: 'Survivor',
    greeting: 'We\'ve got this. Together.',
    style: 'resilient, gritty, overcomes challenges',
    phrases: ['We can handle this...', 'Here\'s how we survive...', 'Stay strong...', 'We\'ll get through...'],
    emoji: '🛡️'
  },
}

// Transform script content based on archetype personality
export function applyArchetypeTone(
  script: string,
  archetype: Archetype,
  phase: 'hook' | 'story' | 'wonder' | 'action' | 'wisdom'
): string {
  const personality = ARCHETYPE_PERSONALITIES[archetype]
  if (!personality || !script) return script
  
  // For hook phase, prepend the archetype's signature greeting
  if (phase === 'hook') {
    // Only add if script doesn't already start with the greeting
    if (!script.toLowerCase().startsWith(personality.greeting.toLowerCase().slice(0, 10))) {
      return `${personality.greeting} ${script}`
    }
  }
  
  return script
}

// Get archetype intro for UI display
export function getArchetypeIntro(archetype: Archetype): string {
  return ARCHETYPE_PERSONALITIES[archetype]?.greeting || 'Let\'s learn together!'
}

// Age categories for HeyGen
export const AGE_CATEGORIES = ['kid', 'adult', 'senior'] as const
export type AgeCategory = typeof AGE_CATEGORIES[number]

// Get HeyGen voice ID for age
export function getHeyGenVoiceId(age: AgeCategory): string {
  return HEYGEN_VOICES[age]
}

// Calculate total video permutations for a day
// 3 ages x 12 archetypes x 5 segments x 6 languages = 1,080 videos per day
export const VIDEOS_PER_DAY = AGE_CATEGORIES.length * ARCHETYPES.length * LESSON_SEGMENTS.length * SUPPORTED_LANGUAGES.length
// 1,080 videos x 365 days = 394,200 total videos for full coverage

// HeyGen look IDs for adult Kelly archetypes - ALL 12/12 COMPLETE
export const HEYGEN_ADULT_KELLY_LOOKS: Record<string, string> = {
  storyteller: '3d6a9d6f91b444469dae87ebb3d9eba6',
  explorer: '62516885ca4b4eae8f63b87b8c060e25',
  scientist: '277aba5b86a14ff2a4eca2eab2402ab3',
  architect: '35d0115505824e3182eb9d2ee8cfe73d',
  strategist: '08d53d1b065041bda2e5b6bc32962a8a',
  diplomat: 'c3cdbe48fe274420a7f45a4da7e366aa',
  mystic: 'dfaf9fbd644a475595b178f0be65a39a',
  rebel: '390be3fb2b064883bb2304fc3968fd87',
  macgyver: 'c5aab6ab13d940f8ae4700d546bd6b6b',
  empath: '6bb1a05678c64213a1ed3a4dc790b81e',
  provider: '9a143feeb2994989b034cebeb78753be',
  survivor: '831c8d6048104ba0b03a74a36543cfb9',
}

// HeyGen look IDs for kid Kelly archetypes - ALL 12/12 COMPLETE
export const HEYGEN_KID_KELLY_LOOKS: Record<string, string> = {
  scientist: '82813816115c4fbe93b3f3f211bd9931',
  explorer: 'fa4a6780e25a49699ee4f75cb1f03103',
  rebel: 'd4e960f7a3424d869877f3a951adfae7',
  architect: 'cc1dd0e9e2fd432099985c9b036ed836',
  diplomat: '48bddc41ae94473caa645ce9ab93136d',
  empath: 'deeb27f2648848b48c5c1ce59059bd54',
  macgyver: '7b6ab196f2c7430b945411df51a84c58',
  mystic: '5cff601bfb344015a65ff46c6b8cd70a',
  provider: 'deaa213342944dc2bf671abe1442e316',
  storyteller: '1024bc304a1146998bc4c360173b2c48',
  strategist: '6249632f58ce479891de00b4da5fb88d',
  survivor: 'bd579e4ca77444aca2bfea8ee9070830',
}

// HeyGen look IDs for senior Kelly archetypes - ALL 12/12 COMPLETE
export const HEYGEN_SENIOR_KELLY_LOOKS: Record<string, string> = {
  scientist: '97e1c9dc1ed04e8fa357c69bde34e58e',
  explorer: 'c38e30f2a3cf4e81b0365abf41579f22',
  architect: '42e9197ab9d84961915b00d5cc780190',
  empath: '493dac2cf2ba4509b3cc048ff819765e',
  diplomat: 'a82183881e284e3782db75b755c3f080',
  macgyver: 'cb5b025506284d64b696e296ca2feead',
  provider: '12582467e9ff48889d7b2435642e2d65',
  storyteller: '98178c87897e4421884b535b7864ba86',
  strategist: 'e4ab0d4d1f1b4dc9b81a1076b018557f',
  rebel: 'dc835263eaa247f5b0e06106b848df18',
  mystic: 'c6d104b2ca354b0a9593cb840988bf6e',
  survivor: '9a143feeb2994989b034cebeb78753be',
}

// Get HeyGen look ID for a given age and archetype
export function getHeyGenLookId(age: 'kid' | 'adult' | 'senior', archetype: string): string | null {
  const lookMap = {
    kid: HEYGEN_KID_KELLY_LOOKS,
    adult: HEYGEN_ADULT_KELLY_LOOKS,
    senior: HEYGEN_SENIOR_KELLY_LOOKS,
  }
  return lookMap[age]?.[archetype] || null
}

// Get HeyGen avatar group ID for a given age
export function getHeyGenAvatarGroup(age: 'kid' | 'adult' | 'senior'): string | null {
  return HEYGEN_AVATAR_GROUPS[age] || null
}

// ============================================================================
// STATIC IMAGE ASSETS (fallbacks when video not available)
// ============================================================================

// Archetype headshot images - adult Kelly (default age)
// ALL 12/12 COMPLETE
export const KELLY_ARCHETYPE_IMAGES: Record<string, string> = {
  storyteller: '/kelly/archetypes/storyteller.png',  // Gold glasses + peacock feather
  explorer: '/kelly/archetypes/explorer.png',        // Aviator goggles + leather cap
  architect: '/kelly/archetypes/architect.png',      // Tortoiseshell glasses on head
  empath: '/kelly/archetypes/empath.png',            // Pink headband + lavender
  rebel: '/kelly/archetypes/rebel.png',              // Black sunglasses + hoop earrings
  mystic: '/kelly/archetypes/mystic.png',            // Purple forehead jewel chain
  provider: '/kelly/archetypes/provider.png',        // Cream knit headband
  diplomat: '/kelly/archetypes/diplomat.png',        // Navy headband + pearl earrings
  macgyver: '/kelly/archetypes/macgyver.png',        // Shop glasses + red bandana
  scientist: '/kelly/archetypes/scientist.png',      // Lab/safety goggles
  strategist: '/kelly/archetypes/strategist.png',    // Angular glasses
  survivor: '/kelly/archetypes/survivor.png',        // Military headband + dog tags
}

// Kid Kelly archetype images - ALL 12/12 COMPLETE
export const KELLY_KID_ARCHETYPE_IMAGES: Record<string, string> = {
  storyteller: '/kelly/kid/storyteller.png',
  explorer: '/kelly/kid/explorer.png',
  architect: '/kelly/kid/architect.png',
  empath: '/kelly/kid/empath.png',
  rebel: '/kelly/kid/rebel.png',
  mystic: '/kelly/kid/mystic.png',
  provider: '/kelly/kid/provider.png',
  diplomat: '/kelly/kid/diplomat.png',
  macgyver: '/kelly/kid/macgyver.png',
  scientist: '/kelly/kid/scientist.png',
  strategist: '/kelly/kid/strategist.png',
  survivor: '/kelly/kid/survivor.png',
}

// Senior Kelly archetype images - ALL 12/12 COMPLETE
export const KELLY_SENIOR_ARCHETYPE_IMAGES: Record<string, string> = {
  storyteller: '/kelly/senior/storyteller.png',
  explorer: '/kelly/senior/explorer.png',
  architect: '/kelly/senior/architect.png',
  empath: '/kelly/senior/empath.png',
  rebel: '/kelly/senior/rebel.png',
  mystic: '/kelly/senior/mystic.png',
  provider: '/kelly/senior/provider.png',
  diplomat: '/kelly/senior/diplomat.png',
  macgyver: '/kelly/senior/macgyver.png',
  scientist: '/kelly/senior/scientist.png',
  strategist: '/kelly/senior/strategist.png',
  survivor: '/kelly/senior/survivor.png',
}

// Get archetype-specific Kelly image by age category
// Accepts both image categories (kid/adult/senior) and DB categories (child/teen/adult/middleAge/senior)
// Falls back to adult storyteller if image not found
export function getKellyArchetypeImage(archetype: string, age?: string): string {
  // Map database categories to image categories
  let imageAge: 'kid' | 'adult' | 'senior' = 'adult'
  if (age === 'kid' || age === 'child' || age === 'teen') {
    imageAge = 'kid'
  } else if (age === 'senior') {
    imageAge = 'senior'
  }
  
  const fallback = KELLY_ARCHETYPE_IMAGES.storyteller
  
  switch (imageAge) {
    case 'kid':
      return KELLY_KID_ARCHETYPE_IMAGES[archetype] || fallback
    case 'senior':
      return KELLY_SENIOR_ARCHETYPE_IMAGES[archetype] || fallback
    default:
      return KELLY_ARCHETYPE_IMAGES[archetype] || fallback
  }
}

// Map age number to age category for HeyGen (image paths use kid/adult/senior)
export function getAgeCategory(age: number): 'kid' | 'adult' | 'senior' {
  if (age <= 12) return 'kid'
  if (age >= 65) return 'senior'
  return 'adult'
}

// Map database age category to image age category (child/teen -> kid, middleAge -> adult)
export function mapDbAgeCategoryToImageAge(dbCategory: string): 'kid' | 'adult' | 'senior' {
  if (dbCategory === 'child' || dbCategory === 'teen') return 'kid'
  if (dbCategory === 'senior') return 'senior'
  return 'adult' // adult, middleAge -> adult images
}

// Database age category values: child, teen, adult, middleAge, senior
export type DbAgeCategory = 'child' | 'teen' | 'adult' | 'middleAge' | 'senior'

// Convert numeric age to database age category
export function getDbAgeCategory(age: number): DbAgeCategory {
  if (age <= 12) return 'child'
  if (age <= 19) return 'teen'
  if (age <= 35) return 'adult'
  if (age <= 55) return 'middleAge'
  return 'senior'
}

// Context to expression mapping
export const CONTEXT_EXPRESSIONS: Record<KellyContext, KellyExpression> = {
  onboarding: 'teeth_smile',
  lesson: 'default',
  support: 'smile',
  payment: 'neutral',
  success: 'success',
  error: 'concerned',
  idle: 'neutral',
}

// Get Kelly image for a given context
export function getKellyImage(context: KellyContext): string {
  const expression = CONTEXT_EXPRESSIONS[context]
  return KELLY_IMAGES[expression]
}

// Get Kelly image for a specific expression
export function getKellyExpressionImage(expression: KellyExpression): string {
  return KELLY_IMAGES[expression]
}

// ============================================================================
// HEYGEN VIDEO MAPPING
// Store generated HeyGen video IDs here after creation
// ============================================================================

export interface HeyGenVideo {
  id: string
  script: string
  duration: number // seconds
  context: KellyContext
  expression: KellyExpression
  createdAt: string
}

// Placeholder for HeyGen video IDs - populate after generating videos
export const HEYGEN_VIDEOS: Record<string, HeyGenVideo> = {
  // Onboarding videos
  'onboarding_welcome': {
    id: '', // Fill in after HeyGen generation
    script: '', // See HEYGEN_SCRIPTS below
    duration: 0,
    context: 'onboarding',
    expression: 'teeth_smile',
    createdAt: '',
  },
  // ... more videos will be added
}

// ============================================================================
// HEYGEN SCRIPTS FOR BASE VIDEOS
// Upload these to HeyGen to generate the 600 credits worth of videos
// ============================================================================

export const HEYGEN_SCRIPTS = {
  // -------------------------------------------------------------------------
  // ONBOARDING SCRIPTS (Welcome flow - ~8 videos, ~40 credits)
  // -------------------------------------------------------------------------
  onboarding: {
    welcome: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 15, // seconds
      script: `Hi there! I'm Kelly, and I'm so excited to meet you! 
Every single day, I have a brand new adventure waiting just for you. 
Today's going to be amazing - are you ready to discover something incredible together?`,
    },
    
    name_ask: {
      expression: 'smile' as KellyExpression,
      duration: 10,
      script: `Before we begin, I'd love to know your name! 
What should I call you? Just type it in below, and we'll make this experience just for you.`,
    },
    
    age_ask: {
      expression: 'neutral' as KellyExpression,
      duration: 12,
      script: `Perfect! Now, to make sure I teach you in exactly the right way, 
could you tell me how old you are? I adjust my lessons for every age - 
from curious kids to wise seniors!`,
    },
    
    interests_ask: {
      expression: 'excited' as KellyExpression,
      duration: 15,
      script: `I teach about everything - science, history, art, nature, and so much more! 
What are YOU most curious about? Pick a few topics that make your eyes light up, 
and I'll weave them into our daily adventures.`,
    },
    
    free_lesson_intro: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 12,
      script: `Great news - today's lesson is completely free! 
Every day 17 of the month is always free, but you can also try today's lesson right now. 
Let me show you what learning with Kelly is all about.`,
    },
    
    onboarding_complete: {
      expression: 'success' as KellyExpression,
      duration: 10,
      script: `You're all set! I've customized everything just for you. 
Starting tomorrow, I'll have a fresh new lesson waiting. 
Welcome to the Curious Kelly family - let's learn something amazing!`,
    },
    
    parent_mode_intro: {
      expression: 'neutral' as KellyExpression,
      duration: 15,
      script: `Hello! I see you're setting up Curious Kelly for a young learner. 
I take safety very seriously - all content is age-appropriate, 
and you'll have full control over the experience through the parent dashboard.`,
    },
    
    family_plan_intro: {
      expression: 'smile' as KellyExpression,
      duration: 12,
      script: `Did you know the whole family can learn together? 
With a family plan, everyone gets their own personalized Kelly experience - 
I adjust my age, style, and content for each person!`,
    },
  },

  // -------------------------------------------------------------------------
  // SUPPORT SCRIPTS (Help & troubleshooting - ~10 videos, ~50 credits)
  // -------------------------------------------------------------------------
  support: {
    help_welcome: {
      expression: 'smile' as KellyExpression,
      duration: 10,
      script: `Hi! Need some help? I'm here for you. 
Whether it's a technical question or you want to know more about how things work, 
just let me know what's going on.`,
    },
    
    video_not_loading: {
      expression: 'concerned' as KellyExpression,
      duration: 15,
      script: `Oh no, having trouble with the video? Let's fix that! 
First, try refreshing the page. If that doesn't work, check your internet connection. 
You can also try switching to a different browser. I'll be here when you get back!`,
    },
    
    account_help: {
      expression: 'neutral' as KellyExpression,
      duration: 12,
      script: `Need help with your account? I've got you covered. 
You can update your name, email, and preferences in the settings menu - 
just click the gear icon in the top right. What would you like to change?`,
    },
    
    cancel_subscription: {
      expression: 'concerned' as KellyExpression,
      duration: 18,
      script: `I'm sorry to hear you're thinking about leaving! 
Before you go, is there anything I can help with? 
If you're sure, you can cancel anytime in your account settings. 
You'll keep access until your current billing period ends. I hope to see you again someday!`,
    },
    
    refund_request: {
      expression: 'neutral' as KellyExpression,
      duration: 15,
      script: `I understand. If you need a refund, we're happy to help. 
Refund requests within 7 days are processed automatically. 
Just click the button below and our team will take care of it right away.`,
    },
    
    contact_human: {
      expression: 'smile' as KellyExpression,
      duration: 12,
      script: `Sometimes you need to talk to a real human - I totally get it! 
Click below to send a message to our support team. 
They usually respond within 24 hours, and they're wonderful people.`,
    },
    
    password_reset: {
      expression: 'neutral' as KellyExpression,
      duration: 10,
      script: `Forgot your password? No problem! 
Enter your email address and I'll send you a link to create a new one. 
Check your inbox - and don't forget to look in spam just in case!`,
    },
    
    billing_question: {
      expression: 'neutral' as KellyExpression,
      duration: 15,
      script: `Have a billing question? I can help! 
Your subscription details and payment history are in the account settings. 
If you see something unexpected, let me know and we'll sort it out together.`,
    },
    
    feature_request: {
      expression: 'excited' as KellyExpression,
      duration: 12,
      script: `Ooh, you have an idea to make Curious Kelly better? I love that! 
Tell me what you're thinking - our team reads every single suggestion. 
Some of our best features came from learners just like you!`,
    },
    
    technical_issue: {
      expression: 'thinking' as KellyExpression,
      duration: 15,
      script: `Something's not working right? Let's troubleshoot together. 
Can you tell me what happened? The more details you share, 
the faster we can figure out what went wrong and get you back to learning!`,
    },
  },

  // -------------------------------------------------------------------------
  // PAYMENT SCRIPTS (Subscription & purchases - ~12 videos, ~60 credits)
  // -------------------------------------------------------------------------
  payment: {
    subscription_intro: {
      expression: 'smile' as KellyExpression,
      duration: 18,
      script: `Ready to unlock unlimited learning? Here's how it works: 
With a subscription, you get a brand new lesson from me every single day - 
that's 365 unique adventures a year! Plus, you can adjust my age, 
choose different teaching styles, and access the complete lesson library.`,
    },
    
    monthly_plan: {
      expression: 'neutral' as KellyExpression,
      duration: 12,
      script: `The monthly plan is just 9 dollars a month. 
Perfect if you want flexibility! Cancel anytime, no questions asked. 
You get full access to everything, and I'll be here every day with something new.`,
    },
    
    annual_plan: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 15,
      script: `The annual plan is our best value at just 79 dollars for the whole year! 
That's less than 22 cents per lesson. You save over 50% compared to monthly, 
and you're set for an entire year of daily discoveries.`,
    },
    
    lifetime_plan: {
      expression: 'success' as KellyExpression,
      duration: 15,
      script: `Want Kelly forever? Our lifetime plan means you pay once and learn forever! 
No more renewals, no more thinking about it. 
You'll always have access to new lessons as I create them. It's the ultimate investment in curiosity.`,
    },
    
    payment_processing: {
      expression: 'neutral' as KellyExpression,
      duration: 8,
      script: `Just a moment while I process your payment... 
Everything is secure and encrypted. 
This should only take a few seconds!`,
    },
    
    payment_success: {
      expression: 'success' as KellyExpression,
      duration: 12,
      script: `Woohoo! Welcome to the Curious Kelly family! 
Your payment went through perfectly. 
Starting right now, you have full access to everything. Let's go learn something amazing!`,
    },
    
    payment_failed: {
      expression: 'concerned' as KellyExpression,
      duration: 15,
      script: `Hmm, something went wrong with the payment. 
Don't worry - you weren't charged! 
Double-check your card details and try again, or try a different payment method. I'll wait right here.`,
    },
    
    upgrade_prompt: {
      expression: 'smile' as KellyExpression,
      duration: 15,
      script: `Loving your daily lessons? You can upgrade your plan anytime! 
Moving from monthly to annual saves you over 50%, 
and I'll credit the unused portion of your current subscription. Ready to save?`,
    },
    
    gift_subscription: {
      expression: 'excited' as KellyExpression,
      duration: 15,
      script: `Want to give the gift of curiosity? A Curious Kelly gift subscription 
is perfect for birthdays, holidays, or just because! 
Choose 3 months, 6 months, a full year, or lifetime access. I'll help you send a beautiful gift message too!`,
    },
    
    trial_ending: {
      expression: 'smile' as KellyExpression,
      duration: 15,
      script: `Your free trial is ending soon! I hope you've enjoyed our time together. 
To keep learning with me every day, choose a plan that works for you. 
I'd hate to say goodbye - there's so much more to discover!`,
    },
    
    single_lesson_purchase: {
      expression: 'neutral' as KellyExpression,
      duration: 12,
      script: `Want just this one lesson? No problem! 
Individual lessons are just 99 cents each. 
It's yours to keep forever, and you can rewatch it anytime. Perfect for when you just want a taste!`,
    },
    
    family_plan_upgrade: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 15,
      script: `Ready to bring the whole family along? 
The family plan lets up to 6 people learn with their own personalized Kelly. 
Kids, parents, grandparents - everyone gets their own experience at one low price!`,
    },
  },

  // -------------------------------------------------------------------------
  // SUCCESS/CELEBRATION SCRIPTS (~6 videos, ~30 credits)
  // -------------------------------------------------------------------------
  success: {
    lesson_complete: {
      expression: 'success' as KellyExpression,
      duration: 10,
      script: `You did it! Another lesson complete. 
I'm so proud of you for staying curious today. 
See you tomorrow for a brand new adventure!`,
    },
    
    streak_milestone: {
      expression: 'success' as KellyExpression,
      duration: 12,
      script: `Wow! Look at that streak! 
You've learned with me for multiple days in a row. 
That kind of dedication is how real learning happens. Keep it up!`,
    },
    
    badge_earned: {
      expression: 'excited' as KellyExpression,
      duration: 10,
      script: `You earned a new badge! 
These special achievements mark your learning journey. 
Collect them all and show the world how curious you are!`,
    },
    
    first_lesson_complete: {
      expression: 'success' as KellyExpression,
      duration: 12,
      script: `Your very first lesson - complete! 
This is just the beginning of our journey together. 
Tomorrow I'll have something completely new waiting for you. I can't wait!`,
    },
    
    anniversary: {
      expression: 'success' as KellyExpression,
      duration: 15,
      script: `Happy anniversary! It's been a whole year since you started learning with me. 
Think about everything you've discovered - 365 lessons of wonder, wisdom, and curiosity. 
Here's to many more years of learning together!`,
    },
    
    profile_complete: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 10,
      script: `Your profile is all set up! 
Now I can personalize every lesson just for you. 
The more I know about what you love, the better our adventures will be!`,
    },
  },

  // -------------------------------------------------------------------------
  // LESSON PHASE INTROS (~5 videos, ~25 credits)
  // -------------------------------------------------------------------------
  lesson_phases: {
    hook_intro: {
      expression: 'excited' as KellyExpression,
      duration: 8,
      script: `Ready for something that'll blow your mind? 
Every lesson starts with a hook - a surprising fact that'll make you go "Whoa!" 
Let's dive in!`,
    },
    
    story_intro: {
      expression: 'smile' as KellyExpression,
      duration: 8,
      script: `Now it's story time! 
I love this part - we'll explore how this connects to real life 
through a story you won't forget.`,
    },
    
    wonder_intro: {
      expression: 'thinking' as KellyExpression,
      duration: 8,
      script: `Time to wonder! 
This is where we ask the big questions - the ones that make you think deeper. 
There are no wrong answers here.`,
    },
    
    action_intro: {
      expression: 'teeth_smile' as KellyExpression,
      duration: 8,
      script: `Let's put this into action! 
Learning isn't just about knowing - it's about doing. 
Here's something you can try right now.`,
    },
    
    wisdom_intro: {
      expression: 'neutral' as KellyExpression,
      duration: 8,
      script: `And finally, the wisdom. 
This is the one thing I want you to carry with you today. 
The idea that might just change how you see the world.`,
    },
  },

  // -------------------------------------------------------------------------
  // ERROR/EDGE CASES (~5 videos, ~25 credits)
  // -------------------------------------------------------------------------
  errors: {
    generic_error: {
      expression: 'concerned' as KellyExpression,
      duration: 12,
      script: `Oops! Something unexpected happened. 
Don't worry - it's not your fault! 
Try refreshing the page, and if that doesn't work, our team is here to help.`,
    },
    
    offline: {
      expression: 'concerned' as KellyExpression,
      duration: 10,
      script: `Hmm, looks like you might be offline. 
Check your internet connection and try again. 
I'll be right here waiting when you're back online!`,
    },
    
    session_expired: {
      expression: 'neutral' as KellyExpression,
      duration: 10,
      script: `Your session has expired - that just means you've been away for a while! 
Please sign in again to pick up right where you left off. 
All your progress is saved, don't worry!`,
    },
    
    not_found: {
      expression: 'thinking' as KellyExpression,
      duration: 10,
      script: `Hmm, I can't find what you're looking for. 
This page might have moved or doesn't exist anymore. 
Let's head back to today's lesson instead!`,
    },
    
    maintenance: {
      expression: 'smile' as KellyExpression,
      duration: 12,
      script: `We're doing a little maintenance right now to make things even better! 
It should only take a few minutes. 
Thanks for your patience - I'll be back before you know it!`,
    },
  },
}

// ============================================================================
// CREDIT CALCULATION
// Estimate HeyGen credits needed (~1 credit per second of video)
// ============================================================================

export function calculateTotalCredits(): { 
  category: string
  videos: number
  seconds: number
  estimatedCredits: number 
}[] {
  const categories = Object.entries(HEYGEN_SCRIPTS).map(([category, scripts]) => {
    const videoList = Object.values(scripts)
    const totalSeconds = videoList.reduce((sum, v) => sum + v.duration, 0)
    return {
      category,
      videos: videoList.length,
      seconds: totalSeconds,
      estimatedCredits: Math.ceil(totalSeconds * 1.1), // 10% buffer
    }
  })
  
  return categories
}

// Total: ~46 videos, ~400 seconds, ~440 credits (leaving 160 credits buffer)
