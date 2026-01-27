/**
 * 🎭 Kelly Personas - Shared Types & Data
 * 
 * Central source of truth for all 12 archetypes.
 * Import this in any v0 template that needs archetype data.
 */

// =============================================================================
// TYPES
// =============================================================================

export type PersonaId = 
  | 'scientist' | 'explorer' | 'rebel' | 'architect'
  | 'diplomat' | 'empath' | 'macgyver' | 'mystic'
  | 'provider' | 'storyteller' | 'strategist' | 'survivor';

export type Phase = 'Hook' | 'Fact1' | 'Fact2' | 'Fact3' | 'Wisdom';
export type RuntimePhase = 'welcome' | 'q1' | 'q2' | 'q3' | 'wisdom';
export type Language = 'en' | 'es' | 'fr';
export type AgeBucket = '5-7' | '8-12' | '13-17' | '18-35' | '36-60' | '61+';
export type Tone = 'playful' | 'conversational' | 'reflective';

export interface Persona {
  id: PersonaId;
  name: string;           // "The Scientist"
  icon: string;           // "🔬"
  tagline: string;        // "Data-driven precision"
  description: string;    // "Lab goggles on forehead"
  accessory: string;      // Full accessory description
  expression: string;     // Expression/pose description
  color: string;          // Hex color "#3b82f6"
  images: {
    head: string;         // CDN path to head-only image
    clean: string;        // CDN path to clean image
    prop: string;         // CDN path to prop image
  };
}

export interface LessonAtom {
  id: string;
  core_lesson_id?: string;  // Optional - may not be included in all queries
  archetype: string;
  phase: Phase | string;    // Allow string for flexibility
  content: {
    script?: string;
    kellyPose?: string;
    kellyEmotion?: string;
    optionIntro?: string;
    options?: Array<{
      letter: 'A' | 'B' | 'C';
      text: string;
      quality: 'redirect' | 'good' | 'best';
      response: string;
    }>;
  } | null;
}

export interface CoreLesson {
  id: string;
  day_number: number;
  topic: string;
  universal_truth: string;
  marketing_headline?: string;
  marketing_tagline?: string;
}

export interface KellyVideoAsset {
  id: string;
  lesson_day: number;
  phase: string;
  archetype?: string;
  age_bucket: string;
  language: string;
  status: 'pending' | 'generating' | 'completed' | 'failed' | 'expired';
  video_public_url?: string;
  video_duration_ms?: number;
}

// =============================================================================
// CONSTANTS
// =============================================================================

export const SUPABASE_CDN = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates';

export const PHASES: Phase[] = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];

export const PHASE_TO_RUNTIME: Record<Phase, RuntimePhase> = {
  Hook: 'welcome',
  Fact1: 'q1',
  Fact2: 'q2',
  Fact3: 'q3',
  Wisdom: 'wisdom',
};

export const RUNTIME_TO_PHASE: Record<RuntimePhase, Phase> = {
  welcome: 'Hook',
  q1: 'Fact1',
  q2: 'Fact2',
  q3: 'Fact3',
  wisdom: 'Wisdom',
};

export const PERSONA_IDS: PersonaId[] = [
  'scientist', 'explorer', 'rebel', 'architect',
  'diplomat', 'empath', 'macgyver', 'mystic',
  'provider', 'storyteller', 'strategist', 'survivor',
];

export const PRIMARY_THREE: PersonaId[] = ['scientist', 'explorer', 'rebel'];

// =============================================================================
// PERSONAS DATA
// =============================================================================

export const PERSONAS: Record<PersonaId, Persona> = {
  scientist: {
    id: 'scientist',
    name: 'The Scientist',
    icon: '🔬',
    tagline: 'Data-driven precision',
    description: 'Lab goggles on forehead',
    accessory: 'clear laboratory safety goggles with elastic strap pushed up onto forehead',
    expression: 'focused analytical gaze with one eyebrow slightly raised, knowing intellectual smile',
    color: '#3b82f6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_scientist_head.png',
      clean: 'heygen/archetypes-head-only/kelly_scientist_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_scientist_prop_head.png',
    },
  },
  explorer: {
    id: 'explorer',
    name: 'The Explorer',
    icon: '🧭',
    tagline: 'Wonder and discovery',
    description: 'Aviator goggles + bandana',
    accessory: 'vintage brass and leather aviator flight goggles pushed up on top of head, weathered tan leather headband bandana',
    expression: 'wide eyes sparkling with wonder and excitement, bright genuine smile showing enthusiasm',
    color: '#eab308',
    images: {
      head: 'heygen/archetypes-head-only/kelly_explorer_head.png',
      clean: 'heygen/archetypes-head-only/kelly_explorer_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_explorer_prop_head.png',
    },
  },
  rebel: {
    id: 'rebel',
    name: 'The Rebel',
    icon: '⚡',
    tagline: 'Bold challenging spirit',
    description: 'Sunglasses in hair + earring',
    accessory: 'classic black wayfarer sunglasses pushed up and resting on top of head in hair, small silver hoop earring',
    expression: 'confident asymmetric smirk with one corner of mouth raised defiantly, intense direct eye contact',
    color: '#ef4444',
    images: {
      head: 'heygen/archetypes-head-only/kelly_rebel_head.png',
      clean: 'heygen/archetypes-head-only/kelly_rebel_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_rebel_prop_head.png',
    },
  },
  architect: {
    id: 'architect',
    name: 'The Architect',
    icon: '🏛️',
    tagline: 'Methodical structure',
    description: 'Pencil behind ear + glasses',
    accessory: 'classic yellow drafting pencil tucked behind right ear, tortoiseshell reading glasses pushed up on top of head',
    expression: 'thoughtful concentrated look with lips pressed together slightly, eyes showing deep analytical focus',
    color: '#6b7280',
    images: {
      head: 'heygen/archetypes-head-only/kelly_architect_head.png',
      clean: 'heygen/archetypes-head-only/kelly_architect_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_architect_prop_head.png',
    },
  },
  diplomat: {
    id: 'diplomat',
    name: 'The Diplomat',
    icon: '🤝',
    tagline: 'Inclusive harmony',
    description: 'Pearl studs + velvet headband',
    accessory: 'elegant classic pearl stud earrings, thin navy blue velvet headband pushing hair back slightly',
    expression: 'warm welcoming diplomatic smile, soft approachable eyes radiating understanding and openness',
    color: '#22c55e',
    images: {
      head: 'heygen/archetypes-head-only/kelly_diplomat_head.png',
      clean: 'heygen/archetypes-head-only/kelly_diplomat_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_diplomat_prop_head.png',
    },
  },
  empath: {
    id: 'empath',
    name: 'The Empath',
    icon: '💗',
    tagline: 'Nurturing warmth',
    description: 'Pink headband + lavender',
    accessory: 'soft dusty rose pink fabric headband, tiny dried lavender sprig tucked behind left ear',
    expression: 'gentle compassionate smile full of warmth, eyes radiating deep understanding and emotional connection',
    color: '#ec4899',
    images: {
      head: 'heygen/archetypes-head-only/kelly_empath_head.png',
      clean: 'heygen/archetypes-head-only/kelly_empath_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_empath_prop_head.png',
    },
  },
  macgyver: {
    id: 'macgyver',
    name: 'The MacGyver',
    icon: '🔧',
    tagline: 'Hands-on problem solver',
    description: 'Shop glasses + red bandana',
    accessory: 'clear protective shop safety glasses with side shields pushed up on forehead, red paisley utility bandana',
    expression: 'practical creative grin of someone with a clever solution, eyes bright and sparkling with ingenious idea',
    color: '#f97316',
    images: {
      head: 'heygen/archetypes-head-only/kelly_macgyver_head.png',
      clean: 'heygen/archetypes-head-only/kelly_macgyver_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_macgyver_prop_head.png',
    },
  },
  mystic: {
    id: 'mystic',
    name: 'The Mystic',
    icon: '✨',
    tagline: 'Profound serenity',
    description: 'Third eye amethyst + gold chain',
    accessory: 'small teardrop amethyst crystal gem at third eye position, delicate thin gold chain headpiece',
    expression: 'serene knowing smile with ancient wisdom in eyes, peaceful profound gaze seeing beyond the visible',
    color: '#a855f7',
    images: {
      head: 'heygen/archetypes-head-only/kelly_mystic_head.png',
      clean: 'heygen/archetypes-head-only/kelly_mystic_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_mystic_prop_head.png',
    },
  },
  provider: {
    id: 'provider',
    name: 'The Provider',
    icon: '🛡️',
    tagline: 'Reassuring strength',
    description: 'Cream knit headband',
    accessory: 'wide cozy cream-colored cable knit headband ear warmer wrapped around head',
    expression: 'warm protective nurturing smile, reassuring steady eyes that promise safety and care',
    color: '#14b8a6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_provider_head.png',
      clean: 'heygen/archetypes-head-only/kelly_provider_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_provider_prop_head.png',
    },
  },
  storyteller: {
    id: 'storyteller',
    name: 'The Storyteller',
    icon: '📖',
    tagline: 'Theatrical captivation',
    description: 'Gold glasses + peacock feather',
    accessory: 'vintage round gold-rimmed reading glasses pushed up on top of head, small peacock feather in hair',
    expression: 'animated expressive face mid-story, eyes sparkling with secrets to share, dramatic engaging smile',
    color: '#f472b6',
    images: {
      head: 'heygen/archetypes-head-only/kelly_storyteller_head.png',
      clean: 'heygen/archetypes-head-only/kelly_storyteller_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_storyteller_prop_head.png',
    },
  },
  strategist: {
    id: 'strategist',
    name: 'The Strategist',
    icon: '🎯',
    tagline: 'Sharp tactical mind',
    description: 'Angular glasses + chess clip',
    accessory: 'sharp modern angular black-framed glasses pushed up on top of head, small gold chess queen hair clip',
    expression: 'sharp focused calculating gaze, confident knowing look of someone thinking several moves ahead',
    color: '#6366f1',
    images: {
      head: 'heygen/archetypes-head-only/kelly_strategist_head.png',
      clean: 'heygen/archetypes-head-only/kelly_strategist_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_strategist_prop_head.png',
    },
  },
  survivor: {
    id: 'survivor',
    name: 'The Survivor',
    icon: '🏕️',
    tagline: 'Grounded resilience',
    description: 'Military bandana + dog tags',
    accessory: 'olive green tactical military bandana tied around forehead, silver military dog tags on ball chain',
    expression: 'serious determined look with no-nonsense direct gaze, eyes showing resilience and hard-won wisdom',
    color: '#84cc16',
    images: {
      head: 'heygen/archetypes-head-only/kelly_survivor_head.png',
      clean: 'heygen/archetypes-head-only/kelly_survivor_clean.png',
      prop: 'heygen/archetypes-head-only/kelly_survivor_prop_head.png',
    },
  },
};

// =============================================================================
// HELPERS
// =============================================================================

/**
 * Get the full CDN URL for a persona image
 */
export function getPersonaImageUrl(
  personaId: PersonaId, 
  variant: 'head' | 'clean' | 'prop' = 'head'
): string {
  const persona = PERSONAS[personaId];
  if (!persona) throw new Error(`Unknown persona: ${personaId}`);
  return `${SUPABASE_CDN}/${persona.images[variant]}`;
}

/**
 * Get persona by name (handles "The Scientist" → scientist)
 */
export function getPersonaByName(name: string): Persona | undefined {
  const id = name.toLowerCase().replace('the ', '') as PersonaId;
  return PERSONAS[id];
}

/**
 * Get all personas as an array (ordered)
 */
export function getPersonasArray(): Persona[] {
  return PERSONA_IDS.map(id => PERSONAS[id]);
}

/**
 * Get archetype color with optional opacity
 */
export function getArchetypeColor(personaId: PersonaId, opacity?: number): string {
  const color = PERSONAS[personaId]?.color || '#6b7280';
  if (opacity === undefined) return color;
  
  // Convert hex to rgba
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/**
 * Map database archetype string to PersonaId
 * Handles "The Scientist", "scientist", "SCIENTIST" etc.
 */
export function normalizeArchetypeId(archetype: string): PersonaId | null {
  const normalized = archetype.toLowerCase().replace('the ', '').trim();
  if (PERSONA_IDS.includes(normalized as PersonaId)) {
    return normalized as PersonaId;
  }
  return null;
}





















