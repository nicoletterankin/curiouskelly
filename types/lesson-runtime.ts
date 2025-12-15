/**
 * CURIOUS KELLY - CANONICAL LESSON CONTRACT v1.0
 *
 * This defines THE data shape for all lesson operations.
 * DO NOT MODIFY without architectural review.
 *
 * @version 1.0.0
 * @lastUpdated 2025-12-15
 */

// === CORE ENUMS ===

export type LessonPhase =
  | 'hook'
  | 'cliff'
  | 'fact1'
  | 'fact2'
  | 'fact3'
  | 'wisdom'
  | 'outro';

export type Archetype =
  | 'scientist'
  | 'architect'
  | 'diplomat'
  | 'empath'
  | 'explorer'
  | 'macgyver'
  | 'mystic'
  | 'provider'
  | 'rebel'
  | 'storyteller'
  | 'strategist'
  | 'survivor';

export type AgeBucket = 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder';

export type LoadSource =
  | 'local_pack'
  | 'supabase'
  | 'd1_cache'
  | 'static_json'
  | 'golden_lesson';

// === CONTENT STRUCTURES ===

export interface LessonChoice {
  title: string;
  description: string;
  visual_id?: string;
  kelly_script: string;
}

export interface LessonOutcome {
  outcome_text: string;
  visual_id?: string;
  kelly_script: string;
}

export interface AtomContent {
  choice_intro: string;
  option_a: LessonChoice;
  option_b: LessonChoice;
  success_a?: LessonOutcome;
  success_b?: LessonOutcome;
  alt_a?: LessonOutcome;
  alt_b?: LessonOutcome;
}

// === MAIN TYPES ===

export interface Lesson {
  day_number: number;
  topic: string;
  headline: string;
  universal_truth: string;
  thumbnail_url?: string;
  hero_image_url?: string;
  updated_at?: string;
}

export interface LessonAtom {
  id?: string;
  day_number: number;
  phase: LessonPhase;
  archetype: Archetype;
  age_bucket: AgeBucket;
  content: AtomContent;
  created_at?: string;
}

export interface LessonVisual {
  id: string;
  day_number: number;
  url: string;
  alt_text: string;
  category: 'hero' | 'thumbnail' | 'diagram' | 'illustration' | 'animation';
  format: 'webp' | 'png' | 'jpg' | 'gif' | 'mp4' | 'webm';
}

// === LOADER TYPES ===

export interface LocalPackMeta {
  created_at: string;
  day_number: number;
  version: string;
  source: string;
}

export interface LocalPack {
  meta: LocalPackMeta;
  lesson: Lesson;
  atoms: LessonAtom[];
  visuals?: LessonVisual[];
}

export interface LessonPackage {
  lesson: Lesson;
  atoms: LessonAtom[];
  shards?: any[];
  source: LoadSource;
}

// === CONSTANTS ===

export const ALL_PHASES: LessonPhase[] = [
  'hook',
  'cliff',
  'fact1',
  'fact2',
  'fact3',
  'wisdom',
  'outro',
];

export const ALL_ARCHETYPES: Archetype[] = [
  'scientist',
  'architect',
  'diplomat',
  'empath',
  'explorer',
  'macgyver',
  'mystic',
  'provider',
  'rebel',
  'storyteller',
  'strategist',
  'survivor',
];

export const ALL_AGE_BUCKETS: AgeBucket[] = [
  'kid',
  'teen',
  'adult',
  'elder',
  'super_elder',
];

// === WINDOW GLOBALS ===

declare global {
  interface Window {
    CURIOUS_KELLY?: {
      DAY_017?: LocalPack;
      LOCAL_PACKS?: Record<string, LocalPack>;
      GOLDEN_LESSON?: LocalPack;
    };
    KellyLessonLoader?: {
      loadLesson: (dayNumber: number, options?: any) => Promise<LessonPackage>;
    };
  }
}

export {};

/**
 * CURIOUS KELLY - CANONICAL LESSON CONTRACT v1.0
 * DO NOT MODIFY without architectural review
 */

export type LessonPhase = 'hook' | 'cliff' | 'fact1' | 'fact2' | 'fact3' | 'wisdom' | 'outro';

export type Archetype =
  | 'scientist'
  | 'architect'
  | 'diplomat'
  | 'empath'
  | 'explorer'
  | 'macgyver'
  | 'mystic'
  | 'provider'
  | 'rebel'
  | 'storyteller'
  | 'strategist'
  | 'survivor';

export type AgeBucket = 'kid' | 'teen' | 'adult' | 'elder' | 'super_elder';

export interface LessonChoice {
  title: string;
  description: string;
  visual_id?: string;
  kelly_script: string;
}

export interface AtomContent {
  choice_intro: string;
  option_a: LessonChoice;
  option_b: LessonChoice;
  success_a?: { outcome_text: string; kelly_script: string };
  success_b?: { outcome_text: string; kelly_script: string };
}

export interface LessonAtom {
  day_number: number;
  phase: LessonPhase;
  archetype: Archetype;
  age_bucket: AgeBucket;
  content: AtomContent;
}

export interface Lesson {
  day_number: number;
  topic: string;
  headline: string;
  universal_truth: string;
  thumbnail_url?: string;
}

export interface LocalPack {
  meta: { created_at: string; day_number: number; version: string };
  lesson: Lesson;
  atoms: LessonAtom[];
}

// Window globals
declare global {
  interface Window {
    CURIOUS_KELLY?: {
      LOCAL_PACKS?: Record<string, LocalPack>;
      GOLDEN_LESSON?: LocalPack;
    };
  }
}
