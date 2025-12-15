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
