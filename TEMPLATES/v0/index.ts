/**
 * 🎨 Lotd v0 Templates - Main Export
 * 
 * Import everything from this single entry point:
 * 
 *   import { 
 *     FactoryDayView, 
 *     ArchetypeCard, 
 *     PERSONAS,
 *     useLessonWithAtoms 
 *   } from '@/templates/v0';
 */

// =============================================================================
// COMPONENTS
// =============================================================================

export { default as FactoryDayView } from './FactoryDayView';
export { 
  default as ArchetypeCard, 
  ArchetypeGrid, 
  ArchetypeBadge 
} from './ArchetypeCard';
export { default as LessonPreviewCard } from './LessonPreviewCard';

// =============================================================================
// LIB - Personas
// =============================================================================

export {
  // Types
  type PersonaId,
  type Persona,
  type Phase,
  type RuntimePhase,
  type Language,
  type AgeBucket,
  type Tone,
  type LessonAtom,
  type CoreLesson,
  type KellyVideoAsset,
  
  // Constants
  SUPABASE_CDN,
  PHASES,
  PHASE_TO_RUNTIME,
  RUNTIME_TO_PHASE,
  PERSONA_IDS,
  PRIMARY_THREE,
  PERSONAS,
  
  // Helpers
  getPersonaImageUrl,
  getPersonaByName,
  getPersonasArray,
  getArchetypeColor,
  normalizeArchetypeId,
} from './lib/personas';

// =============================================================================
// LIB - Supabase
// =============================================================================

export {
  // Client
  getSupabaseClient,
  
  // Hooks
  useCoreLesson,
  useLessonWithAtoms,
  useVideoAssets,
  useDayStats,
  useAllLessons,
  
  // Direct queries
  fetchCoreLesson,
  fetchLessonAtoms,
  fetchVideoUrl,
  
  // Subscriptions
  subscribeToVideoUpdates,
} from './lib/supabase';










