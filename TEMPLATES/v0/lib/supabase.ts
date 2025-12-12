/**
 * 🔌 Supabase Client & Hooks for v0 Templates
 * 
 * Provides typed queries for the Lotd database schema.
 * Use these hooks in any v0 template that needs data.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { useEffect, useState, useCallback } from 'react';
import type { 
  CoreLesson, 
  LessonAtom, 
  KellyVideoAsset, 
  PersonaId, 
  Phase 
} from './personas';

// =============================================================================
// CLIENT
// =============================================================================

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

let supabaseClient: SupabaseClient | null = null;

export function getSupabaseClient(): SupabaseClient {
  if (!supabaseClient) {
    supabaseClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  }
  return supabaseClient;
}

// =============================================================================
// QUERY RESULT TYPES
// =============================================================================

interface QueryResult<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refetch: () => Promise<void>;
}

interface LessonWithAtoms extends CoreLesson {
  atoms: LessonAtom[];
}

interface DayStats {
  totalAtoms: number;
  atomsWithContent: number;
  atomsWithVideo: number;
  atomsGenerating: number;
  atomsFailed: number;
}

// =============================================================================
// HOOKS
// =============================================================================

/**
 * Fetch a single lesson by day number
 */
export function useCoreLesson(dayNumber: number): QueryResult<CoreLesson> {
  const [data, setData] = useState<CoreLesson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const supabase = getSupabaseClient();
      const { data: lesson, error: err } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline')
        .eq('day_number', dayNumber)
        .single();

      if (err) throw new Error(err.message);
      setData(lesson);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dayNumber]);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, error, loading, refetch: fetch };
}

/**
 * Fetch a lesson with all its atoms for a specific archetype
 */
export function useLessonWithAtoms(
  dayNumber: number, 
  archetype?: PersonaId
): QueryResult<LessonWithAtoms> {
  const [data, setData] = useState<LessonWithAtoms | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const supabase = getSupabaseClient();
      
      // Get core lesson
      const { data: lesson, error: lessonErr } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline')
        .eq('day_number', dayNumber)
        .single();

      if (lessonErr) throw new Error(lessonErr.message);

      // Get atoms (optionally filtered by archetype)
      let atomQuery = supabase
        .from('lesson_atoms')
        .select('id, archetype, phase, content')
        .eq('core_lesson_id', lesson.id);

      if (archetype) {
        atomQuery = atomQuery.ilike('archetype', `%${archetype}%`);
      }

      const { data: atoms, error: atomsErr } = await atomQuery.order('phase');

      if (atomsErr) throw new Error(atomsErr.message);

      setData({
        ...lesson,
        atoms: atoms || [],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dayNumber, archetype]);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, error, loading, refetch: fetch };
}

/**
 * Fetch video assets for a specific day
 */
export function useVideoAssets(dayNumber: number): QueryResult<KellyVideoAsset[]> {
  const [data, setData] = useState<KellyVideoAsset[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const supabase = getSupabaseClient();
      const { data: videos, error: err } = await supabase
        .from('kelly_video_assets')
        .select('*')
        .eq('lesson_day', dayNumber);

      if (err) throw new Error(err.message);
      setData(videos || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dayNumber]);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, error, loading, refetch: fetch };
}

/**
 * Fetch day stats (content/video completion status)
 */
export function useDayStats(dayNumber: number): QueryResult<DayStats> {
  const [data, setData] = useState<DayStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const supabase = getSupabaseClient();

      // Get lesson ID
      const { data: lesson, error: lessonErr } = await supabase
        .from('core_lessons')
        .select('id')
        .eq('day_number', dayNumber)
        .single();

      if (lessonErr) throw new Error(lessonErr.message);

      // Count atoms with content
      const { count: atomCount, error: atomErr } = await supabase
        .from('lesson_atoms')
        .select('id', { count: 'exact' })
        .eq('core_lesson_id', lesson.id)
        .not('content', 'is', null);

      if (atomErr) throw new Error(atomErr.message);

      // Get video stats
      const { data: videos, error: videoErr } = await supabase
        .from('kelly_video_assets')
        .select('status')
        .eq('lesson_day', dayNumber);

      if (videoErr) throw new Error(videoErr.message);

      const videoCompleted = videos?.filter(v => v.status === 'completed').length || 0;
      const videoGenerating = videos?.filter(v => v.status === 'generating').length || 0;
      const videoFailed = videos?.filter(v => v.status === 'failed').length || 0;

      setData({
        totalAtoms: 60, // 12 archetypes × 5 phases
        atomsWithContent: atomCount || 0,
        atomsWithVideo: videoCompleted,
        atomsGenerating: videoGenerating,
        atomsFailed: videoFailed,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [dayNumber]);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, error, loading, refetch: fetch };
}

/**
 * Fetch all lessons (for calendar/sidebar)
 */
export function useAllLessons(): QueryResult<CoreLesson[]> {
  const [data, setData] = useState<CoreLesson[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const supabase = getSupabaseClient();
      const { data: lessons, error: err } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, universal_truth')
        .order('day_number');

      if (err) throw new Error(err.message);
      setData(lessons || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, error, loading, refetch: fetch };
}

// =============================================================================
// DIRECT QUERIES (for server components or non-React contexts)
// =============================================================================

/**
 * Get a single lesson by day number
 */
export async function fetchCoreLesson(dayNumber: number): Promise<CoreLesson | null> {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline')
    .eq('day_number', dayNumber)
    .single();

  if (error) throw new Error(error.message);
  return data;
}

/**
 * Get atoms for a lesson, optionally filtered by archetype
 */
export async function fetchLessonAtoms(
  lessonId: string, 
  archetype?: PersonaId
): Promise<LessonAtom[]> {
  const supabase = getSupabaseClient();
  
  let query = supabase
    .from('lesson_atoms')
    .select('id, archetype, phase, content')
    .eq('core_lesson_id', lessonId);

  if (archetype) {
    query = query.ilike('archetype', `%${archetype}%`);
  }

  const { data, error } = await query.order('phase');

  if (error) throw new Error(error.message);
  return data || [];
}

/**
 * Get video URL for a specific lesson/phase/variant
 */
export async function fetchVideoUrl(
  dayNumber: number,
  phase: Phase,
  options: {
    archetype?: PersonaId;
    ageBucket?: string;
    language?: string;
  } = {}
): Promise<string | null> {
  const supabase = getSupabaseClient();
  
  let query = supabase
    .from('kelly_video_assets')
    .select('video_public_url')
    .eq('lesson_day', dayNumber)
    .eq('phase', phase.toLowerCase())
    .eq('status', 'completed');

  if (options.ageBucket) query = query.eq('age_bucket', options.ageBucket);
  if (options.language) query = query.eq('language', options.language);
  if (options.archetype) query = query.ilike('archetype', `%${options.archetype}%`);

  const { data, error } = await query.limit(1).single();

  if (error || !data) return null;
  return data.video_public_url;
}

// =============================================================================
// REAL-TIME SUBSCRIPTIONS
// =============================================================================

/**
 * Subscribe to video generation updates for a day
 */
export function subscribeToVideoUpdates(
  dayNumber: number,
  onUpdate: (video: KellyVideoAsset) => void
): () => void {
  const supabase = getSupabaseClient();
  
  const subscription = supabase
    .channel(`videos-day-${dayNumber}`)
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'kelly_video_assets',
        filter: `lesson_day=eq.${dayNumber}`,
      },
      (payload) => {
        onUpdate(payload.new as KellyVideoAsset);
      }
    )
    .subscribe();

  return () => {
    supabase.removeChannel(subscription);
  };
}


