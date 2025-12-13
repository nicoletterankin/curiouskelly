import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

/**
 * Supabase Service - Browser-compatible client
 * 
 * Credentials are loaded from:
 * 1. Window config (set by hosting page)
 * 2. Environment variables (for Node.js/SSR)
 * 3. Default fallback (development only - NEVER commit real keys)
 */

// Try multiple sources for credentials
const getSupabaseUrl = () => {
  // Window config (set in HTML or by build process)
  if (typeof window !== 'undefined' && window.SUPABASE_URL) {
    return window.SUPABASE_URL;
  }
  // Environment variables (Node.js/build time)
  if (typeof process !== 'undefined' && process.env?.PUBLIC_SUPABASE_URL) {
    return process.env.PUBLIC_SUPABASE_URL;
  }
  // Import meta env (Vite/Astro)
  if (typeof import.meta !== 'undefined' && import.meta.env?.PUBLIC_SUPABASE_URL) {
    return import.meta.env.PUBLIC_SUPABASE_URL;
  }
  // Fallback for development (should be replaced in production)
  console.warn('[SupabaseService] Using fallback URL - configure window.SUPABASE_URL for production');
  return 'https://tvjalxxsyryjphkforjv.supabase.co'; // Project: forjv
};

const getSupabaseKey = () => {
  // Window config
  if (typeof window !== 'undefined' && window.SUPABASE_ANON_KEY) {
    return window.SUPABASE_ANON_KEY;
  }
  // Environment variables
  if (typeof process !== 'undefined' && process.env?.PUBLIC_SUPABASE_ANON_KEY) {
    return process.env.PUBLIC_SUPABASE_ANON_KEY;
  }
  // Import meta env
  if (typeof import.meta !== 'undefined' && import.meta.env?.PUBLIC_SUPABASE_ANON_KEY) {
    return import.meta.env.PUBLIC_SUPABASE_ANON_KEY;
  }
  // Fallback for development
  console.warn('[SupabaseService] Using fallback key - configure window.SUPABASE_ANON_KEY for production');
  return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4NjI0NzgsImV4cCI6MjA0OTQzODQ3OH0.qfTs_t0tLmVHFNlKlOqXxvbmEgUEZpHdnVAFbQdJv1c'; // Project: forjv
};

const SUPABASE_URL = getSupabaseUrl();
const SUPABASE_KEY = getSupabaseKey();

class SupabaseService {
  constructor() {
    this.client = createClient(SUPABASE_URL, SUPABASE_KEY);
  }

  /**
   * Fetch a single core lesson by day number
   * @param {number} dayNumber 
   * @returns {Promise<Object|null>}
   */
  async getCoreLesson(dayNumber) {
    const { data, error } = await this.client
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNumber)
      .single();
    
    if (error) {
      console.error(`Error fetching core lesson day ${dayNumber}:`, error);
      return null;
    }
    return data;
  }

  /**
   * Fetch minimal data for all core lessons (for calendar)
   * @returns {Promise<Array>}
   */
  async getAllCoreLessons() {
    const { data, error } = await this.client
      .from('core_lessons')
      .select('id, day_number, topic, universal_truth')
      .order('day_number', { ascending: true });

    if (error) {
      console.error('Error fetching calendar lessons:', error);
      return [];
    }
    return data;
  }

  /**
   * Fetch a specific atom for the player
   * @param {string} coreLessonId 
   * @param {string} archetype 
   * @param {string} phase 
   * @returns {Promise<Object|null>}
   */
  async getAtom(coreLessonId, archetype, phase) {
    const { data, error } = await this.client
      .from('lesson_atoms')
      .select('*')
      .eq('core_lesson_id', coreLessonId)
      .eq('archetype', archetype)
      .eq('phase', phase)
      .single();

    if (error) {
      console.error(`Error fetching atom (${archetype}/${phase}):`, error);
      return null;
    }
    return data;
  }

  /**
   * Fetch all atoms for a lesson and archetype (for full session load)
   * @param {string} coreLessonId 
   * @param {string} archetype 
   * @returns {Promise<Object>} Map of phase -> content
   */
  async getAtomsForLesson(coreLessonId, archetype) {
      const { data, error } = await this.client
        .from('lesson_atoms')
        .select('phase, content')
        .eq('core_lesson_id', coreLessonId)
        .eq('archetype', archetype);
      
      if (error) {
          console.error('Error fetching lesson atoms:', error);
          return {};
      }
      
      // Convert array to map: { "Hook": content, ... }
      return data.reduce((acc, item) => {
          acc[item.phase] = item.content;
          return acc;
      }, {});
  }
  
  /**
   * Helper to calculate today's day number (1-365)
   * @returns {number}
   */
  getTodayNumber() {
      const now = new Date();
      const start = new Date(now.getFullYear(), 0, 0);
      const diff = now - start;
      const oneDay = 1000 * 60 * 60 * 24;
      const day = Math.floor(diff / oneDay);
      return Math.max(1, Math.min(365, day));
  }

  async getTodayLesson() {
      const day = this.getTodayNumber();
      return this.getCoreLesson(day);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // AGE/TONE/LANGUAGE PERSONALIZATION - NEW METHODS
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Fetch age-specific hook for a lesson day
   * @param {number} dayNumber - Day of the year (1-365)
   * @param {string} ageBucket - e.g., "2-5", "6-12", "13-17", "18-29", "30-54", "55+"
   * @returns {Promise<string|null>}
   */
  async getAgeHook(dayNumber, ageBucket) {
    const { data, error } = await this.client
      .from('lesson_age_hooks')
      .select('hook')
      .eq('day_number', dayNumber)
      .eq('age_bucket', ageBucket)
      .single();

    if (error) {
      // Not an error - just means no hook for this combo
      console.log(`[Supabase] No age hook for day ${dayNumber}, bucket ${ageBucket}`);
      return null;
    }
    return data?.hook || null;
  }

  /**
   * Fetch lesson shards (age/region/tone variants) for a lesson
   * @param {string} coreLessonId - UUID of the core lesson
   * @returns {Promise<Array>}
   */
  async getLessonShards(coreLessonId) {
    const { data, error } = await this.client
      .from('lesson_shards')
      .select('age, region, tone, birth_year, script_content')
      .eq('core_lesson_id', coreLessonId);

    if (error) {
      console.warn(`[Supabase] Error loading shards:`, error);
      return [];
    }
    console.log(`[Supabase] Loaded ${data?.length || 0} shards for lesson`);
    return data || [];
  }

  /**
   * Fetch available archetypes for a lesson (for debugging/fallback)
   * @param {string} coreLessonId 
   * @returns {Promise<Array<string>>}
   */
  async getAvailableArchetypes(coreLessonId) {
    const { data, error } = await this.client
      .from('lesson_atoms')
      .select('archetype')
      .eq('core_lesson_id', coreLessonId);

    if (error) {
      console.error('Error fetching archetypes:', error);
      return [];
    }
    
    // Return unique archetypes
    return [...new Set(data.map(d => d.archetype))];
  }
}

export default new SupabaseService();

