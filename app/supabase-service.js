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
  return 'https://tvjalxxsyryjphkforjv.supabase.co';
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
  return 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcnp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5NjI3NTcsImV4cCI6MjA0NzUzODc1N30.kLMlC14ckEp-XoL8RX5liw_cMdGs8lR';
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
}

export default new SupabaseService();

