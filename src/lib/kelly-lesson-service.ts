/**
 * Kelly Lesson Service
 * Fetches lesson assets from the kelly_lesson_assets registry in Supabase
 * 
 * Usage:
 *   const lesson = await KellyLessonService.getLesson(1, 'story', 35, 'en');
 *   if (lesson.videoUrl) { playVideo(lesson.videoUrl); }
 *   else if (lesson.audioUrl) { playAudioWithPixiJS(lesson.audioUrl, lesson.visemeData); }
 */

import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = import.meta.env.PUBLIC_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;

const supabase = createClient(supabaseUrl, supabaseAnonKey);

export interface LessonAsset {
  id: string;
  dayNumber: number;
  phase: 'hook' | 'story' | 'wonder' | 'action' | 'wisdom';
  ageGroup: number;
  language: string;
  script: string | null;
  audioUrl: string | null;
  audioDuration: number | null;
  visemeData: object | null;
  videoUrl: string | null;
  videoSource: string | null;
  videoDuration: number | null;
  status: 'pending' | 'script_ready' | 'audio_ready' | 'video_processing' | 'complete' | 'error';
}

export interface FactoryStats {
  totalAssets: number;
  pending: number;
  scriptReady: number;
  audioReady: number;
  complete: number;
  progressPct: number;
}

export class KellyLessonService {
  /**
   * Get a single lesson asset
   */
  static async getLesson(
    dayNumber: number,
    phase: string,
    ageGroup: number = 35,
    language: string = 'en'
  ): Promise<LessonAsset | null> {
    const { data, error } = await supabase
      .from('kelly_lesson_assets')
      .select('*')
      .eq('day_number', dayNumber)
      .eq('phase', phase)
      .eq('age_group', ageGroup)
      .eq('language', language)
      .single();

    if (error || !data) {
      console.error('Error fetching lesson:', error);
      return null;
    }

    return this.mapToLessonAsset(data);
  }

  /**
   * Get all phases for a day/age/language
   */
  static async getDayLessons(
    dayNumber: number,
    ageGroup: number = 35,
    language: string = 'en'
  ): Promise<LessonAsset[]> {
    const { data, error } = await supabase
      .from('kelly_lesson_assets')
      .select('*')
      .eq('day_number', dayNumber)
      .eq('age_group', ageGroup)
      .eq('language', language)
      .order('phase');

    if (error || !data) {
      console.error('Error fetching day lessons:', error);
      return [];
    }

    return data.map(this.mapToLessonAsset);
  }

  /**
   * Get factory production stats
   */
  static async getFactoryStats(): Promise<FactoryStats | null> {
    const { data, error } = await supabase.rpc('get_factory_stats');

    if (error || !data || data.length === 0) {
      console.error('Error fetching factory stats:', error);
      return null;
    }

    const stats = data[0];
    return {
      totalAssets: stats.total_assets,
      pending: stats.pending,
      scriptReady: stats.script_ready,
      audioReady: stats.audio_ready,
      complete: stats.complete,
      progressPct: parseFloat(stats.progress_pct) || 0
    };
  }

  /**
   * Get dashboard data for a range of days
   */
  static async getDashboard(startDay: number = 1, endDay: number = 10): Promise<any[]> {
    const { data, error } = await supabase
      .from('kelly_factory_dashboard')
      .select('*')
      .gte('day_number', startDay)
      .lte('day_number', endDay);

    if (error) {
      console.error('Error fetching dashboard:', error);
      return [];
    }

    return data || [];
  }

  /**
   * Map database row to LessonAsset interface
   */
  private static mapToLessonAsset(row: any): LessonAsset {
    return {
      id: row.id,
      dayNumber: row.day_number,
      phase: row.phase,
      ageGroup: row.age_group,
      language: row.language,
      script: row.script,
      audioUrl: row.audio_url,
      audioDuration: row.audio_duration,
      visemeData: row.viseme_data,
      videoUrl: row.video_url,
      videoSource: row.video_source,
      videoDuration: row.video_duration,
      status: row.status
    };
  }
}

// Export singleton instance for convenience
export const kellyLessonService = KellyLessonService;
