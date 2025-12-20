/**
 * ⚠️ DEPRECATED - DO NOT USE
 * Canonical loader: public/js/kelly-lesson-loader.js
 * This file will be removed after 2025-01-15
 * Last audit: 2025-12-15
 */

// Debug mode
const __GOLDEN_DEBUG = (
  (typeof location !== 'undefined' && location.search.includes('debug')) ||
  (typeof localStorage !== 'undefined' && localStorage.getItem('kellyDebug') === '1')
);

if (__GOLDEN_DEBUG) console.warn('[DEPRECATED] golden-v5-data-loader.js - Use kelly-lesson-loader.js');

// golden-v5-data-loader.js
// Loads real lesson content from Supabase for Golden V5

class GoldenV5DataLoader {
  constructor(supabaseClient) {
    this.supabase = supabaseClient;
    this.cache = {};
  }

  /**
   * Load complete lesson data for a specific day and archetype
   * @param {number} dayNumber - Day number (1-365)
   * @param {string} archetype - Archetype name (e.g., "The Explorer")
   * @returns {Promise<Object>} Complete lesson data
   */
  async loadLesson(dayNumber, archetype = 'The Explorer') {
    const cacheKey = `${dayNumber}-${archetype}`;
    
    if (this.cache[cacheKey]) {
      if (__GOLDEN_DEBUG) console.log(`[GoldenV5DataLoader] Using cached data for Day ${dayNumber}`);
      return this.cache[cacheKey];
    }

    if (__GOLDEN_DEBUG) console.log(`[GoldenV5DataLoader] Loading Day ${dayNumber} (${archetype})...`);

    try {
      // Step 1: Load core lesson data
      const { data: coreLesson, error: coreError } = await this.supabase
        .from('core_lessons')
        .select('*')
        .eq('day_number', dayNumber)
        .single();

      if (coreError) throw coreError;

      // Step 2: Load lesson atoms (phase-specific content)
      const { data: atoms, error: atomsError } = await this.supabase
        .from('lesson_atoms')
        .select('*')
        .eq('core_lesson_id', coreLesson.id)
        .eq('archetype', archetype)
        .order('phase');

      if (atomsError) throw atomsError;

      // Step 3: Organize atoms by phase
      const phases = {};
      atoms.forEach(atom => {
        phases[atom.phase] = {
          ...atom.content,
          kellyPose: atom.content.kellyPose || 'explaining',
          kellyEmotion: atom.content.kellyEmotion || 'curious',
          visualUrl: atom.visual_url,
          hdVideoUrl: atom.hd_video_url // NEW: HD video from pipeline
        };
      });

      // Step 4: Construct complete lesson object
      const lesson = {
        dayNumber: coreLesson.day_number,
        topic: coreLesson.topic,
        universalTruth: coreLesson.universal_truth,
        iconEmoji: coreLesson.icon_emoji || '📚',
        estimatedDuration: coreLesson.estimated_duration || 8,
        idealAgeRange: coreLesson.ideal_age_range,
        difficultyLevel: coreLesson.difficulty_level,
        archetype: archetype,
        phases: {
          Hook: phases.Hook || this.getPlaceholderPhase('Hook'),
          Fact1: phases.Fact1 || this.getPlaceholderPhase('Fact1'),
          Fact2: phases.Fact2 || this.getPlaceholderPhase('Fact2'),
          Fact3: phases.Fact3 || this.getPlaceholderPhase('Fact3'),
          Wisdom: phases.Wisdom || this.getPlaceholderPhase('Wisdom')
        }
      };

      // Cache the result
      this.cache[cacheKey] = lesson;

      if (__GOLDEN_DEBUG) console.log(`[GoldenV5DataLoader] ✅ Loaded Day ${dayNumber}:`, lesson);
      return lesson;

    } catch (error) {
      console.error(`[GoldenV5DataLoader] ❌ Failed to load Day ${dayNumber}:`, error);
      return this.getPlaceholderLesson(dayNumber, archetype);
    }
  }

  /**
   * Load video assets for a specific day and phase
   * @param {number} dayNumber - Day number (1-365)
   * @param {string} phase - Phase name (Hook, Fact1, Fact2, Fact3, Wisdom)
   * @returns {Promise<Object>} Video asset data
   */
  async loadVideoAsset(dayNumber, phase) {
    if (__GOLDEN_DEBUG) console.log(`[GoldenV5DataLoader] Loading video for Day ${dayNumber}, Phase ${phase}...`);

    try {
      const { data: asset, error } = await this.supabase
        .from('kelly_video_assets')
        .select('*')
        .eq('day_number', dayNumber)
        .eq('phase', phase.toLowerCase())
        .eq('asset_type', 'video')
        .eq('status', 'published')
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (error) throw error;

      if (__GOLDEN_DEBUG) console.log(`[GoldenV5DataLoader] ✅ Loaded video asset:`, asset);
      return {
        videoUrl: asset.public_url,
        manifestUrl: asset.public_url.replace('.mp4', '-safe-zones.json'),
        duration: asset.duration_seconds,
        resolution: asset.resolution,
        qualityTier: asset.quality_tier
      };

    } catch (error) {
      if (__GOLDEN_DEBUG) console.warn(`[GoldenV5DataLoader] ⚠️ No video found for Day ${dayNumber}, Phase ${phase}. Using fallback.`);
      return this.getFallbackVideo(dayNumber, phase);
    }
  }

  /**
   * Get placeholder phase data (fallback)
   */
  getPlaceholderPhase(phaseName) {
    return {
      script: `This is a placeholder for the ${phaseName} phase. Real content will be loaded from the database.`,
      options: [
        {
          letter: 'A',
          text: `Option A for ${phaseName}`,
          quality: 'good',
          response: 'This is a placeholder response.'
        },
        {
          letter: 'B',
          text: `Option B for ${phaseName}`,
          quality: 'best',
          response: 'This is a placeholder response.'
        }
      ],
      kellyPose: 'explaining',
      kellyEmotion: 'curious'
    };
  }

  /**
   * Get placeholder lesson data (fallback)
   */
  getPlaceholderLesson(dayNumber, archetype) {
    return {
      dayNumber: dayNumber,
      topic: 'Loading...',
      universalTruth: 'Loading...',
      iconEmoji: '📚',
      estimatedDuration: 8,
      idealAgeRange: 'All ages',
      difficultyLevel: 'Beginner',
      archetype: archetype,
      phases: {
        Hook: this.getPlaceholderPhase('Hook'),
        Fact1: this.getPlaceholderPhase('Fact1'),
        Fact2: this.getPlaceholderPhase('Fact2'),
        Fact3: this.getPlaceholderPhase('Fact3'),
        Wisdom: this.getPlaceholderPhase('Wisdom')
      }
    };
  }

  /**
   * Get fallback video (local test video)
   */
  getFallbackVideo(dayNumber, phase) {
    return {
      videoUrl: '/kelly/videos/001/welcome.mp4',
      manifestUrl: '/kelly/videos/001/welcome-safe-zones.json',
      duration: 30,
      resolution: '1920x1080',
      qualityTier: 'standard'
    };
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache = {};
    if (__GOLDEN_DEBUG) console.log('[GoldenV5DataLoader] Cache cleared');
  }
}

// Export for use in Golden V5
window.GoldenV5DataLoader = GoldenV5DataLoader;

