/**
 * Two-Tier Phase Loader for Curious Kelly
 * 
 * TIER 1: Pre-computed Daily Lessons (365)
 *   - Audio files stored in CDN/Supabase Storage
 *   - Expression data stored in lesson_atoms.expression_data
 *   - Loading: Instant (like Spotify)
 * 
 * TIER 2: Custom Lessons (Real-Time Generated)
 *   - Audio generated via ElevenLabs API on-demand
 *   - Expression generated via ExpressionGenerator
 *   - Caching: Saved to Supabase for future users
 * 
 * @module phase-loader
 */

import ElevenLabsVoiceEngine, { getAgeBucket } from './elevenlabs-voice-engine.js';
import { ExpressionGenerator } from './expression-generator.js';
import CacheManager from './cache-manager.js';
import supabaseService from './supabase-service.js';

// =============================================================================
// CONSTANTS & CONFIGURATION
// =============================================================================

/**
 * CDN/Storage base URLs
 * Production: Use Cloudflare R2 or Supabase Storage
 */
const STORAGE_CONFIG = {
  // Pre-computed audio storage base URL
  precomputedAudioBase: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-audio/precomputed',
  
  // Generated audio storage base URL (cached custom lessons)
  generatedAudioBase: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-audio/generated',
  
  // Fallback local path for development
  localFallbackBase: '/lessons/audio',
  
  // Audio file format
  audioFormat: 'mp3',
  
  // Storage bucket name
  storageBucket: 'lesson-audio',
};

/**
 * Phase definitions matching Curious Kelly lesson structure
 */
const PHASES = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];

/**
 * Age buckets matching ElevenLabs voice engine
 */
const AGE_BUCKETS = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];

/**
 * Supported languages (precomputed)
 */
const LANGUAGES = ['en', 'es', 'fr'];

/**
 * Prefetch configuration
 */
const PREFETCH_CONFIG = {
  // Prefetch next phase when current is 75% complete
  prefetchThreshold: 0.75,
  
  // Maximum concurrent prefetch requests
  maxConcurrentPrefetch: 2,
  
  // Timeout for prefetch requests (ms)
  prefetchTimeout: 30000,
};

// =============================================================================
// PHASE LOADER CLASS
// =============================================================================

export default class PhaseLoader {
  /**
   * Create a new PhaseLoader instance
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    // State reference
    this.stateManager = options.stateManager || null;
    
    // Unity bridge reference for sending audio/expressions
    this.unityBridge = options.unityBridge || null;
    
    // Initialize sub-systems
    this.voiceEngine = new ElevenLabsVoiceEngine({
      apiKey: options.elevenLabsApiKey || window.ELEVENLABS_API_KEY,
      voiceId: options.voiceId || 'Kelly_v1', // Default Kelly voice
    });
    
    this.expressionGenerator = new ExpressionGenerator();
    this.cacheManager = new CacheManager();
    
    // Loading state
    this.loadingPhases = new Map(); // phase -> Promise
    this.prefetchedPhases = new Map(); // phase -> { audioUrl, expressions }
    this.prefetchQueue = new Set();
    this.currentPrefetchController = null;
    
    // Callbacks
    this.onLoadingStart = options.onLoadingStart || null;
    this.onLoadingEnd = options.onLoadingEnd || null;
    this.onError = options.onError || null;
    
    // Analytics
    this.loadStats = {
      cacheHits: 0,
      cacheMisses: 0,
      precomputedLoads: 0,
      generatedLoads: 0,
      totalLoadTimeMs: 0,
    };
  }

  // ===========================================================================
  // MAIN API
  // ===========================================================================

  /**
   * Load a specific phase with audio and expression data
   * Main entry point - handles both pre-computed and custom lessons
   * 
   * @param {string} phase - Phase name ('welcome', 'q1', 'q2', 'q3', 'wisdom')
   * @param {Object} state - Current app state (optional, uses stateManager if available)
   * @returns {Promise<Object>} Loaded phase content { audioUrl, expressions, metadata }
   */
  async loadPhase(phase, state = null) {
    const currentState = state || this.getState();
    const startTime = performance.now();
    
    // Validate phase
    if (!PHASES.includes(phase)) {
      throw new Error(`Invalid phase: ${phase}. Expected one of: ${PHASES.join(', ')}`);
    }
    
    // Check if already loading
    if (this.loadingPhases.has(phase)) {
      return this.loadingPhases.get(phase);
    }
    
    // Check prefetch cache first
    const prefetchKey = this.getPrefetchKey(currentState, phase);
    if (this.prefetchedPhases.has(prefetchKey)) {
      console.log(`[PhaseLoader] Using prefetched content for ${phase}`);
      const prefetched = this.prefetchedPhases.get(prefetchKey);
      this.prefetchedPhases.delete(prefetchKey);
      this.loadStats.cacheHits++;
      return prefetched;
    }
    
    // Start loading
    this.notifyLoadingStart(phase);
    
    const loadPromise = this.performPhaseLoad(phase, currentState)
      .then(result => {
        const loadTime = performance.now() - startTime;
        this.loadStats.totalLoadTimeMs += loadTime;
        console.log(`[PhaseLoader] Loaded ${phase} in ${loadTime.toFixed(0)}ms`);
        
        this.loadingPhases.delete(phase);
        this.notifyLoadingEnd(phase, result);
        
        // Trigger prefetch for next phase
        this.triggerPrefetch(phase, currentState);
        
        return result;
      })
      .catch(error => {
        this.loadingPhases.delete(phase);
        this.notifyLoadingEnd(phase, null, error);
        throw error;
      });
    
    this.loadingPhases.set(phase, loadPromise);
    return loadPromise;
  }

  /**
   * Perform the actual phase loading based on lesson type
   * @private
   */
  async performPhaseLoad(phase, state) {
    const lesson = state.selectedLesson || state.currentLesson;
    
    if (!lesson) {
      throw new Error('No lesson selected');
    }
    
    // Determine if this is a pre-computed daily lesson
    const isDailyLesson = this.isDailyLesson(lesson);
    
    if (isDailyLesson) {
      // TIER 1: Load from CDN/Supabase Storage (instant)
      return this.loadPrecomputedPhase(phase, state, lesson);
    } else {
      // TIER 2: Generate real-time or load from cache
      return this.loadCustomPhase(phase, state, lesson);
    }
  }

  // ===========================================================================
  // TIER 1: PRE-COMPUTED DAILY LESSONS
  // ===========================================================================

  /**
   * Load pre-computed content for a daily lesson phase
   * @private
   */
  async loadPrecomputedPhase(phase, state, lesson) {
    const ageBucket = this.getAgeBucket(state.age);
    const language = state.language || 'en';
    
    console.log(`[PhaseLoader] Loading pre-computed: ${lesson.slug || lesson.topic} - ${ageBucket} - ${language} - ${phase}`);
    
    // Get audio URL from CDN
    const audioUrl = this.getPrecomputedAudioUrl(lesson, ageBucket, language, phase);
    
    // Get expressions from database (lesson_atoms.expression_data)
    const expressions = await this.getPrecomputedExpressions(lesson, ageBucket, language, phase);
    
    // Verify audio exists (optional - can be disabled for performance)
    const audioExists = await this.verifyAudioUrl(audioUrl);
    
    if (!audioExists) {
      console.warn(`[PhaseLoader] Pre-computed audio not found: ${audioUrl}`);
      // Fallback to generation
      return this.loadCustomPhase(phase, state, lesson);
    }
    
    this.loadStats.precomputedLoads++;
    
    const result = {
      audioUrl,
      expressions,
      metadata: {
        tier: 1,
        type: 'precomputed',
        lesson: lesson.slug || lesson.topic,
        ageBucket,
        language,
        phase,
        loadedAt: new Date().toISOString(),
      },
    };
    
    // Send to Unity
    this.sendToUnity(audioUrl, expressions, phase);
    
    return result;
  }

  /**
   * Build the CDN URL for pre-computed audio
   * Path: /precomputed/{lesson-slug}/{age-bucket}-{language}-{phase}.mp3
   */
  getPrecomputedAudioUrl(lesson, ageBucket, language, phase) {
    const lessonSlug = this.getLessonSlug(lesson);
    const filename = `${ageBucket}-${language}-${phase}.${STORAGE_CONFIG.audioFormat}`;
    return `${STORAGE_CONFIG.precomputedAudioBase}/${lessonSlug}/${filename}`;
  }

  /**
   * Fetch pre-computed expressions from Supabase
   */
  async getPrecomputedExpressions(lesson, ageBucket, language, phase) {
    try {
      const { data, error } = await supabaseService.client
        .from('lesson_atoms')
        .select('expression_data')
        .eq('core_lesson_id', lesson.id)
        .eq('phase', phase)
        .single();
      
      if (error || !data?.expression_data) {
        console.warn(`[PhaseLoader] No expression data found for ${lesson.topic} - ${phase}`);
        return this.getDefaultExpressions(phase);
      }
      
      // Expression data is stored with age-bucket keys
      const variantKey = `${ageBucket}-${language}`;
      const expressionData = data.expression_data[variantKey] || data.expression_data.default;
      
      return expressionData || this.getDefaultExpressions(phase);
    } catch (error) {
      console.error('[PhaseLoader] Failed to fetch expressions:', error);
      return this.getDefaultExpressions(phase);
    }
  }

  // ===========================================================================
  // TIER 2: CUSTOM LESSONS (REAL-TIME GENERATION)
  // ===========================================================================

  /**
   * Load custom lesson content - generate in real-time or use cache
   * @private
   */
  async loadCustomPhase(phase, state, lesson) {
    const ageBucket = this.getAgeBucket(state.age);
    const language = state.language || 'en';
    const archetype = state.archetype || 'The Scientist';
    
    console.log(`[PhaseLoader] Loading custom: ${lesson.title || lesson.topic} - ${ageBucket} - ${language} - ${phase}`);
    
    // Check cache first
    const cacheKey = this.getCustomCacheKey(lesson, ageBucket, language, phase);
    const cached = await this.cacheManager.get(cacheKey);
    
    if (cached) {
      console.log(`[PhaseLoader] Cache hit for custom lesson: ${cacheKey}`);
      this.loadStats.cacheHits++;
      this.sendToUnity(cached.audioUrl, cached.expressions, phase);
      return {
        ...cached,
        metadata: {
          ...cached.metadata,
          fromCache: true,
        },
      };
    }
    
    this.loadStats.cacheMisses++;
    
    // Get lesson content/script for this phase
    const phaseContent = await this.getPhaseContent(lesson, phase, state);
    
    if (!phaseContent || !phaseContent.script) {
      throw new Error(`No content available for phase: ${phase}`);
    }
    
    // Generate audio via ElevenLabs
    const audioResult = await this.generatePhaseAudio(
      phaseContent.script,
      state.age,
      language,
      archetype,
      phaseContent.tone || 'warm'
    );
    
    // Generate expressions
    const expressions = this.expressionGenerator.generate({
      text: phaseContent.script,
      elevenLabsResponse: audioResult,
      archetype,
      tone: phaseContent.tone || 'warm',
      ageBucket,
      language,
      phase,
      totalDuration: audioResult.duration || 60,
    });
    
    // Cache for future users
    const audioUrl = audioResult.audioUrl;
    await this.cacheGeneratedContent(cacheKey, lesson, ageBucket, language, phase, audioUrl, expressions);
    
    this.loadStats.generatedLoads++;
    
    const result = {
      audioUrl,
      expressions,
      metadata: {
        tier: 2,
        type: 'generated',
        lesson: lesson.title || lesson.topic,
        ageBucket,
        language,
        phase,
        archetype,
        characterCount: phaseContent.script.length,
        loadedAt: new Date().toISOString(),
      },
    };
    
    // Send to Unity
    this.sendToUnity(audioUrl, expressions, phase);
    
    return result;
  }

  /**
   * Generate audio via ElevenLabs
   */
  async generatePhaseAudio(script, age, language, archetype, tone) {
    try {
      return await this.voiceEngine.generatePhaseAudio(
        script,
        age,
        language,
        archetype,
        tone
      );
    } catch (error) {
      console.error('[PhaseLoader] Audio generation failed:', error);
      throw new Error(`Failed to generate audio: ${error.message}`);
    }
  }

  /**
   * Cache generated content for future users
   */
  async cacheGeneratedContent(cacheKey, lesson, ageBucket, language, phase, audioUrl, expressions) {
    try {
      // Save to local cache
      await this.cacheManager.set(cacheKey, {
        audioUrl,
        expressions,
        metadata: {
          tier: 2,
          type: 'generated-cached',
          lesson: lesson.title || lesson.topic,
          ageBucket,
          language,
          phase,
          cachedAt: new Date().toISOString(),
        },
      });
      
      // Optionally upload to Supabase Storage for global caching
      // This creates a permanent cached version that other users can benefit from
      const shouldPersist = true; // Could be controlled by config
      if (shouldPersist) {
        await this.persistGeneratedAudio(lesson, ageBucket, language, phase, audioUrl);
      }
      
      console.log(`[PhaseLoader] Cached generated content: ${cacheKey}`);
    } catch (error) {
      console.warn('[PhaseLoader] Failed to cache content:', error);
      // Don't throw - caching failure shouldn't break the experience
    }
  }

  /**
   * Upload generated audio to Supabase Storage for global caching
   */
  async persistGeneratedAudio(lesson, ageBucket, language, phase, audioUrl) {
    try {
      // Fetch the audio blob
      const response = await fetch(audioUrl);
      const audioBlob = await response.blob();
      
      const lessonSlug = this.getLessonSlug(lesson);
      const filename = `${ageBucket}-${language}-${phase}.${STORAGE_CONFIG.audioFormat}`;
      const storagePath = `generated/${lessonSlug}/${filename}`;
      
      const { error } = await supabaseService.client.storage
        .from(STORAGE_CONFIG.storageBucket)
        .upload(storagePath, audioBlob, {
          contentType: 'audio/mpeg',
          upsert: true,
        });
      
      if (error) {
        console.warn('[PhaseLoader] Failed to persist audio to storage:', error);
      } else {
        console.log(`[PhaseLoader] Persisted audio to: ${storagePath}`);
      }
    } catch (error) {
      console.warn('[PhaseLoader] Failed to persist audio:', error);
    }
  }

  // ===========================================================================
  // PREFETCHING
  // ===========================================================================

  /**
   * Trigger prefetch for the next phase
   */
  triggerPrefetch(currentPhase, state) {
    const nextPhaseIndex = PHASES.indexOf(currentPhase) + 1;
    
    if (nextPhaseIndex < PHASES.length) {
      const nextPhase = PHASES[nextPhaseIndex];
      this.prefetchPhase(nextPhase, state);
    }
  }

  /**
   * Prefetch a phase in the background
   */
  async prefetchPhase(phase, state) {
    const prefetchKey = this.getPrefetchKey(state, phase);
    
    // Skip if already prefetched or currently prefetching
    if (this.prefetchedPhases.has(prefetchKey) || this.prefetchQueue.has(prefetchKey)) {
      return;
    }
    
    // Check concurrent prefetch limit
    if (this.prefetchQueue.size >= PREFETCH_CONFIG.maxConcurrentPrefetch) {
      return;
    }
    
    this.prefetchQueue.add(prefetchKey);
    console.log(`[PhaseLoader] Prefetching: ${phase}`);
    
    try {
      const result = await this.performPhaseLoad(phase, state);
      this.prefetchedPhases.set(prefetchKey, result);
      console.log(`[PhaseLoader] Prefetched: ${phase}`);
    } catch (error) {
      console.warn(`[PhaseLoader] Prefetch failed for ${phase}:`, error);
    } finally {
      this.prefetchQueue.delete(prefetchKey);
    }
  }

  /**
   * Cancel all ongoing prefetch operations (e.g., when user changes settings)
   */
  cancelPrefetch() {
    this.prefetchQueue.clear();
    this.prefetchedPhases.clear();
    console.log('[PhaseLoader] Prefetch cancelled');
  }

  /**
   * Generate a prefetch cache key
   */
  getPrefetchKey(state, phase) {
    const ageBucket = this.getAgeBucket(state.age);
    const language = state.language || 'en';
    const lessonId = state.selectedLesson?.id || state.currentLesson?.id || 'unknown';
    return `${lessonId}-${ageBucket}-${language}-${phase}`;
  }

  // ===========================================================================
  // STATE CHANGE HANDLING
  // ===========================================================================

  /**
   * Handle age or language change mid-phase
   * Reloads current phase with new settings
   * 
   * @param {string} changeType - 'age' or 'language'
   * @param {*} newValue - New value
   */
  async handleStateChange(changeType, newValue) {
    // Cancel any ongoing prefetch
    this.cancelPrefetch();
    
    // Get updated state
    const state = this.getState();
    const currentPhase = state.currentPhase || 'welcome';
    
    console.log(`[PhaseLoader] State change: ${changeType} = ${newValue}, reloading ${currentPhase}`);
    
    // Reload current phase with new settings
    return this.loadPhase(currentPhase, state);
  }

  // ===========================================================================
  // UNITY INTEGRATION
  // ===========================================================================

  /**
   * Send loaded content to Unity
   */
  sendToUnity(audioUrl, expressions, phase) {
    if (!this.unityBridge) {
      console.log('[PhaseLoader] No Unity bridge - skipping send');
      return;
    }
    
    // Send audio load event
    this.unityBridge.emit('audio-load', {
      url: audioUrl,
      phase,
      autoplay: true,
    });
    
    // Send expression data
    this.unityBridge.emit('expression-data', {
      phase,
      expressions: expressions.expressions || [],
      gestures: expressions.gestures || [],
      blendShapeTimeline: expressions.blendShapeTimeline || [],
      metadata: expressions.metadata || {},
    });
  }

  // ===========================================================================
  // HELPER METHODS
  // ===========================================================================

  /**
   * Get current app state from state manager
   */
  getState() {
    if (this.stateManager) {
      return this.stateManager.getState();
    }
    // Fallback to global state if available
    if (typeof window !== 'undefined' && window.appState) {
      return window.appState;
    }
    return {};
  }

  /**
   * Check if a lesson is a pre-computed daily lesson
   */
  isDailyLesson(lesson) {
    // Daily lessons have day_number 1-365
    return lesson.day_number && lesson.day_number >= 1 && lesson.day_number <= 365;
  }

  /**
   * Get age bucket from numeric age
   */
  getAgeBucket(age) {
    return getAgeBucket(age) || '18-35';
  }

  /**
   * Generate URL-safe lesson slug
   */
  getLessonSlug(lesson) {
    if (lesson.slug) return lesson.slug;
    
    const text = lesson.topic || lesson.title || `day-${lesson.day_number}`;
    return text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .substring(0, 50);
  }

  /**
   * Generate cache key for custom lessons
   */
  getCustomCacheKey(lesson, ageBucket, language, phase) {
    const lessonId = lesson.id || this.getLessonSlug(lesson);
    return `custom-${lessonId}-${ageBucket}-${language}-${phase}`;
  }

  /**
   * Verify audio URL exists (HEAD request)
   */
  async verifyAudioUrl(url) {
    try {
      const response = await fetch(url, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get phase content from lesson data
   */
  async getPhaseContent(lesson, phase, state) {
    // Try to get from lesson_atoms first
    try {
      const atom = await supabaseService.getAtom(
        lesson.id,
        state.archetype || 'The Scientist',
        phase
      );
      
      if (atom?.content) {
        return {
          script: atom.content.script || atom.content.text || JSON.stringify(atom.content),
          tone: atom.content.tone || 'warm',
        };
      }
    } catch (error) {
      console.warn('[PhaseLoader] Failed to fetch atom:', error);
    }
    
    // Fallback to lesson content
    if (lesson.content && lesson.content[phase]) {
      return {
        script: lesson.content[phase].script || lesson.content[phase],
        tone: lesson.content[phase].tone || 'warm',
      };
    }
    
    return null;
  }

  /**
   * Get default expressions for a phase
   */
  getDefaultExpressions(phase) {
    // Return minimal default expressions based on phase
    const defaults = {
      welcome: {
        expressions: [
          { timestamp: 0, emotion: 'warm', intensity: 0.7 },
          { timestamp: 2, emotion: 'excited', intensity: 0.8 },
        ],
        gestures: [
          { timestamp: 0.5, gesture: 'open_arms_welcome', duration: 2.0, intensity: 0.7 },
        ],
      },
      q1: {
        expressions: [
          { timestamp: 0, emotion: 'curious', intensity: 0.8 },
          { timestamp: 5, emotion: 'explaining', intensity: 0.7 },
        ],
        gestures: [
          { timestamp: 1, gesture: 'point_up_dramatic', duration: 1.5, intensity: 0.6 },
        ],
      },
      q2: {
        expressions: [
          { timestamp: 0, emotion: 'encouraging', intensity: 0.7 },
        ],
        gestures: [],
      },
      q3: {
        expressions: [
          { timestamp: 0, emotion: 'thoughtful', intensity: 0.7 },
        ],
        gestures: [],
      },
      wisdom: {
        expressions: [
          { timestamp: 0, emotion: 'serene', intensity: 0.7 },
          { timestamp: 5, emotion: 'warm', intensity: 0.8 },
        ],
        gestures: [
          { timestamp: 2, gesture: 'heart_open', duration: 2.5, intensity: 0.6 },
        ],
      },
    };
    
    return defaults[phase] || { expressions: [], gestures: [] };
  }

  // ===========================================================================
  // CALLBACKS & NOTIFICATIONS
  // ===========================================================================

  notifyLoadingStart(phase) {
    if (typeof this.onLoadingStart === 'function') {
      this.onLoadingStart(phase);
    }
  }

  notifyLoadingEnd(phase, result, error = null) {
    if (typeof this.onLoadingEnd === 'function') {
      this.onLoadingEnd(phase, result, error);
    }
  }

  // ===========================================================================
  // STATISTICS & DIAGNOSTICS
  // ===========================================================================

  /**
   * Get loading statistics
   */
  getStats() {
    const totalLoads = this.loadStats.precomputedLoads + this.loadStats.generatedLoads;
    return {
      ...this.loadStats,
      totalLoads,
      averageLoadTimeMs: totalLoads > 0 
        ? (this.loadStats.totalLoadTimeMs / totalLoads).toFixed(2) 
        : 0,
      cacheHitRate: (this.loadStats.cacheHits + this.loadStats.cacheMisses) > 0
        ? ((this.loadStats.cacheHits / (this.loadStats.cacheHits + this.loadStats.cacheMisses)) * 100).toFixed(1) + '%'
        : '0%',
      tier1Percentage: totalLoads > 0
        ? ((this.loadStats.precomputedLoads / totalLoads) * 100).toFixed(1) + '%'
        : '0%',
    };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.loadStats = {
      cacheHits: 0,
      cacheMisses: 0,
      precomputedLoads: 0,
      generatedLoads: 0,
      totalLoadTimeMs: 0,
    };
  }
}

// =============================================================================
// STANDALONE HELPER FUNCTIONS
// =============================================================================

/**
 * Create a PhaseLoader instance with common defaults
 */
export function createPhaseLoader(options = {}) {
  return new PhaseLoader({
    elevenLabsApiKey: options.apiKey || window.ELEVENLABS_API_KEY,
    stateManager: options.stateManager,
    unityBridge: options.unityBridge,
    ...options,
  });
}

/**
 * Get the audio URL for a pre-computed lesson phase
 * Useful for direct access without instantiating PhaseLoader
 */
export function getPrecomputedAudioUrl(lessonSlug, ageBucket, language, phase) {
  const filename = `${ageBucket}-${language}-${phase}.${STORAGE_CONFIG.audioFormat}`;
  return `${STORAGE_CONFIG.precomputedAudioBase}/${lessonSlug}/${filename}`;
}

/**
 * Get the audio URL for a generated/cached lesson phase
 */
export function getGeneratedAudioUrl(lessonSlug, ageBucket, language, phase) {
  const filename = `${ageBucket}-${language}-${phase}.${STORAGE_CONFIG.audioFormat}`;
  return `${STORAGE_CONFIG.generatedAudioBase}/${lessonSlug}/${filename}`;
}

// Export constants for external use
export { PHASES, AGE_BUCKETS, LANGUAGES, STORAGE_CONFIG, PREFETCH_CONFIG };










