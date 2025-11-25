/**
 * ElevenLabs Voice Engine - Age-Based Pitch Modulation System
 * 
 * Provides voice synthesis with:
 * - Age-based pitch modulation (2-102 years)
 * - Archetype-specific voice settings (all 12 archetypes)
 * - Tone-based adjustments (enthusiastic, serious, playful, thoughtful)
 * - Retry logic with exponential backoff
 * - Fallback to cached audio
 * 
 * @requires ELEVENLABS_API_KEY environment variable
 */

/* eslint-env browser */

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

/**
 * Age-based pitch modulation mapping
 * Base voice is calibrated for age 27 (within 18-35 bracket)
 */
const AGE_PITCH_MAP = {
  '2-5':   { minAge: 2,  maxAge: 5,   pitchShift: 0.20,  description: 'Childlike, higher' },
  '6-12':  { minAge: 6,  maxAge: 12,  pitchShift: 0.12,  description: 'Pre-teen' },
  '13-17': { minAge: 13, maxAge: 17,  pitchShift: 0.05,  description: 'Teen' },
  '18-35': { minAge: 18, maxAge: 35,  pitchShift: 0.00,  description: 'Base voice (27yo)' },
  '36-60': { minAge: 36, maxAge: 60,  pitchShift: -0.05, description: 'Mature' },
  '61-102': { minAge: 61, maxAge: 102, pitchShift: -0.12, description: 'Elder' },
};

/**
 * Voice settings for all 12 archetypes
 * Each archetype has specific stability and similarity_boost settings
 * that affect the voice character and consistency
 * 
 * stability: 0.0-1.0 (lower = more varied/expressive, higher = more consistent)
 * similarity_boost: 0.0-1.0 (higher = closer to original voice)
 * style: 0.0-1.0 (exaggeration of style, available on some models)
 * speakingRate: multiplier for speaking speed (1.0 = normal)
 */
const ARCHETYPE_VOICE_SETTINGS = {
  'The Scientist': {
    stability: 0.70,
    similarity_boost: 0.80,
    style: 0.30,
    speakingRate: 0.95,
    description: 'Measured, clear, analytical delivery',
    voiceCharacter: 'Precise articulation, even pacing, thoughtful pauses',
  },
  'The Explorer': {
    stability: 0.50,
    similarity_boost: 0.70,
    style: 0.50,
    speakingRate: 1.10,
    description: 'Energetic, varied, adventurous tone',
    voiceCharacter: 'Dynamic range, excited inflections, forward momentum',
  },
  'The Artist': {
    stability: 0.40,
    similarity_boost: 0.90,
    style: 0.70,
    speakingRate: 0.90,
    description: 'Expressive, emotional, creative flow',
    voiceCharacter: 'Rich emotion, dramatic pauses, lyrical quality',
  },
  'The Survivor': {
    stability: 0.75,
    similarity_boost: 0.75,
    style: 0.25,
    speakingRate: 0.90,
    description: 'Practical, serious, grounded delivery',
    voiceCharacter: 'Direct, no-nonsense, focused urgency',
  },
  'The Strategist': {
    stability: 0.80,
    similarity_boost: 0.85,
    style: 0.20,
    speakingRate: 0.85,
    description: 'Structured, deliberate, authoritative',
    voiceCharacter: 'Methodical pacing, clear hierarchy of ideas',
  },
  'The MacGyver': {
    stability: 0.60,
    similarity_boost: 0.75,
    style: 0.40,
    speakingRate: 1.05,
    description: 'Resourceful, quick-thinking, adaptive',
    voiceCharacter: 'Problem-solving energy, confident improvisation',
  },
  'The Architect': {
    stability: 0.75,
    similarity_boost: 0.80,
    style: 0.35,
    speakingRate: 0.90,
    description: 'Visionary, structured, building momentum',
    voiceCharacter: 'Blueprint clarity, systematic revelation',
  },
  'The Rebel': {
    stability: 0.45,
    similarity_boost: 0.65,
    style: 0.60,
    speakingRate: 1.05,
    description: 'Chaotic, challenging, unconventional',
    voiceCharacter: 'Edge and attitude, questioning inflection',
  },
  'The Diplomat': {
    stability: 0.65,
    similarity_boost: 0.85,
    style: 0.30,
    speakingRate: 0.95,
    description: 'Balanced, warm, connecting',
    voiceCharacter: 'Bridge-building tone, inclusive warmth',
  },
  'The Provider': {
    stability: 0.70,
    similarity_boost: 0.90,
    style: 0.35,
    speakingRate: 0.90,
    description: 'Nurturing, warm, supportive',
    voiceCharacter: 'Comforting presence, gentle encouragement',
  },
  'The Storyteller': {
    stability: 0.45,
    similarity_boost: 0.85,
    style: 0.65,
    speakingRate: 1.00,
    description: 'Narrative flow, engaging, captivating',
    voiceCharacter: 'Story arc pacing, character voices, dramatic tension',
  },
  'The Mystic': {
    stability: 0.55,
    similarity_boost: 0.80,
    style: 0.55,
    speakingRate: 0.80,
    description: 'Deep, contemplative, profound',
    voiceCharacter: 'Resonant depth, meaningful pauses, wisdom weight',
  },
  'The Empath': {
    stability: 0.50,
    similarity_boost: 0.90,
    style: 0.50,
    speakingRate: 0.85,
    description: 'Emotionally attuned, warm, understanding',
    voiceCharacter: 'Feeling-forward, mirroring tone, heart connection',
  },
};

/**
 * Tone-based voice adjustments
 * These modify the base archetype settings
 */
const TONE_MODIFIERS = {
  enthusiastic: {
    stabilityMod: 0.05,      // Slightly more stable for clarity
    speakingRateMod: 1.15,   // 15% faster
    styleMod: 0.10,          // More expressive
    description: 'Upbeat, energetic, excited delivery',
  },
  serious: {
    stabilityMod: 0.10,      // More consistent
    speakingRateMod: 0.90,   // 10% slower
    styleMod: -0.10,         // Less stylized
    description: 'Grave, measured, weighty delivery',
  },
  playful: {
    stabilityMod: -0.10,     // More varied
    speakingRateMod: 1.05,   // Slightly faster
    styleMod: 0.15,          // More expressive
    description: 'Light, fun, bouncy delivery',
  },
  thoughtful: {
    stabilityMod: 0.05,      // Slightly more stable
    speakingRateMod: 0.85,   // 15% slower
    styleMod: 0.05,          // Slight style
    description: 'Reflective, contemplative, measured delivery',
  },
  neutral: {
    stabilityMod: 0.0,
    speakingRateMod: 1.0,
    styleMod: 0.0,
    description: 'Default delivery, no modifications',
  },
};

/**
 * Language-specific adjustments
 * Some languages benefit from different stability settings
 */
const LANGUAGE_ADJUSTMENTS = {
  en: { stabilityMod: 0.0, description: 'English - base settings' },
  es: { stabilityMod: 0.05, description: 'Spanish - slightly more stable for accent consistency' },
  fr: { stabilityMod: 0.05, description: 'French - slightly more stable for accent consistency' },
};

/**
 * API retry configuration
 */
const RETRY_CONFIG = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
  backoffMultiplier: 2,
};

/**
 * Default ElevenLabs voice model settings
 */
const DEFAULT_MODEL = 'eleven_multilingual_v2';

// ============================================================================
// ELEVENLABS VOICE ENGINE CLASS
// ============================================================================

export default class ElevenLabsVoiceEngine {
  /**
   * Create a new ElevenLabs Voice Engine instance
   * @param {Object} options - Configuration options
   * @param {string} options.apiKey - ElevenLabs API key (or from env)
   * @param {string} options.voiceId - Default voice ID to use
   * @param {string} options.model - ElevenLabs model (default: eleven_multilingual_v2)
   * @param {string} options.cacheDir - Directory for cached audio (browser: IndexedDB)
   */
  constructor(options = {}) {
    this.apiKey = options.apiKey || this.getApiKey();
    this.voiceId = options.voiceId || null;
    this.model = options.model || DEFAULT_MODEL;
    this.baseUrl = 'https://api.elevenlabs.io/v1';
    
    // Audio cache (Map for in-memory, IndexedDB for persistence)
    this.audioCache = new Map();
    this.cacheDbName = 'elevenlabs-audio-cache';
    this.cacheStoreName = 'audio-files';
    
    // Request tracking
    this.pendingRequests = new Map();
    this.requestHistory = [];
    
    // Callbacks
    this.onError = options.onError || null;
    this.onCacheHit = options.onCacheHit || null;
    this.onGenerated = options.onGenerated || null;
    
    // Initialize IndexedDB cache
    this.initCache();
  }

  /**
   * Get API key from environment or window config
   * @returns {string|null}
   */
  getApiKey() {
    // Check window config (for browser apps)
    if (typeof window !== 'undefined' && window.ELEVENLABS_API_KEY) {
      return window.ELEVENLABS_API_KEY;
    }
    // Check process.env (for Node.js)
    if (typeof process !== 'undefined' && process.env?.ELEVENLABS_API_KEY) {
      return process.env.ELEVENLABS_API_KEY;
    }
    return null;
  }

  /**
   * Initialize IndexedDB cache for persistent audio storage
   */
  async initCache() {
    if (typeof indexedDB === 'undefined') {
      console.warn('[ElevenLabsVoiceEngine] IndexedDB not available, using memory cache only');
      return;
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.cacheDbName, 1);
      
      request.onerror = () => {
        console.warn('[ElevenLabsVoiceEngine] Failed to open IndexedDB cache');
        resolve(); // Continue without persistence
      };
      
      request.onsuccess = (event) => {
        this.cacheDb = event.target.result;
        console.log('[ElevenLabsVoiceEngine] IndexedDB cache initialized');
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(this.cacheStoreName)) {
          const store = db.createObjectStore(this.cacheStoreName, { keyPath: 'cacheKey' });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('age', 'age', { unique: false });
          store.createIndex('archetype', 'archetype', { unique: false });
        }
      };
    });
  }

  // ==========================================================================
  // CORE VOICE SETTINGS CALCULATION
  // ==========================================================================

  /**
   * Calculate the age bucket from a numeric age
   * @param {number} age - Age in years (2-102)
   * @returns {string} Age bucket key (e.g., '18-35')
   */
  getAgeBucket(age) {
    const normalizedAge = Math.max(2, Math.min(102, age));
    
    for (const [bucket, config] of Object.entries(AGE_PITCH_MAP)) {
      if (normalizedAge >= config.minAge && normalizedAge <= config.maxAge) {
        return bucket;
      }
    }
    
    return '18-35'; // Default fallback
  }

  /**
   * Calculate pitch shift percentage for a given age
   * Uses linear interpolation within age brackets for smoother transitions
   * @param {number} age - Age in years (2-102)
   * @returns {number} Pitch shift as decimal (e.g., 0.20 for +20%)
   */
  calculatePitchFromAge(age) {
    const bucket = this.getAgeBucket(age);
    const config = AGE_PITCH_MAP[bucket];
    
    // Linear interpolation within bracket for smoother transitions
    const range = config.maxAge - config.minAge;
    const position = (age - config.minAge) / range;
    
    // Get adjacent bucket for interpolation
    const bucketKeys = Object.keys(AGE_PITCH_MAP);
    const bucketIndex = bucketKeys.indexOf(bucket);
    
    // If not at boundaries, interpolate toward next bucket
    if (bucketIndex < bucketKeys.length - 1) {
      const nextBucket = AGE_PITCH_MAP[bucketKeys[bucketIndex + 1]];
      const transitionStart = 0.7; // Start transitioning at 70% through bracket
      
      if (position > transitionStart) {
        const transitionProgress = (position - transitionStart) / (1 - transitionStart);
        const pitchDiff = nextBucket.pitchShift - config.pitchShift;
        return config.pitchShift + (pitchDiff * transitionProgress * 0.3); // 30% blend
      }
    }
    
    return config.pitchShift;
  }

  /**
   * Get voice settings for an archetype
   * @param {string} archetype - Archetype name (e.g., 'The Scientist')
   * @returns {Object} Voice settings for the archetype
   */
  getArchetypeSettings(archetype) {
    const normalized = archetype?.trim() || 'The Scientist';
    return ARCHETYPE_VOICE_SETTINGS[normalized] || ARCHETYPE_VOICE_SETTINGS['The Scientist'];
  }

  /**
   * Get tone modifier settings
   * @param {string} tone - Tone name (e.g., 'enthusiastic')
   * @returns {Object} Tone modifier settings
   */
  getToneModifiers(tone) {
    const normalized = tone?.toLowerCase()?.trim() || 'neutral';
    return TONE_MODIFIERS[normalized] || TONE_MODIFIERS.neutral;
  }

  /**
   * Get language adjustment settings
   * @param {string} language - Language code (e.g., 'en', 'es', 'fr')
   * @returns {Object} Language adjustment settings
   */
  getLanguageAdjustments(language) {
    const normalized = language?.toLowerCase()?.trim() || 'en';
    return LANGUAGE_ADJUSTMENTS[normalized] || LANGUAGE_ADJUSTMENTS.en;
  }

  /**
   * Calculate complete voice settings from age, archetype, tone, and language
   * @param {number} age - Age in years (2-102)
   * @param {string} archetype - Archetype name
   * @param {string} tone - Tone name
   * @param {string} language - Language code
   * @returns {Object} Complete voice settings for ElevenLabs API
   */
  calculateVoiceSettings(age, archetype, tone = 'neutral', language = 'en') {
    // Get base settings
    const archetypeSettings = this.getArchetypeSettings(archetype);
    const toneModifiers = this.getToneModifiers(tone);
    const languageAdjustments = this.getLanguageAdjustments(language);
    const pitchShift = this.calculatePitchFromAge(age);
    const ageBucket = this.getAgeBucket(age);

    // Combine settings with modifiers (clamped to valid ranges)
    const stability = this.clamp(
      archetypeSettings.stability + toneModifiers.stabilityMod + languageAdjustments.stabilityMod,
      0.0, 1.0
    );
    
    const similarityBoost = this.clamp(
      archetypeSettings.similarity_boost,
      0.0, 1.0
    );
    
    const style = this.clamp(
      archetypeSettings.style + toneModifiers.styleMod,
      0.0, 1.0
    );
    
    const speakingRate = this.clamp(
      archetypeSettings.speakingRate * toneModifiers.speakingRateMod,
      0.5, 2.0
    );

    return {
      // ElevenLabs API voice_settings
      voice_settings: {
        stability,
        similarity_boost: similarityBoost,
        style,
        use_speaker_boost: true,
      },
      
      // Additional metadata
      pitchShift,
      pitchShiftPercent: `${pitchShift >= 0 ? '+' : ''}${(pitchShift * 100).toFixed(0)}%`,
      speakingRate,
      ageBucket,
      
      // Debug info
      debug: {
        archetype,
        archetypeDescription: archetypeSettings.description,
        tone,
        toneDescription: toneModifiers.description,
        language,
        age,
        ageBucketDescription: AGE_PITCH_MAP[ageBucket].description,
      },
    };
  }

  /**
   * Clamp a value between min and max
   * @param {number} value - Value to clamp
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @returns {number} Clamped value
   */
  clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  // ==========================================================================
  // AUDIO GENERATION
  // ==========================================================================

  /**
   * Generate audio for a lesson phase
   * Main entry point for audio generation with all voice parameters
   * 
   * @param {string} text - Text to synthesize
   * @param {number} age - Age in years (2-102)
   * @param {string} language - Language code ('en', 'es', 'fr')
   * @param {string} archetype - Archetype name
   * @param {string} tone - Tone name
   * @param {Object} options - Additional options
   * @param {string} options.voiceId - Override default voice ID
   * @param {boolean} options.useCache - Use cached audio if available (default: true)
   * @param {string} options.cacheKey - Custom cache key
   * @returns {Promise<Object>} Generated audio result with URL and metadata
   */
  async generatePhaseAudio(text, age, language, archetype, tone, options = {}) {
    // Validate inputs
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      throw new Error('Text is required for audio generation');
    }
    
    if (!this.apiKey) {
      throw new Error('ElevenLabs API key not configured. Set ELEVENLABS_API_KEY.');
    }
    
    const voiceId = options.voiceId || this.voiceId;
    if (!voiceId) {
      throw new Error('Voice ID is required. Set in constructor or options.');
    }

    // Calculate voice settings
    const voiceSettings = this.calculateVoiceSettings(age, archetype, tone, language);
    
    // Generate cache key
    const cacheKey = options.cacheKey || this.generateCacheKey(text, age, language, archetype, tone);
    
    // Check cache first
    if (options.useCache !== false) {
      const cached = await this.getFromCache(cacheKey);
      if (cached) {
        console.log(`[ElevenLabsVoiceEngine] Cache hit: ${cacheKey}`);
        if (this.onCacheHit) {
          this.onCacheHit({ cacheKey, cached });
        }
        return {
          ...cached,
          fromCache: true,
          voiceSettings,
        };
      }
    }

    // Check for pending request with same key
    if (this.pendingRequests.has(cacheKey)) {
      console.log(`[ElevenLabsVoiceEngine] Waiting for pending request: ${cacheKey}`);
      return this.pendingRequests.get(cacheKey);
    }

    // Create generation promise
    const generationPromise = this.performGeneration(
      text, voiceId, voiceSettings, language, cacheKey
    );
    
    this.pendingRequests.set(cacheKey, generationPromise);
    
    try {
      const result = await generationPromise;
      this.pendingRequests.delete(cacheKey);
      
      // Cache the result
      await this.saveToCache(cacheKey, result);
      
      if (this.onGenerated) {
        this.onGenerated({ cacheKey, result });
      }
      
      return {
        ...result,
        fromCache: false,
        voiceSettings,
      };
    } catch (error) {
      this.pendingRequests.delete(cacheKey);
      
      // Try fallback to cached audio
      const fallback = await this.getFallbackAudio(cacheKey, age, language, archetype);
      if (fallback) {
        console.warn(`[ElevenLabsVoiceEngine] Using fallback audio for: ${cacheKey}`);
        return {
          ...fallback,
          fromCache: true,
          fallback: true,
          voiceSettings,
          originalError: error.message,
        };
      }
      
      throw error;
    }
  }

  /**
   * Perform the actual API call with retry logic
   * @private
   */
  async performGeneration(text, voiceId, voiceSettings, language, cacheKey) {
    const endpoint = `${this.baseUrl}/text-to-speech/${voiceId}`;
    
    const requestBody = {
      text,
      model_id: this.model,
      voice_settings: voiceSettings.voice_settings,
    };

    let lastError = null;
    
    for (let attempt = 0; attempt < RETRY_CONFIG.maxRetries; attempt++) {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': this.apiKey,
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorBody = await response.text();
          throw new Error(`ElevenLabs API error ${response.status}: ${errorBody}`);
        }

        // Get audio blob
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Extract metadata from response headers
        const characterCount = parseInt(response.headers.get('x-character-count') || text.length);
        
        // Track request
        this.trackRequest({
          cacheKey,
          timestamp: new Date().toISOString(),
          characterCount,
          attempt: attempt + 1,
          success: true,
        });

        return {
          audioUrl,
          audioBlob,
          text,
          characterCount,
          timestamp: new Date().toISOString(),
          voiceId,
          model: this.model,
          pitchShift: voiceSettings.pitchShift,
          ageBucket: voiceSettings.ageBucket,
          debug: voiceSettings.debug,
        };
        
      } catch (error) {
        lastError = error;
        console.warn(`[ElevenLabsVoiceEngine] Attempt ${attempt + 1} failed:`, error.message);
        
        // Track failed attempt
        this.trackRequest({
          cacheKey,
          timestamp: new Date().toISOString(),
          attempt: attempt + 1,
          success: false,
          error: error.message,
        });

        // Calculate backoff delay
        if (attempt < RETRY_CONFIG.maxRetries - 1) {
          const delay = Math.min(
            RETRY_CONFIG.baseDelayMs * Math.pow(RETRY_CONFIG.backoffMultiplier, attempt),
            RETRY_CONFIG.maxDelayMs
          );
          console.log(`[ElevenLabsVoiceEngine] Retrying in ${delay}ms...`);
          await this.sleep(delay);
        }
      }
    }

    // All retries failed
    if (this.onError) {
      this.onError({
        cacheKey,
        error: lastError,
        retriesExhausted: true,
      });
    }
    
    throw lastError;
  }

  /**
   * Generate cache key for audio
   * @private
   */
  generateCacheKey(text, age, language, archetype, tone) {
    // Create deterministic key from parameters
    const textHash = this.simpleHash(text);
    const ageBucket = this.getAgeBucket(age);
    return `${ageBucket}-${language}-${archetype}-${tone}-${textHash}`;
  }

  /**
   * Simple hash function for cache keys
   * @private
   */
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Sleep for specified milliseconds
   * @private
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Track API request for analytics
   * @private
   */
  trackRequest(data) {
    this.requestHistory.push(data);
    // Keep only last 100 requests
    if (this.requestHistory.length > 100) {
      this.requestHistory.shift();
    }
  }

  // ==========================================================================
  // CACHING
  // ==========================================================================

  /**
   * Get audio from cache
   * @param {string} cacheKey - Cache key
   * @returns {Promise<Object|null>} Cached audio data or null
   */
  async getFromCache(cacheKey) {
    // Check memory cache first
    if (this.audioCache.has(cacheKey)) {
      return this.audioCache.get(cacheKey);
    }

    // Check IndexedDB
    if (!this.cacheDb) return null;

    return new Promise((resolve) => {
      try {
        const transaction = this.cacheDb.transaction([this.cacheStoreName], 'readonly');
        const store = transaction.objectStore(this.cacheStoreName);
        const request = store.get(cacheKey);
        
        request.onsuccess = () => {
          if (request.result) {
            // Reconstruct audio URL from blob
            const audioUrl = URL.createObjectURL(request.result.audioBlob);
            const result = { ...request.result, audioUrl };
            this.audioCache.set(cacheKey, result); // Populate memory cache
            resolve(result);
          } else {
            resolve(null);
          }
        };
        
        request.onerror = () => resolve(null);
      } catch (error) {
        console.warn('[ElevenLabsVoiceEngine] Cache read error:', error);
        resolve(null);
      }
    });
  }

  /**
   * Save audio to cache
   * @param {string} cacheKey - Cache key
   * @param {Object} data - Audio data to cache
   */
  async saveToCache(cacheKey, data) {
    // Save to memory cache
    this.audioCache.set(cacheKey, data);

    // Save to IndexedDB
    if (!this.cacheDb) return;

    return new Promise((resolve) => {
      try {
        const transaction = this.cacheDb.transaction([this.cacheStoreName], 'readwrite');
        const store = transaction.objectStore(this.cacheStoreName);
        
        const cacheData = {
          cacheKey,
          ...data,
          timestamp: new Date().toISOString(),
        };
        
        const request = store.put(cacheData);
        request.onsuccess = () => resolve(true);
        request.onerror = () => resolve(false);
      } catch (error) {
        console.warn('[ElevenLabsVoiceEngine] Cache write error:', error);
        resolve(false);
      }
    });
  }

  /**
   * Get fallback audio when generation fails
   * Tries similar cached entries with different parameters
   * @private
   */
  async getFallbackAudio(originalKey, age, language, archetype) {
    const ageBucket = this.getAgeBucket(age);
    
    // Fallback strategies in order of preference
    const fallbackStrategies = [
      // 1. Same bucket, default language
      { ageBucket, language: 'en', archetype },
      // 2. Default bucket, same language
      { ageBucket: '18-35', language, archetype },
      // 3. Default everything
      { ageBucket: '18-35', language: 'en', archetype },
      // 4. Different archetype (The Scientist as safe default)
      { ageBucket: '18-35', language: 'en', archetype: 'The Scientist' },
    ];

    for (const strategy of fallbackStrategies) {
      // Search cache for matching entries
      const match = await this.searchCacheByParams(strategy);
      if (match) {
        return match;
      }
    }

    return null;
  }

  /**
   * Search cache for entries matching parameters
   * @private
   */
  async searchCacheByParams(params) {
    if (!this.cacheDb) return null;

    return new Promise((resolve) => {
      try {
        const transaction = this.cacheDb.transaction([this.cacheStoreName], 'readonly');
        const store = transaction.objectStore(this.cacheStoreName);
        const request = store.openCursor();
        
        request.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            const data = cursor.value;
            if (data.ageBucket === params.ageBucket &&
                data.debug?.language === params.language &&
                data.debug?.archetype === params.archetype) {
              const audioUrl = URL.createObjectURL(data.audioBlob);
              resolve({ ...data, audioUrl });
              return;
            }
            cursor.continue();
          } else {
            resolve(null);
          }
        };
        
        request.onerror = () => resolve(null);
      } catch (error) {
        resolve(null);
      }
    });
  }

  /**
   * Clear the audio cache
   * @param {Object} options - Clear options
   * @param {boolean} options.memoryOnly - Only clear memory cache
   * @param {string} options.olderThan - Clear entries older than ISO date string
   */
  async clearCache(options = {}) {
    // Clear memory cache
    this.audioCache.clear();
    
    if (options.memoryOnly || !this.cacheDb) return;

    return new Promise((resolve) => {
      try {
        const transaction = this.cacheDb.transaction([this.cacheStoreName], 'readwrite');
        const store = transaction.objectStore(this.cacheStoreName);
        
        if (options.olderThan) {
          // Clear entries older than specified date
          const index = store.index('timestamp');
          const range = IDBKeyRange.upperBound(options.olderThan);
          const request = index.openCursor(range);
          
          request.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
              cursor.delete();
              cursor.continue();
            } else {
              resolve(true);
            }
          };
        } else {
          // Clear all entries
          const request = store.clear();
          request.onsuccess = () => resolve(true);
        }
        
        request.onerror = () => resolve(false);
      } catch (error) {
        console.warn('[ElevenLabsVoiceEngine] Cache clear error:', error);
        resolve(false);
      }
    });
  }

  // ==========================================================================
  // UTILITY METHODS
  // ==========================================================================

  /**
   * Get all available archetypes with their settings
   * @returns {Object} Map of archetype names to settings
   */
  getArchetypes() {
    return { ...ARCHETYPE_VOICE_SETTINGS };
  }

  /**
   * Get all available tones with their modifiers
   * @returns {Object} Map of tone names to modifiers
   */
  getTones() {
    return { ...TONE_MODIFIERS };
  }

  /**
   * Get age bucket information
   * @returns {Object} Map of age buckets to pitch info
   */
  getAgeBuckets() {
    return { ...AGE_PITCH_MAP };
  }

  /**
   * Get request history for debugging/analytics
   * @returns {Array} Array of request tracking data
   */
  getRequestHistory() {
    return [...this.requestHistory];
  }

  /**
   * Estimate cost for text (characters × rate)
   * ElevenLabs charges per character
   * @param {string} text - Text to estimate
   * @returns {Object} Cost estimate
   */
  estimateCost(text) {
    const characterCount = text.length;
    // ElevenLabs pricing varies by plan, using approximate rate
    const ratePerThousandChars = 0.30; // $0.30 per 1000 chars (approximate)
    const estimatedCost = (characterCount / 1000) * ratePerThousandChars;
    
    return {
      characterCount,
      estimatedCostUSD: estimatedCost.toFixed(4),
      note: 'Estimate only. Actual cost depends on your ElevenLabs plan.',
    };
  }

  /**
   * Validate that voice ID exists (makes API call)
   * @param {string} voiceId - Voice ID to validate
   * @returns {Promise<Object>} Voice info or throws error
   */
  async validateVoiceId(voiceId) {
    if (!this.apiKey) {
      throw new Error('API key required to validate voice ID');
    }

    const response = await fetch(`${this.baseUrl}/voices/${voiceId}`, {
      headers: {
        'xi-api-key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Invalid voice ID: ${voiceId}`);
    }

    return response.json();
  }

  /**
   * Set the default voice ID
   * @param {string} voiceId - Voice ID to use
   */
  setVoiceId(voiceId) {
    this.voiceId = voiceId;
  }

  /**
   * Set the API key
   * @param {string} apiKey - ElevenLabs API key
   */
  setApiKey(apiKey) {
    this.apiKey = apiKey;
  }
}

// ============================================================================
// STANDALONE HELPER FUNCTIONS
// ============================================================================

/**
 * Calculate voice settings without instantiating the engine
 * Useful for preview/display purposes
 * @param {number} age - Age in years
 * @param {string} archetype - Archetype name
 * @param {string} tone - Tone name
 * @param {string} language - Language code
 * @returns {Object} Calculated voice settings
 */
export function calculateVoiceSettings(age, archetype, tone = 'neutral', language = 'en') {
  const engine = new ElevenLabsVoiceEngine();
  return engine.calculateVoiceSettings(age, archetype, tone, language);
}

/**
 * Get pitch shift for an age
 * @param {number} age - Age in years
 * @returns {number} Pitch shift as decimal
 */
export function getPitchForAge(age) {
  const engine = new ElevenLabsVoiceEngine();
  return engine.calculatePitchFromAge(age);
}

/**
 * Get age bucket from age
 * @param {number} age - Age in years
 * @returns {string} Age bucket key
 */
export function getAgeBucket(age) {
  const engine = new ElevenLabsVoiceEngine();
  return engine.getAgeBucket(age);
}

// Export constants for external use
export {
  AGE_PITCH_MAP,
  ARCHETYPE_VOICE_SETTINGS,
  TONE_MODIFIERS,
  LANGUAGE_ADJUSTMENTS,
  RETRY_CONFIG,
  DEFAULT_MODEL,
};


