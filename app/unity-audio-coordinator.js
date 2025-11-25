/**
 * Unity Audio Coordinator - Handles audio URL calculation and playback coordination
 * 
 * Responsibilities:
 * - Calculate audio URLs based on lesson, phase, age, language
 * - Coordinate audio loading with Unity
 * - Handle audio fallbacks when files missing
 * - Manage audio playback state
 */

/* eslint-env browser */
export default class UnityAudioCoordinator {
  constructor(unityBridge) {
    this.unityBridge = unityBridge;
    this.currentAudio = null;
    this.audioCache = new Map(); // Cache: url -> metadata
    this.loadingPromises = new Map(); // Track in-flight loads

    // Phase to audio phase name mapping
    this.phaseToAudioPhase = {
      'welcome': 'welcome',
      'teaching': 'mainContent',  // q1 maps to mainContent
      'practice': 'mainContent',   // q2, q3 map to mainContent
      'wisdom': 'wisdomMoment',
      // Legacy support
      'q1': 'mainContent',
      'q2': 'mainContent',
      'q3': 'mainContent',
      'q4': 'wisdomMoment',
    };

    // Listen for Unity responses
    this.setupListeners();
  }

  /**
   * Setup Unity bridge event listeners
   */
  setupListeners() {
    if (!this.unityBridge) return;

    window.addEventListener('message', (event) => {
      if (event.data?.type === 'unity-bridge-command') {
        const { event: eventName, payload } = event.data;
        
        if (eventName === 'audio-ready') {
          this.handleAudioReady(payload);
        } else if (eventName === 'playback-started') {
          this.handlePlaybackStarted(payload);
        } else if (eventName === 'playback-complete') {
          this.handlePlaybackComplete(payload);
        } else if (eventName === 'error' && payload.context === 'audio-load') {
          this.handleAudioError(payload);
        }
      }
    });
  }

  /**
   * Calculate audio URL for current state and phase
   */
  calculateAudioUrl(state, phase) {
    if (!state.selectedLesson) {
      console.warn('[UnityAudioCoordinator] No lesson selected');
      return null;
    }

    // Get lesson slug
    const lessonSlug = state.selectedLesson.slug || 
                       this.slugify(state.selectedLesson.topic || state.selectedLesson.title || 'unknown');
    
    // Map phase to audio phase name
    const audioPhase = this.phaseToAudioPhase[phase] || 'mainContent';
    
    // Build URL: /lessons/audio/{lesson-slug}/{ageBucket}-{language}-{audioPhase}.mp3
    const url = `/lessons/audio/${lessonSlug}/${state.ageBucket}-${state.language}-${audioPhase}.mp3`;
    
    return url;
  }

  /**
   * Load audio for current phase
   */
  async loadAudio(state, phase) {
    const url = this.calculateAudioUrl(state, phase);
    if (!url) {
      console.warn('[UnityAudioCoordinator] Cannot calculate audio URL', { state, phase });
      return null;
    }

    // Check cache
    if (this.audioCache.has(url)) {
      const cached = this.audioCache.get(url);
      console.log(`[UnityAudioCoordinator] Using cached audio: ${url}`);
      return cached;
    }

    // Check if already loading
    if (this.loadingPromises.has(url)) {
      return this.loadingPromises.get(url);
    }

    // Start loading
    const loadPromise = this.performLoad(url, state, phase);
    this.loadingPromises.set(url, loadPromise);

    try {
      const result = await loadPromise;
      this.audioCache.set(url, result);
      this.currentAudio = url;
      return result;
    } catch (error) {
      console.error(`[UnityAudioCoordinator] Failed to load audio: ${url}`, error);
      // Try fallback
      return this.loadFallbackAudio(state, phase, url);
    } finally {
      this.loadingPromises.delete(url);
    }
  }

  /**
   * Perform actual audio load via Unity bridge
   */
  async performLoad(url, state, phase) {
    if (!this.unityBridge) {
      throw new Error('Unity bridge not available');
    }

    // Emit load event to Unity
    this.unityBridge.emit('audio-load', {
      url,
      phase,
      ageBucket: state.ageBucket,
      language: state.language,
      sessionId: state.sessionId,
      timestamp: new Date().toISOString(),
    });

    // Wait for Unity confirmation (with timeout)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Audio load timeout: ${url}`));
      }, 10000); // 10 second timeout

      const handler = (event) => {
        if (event.data?.type === 'unity-bridge-command') {
          const { event: eventName, payload } = event.data;
          
          if (eventName === 'audio-ready' && payload.url === url) {
            clearTimeout(timeout);
            window.removeEventListener('message', handler);
            resolve(payload);
          } else if (eventName === 'error' && 
                     payload.context === 'audio-load' && 
                     payload.url === url) {
            clearTimeout(timeout);
            window.removeEventListener('message', handler);
            reject(new Error(payload.message || 'Audio load error'));
          }
        }
      };

      window.addEventListener('message', handler);
    });
  }

  /**
   * Handle audio ready confirmation from Unity
   */
  handleAudioReady(payload) {
    const { url, duration } = payload;
    console.log(`[UnityAudioCoordinator] Audio ready: ${url} (${duration}s)`);
    this.audioCache.set(url, payload);
  }

  /**
   * Handle playback started from Unity
   */
  handlePlaybackStarted(payload) {
    const { url, phase } = payload;
    console.log(`[UnityAudioCoordinator] Playback started: ${phase} - ${url}`);
    this.currentAudio = url;
  }

  /**
   * Handle playback complete from Unity
   */
  handlePlaybackComplete(payload) {
    const { url, phase } = payload;
    console.log(`[UnityAudioCoordinator] Playback complete: ${phase} - ${url}`);
    
    // Emit custom event for app to handle
    window.dispatchEvent(new CustomEvent('unity-audio-complete', {
      detail: { url, phase }
    }));
  }

  /**
   * Handle audio load error from Unity
   */
  handleAudioError(payload) {
    const { url, message } = payload;
    console.error(`[UnityAudioCoordinator] Audio load error: ${url}`, message);
  }

  /**
   * Load fallback audio when primary fails
   */
  async loadFallbackAudio(state, phase, originalUrl) {
    console.log(`[UnityAudioCoordinator] Attempting fallback audio for: ${originalUrl}`);

    // Fallback strategy: Try different language/age combinations
    const fallbackStrategies = [
      // 1. Same age, default language (en)
      { ...state, language: 'en' },
      // 2. Default age (18-35), same language
      { ...state, ageBucket: '18-35' },
      // 3. Default age and language
      { ...state, ageBucket: '18-35', language: 'en' },
    ];

    for (const fallbackState of fallbackStrategies) {
      const fallbackUrl = this.calculateAudioUrl(fallbackState, phase);
      
      if (fallbackUrl && fallbackUrl !== originalUrl) {
        console.log(`[UnityAudioCoordinator] Trying fallback: ${fallbackUrl}`);
        
        try {
          const result = await this.loadAudio(fallbackState, phase);
          if (result) {
            console.log(`[UnityAudioCoordinator] Fallback audio loaded: ${fallbackUrl}`);
            return result;
          }
        } catch (error) {
          console.debug(`[UnityAudioCoordinator] Fallback failed: ${fallbackUrl}`, error);
          continue;
        }
      }
    }

    // All fallbacks failed
    console.warn(`[UnityAudioCoordinator] All fallback strategies failed for phase: ${phase}`);
    
    // Show user-friendly message
    window.dispatchEvent(new CustomEvent('unity-audio-unavailable', {
      detail: { phase, originalUrl }
    }));

    return null;
  }

  /**
   * Handle real-time state changes (age/language)
   */
  async updateAudioForStateChange(state, changeType) {
    const currentPhase = state.currentPhase;
    
    if (changeType === 'language' || changeType === 'age') {
      console.log(`[UnityAudioCoordinator] State change (${changeType}), reloading audio for phase: ${currentPhase}`);
      
      // Clear current audio cache for this phase
      const currentUrl = this.calculateAudioUrl(state, currentPhase);
      if (currentUrl) {
        this.audioCache.delete(currentUrl);
      }
      
      // Reload audio with new state
      return this.loadAudio(state, currentPhase);
    }
    
    return null;
  }

  /**
   * Preload audio for next phase (performance optimization)
   */
  async preloadNextPhase(state, currentPhase) {
    const phaseOrder = ['welcome', 'teaching', 'practice', 'wisdom'];
    const currentIndex = phaseOrder.indexOf(currentPhase);
    
    if (currentIndex === -1 || currentIndex >= phaseOrder.length - 1) {
      return; // No next phase
    }

    const nextPhase = phaseOrder[currentIndex + 1];
    const nextUrl = this.calculateAudioUrl(state, nextPhase);
    
    if (nextUrl && !this.audioCache.has(nextUrl) && !this.loadingPromises.has(nextUrl)) {
      console.log(`[UnityAudioCoordinator] Preloading next phase audio: ${nextPhase}`);
      // Load in background (don't await)
      this.loadAudio(state, nextPhase).catch(err => {
        console.debug(`[UnityAudioCoordinator] Preload failed for ${nextPhase}:`, err);
      });
    }
  }

  /**
   * Get current audio URL
   */
  getCurrentAudio() {
    return this.currentAudio;
  }

  /**
   * Check if audio is loaded for URL
   */
  isAudioLoaded(url) {
    return this.audioCache.has(url);
  }

  /**
   * Utility: Convert text to URL-friendly slug
   */
  slugify(text) {
    if (!text) return 'unknown';
    
    return text.toLowerCase()
      .replace(/[^\w\s-]/g, '') // Remove special chars
      .replace(/\s+/g, '-')      // Replace spaces with hyphens
      .replace(/-+/g, '-')      // Replace multiple hyphens with single
      .trim();
  }
}

