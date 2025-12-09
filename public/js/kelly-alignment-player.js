/**
 * Kelly Alignment Player
 * 
 * Integrates pre-computed lipsync alignments with the lesson player.
 * Fetches alignment data from Supabase and plays synchronized blendshapes.
 * 
 * Usage:
 *   const player = new KellyAlignmentPlayer();
 *   await player.loadAlignment(dayNumber, ageBucket, language, phase);
 *   player.playWithAudio(audioElement);
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const ALIGNMENT_API_URL = '/api/lipsync-alignment';
const FALLBACK_TO_REALTIME = true;

// =============================================================================
// ALIGNMENT CACHE
// =============================================================================

const alignmentCache = new Map();

function getCacheKey(day, age, lang, phase) {
  return `${day}_${age}_${lang}_${phase}`;
}

// =============================================================================
// KELLY ALIGNMENT PLAYER
// =============================================================================

class KellyAlignmentPlayer {
  constructor(options = {}) {
    this.options = {
      fps: 30,
      sendToUnity: true,
      sendTo2D: true,
      fallbackToRealtime: FALLBACK_TO_REALTIME,
      ...options,
    };
    
    this.currentAlignment = null;
    this.currentTimeline = null;
    this.timelineIndex = 0;
    this.isPlaying = false;
    this.animationFrameId = null;
    this.audioElement = null;
    
    // Blendshapes state
    this.currentBlendshapes = this.getRestingFace();
    
    // Callbacks
    this.onBlendshapesUpdate = null;
    this.onPlaybackStart = null;
    this.onPlaybackEnd = null;
    
    console.log('[KellyAlignmentPlayer] Initialized');
  }
  
  // ===========================================================================
  // LOADING
  // ===========================================================================
  
  /**
   * Load alignment data from API
   * @param {number} day - Day number (1-365)
   * @param {string} ageBucket - Age bucket (e.g., '6-12')
   * @param {string} language - Language code (e.g., 'en')
   * @param {string} phase - Phase type (e.g., 'script', 'response_A')
   * @returns {Promise<boolean>} Whether alignment was loaded successfully
   */
  async loadAlignment(day, ageBucket, language = 'en', phase = 'script') {
    const cacheKey = getCacheKey(day, ageBucket, language, phase);
    
    // Check cache first
    if (alignmentCache.has(cacheKey)) {
      this.currentAlignment = alignmentCache.get(cacheKey);
      this.currentTimeline = this.currentAlignment.blendshapeTimeline;
      console.log(`[KellyAlignmentPlayer] Loaded from cache: ${cacheKey}`);
      return true;
    }
    
    try {
      const url = `${ALIGNMENT_API_URL}?day=${day}&age=${ageBucket}&lang=${language}&phase=${phase}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        if (response.status === 404) {
          console.warn(`[KellyAlignmentPlayer] No alignment for ${cacheKey}`);
          return false;
        }
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      this.currentAlignment = data;
      this.currentTimeline = data.blendshapeTimeline || [];
      
      // Cache it
      alignmentCache.set(cacheKey, data);
      
      console.log(`[KellyAlignmentPlayer] Loaded alignment: ${cacheKey}`);
      console.log(`  - Words: ${data.words?.length || 0}`);
      console.log(`  - Phones: ${data.phones?.length || 0}`);
      console.log(`  - Frames: ${this.currentTimeline.length}`);
      console.log(`  - Duration: ${data.duration}s`);
      console.log(`  - Method: ${data.method}`);
      
      return true;
      
    } catch (error) {
      console.error('[KellyAlignmentPlayer] Load error:', error);
      return false;
    }
  }
  
  /**
   * Check if alignment is available for given parameters
   */
  hasAlignment() {
    return this.currentTimeline && this.currentTimeline.length > 0;
  }
  
  // ===========================================================================
  // PLAYBACK
  // ===========================================================================
  
  /**
   * Start playback synchronized with an audio element
   * @param {HTMLAudioElement} audioElement - Audio element to sync with
   */
  playWithAudio(audioElement) {
    if (!this.hasAlignment()) {
      console.warn('[KellyAlignmentPlayer] No alignment loaded, using realtime fallback');
      if (this.options.fallbackToRealtime && window.KellyLipSync) {
        window.KellyLipSync.startFromAudioElement(audioElement);
      }
      return;
    }
    
    this.audioElement = audioElement;
    this.timelineIndex = 0;
    
    // Event handlers
    const onPlay = () => {
      this.isPlaying = true;
      this.startPlayback();
      if (this.onPlaybackStart) this.onPlaybackStart();
    };
    
    const onPause = () => {
      // Don't stop immediately - allow for resume
    };
    
    const onSeeked = () => {
      // Sync timeline index to current time
      if (this.currentTimeline) {
        const currentTime = audioElement.currentTime;
        this.timelineIndex = Math.floor(currentTime * this.options.fps);
      }
    };
    
    const onEnded = () => {
      this.stop();
      if (this.onPlaybackEnd) this.onPlaybackEnd();
    };
    
    const onTimeUpdate = () => {
      if (!this.isPlaying) return;
      // Keep timeline synced to audio time
      const currentTime = audioElement.currentTime;
      this.timelineIndex = Math.floor(currentTime * this.options.fps);
    };
    
    // Remove any existing listeners
    this.cleanupAudioListeners();
    
    // Add listeners
    audioElement.addEventListener('play', onPlay);
    audioElement.addEventListener('pause', onPause);
    audioElement.addEventListener('seeked', onSeeked);
    audioElement.addEventListener('ended', onEnded);
    audioElement.addEventListener('timeupdate', onTimeUpdate);
    
    // Store for cleanup
    this._audioListeners = {
      onPlay, onPause, onSeeked, onEnded, onTimeUpdate, audioElement
    };
    
    // If audio is already playing, start immediately
    if (!audioElement.paused) {
      this.isPlaying = true;
      this.startPlayback();
    }
    
    console.log('[KellyAlignmentPlayer] Bound to audio element');
  }
  
  /**
   * Start the playback animation loop
   * @private
   */
  startPlayback() {
    if (this.animationFrameId) return;
    
    const updateFrame = () => {
      if (!this.isPlaying) {
        this.animationFrameId = null;
        return;
      }
      
      // Get current frame from timeline
      if (this.currentTimeline && this.timelineIndex < this.currentTimeline.length) {
        const frame = this.currentTimeline[this.timelineIndex];
        
        if (frame && frame.blendshapes) {
          // Smooth transition
          this.currentBlendshapes = this.smoothBlendshapes(
            this.currentBlendshapes,
            frame.blendshapes
          );
          
          // Send to outputs
          this.sendBlendshapes(this.currentBlendshapes);
        }
      } else {
        // Beyond timeline - decay to rest
        this.currentBlendshapes = this.smoothBlendshapes(
          this.currentBlendshapes,
          this.getRestingFace()
        );
        this.sendBlendshapes(this.currentBlendshapes);
      }
      
      this.animationFrameId = requestAnimationFrame(updateFrame);
    };
    
    updateFrame();
  }
  
  /**
   * Stop playback
   */
  stop() {
    this.isPlaying = false;
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Reset to resting face
    this.currentBlendshapes = this.getRestingFace();
    this.sendBlendshapes(this.currentBlendshapes);
    
    this.cleanupAudioListeners();
    
    console.log('[KellyAlignmentPlayer] Stopped');
  }
  
  // ===========================================================================
  // OUTPUT
  // ===========================================================================
  
  /**
   * Send blendshapes to outputs
   * @private
   */
  sendBlendshapes(blendshapes) {
    // Callback
    if (this.onBlendshapesUpdate) {
      this.onBlendshapesUpdate(blendshapes);
    }
    
    // Unity
    if (this.options.sendToUnity && window.unityInstance) {
      try {
        window.unityInstance.SendMessage('kelly_fbx_v4', 'SetBlendshapes', JSON.stringify(blendshapes));
      } catch (e) {
        // Unity not loaded
      }
    }
    
    // 2D Avatar
    if (this.options.sendTo2D) {
      if (window.KellyPoseManager && window.KellyPoseManager.setMouthState) {
        const jawOpen = blendshapes.jawOpen || 0;
        if (jawOpen > 30) {
          window.KellyPoseManager.setMouthState('speaking');
        } else if (jawOpen > 10) {
          window.KellyPoseManager.setMouthState('talking');
        } else {
          window.KellyPoseManager.setMouthState('idle');
        }
      }
    }
  }
  
  // ===========================================================================
  // UTILITIES
  // ===========================================================================
  
  /**
   * Get resting face blendshapes
   */
  getRestingFace() {
    return {
      jawOpen: 0,
      mouthOpen: 0,
      mouthFunnel: 0,
      mouthPucker: 0,
      mouthStretchLeft: 0,
      mouthStretchRight: 0,
      mouthSmileLeft: 15,
      mouthSmileRight: 15,
      mouthPressLeft: 0,
      mouthPressRight: 0,
      mouthUpperUpLeft: 0,
      mouthUpperUpRight: 0,
      mouthLowerDownLeft: 0,
      mouthLowerDownRight: 0,
      mouthClose: 15,
    };
  }
  
  /**
   * Smooth transition between blendshape states
   */
  smoothBlendshapes(from, to) {
    const smoothing = 0.3;
    const result = {};
    const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
    
    for (const key of allKeys) {
      const fromValue = from[key] || 0;
      const toValue = to[key] || 0;
      result[key] = fromValue + (toValue - fromValue) * (1 - smoothing);
    }
    
    return result;
  }
  
  /**
   * Clean up audio element listeners
   */
  cleanupAudioListeners() {
    if (this._audioListeners) {
      const { onPlay, onPause, onSeeked, onEnded, onTimeUpdate, audioElement } = this._audioListeners;
      audioElement.removeEventListener('play', onPlay);
      audioElement.removeEventListener('pause', onPause);
      audioElement.removeEventListener('seeked', onSeeked);
      audioElement.removeEventListener('ended', onEnded);
      audioElement.removeEventListener('timeupdate', onTimeUpdate);
      this._audioListeners = null;
    }
  }
  
  /**
   * Get current alignment info
   */
  getAlignmentInfo() {
    if (!this.currentAlignment) return null;
    
    return {
      words: this.currentAlignment.words?.length || 0,
      phones: this.currentAlignment.phones?.length || 0,
      frames: this.currentTimeline?.length || 0,
      duration: this.currentAlignment.duration,
      method: this.currentAlignment.method,
      confidence: this.currentAlignment.confidence,
    };
  }
  
  /**
   * Preload alignments for multiple segments
   */
  async preloadAlignments(segments) {
    const promises = segments.map(s => 
      this.loadAlignment(s.day, s.ageBucket, s.language, s.phase)
        .catch(() => false)
    );
    
    const results = await Promise.all(promises);
    const loaded = results.filter(Boolean).length;
    
    console.log(`[KellyAlignmentPlayer] Preloaded ${loaded}/${segments.length} alignments`);
    return loaded;
  }
  
  /**
   * Clear the alignment cache
   */
  clearCache() {
    alignmentCache.clear();
    console.log('[KellyAlignmentPlayer] Cache cleared');
  }
  
  /**
   * Dispose of all resources
   */
  dispose() {
    this.stop();
    this.currentAlignment = null;
    this.currentTimeline = null;
    console.log('[KellyAlignmentPlayer] Disposed');
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyAlignmentPlayer = KellyAlignmentPlayer;
}

// =============================================================================
// AUTO-INITIALIZE WITH LESSON PLAYER
// =============================================================================

// If KellyLessonPlayer exists, extend it to use alignments
if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    // Create global instance for easy access
    window.kellyAlignmentPlayer = new KellyAlignmentPlayer();
    console.log('[KellyAlignmentPlayer] Global instance ready as window.kellyAlignmentPlayer');
  });
}


