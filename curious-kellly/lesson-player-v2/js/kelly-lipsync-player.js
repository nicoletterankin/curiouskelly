/**
 * Kelly Lip-Sync Player for Lesson Player v2
 * 
 * Loads and plays pre-computed lip-sync data from lipsync.json files.
 * Converts viseme keyframes to Unity blendshapes and 2D avatar states.
 * 
 * This bridges:
 * - Pre-generated lipsync.json files (from golden-lesson-hd pipeline)
 * - Unity 3D avatar (via SendMessage)
 * - 2D avatar (via Kelly2DAvatar or KellyPoseManager)
 * - KellyLipSync real-time fallback (when no pre-computed data)
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// VISEME TO BLENDSHAPE MAPPING
// =============================================================================

/**
 * Maps CC4/iClone viseme names to ARKit-compatible blendshapes
 * This matches the output format from kelly-lipsync-engine.ts
 */
const VISEME_TO_BLENDSHAPE = {
  // Core visemes from the lipsync.json files
  'V_Wide': { jawOpen: 60, mouthOpen: 50, mouthStretchLeft: 40, mouthStretchRight: 40 },
  'V_Open': { jawOpen: 80, mouthOpen: 70, mouthLowerDownLeft: 30, mouthLowerDownRight: 30 },
  'V_Tight_O': { jawOpen: 30, mouthFunnel: 60, mouthPucker: 40 },
  'V_Explosive': { jawOpen: 10, mouthPressLeft: 70, mouthPressRight: 70, mouthClose: 50 },
  'V_Dental_Lip': { jawOpen: 20, mouthUpperUpLeft: 40, mouthUpperUpRight: 40, mouthLowerDownLeft: 20, mouthLowerDownRight: 20 },
  
  // Standard viseme set (OVR Lip Sync / ARKit)
  'V_Sil': { jawOpen: 0, mouthClose: 20 },
  'V_PP': { mouthPressLeft: 80, mouthPressRight: 80, mouthClose: 60 },
  'V_FF': { mouthUpperUpLeft: 50, mouthUpperUpRight: 50, mouthLowerDownLeft: 30, mouthLowerDownRight: 30 },
  'V_TH': { jawOpen: 15, mouthUpperUpLeft: 30, mouthUpperUpRight: 30 },
  'V_DD': { jawOpen: 25, mouthStretchLeft: 20, mouthStretchRight: 20 },
  'V_kk': { jawOpen: 15, mouthFunnel: 20 },
  'V_CH': { jawOpen: 20, mouthFunnel: 50, mouthPucker: 30 },
  'V_SS': { jawOpen: 10, mouthStretchLeft: 40, mouthStretchRight: 40 },
  'V_nn': { jawOpen: 20, mouthPressLeft: 30, mouthPressRight: 30 },
  'V_RR': { jawOpen: 25, mouthFunnel: 40 },
  'V_aa': { jawOpen: 70, mouthOpen: 60, mouthStretchLeft: 30, mouthStretchRight: 30 },
  'V_E': { jawOpen: 40, mouthStretchLeft: 50, mouthStretchRight: 50 },
  'V_ih': { jawOpen: 30, mouthStretchLeft: 40, mouthStretchRight: 40 },
  'V_oh': { jawOpen: 50, mouthFunnel: 50 },
  'V_ou': { jawOpen: 40, mouthPucker: 60, mouthFunnel: 40 },
};

/**
 * Resting face blendshapes (neutral position)
 */
const RESTING_FACE = {
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

// =============================================================================
// KELLY LIPSYNC PLAYER CLASS
// =============================================================================

class KellyLipSyncPlayer {
  constructor(options = {}) {
    this.options = {
      fps: 30,
      smoothing: 0.35, // Higher = smoother but less responsive
      sendToUnity: true,
      sendTo2D: true,
      fallbackToRealtime: true,
      basePath: '/generated-videos/golden-lesson-hd', // Default path to lipsync files
      ...options,
    };
    
    // State
    this.currentTrack = null;
    this.keyframes = [];
    this.keyframeIndex = 0;
    this.isPlaying = false;
    this.isPaused = false;
    this.animationFrameId = null;
    
    // Audio binding
    this.audioElement = null;
    this._audioListeners = null;
    
    // Blendshapes
    this.currentBlendshapes = { ...RESTING_FACE };
    this.targetBlendshapes = { ...RESTING_FACE };
    
    // Cache for loaded tracks
    this._cache = new Map();
    
    // Callbacks
    this.onBlendshapesUpdate = null;
    this.onPlaybackStart = null;
    this.onPlaybackEnd = null;
    
    console.log('[KellyLipSyncPlayer] Initialized');
  }
  
  // ===========================================================================
  // TRACK LOADING
  // ===========================================================================
  
  /**
   * Build the path to a lipsync.json file
   * @param {number} dayNumber - Lesson day number
   * @param {string} phase - Phase name (Hook, Fact1, Fact2, Fact3, Wisdom)
   * @param {string} archetype - Archetype name (The_Explorer, The_Rebel, The_Scientist)
   * @returns {string} Path to lipsync.json
   */
  buildLipSyncPath(dayNumber, phase, archetype = 'The_Explorer') {
    // Normalize archetype name (replace spaces with underscores)
    const archetypeKey = archetype.replace(/\s+/g, '_');
    
    // Normalize phase name (capitalize first letter)
    const phaseKey = phase.charAt(0).toUpperCase() + phase.slice(1);
    
    // Format: day_XXX_Phase_Archetype/lipsync.json
    const dayStr = String(dayNumber).padStart(3, '0');
    const folder = `day_${dayStr}_${phaseKey}_${archetypeKey}`;
    
    return `${this.options.basePath}/${folder}/lipsync.json`;
  }
  
  /**
   * Load a lip-sync track from file
   * @param {number} dayNumber - Lesson day number  
   * @param {string} phase - Phase name
   * @param {string} archetype - Archetype name
   * @returns {Promise<boolean>} Whether track was loaded
   */
  async loadTrack(dayNumber, phase, archetype = 'The Explorer') {
    const path = this.buildLipSyncPath(dayNumber, phase, archetype);
    const cacheKey = `${dayNumber}_${phase}_${archetype}`;
    
    // Check cache
    if (this._cache.has(cacheKey)) {
      this.currentTrack = this._cache.get(cacheKey);
      this.keyframes = this.currentTrack.keyframes || [];
      console.log(`[KellyLipSyncPlayer] Loaded from cache: ${cacheKey}`);
      return true;
    }
    
    try {
      console.log(`[KellyLipSyncPlayer] Loading: ${path}`);
      const response = await fetch(path);
      
      if (!response.ok) {
        console.warn(`[KellyLipSyncPlayer] No lipsync at ${path}`);
        return false;
      }
      
      const data = await response.json();
      
      // Validate structure
      if (!data.keyframes || !Array.isArray(data.keyframes)) {
        console.warn('[KellyLipSyncPlayer] Invalid lipsync.json structure');
        return false;
      }
      
      // Store track
      this.currentTrack = data;
      this.keyframes = data.keyframes;
      this._cache.set(cacheKey, data);
      
      console.log(`[KellyLipSyncPlayer] Loaded: ${data.keyframes.length} keyframes, ${data.duration?.toFixed(2)}s`);
      return true;
      
    } catch (error) {
      console.error('[KellyLipSyncPlayer] Load error:', error);
      return false;
    }
  }
  
  /**
   * Check if a track is loaded
   */
  hasTrack() {
    return this.keyframes && this.keyframes.length > 0;
  }
  
  /**
   * Preload multiple tracks
   * @param {Array} tracks - Array of {dayNumber, phase, archetype}
   * @returns {Promise<number>} Number of tracks loaded
   */
  async preloadTracks(tracks) {
    const results = await Promise.all(
      tracks.map(t => this.loadTrack(t.dayNumber, t.phase, t.archetype).catch(() => false))
    );
    const loaded = results.filter(Boolean).length;
    console.log(`[KellyLipSyncPlayer] Preloaded ${loaded}/${tracks.length} tracks`);
    return loaded;
  }
  
  // ===========================================================================
  // PLAYBACK
  // ===========================================================================
  
  /**
   * Start playback synchronized with an audio element
   * @param {HTMLAudioElement} audioElement - Audio element to sync with
   */
  playWithAudio(audioElement) {
    this.audioElement = audioElement;
    this.keyframeIndex = 0;
    
    // If no track loaded, try realtime fallback
    if (!this.hasTrack()) {
      console.warn('[KellyLipSyncPlayer] No track loaded, using realtime fallback');
      if (this.options.fallbackToRealtime && window.KellyLipSync) {
        window.KellyLipSync.startFromAudioElement(audioElement);
      }
      return;
    }
    
    // Remove existing listeners
    this._cleanupAudioListeners();
    
    // Create listeners
    const onPlay = () => {
      this.isPlaying = true;
      this.isPaused = false;
      this._startPlaybackLoop();
      if (this.onPlaybackStart) this.onPlaybackStart();
    };
    
    const onPause = () => {
      this.isPaused = true;
    };
    
    const onSeeked = () => {
      // Resync keyframe index to audio time
      if (this.currentTrack) {
        const currentTime = audioElement.currentTime;
        this._seekToTime(currentTime);
      }
    };
    
    const onEnded = () => {
      this.stop();
      if (this.onPlaybackEnd) this.onPlaybackEnd();
    };
    
    const onTimeUpdate = () => {
      if (!this.isPlaying || this.isPaused) return;
      // Keep synced to audio time
      const currentTime = audioElement.currentTime;
      this._updateBlendshapesForTime(currentTime);
    };
    
    // Add listeners
    audioElement.addEventListener('play', onPlay);
    audioElement.addEventListener('pause', onPause);
    audioElement.addEventListener('seeked', onSeeked);
    audioElement.addEventListener('ended', onEnded);
    audioElement.addEventListener('timeupdate', onTimeUpdate);
    
    // Store for cleanup
    this._audioListeners = { onPlay, onPause, onSeeked, onEnded, onTimeUpdate, audioElement };
    
    // If already playing, start immediately
    if (!audioElement.paused) {
      onPlay();
    }
    
    console.log('[KellyLipSyncPlayer] Bound to audio element');
  }
  
  /**
   * Start the playback animation loop
   * @private
   */
  _startPlaybackLoop() {
    if (this.animationFrameId) return;
    
    const fps = this.options.fps;
    const frameInterval = 1000 / fps;
    let lastFrameTime = performance.now();
    
    const update = (now) => {
      if (!this.isPlaying) {
        this.animationFrameId = null;
        return;
      }
      
      const elapsed = now - lastFrameTime;
      
      // Throttle to target FPS
      if (elapsed >= frameInterval) {
        lastFrameTime = now - (elapsed % frameInterval);
        
        // Smooth transition to target
        this.currentBlendshapes = this._smoothBlendshapes(
          this.currentBlendshapes,
          this.targetBlendshapes
        );
        
        // Send to outputs
        this._sendBlendshapes(this.currentBlendshapes);
      }
      
      this.animationFrameId = requestAnimationFrame(update);
    };
    
    this.animationFrameId = requestAnimationFrame(update);
  }
  
  /**
   * Update blendshapes for a specific time
   * @private
   */
  _updateBlendshapesForTime(time) {
    if (!this.keyframes || this.keyframes.length === 0) {
      this.targetBlendshapes = { ...RESTING_FACE };
      return;
    }
    
    // Find active keyframe(s) for this time
    const blendshapes = { ...RESTING_FACE };
    
    for (const keyframe of this.keyframes) {
      const keyframeEnd = keyframe.time + (keyframe.duration || 0.1);
      
      if (time >= keyframe.time && time < keyframeEnd) {
        // Calculate interpolation within keyframe (bell curve)
        const progress = (time - keyframe.time) / (keyframe.duration || 0.1);
        const intensity = Math.sin(progress * Math.PI); // 0 -> 1 -> 0
        
        // Convert visemes to blendshapes
        if (keyframe.visemes) {
          for (const [visemeName, weight] of Object.entries(keyframe.visemes)) {
            const mapping = VISEME_TO_BLENDSHAPE[visemeName];
            if (mapping) {
              for (const [shapeName, baseValue] of Object.entries(mapping)) {
                const scaledValue = (baseValue * weight / 100) * intensity;
                blendshapes[shapeName] = Math.max(blendshapes[shapeName] || 0, scaledValue);
              }
            }
          }
        }
      }
    }
    
    this.targetBlendshapes = blendshapes;
  }
  
  /**
   * Seek to a specific time (update keyframe index)
   * @private
   */
  _seekToTime(time) {
    // Binary search would be faster but linear is fine for our data size
    this.keyframeIndex = 0;
    for (let i = 0; i < this.keyframes.length; i++) {
      if (this.keyframes[i].time > time) break;
      this.keyframeIndex = i;
    }
    this._updateBlendshapesForTime(time);
  }
  
  /**
   * Stop playback
   */
  stop() {
    this.isPlaying = false;
    this.isPaused = false;
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Decay to resting face
    this.targetBlendshapes = { ...RESTING_FACE };
    this._animateToRest();
    
    this._cleanupAudioListeners();
    
    console.log('[KellyLipSyncPlayer] Stopped');
  }
  
  /**
   * Animate smoothly back to resting face
   * @private
   */
  _animateToRest() {
    const decay = () => {
      let atRest = true;
      
      for (const key of Object.keys(this.currentBlendshapes)) {
        const current = this.currentBlendshapes[key];
        const target = RESTING_FACE[key] || 0;
        
        if (Math.abs(current - target) > 0.5) {
          atRest = false;
          this.currentBlendshapes[key] = current + (target - current) * 0.15;
        } else {
          this.currentBlendshapes[key] = target;
        }
      }
      
      this._sendBlendshapes(this.currentBlendshapes);
      
      if (!atRest) {
        requestAnimationFrame(decay);
      }
    };
    
    requestAnimationFrame(decay);
  }
  
  // ===========================================================================
  // OUTPUT
  // ===========================================================================
  
  /**
   * Smooth transition between blendshape states
   * @private
   */
  _smoothBlendshapes(from, to) {
    const result = {};
    const smoothing = this.options.smoothing;
    const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
    
    for (const key of allKeys) {
      const fromValue = from[key] || 0;
      const toValue = to[key] || 0;
      result[key] = fromValue + (toValue - fromValue) * (1 - smoothing);
    }
    
    return result;
  }
  
  /**
   * Send blendshapes to all outputs
   * @private
   */
  _sendBlendshapes(blendshapes) {
    // Callback
    if (this.onBlendshapesUpdate) {
      this.onBlendshapesUpdate(blendshapes);
    }
    
    // Unity 3D
    if (this.options.sendToUnity && window.unityInstance) {
      try {
        window.unityInstance.SendMessage('kelly_fbx_v4', 'SetBlendshapes', JSON.stringify(blendshapes));
      } catch (e) {
        // Unity not loaded
      }
    }
    
    // 2D Avatar systems
    if (this.options.sendTo2D) {
      const jawOpen = blendshapes.jawOpen || 0;
      
      // KellyPoseManager
      if (window.KellyPoseManager && window.KellyPoseManager.setMouthState) {
        if (jawOpen > 30) {
          window.KellyPoseManager.setMouthState('speaking');
        } else if (jawOpen > 10) {
          window.KellyPoseManager.setMouthState('talking');
        } else {
          window.KellyPoseManager.setMouthState('idle');
        }
      }
      
      // Kelly2DAvatar
      if (window.kelly2DAvatar && window.kelly2DAvatar.setMouthOpenness) {
        window.kelly2DAvatar.setMouthOpenness(jawOpen / 100);
      }
    }
  }
  
  // ===========================================================================
  // CLEANUP
  // ===========================================================================
  
  /**
   * Clean up audio element listeners
   * @private
   */
  _cleanupAudioListeners() {
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
   * Clear the track cache
   */
  clearCache() {
    this._cache.clear();
    console.log('[KellyLipSyncPlayer] Cache cleared');
  }
  
  /**
   * Get current track info
   */
  getTrackInfo() {
    if (!this.currentTrack) return null;
    return {
      duration: this.currentTrack.duration,
      fps: this.currentTrack.fps || this.options.fps,
      keyframes: this.keyframes.length,
      metadata: this.currentTrack.metadata
    };
  }
  
  /**
   * Dispose of resources
   */
  dispose() {
    this.stop();
    this.clearCache();
    this.currentTrack = null;
    this.keyframes = [];
    console.log('[KellyLipSyncPlayer] Disposed');
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyLipSyncPlayer = KellyLipSyncPlayer;
  
  // Create global instance on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      window.kellyLipSyncPlayer = new KellyLipSyncPlayer();
      console.log('[KellyLipSyncPlayer] Global instance ready as window.kellyLipSyncPlayer');
    });
  } else {
    window.kellyLipSyncPlayer = new KellyLipSyncPlayer();
    console.log('[KellyLipSyncPlayer] Global instance ready as window.kellyLipSyncPlayer');
  }
}

// ES Module export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyLipSyncPlayer;
}






