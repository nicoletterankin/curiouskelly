/**
 * Kelly Lip-Sync Orchestrator
 * 
 * Unified lip-sync system that combines:
 * - Forced alignment (for pre-rendered content)
 * - Real-time audio analysis (for conversations)
 * - Expression system integration (eyes, brows, emotions)
 * 
 * @module kelly-lipsync-orchestrator
 */

import { 
  generateBlendshapeTimeline, 
  getBlendshapesForPhoneme,
  interpolateBlendshapes,
  applyCoarticulation,
} from './phoneme-viseme-map.js';

import { 
  RealtimeLipSync, 
  AudioElementLipSync, 
  StreamingLipSync 
} from './realtime-lipsync.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Default orchestrator configuration
 */
export const DEFAULT_ORCHESTRATOR_CONFIG = {
  // Lip-sync method priority: 'alignment' > 'realtime' > 'estimation'
  preferredMethod: 'alignment',
  
  // Enable expression overlay (eyes, brows)
  enableExpressions: true,
  
  // Frame rate for blendshape updates
  fps: 30,
  
  // Blend ratio between lip-sync and expressions (0-1)
  // Higher = more lip-sync influence on mouth
  lipSyncWeight: 0.9,
  
  // Enable smooth transitions between phonemes
  smoothTransitions: true,
  
  // Transition duration in ms
  transitionDuration: 50,
  
  // API endpoint for alignment service
  alignmentApiUrl: '/api/align',
  
  // Cache alignments for reuse
  cacheAlignments: true,
  
  // Max cache size
  maxCacheSize: 50,
};

// =============================================================================
// KELLY LIPSYNC ORCHESTRATOR
// =============================================================================

/**
 * Main orchestrator class for Kelly lip-sync
 */
export class KellyLipSyncOrchestrator {
  /**
   * Create the lip-sync orchestrator
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    this.config = { ...DEFAULT_ORCHESTRATOR_CONFIG, ...options };
    
    // Lip-sync engines
    this.realtimeLipSync = null;
    this.streamingLipSync = null;
    
    // State
    this.isActive = false;
    this.currentMode = null; // 'alignment', 'realtime', 'streaming'
    this.currentAlignment = null;
    this.currentTimeline = null;
    this.timelineIndex = 0;
    
    // Playback
    this.playbackStartTime = 0;
    this.playbackDuration = 0;
    this.animationFrameId = null;
    
    // Expression overlay
    this.currentExpression = null;
    this.expressionCallback = null;
    
    // Callbacks
    this.onBlendshapesUpdate = null;
    this.onPlaybackComplete = null;
    this.onError = null;
    
    // Cache
    this.alignmentCache = new Map();
    
    // Current blendshapes
    this.currentBlendshapes = {};
    this.targetBlendshapes = {};
    
    // Unity bridge
    this.unityBridge = null;
    
    console.log('[KellyLipSync] Orchestrator initialized');
  }
  
  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================
  
  /**
   * Initialize with optional Unity bridge
   * @param {Object} unityInstance - Unity WebGL instance
   */
  setUnityBridge(unityInstance) {
    this.unityBridge = unityInstance;
    console.log('[KellyLipSync] Unity bridge connected');
    return this;
  }
  
  /**
   * Set expression callback for emotion overlay
   * @param {Function} callback - Function that returns current expression blendshapes
   */
  setExpressionCallback(callback) {
    this.expressionCallback = callback;
    return this;
  }
  
  // ===========================================================================
  // ALIGNMENT-BASED LIP-SYNC (Pre-rendered)
  // ===========================================================================
  
  /**
   * Generate alignment from audio and text
   * @param {string|Blob} audio - Audio URL or Blob
   * @param {string} transcript - Text transcript
   * @returns {Promise<Object>} Alignment data
   */
  async generateAlignment(audio, transcript) {
    // Check cache
    const cacheKey = this.getCacheKey(transcript);
    if (this.config.cacheAlignments && this.alignmentCache.has(cacheKey)) {
      console.log('[KellyLipSync] Using cached alignment');
      return this.alignmentCache.get(cacheKey);
    }
    
    try {
      // Convert audio to form data if it's a Blob
      const formData = new FormData();
      
      if (audio instanceof Blob) {
        formData.append('audio', audio, 'audio.wav');
      } else if (typeof audio === 'string') {
        formData.append('audio_url', audio);
      }
      
      formData.append('transcript', transcript);
      
      // Call alignment API
      const response = await fetch(this.config.alignmentApiUrl, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Alignment API error: ${response.status}`);
      }
      
      const alignment = await response.json();
      
      // Cache result
      if (this.config.cacheAlignments) {
        this.cacheAlignment(cacheKey, alignment);
      }
      
      return alignment;
      
    } catch (error) {
      console.error('[KellyLipSync] Alignment generation failed:', error);
      
      // Fall back to estimation
      return this.estimateAlignment(transcript);
    }
  }
  
  /**
   * Simple alignment estimation from text
   * @param {string} transcript - Text transcript
   * @param {number} duration - Estimated duration in seconds
   * @returns {Object} Estimated alignment
   */
  estimateAlignment(transcript, duration = null) {
    const words = transcript.trim().split(/\s+/);
    const estimatedDuration = duration || (words.length * 0.4); // ~0.4s per word
    const wordDuration = estimatedDuration / words.length;
    
    const wordAlignments = [];
    const phoneAlignments = [];
    let currentTime = 0;
    
    for (const word of words) {
      const wordEnd = currentTime + wordDuration;
      
      wordAlignments.push({
        word,
        start: currentTime,
        end: wordEnd,
        confidence: 0.5,
      });
      
      // Estimate phonemes from letters
      const phonemes = this.estimatePhonemesFromWord(word);
      const phoneDuration = wordDuration / phonemes.length;
      
      for (let i = 0; i < phonemes.length; i++) {
        phoneAlignments.push({
          phone: phonemes[i],
          start: currentTime + (i * phoneDuration),
          end: currentTime + ((i + 1) * phoneDuration),
          word,
        });
      }
      
      currentTime = wordEnd + 0.05;
    }
    
    return {
      words: wordAlignments,
      phones: phoneAlignments,
      duration: estimatedDuration,
      method: 'estimation',
      confidence: 0.5,
    };
  }
  
  /**
   * Play lip-sync from alignment data
   * @param {Object} alignment - Alignment data with words and phones
   * @param {HTMLAudioElement} audioElement - Audio element to sync with
   */
  playFromAlignment(alignment, audioElement) {
    this.currentMode = 'alignment';
    this.currentAlignment = alignment;
    
    // Generate blendshape timeline
    this.currentTimeline = generateBlendshapeTimeline(
      alignment.phones,
      this.config.fps
    );
    
    this.timelineIndex = 0;
    this.playbackDuration = alignment.duration;
    
    // Start playback on audio play
    const handlePlay = () => {
      this.playbackStartTime = performance.now();
      this.isActive = true;
      this.updateFromTimeline();
    };
    
    const handleTimeUpdate = () => {
      if (!this.isActive) return;
      
      // Sync timeline index to audio time
      const audioTime = audioElement.currentTime;
      this.timelineIndex = Math.floor(audioTime * this.config.fps);
    };
    
    const handleEnded = () => {
      this.stop();
      if (this.onPlaybackComplete) {
        this.onPlaybackComplete();
      }
    };
    
    // Clean up previous listeners
    audioElement.removeEventListener('play', handlePlay);
    audioElement.removeEventListener('timeupdate', handleTimeUpdate);
    audioElement.removeEventListener('ended', handleEnded);
    
    // Add listeners
    audioElement.addEventListener('play', handlePlay);
    audioElement.addEventListener('timeupdate', handleTimeUpdate);
    audioElement.addEventListener('ended', handleEnded);
    
    // Store for cleanup
    this._audioListeners = { handlePlay, handleTimeUpdate, handleEnded, audioElement };
    
    console.log('[KellyLipSync] Ready to play from alignment');
    return this;
  }
  
  /**
   * Update blendshapes from pre-computed timeline
   * @private
   */
  updateFromTimeline() {
    if (!this.isActive || !this.currentTimeline) return;
    
    // Get current frame
    const frame = this.currentTimeline[this.timelineIndex];
    
    if (frame) {
      // Merge with expression overlay
      this.targetBlendshapes = this.mergeWithExpression(frame.blendshapes);
      
      // Apply smoothing
      this.currentBlendshapes = this.smoothBlendshapes(
        this.currentBlendshapes,
        this.targetBlendshapes
      );
      
      // Send to Unity
      this.sendToUnity(this.currentBlendshapes);
      
      // Callback
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(this.currentBlendshapes);
      }
      
      this.timelineIndex++;
    }
    
    // Continue if active
    if (this.isActive && this.timelineIndex < this.currentTimeline.length) {
      this.animationFrameId = requestAnimationFrame(() => this.updateFromTimeline());
    }
  }
  
  // ===========================================================================
  // REAL-TIME LIP-SYNC (Conversations)
  // ===========================================================================
  
  /**
   * Start real-time lip-sync from audio element
   * @param {HTMLAudioElement} audioElement - Audio element to analyze
   */
  startRealtimeFromAudio(audioElement) {
    this.currentMode = 'realtime';
    
    // Create real-time analyzer
    this.realtimeLipSync = new AudioElementLipSync(audioElement, {
      smoothing: 0.6,
      sensitivity: 1.5,
    });
    
    // Set up update callback
    this.realtimeLipSync.onBlendshapesUpdate = (blendshapes) => {
      const merged = this.mergeWithExpression(blendshapes);
      this.currentBlendshapes = merged;
      
      this.sendToUnity(merged);
      
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(merged);
      }
    };
    
    this.isActive = true;
    console.log('[KellyLipSync] Real-time lip-sync started');
    
    return this;
  }
  
  /**
   * Start real-time lip-sync from microphone
   * @param {MediaStream} mediaStream - Microphone stream
   */
  startRealtimeFromMicrophone(mediaStream) {
    this.currentMode = 'realtime';
    
    // Create real-time analyzer
    this.realtimeLipSync = new RealtimeLipSync({
      smoothing: 0.5,
      sensitivity: 2.0,
    });
    
    this.realtimeLipSync.connectMediaStream(mediaStream);
    
    // Set up update callback
    this.realtimeLipSync.onBlendshapesUpdate = (blendshapes) => {
      const merged = this.mergeWithExpression(blendshapes);
      this.currentBlendshapes = merged;
      
      this.sendToUnity(merged);
      
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(merged);
      }
    };
    
    this.realtimeLipSync.start();
    this.isActive = true;
    
    console.log('[KellyLipSync] Real-time lip-sync from microphone started');
    return this;
  }
  
  // ===========================================================================
  // STREAMING LIP-SYNC (ElevenLabs Streaming)
  // ===========================================================================
  
  /**
   * Start streaming lip-sync for ElevenLabs WebSocket audio
   */
  startStreamingLipSync() {
    this.currentMode = 'streaming';
    
    // Create streaming analyzer
    this.streamingLipSync = new StreamingLipSync({
      smoothing: 0.55,
      sensitivity: 1.6,
    });
    
    this.streamingLipSync.initStreaming();
    
    // Set up update callback
    this.streamingLipSync.onBlendshapesUpdate = (blendshapes) => {
      const merged = this.mergeWithExpression(blendshapes);
      this.currentBlendshapes = merged;
      
      this.sendToUnity(merged);
      
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(merged);
      }
    };
    
    this.streamingLipSync.start();
    this.isActive = true;
    
    console.log('[KellyLipSync] Streaming lip-sync started');
    return this;
  }
  
  /**
   * Add audio chunk for streaming lip-sync
   * @param {ArrayBuffer|Uint8Array} audioChunk - Encoded audio data
   */
  addStreamingAudioChunk(audioChunk) {
    if (this.streamingLipSync && this.currentMode === 'streaming') {
      this.streamingLipSync.addAudioChunk(audioChunk);
    }
  }
  
  // ===========================================================================
  // EXPRESSION INTEGRATION
  // ===========================================================================
  
  /**
   * Set current expression for overlay
   * @param {Object} expression - Expression blendshapes (eyes, brows)
   */
  setExpression(expression) {
    this.currentExpression = expression;
  }
  
  /**
   * Merge lip-sync blendshapes with expression overlay
   * @param {Object} lipSyncBlendshapes - Lip-sync mouth shapes
   * @returns {Object} Merged blendshapes
   * @private
   */
  mergeWithExpression(lipSyncBlendshapes) {
    if (!this.config.enableExpressions) {
      return lipSyncBlendshapes;
    }
    
    // Get current expression
    let expression = this.currentExpression;
    if (this.expressionCallback) {
      expression = this.expressionCallback();
    }
    
    if (!expression) {
      return lipSyncBlendshapes;
    }
    
    // Separate mouth shapes from expression shapes
    const mouthShapes = [
      'jawOpen', 'mouthOpen', 'mouthClose',
      'mouthFunnel', 'mouthPucker',
      'mouthStretchLeft', 'mouthStretchRight',
      'mouthSmileLeft', 'mouthSmileRight',
      'mouthFrownLeft', 'mouthFrownRight',
      'mouthPressLeft', 'mouthPressRight',
      'mouthUpperUpLeft', 'mouthUpperUpRight',
      'mouthLowerDownLeft', 'mouthLowerDownRight',
      'mouthRollLower', 'mouthRollUpper',
      'mouthShrugLower', 'mouthShrugUpper',
    ];
    
    const merged = {};
    
    // For mouth shapes: primarily use lip-sync
    for (const shape of mouthShapes) {
      const lipValue = lipSyncBlendshapes[shape] || 0;
      const expValue = expression[shape] || 0;
      
      // Weight toward lip-sync for mouth
      merged[shape] = (lipValue * this.config.lipSyncWeight) + 
                      (expValue * (1 - this.config.lipSyncWeight));
    }
    
    // For other shapes (eyes, brows): use expression
    for (const [key, value] of Object.entries(expression)) {
      if (!mouthShapes.includes(key)) {
        merged[key] = value;
      }
    }
    
    return merged;
  }
  
  // ===========================================================================
  // UNITY COMMUNICATION
  // ===========================================================================
  
  /**
   * Send blendshapes to Unity
   * @param {Object} blendshapes - Blendshape values
   * @private
   */
  sendToUnity(blendshapes) {
    if (!this.unityBridge) return;
    
    try {
      // Convert to JSON string for Unity
      const json = JSON.stringify(blendshapes);
      
      // Send to Kelly character
      this.unityBridge.SendMessage('kelly_fbx_v4', 'SetBlendshapes', json);
    } catch (error) {
      // Unity not loaded, ignore
    }
  }
  
  // ===========================================================================
  // UTILITIES
  // ===========================================================================
  
  /**
   * Smooth transition between blendshape states
   * @param {Object} from - Current blendshapes
   * @param {Object} to - Target blendshapes
   * @returns {Object} Smoothed blendshapes
   * @private
   */
  smoothBlendshapes(from, to) {
    const result = {};
    const smoothing = this.config.smoothTransitions ? 0.3 : 0;
    
    const allKeys = new Set([...Object.keys(from || {}), ...Object.keys(to || {})]);
    
    for (const key of allKeys) {
      const fromValue = (from && from[key]) || 0;
      const toValue = (to && to[key]) || 0;
      result[key] = fromValue + (toValue - fromValue) * (1 - smoothing);
    }
    
    return result;
  }
  
  /**
   * Estimate phonemes from word spelling
   * @param {string} word - Word to analyze
   * @returns {Array} Estimated phonemes
   * @private
   */
  estimatePhonemesFromWord(word) {
    const LETTER_TO_PHONEME = {
      'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH',
      'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
      'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
      'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
      't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
    };
    
    const phonemes = [];
    word = word.toLowerCase().replace(/[^a-z]/g, '');
    
    for (let i = 0; i < word.length; i++) {
      const char = word[i];
      if (LETTER_TO_PHONEME[char]) {
        phonemes.push(LETTER_TO_PHONEME[char]);
      }
    }
    
    return phonemes.length > 0 ? phonemes : ['SIL'];
  }
  
  /**
   * Get cache key for alignment
   * @param {string} transcript - Transcript text
   * @returns {string} Cache key
   * @private
   */
  getCacheKey(transcript) {
    // Simple hash
    let hash = 0;
    for (let i = 0; i < transcript.length; i++) {
      const char = transcript.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return `align_${Math.abs(hash)}`;
  }
  
  /**
   * Cache alignment data
   * @param {string} key - Cache key
   * @param {Object} alignment - Alignment data
   * @private
   */
  cacheAlignment(key, alignment) {
    // Enforce max cache size
    if (this.alignmentCache.size >= this.config.maxCacheSize) {
      const firstKey = this.alignmentCache.keys().next().value;
      this.alignmentCache.delete(firstKey);
    }
    
    this.alignmentCache.set(key, alignment);
  }
  
  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================
  
  /**
   * Stop lip-sync
   */
  stop() {
    this.isActive = false;
    
    // Stop animation frame
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Stop real-time analyzer
    if (this.realtimeLipSync) {
      this.realtimeLipSync.stop();
    }
    
    // Stop streaming analyzer
    if (this.streamingLipSync) {
      this.streamingLipSync.stop();
    }
    
    // Reset state
    this.currentTimeline = null;
    this.timelineIndex = 0;
    
    // Reset to neutral face
    this.currentBlendshapes = {};
    this.sendToUnity({});
    
    console.log('[KellyLipSync] Stopped');
    return this;
  }
  
  /**
   * Clean up resources
   */
  dispose() {
    this.stop();
    
    // Clean up audio listeners
    if (this._audioListeners) {
      const { handlePlay, handleTimeUpdate, handleEnded, audioElement } = this._audioListeners;
      audioElement.removeEventListener('play', handlePlay);
      audioElement.removeEventListener('timeupdate', handleTimeUpdate);
      audioElement.removeEventListener('ended', handleEnded);
      this._audioListeners = null;
    }
    
    // Dispose analyzers
    if (this.realtimeLipSync) {
      this.realtimeLipSync.dispose();
      this.realtimeLipSync = null;
    }
    
    if (this.streamingLipSync) {
      this.streamingLipSync.dispose();
      this.streamingLipSync = null;
    }
    
    // Clear cache
    this.alignmentCache.clear();
    
    console.log('[KellyLipSync] Disposed');
  }
  
  // ===========================================================================
  // STATIC FACTORY METHODS
  // ===========================================================================
  
  /**
   * Create orchestrator for pre-rendered lessons
   * @param {Object} options - Configuration options
   * @returns {KellyLipSyncOrchestrator} Configured orchestrator
   */
  static forLessons(options = {}) {
    return new KellyLipSyncOrchestrator({
      ...options,
      preferredMethod: 'alignment',
      cacheAlignments: true,
    });
  }
  
  /**
   * Create orchestrator for real-time conversations
   * @param {Object} options - Configuration options
   * @returns {KellyLipSyncOrchestrator} Configured orchestrator
   */
  static forConversation(options = {}) {
    return new KellyLipSyncOrchestrator({
      ...options,
      preferredMethod: 'realtime',
      smoothing: 0.5,
    });
  }
  
  /**
   * Create orchestrator for ElevenLabs streaming
   * @param {Object} options - Configuration options
   * @returns {KellyLipSyncOrchestrator} Configured orchestrator
   */
  static forStreaming(options = {}) {
    return new KellyLipSyncOrchestrator({
      ...options,
      preferredMethod: 'streaming',
    });
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default KellyLipSyncOrchestrator;

