/**
 * Kelly Lip-Sync System v1.0
 * 
 * Browser-ready lip-sync for Kelly's avatar.
 * Combines real-time audio analysis with optional forced alignment.
 * 
 * Features:
 * - Real-time amplitude-based lip-sync for streaming audio
 * - Frequency analysis for improved mouth shapes
 * - Smooth transitions and natural decay
 * - Unity bridge integration for 3D avatar
 * - 2D avatar fallback support
 * 
 * Usage:
 *   KellyLipSync.init();
 *   KellyLipSync.startStreaming();  // For ElevenLabs streaming
 *   KellyLipSync.addAudioChunk(base64Audio);  // Feed audio chunks
 *   KellyLipSync.stop();
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const LIPSYNC_CONFIG = {
  // Smoothing factor (0-1, higher = smoother but less responsive)
  smoothing: 0.55,
  
  // Minimum amplitude to trigger mouth movement (0-1)
  minAmplitude: 0.02,
  
  // Maximum jaw opening (0-100)
  maxJawOpen: 85,
  
  // Mouth movement sensitivity multiplier
  sensitivity: 1.6,
  
  // Frame rate for updates (fps)
  updateRate: 30,
  
  // Frequency band thresholds (Hz)
  frequencyBands: {
    low: { min: 80, max: 300 },      // Vowels, fundamental frequency
    mid: { min: 300, max: 2000 },    // Consonants, formants
    high: { min: 2000, max: 8000 },  // Sibilants, fricatives
  },
  
  // Decay rate when audio stops (per frame)
  decayRate: 0.18,
  
  // Send to Unity bridge
  sendToUnity: true,
  
  // Send to 2D avatar
  sendTo2D: true,
  
  // Debug logging
  debug: false,
};

// =============================================================================
// PHONEME TO VISEME MAPPING (Simplified for real-time)
// =============================================================================

const AMPLITUDE_TO_MOUTH = {
  // Maps normalized amplitude ranges to mouth shapes
  0.0: { jawOpen: 0, mouthOpen: 0, mouthSmile: 15 },      // Silence
  0.2: { jawOpen: 20, mouthOpen: 15, mouthSmile: 20 },    // Soft sound
  0.4: { jawOpen: 40, mouthOpen: 30, mouthSmile: 15 },    // Medium
  0.6: { jawOpen: 60, mouthOpen: 50, mouthSmile: 10 },    // Loud
  0.8: { jawOpen: 75, mouthOpen: 65, mouthSmile: 5 },     // Very loud
  1.0: { jawOpen: 85, mouthOpen: 80, mouthSmile: 0 },     // Maximum
};

// =============================================================================
// KELLY LIPSYNC SYSTEM
// =============================================================================

const KellyLipSync = {
  // State
  isActive: false,
  isStreaming: false,
  isInitialized: false,
  
  // Audio context and nodes
  audioContext: null,
  analyser: null,
  sourceNode: null,
  gainNode: null,
  
  // Analysis data
  timeDomainData: null,
  frequencyData: null,
  fftSize: 2048,
  
  // Blendshapes
  currentBlendshapes: {},
  targetBlendshapes: {},
  restingFace: {
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
  },
  
  // Animation frame
  animationFrameId: null,
  lastUpdateTime: 0,
  frameInterval: 1000 / 30,
  
  // Energy tracking
  energyHistory: [],
  energyHistorySize: 5,
  
  // Speaking state
  isSpeaking: false,
  silenceFrames: 0,
  silenceThreshold: 8,
  
  // Streaming audio queue
  audioQueue: [],
  isProcessingQueue: false,
  currentAudioSource: null,
  
  // Callbacks
  onBlendshapesUpdate: null,
  onSpeakingStateChange: null,
  
  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================
  
  /**
   * Initialize the lip-sync system
   * @param {Object} options - Configuration overrides
   */
  init(options = {}) {
    if (this.isInitialized) {
      console.log('[KellyLipSync] Already initialized');
      return this;
    }
    
    // Merge config
    Object.assign(LIPSYNC_CONFIG, options);
    
    // Create audio context
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Create analyser node
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize;
    this.analyser.smoothingTimeConstant = 0.3;
    
    // Create gain node for output
    this.gainNode = this.audioContext.createGain();
    this.gainNode.gain.value = 1.0;
    this.analyser.connect(this.gainNode);
    this.gainNode.connect(this.audioContext.destination);
    
    // Create data arrays
    this.timeDomainData = new Float32Array(this.analyser.fftSize);
    this.frequencyData = new Float32Array(this.analyser.frequencyBinCount);
    
    // Reset blendshapes
    this.currentBlendshapes = { ...this.restingFace };
    this.targetBlendshapes = { ...this.restingFace };
    
    this.isInitialized = true;
    this.frameInterval = 1000 / LIPSYNC_CONFIG.updateRate;
    
    console.log('[KellyLipSync] Initialized');
    return this;
  },
  
  /**
   * Ensure audio context is resumed (call after user interaction)
   */
  async resume() {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
      console.log('[KellyLipSync] AudioContext resumed');
    }
    return this;
  },
  
  // ===========================================================================
  // STREAMING MODE (ElevenLabs WebSocket)
  // ===========================================================================
  
  /**
   * Start streaming lip-sync mode
   */
  startStreaming() {
    if (!this.isInitialized) {
      this.init();
    }
    
    this.resume();
    this.isStreaming = true;
    this.isActive = true;
    this.audioQueue = [];
    this.isProcessingQueue = false;
    
    // Start animation loop
    this.lastUpdateTime = performance.now();
    this.processFrame();
    
    console.log('[KellyLipSync] Streaming mode started');
    return this;
  },
  
  /**
   * Add audio chunk to processing queue (for ElevenLabs streaming)
   * @param {string|ArrayBuffer|Uint8Array} audioData - Audio data (base64 or buffer)
   */
  async addAudioChunk(audioData) {
    if (!this.isStreaming) return;
    
    let buffer;
    
    // Handle base64 string
    if (typeof audioData === 'string') {
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      buffer = bytes.buffer;
    } else if (audioData instanceof ArrayBuffer) {
      buffer = audioData;
    } else if (audioData instanceof Uint8Array) {
      buffer = audioData.buffer.slice(audioData.byteOffset, audioData.byteOffset + audioData.byteLength);
    }
    
    if (buffer) {
      this.audioQueue.push(buffer);
      
      if (!this.isProcessingQueue) {
        this.processAudioQueue();
      }
    }
  },
  
  /**
   * Process queued audio chunks
   * @private
   */
  async processAudioQueue() {
    if (this.audioQueue.length === 0) {
      this.isProcessingQueue = false;
      return;
    }
    
    this.isProcessingQueue = true;
    const chunk = this.audioQueue.shift();
    
    try {
      // Decode audio data
      const audioBuffer = await this.audioContext.decodeAudioData(chunk.slice(0));
      
      // Create buffer source
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.analyser);
      
      this.currentAudioSource = source;
      
      // Play audio
      source.onended = () => {
        this.currentAudioSource = null;
        this.processAudioQueue();
      };
      source.start();
      
    } catch (error) {
      if (LIPSYNC_CONFIG.debug) {
        console.warn('[KellyLipSync] Failed to decode audio chunk:', error);
      }
      this.processAudioQueue(); // Try next chunk
    }
  },
  
  // ===========================================================================
  // AUDIO ELEMENT MODE (Pre-rendered audio)
  // ===========================================================================
  
  /**
   * Start lip-sync for an HTML audio element
   * @param {HTMLAudioElement} audioElement - Audio element to analyze
   */
  startFromAudioElement(audioElement) {
    if (!this.isInitialized) {
      this.init();
    }
    
    this.resume();
    
    // Create media element source (only once per element)
    if (!audioElement._lipSyncConnected) {
      const source = this.audioContext.createMediaElementSource(audioElement);
      source.connect(this.analyser);
      audioElement._lipSyncConnected = true;
      audioElement._lipSyncSource = source;
    }
    
    // Event handlers
    const handlePlay = () => {
      this.isActive = true;
      if (!this.animationFrameId) {
        this.lastUpdateTime = performance.now();
        this.processFrame();
      }
    };
    
    const handlePause = () => {
      // Keep analyzing briefly for smooth decay
    };
    
    const handleEnded = () => {
      // Allow natural decay to resting face
      setTimeout(() => {
        if (!audioElement.paused) return;
        this.isActive = false;
      }, 500);
    };
    
    audioElement.addEventListener('play', handlePlay);
    audioElement.addEventListener('pause', handlePause);
    audioElement.addEventListener('ended', handleEnded);
    
    // Store for cleanup
    audioElement._lipSyncHandlers = { handlePlay, handlePause, handleEnded };
    
    console.log('[KellyLipSync] Connected to audio element');
    return this;
  },
  
  /**
   * Stop lip-sync for an audio element
   * @param {HTMLAudioElement} audioElement - Audio element to disconnect
   */
  stopFromAudioElement(audioElement) {
    if (audioElement._lipSyncHandlers) {
      const { handlePlay, handlePause, handleEnded } = audioElement._lipSyncHandlers;
      audioElement.removeEventListener('play', handlePlay);
      audioElement.removeEventListener('pause', handlePause);
      audioElement.removeEventListener('ended', handleEnded);
      delete audioElement._lipSyncHandlers;
    }
  },
  
  // ===========================================================================
  // MAIN PROCESSING LOOP
  // ===========================================================================
  
  /**
   * Main animation frame processing
   * @private
   */
  processFrame() {
    if (!this.isActive && !this.isStreaming) return;
    
    const now = performance.now();
    const elapsed = now - this.lastUpdateTime;
    
    // Throttle to target frame rate
    if (elapsed >= this.frameInterval) {
      this.lastUpdateTime = now - (elapsed % this.frameInterval);
      
      // Get audio data
      this.analyser.getFloatTimeDomainData(this.timeDomainData);
      this.analyser.getFloatFrequencyData(this.frequencyData);
      
      // Analyze and generate blendshapes
      const newBlendshapes = this.analyzeAudio();
      
      // Smooth transition
      this.currentBlendshapes = this.smoothBlendshapes(
        this.currentBlendshapes,
        newBlendshapes
      );
      
      // Send to Unity
      if (LIPSYNC_CONFIG.sendToUnity) {
        this.sendToUnity();
      }
      
      // Send to 2D avatar
      if (LIPSYNC_CONFIG.sendTo2D) {
        this.sendTo2D();
      }
      
      // Callback
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(this.currentBlendshapes);
      }
    }
    
    this.animationFrameId = requestAnimationFrame(() => this.processFrame());
  },
  
  /**
   * Analyze current audio frame
   * @returns {Object} Blendshape values
   * @private
   */
  analyzeAudio() {
    // Calculate RMS amplitude
    let sum = 0;
    for (let i = 0; i < this.timeDomainData.length; i++) {
      sum += this.timeDomainData[i] * this.timeDomainData[i];
    }
    const rms = Math.sqrt(sum / this.timeDomainData.length);
    
    // Track energy history
    this.energyHistory.push(rms);
    if (this.energyHistory.length > this.energyHistorySize) {
      this.energyHistory.shift();
    }
    const avgEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
    
    // Update speaking state
    this.updateSpeakingState(avgEnergy);
    
    // If below threshold, decay to rest
    if (avgEnergy < LIPSYNC_CONFIG.minAmplitude) {
      return this.decayToRest();
    }
    
    // Normalize energy
    const normalizedEnergy = Math.min(1, avgEnergy * LIPSYNC_CONFIG.sensitivity);
    
    // Get frequency band energies
    const lowEnergy = this.getFrequencyBandEnergy(
      LIPSYNC_CONFIG.frequencyBands.low.min,
      LIPSYNC_CONFIG.frequencyBands.low.max
    );
    const midEnergy = this.getFrequencyBandEnergy(
      LIPSYNC_CONFIG.frequencyBands.mid.min,
      LIPSYNC_CONFIG.frequencyBands.mid.max
    );
    const highEnergy = this.getFrequencyBandEnergy(
      LIPSYNC_CONFIG.frequencyBands.high.min,
      LIPSYNC_CONFIG.frequencyBands.high.max
    );
    
    // Generate blendshapes
    return this.generateBlendshapes(normalizedEnergy, lowEnergy, midEnergy, highEnergy);
  },
  
  /**
   * Generate blendshape values from audio analysis
   * @private
   */
  generateBlendshapes(energy, lowEnergy, midEnergy, highEnergy) {
    // Normalize frequency energies
    const totalFreqEnergy = lowEnergy + midEnergy + highEnergy || 1;
    const lowRatio = lowEnergy / totalFreqEnergy;
    const midRatio = midEnergy / totalFreqEnergy;
    const highRatio = highEnergy / totalFreqEnergy;
    
    // Base mouth opening from overall energy
    const baseJawOpen = energy * LIPSYNC_CONFIG.maxJawOpen;
    
    // Determine mouth shape from frequency ratios
    let mouthFunnel = 0;
    let mouthStretch = 0;
    let lipsPursed = 0;
    
    if (lowRatio > 0.5) {
      mouthFunnel = 20 + (lowRatio * 30);
    }
    
    if (midRatio > 0.4) {
      mouthStretch = midRatio * 40;
    }
    
    if (highRatio > 0.3) {
      mouthStretch = Math.max(mouthStretch, highRatio * 50);
      lipsPursed = highRatio * 30;
    }
    
    // Add natural variation
    const variation = (Math.random() - 0.5) * 4;
    
    return {
      jawOpen: Math.max(0, Math.min(100, baseJawOpen + variation)),
      mouthOpen: baseJawOpen * 0.8,
      mouthFunnel: mouthFunnel,
      mouthPucker: lipsPursed * 0.5,
      mouthStretchLeft: mouthStretch,
      mouthStretchRight: mouthStretch,
      mouthSmileLeft: 15 + (energy * 10),
      mouthSmileRight: 15 + (energy * 10),
      mouthPressLeft: midRatio * 15,
      mouthPressRight: midRatio * 15,
      mouthUpperUpLeft: energy * 8,
      mouthUpperUpRight: energy * 8,
      mouthLowerDownLeft: baseJawOpen * 0.3,
      mouthLowerDownRight: baseJawOpen * 0.3,
      mouthClose: Math.max(0, 15 - baseJawOpen),
    };
  },
  
  /**
   * Get energy in a frequency band
   * @private
   */
  getFrequencyBandEnergy(minHz, maxHz) {
    const nyquist = this.audioContext.sampleRate / 2;
    const binSize = nyquist / this.frequencyData.length;
    
    const minBin = Math.floor(minHz / binSize);
    const maxBin = Math.min(Math.ceil(maxHz / binSize), this.frequencyData.length - 1);
    
    let sum = 0;
    let count = 0;
    
    for (let i = minBin; i <= maxBin; i++) {
      const linearValue = Math.pow(10, this.frequencyData[i] / 20);
      sum += linearValue;
      count++;
    }
    
    return count > 0 ? sum / count : 0;
  },
  
  /**
   * Decay current blendshapes toward resting position
   * @private
   */
  decayToRest() {
    const decayed = {};
    
    for (const [key, restValue] of Object.entries(this.restingFace)) {
      const currentValue = this.currentBlendshapes[key] || 0;
      const diff = restValue - currentValue;
      decayed[key] = currentValue + (diff * LIPSYNC_CONFIG.decayRate);
    }
    
    return decayed;
  },
  
  /**
   * Smooth transition between blendshape states
   * @private
   */
  smoothBlendshapes(from, to) {
    const result = {};
    const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
    
    for (const key of allKeys) {
      const fromValue = from[key] || 0;
      const toValue = to[key] || 0;
      result[key] = fromValue + (toValue - fromValue) * (1 - LIPSYNC_CONFIG.smoothing);
    }
    
    return result;
  },
  
  /**
   * Update speaking state
   * @private
   */
  updateSpeakingState(energy) {
    const wasSpeaking = this.isSpeaking;
    
    if (energy >= LIPSYNC_CONFIG.minAmplitude) {
      this.silenceFrames = 0;
      this.isSpeaking = true;
    } else {
      this.silenceFrames++;
      if (this.silenceFrames >= this.silenceThreshold) {
        this.isSpeaking = false;
      }
    }
    
    if (wasSpeaking !== this.isSpeaking && this.onSpeakingStateChange) {
      this.onSpeakingStateChange(this.isSpeaking);
    }
  },
  
  // ===========================================================================
  // OUTPUT TO AVATARS
  // ===========================================================================
  
  /**
   * Send blendshapes to Unity 3D avatar
   * @private
   */
  sendToUnity() {
    if (!window.unityInstance) return;
    
    try {
      const json = JSON.stringify(this.currentBlendshapes);
      window.unityInstance.SendMessage('kelly_fbx_v4', 'SetBlendshapes', json);
    } catch (e) {
      // Unity not loaded, ignore
    }
  },
  
  /**
   * Send blendshapes to 2D avatar
   * @private
   */
  sendTo2D() {
    // Update KellyPoseManager if available
    if (window.KellyPoseManager && window.KellyPoseManager.setMouthState) {
      const jawOpen = this.currentBlendshapes.jawOpen || 0;
      
      if (jawOpen > 30) {
        window.KellyPoseManager.setMouthState('speaking');
      } else if (jawOpen > 10) {
        window.KellyPoseManager.setMouthState('talking');
      } else {
        window.KellyPoseManager.setMouthState('idle');
      }
    }
    
    // Update Kelly 2D Avatar component if available
    if (window.kelly2DAvatar && window.kelly2DAvatar.setMouthOpenness) {
      window.kelly2DAvatar.setMouthOpenness(this.currentBlendshapes.jawOpen / 100);
    }
  },
  
  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================
  
  /**
   * Get current blendshapes
   * @returns {Object} Current blendshape values
   */
  getBlendshapes() {
    return { ...this.currentBlendshapes };
  },
  
  /**
   * Get speaking state
   * @returns {boolean} Whether audio is speaking
   */
  getIsSpeaking() {
    return this.isSpeaking;
  },
  
  /**
   * Update configuration
   * @param {Object} newConfig - New configuration values
   */
  updateConfig(newConfig) {
    Object.assign(LIPSYNC_CONFIG, newConfig);
    this.frameInterval = 1000 / LIPSYNC_CONFIG.updateRate;
  },
  
  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================
  
  /**
   * Stop lip-sync
   */
  stop() {
    this.isActive = false;
    this.isStreaming = false;
    
    // Stop animation frame
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Clear audio queue
    this.audioQueue = [];
    this.isProcessingQueue = false;
    
    if (this.currentAudioSource) {
      try {
        this.currentAudioSource.stop();
      } catch (e) {}
      this.currentAudioSource = null;
    }
    
    // Reset to resting face
    this.currentBlendshapes = { ...this.restingFace };
    this.sendToUnity();
    this.sendTo2D();
    
    console.log('[KellyLipSync] Stopped');
    return this;
  },
  
  /**
   * Clean up resources
   */
  dispose() {
    this.stop();
    
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    
    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }
    
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    this.isInitialized = false;
    console.log('[KellyLipSync] Disposed');
  },
};

// =============================================================================
// EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyLipSync = KellyLipSync;
}

// Auto-init on DOM ready
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyLipSync.init());
  } else {
    KellyLipSync.init();
  }
}

