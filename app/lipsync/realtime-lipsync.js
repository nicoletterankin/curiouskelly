/**
 * Real-Time Audio Analysis Lip-Sync for Kelly
 * 
 * Generates lip-sync blendshapes from audio amplitude/frequency analysis.
 * Used for live conversations when phoneme alignment isn't available.
 * 
 * Features:
 * - Real-time audio amplitude analysis
 * - Frequency-based viseme estimation
 * - Smooth transitions with configurable parameters
 * - WebAudio API integration
 * 
 * @module realtime-lipsync
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Default configuration for real-time lip-sync
 */
export const DEFAULT_CONFIG = {
  // Smoothing factor (0-1, higher = smoother but less responsive)
  smoothing: 0.65,
  
  // Minimum amplitude to trigger mouth movement (0-1)
  minAmplitude: 0.02,
  
  // Maximum jaw opening (0-100)
  maxJawOpen: 85,
  
  // Mouth movement sensitivity multiplier
  sensitivity: 1.5,
  
  // Frame rate for updates (fps)
  updateRate: 30,
  
  // Frequency band thresholds (Hz)
  frequencyBands: {
    low: { min: 80, max: 300 },      // Vowels, fundamental frequency
    mid: { min: 300, max: 2000 },    // Consonants, formants
    high: { min: 2000, max: 8000 },  // Sibilants, fricatives
  },
  
  // Decay rate when audio stops (per frame)
  decayRate: 0.15,
  
  // Enable frequency-based viseme hints
  useFrequencyAnalysis: true,
  
  // Enable formant tracking for vowel detection
  useFormantTracking: false,
};

// =============================================================================
// REALTIME LIPSYNC CLASS
// =============================================================================

/**
 * Real-time lip-sync generator from audio analysis
 */
export class RealtimeLipSync {
  /**
   * Create a real-time lip-sync analyzer
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    this.config = { ...DEFAULT_CONFIG, ...options };
    
    // Audio context and nodes
    this.audioContext = null;
    this.analyser = null;
    this.sourceNode = null;
    
    // Analysis data
    this.timeDomainData = null;
    this.frequencyData = null;
    this.fftSize = 2048;
    
    // State
    this.isActive = false;
    this.currentBlendshapes = this.getRestingFace();
    this.previousBlendshapes = this.getRestingFace();
    
    // Animation frame tracking
    this.animationFrameId = null;
    this.lastUpdateTime = 0;
    this.frameInterval = 1000 / this.config.updateRate;
    
    // Callbacks
    this.onBlendshapesUpdate = null;
    this.onSpeakingStateChange = null;
    
    // Speaking state tracking
    this.isSpeaking = false;
    this.silenceFrames = 0;
    this.silenceThreshold = 10; // Frames of silence before "not speaking"
    
    // Energy history for smoothing
    this.energyHistory = [];
    this.energyHistorySize = 5;
  }
  
  /**
   * Initialize with an AudioContext
   * @param {AudioContext} audioContext - Existing audio context
   */
  init(audioContext) {
    this.audioContext = audioContext;
    
    // Create analyser node
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = this.fftSize;
    this.analyser.smoothingTimeConstant = 0.3;
    
    // Create data arrays
    this.timeDomainData = new Float32Array(this.analyser.fftSize);
    this.frequencyData = new Float32Array(this.analyser.frequencyBinCount);
    
    console.log('[RealtimeLipSync] Initialized with AudioContext');
    return this;
  }
  
  /**
   * Connect an audio source to the analyzer
   * @param {AudioNode} sourceNode - Audio source node (MediaStreamSource, etc.)
   */
  connectSource(sourceNode) {
    this.sourceNode = sourceNode;
    this.sourceNode.connect(this.analyser);
    console.log('[RealtimeLipSync] Audio source connected');
    return this;
  }
  
  /**
   * Create analyzer from MediaStream (microphone, etc.)
   * @param {MediaStream} mediaStream - Media stream from getUserMedia
   */
  connectMediaStream(mediaStream) {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (!this.analyser) {
      this.init(this.audioContext);
    }
    
    const source = this.audioContext.createMediaStreamSource(mediaStream);
    this.connectSource(source);
    
    return this;
  }
  
  /**
   * Connect to an AudioElement for playback analysis
   * @param {HTMLAudioElement} audioElement - Audio element to analyze
   */
  connectAudioElement(audioElement) {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (!this.analyser) {
      this.init(this.audioContext);
    }
    
    const source = this.audioContext.createMediaElementSource(audioElement);
    source.connect(this.analyser);
    this.analyser.connect(this.audioContext.destination);
    this.sourceNode = source;
    
    return this;
  }
  
  /**
   * Start real-time analysis
   * @param {Function} onUpdate - Callback for blendshape updates
   */
  start(onUpdate = null) {
    if (this.isActive) return;
    
    if (!this.analyser) {
      console.error('[RealtimeLipSync] Not initialized. Call init() first.');
      return;
    }
    
    this.isActive = true;
    this.onBlendshapesUpdate = onUpdate;
    this.lastUpdateTime = performance.now();
    
    this.processFrame();
    console.log('[RealtimeLipSync] Started real-time analysis');
    
    return this;
  }
  
  /**
   * Stop real-time analysis
   */
  stop() {
    this.isActive = false;
    
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    
    // Reset to resting face
    this.currentBlendshapes = this.getRestingFace();
    
    console.log('[RealtimeLipSync] Stopped');
    return this;
  }
  
  /**
   * Main processing loop
   * @private
   */
  processFrame() {
    if (!this.isActive) return;
    
    const now = performance.now();
    const elapsed = now - this.lastUpdateTime;
    
    // Throttle updates to target frame rate
    if (elapsed >= this.frameInterval) {
      this.lastUpdateTime = now - (elapsed % this.frameInterval);
      
      // Get audio data
      this.analyser.getFloatTimeDomainData(this.timeDomainData);
      this.analyser.getFloatFrequencyData(this.frequencyData);
      
      // Analyze and generate blendshapes
      const blendshapes = this.analyzeAudio();
      
      // Apply smoothing
      this.currentBlendshapes = this.smoothBlendshapes(
        this.previousBlendshapes,
        blendshapes
      );
      this.previousBlendshapes = { ...this.currentBlendshapes };
      
      // Callback
      if (this.onBlendshapesUpdate) {
        this.onBlendshapesUpdate(this.currentBlendshapes);
      }
    }
    
    this.animationFrameId = requestAnimationFrame(() => this.processFrame());
  }
  
  /**
   * Analyze current audio frame and generate blendshapes
   * @returns {Object} Blendshape values
   * @private
   */
  analyzeAudio() {
    // Calculate RMS amplitude
    const rms = this.calculateRMS(this.timeDomainData);
    
    // Track energy history for smoothing
    this.energyHistory.push(rms);
    if (this.energyHistory.length > this.energyHistorySize) {
      this.energyHistory.shift();
    }
    const avgEnergy = this.energyHistory.reduce((a, b) => a + b, 0) / this.energyHistory.length;
    
    // Update speaking state
    this.updateSpeakingState(avgEnergy);
    
    // If below threshold, decay to rest
    if (avgEnergy < this.config.minAmplitude) {
      return this.decayToRest();
    }
    
    // Normalize energy (0-1)
    const normalizedEnergy = Math.min(1, avgEnergy * this.config.sensitivity);
    
    // Get frequency band energies
    let lowEnergy = 0, midEnergy = 0, highEnergy = 0;
    
    if (this.config.useFrequencyAnalysis) {
      lowEnergy = this.getFrequencyBandEnergy(
        this.config.frequencyBands.low.min,
        this.config.frequencyBands.low.max
      );
      midEnergy = this.getFrequencyBandEnergy(
        this.config.frequencyBands.mid.min,
        this.config.frequencyBands.mid.max
      );
      highEnergy = this.getFrequencyBandEnergy(
        this.config.frequencyBands.high.min,
        this.config.frequencyBands.high.max
      );
    }
    
    // Generate blendshapes from analysis
    return this.generateBlendshapes(normalizedEnergy, lowEnergy, midEnergy, highEnergy);
  }
  
  /**
   * Generate blendshape values from audio analysis
   * @param {number} energy - Overall energy (0-1)
   * @param {number} lowEnergy - Low frequency energy
   * @param {number} midEnergy - Mid frequency energy
   * @param {number} highEnergy - High frequency energy
   * @returns {Object} Blendshape values
   * @private
   */
  generateBlendshapes(energy, lowEnergy, midEnergy, highEnergy) {
    // Normalize frequency energies
    const totalFreqEnergy = lowEnergy + midEnergy + highEnergy || 1;
    const lowRatio = lowEnergy / totalFreqEnergy;
    const midRatio = midEnergy / totalFreqEnergy;
    const highRatio = highEnergy / totalFreqEnergy;
    
    // Base mouth opening from overall energy
    const baseJawOpen = energy * this.config.maxJawOpen;
    
    // Determine mouth shape from frequency ratios
    let mouthFunnel = 0;   // Rounded (O, U)
    let mouthStretch = 0;  // Wide (E, I)
    let lipsPursed = 0;    // Pursed
    
    // High low-frequency = open vowels (AA, AH, AO)
    // High mid-frequency = consonants
    // High high-frequency = sibilants (S, SH, F)
    
    if (lowRatio > 0.5) {
      // Open vowel shape
      mouthFunnel = 20 + (lowRatio * 30);
    }
    
    if (midRatio > 0.4) {
      // Consonant - more stretch
      mouthStretch = midRatio * 40;
    }
    
    if (highRatio > 0.3) {
      // Sibilant - teeth together, stretched
      mouthStretch = Math.max(mouthStretch, highRatio * 50);
      lipsPursed = highRatio * 30;
    }
    
    // Apply some randomness for natural variation
    const variation = (Math.random() - 0.5) * 5;
    
    return {
      // Jaw movement (primary driver)
      jawOpen: Math.max(0, Math.min(100, baseJawOpen + variation)),
      
      // Mouth shape
      mouthOpen: baseJawOpen * 0.8,
      mouthFunnel: mouthFunnel,
      mouthPucker: lipsPursed * 0.5,
      
      // Lip stretch for consonants/vowels
      mouthStretchLeft: mouthStretch,
      mouthStretchRight: mouthStretch,
      
      // Subtle smile to prevent dead face
      mouthSmileLeft: 15 + (energy * 10),
      mouthSmileRight: 15 + (energy * 10),
      
      // Slight movement for liveliness
      mouthPressLeft: midRatio * 15,
      mouthPressRight: midRatio * 15,
      
      // Upper lip subtle movement
      mouthUpperUpLeft: energy * 8,
      mouthUpperUpRight: energy * 8,
      
      // Lower lip movement
      mouthLowerDownLeft: baseJawOpen * 0.3,
      mouthLowerDownRight: baseJawOpen * 0.3,
    };
  }
  
  /**
   * Get resting face blendshapes
   * @returns {Object} Rest position blendshapes
   */
  getRestingFace() {
    return {
      jawOpen: 0,
      mouthOpen: 0,
      mouthFunnel: 0,
      mouthPucker: 0,
      mouthStretchLeft: 0,
      mouthStretchRight: 0,
      mouthSmileLeft: 15,  // Slight smile at rest
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
   * Decay current blendshapes toward resting position
   * @returns {Object} Decayed blendshapes
   * @private
   */
  decayToRest() {
    const rest = this.getRestingFace();
    const decayed = {};
    
    for (const [key, restValue] of Object.entries(rest)) {
      const currentValue = this.currentBlendshapes[key] || 0;
      const diff = restValue - currentValue;
      decayed[key] = currentValue + (diff * this.config.decayRate);
    }
    
    return decayed;
  }
  
  /**
   * Smooth transition between blendshape states
   * @param {Object} from - Previous blendshapes
   * @param {Object} to - Target blendshapes
   * @returns {Object} Smoothed blendshapes
   * @private
   */
  smoothBlendshapes(from, to) {
    const result = {};
    const allKeys = new Set([...Object.keys(from), ...Object.keys(to)]);
    
    for (const key of allKeys) {
      const fromValue = from[key] || 0;
      const toValue = to[key] || 0;
      result[key] = fromValue + (toValue - fromValue) * (1 - this.config.smoothing);
    }
    
    return result;
  }
  
  /**
   * Calculate RMS (Root Mean Square) amplitude
   * @param {Float32Array} data - Time domain audio data
   * @returns {number} RMS value (0-1)
   * @private
   */
  calculateRMS(data) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i];
    }
    return Math.sqrt(sum / data.length);
  }
  
  /**
   * Get energy in a frequency band
   * @param {number} minHz - Minimum frequency
   * @param {number} maxHz - Maximum frequency
   * @returns {number} Normalized energy (0-1)
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
      // Convert dB to linear scale
      const linearValue = Math.pow(10, this.frequencyData[i] / 20);
      sum += linearValue;
      count++;
    }
    
    return count > 0 ? sum / count : 0;
  }
  
  /**
   * Update speaking state based on energy
   * @param {number} energy - Current energy level
   * @private
   */
  updateSpeakingState(energy) {
    const wasSpeaking = this.isSpeaking;
    
    if (energy >= this.config.minAmplitude) {
      this.silenceFrames = 0;
      this.isSpeaking = true;
    } else {
      this.silenceFrames++;
      if (this.silenceFrames >= this.silenceThreshold) {
        this.isSpeaking = false;
      }
    }
    
    // Fire callback on state change
    if (wasSpeaking !== this.isSpeaking && this.onSpeakingStateChange) {
      this.onSpeakingStateChange(this.isSpeaking);
    }
  }
  
  /**
   * Get current blendshapes without processing
   * @returns {Object} Current blendshape values
   */
  getCurrentBlendshapes() {
    return { ...this.currentBlendshapes };
  }
  
  /**
   * Get current speaking state
   * @returns {boolean} Whether audio is currently speaking
   */
  getIsSpeaking() {
    return this.isSpeaking;
  }
  
  /**
   * Update configuration
   * @param {Object} newConfig - New configuration values
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    this.frameInterval = 1000 / this.config.updateRate;
  }
  
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
    
    this.audioContext = null;
    console.log('[RealtimeLipSync] Disposed');
  }
}

// =============================================================================
// AUDIO ELEMENT WRAPPER
// =============================================================================

/**
 * Simple wrapper for analyzing HTMLAudioElement playback
 */
export class AudioElementLipSync extends RealtimeLipSync {
  /**
   * Create lip-sync analyzer for an audio element
   * @param {HTMLAudioElement} audioElement - Audio element to analyze
   * @param {Object} options - Configuration options
   */
  constructor(audioElement, options = {}) {
    super(options);
    
    this.audioElement = audioElement;
    this.isPlaying = false;
    
    // Bind event handlers
    this.handlePlay = () => {
      this.isPlaying = true;
      this.start();
    };
    this.handlePause = () => {
      this.isPlaying = false;
      this.stop();
    };
    this.handleEnded = () => {
      this.isPlaying = false;
      this.stop();
    };
    
    // Setup
    this.setup();
  }
  
  /**
   * Set up audio element connection and event listeners
   * @private
   */
  setup() {
    // Connect audio element
    this.connectAudioElement(this.audioElement);
    
    // Add event listeners
    this.audioElement.addEventListener('play', this.handlePlay);
    this.audioElement.addEventListener('pause', this.handlePause);
    this.audioElement.addEventListener('ended', this.handleEnded);
    
    console.log('[AudioElementLipSync] Set up for audio element');
  }
  
  /**
   * Clean up resources
   */
  dispose() {
    // Remove event listeners
    if (this.audioElement) {
      this.audioElement.removeEventListener('play', this.handlePlay);
      this.audioElement.removeEventListener('pause', this.handlePause);
      this.audioElement.removeEventListener('ended', this.handleEnded);
    }
    
    super.dispose();
  }
}

// =============================================================================
// WEBSOCKET STREAM WRAPPER
// =============================================================================

/**
 * Lip-sync analyzer for WebSocket audio streams (ElevenLabs streaming)
 */
export class StreamingLipSync extends RealtimeLipSync {
  /**
   * Create lip-sync analyzer for streaming audio
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    super(options);
    
    this.audioQueue = [];
    this.isProcessingQueue = false;
    this.currentAudioSource = null;
  }
  
  /**
   * Initialize for streaming
   */
  initStreaming() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    this.init(this.audioContext);
    
    // Create a gain node for output
    this.gainNode = this.audioContext.createGain();
    this.analyser.connect(this.gainNode);
    this.gainNode.connect(this.audioContext.destination);
    
    console.log('[StreamingLipSync] Initialized for streaming');
    return this;
  }
  
  /**
   * Add audio chunk to processing queue
   * @param {ArrayBuffer|Uint8Array} audioData - Encoded audio data (MP3, etc.)
   */
  async addAudioChunk(audioData) {
    // Convert to ArrayBuffer if needed
    const buffer = audioData instanceof ArrayBuffer 
      ? audioData 
      : audioData.buffer.slice(audioData.byteOffset, audioData.byteOffset + audioData.byteLength);
    
    this.audioQueue.push(buffer);
    
    if (!this.isProcessingQueue) {
      this.processAudioQueue();
    }
  }
  
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
      
      // Start analysis if not already running
      if (!this.isActive) {
        this.start();
      }
      
      // Play audio
      source.onended = () => {
        this.processAudioQueue();
      };
      source.start();
      
    } catch (error) {
      console.warn('[StreamingLipSync] Failed to decode audio chunk:', error);
      this.processAudioQueue(); // Try next chunk
    }
  }
  
  /**
   * Clear audio queue and stop current playback
   */
  clearQueue() {
    this.audioQueue = [];
    
    if (this.currentAudioSource) {
      try {
        this.currentAudioSource.stop();
      } catch (e) {
        // Ignore if already stopped
      }
      this.currentAudioSource = null;
    }
    
    this.isProcessingQueue = false;
  }
  
  /**
   * Clean up resources
   */
  dispose() {
    this.clearQueue();
    
    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }
    
    super.dispose();
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default RealtimeLipSync;

