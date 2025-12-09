/**
 * ═══════════════════════════════════════════════════════════════════════════
 * KELLY PERFORMANCE ENGINE v1.0
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Orchestrates Kelly's real-time performance by coordinating:
 * - Intelligent Director (emotion analysis)
 * - Expression Bridge (facial animations)
 * - Lip-Sync System (mouth movements)
 * - Audio System (ElevenLabs voice)
 * - Unity Bridge (3D avatar)
 * 
 * This is the CONDUCTOR that makes Kelly come alive.
 * 
 * Features:
 * - Automatic performance orchestration
 * - Real-time emotion-to-expression mapping
 * - Coordinated audio + visual performance
 * - Lesson phase awareness
 * - User interaction reactions
 * 
 * Usage:
 *   KellyPerformance.init();
 *   await KellyPerformance.perform("Today we're learning about something amazing!");
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 */

const KellyPerformance = {
  // ═══════════════════════════════════════════════════════════════════════════
  // STATE
  // ═══════════════════════════════════════════════════════════════════════════
  
  isInitialized: false,
  isPerforming: false,
  currentPhase: null,
  
  // System references
  systems: {
    director: null,      // KellyDirector
    expressions: null,   // KellyExpressionBridge
    lipsync: null,       // KellyLipSync
    audio: null,         // KellyAudio
    unity: null,         // UnityBridge
    lesson: null,        // KellyLessonSystem
  },
  
  // Performance queue
  performanceQueue: [],
  isProcessingQueue: false,
  
  // Current performance state
  current: {
    text: '',
    expression: 'neutral',
    phase: null,
    audioElement: null,
    startTime: 0,
  },
  
  // Callbacks
  callbacks: {
    onPerformanceStart: null,
    onPerformanceEnd: null,
    onExpressionChange: null,
    onPhaseChange: null,
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Initialize the Performance Engine
   * @param {Object} options - Configuration options
   */
  init(options = {}) {
    if (this.isInitialized) {
      console.log('[KellyPerformance] Already initialized');
      return this;
    }
    
    // Get system references
    this.systems.director = window.KellyDirector;
    this.systems.expressions = window.KellyExpressionBridge;
    this.systems.lipsync = window.KellyLipSync;
    this.systems.audio = window.KellyAudio;
    this.systems.unity = window.unityBridge;
    this.systems.lesson = window.kellyLessonSystem;
    
    // Initialize systems that need it
    if (this.systems.director && !this.systems.director.isInitialized) {
      this.systems.director.init();
    }
    if (this.systems.expressions && !this.systems.expressions.isInitialized) {
      this.systems.expressions.init();
    }
    if (this.systems.lipsync && !this.systems.lipsync.isInitialized) {
      this.systems.lipsync.init();
    }
    
    // Apply options
    if (options.callbacks) {
      Object.assign(this.callbacks, options.callbacks);
    }
    
    // Hook into lesson system if available
    this.hookIntoLessonSystem();
    
    this.isInitialized = true;
    console.log('[KellyPerformance] 🎭 Performance Engine initialized');
    console.log('[KellyPerformance] Systems:', {
      director: !!this.systems.director,
      expressions: !!this.systems.expressions,
      lipsync: !!this.systems.lipsync,
      audio: !!this.systems.audio,
      unity: !!this.systems.unity,
    });
    
    return this;
  },
  
  /**
   * Hook into the lesson system to automatically direct performance
   */
  hookIntoLessonSystem() {
    if (!this.systems.lesson) return;
    
    // Store original methods to wrap
    const originalSetLessonState = this.systems.lesson.setLessonState?.bind(this.systems.lesson);
    const originalAdvancePhase = this.systems.lesson.advancePhase?.bind(this.systems.lesson);
    
    if (originalSetLessonState) {
      this.systems.lesson.setLessonState = (lesson) => {
        originalSetLessonState(lesson);
        this.onLessonStart(lesson);
      };
    }
    
    if (originalAdvancePhase) {
      this.systems.lesson.advancePhase = () => {
        const phase = originalAdvancePhase();
        if (phase) {
          this.onPhaseAdvance(phase);
        }
        return phase;
      };
    }
    
    console.log('[KellyPerformance] Hooked into lesson system');
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MAIN PERFORMANCE API
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Perform text with intelligent expression and optional audio
   * @param {string} text - Text to perform
   * @param {Object} options - Performance options
   * @returns {Promise} Resolves when performance completes
   */
  async perform(text, options = {}) {
    if (!this.isInitialized) this.init();
    
    // Queue if already performing
    if (this.isPerforming && !options.interrupt) {
      this.performanceQueue.push({ text, options });
      return;
    }
    
    this.isPerforming = true;
    this.current.text = text;
    this.current.startTime = Date.now();
    
    // Notify start
    if (this.callbacks.onPerformanceStart) {
      this.callbacks.onPerformanceStart({ text, options });
    }
    
    try {
      // Step 1: Analyze text and set expression
      const analysis = this.systems.director?.analyzeAndDirect(text) || {
        dominantExpression: 'neutral',
        confidence: 0.5,
      };
      
      this.current.expression = analysis.dominantExpression;
      
      // Notify expression change
      if (this.callbacks.onExpressionChange) {
        this.callbacks.onExpressionChange(analysis.dominantExpression, analysis.confidence);
      }
      
      // Step 2: Generate/play audio if requested
      if (options.withAudio !== false && this.systems.audio) {
        await this.performWithAudio(text, analysis, options);
      } else {
        // Just perform silently (expression only)
        await this.performSilent(text, analysis, options);
      }
      
    } catch (error) {
      console.error('[KellyPerformance] Performance error:', error);
    } finally {
      this.isPerforming = false;
      
      // Notify end
      if (this.callbacks.onPerformanceEnd) {
        this.callbacks.onPerformanceEnd({ text, duration: Date.now() - this.current.startTime });
      }
      
      // Process queue
      this.processQueue();
    }
  },
  
  /**
   * Perform text with audio (TTS)
   */
  async performWithAudio(text, analysis, options) {
    // Get voice settings from analysis
    const voiceSettings = analysis.suggestedVoice || {};
    
    // Start lip-sync
    if (this.systems.lipsync) {
      this.systems.lipsync.startStreaming();
    }
    
    // Notify Unity of speaking state
    if (this.systems.unity) {
      this.systems.unity.setSpeaking(true);
      this.systems.unity.startLipSync(text);
    }
    
    // Generate and play audio
    if (this.systems.audio) {
      try {
        await this.systems.audio.speak(text, {
          ...voiceSettings,
          onAudioChunk: (chunk) => {
            // Feed to lip-sync
            if (this.systems.lipsync) {
              this.systems.lipsync.addAudioChunk(chunk);
            }
          },
        });
      } catch (error) {
        console.warn('[KellyPerformance] Audio failed, continuing silently:', error);
      }
    }
    
    // Stop lip-sync and speaking state
    if (this.systems.lipsync) {
      this.systems.lipsync.stop();
    }
    if (this.systems.unity) {
      this.systems.unity.setSpeaking(false);
      this.systems.unity.stopLipSync();
    }
  },
  
  /**
   * Perform text silently (expression only)
   */
  async performSilent(text, analysis, options) {
    // Calculate how long to hold the expression based on text length
    const wordsPerSecond = options.wordsPerSecond || 2.5;
    const wordCount = text.split(/\s+/).length;
    const duration = (wordCount / wordsPerSecond) * 1000;
    
    // Hold the expression
    await new Promise(resolve => setTimeout(resolve, Math.min(duration, 5000)));
  },
  
  /**
   * Process queued performances
   */
  async processQueue() {
    if (this.isProcessingQueue || this.performanceQueue.length === 0) return;
    
    this.isProcessingQueue = true;
    
    while (this.performanceQueue.length > 0 && !this.isPerforming) {
      const next = this.performanceQueue.shift();
      await this.perform(next.text, next.options);
    }
    
    this.isProcessingQueue = false;
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PHASE-AWARE PERFORMANCE
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Perform a lesson phase
   * @param {Object} phase - Phase data from lesson system
   * @param {Object} options - Performance options
   */
  async performPhase(phase, options = {}) {
    if (!phase) return;
    
    this.currentPhase = phase;
    
    // Notify phase change
    if (this.callbacks.onPhaseChange) {
      this.callbacks.onPhaseChange(phase);
    }
    
    // Direct the performance based on phase type
    if (this.systems.director) {
      this.systems.director.directPhase(phase.visualPhase || phase.type, phase.text);
    }
    
    // Set phase in Unity
    if (this.systems.unity) {
      this.systems.unity.setPhase(phase.visualPhase || phase.type);
    }
    
    // Perform the phase text
    if (phase.text && options.withAudio !== false) {
      await this.perform(phase.text, {
        phase: phase.type,
        ...options,
      });
    }
  },
  
  /**
   * React to user interaction
   * @param {string} action - User action type (correct, incorrect, skip, etc.)
   * @param {string} message - Optional reaction message
   */
  async reactToUser(action, message = '') {
    // Use director to set appropriate expression
    if (this.systems.director) {
      this.systems.director.reactToUser(action);
    }
    
    // Perform reaction message if provided
    if (message) {
      await this.perform(message, { withAudio: true });
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // LESSON SYSTEM HOOKS
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Called when a lesson starts
   */
  onLessonStart(lesson) {
    console.log('[KellyPerformance] Lesson started:', lesson?.topic);
    
    // Set excited expression for lesson start
    if (this.systems.director) {
      this.systems.director.applyExpression('excited', { confidence: 0.9 });
    }
    
    // Start real-time direction
    if (this.systems.director) {
      this.systems.director.startRealTimeDirection();
    }
  },
  
  /**
   * Called when phase advances
   */
  onPhaseAdvance(phase) {
    console.log('[KellyPerformance] Phase advanced:', phase?.type);
    
    // Direct the phase
    if (this.systems.director) {
      this.systems.director.directPhase(phase.visualPhase || phase.type, phase.text);
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DIRECT EXPRESSION CONTROL
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Set expression directly
   * @param {string} expression - Expression name
   * @param {Object} options - Options
   */
  setExpression(expression, options = {}) {
    if (this.systems.director) {
      this.systems.director.applyExpression(expression, options);
    }
    
    // Also set in expression bridge
    if (this.systems.expressions) {
      this.systems.expressions.setExpression(expression, options);
    }
    
    // And Unity
    if (this.systems.unity) {
      this.systems.unity.setExpression(expression);
    }
    
    this.current.expression = expression;
  },
  
  /**
   * Set phase context
   * @param {string} phase - Phase name
   */
  setPhase(phase) {
    if (this.systems.director) {
      this.systems.director.directPhase(phase);
    }
    
    if (this.systems.expressions) {
      this.systems.expressions.setPhaseExpression(phase);
    }
    
    if (this.systems.unity) {
      this.systems.unity.setPhase(phase);
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // UTILITY METHODS
  // ═══════════════════════════════════════════════════════════════════════════
  
  /**
   * Get current performance state
   */
  getState() {
    return {
      isPerforming: this.isPerforming,
      currentPhase: this.currentPhase,
      currentExpression: this.current.expression,
      queueLength: this.performanceQueue.length,
      systems: {
        director: !!this.systems.director?.isInitialized,
        expressions: !!this.systems.expressions?.isInitialized,
        lipsync: !!this.systems.lipsync?.isInitialized,
        audio: !!this.systems.audio,
        unity: !!this.systems.unity?.ready,
      },
    };
  },
  
  /**
   * Get available expressions
   */
  getExpressions() {
    return this.systems.director?.getExpressions() || {};
  },
  
  /**
   * Stop current performance
   */
  stop() {
    this.isPerforming = false;
    this.performanceQueue = [];
    
    // Stop all systems
    if (this.systems.lipsync) {
      this.systems.lipsync.stop();
    }
    if (this.systems.director) {
      this.systems.director.stopRealTimeDirection();
    }
    if (this.systems.unity) {
      this.systems.unity.stopLipSync();
      this.systems.unity.setSpeaking(false);
    }
    
    // Return to neutral
    this.setExpression('neutral');
    
    console.log('[KellyPerformance] Stopped');
  },
  
  /**
   * Clean up
   */
  dispose() {
    this.stop();
    this.isInitialized = false;
    this.systems = {
      director: null,
      expressions: null,
      lipsync: null,
      audio: null,
      unity: null,
      lesson: null,
    };
  },
};

// =============================================================================
// EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyPerformance = KellyPerformance;
}

// Auto-initialize after other systems are ready
if (typeof document !== 'undefined') {
  const initWhenReady = () => {
    // Wait for other systems
    setTimeout(() => {
      if (!KellyPerformance.isInitialized) {
        KellyPerformance.init();
      }
    }, 500);
  };
  
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWhenReady);
  } else {
    initWhenReady();
  }
}

