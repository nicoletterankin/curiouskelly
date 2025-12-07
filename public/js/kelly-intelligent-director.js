/**
 * ═══════════════════════════════════════════════════════════════════════════
 * KELLY INTELLIGENT DIRECTOR v1.0
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * The brain behind Kelly's performances. Analyzes text for emotional content
 * and automatically directs Kelly's expressions, gestures, and vocal tone
 * in real-time during lessons.
 * 
 * This system turns static lesson content into dynamic, emotionally-intelligent
 * performances that respond to the meaning of what Kelly is saying.
 * 
 * Features:
 * - Real-time text emotion analysis
 * - Automatic expression transitions
 * - Voice modulation suggestions
 * - Performance timeline generation
 * - Unity 3D and 2D avatar support
 * 
 * Architecture:
 *   Text Input → Emotion Analysis → Expression Selection → Avatar Direction
 * 
 * Usage:
 *   KellyDirector.init();
 *   KellyDirector.analyzeAndDirect("Today we're going to learn something amazing!");
 *   // Kelly automatically shows excitement
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 */

// =============================================================================
// CANONICAL EXPRESSION CONFIGURATION
// =============================================================================

const KELLY_EXPRESSIONS = {
  // ─────────────────────────────────────────────────────────────────────────
  // Core Emotional Expressions
  // ─────────────────────────────────────────────────────────────────────────
  
  happy: {
    name: 'Happy',
    description: 'Warm, friendly smile - default positive state',
    blendshapes: {
      'Mouth_Smile_L': 70, 'Mouth_Smile_R': 70,
      'Cheek_Raise_L': 40, 'Cheek_Raise_R': 40,
      'Eye_Squint_L': 20, 'Eye_Squint_R': 20,
    },
    videoKeywords: ['warm smile', 'friendly expression', 'bright eyes'],
    voiceSettings: { stability: 0.6, similarity_boost: 0.8, style: 0.3 },
    triggers: ['wonderful', 'great', 'love', 'happy', 'joy', 'excited', 'amazing', 'beautiful'],
    duration: 'sustained',
    intensity: 0.7,
  },

  curious: {
    name: 'Curious',
    description: 'Raised brows, attentive gaze - discovery mode',
    blendshapes: {
      'Brow_Raise_Inner_L': 60, 'Brow_Raise_Inner_R': 60,
      'Brow_Raise_Outer_L': 40, 'Brow_Raise_Outer_R': 40,
      'Eye_Wide_L': 30, 'Eye_Wide_R': 30,
      'Mouth_Smile_L': 25, 'Mouth_Smile_R': 25,
    },
    videoKeywords: ['curious expression', 'raised eyebrows', 'interested look'],
    voiceSettings: { stability: 0.5, similarity_boost: 0.75, style: 0.4 },
    triggers: ['wonder', 'curious', 'what if', 'how', 'why', 'imagine', 'discover', 'question'],
    duration: 'medium',
    intensity: 0.6,
  },

  wisdom: {
    name: 'Wisdom',
    description: 'Knowing smile, gentle eyes - imparting insight',
    blendshapes: {
      'Mouth_Smile_L': 45, 'Mouth_Smile_R': 45,
      'Eye_Squint_L': 35, 'Eye_Squint_R': 35,
      'Brow_Raise_Inner_L': 20, 'Brow_Raise_Inner_R': 20,
    },
    videoKeywords: ['wise expression', 'knowing smile', 'thoughtful gaze'],
    voiceSettings: { stability: 0.75, similarity_boost: 0.85, style: 0.2 },
    triggers: ['truth', 'wisdom', 'understand', 'realize', 'deep', 'important', 'remember', 'lesson'],
    duration: 'sustained',
    intensity: 0.5,
  },

  thinking: {
    name: 'Thinking',
    description: 'Slight squint, processing - contemplation',
    blendshapes: {
      'Eye_Squint_L': 30, 'Eye_Squint_R': 30,
      'Brow_Raise_Inner_L': 35, 'Brow_Raise_Inner_R': 35,
      'Mouth_Shrug_Upper': 15,
    },
    videoKeywords: ['thoughtful expression', 'contemplating', 'considering'],
    voiceSettings: { stability: 0.7, similarity_boost: 0.7, style: 0.25 },
    triggers: ['think', 'consider', 'perhaps', 'maybe', 'might', 'ponder', 'reflect'],
    duration: 'brief',
    intensity: 0.4,
  },

  excited: {
    name: 'Excited',
    description: 'Wide eyes, big smile - high energy',
    blendshapes: {
      'Mouth_Smile_L': 90, 'Mouth_Smile_R': 90,
      'Eye_Wide_L': 50, 'Eye_Wide_R': 50,
      'Cheek_Raise_L': 60, 'Cheek_Raise_R': 60,
      'Brow_Raise_Inner_L': 50, 'Brow_Raise_Inner_R': 50,
    },
    videoKeywords: ['excited expression', 'wide eyes', 'enthusiastic'],
    voiceSettings: { stability: 0.4, similarity_boost: 0.8, style: 0.5 },
    triggers: ['wow', 'incredible', 'unbelievable', 'fantastic', 'awesome', 'surprise'],
    duration: 'brief',
    intensity: 0.9,
  },

  explaining: {
    name: 'Explaining',
    description: 'Engaged expression - teaching moment',
    blendshapes: {
      'Brow_Raise_Inner_L': 40, 'Brow_Raise_Inner_R': 40,
      'Mouth_Smile_L': 35, 'Mouth_Smile_R': 35,
      'Eye_Wide_L': 15, 'Eye_Wide_R': 15,
    },
    videoKeywords: ['explaining', 'teaching expression', 'engaged look'],
    voiceSettings: { stability: 0.65, similarity_boost: 0.75, style: 0.35 },
    triggers: ['because', 'therefore', 'this means', 'actually', 'in fact', 'for example'],
    duration: 'sustained',
    intensity: 0.5,
  },

  encouraging: {
    name: 'Encouraging',
    description: 'Warm supportive smile - motivational',
    blendshapes: {
      'Mouth_Smile_L': 65, 'Mouth_Smile_R': 65,
      'Cheek_Raise_L': 35, 'Cheek_Raise_R': 35,
      'Brow_Raise_Inner_L': 30, 'Brow_Raise_Inner_R': 30,
    },
    videoKeywords: ['encouraging smile', 'supportive expression', 'warm look'],
    voiceSettings: { stability: 0.6, similarity_boost: 0.85, style: 0.3 },
    triggers: ['you can', 'try', 'believe', 'possible', 'capable', 'strength', 'together'],
    duration: 'medium',
    intensity: 0.6,
  },

  listening: {
    name: 'Listening',
    description: 'Attentive, open expression - receiving mode',
    blendshapes: {
      'Brow_Raise_Inner_L': 25, 'Brow_Raise_Inner_R': 25,
      'Eye_Wide_L': 20, 'Eye_Wide_R': 20,
      'Mouth_Smile_L': 20, 'Mouth_Smile_R': 20,
    },
    videoKeywords: ['attentive expression', 'listening', 'open look'],
    voiceSettings: { stability: 0.7, similarity_boost: 0.75, style: 0.2 },
    triggers: ['what do you think', 'tell me', 'share', 'your turn'],
    duration: 'sustained',
    intensity: 0.3,
  },

  celebrating: {
    name: 'Celebrating',
    description: 'Full joy expression - achievement moment',
    blendshapes: {
      'Mouth_Smile_L': 100, 'Mouth_Smile_R': 100,
      'Mouth_Open': 30,
      'Eye_Wide_L': 40, 'Eye_Wide_R': 40,
      'Cheek_Raise_L': 70, 'Cheek_Raise_R': 70,
    },
    videoKeywords: ['celebrating', 'joyful expression', 'triumph'],
    voiceSettings: { stability: 0.35, similarity_boost: 0.85, style: 0.6 },
    triggers: ['congratulations', 'you did it', 'perfect', 'correct', 'excellent', 'bravo'],
    duration: 'brief',
    intensity: 1.0,
  },

  empathetic: {
    name: 'Empathetic',
    description: 'Soft, understanding expression - connecting',
    blendshapes: {
      'Brow_Raise_Inner_L': 35, 'Brow_Raise_Inner_R': 35,
      'Mouth_Smile_L': 30, 'Mouth_Smile_R': 30,
      'Eye_Squint_L': 15, 'Eye_Squint_R': 15,
    },
    videoKeywords: ['empathetic look', 'understanding expression', 'compassionate'],
    voiceSettings: { stability: 0.7, similarity_boost: 0.9, style: 0.2 },
    triggers: ['understand', 'feel', 'know how', 'difficult', 'challenge', 'struggle'],
    duration: 'medium',
    intensity: 0.5,
  },

  neutral: {
    name: 'Neutral',
    description: 'Calm, pleasant baseline',
    blendshapes: {
      'Mouth_Smile_L': 15, 'Mouth_Smile_R': 15,
    },
    videoKeywords: ['neutral expression', 'calm', 'pleasant'],
    voiceSettings: { stability: 0.7, similarity_boost: 0.75, style: 0.25 },
    triggers: [],
    duration: 'default',
    intensity: 0.2,
  },
};

// =============================================================================
// PHASE EXPRESSION MAPPINGS
// =============================================================================

const PHASE_EXPRESSIONS = {
  welcome: { primary: 'happy', secondary: 'excited', transition: 'warm' },
  hook: { primary: 'curious', secondary: 'excited', transition: 'intrigued' },
  q1: { primary: 'curious', secondary: 'explaining', transition: 'engaged' },
  q2: { primary: 'explaining', secondary: 'thinking', transition: 'focused' },
  q3: { primary: 'thinking', secondary: 'wisdom', transition: 'contemplative' },
  wisdom: { primary: 'wisdom', secondary: 'empathetic', transition: 'profound' },
  celebrating: { primary: 'celebrating', secondary: 'encouraging', transition: 'triumphant' },
  question: { primary: 'listening', secondary: 'curious', transition: 'receptive' },
};

// =============================================================================
// EMOTION DETECTION PATTERNS
// =============================================================================

const EMOTION_PATTERNS = {
  // Questions and curiosity
  curiosity: {
    patterns: [
      /\?/,
      /\bwhat\b/i, /\bhow\b/i, /\bwhy\b/i, /\bwhen\b/i, /\bwhere\b/i,
      /\bwonder\b/i, /\bcurious\b/i, /\bimagine\b/i, /\bdiscover\b/i,
    ],
    expression: 'curious',
    weight: 0.8,
  },

  // Excitement and surprise
  excitement: {
    patterns: [
      /!/,
      /\bwow\b/i, /\bamazing\b/i, /\bincredible\b/i, /\bunbelievable\b/i,
      /\bfantastic\b/i, /\bawesome\b/i, /\bexciting\b/i,
    ],
    expression: 'excited',
    weight: 0.9,
  },

  // Teaching and explaining
  explaining: {
    patterns: [
      /\bbecause\b/i, /\btherefore\b/i, /\bactually\b/i, /\bin fact\b/i,
      /\bthis means\b/i, /\bfor example\b/i, /\bhere's why\b/i,
      /\blet me explain\b/i, /\bthe reason\b/i,
    ],
    expression: 'explaining',
    weight: 0.7,
  },

  // Wisdom and insight
  wisdom: {
    patterns: [
      /\btruth\b/i, /\bwisdom\b/i, /\bremember\b/i, /\blesson\b/i,
      /\bimportant\b/i, /\brealize\b/i, /\bunderstand\b/i,
      /\bthe key is\b/i, /\bwhat matters\b/i,
    ],
    expression: 'wisdom',
    weight: 0.75,
  },

  // Encouragement
  encouragement: {
    patterns: [
      /\byou can\b/i, /\btry\b/i, /\bbelieve\b/i, /\bpossible\b/i,
      /\bdon't give up\b/i, /\btogether\b/i, /\bstrong\b/i,
    ],
    expression: 'encouraging',
    weight: 0.7,
  },

  // Celebration
  celebration: {
    patterns: [
      /\bcongrat/i, /\bperfect\b/i, /\bcorrect\b/i, /\bexcellent\b/i,
      /\byou did it\b/i, /\bwell done\b/i, /\bgreat job\b/i,
    ],
    expression: 'celebrating',
    weight: 0.95,
  },

  // Empathy
  empathy: {
    patterns: [
      /\bi understand\b/i, /\bi know how\b/i, /\bdifficult\b/i,
      /\bchalleng/i, /\bstruggle\b/i, /\bfeel\b/i,
    ],
    expression: 'empathetic',
    weight: 0.65,
  },

  // Thinking
  thinking: {
    patterns: [
      /\bthink\b/i, /\bconsider\b/i, /\bperhaps\b/i, /\bmaybe\b/i,
      /\bmight\b/i, /\bponder\b/i, /\bhmm\b/i,
    ],
    expression: 'thinking',
    weight: 0.6,
  },
};

// =============================================================================
// INTELLIGENT DIRECTOR CLASS
// =============================================================================

const KellyDirector = {
  // State
  isInitialized: false,
  isDirecting: false,
  currentExpression: 'neutral',
  expressionQueue: [],
  performanceTimeline: [],
  
  // Timing
  lastExpressionChange: 0,
  minExpressionDuration: 1500, // ms - don't change expressions too fast
  transitionDuration: 400, // ms
  
  // Integration references
  expressionBridge: null,
  unityBridge: null,
  lessonSystem: null,
  
  // Performance stats
  stats: {
    expressionsTriggered: 0,
    emotionsDetected: 0,
    averageConfidence: 0,
  },

  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================

  /**
   * Initialize the Intelligent Director
   */
  init(options = {}) {
    if (this.isInitialized) return this;

    // Store references to other systems
    this.expressionBridge = window.KellyExpressionBridge;
    this.unityBridge = window.unityBridge;
    this.lessonSystem = window.kellyLessonSystem;

    // Initialize expression bridge if not done
    if (this.expressionBridge && !this.expressionBridge.isInitialized) {
      this.expressionBridge.init();
    }

    // Apply options
    if (options.minExpressionDuration) {
      this.minExpressionDuration = options.minExpressionDuration;
    }
    if (options.transitionDuration) {
      this.transitionDuration = options.transitionDuration;
    }

    this.isInitialized = true;
    console.log('[KellyDirector] 🎬 Intelligent Director initialized');
    return this;
  },

  // ===========================================================================
  // MAIN ANALYSIS & DIRECTION
  // ===========================================================================

  /**
   * Analyze text and automatically direct Kelly's performance
   * @param {string} text - Text to analyze
   * @param {Object} options - Direction options
   * @returns {Object} Analysis result with expression decisions
   */
  analyzeAndDirect(text, options = {}) {
    if (!this.isInitialized) this.init();

    const analysis = this.analyzeText(text);
    
    // Check if enough time has passed since last expression change
    const now = Date.now();
    const timeSinceLastChange = now - this.lastExpressionChange;
    
    if (timeSinceLastChange < this.minExpressionDuration && !options.force) {
      // Queue the expression instead
      this.expressionQueue.push({
        expression: analysis.dominantExpression,
        confidence: analysis.confidence,
        timestamp: now + this.minExpressionDuration - timeSinceLastChange,
      });
      return analysis;
    }

    // Apply the expression
    this.applyExpression(analysis.dominantExpression, {
      duration: this.transitionDuration,
      source: 'director',
      confidence: analysis.confidence,
    });

    // Record to performance timeline
    this.performanceTimeline.push({
      timestamp: now,
      text: text.substring(0, 50) + '...',
      expression: analysis.dominantExpression,
      confidence: analysis.confidence,
      emotions: analysis.emotions,
    });

    // Update stats
    this.stats.expressionsTriggered++;
    this.stats.emotionsDetected += Object.keys(analysis.emotions).length;
    this.stats.averageConfidence = 
      (this.stats.averageConfidence * (this.stats.expressionsTriggered - 1) + analysis.confidence) / 
      this.stats.expressionsTriggered;

    return analysis;
  },

  /**
   * Analyze text for emotional content
   * @param {string} text - Text to analyze
   * @returns {Object} Analysis results
   */
  analyzeText(text) {
    const emotions = {};
    let maxScore = 0;
    let dominantEmotion = null;

    // Check each emotion pattern
    for (const [emotionKey, config] of Object.entries(EMOTION_PATTERNS)) {
      let score = 0;
      let matches = 0;

      for (const pattern of config.patterns) {
        const patternMatches = text.match(pattern);
        if (patternMatches) {
          matches += patternMatches.length;
        }
      }

      if (matches > 0) {
        // Calculate score based on matches and weight
        score = Math.min(1, (matches * 0.3) * config.weight);
        emotions[emotionKey] = {
          score,
          expression: config.expression,
          matches,
        };

        if (score > maxScore) {
          maxScore = score;
          dominantEmotion = emotionKey;
        }
      }
    }

    // Get the expression for the dominant emotion
    const dominantExpression = dominantEmotion 
      ? EMOTION_PATTERNS[dominantEmotion].expression 
      : 'neutral';

    // Calculate overall confidence
    const confidence = maxScore > 0 ? maxScore : 0.3;

    return {
      text,
      emotions,
      dominantEmotion,
      dominantExpression,
      confidence,
      suggestedVoice: KELLY_EXPRESSIONS[dominantExpression]?.voiceSettings || {},
    };
  },

  // ===========================================================================
  // EXPRESSION APPLICATION
  // ===========================================================================

  /**
   * Apply an expression to Kelly
   * @param {string} expressionName - Name of the expression
   * @param {Object} options - Application options
   */
  applyExpression(expressionName, options = {}) {
    const expression = KELLY_EXPRESSIONS[expressionName];
    if (!expression) {
      console.warn(`[KellyDirector] Unknown expression: ${expressionName}`);
      return;
    }

    const duration = options.duration || this.transitionDuration;

    // Update state
    this.currentExpression = expressionName;
    this.lastExpressionChange = Date.now();

    // Apply via expression bridge (handles 2D and Unity)
    if (this.expressionBridge) {
      this.expressionBridge.setCustomExpression(
        this.expressionToARKit(expression.blendshapes),
        duration
      );
    }

    // Also send directly to Unity if available
    if (window.unityInstance) {
      try {
        window.unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', expressionName);
      } catch (e) {
        // Unity not ready
      }
    }

    // Log
    console.log(`[KellyDirector] 🎭 Expression: ${expressionName} (confidence: ${(options.confidence * 100).toFixed(0)}%)`);
  },

  /**
   * Convert our blendshape format to ARKit-compatible format
   * @param {Object} blendshapes - Our blendshape format
   * @returns {Object} ARKit-compatible blendshapes
   */
  expressionToARKit(blendshapes) {
    const arkitMap = {
      'Mouth_Smile_L': 'mouthSmileLeft',
      'Mouth_Smile_R': 'mouthSmileRight',
      'Cheek_Raise_L': 'cheekSquintLeft',
      'Cheek_Raise_R': 'cheekSquintRight',
      'Eye_Squint_L': 'eyeSquintLeft',
      'Eye_Squint_R': 'eyeSquintRight',
      'Eye_Wide_L': 'eyeWideLeft',
      'Eye_Wide_R': 'eyeWideRight',
      'Brow_Raise_Inner_L': 'browInnerUp',
      'Brow_Raise_Inner_R': 'browInnerUp',
      'Brow_Raise_Outer_L': 'browOuterUpLeft',
      'Brow_Raise_Outer_R': 'browOuterUpRight',
      'Mouth_Open': 'jawOpen',
      'Mouth_Shrug_Upper': 'mouthShrugUpper',
    };

    const arkitBlendshapes = {};
    for (const [key, value] of Object.entries(blendshapes)) {
      const arkitKey = arkitMap[key];
      if (arkitKey) {
        // If the ARKit key already exists, average the values
        if (arkitBlendshapes[arkitKey] !== undefined) {
          arkitBlendshapes[arkitKey] = (arkitBlendshapes[arkitKey] + value) / 2;
        } else {
          arkitBlendshapes[arkitKey] = value;
        }
      }
    }

    return arkitBlendshapes;
  },

  // ===========================================================================
  // PHASE DIRECTION
  // ===========================================================================

  /**
   * Direct Kelly based on lesson phase
   * @param {string} phase - Lesson phase name
   * @param {string} text - Optional text content for the phase
   */
  directPhase(phase, text = '') {
    const phaseConfig = PHASE_EXPRESSIONS[phase];
    if (!phaseConfig) {
      console.warn(`[KellyDirector] Unknown phase: ${phase}`);
      return;
    }

    // If text is provided, analyze it first
    if (text) {
      const analysis = this.analyzeText(text);
      // If text has strong emotion, use that; otherwise use phase default
      if (analysis.confidence > 0.6) {
        this.applyExpression(analysis.dominantExpression, { confidence: analysis.confidence });
        return;
      }
    }

    // Use phase default expression
    this.applyExpression(phaseConfig.primary, { confidence: 0.7 });
  },

  // ===========================================================================
  // PERFORMANCE DIRECTION
  // ===========================================================================

  /**
   * Direct a full performance from a script
   * @param {string} script - Full script text
   * @param {Object} options - Performance options
   * @returns {Array} Performance timeline
   */
  directPerformance(script, options = {}) {
    const sentences = this.splitIntoSentences(script);
    const timeline = [];
    let currentTime = 0;
    const wordsPerSecond = options.wordsPerSecond || 2.5;

    for (const sentence of sentences) {
      const analysis = this.analyzeText(sentence);
      const duration = (sentence.split(/\s+/).length / wordsPerSecond) * 1000;

      timeline.push({
        start: currentTime,
        end: currentTime + duration,
        text: sentence,
        expression: analysis.dominantExpression,
        confidence: analysis.confidence,
        voiceSettings: analysis.suggestedVoice,
      });

      currentTime += duration;
    }

    return timeline;
  },

  /**
   * Split text into sentences
   * @param {string} text - Text to split
   * @returns {Array} Array of sentences
   */
  splitIntoSentences(text) {
    // Split on sentence-ending punctuation
    return text
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  },

  // ===========================================================================
  // REAL-TIME DIRECTION
  // ===========================================================================

  /**
   * Start real-time direction mode
   * Automatically directs Kelly based on lesson content as it progresses
   */
  startRealTimeDirection() {
    if (this.isDirecting) return;
    this.isDirecting = true;

    // Process expression queue
    this.queueProcessorInterval = setInterval(() => {
      this.processExpressionQueue();
    }, 100);

    console.log('[KellyDirector] 🎬 Real-time direction started');
  },

  /**
   * Process queued expressions
   */
  processExpressionQueue() {
    if (this.expressionQueue.length === 0) return;

    const now = Date.now();
    const timeSinceLastChange = now - this.lastExpressionChange;

    if (timeSinceLastChange >= this.minExpressionDuration) {
      const next = this.expressionQueue.shift();
      if (next) {
        this.applyExpression(next.expression, {
          confidence: next.confidence,
          source: 'queue',
        });
      }
    }
  },

  /**
   * Stop real-time direction
   */
  stopRealTimeDirection() {
    this.isDirecting = false;
    if (this.queueProcessorInterval) {
      clearInterval(this.queueProcessorInterval);
      this.queueProcessorInterval = null;
    }
    this.expressionQueue = [];
    console.log('[KellyDirector] 🎬 Real-time direction stopped');
  },

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Get available expressions
   * @returns {Object} Expression definitions
   */
  getExpressions() {
    return KELLY_EXPRESSIONS;
  },

  /**
   * Get phase mappings
   * @returns {Object} Phase to expression mappings
   */
  getPhaseExpressions() {
    return PHASE_EXPRESSIONS;
  },

  /**
   * Get current performance stats
   * @returns {Object} Performance statistics
   */
  getStats() {
    return {
      ...this.stats,
      timelineLength: this.performanceTimeline.length,
      currentExpression: this.currentExpression,
      isDirecting: this.isDirecting,
    };
  },

  /**
   * Get performance timeline
   * @returns {Array} Timeline of expression changes
   */
  getTimeline() {
    return this.performanceTimeline;
  },

  /**
   * Clear performance timeline
   */
  clearTimeline() {
    this.performanceTimeline = [];
    this.stats = {
      expressionsTriggered: 0,
      emotionsDetected: 0,
      averageConfidence: 0,
    };
  },

  /**
   * React to user action (correct answer, wrong answer, etc.)
   * @param {string} action - Action type
   */
  reactToUser(action) {
    const reactions = {
      correct: 'celebrating',
      incorrect: 'encouraging',
      timeout: 'empathetic',
      skip: 'thinking',
      start: 'excited',
      complete: 'celebrating',
    };

    const expression = reactions[action] || 'neutral';
    this.applyExpression(expression, { confidence: 0.9, force: true });
  },
};

// =============================================================================
// EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyDirector = KellyDirector;
  window.KELLY_EXPRESSIONS = KELLY_EXPRESSIONS;
  window.PHASE_EXPRESSIONS = PHASE_EXPRESSIONS;
}

// Auto-initialize on DOM ready
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyDirector.init());
  } else {
    KellyDirector.init();
  }
}

