/**
 * Kelly Engine - Master Integration Controller
 * 
 * Connects: Voice Engine + Expression Generator + Phase Loader + Unity Bridge
 * This is the main entry point for the Curious Kelly application.
 * 
 * @module kelly-engine
 */

import ElevenLabsVoiceEngine from './elevenlabs-voice-engine.js';
import { ExpressionGenerator } from './expression-generator.js';
import PhaseLoader from './phase-loader.js';
import UnityBridge from './unity-bridge.js';
import UnityLoader from './unity-loader.js';
import UnityAssetManager from './unity-asset-manager.js';
import UnityAudioCoordinator from './unity-audio-coordinator.js';
import CacheManager from './cache-manager.js';
import supabaseService from './supabase-service.js';

// =============================================================================
// CONSTANTS
// =============================================================================

const PHASES = ['welcome', 'q1', 'q2', 'q3', 'wisdom'];

const AGE_BUCKETS = {
  '2-5':   { minAge: 2,  maxAge: 5,   description: 'Toddler/Preschool' },
  '6-12':  { minAge: 6,  maxAge: 12,  description: 'Elementary' },
  '13-17': { minAge: 13, maxAge: 17,  description: 'Teen' },
  '18-35': { minAge: 18, maxAge: 35,  description: 'Young Adult' },
  '36-60': { minAge: 36, maxAge: 60,  description: 'Adult' },
  '61-102': { minAge: 61, maxAge: 102, description: 'Wisdom Years' },
};

const DEFAULT_ARCHETYPES = [
  'The Scientist',
  'The Explorer', 
  'The Artist',
  'The Storyteller',
  'The Empath',
  'The Mystic',
  'The Provider',
  'The Diplomat',
  'The Architect',
  'The Rebel',
  'The Strategist',
  'The MacGyver',
  'The Survivor',
];

// =============================================================================
// KELLY ENGINE CLASS
// =============================================================================

export class KellyEngine {
  /**
   * Create a new Kelly Engine instance
   * @param {Object} config - Configuration options
   */
  constructor(config = {}) {
    // Configuration
    this.config = {
      unityContainerId: config.unityContainerId || 'unity-container',
      autoInitUnity: config.autoInitUnity !== false,
      ...config,
    };

    // Get API keys from config or environment
    const apiKey = config.elevenLabsApiKey || 
                   (typeof window !== 'undefined' && window.ELEVENLABS_API_KEY) ||
                   (typeof process !== 'undefined' && process.env?.ELEVENLABS_API_KEY);
    
    const voiceId = config.voiceId || 
                    (typeof window !== 'undefined' && window.ELEVENLABS_KELLY_VOICE_ID) ||
                    (typeof process !== 'undefined' && process.env?.ELEVENLABS_KELLY_VOICE_ID);

    // Initialize sub-systems
    this.voiceEngine = new ElevenLabsVoiceEngine({
      apiKey,
      voiceId,
    });
    
    this.expressionGenerator = new ExpressionGenerator();
    this.cacheManager = new CacheManager();
    this.supabase = supabaseService.client;
    
    // Unity components (initialized later)
    this.unityBridge = null;
    this.unityLoader = null;
    this.unityAssetManager = null;
    this.unityAudioCoordinator = null;
    this.phaseLoader = null;
    
    // Application state
    this.state = {
      initialized: false,
      unityReady: false,
      currentLesson: null,
      currentPhase: null,
      age: 27,
      ageBucket: '18-35',
      language: 'en',
      archetype: 'The Explorer',
      tone: 'enthusiastic',
      sessionId: this.generateSessionId(),
    };

    // Event callbacks
    this.callbacks = {
      onStateChange: config.onStateChange || null,
      onPhaseStart: config.onPhaseStart || null,
      onPhaseComplete: config.onPhaseComplete || null,
      onLessonComplete: config.onLessonComplete || null,
      onError: config.onError || null,
      onUnityReady: config.onUnityReady || null,
    };

    // Bind methods
    this.handleUnityReady = this.handleUnityReady.bind(this);
    this.handleAudioComplete = this.handleAudioComplete.bind(this);
  }

  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================

  /**
   * Initialize the Kelly Engine
   * @param {HTMLElement|string} unityContainer - Unity container element or ID
   * @returns {Promise<boolean>}
   */
  async initialize(unityContainer = null) {
    console.log('🚀 Initializing Kelly Engine...');
    
    try {
      // Resolve Unity container
      const container = unityContainer || 
                        document.getElementById(this.config.unityContainerId);
      
      // Initialize Unity Bridge
      this.unityBridge = new UnityBridge({
        bridgeVersion: '1.0.0',
        onStatusChange: (status) => console.log(`[Unity] ${status}`),
        onConnectionChange: (channel, state) => {
          if (state === 'connected') {
            this.handleUnityReady();
          }
        },
      });

      // Initialize Unity Loader
      if (container && this.config.autoInitUnity) {
        this.unityLoader = new UnityLoader({
          canvasId: 'unity-canvas',
          iframeId: 'unity-iframe',
          onLoad: () => this.handleUnityReady(),
          onError: (type, error) => this.handleError('unity_load', error),
        });
      }

      // Initialize Unity Asset Manager
      this.unityAssetManager = new UnityAssetManager(this.unityBridge);
      
      // Initialize Unity Audio Coordinator
      this.unityAudioCoordinator = new UnityAudioCoordinator(this.unityBridge);

      // Initialize Phase Loader
      this.phaseLoader = new PhaseLoader({
        stateManager: { getState: () => this.state },
        unityBridge: this.unityBridge,
        elevenLabsApiKey: this.voiceEngine.apiKey,
        voiceId: this.voiceEngine.voiceId,
        onLoadingStart: (phase) => console.log(`[PhaseLoader] Loading ${phase}...`),
        onLoadingEnd: (phase, result) => console.log(`[PhaseLoader] Loaded ${phase}`),
      });

      // Setup event listeners
      this.setupEventListeners();

      // Mark as initialized
      this.state.initialized = true;
      console.log('✅ Kelly Engine initialized!');
      
      return true;
    } catch (error) {
      console.error('❌ Kelly Engine initialization failed:', error);
      this.handleError('initialization', error);
      return false;
    }
  }

  /**
   * Setup global event listeners
   */
  setupEventListeners() {
    // Listen for Unity audio complete events
    if (typeof window !== 'undefined') {
      window.addEventListener('unity-audio-complete', this.handleAudioComplete);
      window.addEventListener('unity-disabled', () => {
        console.log('[KellyEngine] Unity disabled, continuing without avatar');
        this.state.unityReady = false;
      });
    }
  }

  /**
   * Handle Unity ready event
   */
  handleUnityReady() {
    console.log('🎮 Unity is ready!');
    this.state.unityReady = true;
    
    // Load character model for current age
    if (this.unityAssetManager) {
      this.unityAssetManager.loadCharacterModel(this.state.ageBucket, this.state.sessionId);
    }
    
    if (this.callbacks.onUnityReady) {
      this.callbacks.onUnityReady();
    }
  }

  /**
   * Handle audio playback complete
   */
  handleAudioComplete(event) {
    const { phase } = event.detail || {};
    console.log(`[KellyEngine] Audio complete for phase: ${phase}`);
    
    if (this.callbacks.onPhaseComplete) {
      this.callbacks.onPhaseComplete(phase);
    }

    // Auto-advance to next phase if configured
    // this.advancePhase();
  }

  // ===========================================================================
  // LESSON MANAGEMENT
  // ===========================================================================

  /**
   * Load a lesson by day number
   * @param {number} dayNumber - Day number (1-365)
   * @returns {Promise<Object>}
   */
  async loadLesson(dayNumber) {
    console.log(`📚 Loading lesson for day ${dayNumber}...`);
    
    try {
      // Fetch lesson from Supabase
      const lesson = await supabaseService.getCoreLesson(dayNumber);
      
      if (!lesson) {
        throw new Error(`Lesson not found for day ${dayNumber}`);
      }
      
      this.state.currentLesson = lesson;
      this.state.currentPhase = null;
      
      console.log(`✅ Loaded: ${lesson.topic}`);
      this.emitStateChange();
      
      // Start with welcome phase
      await this.loadPhase('welcome');
      
      return lesson;
    } catch (error) {
      console.error(`❌ Failed to load lesson ${dayNumber}:`, error);
      this.handleError('lesson_load', error);
      throw error;
    }
  }

  /**
   * Load today's lesson
   * @returns {Promise<Object>}
   */
  async loadTodayLesson() {
    const dayNumber = supabaseService.getTodayNumber();
    return this.loadLesson(dayNumber);
  }

  /**
   * Load a specific phase
   * @param {string} phase - Phase name ('welcome', 'q1', 'q2', 'q3', 'wisdom')
   * @returns {Promise<Object>}
   */
  async loadPhase(phase) {
    if (!this.state.currentLesson) {
      throw new Error('No lesson loaded. Call loadLesson() first.');
    }
    
    if (!PHASES.includes(phase)) {
      throw new Error(`Invalid phase: ${phase}. Expected one of: ${PHASES.join(', ')}`);
    }

    console.log(`🎬 Loading phase: ${phase}...`);
    
    try {
      this.state.currentPhase = phase;
      
      // Notify phase start
      if (this.callbacks.onPhaseStart) {
        this.callbacks.onPhaseStart(phase);
      }
      
      // Use PhaseLoader for optimized loading
      if (this.phaseLoader) {
        const result = await this.phaseLoader.loadPhase(phase, {
          ...this.state,
          selectedLesson: this.state.currentLesson,
        });
        
        this.emitStateChange();
        return result;
      }
      
      // Fallback: Manual loading
      return await this.loadPhaseManually(phase);
    } catch (error) {
      console.error(`❌ Failed to load phase ${phase}:`, error);
      this.handleError('phase_load', error);
      throw error;
    }
  }

  /**
   * Manual phase loading (fallback when PhaseLoader unavailable)
   * @private
   */
  async loadPhaseManually(phase) {
    const lesson = this.state.currentLesson;
    
    // Get phase content
    const content = await this.getPhaseContent(lesson, phase);
    
    if (!content || !content.script) {
      throw new Error(`No content available for phase: ${phase}`);
    }
    
    // Generate audio
    const audio = await this.voiceEngine.generatePhaseAudio(
      content.script,
      this.state.age,
      this.state.language,
      this.state.archetype,
      this.state.tone
    );
    
    // Generate expressions
    const expressions = this.expressionGenerator.generate({
      text: content.script,
      archetype: this.state.archetype,
      tone: this.state.tone,
      ageBucket: this.state.ageBucket,
      language: this.state.language,
      phase: phase,
    });
    
    // Send to Unity
    if (this.unityBridge && this.state.unityReady) {
      this.unityBridge.emit('audio-load', {
        url: audio.audioUrl,
        phase,
        autoplay: true,
      });
      
      this.unityBridge.emit('expression-data', {
        phase,
        expressions: expressions.expressions,
        gestures: expressions.gestures,
      });
    }
    
    return { content, audio, expressions };
  }

  /**
   * Get phase content from lesson
   * @private
   */
  async getPhaseContent(lesson, phase) {
    // Try to get from lesson_atoms
    const atom = await supabaseService.getAtom(
      lesson.id,
      this.state.archetype,
      phase
    );
    
    if (atom?.content) {
      return {
        script: atom.content.script || atom.content.text || JSON.stringify(atom.content),
        tone: atom.content.tone || this.state.tone,
      };
    }
    
    // Fallback: Use lesson content directly
    if (lesson.content && lesson.content[phase]) {
      return {
        script: lesson.content[phase],
        tone: this.state.tone,
      };
    }
    
    // Generate default content based on lesson topic
    return this.generateDefaultContent(lesson, phase);
  }

  /**
   * Generate default phase content
   * @private
   */
  generateDefaultContent(lesson, phase) {
    const topic = lesson.topic || lesson.title || 'today\'s topic';
    
    const templates = {
      welcome: `Hello! I'm Kelly, and today we're going to explore something fascinating: ${topic}! Are you ready to discover something amazing?`,
      q1: `Let me ask you a question about ${topic}. What do you think makes this interesting?`,
      q2: `Here's another thought about ${topic}. Can you imagine how this connects to your own life?`,
      q3: `Now let's think deeper about ${topic}. What patterns do you notice?`,
      wisdom: `Today we learned about ${topic}. Remember: curiosity is the key that opens every door to knowledge!`,
    };
    
    return {
      script: templates[phase] || templates.welcome,
      tone: 'enthusiastic',
    };
  }

  /**
   * Advance to the next phase
   */
  async advancePhase() {
    const currentIndex = PHASES.indexOf(this.state.currentPhase);
    
    if (currentIndex < 0 || currentIndex >= PHASES.length - 1) {
      // Lesson complete
      console.log('🎉 Lesson complete!');
      if (this.callbacks.onLessonComplete) {
        this.callbacks.onLessonComplete(this.state.currentLesson);
      }
      return null;
    }
    
    const nextPhase = PHASES[currentIndex + 1];
    return this.loadPhase(nextPhase);
  }

  // ===========================================================================
  // STATE MANAGEMENT
  // ===========================================================================

  /**
   * Update application state
   * @param {Object} newState - State updates
   */
  async updateState(newState) {
    const oldState = { ...this.state };
    Object.assign(this.state, newState);
    
    // Handle age change
    if (newState.age !== undefined && newState.age !== oldState.age) {
      this.state.ageBucket = this.getAgeBucket(newState.age);
      
      // Reload character model
      if (this.unityAssetManager && this.state.unityReady) {
        await this.unityAssetManager.loadCharacterModel(
          this.state.ageBucket,
          this.state.sessionId
        );
      }
      
      // Reload current phase with new age
      if (this.state.currentPhase && this.state.currentLesson) {
        await this.loadPhase(this.state.currentPhase);
      }
    }
    
    // Handle language change
    if (newState.language !== undefined && newState.language !== oldState.language) {
      if (this.state.currentPhase && this.state.currentLesson) {
        await this.loadPhase(this.state.currentPhase);
      }
    }
    
    // Handle archetype change
    if (newState.archetype !== undefined && newState.archetype !== oldState.archetype) {
      if (this.state.currentPhase && this.state.currentLesson) {
        await this.loadPhase(this.state.currentPhase);
      }
    }
    
    this.emitStateChange();
  }

  /**
   * Get current state
   * @returns {Object}
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Emit state change event
   */
  emitStateChange() {
    if (this.callbacks.onStateChange) {
      this.callbacks.onStateChange(this.state);
    }
  }

  // ===========================================================================
  // UTILITY METHODS
  // ===========================================================================

  /**
   * Get age bucket from numeric age
   * @param {number} age - Age in years (2-102)
   * @returns {string} Age bucket key
   */
  getAgeBucket(age) {
    const normalizedAge = Math.max(2, Math.min(102, age));
    
    for (const [bucket, config] of Object.entries(AGE_BUCKETS)) {
      if (normalizedAge >= config.minAge && normalizedAge <= config.maxAge) {
        return bucket;
      }
    }
    
    return '18-35'; // Default fallback
  }

  /**
   * Generate a unique session ID
   * @returns {string}
   */
  generateSessionId() {
    return `ck-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle errors
   * @private
   */
  handleError(type, error) {
    console.error(`[KellyEngine] Error (${type}):`, error);
    
    if (this.callbacks.onError) {
      this.callbacks.onError({ type, error, message: error.message });
    }
  }

  // ===========================================================================
  // STATISTICS & DIAGNOSTICS
  // ===========================================================================

  /**
   * Get engine statistics
   * @returns {Object}
   */
  getStats() {
    return {
      initialized: this.state.initialized,
      unityReady: this.state.unityReady,
      currentLesson: this.state.currentLesson?.topic || null,
      currentPhase: this.state.currentPhase,
      sessionId: this.state.sessionId,
      phaseLoaderStats: this.phaseLoader?.getStats() || {},
      cacheStats: this.cacheManager.getStats(),
      voiceEngineStats: this.voiceEngine.getRequestHistory(),
    };
  }

  /**
   * Get available archetypes
   * @returns {string[]}
   */
  getArchetypes() {
    return [...DEFAULT_ARCHETYPES];
  }

  /**
   * Get available languages
   * @returns {string[]}
   */
  getLanguages() {
    return ['en', 'es', 'fr'];
  }

  /**
   * Get phase list
   * @returns {string[]}
   */
  getPhases() {
    return [...PHASES];
  }
}

// =============================================================================
// FACTORY FUNCTION
// =============================================================================

/**
 * Create a KellyEngine instance with defaults
 * @param {Object} config - Configuration options
 * @returns {KellyEngine}
 */
export function createKellyEngine(config = {}) {
  return new KellyEngine(config);
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

export default KellyEngine;










