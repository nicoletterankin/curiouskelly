/**
 * Kelly Visual Learning System v1.0
 * 
 * A complete visual storytelling engine for educational content.
 * This system transforms each lesson into an immersive visual journey
 * where Kelly teaches with dynamic backgrounds, props, and choreographed poses.
 * 
 * Philosophy: Every frame should TEACH, not just entertain.
 * - Backgrounds create context and atmosphere
 * - Props are visual aids that reinforce concepts
 * - Kelly's poses guide attention and emotional state
 * - Transitions build narrative momentum
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 * @license Proprietary
 */

// ═══════════════════════════════════════════════════════════════════════════
// VISUAL SYSTEM CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const VISUAL_CONFIG = {
  // Asset paths - NEW: Phase-based structure
  PHASE_ASSETS_BASE: '/kelly/phases',  // New phase visuals
  LESSON_ASSETS_BASE: '/kelly/lessons', // Legacy
  POSES_BASE: '/kelly/poses',
  
  // Phase asset names (in each day folder)
  PHASE_FILES: {
    hook: 'hook.png',
    q1: 'q1.png',
    q2: 'q2.png', 
    q3: 'q3.png',
    wisdom: 'wisdom.png'
  },
  
  // Legacy asset types per lesson
  ASSET_TYPES: {
    BACKGROUND: 'bg',
    HERO: 'hero',           // Kelly full-body in environment
    PROP: 'prop',           // Kelly presenting visual aid
    REACTION: 'reaction',   // Kelly close-up reaction
    GUIDE_POINT: 'guide-point'  // Kelly pointing/explaining
  },
  
  // Phase-to-visual mapping (updated for new system)
  PHASE_VISUALS: {
    welcome: { phaseFile: 'hook', showProp: false, overlay: 'welcome' },
    hook: { phaseFile: 'hook', showProp: false, overlay: 'hook' },
    q1: { phaseFile: 'q1', showProp: false, overlay: 'question' },
    q2: { phaseFile: 'q2', showProp: false, overlay: 'question' },
    q3: { phaseFile: 'q3', showProp: false, overlay: 'question' },
    feedback_correct: { phaseFile: null, showProp: false, overlay: 'success' },
    feedback_incorrect: { phaseFile: null, showProp: false, overlay: 'encourage' },
    wisdom: { phaseFile: 'wisdom', showProp: false, overlay: 'wisdom' },
    complete: { phaseFile: 'wisdom', showProp: false, overlay: 'celebration' }
  },
  
  // Transition timings (ms)
  TRANSITIONS: {
    BACKGROUND_FADE: 800,
    KELLY_CROSSFADE: 400,
    PROP_SLIDE: 600,
    OVERLAY_FADE: 300
  },
  
  // Fallback poses when lesson-specific assets unavailable
  FALLBACK_POSES: {
    hero: 'kelly_welcome.png',
    prop: 'kelly_idle.png',
    reaction: 'kelly_clasp.png',
    'guide-point': 'kelly_hint.png'
  },
  
  // Default gradient backgrounds by lesson category
  CATEGORY_GRADIENTS: {
    science: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
    nature: 'linear-gradient(135deg, #134e5e 0%, #71b280 100%)',
    social: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%)',
    philosophy: 'linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%)',
    health: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    creativity: 'linear-gradient(135deg, #fc466b 0%, #3f5efb 100%)',
    default: 'linear-gradient(135deg, #0f0f11 0%, #1a1a2e 50%, #16213e 100%)'
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// ASSET PRELOADER
// Ensures smooth transitions by preloading all lesson assets
// ═══════════════════════════════════════════════════════════════════════════

class AssetPreloader {
  constructor() {
    this.cache = new Map();
    this.loading = new Map();
  }
  
  /**
   * Preload all assets for a lesson (NEW: Phase-based structure)
   * @param {number} dayNumber - The lesson day (1-365)
   * @returns {Promise<Object>} Map of phase names to loaded image elements
   */
  async preloadLesson(dayNumber) {
    const paddedDay = String(dayNumber).padStart(3, '0');
    const basePath = `${VISUAL_CONFIG.PHASE_ASSETS_BASE}/${paddedDay}`;
    
    const assetPromises = {};
    
    // Load phase-based assets (hook, q1, q2, q3, wisdom)
    for (const [phaseName, fileName] of Object.entries(VISUAL_CONFIG.PHASE_FILES)) {
      const path = `${basePath}/${fileName}`;
      assetPromises[phaseName] = this.loadImage(path);
    }
    
    const results = await Promise.allSettled(Object.values(assetPromises));
    const assets = {};
    
    Object.keys(assetPromises).forEach((phaseName, index) => {
      assets[phaseName] = results[index].status === 'fulfilled' ? results[index].value : null;
    });
    
    console.log(`[VisualSystem] Preloaded lesson ${dayNumber}:`, 
      Object.entries(assets).map(([k, v]) => `${k}:${v ? '✓' : '✗'}`).join(', '));
    
    return assets;
  }
  
  /**
   * Load a single image with caching
   * @param {string} src - Image source URL
   * @returns {Promise<HTMLImageElement>}
   */
  loadImage(src) {
    // Check cache first
    if (this.cache.has(src)) {
      return Promise.resolve(this.cache.get(src));
    }
    
    // Check if already loading
    if (this.loading.has(src)) {
      return this.loading.get(src);
    }
    
    // Start loading
    const promise = new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.cache.set(src, img);
        this.loading.delete(src);
        resolve(img);
      };
      img.onerror = () => {
        this.loading.delete(src);
        reject(new Error(`Failed to load: ${src}`));
      };
      img.src = src;
    });
    
    this.loading.set(src, promise);
    return promise;
  }
  
  /**
   * Preload adjacent lessons for smooth navigation
   * @param {number} currentDay - Current lesson day
   */
  preloadAdjacent(currentDay) {
    // Preload previous and next lessons
    if (currentDay > 1) this.preloadLesson(currentDay - 1);
    if (currentDay < 365) this.preloadLesson(currentDay + 1);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// VISUAL SCENE MANAGER
// Controls the layered visual composition of each lesson
// ═══════════════════════════════════════════════════════════════════════════

class VisualSceneManager {
  constructor(containerElement) {
    this.container = containerElement;
    this.preloader = new AssetPreloader();
    this.currentDay = null;
    this.currentPhase = null;
    this.assets = null;
    
    this.setupDOM();
  }
  
  /**
   * Create the layered DOM structure for visual scenes
   */
  setupDOM() {
    // Clear container
    this.container.innerHTML = '';
    
    // Create visual layers (back to front)
    this.layers = {
      // Layer 1: Background environment
      background: this.createLayer('visual-layer-bg', 1),
      
      // Layer 2: Atmospheric effects (particles, glow)
      atmosphere: this.createLayer('visual-layer-atmosphere', 2),
      
      // Layer 3: Kelly avatar
      kelly: this.createLayer('visual-layer-kelly', 3),
      
      // Layer 4: Props and visual aids
      props: this.createLayer('visual-layer-props', 4),
      
      // Layer 5: UI overlays (text, buttons)
      overlay: this.createLayer('visual-layer-overlay', 5)
    };
    
    // Create Kelly image element
    this.kellyImg = document.createElement('img');
    this.kellyImg.className = 'kelly-visual';
    this.kellyImg.alt = 'Kelly';
    this.layers.kelly.appendChild(this.kellyImg);
    
    // Create prop container
    this.propContainer = document.createElement('div');
    this.propContainer.className = 'prop-container';
    this.layers.props.appendChild(this.propContainer);
    
    // Add base styles
    this.injectStyles();
    
    console.log('[VisualSystem] DOM structure initialized');
  }
  
  /**
   * Create a visual layer
   */
  createLayer(className, zIndex) {
    const layer = document.createElement('div');
    layer.className = `visual-layer ${className}`;
    layer.style.cssText = `
      position: absolute;
      inset: 0;
      z-index: ${zIndex};
      pointer-events: none;
    `;
    this.container.appendChild(layer);
    return layer;
  }
  
  /**
   * Inject CSS styles for the visual system
   */
  injectStyles() {
    if (document.getElementById('kelly-visual-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'kelly-visual-styles';
    style.textContent = `
      /* ═══════════════════════════════════════════════════════════════════
         KELLY VISUAL SYSTEM STYLES
         ═══════════════════════════════════════════════════════════════════ */
      
      /* Background Layer */
      .visual-layer-bg {
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        transition: background-image ${VISUAL_CONFIG.TRANSITIONS.BACKGROUND_FADE}ms ease-out,
                    opacity ${VISUAL_CONFIG.TRANSITIONS.BACKGROUND_FADE}ms ease-out;
      }
      
      .visual-layer-bg::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(
          to bottom,
          rgba(0, 0, 0, 0) 0%,
          rgba(0, 0, 0, 0.3) 70%,
          rgba(0, 0, 0, 0.7) 100%
        );
        pointer-events: none;
      }
      
      /* Atmosphere Layer - Subtle effects */
      .visual-layer-atmosphere {
        opacity: 0.5;
        mix-blend-mode: screen;
      }
      
      /* Kelly Layer */
      .visual-layer-kelly {
        display: flex;
        align-items: center;
        justify-content: center;
      }
      
      .kelly-visual {
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: 100%;
        object-fit: contain;
        object-position: center bottom;
        transition: opacity ${VISUAL_CONFIG.TRANSITIONS.KELLY_CROSSFADE}ms ease-out,
                    transform ${VISUAL_CONFIG.TRANSITIONS.KELLY_CROSSFADE}ms ease-out;
      }
      
      .kelly-visual.transitioning {
        opacity: 0;
        transform: scale(0.98);
      }
      
      /* Prop Container */
      .prop-container {
        position: absolute;
        bottom: 20%;
        left: 10%;
        width: 30%;
        opacity: 0;
        transform: translateY(20px);
        transition: opacity ${VISUAL_CONFIG.TRANSITIONS.PROP_SLIDE}ms ease-out,
                    transform ${VISUAL_CONFIG.TRANSITIONS.PROP_SLIDE}ms ease-out;
        pointer-events: auto;
      }
      
      .prop-container.visible {
        opacity: 1;
        transform: translateY(0);
      }
      
      .prop-container img {
        width: 100%;
        height: auto;
        filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.5));
      }
      
      /* Phase-specific overlays */
      .visual-layer-overlay {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 20px;
        padding-bottom: calc(120px + env(safe-area-inset-bottom, 34px));
        pointer-events: auto;
      }
      
      /* Celebration particles */
      @keyframes float-up {
        0% {
          transform: translateY(100vh) rotate(0deg);
          opacity: 1;
        }
        100% {
          transform: translateY(-100vh) rotate(720deg);
          opacity: 0;
        }
      }
      
      .celebration-particle {
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: float-up 3s ease-out forwards;
      }
      
      /* Success glow effect */
      @keyframes success-pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(34, 197, 94, 0.3); }
        50% { box-shadow: 0 0 40px rgba(34, 197, 94, 0.6); }
      }
      
      .success-glow {
        animation: success-pulse 1s ease-in-out 2;
      }
      
      /* Wisdom moment styling */
      .wisdom-overlay {
        background: linear-gradient(
          to top,
          rgba(139, 92, 246, 0.3) 0%,
          transparent 50%
        );
      }
      
      /* Loading shimmer */
      @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
      }
      
      .loading-shimmer {
        background: linear-gradient(
          90deg,
          rgba(255, 255, 255, 0) 0%,
          rgba(255, 255, 255, 0.1) 50%,
          rgba(255, 255, 255, 0) 100%
        );
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
      }
    `;
    document.head.appendChild(style);
  }
  
  /**
   * Load and display a lesson's visual assets
   * @param {number} dayNumber - Lesson day (1-365)
   * @param {string} initialPhase - Starting phase
   */
  async loadLesson(dayNumber, initialPhase = 'welcome') {
    console.log(`[VisualSystem] Loading lesson ${dayNumber}, phase: ${initialPhase}`);
    
    this.currentDay = dayNumber;
    
    // Show loading state
    this.layers.background.classList.add('loading-shimmer');
    
    try {
      // Preload all assets for this lesson
      this.assets = await this.preloader.preloadLesson(dayNumber);
      
      // Preload adjacent lessons for smooth navigation
      this.preloader.preloadAdjacent(dayNumber);
      
      // Set initial visual state
      await this.setPhase(initialPhase);
      
      // Remove loading state
      this.layers.background.classList.remove('loading-shimmer');
      
      console.log(`[VisualSystem] Lesson ${dayNumber} loaded successfully`);
      
    } catch (error) {
      console.error('[VisualSystem] Failed to load lesson:', error);
      this.showFallbackVisuals(dayNumber);
    }
  }
  
  /**
   * Transition to a new phase with appropriate visuals (NEW: Phase-based)
   * @param {string} phase - The phase name
   * @param {Object} options - Additional options
   */
  async setPhase(phase, options = {}) {
    const phaseConfig = VISUAL_CONFIG.PHASE_VISUALS[phase] || VISUAL_CONFIG.PHASE_VISUALS.welcome;
    this.currentPhase = phase;
    
    console.log(`[VisualSystem] Setting phase: ${phase}`, phaseConfig);
    
    // Update Kelly/scene image based on phase
    if (phaseConfig.phaseFile) {
      await this.setPhaseImage(phaseConfig.phaseFile);
    }
    
    // Handle props (currently disabled - Kelly+context in one image)
    if (phaseConfig.showProp) {
      this.showProp();
    } else {
      this.hideProp();
    }
    
    // Apply phase-specific overlay effects
    this.setOverlayEffect(phaseConfig.overlay);
    
    // Dispatch event for other systems
    document.dispatchEvent(new CustomEvent('visual-phase-change', {
      detail: { phase, dayNumber: this.currentDay, config: phaseConfig }
    }));
  }
  
  /**
   * Set the phase image (Kelly in context)
   * @param {string} phaseName - Phase name (hook, q1, q2, q3, wisdom)
   */
  async setPhaseImage(phaseName) {
    const asset = this.assets?.[phaseName];
    
    // Start transition
    this.kellyImg.classList.add('transitioning');
    
    await new Promise(resolve => setTimeout(resolve, VISUAL_CONFIG.TRANSITIONS.KELLY_CROSSFADE / 2));
    
    if (asset) {
      this.kellyImg.src = asset.src;
      // Clear background since phase image includes environment
      this.layers.background.style.backgroundImage = 'none';
      this.layers.background.style.background = 'transparent';
    } else {
      // Use fallback pose
      const fallback = VISUAL_CONFIG.FALLBACK_POSES.hero || 'kelly_welcome.png';
      this.kellyImg.src = `${VISUAL_CONFIG.POSES_BASE}/${fallback}`;
      // Show gradient background as fallback
      const category = this.getLessonCategory(this.currentDay);
      this.layers.background.style.background = 
        VISUAL_CONFIG.CATEGORY_GRADIENTS[category] || VISUAL_CONFIG.CATEGORY_GRADIENTS.default;
    }
    
    // End transition
    this.kellyImg.classList.remove('transitioning');
  }
  
  /**
   * Set the background image for current lesson
   */
  async setBackground() {
    const bgAsset = this.assets?.bg;
    
    if (bgAsset) {
      this.layers.background.style.backgroundImage = `url(${bgAsset.src})`;
    } else {
      // Use category-based gradient fallback
      const category = this.getLessonCategory(this.currentDay);
      this.layers.background.style.backgroundImage = 
        VISUAL_CONFIG.CATEGORY_GRADIENTS[category] || VISUAL_CONFIG.CATEGORY_GRADIENTS.default;
    }
  }
  
  /**
   * Set Kelly's image with smooth transition
   * @param {string} variant - 'hero', 'prop', 'reaction', or 'guide-point'
   */
  async setKellyImage(variant) {
    const asset = this.assets?.[variant];
    
    // Start transition
    this.kellyImg.classList.add('transitioning');
    
    await new Promise(resolve => setTimeout(resolve, VISUAL_CONFIG.TRANSITIONS.KELLY_CROSSFADE / 2));
    
    if (asset) {
      this.kellyImg.src = asset.src;
    } else {
      // Use fallback pose
      const fallback = VISUAL_CONFIG.FALLBACK_POSES[variant] || VISUAL_CONFIG.FALLBACK_POSES.hero;
      this.kellyImg.src = `${VISUAL_CONFIG.POSES_BASE}/${fallback}`;
    }
    
    // End transition
    this.kellyImg.classList.remove('transitioning');
  }
  
  /**
   * Show the prop/visual aid for current lesson
   */
  showProp() {
    const propAsset = this.assets?.prop;
    
    if (propAsset) {
      this.propContainer.innerHTML = `<img src="${propAsset.src}" alt="Visual aid">`;
      this.propContainer.classList.add('visible');
    }
  }
  
  /**
   * Hide the prop/visual aid
   */
  hideProp() {
    this.propContainer.classList.remove('visible');
  }
  
  /**
   * Apply phase-specific overlay effects
   * @param {string} overlayType - Type of overlay effect
   */
  setOverlayEffect(overlayType) {
    // Remove existing overlays
    this.layers.overlay.className = 'visual-layer visual-layer-overlay';
    
    switch (overlayType) {
      case 'success':
        this.layers.overlay.classList.add('success-glow');
        this.triggerCelebration(5); // Subtle celebration
        break;
        
      case 'celebration':
        this.triggerCelebration(20); // Full celebration
        break;
        
      case 'wisdom':
        this.layers.overlay.classList.add('wisdom-overlay');
        break;
        
      case 'encourage':
        // Warm encouraging overlay
        this.layers.background.style.filter = 'brightness(1.1) saturate(1.1)';
        setTimeout(() => {
          this.layers.background.style.filter = '';
        }, 1000);
        break;
    }
  }
  
  /**
   * Trigger celebration particle effect
   * @param {number} count - Number of particles
   */
  triggerCelebration(count = 10) {
    const colors = ['#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];
    
    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.className = 'celebration-particle';
      particle.style.cssText = `
        left: ${Math.random() * 100}%;
        background: ${colors[Math.floor(Math.random() * colors.length)]};
        animation-delay: ${Math.random() * 0.5}s;
        animation-duration: ${2 + Math.random() * 2}s;
      `;
      this.layers.atmosphere.appendChild(particle);
      
      // Clean up after animation
      setTimeout(() => particle.remove(), 5000);
    }
  }
  
  /**
   * Show fallback visuals when assets unavailable
   * @param {number} dayNumber - Lesson day
   */
  showFallbackVisuals(dayNumber) {
    const category = this.getLessonCategory(dayNumber);
    this.layers.background.style.background = 
      VISUAL_CONFIG.CATEGORY_GRADIENTS[category] || VISUAL_CONFIG.CATEGORY_GRADIENTS.default;
    
    this.kellyImg.src = `${VISUAL_CONFIG.POSES_BASE}/${VISUAL_CONFIG.FALLBACK_POSES.hero}`;
    this.hideProp();
  }
  
  /**
   * Determine lesson category based on day number for fallback styling
   * @param {number} dayNumber - Lesson day
   * @returns {string} Category name
   */
  getLessonCategory(dayNumber) {
    // Map day ranges to categories based on curriculum themes
    if (dayNumber <= 7 || (dayNumber >= 31 && dayNumber <= 60)) return 'nature';
    if (dayNumber >= 8 && dayNumber <= 15) return 'social';
    if (dayNumber >= 16 && dayNumber <= 22) return 'health';
    if (dayNumber >= 23 && dayNumber <= 30) return 'creativity';
    if (dayNumber >= 61 && dayNumber <= 100) return 'science';
    return 'default';
  }
  
  /**
   * Cleanup resources
   */
  destroy() {
    this.container.innerHTML = '';
    this.assets = null;
    console.log('[VisualSystem] Destroyed');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// VISUAL CHOREOGRAPHER
// Orchestrates visual transitions synchronized with lesson phases and audio
// ═══════════════════════════════════════════════════════════════════════════

class VisualChoreographer {
  constructor(sceneManager, options = {}) {
    this.scene = sceneManager;
    this.options = {
      audioSync: true,
      autoAdvance: true,
      ...options
    };
    
    this.timeline = [];
    this.currentCue = 0;
    this.isPlaying = false;
    
    // Listen for audio events if sync enabled
    if (this.options.audioSync) {
      this.setupAudioSync();
    }
  }
  
  /**
   * Setup audio synchronization
   */
  setupAudioSync() {
    document.addEventListener('kelly-speech-start', (e) => {
      this.onSpeechStart(e.detail);
    });
    
    document.addEventListener('kelly-speech-end', (e) => {
      this.onSpeechEnd(e.detail);
    });
    
    document.addEventListener('kelly-speech-word', (e) => {
      this.onSpeechWord(e.detail);
    });
  }
  
  /**
   * Load choreography for a lesson phase
   * @param {string} phase - Phase name
   * @param {Object} lessonData - Lesson content data
   */
  loadPhaseChoreography(phase, lessonData) {
    // Build timeline of visual cues based on phase
    this.timeline = this.buildTimeline(phase, lessonData);
    this.currentCue = 0;
    
    console.log(`[Choreographer] Loaded ${this.timeline.length} cues for phase: ${phase}`);
  }
  
  /**
   * Build visual timeline for a phase
   * @param {string} phase - Phase name
   * @param {Object} lessonData - Lesson data
   * @returns {Array} Timeline of cues
   */
  buildTimeline(phase, lessonData) {
    const timeline = [];
    
    switch (phase) {
      case 'welcome':
        timeline.push(
          { time: 0, action: 'kelly', value: 'hero', description: 'Kelly enters' },
          { time: 500, action: 'background', value: 'fade-in', description: 'Environment reveals' },
          { time: 1000, action: 'overlay', value: 'title', description: 'Lesson title appears' }
        );
        break;
        
      case 'hook':
        timeline.push(
          { time: 0, action: 'kelly', value: 'guide-point', description: 'Kelly introduces topic' },
          { time: 1000, action: 'prop', value: 'show', description: 'Visual aid appears' },
          { time: 2000, action: 'highlight', value: 'prop', description: 'Focus on visual' }
        );
        break;
        
      case 'q1':
      case 'q2':
      case 'q3':
        timeline.push(
          { time: 0, action: 'kelly', value: 'prop', description: 'Kelly presents question' },
          { time: 500, action: 'prop', value: 'show', description: 'Show related visual' },
          { time: 1500, action: 'overlay', value: 'choices', description: 'Reveal answer choices' }
        );
        break;
        
      case 'wisdom':
        timeline.push(
          { time: 0, action: 'kelly', value: 'hero', description: 'Kelly shares wisdom' },
          { time: 500, action: 'overlay', value: 'wisdom', description: 'Wisdom styling' },
          { time: 2000, action: 'atmosphere', value: 'glow', description: 'Inspirational effect' }
        );
        break;
        
      case 'complete':
        timeline.push(
          { time: 0, action: 'kelly', value: 'hero', description: 'Kelly celebrates' },
          { time: 500, action: 'celebration', value: 'particles', description: 'Celebration effect' },
          { time: 1000, action: 'overlay', value: 'stats', description: 'Show completion stats' }
        );
        break;
    }
    
    return timeline;
  }
  
  /**
   * Start playing the choreography
   */
  play() {
    if (this.isPlaying || this.timeline.length === 0) return;
    
    this.isPlaying = true;
    this.executeCues();
    
    console.log('[Choreographer] Playing timeline');
  }
  
  /**
   * Execute cues in sequence
   */
  executeCues() {
    if (!this.isPlaying || this.currentCue >= this.timeline.length) {
      this.isPlaying = false;
      return;
    }
    
    const cue = this.timeline[this.currentCue];
    const nextCue = this.timeline[this.currentCue + 1];
    
    // Execute current cue
    this.executeCue(cue);
    
    this.currentCue++;
    
    // Schedule next cue
    if (nextCue) {
      const delay = nextCue.time - cue.time;
      setTimeout(() => this.executeCues(), delay);
    } else {
      this.isPlaying = false;
    }
  }
  
  /**
   * Execute a single visual cue
   * @param {Object} cue - Cue object
   */
  executeCue(cue) {
    console.log(`[Choreographer] Executing: ${cue.description}`);
    
    switch (cue.action) {
      case 'kelly':
        this.scene.setKellyImage(cue.value);
        break;
      case 'prop':
        if (cue.value === 'show') {
          this.scene.showProp();
        } else {
          this.scene.hideProp();
        }
        break;
      case 'overlay':
        this.scene.setOverlayEffect(cue.value);
        break;
      case 'celebration':
        this.scene.triggerCelebration(15);
        break;
      case 'atmosphere':
        // Future: atmospheric effects
        break;
    }
    
    // Dispatch cue event for other systems
    document.dispatchEvent(new CustomEvent('visual-cue', { detail: cue }));
  }
  
  /**
   * Pause choreography
   */
  pause() {
    this.isPlaying = false;
  }
  
  /**
   * Reset choreography
   */
  reset() {
    this.isPlaying = false;
    this.currentCue = 0;
    this.timeline = [];
  }
  
  /**
   * Handle speech start
   */
  onSpeechStart(detail) {
    // Kelly's mouth opens
    this.scene.kellyImg.setAttribute('data-speaking', 'true');
  }
  
  /**
   * Handle speech end
   */
  onSpeechEnd(detail) {
    // Kelly's mouth closes
    this.scene.kellyImg.setAttribute('data-speaking', 'false');
  }
  
  /**
   * Handle speech word for lip sync
   */
  onSpeechWord(detail) {
    // Future: Advanced lip sync
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN VISUAL SYSTEM CLASS
// Public API for integrating visual system into lesson player
// ═══════════════════════════════════════════════════════════════════════════

class KellyVisualSystem {
  constructor(containerSelector, options = {}) {
    this.container = typeof containerSelector === 'string' 
      ? document.querySelector(containerSelector)
      : containerSelector;
    
    if (!this.container) {
      throw new Error('[KellyVisualSystem] Container element not found');
    }
    
    this.options = {
      enableChoreography: true,
      enableAudioSync: true,
      ...options
    };
    
    // Initialize components
    this.scene = new VisualSceneManager(this.container);
    
    if (this.options.enableChoreography) {
      this.choreographer = new VisualChoreographer(this.scene, {
        audioSync: this.options.enableAudioSync
      });
    }
    
    this.currentDay = null;
    this.currentPhase = null;
    
    console.log('[KellyVisualSystem] ✨ Initialized');
  }
  
  /**
   * Load a lesson's visual assets
   * @param {number} dayNumber - Lesson day (1-365)
   * @returns {Promise}
   */
  async loadLesson(dayNumber) {
    this.currentDay = dayNumber;
    await this.scene.loadLesson(dayNumber);
    return this;
  }
  
  /**
   * Transition to a lesson phase
   * @param {string} phase - Phase name
   * @param {Object} lessonData - Optional lesson data for choreography
   * @returns {Promise}
   */
  async setPhase(phase, lessonData = null) {
    this.currentPhase = phase;
    await this.scene.setPhase(phase);
    
    if (this.choreographer && lessonData) {
      this.choreographer.loadPhaseChoreography(phase, lessonData);
      this.choreographer.play();
    }
    
    return this;
  }
  
  /**
   * Handle correct answer feedback
   */
  showCorrectFeedback() {
    this.scene.setPhase('feedback_correct');
  }
  
  /**
   * Handle incorrect answer feedback
   */
  showIncorrectFeedback() {
    this.scene.setPhase('feedback_incorrect');
  }
  
  /**
   * Trigger celebration effect
   * @param {number} intensity - Number of particles (5-50)
   */
  celebrate(intensity = 20) {
    this.scene.triggerCelebration(intensity);
  }
  
  /**
   * Get current state
   */
  getState() {
    return {
      day: this.currentDay,
      phase: this.currentPhase,
      hasAssets: this.scene.assets !== null
    };
  }
  
  /**
   * Cleanup
   */
  destroy() {
    if (this.choreographer) {
      this.choreographer.reset();
    }
    this.scene.destroy();
    console.log('[KellyVisualSystem] Destroyed');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════

window.KellyVisualSystem = KellyVisualSystem;
window.VisualSceneManager = VisualSceneManager;
window.VisualChoreographer = VisualChoreographer;
window.VISUAL_CONFIG = VISUAL_CONFIG;

console.log('[KellyVisualSystem] ✅ Module loaded - Visual storytelling for education');

