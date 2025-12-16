/**
 * Kelly Visual System v1.1
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Manages immersive backgrounds, props, and visual choreography for Kelly lessons.
 * Handles phase transitions with appropriate visual theming.
 * Now includes safe-zone metadata for caption positioning and infographic panels.
 * 
 * @module KellyVisualSystem
 */

// Debug mode
const __VISUAL_DEBUG = (
  (typeof location !== 'undefined' && location.search.includes('debug')) ||
  (typeof localStorage !== 'undefined' && localStorage.getItem('kellyDebug') === '1')
);

class KellyVisualSystem {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.options = {
      enableChoreography: options.enableChoreography !== false,
      enableAudioSync: options.enableAudioSync !== false,
      transitionDuration: options.transitionDuration || 500,
      enableSafeZones: options.enableSafeZones !== false,
      ...options
    };
    
    this.currentPhase = null;
    this.currentLesson = null;
    this.isLoading = false;
    
    // Safe zone and visual plan data
    this.safeZones = null;
    this.visualPlan = null;
    this.currentSafeZone = null;
    
    // Visual theme definitions per phase type
    this.phaseThemes = {
      welcome: {
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        ambiance: 'warm',
        particles: false
      },
      question: {
        background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)',
        ambiance: 'focused',
        particles: false
      },
      q1: {
        background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)',
        ambiance: 'focused',
        particles: false
      },
      q2: {
        background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #1a3a5c 100%)',
        ambiance: 'engaged',
        particles: false
      },
      q3: {
        background: 'linear-gradient(135deg, #0f0f1a 0%, #162447 50%, #1f4068 100%)',
        ambiance: 'building',
        particles: false
      },
      wisdom: {
        background: 'linear-gradient(135deg, #1a1a2e 0%, #1f3a5f 50%, #3b82f6 100%)',
        ambiance: 'enlightened',
        particles: true
      }
    };
    
    // Initialize visual layer
    this._initVisualLayer();
    
    if (__VISUAL_DEBUG) console.log('[KellyVisualSystem] ✅ Initialized with choreography:', this.options.enableChoreography);
  }
  
  /**
   * Initialize the visual background layer
   */
  _initVisualLayer() {
    if (!this.container) return;
    
    // Create background layer if not exists
    let bgLayer = this.container.querySelector('.kelly-visual-bg');
    if (!bgLayer) {
      bgLayer = document.createElement('div');
      bgLayer.className = 'kelly-visual-bg';
      bgLayer.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
        transition: background ${this.options.transitionDuration}ms ease;
        pointer-events: none;
      `;
      this.container.style.position = 'relative';
      this.container.insertBefore(bgLayer, this.container.firstChild);
    }
    this.bgLayer = bgLayer;
    
    // Create particle layer for wisdom phase
    let particleLayer = this.container.querySelector('.kelly-particles');
    if (!particleLayer) {
      particleLayer = document.createElement('div');
      particleLayer.className = 'kelly-particles';
      particleLayer.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 0;
        pointer-events: none;
        opacity: 0;
        transition: opacity ${this.options.transitionDuration}ms ease;
      `;
      this.container.appendChild(particleLayer);
    }
    this.particleLayer = particleLayer;
  }
  
  /**
   * Load visual assets for a specific lesson day
   * @param {number} dayNumber - The lesson day number
   * @returns {Promise<void>}
   */
  async loadLesson(dayNumber) {
    if (this.isLoading) return;
    this.isLoading = true;
    
    try {
      this.currentLesson = dayNumber;
      const paddedDay = String(dayNumber).padStart(3, '0');
      
      // Load visual plan (v2 format from content/visual-plans/)
      await this._loadVisualPlan(dayNumber);
      
      // Load safe zones for video-based lessons
      await this._loadSafeZones(dayNumber);
      
      // Apply initial welcome theme
      await this.setPhase('welcome', { dayNumber });
      
    } finally {
      this.isLoading = false;
    }
  }
  
  /**
   * Load visual plan for the lesson
   * @param {number} dayNumber - The lesson day number
   * @private
   */
  async _loadVisualPlan(dayNumber) {
    const paddedDay = String(dayNumber).padStart(3, '0');
    
    // Try v2 format first (content/visual-plans/)
    const visualPlanUrls = [
      `/content/visual-plans/day-${paddedDay}-visual-plan-v2.json`,
      `/content/days/day-${paddedDay}/visual-plan.json`,
      `/kelly/phases/${paddedDay}/visual-plan.json`
    ];
    
    for (const url of visualPlanUrls) {
      try {
        const response = await fetch(url);
        if (response.ok) {
          this.visualPlan = await response.json();
          this.lessonVisuals = this.visualPlan;
          if (__VISUAL_DEBUG) console.log(`[KellyVisualSystem] ✅ Loaded visual plan from ${url}`);
          return;
        }
      } catch (e) {
        // Continue to next URL
      }
    }
    
    this.visualPlan = null;
    this.lessonVisuals = null;
    if (__VISUAL_DEBUG) console.log(`[KellyVisualSystem] No visual plan found for day ${dayNumber}`);
  }
  
  /**
   * Load safe zones JSON for video positioning
   * @param {number} dayNumber - The lesson day number
   * @private
   */
  async _loadSafeZones(dayNumber) {
    const paddedDay = String(dayNumber).padStart(3, '0');
    const safeZoneUrl = `/kelly/videos/${paddedDay}/welcome-safe-zones.json`;
    
    try {
      const response = await fetch(safeZoneUrl);
      if (response.ok) {
        this.safeZones = await response.json();
        if (__VISUAL_DEBUG) console.log(`[KellyVisualSystem] ✅ Loaded safe zones for day ${dayNumber}`);
        
        // Set initial safe zone (first segment)
        if (this.safeZones?.safe_zones?.length > 0) {
          this.currentSafeZone = this.safeZones.safe_zones[0];
        }
      }
    } catch (e) {
      this.safeZones = null;
      if (__VISUAL_DEBUG) console.log(`[KellyVisualSystem] No safe zones found for day ${dayNumber}`);
    }
  }
  
  /**
   * Get the current safe zone based on video time
   * @param {number} videoTime - Current video playback time in seconds
   * @returns {object|null} Safe zone object with position data
   */
  getSafeZoneForTime(videoTime) {
    if (!this.safeZones?.safe_zones) return null;
    
    for (const zone of this.safeZones.safe_zones) {
      if (videoTime >= zone.time_start && videoTime < zone.time_end) {
        this.currentSafeZone = zone;
        return zone;
      }
    }
    
    return this.currentSafeZone;
  }
  
  /**
   * Get optimal caption position based on current safe zone
   * @returns {object} Position object with CSS properties
   */
  getCaptionPosition() {
    const zone = this.currentSafeZone;
    if (!zone?.safe_zones) {
      // Default: bottom center
      return {
        position: 'bottom-center',
        top: 'auto',
        bottom: 'calc(120px + var(--safe-bottom, 0px))',
        left: '50%',
        transform: 'translateX(-50%)',
        maxWidth: '60%'
      };
    }
    
    // Find the best safe zone for captions (prefer bottom zones)
    const bottomZones = zone.safe_zones.filter(z => z.name.includes('bottom'));
    const bestZone = bottomZones.length > 0 
      ? bottomZones.reduce((a, b) => a.score > b.score ? a : b)
      : zone.safe_zones.reduce((a, b) => a.score > b.score ? a : b);
    
    // Convert normalized coordinates to CSS
    return {
      position: bestZone.name,
      top: bestZone.name.includes('top') ? `${bestZone.y * 100}%` : 'auto',
      bottom: bestZone.name.includes('bottom') ? `${(1 - bestZone.y - bestZone.height) * 100}%` : 'auto',
      left: `${bestZone.x * 100}%`,
      width: `${bestZone.width * 100}%`,
      maxWidth: `${bestZone.width * 100}%`
    };
  }
  
  /**
   * Get optimal infographic panel position based on current safe zone
   * @returns {object} Position object with CSS properties for infographic panel
   */
  getInfographicPosition() {
    const zone = this.currentSafeZone;
    if (!zone?.safe_zones) {
      // Default: top right
      return {
        position: 'top-right',
        top: '80px',
        right: '160px',
        bottom: 'auto',
        left: 'auto',
        width: '320px',
        maxWidth: '35%'
      };
    }
    
    // Find the best safe zone for infographic (prefer top-right, avoid Kelly's face)
    const preferredOrder = ['top-right', 'right-mid', 'top-left', 'left-mid'];
    let bestZone = null;
    
    for (const preferred of preferredOrder) {
      const match = zone.safe_zones.find(z => z.name === preferred);
      if (match && match.score >= 0.8) {
        bestZone = match;
        break;
      }
    }
    
    if (!bestZone) {
      bestZone = zone.safe_zones.reduce((a, b) => a.score > b.score ? a : b);
    }
    
    // Convert normalized coordinates to CSS
    const isRight = bestZone.name.includes('right');
    const isTop = bestZone.name.includes('top');
    
    return {
      position: bestZone.name,
      top: isTop ? `${bestZone.y * 100}%` : 'auto',
      bottom: !isTop ? `${(1 - bestZone.y - bestZone.height) * 100}%` : 'auto',
      right: isRight ? `${(1 - bestZone.x - bestZone.width) * 100}%` : 'auto',
      left: !isRight ? `${bestZone.x * 100}%` : 'auto',
      width: `${bestZone.width * 100}%`,
      maxWidth: `${Math.min(bestZone.width * 100, 40)}%`
    };
  }
  
  /**
   * Get visual plan data for a specific phase
   * @param {string} phase - Phase name (hook, q1, q2, q3, wisdom)
   * @returns {object|null} Phase visual data
   */
  getPhaseVisual(phase) {
    if (!this.visualPlan?.phases) return null;
    
    const normalizedPhase = phase.toLowerCase().replace(/[^a-z0-9]/g, '');
    
    // Visual plan v2 uses array format
    if (Array.isArray(this.visualPlan.phases)) {
      return this.visualPlan.phases.find(p => 
        p.phase?.toLowerCase() === normalizedPhase ||
        p.phase?.toLowerCase() === phase.toLowerCase()
      );
    }
    
    // Legacy object format
    return this.visualPlan.phases[normalizedPhase];
  }
  
  /**
   * Get infographic prompt for current phase (for generation)
   * @param {string} phase - Phase name
   * @returns {string|null} Image generation prompt
   */
  getInfographicPrompt(phase) {
    const phaseVisual = this.getPhaseVisual(phase);
    return phaseVisual?.visual?.prompt || null;
  }
  
  /**
   * Set the visual phase with appropriate theming
   * @param {string} phase - Phase name (welcome, q1, q2, q3, wisdom)
   * @param {object} lesson - Current lesson object
   * @returns {Promise<void>}
   */
  async setPhase(phase, lesson = {}) {
    const normalizedPhase = phase.toLowerCase().replace(/[^a-z0-9]/g, '');
    const theme = this.phaseThemes[normalizedPhase] || this.phaseThemes.question;
    
    // Apply background transition
    if (this.bgLayer) {
      this.bgLayer.style.background = theme.background;
    }
    
    // Handle particles for wisdom phase
    if (this.particleLayer) {
      this.particleLayer.style.opacity = theme.particles ? '1' : '0';
      if (theme.particles) {
        this._startParticles();
      }
    }
    
    this.currentPhase = normalizedPhase;
    
    // If we have lesson-specific visuals, apply them
    if (this.lessonVisuals && this.lessonVisuals.phases) {
      const phaseVisual = this.lessonVisuals.phases[normalizedPhase];
      if (phaseVisual && phaseVisual.backgroundImage) {
        this.bgLayer.style.backgroundImage = `url(${phaseVisual.backgroundImage})`;
        this.bgLayer.style.backgroundSize = 'cover';
        this.bgLayer.style.backgroundPosition = 'center';
      }
    }
    
    if (__VISUAL_DEBUG) console.log(`[KellyVisualSystem] Phase set to: ${normalizedPhase}`);
    return Promise.resolve();
  }
  
  /**
   * Show positive feedback animation for correct answers
   */
  showCorrectFeedback() {
    if (!this.options.enableChoreography) return;
    
    // Create celebratory flash effect
    const flash = document.createElement('div');
    flash.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at center, rgba(59, 130, 246, 0.3), transparent 70%);
      pointer-events: none;
      z-index: 9999;
      animation: correctFlash 0.8s ease-out forwards;
    `;
    document.body.appendChild(flash);
    
    setTimeout(() => flash.remove(), 800);
    
    if (__VISUAL_DEBUG) console.log('[KellyVisualSystem] ✨ Correct feedback shown');
  }
  
  /**
   * Start particle animation for wisdom phase
   */
  _startParticles() {
    if (!this.particleLayer || this.particleLayer.children.length > 0) return;
    
    // Create simple floating particles
    for (let i = 0; i < 20; i++) {
      const particle = document.createElement('div');
      particle.className = 'wisdom-particle';
      particle.style.cssText = `
        position: absolute;
        width: ${4 + Math.random() * 4}px;
        height: ${4 + Math.random() * 4}px;
        background: rgba(59, 130, 246, ${0.3 + Math.random() * 0.4});
        border-radius: 50%;
        left: ${Math.random() * 100}%;
        top: ${Math.random() * 100}%;
        animation: floatParticle ${5 + Math.random() * 10}s ease-in-out infinite;
        animation-delay: ${Math.random() * 5}s;
      `;
      this.particleLayer.appendChild(particle);
    }
  }
  
  /**
   * Clean up resources
   */
  destroy() {
    if (this.bgLayer) this.bgLayer.remove();
    if (this.particleLayer) this.particleLayer.remove();
    if (__VISUAL_DEBUG) console.log('[KellyVisualSystem] Destroyed');
  }
}

// Add required CSS animations
const visualSystemStyles = document.createElement('style');
visualSystemStyles.textContent = `
  @keyframes correctFlash {
    0% {
      opacity: 0;
      transform: scale(0.8);
    }
    20% {
      opacity: 1;
      transform: scale(1);
    }
    100% {
      opacity: 0;
      transform: scale(1.2);
    }
  }
  
  @keyframes floatParticle {
    0%, 100% {
      transform: translateY(0) translateX(0);
      opacity: 0.3;
    }
    25% {
      transform: translateY(-20px) translateX(10px);
      opacity: 0.6;
    }
    50% {
      transform: translateY(-10px) translateX(-5px);
      opacity: 0.4;
    }
    75% {
      transform: translateY(-30px) translateX(5px);
      opacity: 0.5;
    }
  }
`;
document.head.appendChild(visualSystemStyles);

// Export globally
window.KellyVisualSystem = KellyVisualSystem;

if (__VISUAL_DEBUG) console.log('[KellyVisualSystem] ✅ Module loaded');
