/**
 * Kelly Visual System v1.0
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Manages immersive backgrounds, props, and visual choreography for Kelly lessons.
 * Handles phase transitions with appropriate visual theming.
 * 
 * @module KellyVisualSystem
 */

class KellyVisualSystem {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    this.options = {
      enableChoreography: options.enableChoreography !== false,
      enableAudioSync: options.enableAudioSync !== false,
      transitionDuration: options.transitionDuration || 500,
      ...options
    };
    
    this.currentPhase = null;
    this.currentLesson = null;
    this.isLoading = false;
    
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
    
    console.log('[KellyVisualSystem] ✅ Initialized with choreography:', this.options.enableChoreography);
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
      
      // Check if we have pre-generated visual assets for this day
      // (Phase images, backgrounds from infographic pipeline)
      const visualPlanUrl = `/content/days/day-${String(dayNumber).padStart(3, '0')}/visual-plan.json`;
      
      try {
        const response = await fetch(visualPlanUrl);
        if (response.ok) {
          const visualPlan = await response.json();
          this.lessonVisuals = visualPlan;
          console.log(`[KellyVisualSystem] Loaded visual plan for day ${dayNumber}`);
        }
      } catch (e) {
        // Visual plan not available - use default theming
        this.lessonVisuals = null;
      }
      
      // Apply initial welcome theme
      await this.setPhase('welcome', { dayNumber });
      
    } finally {
      this.isLoading = false;
    }
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
    
    console.log(`[KellyVisualSystem] Phase set to: ${normalizedPhase}`);
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
    
    console.log('[KellyVisualSystem] ✨ Correct feedback shown');
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
    console.log('[KellyVisualSystem] Destroyed');
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

console.log('[KellyVisualSystem] ✅ Module loaded');
