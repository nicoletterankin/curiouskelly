/**
 * Kelly 2D Avatar Player
 * PNG-based avatar with CSS animations for the learning experience
 *
 * Features:
 * - 5 expressions: curious, explaining, listening, wisdom, celebrating
 * - Smooth crossfade transitions
 * - Breathing animation
 * - Speaking indicator
 * - Phase-based expression mapping
 */

class Kelly2DAvatar {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      imageSet: options.imageSet || 'directors-chair',
      basePath: options.basePath || '/images/kelly/',
      transitionDuration: options.transitionDuration || 400,
      enableBreathing: options.enableBreathing !== false,
      preload: options.preload !== false,
      ...options
    };

    this.state = {
      expression: 'curious',
      isSpeaking: false,
      isTransitioning: false
    };

    this.expressions = ['curious', 'explaining', 'listening', 'wisdom', 'celebrating'];
    this.imageCache = new Map();
    this.avatarElement = null;

    this.init();
  }

  init() {
    this.createDOM();
    if (this.options.preload) {
      this.preloadAllImages();
    }
    console.log('[Kelly2D] Initialized');
  }

  createDOM() {
    // Clear container
    this.container.innerHTML = '';

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'kelly-2d-wrapper';
    wrapper.innerHTML = `
            <img 
                class="kelly-2d-avatar ${this.options.enableBreathing ? 'breathing' : ''}"
                src="${this.getImagePath('curious')}"
                alt="Kelly"
                draggable="false"
            />
            <div class="kelly-speaking-indicator"></div>
        `;

    this.container.appendChild(wrapper);
    this.avatarElement = wrapper.querySelector('.kelly-2d-avatar');
    this.speakingIndicator = wrapper.querySelector('.kelly-speaking-indicator');

    // Add styles if not already present
    this.injectStyles();
  }

  injectStyles() {
    if (document.getElementById('kelly-2d-styles')) return;

    const styles = document.createElement('style');
    styles.id = 'kelly-2d-styles';
    styles.textContent = `
            .kelly-2d-wrapper {
                position: relative;
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .kelly-2d-avatar {
                max-width: 100%;
                max-height: 100%;
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center 15%;
                transition: opacity ${this.options.transitionDuration}ms ease,
                            transform ${this.options.transitionDuration}ms ease;
            }
            
            /* Mobile: Tighter crop on face */
            @media (max-width: 480px) {
                .kelly-2d-avatar {
                    object-position: center 10%;
                    transform: scale(1.1);
                }
            }
            
            /* Desktop: Show more */
            @media (min-width: 769px) {
                .kelly-2d-avatar {
                    object-fit: contain;
                    object-position: center center;
                    transform: scale(1);
                }
            }
            
            /* Breathing animation */
            .kelly-2d-avatar.breathing {
                animation: kelly-breathe 4s ease-in-out infinite;
            }
            
            @keyframes kelly-breathe {
                0%, 100% { transform: scale(1) translateY(0); }
                50% { transform: scale(1.008) translateY(-3px); }
            }
            
            /* Transition state */
            .kelly-2d-avatar.transitioning {
                opacity: 0.7;
            }
            
            /* Speaking state */
            .kelly-2d-avatar.speaking {
                animation: kelly-breathe 2s ease-in-out infinite,
                           kelly-speaking-glow 1s ease-in-out infinite;
            }
            
            @keyframes kelly-speaking-glow {
                0%, 100% { filter: brightness(1); }
                50% { filter: brightness(1.03); }
            }
            
            /* Speaking indicator ring */
            .kelly-speaking-indicator {
                position: absolute;
                bottom: 30%;
                left: 50%;
                transform: translateX(-50%);
                width: 60px;
                height: 60px;
                border-radius: 50%;
                border: 3px solid rgba(59, 130, 246, 0);
                opacity: 0;
                transition: opacity 0.3s ease;
                pointer-events: none;
            }
            
            .kelly-speaking-indicator.active {
                opacity: 1;
                border-color: rgba(59, 130, 246, 0.5);
                animation: speaking-ring 1.5s ease-in-out infinite;
            }
            
            @keyframes speaking-ring {
                0%, 100% { 
                    transform: translateX(-50%) scale(1);
                    border-color: rgba(59, 130, 246, 0.5);
                }
                50% { 
                    transform: translateX(-50%) scale(1.1);
                    border-color: rgba(59, 130, 246, 0.8);
                }
            }
            
            /* Celebration effect */
            .kelly-2d-avatar.celebrating {
                animation: kelly-celebrate 0.6s ease;
            }
            
            @keyframes kelly-celebrate {
                0% { transform: scale(1); }
                30% { transform: scale(1.03) translateY(-8px); }
                60% { transform: scale(0.98) translateY(2px); }
                100% { transform: scale(1) translateY(0); }
            }
            
            /* Reduced motion */
            @media (prefers-reduced-motion: reduce) {
                .kelly-2d-avatar,
                .kelly-2d-avatar.breathing,
                .kelly-2d-avatar.speaking,
                .kelly-speaking-indicator {
                    animation: none !important;
                }
            }
        `;
    document.head.appendChild(styles);
  }

  getImagePath(expression) {
    return `${this.options.basePath}kelly-${this.options.imageSet}-${expression}.png`;
  }

  async preloadAllImages() {
    console.log('[Kelly2D] Preloading images...');

    const promises = this.expressions.map((expr) => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          this.imageCache.set(expr, img.src);
          resolve();
        };
        img.onerror = () => {
          console.warn(`[Kelly2D] Failed to preload: ${expr}`);
          resolve();
        };
        img.src = this.getImagePath(expr);
      });
    });

    await Promise.all(promises);
    console.log('[Kelly2D] Preload complete');
  }

  /**
   * Set Kelly's expression
   */
  async setExpression(expression) {
    if (!this.expressions.includes(expression)) {
      console.warn(`[Kelly2D] Unknown expression: ${expression}`);
      return;
    }

    if (this.state.expression === expression) return;
    if (this.state.isTransitioning) return;

    console.log(`[Kelly2D] Expression: ${this.state.expression} → ${expression}`);

    this.state.isTransitioning = true;
    this.avatarElement.classList.add('transitioning');

    // Wait for fade out
    await this.wait(this.options.transitionDuration / 2);

    // Switch image
    this.avatarElement.src = this.getImagePath(expression);
    this.state.expression = expression;

    // Wait for image load
    await this.waitForImageLoad(this.avatarElement);

    // Fade back in
    this.avatarElement.classList.remove('transitioning');

    await this.wait(this.options.transitionDuration / 2);
    this.state.isTransitioning = false;

    // Dispatch event
    this.dispatchEvent('expression-changed', { expression });
  }

  /**
   * Set speaking state
   */
  setSpeaking(speaking) {
    this.state.isSpeaking = speaking;

    if (speaking) {
      this.avatarElement.classList.add('speaking');
      this.speakingIndicator?.classList.add('active');
    } else {
      this.avatarElement.classList.remove('speaking');
      this.speakingIndicator?.classList.remove('active');
    }

    this.dispatchEvent('speaking-changed', { speaking });
  }

  /**
   * Play celebration animation
   */
  celebrate() {
    this.avatarElement.classList.remove('celebrating');
    // Force reflow
    void this.avatarElement.offsetWidth;
    this.avatarElement.classList.add('celebrating');

    setTimeout(() => {
      this.avatarElement.classList.remove('celebrating');
    }, 600);
  }

  /**
   * Set expression based on lesson phase and choice
   */
  setPhase(phase, choice = null) {
    let expression = 'curious';

    switch (phase) {
      case 'welcome':
        expression = 'curious';
        break;
      case 'q1':
      case 'q2':
        if (choice === 'a') expression = 'explaining';
        else if (choice === 'b') expression = 'celebrating';
        else if (choice === 'c') expression = 'wisdom';
        else expression = 'curious';
        break;
      case 'q3':
        if (choice === 'a') expression = 'explaining';
        else if (choice === 'b') expression = 'celebrating';
        else if (choice === 'c') expression = 'wisdom';
        else expression = 'listening';
        break;
      case 'wisdom':
        expression = 'wisdom';
        break;
      case 'complete':
        expression = 'celebrating';
        this.celebrate();
        break;
      default:
        expression = 'curious';
    }

    this.setExpression(expression);

    // Celebrate on positive choices
    if (choice === 'b' || phase === 'complete') {
      setTimeout(() => this.celebrate(), 200);
    }
  }

  /**
   * Get current expression
   */
  getExpression() {
    return this.state.expression;
  }

  /**
   * Check if speaking
   */
  isSpeaking() {
    return this.state.isSpeaking;
  }

  // Utility methods
  wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  waitForImageLoad(img) {
    return new Promise((resolve) => {
      if (img.complete && img.naturalHeight !== 0) {
        resolve();
      } else {
        img.onload = resolve;
        img.onerror = resolve;
      }
    });
  }

  dispatchEvent(name, detail) {
    document.dispatchEvent(new CustomEvent(`kelly-2d-${name}`, { detail }));
  }

  destroy() {
    this.container.innerHTML = '';
    this.imageCache.clear();
    console.log('[Kelly2D] Destroyed');
  }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Kelly2DAvatar;
}

// Make available globally
window.Kelly2DAvatar = Kelly2DAvatar;

