/**
 * Kelly 2D Avatar - Two Frame Pointing System
 * Simple: Kelly points UP or DOWN to guide user choices
 * 
 * Usage:
 *   const kelly = new Kelly2DAvatar('#kelly-container');
 *   kelly.pointUp();    // Kelly points to top choice
 *   kelly.pointDown();  // Kelly points to bottom choice
 */

const KELLY_IMAGES = {
  up: '/assets/kelly/choices/point-up.jpeg',
  down: '/assets/kelly/choices/point-down.jpeg'
};

// Preload both images immediately for instant switching
Object.values(KELLY_IMAGES).forEach(src => {
  const img = new Image();
  img.src = src;
});

class Kelly2DAvatar {
  constructor(container) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    if (!this.container) {
      console.error('[Kelly] Container not found');
      return;
    }
    
    // Clear container
    this.container.innerHTML = '';
    
    // Create image element
    this.img = document.createElement('img');
    this.img.src = KELLY_IMAGES.up;
    this.img.alt = 'Kelly';
    this.img.className = 'kelly-avatar';
    this.img.style.cssText = 'width:100%;height:100%;object-fit:cover;object-position:center 15%;transition:opacity 0.3s ease;';
    
    this.container.appendChild(this.img);
    console.log('[Kelly] Avatar ready (two-frame pointing system)');
  }
  
  /**
   * Kelly points UP (to top choice)
   */
  pointUp() {
    this.img.src = KELLY_IMAGES.up;
    console.log('[Kelly] Pointing UP');
  }
  
  /**
   * Kelly points DOWN (to bottom choice)
   */
  pointDown() {
    this.img.src = KELLY_IMAGES.down;
    console.log('[Kelly] Pointing DOWN');
  }
  
  /**
   * Set direction by string
   * @param {string} dir - 'up', 'down', 'top', 'bottom', 'first', 'second'
   */
  setDirection(dir) {
    const normalized = (dir || '').toLowerCase();
    if (normalized === 'down' || normalized === 'bottom' || normalized === 'second' || normalized === 'b') {
      this.pointDown();
    } else {
      this.pointUp();
    }
  }
  
  // Backward compatibility with old avatar system
  setExpression(expr) {
    console.log('[Kelly] setExpression called (defaulting to pointUp)');
    this.pointUp();
  }
  
  setSpeaking(speaking) {
    // No-op for compatibility
  }
  
  setPhase(phase, choice = null) {
    // Map phases to pointing
    if (phase === 'q1' || phase === 'q2' || phase === 'q3' || phase === 'question') {
      if (choice === 'b' || choice === 'B') {
        this.pointDown();
      } else {
        this.pointUp();
      }
    } else {
      this.pointUp();
    }
  }
  
  getExpression() {
    return this.img.src.includes('up') ? 'up' : 'down';
  }
  
  destroy() {
    if (this.container) {
      this.container.innerHTML = '';
    }
    console.log('[Kelly] Avatar destroyed');
  }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Kelly2DAvatar;
}

// Make available globally
window.Kelly2DAvatar = Kelly2DAvatar;

console.log('[Kelly] 2D Avatar module loaded (simple two-frame system)');
