/**
 * Kelly 2D Avatar - Two Frame Pointing System
 * Simple: Kelly points UP or DOWN to guide user choices
 * 
 * Usage:
 *   const kelly = new Kelly2DAvatar('#kelly-container');
 *   kelly.pointUp();    // Kelly points to top choice
 *   kelly.pointDown();  // Kelly points to bottom choice
 */

// Kelly 2D Avatar Images - Using /kelly/poses/ directory or /images/expressions/ fallback
const KELLY_IMAGES = {
  up: '/kelly/poses/kelly_choice_left.png',     // Pointing to top/first option
  down: '/kelly/poses/kelly_choice_right.png',  // Pointing to bottom/second option
  welcome: '/kelly/poses/kelly_welcome.png',
  thinking: '/kelly/poses/kelly_hint.png',
  listening: '/kelly/poses/kelly_listening.png',
  // Fallback images (from lesson-player-v2 directory)
  fallback_curious: '/images/expressions/curious-main.jpeg',
  fallback_explaining: '/images/expressions/explaining-main.jpeg',
  fallback_wisdom: '/images/expressions/wisdom-main.jpeg'
};

// Preload images for instant switching
Object.values(KELLY_IMAGES).forEach(src => {
  const img = new Image();
  img.src = src;
});

class Kelly2DAvatar {
  constructor(container, options = {}) {
    this.container = typeof container === 'string' 
      ? document.querySelector(container) 
      : container;
    
    if (!this.container) {
      console.error('[Kelly] Container not found');
      return;
    }
    
    this.options = options;
    this.currentDirection = 'up';
    this.currentExpression = 'curious';
    this.videoMode = false;
    this.currentVideo = null;
    this.isSpeaking = false;
    
    // Don't clear container - work alongside existing image
    const existingImg = this.container.querySelector('img');
    
    // Create or reuse image element
    if (existingImg) {
      this.img = existingImg;
      console.log('[Kelly] Using existing image element');
    } else {
      // Create image element
      this.img = document.createElement('img');
      this.img.src = KELLY_IMAGES.up;
      this.img.alt = 'Kelly';
      this.img.className = 'kelly-avatar';
      this.img.style.cssText = 'width:100%;height:100%;object-fit:cover;object-position:center 15%;transition:opacity 0.3s ease;';
      this.container.appendChild(this.img);
    }
    
    // Add breathing animation CSS
    this.addBreathingAnimation();
    
    // Create video element (hidden by default)
    this.video = document.createElement('video');
    this.video.className = 'kelly-avatar-video';
    this.video.style.cssText = 'width:100%;height:100%;object-fit:cover;object-position:center 20%;display:none;position:absolute;top:0;left:0;z-index:15;';
    this.video.playsInline = true;
    this.video.muted = false;
    
    this.video.onended = () => {
      console.log('[Kelly] Video ended');
      this.showImage();
      if (this.onVideoEnd) this.onVideoEnd();
    };
    
    this.video.onerror = (e) => {
      console.warn('[Kelly] Video error, showing image', e);
      this.showImage();
    };
    
    this.container.style.position = 'relative';
    this.container.appendChild(this.video);
    console.log('[Kelly] Avatar ready (two-frame pointing system v3 + VIDEO SUPPORT)');
  }
  
  /**
   * Add breathing animation to make Kelly feel alive
   */
  addBreathingAnimation() {
    // Check if animation already exists
    if (document.getElementById('kelly-breathing-style')) return;
    
    const style = document.createElement('style');
    style.id = 'kelly-breathing-style';
    style.textContent = `
      @keyframes kelly-breathe {
        0%, 100% { transform: scale(1) translateY(0); }
        50% { transform: scale(1.008) translateY(-3px); }
      }
      
      @keyframes kelly-speaking-pulse {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.05); }
      }
      
      .kelly-avatar.breathing {
        animation: kelly-breathe 4s ease-in-out infinite;
      }
      
      .kelly-avatar.speaking {
        animation: kelly-speaking-pulse 0.5s ease-in-out infinite;
      }
    `;
    document.head.appendChild(style);
    
    // Enable breathing by default
    if (this.img) {
      this.img.classList.add('breathing');
    }
  }
  
  /**
   * Play HD video for a specific phase
   * @param {number} dayNumber - Lesson day
   * @param {string} phase - Phase (hook, q1, q2, q3, wisdom)
   * @param {string} archetype - Archetype (The Explorer, The Rebel, The Scientist)
   */
  async playPhaseVideo(dayNumber, phase, archetype = 'The Explorer') {
    const phaseMap = { 'hook': 'Hook', 'q1': 'Fact1', 'q2': 'Fact2', 'q3': 'Fact3', 'wisdom': 'Wisdom' };
    const videoPhase = phaseMap[phase] || phase;
    const archetypeFormatted = archetype.replace(/ /g, '_');
    const paddedDay = String(dayNumber).padStart(3, '0');
    
    // Use GOLDEN videos for Day 1 (1080p, 4 Mbps)
    const videoUrl = dayNumber === 1 
      ? `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/production/videos/golden/day_${paddedDay}_${videoPhase}_${archetypeFormatted}_golden.mp4`
      : `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/production/videos/day_${paddedDay}_${videoPhase.toLowerCase()}_${archetypeFormatted.toLowerCase()}.mp4`;
    
    console.log(`[Kelly] Trying video: ${videoUrl}`);
    
    try {
      // Quick check if video exists
      const checkResponse = await fetch(videoUrl, { method: 'HEAD' });
      if (!checkResponse.ok) {
        console.log(`[Kelly] Video not found, using image`);
        return false;
      }
      
      this.video.src = videoUrl;
      this.showVideo();
      await this.video.play();
      this.videoMode = true;
      console.log(`[Kelly] Playing HD video: Day ${dayNumber}, ${phase}, ${archetype}`);
      return true;
    } catch (e) {
      console.warn('[Kelly] Video playback failed:', e);
      this.showImage();
      return false;
    }
  }
  
  showVideo() {
    this.video.style.display = 'block';
    this.img.style.opacity = '0.3';
    this.videoMode = true;
  }
  
  showImage() {
    this.video.style.display = 'none';
    this.video.pause();
    this.img.style.opacity = '1';
    this.videoMode = false;
  }
  
  stopVideo() {
    this.video.pause();
    this.video.currentTime = 0;
    this.showImage();
  }
  
  /**
   * Kelly points UP (to top choice)
   */
  pointUp() {
    if (this.img) {
      this.img.src = KELLY_IMAGES.up;
      this.currentDirection = 'up';
      console.log('[Kelly] Pointing UP');
    }
  }
  
  /**
   * Kelly points DOWN (to bottom choice)
   */
  pointDown() {
    if (this.img) {
      this.img.src = KELLY_IMAGES.down;
      this.currentDirection = 'down';
      console.log('[Kelly] Pointing DOWN');
    }
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
  
  /**
   * Set expression - maps expression names to pointing directions
   * For visual variety during lesson phases
   */
  setExpression(expression) {
    this.currentExpression = expression;
    
    // Map expressions to pointing directions for visual variety
    const downExpressions = ['explaining', 'wisdom', 'listening'];
    
    if (downExpressions.includes(expression)) {
      this.pointDown();
    } else {
      this.pointUp();
    }
    console.log(`[Kelly] setExpression('${expression}') -> pointing ${this.currentDirection}`);
  }
  
  /**
   * Set speaking state - adds visual feedback when Kelly is talking
   * @param {boolean} speaking - Whether Kelly is currently speaking
   */
  setSpeaking(speaking) {
    this.isSpeaking = speaking;
    
    if (this.img) {
      if (speaking) {
        this.img.classList.add('speaking');
        this.img.classList.remove('breathing');
      } else {
        this.img.classList.remove('speaking');
        this.img.classList.add('breathing');
      }
    }
    
    console.log(`[Kelly] setSpeaking(${speaking})`);
  }
  
  setPhase(phase, choice = null) {
    // Map phases to pointing
    if (phase === 'q1' || phase === 'q2' || phase === 'q3' || phase === 'question') {
      if (choice === 'b' || choice === 'B') {
        this.pointDown();
      } else {
        this.pointUp();
      }
    } else if (phase === 'wisdom' || phase === 'complete') {
      this.pointDown();
    } else {
      this.pointUp();
    }
  }
  
  /**
   * Load phase-specific visual (Kelly in educational context)
   * @param {number} dayNumber - Lesson day (1-365)
   * @param {string} phase - Phase name ('hook', 'q1', 'q2', 'q3', 'wisdom')
   */
  loadPhaseVisual(dayNumber, phase) {
    const paddedDay = String(dayNumber).padStart(3, '0');
    const phaseMap = {
      'welcome': 'hook',
      'hook': 'hook',
      'q1': 'q1',
      'q2': 'q2', 
      'q3': 'q3',
      'question': 'q1',
      'wisdom': 'wisdom',
      'complete': 'wisdom'
    };
    
    const phaseFile = phaseMap[phase] || 'hook';
    const phasePath = `/kelly/phases/${paddedDay}/${phaseFile}.png`;
    
    // Try to load phase visual, fall back to default pose on error
    const testImg = new Image();
    testImg.onload = () => {
      if (this.img) {
        this.img.src = phasePath;
        console.log(`[Kelly] Phase visual loaded: day ${dayNumber}, ${phaseFile}`);
      }
    };
    testImg.onerror = () => {
      console.log(`[Kelly] Phase visual not found for day ${dayNumber}/${phaseFile}, using default pose`);
      // Keep current pose image
    };
    testImg.src = phasePath;
  }
  
  getExpression() {
    return this.currentExpression || (this.currentDirection === 'up' ? 'curious' : 'explaining');
  }
  
  destroy() {
    if (this.container) {
      // Remove video but keep image (it may be the original)
      if (this.video && this.video.parentNode) {
        this.video.parentNode.removeChild(this.video);
      }
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

console.log('[Kelly] 2D Avatar module loaded (lesson-player-v2 integrated version)');







