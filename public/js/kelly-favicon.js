/**
 * Kelly Living Favicon System
 * 
 * Kelly is not just a logo—she's a living presence in your browser.
 * This system changes Kelly's favicon state based on app activity,
 * creating an emotional connection with learners.
 * 
 * States:
 * - curious (default): Kelly looking up-right, chin on hand
 * - attentive: Kelly looking directly at viewer (notification)
 * - celebrating: Kelly with big smile (achievement)
 * - thinking: Kelly processing (loading)
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 */

class KellyFavicon {
  constructor(options = {}) {
    this.options = {
      basePath: options.basePath || '/icons',
      defaultState: options.defaultState || 'curious',
      enableTitleFlash: options.enableTitleFlash !== false,
      flashDuration: options.flashDuration || 10000,
      celebrationDuration: options.celebrationDuration || 3000,
      ...options
    };

    // State definitions - paths to favicon images
    this.states = {
      curious: `${this.options.basePath}/icon-192.png`,
      attentive: `${this.options.basePath}/icon-192.png`, // TODO: Create attentive variant
      celebrating: `${this.options.basePath}/icon-192.png`, // TODO: Create celebrating variant
      thinking: `${this.options.basePath}/icon-192.png` // TODO: Create thinking variant
    };

    // State meanings (for accessibility and logging)
    this.stateMessages = {
      curious: "What shall we wonder about today?",
      attentive: "I have something exciting to share!",
      celebrating: "You did it! Another day of growth!",
      thinking: "Let me find the perfect lesson..."
    };

    this.currentState = this.options.defaultState;
    this.originalTitle = document.title;
    this.flashInterval = null;
    
    // Find or create the favicon link element
    this.link = this._getOrCreateFaviconLink();
    
    // Initialize
    this._init();
  }

  /**
   * Initialize the favicon system
   */
  _init() {
    // Set initial state
    this.setState(this.options.defaultState);
    
    // Listen for visibility changes to manage resources
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this._onHidden();
      } else {
        this._onVisible();
      }
    });

    // Listen for custom events from the app
    window.addEventListener('kelly:lesson-ready', () => this.notifyLesson());
    window.addEventListener('kelly:lesson-complete', () => this.celebrate());
    window.addEventListener('kelly:loading', () => this.thinking());
    window.addEventListener('kelly:idle', () => this.idle());

    console.log('✨ Kelly Favicon System initialized');
  }

  /**
   * Get existing favicon link or create one
   */
  _getOrCreateFaviconLink() {
    let link = document.querySelector("link[rel~='icon'][sizes='192x192']") ||
               document.querySelector("link[rel~='icon']");
    
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      link.type = 'image/png';
      document.head.appendChild(link);
    }
    
    return link;
  }

  /**
   * Set Kelly's favicon state
   * @param {string} state - One of: curious, attentive, celebrating, thinking
   */
  setState(state) {
    if (!this.states[state]) {
      console.warn(`KellyFavicon: Unknown state "${state}"`);
      return;
    }

    if (state !== this.currentState) {
      this.currentState = state;
      this.link.href = this.states[state] + '?v=' + Date.now(); // Cache bust
      
      // Log state change for debugging
      console.log(`✨ Kelly is now: ${state} - "${this.stateMessages[state]}"`);
    }
  }

  /**
   * Notify user that a new lesson is ready
   * Kelly looks directly at the viewer
   */
  notifyLesson() {
    this.setState('attentive');
    
    if (this.options.enableTitleFlash) {
      this.flashTitle('✨ New Lesson Ready!');
    }
  }

  /**
   * Celebrate a completed lesson or achievement
   * Kelly shows joy
   */
  celebrate() {
    this.setState('celebrating');
    
    if (this.options.enableTitleFlash) {
      this.flashTitle('🎉 Great job!', 3000);
    }
    
    // Return to curious after celebration
    setTimeout(() => this.setState('curious'), this.options.celebrationDuration);
  }

  /**
   * Show thinking state during loading
   */
  thinking() {
    this.setState('thinking');
  }

  /**
   * Return to default curious state
   */
  idle() {
    this.setState('curious');
    this._stopTitleFlash();
  }

  /**
   * Flash the document title to get attention
   * @param {string} message - Message to flash
   * @param {number} duration - How long to flash (ms)
   */
  flashTitle(message, duration = this.options.flashDuration) {
    this._stopTitleFlash();
    
    let showMessage = true;
    this.flashInterval = setInterval(() => {
      document.title = showMessage ? message : this.originalTitle;
      showMessage = !showMessage;
    }, 1000);

    setTimeout(() => this._stopTitleFlash(), duration);
  }

  /**
   * Stop title flashing
   */
  _stopTitleFlash() {
    if (this.flashInterval) {
      clearInterval(this.flashInterval);
      this.flashInterval = null;
      document.title = this.originalTitle;
    }
  }

  /**
   * Called when tab becomes hidden
   */
  _onHidden() {
    // Could pause animations here
  }

  /**
   * Called when tab becomes visible
   */
  _onVisible() {
    // Restore state
    this.setState(this.currentState);
  }

  /**
   * Update the original title (call after programmatic title changes)
   */
  updateOriginalTitle() {
    this.originalTitle = document.title;
  }

  /**
   * Get current state
   */
  getState() {
    return this.currentState;
  }

  /**
   * Get state message
   */
  getMessage() {
    return this.stateMessages[this.currentState];
  }
}

/**
 * Calendar It - Quick capture for curious thoughts
 * Integrates with Kelly's favicon to reinforce the curiosity loop
 */
class KellyCalendarIt {
  constructor(kellyFavicon) {
    this.favicon = kellyFavicon;
    this.pendingThoughts = [];
    
    // Listen for calendar-it requests
    window.addEventListener('kelly:calendar-it', (e) => this.capture(e.detail));
  }

  /**
   * Capture a curious thought for later learning
   * @param {object} thought - { text: string, source?: string }
   */
  capture(thought) {
    this.pendingThoughts.push({
      ...thought,
      timestamp: Date.now(),
      scheduled: false
    });

    // Kelly acknowledges the thought
    this.favicon.thinking();
    
    console.log(`✨ Kelly calendared: "${thought.text}"`);
    
    // Return to curious after brief acknowledgment
    setTimeout(() => this.favicon.idle(), 1500);
    
    // Dispatch event for app to handle
    window.dispatchEvent(new CustomEvent('kelly:thought-captured', {
      detail: thought
    }));
  }

  /**
   * Get all pending thoughts
   */
  getPending() {
    return this.pendingThoughts.filter(t => !t.scheduled);
  }
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initKellyFavicon);
} else {
  initKellyFavicon();
}

function initKellyFavicon() {
  // Only initialize once
  if (window.kellyFavicon) return;
  
  window.kellyFavicon = new KellyFavicon();
  window.kellyCalendarIt = new KellyCalendarIt(window.kellyFavicon);
  
  // Expose convenience methods globally
  window.kellyNotify = () => window.kellyFavicon.notifyLesson();
  window.kellyCelebrate = () => window.kellyFavicon.celebrate();
  window.kellyThinking = () => window.kellyFavicon.thinking();
  window.kellyIdle = () => window.kellyFavicon.idle();
  
  // Quick calendar-it function
  window.calendarIt = (text) => {
    window.dispatchEvent(new CustomEvent('kelly:calendar-it', {
      detail: { text }
    }));
  };
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { KellyFavicon, KellyCalendarIt };
}








