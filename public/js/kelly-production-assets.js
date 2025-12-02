/**
 * Kelly Production Assets System v4.0
 * Manages Kelly poses with complete state mapping for lesson phases
 * 
 * IMAGE INVENTORY (/kelly/poses/):
 * - kelly_welcome.png     → Standing, hand on arm (welcoming)
 * - kelly_idle.png        → Chair, hand gesturing (presenting/explaining)
 * - kelly_hint.png        → Hand on chin (thinking/pondering)
 * - kelly_clasp.png       → Hands clasped in lap (attentive/encouraging)
 * - kelly_choice_left.png → Pointing left (Option A)
 * - kelly_choice_right.png→ Pointing right (Option B)
 * - kelly_listening.png   → Hands clasped, leaning in (conversation mode)
 * - kelly_hint_flip.png   → Alternate thinking pose
 */

// Kelly Production Assets - All available poses
const KELLY_ASSETS = {
  // CORE STATES
  hello: {
    png: '/kelly/poses/kelly_welcome.png',
    description: 'Welcoming pose - standing with hand on arm'
  },
  welcome: {
    png: '/kelly/poses/kelly_welcome.png',
    description: 'Same as hello - warm welcome'
  },
  thinking: {
    png: '/kelly/poses/kelly_hint.png',
    description: 'Thoughtful pose - hand on chin'
  },
  explaining: {
    png: '/kelly/poses/kelly_idle.png',
    description: 'Explaining/presenting - hand gesturing'
  },
  
  // OPTION POINTING
  'pointing-left': {
    png: '/kelly/poses/kelly_choice_left.png',
    description: 'Pointing to Option A (viewer\'s left)'
  },
  'pointing-right': {
    png: '/kelly/poses/kelly_choice_right.png',
    description: 'Pointing to Option B (viewer\'s right)'
  },
  
  // EMOTIONAL STATES
  excited: {
    png: '/kelly/poses/kelly_idle.png',
    description: 'Excited/animated - presenting gesture'
  },
  celebrating: {
    png: '/kelly/poses/kelly_welcome.png',
    description: 'Celebrating success'
  },
  encouraging: {
    png: '/kelly/poses/kelly_clasp.png',
    description: 'Encouraging/supportive - hands clasped'
  },
  proud: {
    png: '/kelly/poses/kelly_welcome.png',
    description: 'Proud of learner'
  },
  
  // INTERACTION STATES
  listening: {
    png: '/kelly/poses/kelly_listening.png',
    description: 'Listening mode - for voice conversation'
  },
  attentive: {
    png: '/kelly/poses/kelly_clasp.png',
    description: 'Attentive/waiting'
  },
  
  // HINT/REVEAL STATES  
  hint: {
    png: '/kelly/poses/kelly_hint.png',
    description: 'About to give a hint'
  },
  'hint-flip': {
    png: '/kelly/poses/kelly_hint_flip.png',
    description: 'Alternate hint pose'
  },
  
  // LEGACY SUPPORT (for older code)
  clasp: {
    png: '/kelly/poses/kelly_clasp.png',
    description: 'Hands clasped'
  },
  'bot-right-index': {
    png: '/kelly/poses/bot_right_index.png',
    description: 'Pointing bottom right'
  },
  'cam-right-index': {
    png: '/kelly/poses/cam_right_index.png',
    description: 'Pointing at camera right'
  },
  'rail-left-thumb': {
    png: '/kelly/poses/rail_left_thumb.png',
    description: 'Thumbs up left'
  }
};

// Map lesson phases to Kelly poses (matches SYSTEM-SPEC)
const KELLY_PHASE_MAP = {
  // From CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
  welcome: 'welcome',      // LESSON_PHASES.WELCOME.kellyPose
  question: 'thinking',    // Q1, Q2, Q3 phases
  q1: 'thinking',
  q2: 'thinking',
  q3: 'thinking',
  hook: 'excited',         // LESSON_PHASES.HOOK.kellyPose
  complete: 'celebrating', // LESSON_PHASES.COMPLETE.kellyPose
  wisdom: 'explaining',    // Wisdom phase
  
  // Additional states for interactions
  choiceA: 'pointing-left',
  choiceB: 'pointing-right',
  feedback_correct: 'celebrating',
  feedback_incorrect: 'encouraging',
  conversation: 'listening',
  speaking: 'explaining'
};

// Legacy state map for backward compatibility
const KELLY_STATE_MAP = KELLY_PHASE_MAP;

class KellyAssetManager {
  constructor(imageElement) {
    this.imageElement = imageElement;
    this.currentState = 'hello';
    this.preloadedImages = new Map();
    this.supportsWebP = this.checkWebPSupport();
    
    // Set initial image
    if (this.imageElement) {
      this.imageElement.src = KELLY_ASSETS.hello.png;
    }
  }
  
  checkWebPSupport() {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
  }
  
  getOptimalSize() {
    const width = window.innerWidth * window.devicePixelRatio;
    if (width <= 640) return '640';
    if (width <= 1280) return '1280';
    if (width <= 1920) return '1920';
    return '2560';
  }
  
  getAssetUrl(state) {
    const asset = KELLY_ASSETS[state];
    if (!asset) {
      console.warn(`[KellyAssets] Unknown state: ${state}, falling back to hello`);
      return KELLY_ASSETS.hello.png;
    }
    return asset.png;
  }
  
  async preloadState(state) {
    if (this.preloadedImages.has(state)) return;
    
    const url = this.getAssetUrl(state);
    const img = new Image();
    
    return new Promise((resolve) => {
      img.onload = () => {
        this.preloadedImages.set(state, img);
        resolve(img);
      };
      img.onerror = () => {
        console.warn(`[KellyAssets] Failed to preload: ${state}`);
        resolve(null);
      };
      img.src = url;
    });
  }
  
  async preloadAll() {
    const states = Object.keys(KELLY_ASSETS);
    console.log(`[KellyAssets] Preloading ${states.length} states...`);
    await Promise.all(states.map(s => this.preloadState(s)));
    console.log('[KellyAssets] ✅ All states preloaded');
  }
  
  async preloadEssential() {
    // Preload the most common lesson states first
    const essential = [
      'hello', 
      'welcome',
      'thinking', 
      'pointing-left', 
      'pointing-right',
      'explaining',
      'celebrating',
      'listening'
    ];
    await Promise.all(essential.map(s => this.preloadState(s)));
    console.log('[KellyAssets] ✅ Essential states preloaded');
  }
  
  setState(state, animate = true) {
    // Normalize state name
    const normalizedState = state?.toLowerCase?.() || 'hello';
    
    if (!KELLY_ASSETS[normalizedState]) {
      console.warn(`[KellyAssets] Invalid state: ${state}, using hello`);
      state = 'hello';
    } else {
      state = normalizedState;
    }
    
    if (state === this.currentState) return;
    
    const url = this.getAssetUrl(state);
    
    if (!this.imageElement) {
      console.warn('[KellyAssets] No image element set');
      return;
    }
    
    if (animate) {
      // Crossfade transition
      this.imageElement.style.opacity = '0';
      
      setTimeout(() => {
        this.imageElement.src = url;
        this.imageElement.style.opacity = '1';
      }, 150);
    } else {
      this.imageElement.src = url;
    }
    
    this.currentState = state;
    console.log(`[KellyAssets] State: ${state}`);
  }
  
  // Set state based on lesson phase
  setStateForPhase(phaseType) {
    const state = KELLY_PHASE_MAP[phaseType] || KELLY_PHASE_MAP[phaseType?.toLowerCase?.()] || 'thinking';
    this.setState(state);
  }
  
  // Convenience methods
  pointLeft() { this.setState('pointing-left'); }
  pointRight() { this.setState('pointing-right'); }
  think() { this.setState('thinking'); }
  greet() { this.setState('hello'); }
  welcome() { this.setState('welcome'); }
  celebrate() { this.setState('celebrating'); }
  listen() { this.setState('listening'); }
  explain() { this.setState('explaining'); }
  encourage() { this.setState('encouraging'); }
}

// Export
window.KELLY_ASSETS = KELLY_ASSETS;
window.KELLY_PHASE_MAP = KELLY_PHASE_MAP;
window.KELLY_STATE_MAP = KELLY_STATE_MAP;
window.KellyAssetManager = KellyAssetManager;

console.log('[KellyAssets] ✅ v4.0 loaded - Complete pose system with', Object.keys(KELLY_ASSETS).length, 'states');
