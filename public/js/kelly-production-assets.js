/**
 * Kelly Production Assets System
 * Manages optimized Kelly images with responsive loading
 */

// Kelly Production Assets v3.0 - Using /kelly/poses/ directory
const KELLY_ASSETS = {
  hello: {
    png: '/kelly/poses/kelly_welcome.png'
  },
  thinking: {
    png: '/kelly/poses/kelly_idle.png'
  },
  'pointing-left': {
    png: '/kelly/poses/kelly_choice_left.png'
  },
  'pointing-right': {
    png: '/kelly/poses/kelly_choice_right.png'
  },
  'hint': {
    png: '/kelly/poses/kelly_hint.png'
  },
  'hint-flip': {
    png: '/kelly/poses/kelly_hint_flip.png'
  },
  'listening': {
    png: '/kelly/poses/kelly_listening.png'
  },
  'clasp': {
    png: '/kelly/poses/kelly_clasp.png'
  },
  'bot-right-index': {
    png: '/kelly/poses/bot_right_index.png'
  },
  'cam-right-index': {
    png: '/kelly/poses/cam_right_index.png'
  },
  'rail-left-thumb': {
    png: '/kelly/poses/rail_left_thumb.png'
  }
};

// State mapping for lesson phases
const KELLY_STATE_MAP = {
  welcome: 'hello',
  question: 'thinking',
  choiceA: 'pointing-left',
  choiceB: 'pointing-right',
  wisdom: 'hello',
  celebrating: 'hello'
};

class KellyAssetManager {
  constructor(imageElement) {
    this.imageElement = imageElement;
    this.currentState = 'hello';
    this.preloadedImages = new Map();
    this.supportsWebP = this.checkWebPSupport();
  }
  
  checkWebPSupport() {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
  }
  
  getOptimalSize() {
    // Use device pixel ratio for crisp images on retina/high-DPI displays
    const width = window.innerWidth * window.devicePixelRatio;
    if (width <= 640) return '640';
    if (width <= 1280) return '1280';
    if (width <= 1920) return '1920';
    return '2560'; // 4K quality for high-DPI displays
  }
  
  getAssetUrl(state) {
    const asset = KELLY_ASSETS[state];
    if (!asset) {
      console.warn(`[KellyAssets] Unknown state: ${state}`);
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
    // Preload the most common states first
    const essential = ['hello', 'thinking', 'pointing-left', 'pointing-right'];
    await Promise.all(essential.map(s => this.preloadState(s)));
    console.log('[KellyAssets] ✅ Essential states preloaded');
  }
  
  setState(state, animate = true) {
    if (!KELLY_ASSETS[state]) {
      console.warn(`[KellyAssets] Invalid state: ${state}`);
      return;
    }
    
    if (state === this.currentState) return;
    
    const url = this.getAssetUrl(state);
    
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
  
  setStateForPhase(phaseType) {
    const state = KELLY_STATE_MAP[phaseType] || 'thinking';
    this.setState(state);
  }
  
  pointLeft() {
    this.setState('pointing-left');
  }
  
  pointRight() {
    this.setState('pointing-right');
  }
  
  think() {
    this.setState('thinking');
  }
  
  greet() {
    this.setState('hello');
  }
}

// Export
window.KELLY_ASSETS = KELLY_ASSETS;
window.KellyAssetManager = KellyAssetManager;

