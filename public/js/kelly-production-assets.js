/**
 * Kelly Production Assets System
 * Manages optimized Kelly images with responsive loading
 */

// Kelly Production Assets v2.0 - HIGH QUALITY
const KELLY_ASSETS = {
  hello: {
    webp: {
      640: '/assets/kelly/production/webp/hello-640.webp',
      1280: '/assets/kelly/production/webp/hello-1280.webp',
      1920: '/assets/kelly/production/webp/hello-1920.webp',
      2560: '/assets/kelly/production/webp/hello-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/hello.jpeg'
  },
  thinking: {
    webp: {
      640: '/assets/kelly/production/webp/thinking-640.webp',
      1280: '/assets/kelly/production/webp/thinking-1280.webp',
      1920: '/assets/kelly/production/webp/thinking-1920.webp',
      2560: '/assets/kelly/production/webp/thinking-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/thinking.jpeg'
  },
  'pointing-left': {
    webp: {
      640: '/assets/kelly/production/webp/pointing-left-640.webp',
      1280: '/assets/kelly/production/webp/pointing-left-1280.webp',
      1920: '/assets/kelly/production/webp/pointing-left-1920.webp',
      2560: '/assets/kelly/production/webp/pointing-left-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/pointing-left.jpeg'
  },
  'pointing-right': {
    webp: {
      640: '/assets/kelly/production/webp/pointing-right-640.webp',
      1280: '/assets/kelly/production/webp/pointing-right-1280.webp',
      1920: '/assets/kelly/production/webp/pointing-right-1920.webp',
      2560: '/assets/kelly/production/webp/pointing-right-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/pointing-right.jpeg'
  },
  'out-left': {
    webp: {
      640: '/assets/kelly/production/webp/out-left-640.webp',
      1280: '/assets/kelly/production/webp/out-left-1280.webp',
      1920: '/assets/kelly/production/webp/out-left-1920.webp',
      2560: '/assets/kelly/production/webp/out-left-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/out-left.jpeg'
  },
  'out-right': {
    webp: {
      640: '/assets/kelly/production/webp/out-right-640.webp',
      1280: '/assets/kelly/production/webp/out-right-1280.webp',
      1920: '/assets/kelly/production/webp/out-right-1920.webp',
      2560: '/assets/kelly/production/webp/out-right-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/out-right.jpeg'
  },
  'mid-left': {
    webp: {
      640: '/assets/kelly/production/webp/mid-left-640.webp',
      1280: '/assets/kelly/production/webp/mid-left-1280.webp',
      1920: '/assets/kelly/production/webp/mid-left-1920.webp',
      2560: '/assets/kelly/production/webp/mid-left-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/mid-left.jpeg'
  },
  'mid-right': {
    webp: {
      640: '/assets/kelly/production/webp/mid-right-640.webp',
      1280: '/assets/kelly/production/webp/mid-right-1280.webp',
      1920: '/assets/kelly/production/webp/mid-right-1920.webp',
      2560: '/assets/kelly/production/webp/mid-right-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/mid-right.jpeg'
  },
  'in-left': {
    webp: {
      640: '/assets/kelly/production/webp/in-left-640.webp',
      1280: '/assets/kelly/production/webp/in-left-1280.webp',
      1920: '/assets/kelly/production/webp/in-left-1920.webp',
      2560: '/assets/kelly/production/webp/in-left-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/in-left.jpeg'
  },
  'in-right': {
    webp: {
      640: '/assets/kelly/production/webp/in-right-640.webp',
      1280: '/assets/kelly/production/webp/in-right-1280.webp',
      1920: '/assets/kelly/production/webp/in-right-1920.webp',
      2560: '/assets/kelly/production/webp/in-right-2560.webp'
    },
    jpeg: '/assets/kelly/production/jpeg/in-right.jpeg'
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
      return KELLY_ASSETS.hello.jpeg;
    }
    
    if (this.supportsWebP) {
      const size = this.getOptimalSize();
      return asset.webp[size];
    }
    return asset.jpeg;
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

