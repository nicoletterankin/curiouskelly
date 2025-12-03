/**
 * Unity Kelly Loader - Production Version
 * Loads 3D Kelly from local Unity WebGL build
 */

class UnityKellyLoader {
  constructor() {
    // Production Unity build on Netlify
    this.buildUrl = 'https://meek-hamster-500f3c.netlify.app/Build';
    this.unityInstance = null;
    this.isLoading = false;
    this.isLoaded = false;
    this.loadPromise = null;
  }

  /**
   * Load Unity - returns the Unity instance
   */
  async load() {
    if (this.isLoaded && this.unityInstance) {
      console.log('[Unity] Already loaded, returning cached instance');
      return this.unityInstance;
    }
    if (this.isLoading && this.loadPromise) {
      console.log('[Unity] Already loading, waiting...');
      return this.loadPromise;
    }

    this.isLoading = true;
    this.loadPromise = this._doLoad();

    try {
      this.unityInstance = await this.loadPromise;
      this.isLoaded = true;
      return this.unityInstance;
    } catch (error) {
      console.error('[Unity] Load failed:', error);
      this._showError(error.message);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  async _doLoad() {
    const canvas = document.getElementById('unity-canvas');
    const loadingOverlay = document.getElementById('unity-loading');
    const progressBar = document.getElementById('unity-progress-bar');
    const loadingText = document.querySelector('.unity-loading-text');

    if (!canvas) {
      throw new Error('Unity canvas element #unity-canvas not found');
    }

    // Show loading overlay
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Loading Unity engine...', 5);

    // Step 1: Load the Unity loader script
    await this._loadScript(`${this.buildUrl}/WebGL.loader.js`);

    if (typeof createUnityInstance !== 'function') {
      throw new Error('Unity loader script failed - createUnityInstance not found');
    }

    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Preparing Kelly 3D...', 15);

    // Step 2: Configure Unity with Netlify hosted files (uncompressed dev build)
    const config = {
      dataUrl: `${this.buildUrl}/WebGL.data`,
      frameworkUrl: `${this.buildUrl}/WebGL.framework.js`,
      codeUrl: `${this.buildUrl}/WebGL.wasm`,
      streamingAssetsUrl: `${this.buildUrl}/../StreamingAssets`,
      companyName: 'LessonOfTheDay',
      productName: 'CuriousKelly',
      productVersion: '2.0'
    };

    console.log('[Unity] Starting Kelly 3D with config:', config);
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Loading Kelly 3D model...', 20);

    // Step 3: Create Unity instance with progress callback
    const instance = await createUnityInstance(canvas, config, (progress) => {
      const percent = Math.round(20 + progress * 75);
      const msg = progress < 0.5 
        ? 'Downloading Kelly...' 
        : progress < 0.9 
          ? 'Preparing Kelly...'
          : 'Kelly is almost ready!';
      this._updateLoading(loadingOverlay, loadingText, progressBar, `${msg} ${Math.round(progress * 100)}%`, percent);
    });

    // Step 4: Hide loading overlay
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Kelly is ready!', 100);
    
    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.style.display = 'none';
    }, 500);

    console.log('[Unity] Kelly 3D loaded successfully!');
    return instance;
  }

  _loadScript(src) {
    return new Promise((resolve, reject) => {
      // Check if already loaded
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = () => reject(new Error(`Failed to load: ${src}`));
      document.head.appendChild(script);
    });
  }

  _updateLoading(overlay, text, bar, message, percent) {
    if (text) text.textContent = message;
    if (bar) bar.style.width = `${percent}%`;
    if (overlay) overlay.style.display = 'flex';
    console.log(`[Unity] ${message} (${percent}%)`);
  }

  _showError(message) {
    const loadingOverlay = document.getElementById('unity-loading');
    const loadingText = document.querySelector('.unity-loading-text');
    
    if (loadingText) {
      loadingText.textContent = `Error: ${message}`;
      loadingText.style.color = '#ff6b6b';
    }
    
    console.error('[Unity] Error:', message);
  }

  // Public API for controlling Kelly
  setExpression(expression) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'SetExpression', expression);
    }
  }

  startLipSync(text) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'StartLipSync', text);
    }
  }

  stopLipSync() {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'StopLipSync');
    }
  }

  setPhase(phase) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'SetPhase', phase);
    }
  }

  setSpeaking(speaking) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'SetSpeaking', speaking ? '1' : '0');
    }
  }

  playAnimation(animName) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('KellyAvatar', 'PlayAnimation', animName);
    }
  }

  unload() {
    if (this.unityInstance) {
      this.unityInstance.Quit().then(() => {
        console.log('[Unity] Kelly 3D unloaded');
        this.unityInstance = null;
        this.isLoaded = false;
      });
    }
  }
}

// Create global instance
window.unityKellyLoader = new UnityKellyLoader();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnityKellyLoader;
}
