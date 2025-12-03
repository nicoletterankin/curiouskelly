/**
 * Unity Kelly Loader - Production Version
 * Loads 3D Kelly from Cloudflare Worker + R2 CDN
 * 
 * CDN: unity-cdn.nicoletterankin.workers.dev (serves R2 with proper headers)
 * R2 Bucket: curious-kelly-unity
 * Files: Brotli-compressed Unity WebGL build
 */

class UnityKellyLoader {
  constructor(options = {}) {
    // Cloudflare Worker CDN - serves R2 with CORS and Content-Encoding headers
    this.cdnUrl = options.cdnUrl || 'https://unity-cdn.nicoletterankin.workers.dev';
    
    // Determine environment
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    // Fallback URLs with priority based on environment
    this.fallbackUrls = isLocalhost 
      ? [
          '/unity/kelly-live/Build',                                    // Local dev: fastest (uncompressed)
          'https://unity-cdn.nicoletterankin.workers.dev',              // Fallback: Worker CDN (Brotli)
          'https://meek-hamster-500f3c.netlify.app/Build',              // Last resort: Netlify
        ]
      : [
          'https://unity-cdn.nicoletterankin.workers.dev',              // Production: Worker CDN (Brotli, CORS)
          '/unity/kelly-live/Build',                                    // Fallback: Local if bundled
          'https://meek-hamster-500f3c.netlify.app/Build',              // Last resort: Netlify
        ];
    this.currentUrlIndex = 0;
    
    this.unityInstance = null;
    this.isLoading = false;
    this.isLoaded = false;
    this.loadPromise = null;
    
    // Track load metrics
    this.metrics = {
      startTime: null,
      loadTime: null,
      source: null,
      fallbackAttempts: 0
    };
    
    // Performance monitor integration
    this.perfMonitor = window.unityPerfMonitor || null;
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
    this.metrics.startTime = performance.now();
    
    // Start performance tracking
    if (this.perfMonitor) {
      this.perfMonitor.startTracking();
    }
    
    this.loadPromise = this._loadWithFallback();

    try {
      this.unityInstance = await this.loadPromise;
      this.isLoaded = true;
      this.metrics.loadTime = performance.now() - this.metrics.startTime;
      console.log(`[Unity] Loaded successfully in ${Math.round(this.metrics.loadTime)}ms from ${this.metrics.source}`);
      
      // Record load completion
      if (this.perfMonitor) {
        this.perfMonitor.recordLoaded();
      }
      
      return this.unityInstance;
    } catch (error) {
      console.error('[Unity] Load failed:', error);
      this._showError(error.message);
      
      // Record error
      if (this.perfMonitor) {
        this.perfMonitor.recordError(error, { phase: 'load' });
      }
      
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Try loading from each URL until one works
   */
  async _loadWithFallback() {
    for (let i = 0; i < this.fallbackUrls.length; i++) {
      const buildUrl = this.fallbackUrls[i];
      try {
        console.log(`[Unity] Trying source ${i + 1}/${this.fallbackUrls.length}: ${buildUrl}`);
        this.metrics.source = buildUrl;
        return await this._doLoad(buildUrl);
      } catch (error) {
        console.warn(`[Unity] Failed to load from ${buildUrl}:`, error.message);
        
        // Record fallback attempt
        this.metrics.fallbackAttempts++;
        if (this.perfMonitor && i < this.fallbackUrls.length - 1) {
          this.perfMonitor.recordFallback(buildUrl, this.fallbackUrls[i + 1], error);
        }
        
        if (i === this.fallbackUrls.length - 1) {
          throw new Error(`All Unity sources failed. Last error: ${error.message}`);
        }
      }
    }
  }

  async _doLoad(buildUrl) {
    const canvas = document.getElementById('unity-canvas');
    const loadingOverlay = document.getElementById('unity-loading');
    const progressBar = document.getElementById('unity-progress-bar');
    const loadingText = document.querySelector('.unity-loading-text');

    if (!canvas) {
      throw new Error('Unity canvas element #unity-canvas not found');
    }

    // Show loading overlay
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Loading Unity engine...', 5);

    // Determine file paths based on source
    const isWorkerCDN = buildUrl.includes('workers.dev');
    const isR2Direct = buildUrl.includes('r2.dev');
    const isNetlify = buildUrl.includes('netlify.app');
    const isLocal = buildUrl.startsWith('/');
    
    let loaderUrl, config;
    
    if (isWorkerCDN || isR2Direct) {
      // Cloudflare Worker CDN or R2 - Brotli compressed files with Kelly_Web_Build prefix
      loaderUrl = `${buildUrl}/Kelly_Web_Build.loader.js`;
      config = {
        dataUrl: `${buildUrl}/Kelly_Web_Build.data.br`,
        frameworkUrl: `${buildUrl}/Kelly_Web_Build.framework.js.br`,
        codeUrl: `${buildUrl}/Kelly_Web_Build.wasm.br`,
        streamingAssetsUrl: `${buildUrl}`,
        companyName: 'LessonOfTheDay',
        productName: 'CuriousKelly',
        productVersion: '2.0'
      };
    } else if (isNetlify) {
      // Netlify - uncompressed WebGL files
      loaderUrl = `${buildUrl}/WebGL.loader.js`;
      config = {
        dataUrl: `${buildUrl}/WebGL.data`,
        frameworkUrl: `${buildUrl}/WebGL.framework.js`,
        codeUrl: `${buildUrl}/WebGL.wasm`,
        streamingAssetsUrl: `${buildUrl}/../StreamingAssets`,
        companyName: 'LessonOfTheDay',
        productName: 'CuriousKelly',
        productVersion: '2.0'
      };
    } else {
      // Local build - uncompressed
      loaderUrl = `${buildUrl}/WebGL.loader.js`;
      config = {
        dataUrl: `${buildUrl}/WebGL.data`,
        frameworkUrl: `${buildUrl}/WebGL.framework.js`,
        codeUrl: `${buildUrl}/WebGL.wasm`,
        streamingAssetsUrl: `${buildUrl}/../StreamingAssets`,
        companyName: 'LessonOfTheDay',
        productName: 'CuriousKelly',
        productVersion: '2.0'
      };
    }

    // Step 1: Load the Unity loader script
    await this._loadScript(loaderUrl, !isLocal);
    
    // Record loader loaded
    if (this.perfMonitor) {
      this.perfMonitor.recordLoaderLoaded(buildUrl);
    }

    if (typeof createUnityInstance !== 'function') {
      throw new Error('Unity loader script failed - createUnityInstance not found');
    }

    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Preparing Kelly 3D...', 15);

    console.log('[Unity] Starting Kelly 3D with config:', config);
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Loading Kelly 3D model...', 20);

    // Step 2: Create Unity instance with progress callback
    const instance = await createUnityInstance(canvas, config, (progress) => {
      const percent = Math.round(20 + progress * 75);
      const msg = progress < 0.5 
        ? 'Downloading Kelly...' 
        : progress < 0.9 
          ? 'Preparing Kelly...'
          : 'Kelly is almost ready!';
      this._updateLoading(loadingOverlay, loadingText, progressBar, `${msg} ${Math.round(progress * 100)}%`, percent);
      
      // Track progress
      if (this.perfMonitor) {
        this.perfMonitor.recordProgress(progress);
      }
    });

    // Step 3: Hide loading overlay
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Kelly is ready!', 100);
    
    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.style.display = 'none';
    }, 500);

    console.log('[Unity] Kelly 3D loaded successfully!');
    return instance;
  }

  _loadScript(src, useCrossOrigin = false) {
    return new Promise((resolve, reject) => {
      // Check if already loaded
      if (document.querySelector(`script[src="${src}"]`)) {
        console.log('[Unity] Loader script already loaded');
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = src;
      // Use crossOrigin for external scripts (Worker CDN has CORS headers)
      if (useCrossOrigin) {
        script.crossOrigin = 'anonymous';
      }
      script.onload = () => {
        console.log('[Unity] Loader script loaded:', src);
        resolve();
      };
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
      this.unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', expression);
    }
  }

  startLipSync(text) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('kelly_fbx_v4', 'StartLipSync', text);
    }
  }

  stopLipSync() {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('kelly_fbx_v4', 'StopLipSync');
    }
  }

  setPhase(phase) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('kelly_fbx_v4', 'SetPhase', phase);
    }
  }

  setSpeaking(speaking) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('kelly_fbx_v4', 'SetSpeaking', speaking ? '1' : '0');
    }
  }

  playAnimation(animName) {
    if (this.unityInstance) {
      this.unityInstance.SendMessage('kelly_fbx_v4', 'PlayAnimation', animName);
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
  
  // Get load metrics for analytics
  getMetrics() {
    const basicMetrics = {
      ...this.metrics,
      isLoaded: this.isLoaded
    };
    
    // Include performance monitor metrics if available
    if (this.perfMonitor) {
      return {
        ...basicMetrics,
        performance: this.perfMonitor.getMetrics()
      };
    }
    
    return basicMetrics;
  }
  
  // Print performance report to console
  printPerformanceReport() {
    if (this.perfMonitor) {
      return this.perfMonitor.printReport();
    }
    
    console.group('🎮 Unity Load Report');
    console.log(`Load Time: ${this.metrics.loadTime ? Math.round(this.metrics.loadTime) + 'ms' : 'N/A'}`);
    console.log(`Source: ${this.metrics.source || 'N/A'}`);
    console.log(`Fallback Attempts: ${this.metrics.fallbackAttempts}`);
    console.log(`Status: ${this.isLoaded ? 'Loaded' : 'Not Loaded'}`);
    console.groupEnd();
    
    return this.getMetrics();
  }
}

// Create global instance with Worker CDN
window.unityKellyLoader = new UnityKellyLoader({
  cdnUrl: 'https://unity-cdn.nicoletterankin.workers.dev'
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnityKellyLoader;
}
