/**
 * Unity Loader - Handles Unity WebGL initialization and iframe management
 * 
 * Responsibilities:
 * - Load Unity WebGL build (iframe or direct canvas)
 * - Handle initialization errors
 * - Manage Unity instance lifecycle
 * - Provide retry/fallback mechanisms
 */

/* eslint-env browser */
export default class UnityLoader {
  constructor(options = {}) {
    this.config = {
      buildUrl: options.buildUrl || '/unity/kelly-live/Build',
      loaderScript: options.loaderScript || 'Kelly_Web_Build.loader.js',
      canvasId: options.canvasId || 'unity-canvas',
      iframeId: options.iframeId || 'unity-iframe',
      useIframe: options.useIframe !== false, // Default to iframe
      onLoad: options.onLoad || null,
      onError: options.onError || null,
      onProgress: options.onProgress || null,
    };
    
    this.unityInstance = null;
    this.loadAttempts = 0;
    this.maxRetries = 3;
    this.isLoading = false;
  }

  /**
   * Load Unity build (iframe or direct canvas)
   */
  async load() {
    if (this.isLoading) {
      console.warn('[UnityLoader] Load already in progress');
      return null;
    }

    this.isLoading = true;
    this.loadAttempts += 1;

    try {
      if (this.config.useIframe) {
        return await this.loadViaIframe();
      } else {
        return await this.loadViaCanvas();
      }
    } catch (error) {
      this.isLoading = false;
      this.handleLoadError('load_failed', error);
      
      // Retry if attempts remaining
      if (this.loadAttempts < this.maxRetries) {
        console.log(`[UnityLoader] Retrying load (attempt ${this.loadAttempts + 1}/${this.maxRetries})`);
        await this.delay(2000 * this.loadAttempts); // Exponential backoff
        return this.load();
      }
      
      throw error;
    }
  }

  /**
   * Load Unity via iframe (recommended for production)
   */
  async loadViaIframe() {
    const iframe = document.getElementById(this.config.iframeId);
    if (!iframe) {
      throw new Error(`Unity iframe not found: #${this.config.iframeId}`);
    }

    const src = iframe.dataset.src || `${this.config.buildUrl}/../index.html`;
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Unity iframe load timeout'));
      }, 30000);

      iframe.onload = () => {
        clearTimeout(timeout);
        this.isLoading = false;
        this.unityInstance = { iframe, type: 'iframe' };
        
        if (this.config.onLoad) {
          this.config.onLoad(this.unityInstance);
        }
        
        resolve(this.unityInstance);
      };

      iframe.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Unity iframe load error'));
      };

      iframe.src = src;
    });
  }

  /**
   * Load Unity via direct canvas embed (for development)
   */
  async loadViaCanvas() {
    const canvas = document.getElementById(this.config.canvasId);
    if (!canvas) {
      throw new Error(`Unity canvas not found: #${this.config.canvasId}`);
    }

    const loaderUrl = `${this.config.buildUrl}/${this.config.loaderScript}`;
    const buildUrl = this.config.buildUrl;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Unity script load timeout'));
      }, 30000);

      const script = document.createElement('script');
      script.src = loaderUrl;
      
      script.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Unity loader script failed to load'));
      };

      script.onload = () => {
        clearTimeout(timeout);
        
        if (typeof createUnityInstance === 'undefined') {
          reject(new Error('Unity instance creator not found'));
          return;
        }

        const config = {
          dataUrl: `${buildUrl}/Kelly_Web_Build.data.br`,
          frameworkUrl: `${buildUrl}/Kelly_Web_Build.framework.js.br`,
          codeUrl: `${buildUrl}/Kelly_Web_Build.wasm.br`,
          streamingAssetsUrl: 'StreamingAssets',
          companyName: 'Curious Kelly PBC',
          productName: 'Curious Kelly',
          productVersion: '1.0',
        };

        createUnityInstance(canvas, config, (progress) => {
          if (this.config.onProgress) {
            this.config.onProgress(progress);
          }
        })
        .then((instance) => {
          this.isLoading = false;
          this.unityInstance = instance;
          
          if (this.config.onLoad) {
            this.config.onLoad(this.unityInstance);
          }
          
          resolve(this.unityInstance);
        })
        .catch((error) => {
          reject(error);
        });
      };

      document.body.appendChild(script);
    });
  }

  /**
   * Handle load errors with user-friendly UI
   */
  handleLoadError(type, error = null) {
    console.error(`[UnityLoader] ${type}:`, error);

    const errorInfo = {
      'load_failed': {
        title: 'Avatar System Unavailable',
        message: 'Kelly\'s avatar is temporarily offline. You can still learn with text and audio.',
      },
      'script_load_failed': {
        title: 'Avatar System Unavailable',
        message: 'Unable to load avatar system. Please check your internet connection and try again.',
      },
      'unity_instance_undefined': {
        title: 'Avatar System Error',
        message: 'Avatar system failed to initialize. Please refresh the page.',
      },
    };

    const info = errorInfo[type] || errorInfo['load_failed'];
    
    this.showErrorUI({
      ...info,
      actions: [
        { label: 'Retry', action: () => this.retry() },
        { label: 'Continue Without Avatar', action: () => this.disableUnity() },
      ],
    });

    if (this.config.onError) {
      this.config.onError(type, error);
    }

    // Analytics
    if (window.gtag) {
      gtag('event', 'unity_load_error', {
        error_type: type,
        error_message: error?.message || 'unknown',
      });
    }
  }

  /**
   * Show error UI overlay
   */
  showErrorUI(errorInfo) {
    const overlay = document.getElementById('unity-overlay');
    if (!overlay) {
      console.warn('[UnityLoader] Error overlay not found');
      return;
    }

    overlay.innerHTML = `
      <div class="unity-error">
        <div class="error-icon">⚠️</div>
        <h3>${errorInfo.title}</h3>
        <p>${errorInfo.message}</p>
        <div class="error-actions">
          ${errorInfo.actions.map((action, i) => 
            `<button class="error-action-btn" data-action="${i}">${action.label}</button>`
          ).join('')}
        </div>
      </div>
    `;
    overlay.classList.remove('hidden');

    // Bind action buttons
    overlay.querySelectorAll('.error-action-btn').forEach((btn, i) => {
      btn.addEventListener('click', () => {
        errorInfo.actions[i].action();
      });
    });
  }

  /**
   * Retry loading Unity
   */
  async retry() {
    const overlay = document.getElementById('unity-overlay');
    if (overlay) {
      overlay.innerHTML = '<div class="unity-loading">Retrying...</div>';
    }
    
    try {
      await this.load();
    } catch (error) {
      // Error handling will show UI
    }
  }

  /**
   * Disable Unity and show fallback UI
   */
  disableUnity() {
    const overlay = document.getElementById('unity-overlay');
    if (overlay) {
      overlay.innerHTML = `
        <div class="unity-disabled">
          <div class="disabled-icon">📚</div>
          <h3>Learning Mode</h3>
          <p>Continuing without avatar. All lesson content is still available.</p>
        </div>
      `;
    }

    // Hide Unity container
    const container = document.getElementById('unity-container');
    if (container) {
      container.style.display = 'none';
    }

    // Emit event for app to handle
    window.dispatchEvent(new CustomEvent('unity-disabled'));
  }

  /**
   * Get Unity instance (if loaded)
   */
  getInstance() {
    return this.unityInstance;
  }

  /**
   * Check if Unity is loaded
   */
  isLoaded() {
    return this.unityInstance !== null;
  }

  /**
   * Utility: Delay promise
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

