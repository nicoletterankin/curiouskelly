/**
 * Unity Kelly Loader
 * Loads Unity WebGL build for 3D Kelly avatar
 *
 * Features:
 * - WebGL capability detection
 * - Memory check before loading
 * - Progress tracking
 * - Graceful fallback to 2D
 * - Timeout handling
 */

class UnityKellyLoader {
  constructor(options = {}) {
    this.options = {
      canvasId: options.canvasId || 'unity-canvas',
      buildPath: options.buildPath || 'https://nicoletterankin.github.io/kelly-v2/Build',
      buildName: options.buildName || 'kelly',
      timeout: options.timeout || 45000, // 45 seconds
      onProgress: options.onProgress || null,
      onLoad: options.onLoad || null,
      onError: options.onError || null,
      ...options
    };

    this.unityInstance = null;
    this.isLoading = false;
    this.isLoaded = false;
    this.loadFailed = false;
    this.abortController = null;
  }

  /**
   * Check if device supports Unity WebGL
   */
  static isSupported() {
    // Check WebGL support
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) {
        console.warn('[UnityLoader] WebGL not supported');
        return false;
      }
    } catch (e) {
      console.warn('[UnityLoader] WebGL check failed:', e);
      return false;
    }

    // Check device memory (need ~500MB free)
    if (navigator.deviceMemory && navigator.deviceMemory < 2) {
      console.warn('[UnityLoader] Low memory device (<2GB)');
      return false;
    }

    // Check if mobile with low-end indicators
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    const isLowEnd = /iPhone [5-7]|iPad [2-4]/i.test(navigator.userAgent);

    if (isMobile && isLowEnd) {
      console.warn('[UnityLoader] Low-end mobile device');
      return false;
    }

    return true;
  }

  /**
   * Get device tier for performance expectations
   */
  static getDeviceTier() {
    const memory = navigator.deviceMemory || 4;
    const cores = navigator.hardwareConcurrency || 4;
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    if (memory >= 8 && cores >= 8 && !isMobile) {
      return 'high'; // Desktop/laptop with good specs
    } else if (memory >= 4 && cores >= 4) {
      return 'mid'; // Average device
    } else {
      return 'low'; // Low-end device
    }
  }

  /**
   * Load Unity WebGL build
   */
  async load() {
    if (this.isLoading) {
      console.warn('[UnityLoader] Load already in progress');
      return null;
    }

    if (this.isLoaded && this.unityInstance) {
      console.log('[UnityLoader] Already loaded, returning instance');
      return this.unityInstance;
    }

    if (!UnityKellyLoader.isSupported()) {
      const error = new Error('Device does not support Unity WebGL');
      this.handleError('not_supported', error);
      return null;
    }

    this.isLoading = true;
    this.loadFailed = false;

    const canvas = document.getElementById(this.options.canvasId);
    if (!canvas) {
      const error = new Error(`Canvas element not found: ${this.options.canvasId}`);
      this.handleError('canvas_not_found', error);
      return null;
    }

    console.log('[UnityLoader] Starting Unity load...');

    return new Promise((resolve, reject) => {
      // Set up timeout
      const timeoutId = setTimeout(() => {
        this.handleError('timeout', new Error('Unity load timeout'));
        reject(new Error('Unity load timeout'));
      }, this.options.timeout);

      // Load the Unity loader script
      const loaderUrl = `${this.options.buildPath}/${this.options.buildName}.loader.js`;
      const script = document.createElement('script');
      script.src = loaderUrl;

      script.onerror = () => {
        clearTimeout(timeoutId);
        const error = new Error('Failed to load Unity loader script');
        this.handleError('script_load_failed', error);
        reject(error);
      };

      script.onload = () => {
        // Check if createUnityInstance is available
        if (typeof createUnityInstance === 'undefined') {
          clearTimeout(timeoutId);
          const error = new Error('createUnityInstance not defined');
          this.handleError('loader_invalid', error);
          reject(error);
          return;
        }

        // Unity configuration (loading from GitHub Pages CDN)
        const config = {
          dataUrl: `${this.options.buildPath}/${this.options.buildName}.data.gz`,
          frameworkUrl: `${this.options.buildPath}/${this.options.buildName}.framework.js.gz`,
          codeUrl: `${this.options.buildPath}/${this.options.buildName}.wasm.gz`,
          streamingAssetsUrl: `${this.options.buildPath.replace('/Build', '')}/StreamingAssets`,
          companyName: 'CuriousKelly',
          productName: 'Kelly',
          productVersion: '1.0'
        };

        // Create Unity instance
        createUnityInstance(canvas, config, (progress) => {
          if (this.options.onProgress) {
            this.options.onProgress(progress);
          }
          this.dispatchEvent('progress', { progress });
        })
          .then((instance) => {
            clearTimeout(timeoutId);
            this.unityInstance = instance;
            this.isLoaded = true;
            this.isLoading = false;

            console.log('[UnityLoader] Unity loaded successfully');

            if (this.options.onLoad) {
              this.options.onLoad(instance);
            }

            this.dispatchEvent('loaded', { instance });
            resolve(instance);
          })
          .catch((error) => {
            clearTimeout(timeoutId);
            this.handleError('load_failed', error);
            reject(error);
          });
      };

      document.body.appendChild(script);
    });
  }

  /**
   * Handle load errors
   */
  handleError(type, error) {
    console.error(`[UnityLoader] Error (${type}):`, error);

    this.isLoading = false;
    this.loadFailed = true;

    if (this.options.onError) {
      this.options.onError(type, error);
    }

    this.dispatchEvent('error', { type, error });
  }

  /**
   * Send message to Unity
   */
  sendMessage(objectName, methodName, value = '') {
    if (!this.unityInstance) {
      console.warn('[UnityLoader] Unity not loaded, cannot send message');
      return false;
    }

    try {
      this.unityInstance.SendMessage(objectName, methodName, value);
      return true;
    } catch (e) {
      console.error('[UnityLoader] SendMessage failed:', e);
      return false;
    }
  }

  /**
   * Unity object name (kelly_fbx_v4 is the actual GameObject name in the scene)
   * This must match the GameObject that has KellyWebGLBridge.cs attached
   */
  getKellyObjectName() {
    return 'kelly_fbx_v4';
  }

  /**
   * Set Kelly expression in Unity
   * Note: The KellyAvatarController.cs needs SetExpression method added
   */
  setExpression(expression) {
    return this.sendMessage(this.getKellyObjectName(), 'SetExpression', expression);
  }

  /**
   * Start lip sync with viseme data
   * Note: The KellyAvatarController.cs needs StartLipSync method added
   */
  startLipSync(text) {
    return this.sendMessage(this.getKellyObjectName(), 'StartLipSync', text);
  }

  /**
   * Stop lip sync
   * Note: The KellyAvatarController.cs needs StopLipSync method added
   */
  stopLipSync() {
    return this.sendMessage(this.getKellyObjectName(), 'StopLipSync');
  }

  /**
   * Play animation
   * Note: The KellyAvatarController.cs needs PlayAnimation method added
   */
  playAnimation(animationName) {
    return this.sendMessage(this.getKellyObjectName(), 'PlayAnimation', animationName);
  }

  /**
   * Process viseme for lip sync (this method EXISTS in KellyAvatarController.cs)
   */
  processViseme(visemeName, weight) {
    return this.sendMessage(this.getKellyObjectName(), 'ProcessViseme', `${visemeName}:${weight}`);
  }

  /**
   * Unload Unity to free memory
   */
  async unload() {
    if (!this.unityInstance) return;

    console.log('[UnityLoader] Unloading Unity...');

    try {
      await this.unityInstance.Quit();
      this.unityInstance = null;
      this.isLoaded = false;
      console.log('[UnityLoader] Unity unloaded');
      this.dispatchEvent('unloaded');
    } catch (e) {
      console.error('[UnityLoader] Unload failed:', e);
    }
  }

  /**
   * Check if loaded
   */
  getIsLoaded() {
    return this.isLoaded;
  }

  /**
   * Check if loading
   */
  getIsLoading() {
    return this.isLoading;
  }

  /**
   * Get Unity instance
   */
  getInstance() {
    return this.unityInstance;
  }

  /**
   * Dispatch custom event
   */
  dispatchEvent(name, detail = {}) {
    document.dispatchEvent(new CustomEvent(`unity-kelly-${name}`, { detail }));
  }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnityKellyLoader;
}

// Make available globally
window.UnityKellyLoader = UnityKellyLoader;
