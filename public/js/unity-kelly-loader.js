/**
 * Unity Kelly Loader with Client-Side Decompression
 *
 * Problem: GitHub Pages serves .unityweb files without Content-Encoding headers
 * Solution: Fetch compressed files, decompress client-side with pako, create blob URLs
 *
 * Build location: https://nicoletterankin.github.io/kelly-v2/Build/
 */

class UnityKellyLoader {
  constructor() {
    this.buildUrl = 'https://nicoletterankin.github.io/kelly-v2/Build';
    this.unityInstance = null;
    this.isLoading = false;
    this.isLoaded = false;
    this.loadPromise = null;
    this.pakoLoaded = false;
  }

  /**
   * Load Unity - handles all decompression automatically
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
    this._updateLoading(
      loadingOverlay,
      loadingText,
      progressBar,
      'Loading decompression library...',
      0
    );

    // Step 1: Load pako for gzip decompression
    await this._loadPako();
    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Loading Unity engine...', 5);

    // Step 2: Load the Unity loader script (not compressed)
    await this._loadScript(`${this.buildUrl}/WebGL.loader.js`);

    if (typeof createUnityInstance !== 'function') {
      throw new Error('Unity loader script failed - createUnityInstance not found');
    }

    this._updateLoading(
      loadingOverlay,
      loadingText,
      progressBar,
      'Downloading Kelly 3D assets...',
      10
    );

    // Step 3: Fetch and decompress all Unity files in parallel
    console.log('[Unity] Starting parallel download and decompression...');

    const [frameworkJs, wasmCode, dataFile] = await Promise.all([
      this._fetchAndDecompress(`${this.buildUrl}/WebGL.framework.js.unityweb`, 'framework', (p) => {
        this._updateLoading(
          loadingOverlay,
          loadingText,
          progressBar,
          'Downloading framework...',
          10 + p * 20
        );
      }),
      this._fetchAndDecompress(`${this.buildUrl}/WebGL.wasm.unityweb`, 'wasm', (p) => {
        this._updateLoading(
          loadingOverlay,
          loadingText,
          progressBar,
          'Downloading WebAssembly...',
          30 + p * 30
        );
      }),
      this._fetchAndDecompress(`${this.buildUrl}/WebGL.data.unityweb`, 'data', (p) => {
        this._updateLoading(
          loadingOverlay,
          loadingText,
          progressBar,
          'Downloading Kelly model...',
          60 + p * 20
        );
      })
    ]);

    this._updateLoading(loadingOverlay, loadingText, progressBar, 'Preparing Kelly 3D...', 80);

    // Step 4: Create blob URLs from decompressed content
    const frameworkBlob = new Blob([frameworkJs], { type: 'application/javascript' });
    const wasmBlob = new Blob([wasmCode], { type: 'application/wasm' });
    const dataBlob = new Blob([dataFile], { type: 'application/octet-stream' });

    const config = {
      dataUrl: URL.createObjectURL(dataBlob),
      frameworkUrl: URL.createObjectURL(frameworkBlob),
      codeUrl: URL.createObjectURL(wasmBlob),
      streamingAssetsUrl: `${this.buildUrl}/StreamingAssets`,
      companyName: 'LessonOfTheDay',
      productName: 'CuriousKelly',
      productVersion: '1.0'
    };

    console.log('[Unity] Created blob URLs, starting Unity instance...');
    this._updateLoading(
      loadingOverlay,
      loadingText,
      progressBar,
      'Starting Kelly 3D engine...',
      85
    );

    // Step 5: Create Unity instance
    const instance = await createUnityInstance(canvas, config, (progress) => {
      const percent = Math.round(85 + progress * 15);
      this._updateLoading(
        loadingOverlay,
        loadingText,
        progressBar,
        progress < 1 ? `Initializing Kelly... ${Math.round(progress * 100)}%` : 'Kelly is ready!',
        percent
      );
    });

    // Step 6: Cleanup
    if (loadingOverlay) loadingOverlay.style.display = 'none';

    // Revoke blob URLs to free memory (Unity has already loaded them)
    setTimeout(() => {
      URL.revokeObjectURL(config.dataUrl);
      URL.revokeObjectURL(config.frameworkUrl);
      URL.revokeObjectURL(config.codeUrl);
    }, 1000);

    console.log('[Unity] ✅ Kelly 3D loaded successfully!');

    // Dispatch ready event
    window.dispatchEvent(new CustomEvent('unity-ready', { detail: instance }));

    return instance;
  }

  /**
   * Load pako decompression library from CDN
   */
  async _loadPako() {
    if (this.pakoLoaded || window.pako) {
      this.pakoLoaded = true;
      console.log('[Unity] Pako already loaded');
      return;
    }

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js';
      script.crossOrigin = 'anonymous';

      script.onload = () => {
        this.pakoLoaded = true;
        console.log('[Unity] ✅ Pako decompression library loaded');
        resolve();
      };

      script.onerror = () => {
        reject(new Error('Failed to load pako decompression library'));
      };

      document.head.appendChild(script);
    });
  }

  /**
   * Fetch a file and decompress if needed
   */
  async _fetchAndDecompress(url, label, onProgress) {
    console.log(`[Unity] Fetching ${label} from ${url}`);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${label}: HTTP ${response.status}`);
    }

    // Get total size for progress
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    // Read response as stream for progress tracking
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      received += value.length;

      if (total && onProgress) {
        onProgress(received / total);
      }
    }

    // Combine chunks into single array
    const compressed = new Uint8Array(received);
    let position = 0;
    for (const chunk of chunks) {
      compressed.set(chunk, position);
      position += chunk.length;
    }

    console.log(`[Unity] ${label}: ${(received / 1024 / 1024).toFixed(2)} MB downloaded`);

    // Check for gzip magic number (0x1f 0x8b)
    if (compressed[0] === 0x1f && compressed[1] === 0x8b) {
      console.log(`[Unity] ${label}: Detected gzip compression, decompressing...`);

      try {
        const decompressed = window.pako.inflate(compressed);
        console.log(
          `[Unity] ${label}: ${(decompressed.length / 1024 / 1024).toFixed(2)} MB decompressed`
        );
        return decompressed;
      } catch (e) {
        console.error(`[Unity] ${label}: Decompression failed:`, e);
        throw new Error(`Failed to decompress ${label}: ${e.message}`);
      }
    } else {
      console.log(`[Unity] ${label}: Not gzip compressed, using raw data`);
      return compressed;
    }
  }

  /**
   * Load an external script
   */
  _loadScript(url) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${url}"]`);
      if (existing) {
        console.log('[Unity] Script already exists:', url);
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = url;
      script.crossOrigin = 'anonymous';

      script.onload = () => {
        console.log('[Unity] ✅ Loaded script:', url);
        resolve();
      };

      script.onerror = () => {
        reject(new Error(`Failed to load script: ${url}`));
      };

      document.body.appendChild(script);
    });
  }

  /**
   * Update loading UI
   */
  _updateLoading(overlay, textEl, progressBar, message, percent) {
    if (overlay) overlay.style.display = 'flex';
    if (textEl) textEl.textContent = message;
    if (progressBar) progressBar.style.width = `${percent}%`;
  }

  /**
   * Show error in loading UI
   */
  _showError(message) {
    const loadingText = document.querySelector('.unity-loading-text');
    const loadingOverlay = document.getElementById('unity-loading');

    if (loadingText) {
      loadingText.textContent = `Error: ${message}`;
      loadingText.style.color = '#ef4444';
    }

    // Add retry button if not exists
    if (loadingOverlay && !loadingOverlay.querySelector('.unity-retry-btn')) {
      const retryBtn = document.createElement('button');
      retryBtn.className = 'unity-retry-btn';
      retryBtn.textContent = 'Try Again';
      retryBtn.style.cssText = `
        margin-top: 16px;
        padding: 10px 24px;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        cursor: pointer;
        transition: background 0.2s;
      `;
      retryBtn.onmouseover = () => (retryBtn.style.background = '#1d4ed8');
      retryBtn.onmouseout = () => (retryBtn.style.background = '#2563eb');
      retryBtn.onclick = () => {
        this.isLoading = false;
        this.loadPromise = null;
        if (loadingText) loadingText.style.color = '';
        retryBtn.remove();
        this.load();
      };
      loadingOverlay.appendChild(retryBtn);
    }
  }

  // ==================== PUBLIC API ====================

  /**
   * Send a message to Unity GameObject
   */
  sendMessage(gameObject, method, value) {
    if (!this.unityInstance) {
      console.warn('[Unity] Cannot send message - Unity not loaded');
      return false;
    }

    try {
      console.log(`[Unity] SendMessage: ${gameObject}.${method}(${value})`);
      this.unityInstance.SendMessage(gameObject, method, value);
      return true;
    } catch (e) {
      console.error('[Unity] SendMessage failed:', e);
      return false;
    }
  }

  /**
   * Set Kelly's expression
   */
  setExpression(expression, intensity = 1.0) {
    const data = JSON.stringify({ expression, intensity });
    return this.sendMessage('kelly_fbx_v4', 'SetExpression', data);
  }

  /**
   * Set a specific blendshape
   */
  setBlendshape(name, value) {
    const data = JSON.stringify({ name, value });
    return this.sendMessage('kelly_fbx_v4', 'SetBlendshape', data);
  }

  /**
   * Set viseme for lip sync
   */
  setViseme(viseme, weight) {
    const data = JSON.stringify({ viseme, weight });
    return this.sendMessage('kelly_fbx_v4', 'SetViseme', data);
  }

  /**
   * Reset to neutral expression
   */
  resetExpression() {
    return this.setExpression('neutral', 1.0);
  }

  /**
   * Check if Unity is loaded and ready
   */
  isReady() {
    return this.isLoaded && this.unityInstance !== null;
  }
}

// Create global instance
window.unityKellyLoader = new UnityKellyLoader();

// Log that loader is ready
console.log('[Unity] Kelly loader initialized (with decompression support)');
