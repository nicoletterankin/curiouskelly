/**
 * Kelly Unified Avatar Manager
 * Manages both 2D and 3D avatar modes with automatic fallback
 * 
 * Features:
 * - Automatic WebGL support detection
 * - Unity 3D loading with progress UI
 * - Graceful fallback to 2D on failure
 * - Mode persistence via localStorage
 * - Unified API regardless of mode
 */

class KellyUnifiedAvatar {
  constructor(options = {}) {
    this.options = {
      container: options.container || document.getElementById('layer-background'),
      unityContainer: options.unityContainer || document.getElementById('kelly-unity-container'),
      unityIframe: options.unityIframe || document.getElementById('kelly-unity-iframe'),
      kellyImage: options.kellyImage || document.getElementById('kelly-image'),
      defaultMode: options.defaultMode || localStorage.getItem('kelly_avatar_mode') || '2d',
      autoLoad3D: options.autoLoad3D !== false,
      unityBuildPath: options.unityBuildPath || '/unity/kelly-live/Build',
      unityTimeout: options.unityTimeout || 30000, // 30 seconds
      ...options
    };
    
    // State
    this.currentMode = null;
    this.targetMode = this.options.defaultMode;
    this.isLoading = false;
    this.isReady = false;
    
    // Avatar instances
    this.avatar2D = null;
    this.unityLoader = null;
    this.unityBridge = null;
    
    // Current expression/phase state (synced between modes)
    this.state = {
      expression: 'curious',
      phase: 'welcome',
      isSpeaking: false,
      lessonId: 1
    };
    
    // Callbacks
    this.onModeChange = options.onModeChange || null;
    this.onReady = options.onReady || null;
    this.onError = options.onError || null;
    
    this.init();
  }
  
  /**
   * Initialize the unified avatar system
   */
  init() {
    console.log('[KellyUnified] Initializing unified avatar system...');
    
    // Always initialize 2D first (fast, works everywhere)
    this._init2D();
    
    // Check if 3D is supported and requested
    if (this.targetMode === '3d' && this.isWebGLSupported()) {
      // Start loading 3D in background
      if (this.options.autoLoad3D) {
        this._init3D();
      }
    } else {
      // Just use 2D
      this.currentMode = '2d';
      this.isReady = true;
      this._notifyReady();
    }
    
    // Create mode toggle UI
    this._createModeToggle();
  }
  
  /**
   * Check if WebGL is supported
   */
  isWebGLSupported() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!gl) {
        console.warn('[KellyUnified] WebGL not supported');
        return false;
      }
      
      // Check device memory (Unity needs ~500MB)
      if (navigator.deviceMemory && navigator.deviceMemory < 2) {
        console.warn('[KellyUnified] Low memory device, 3D not recommended');
        return false;
      }
      
      return true;
    } catch (e) {
      console.warn('[KellyUnified] WebGL check failed:', e);
      return false;
    }
  }
  
  /**
   * Initialize 2D avatar
   */
  _init2D() {
    if (this.avatar2D) return;
    
    const container = this.options.kellyImage?.parentElement || this.options.container;
    
    if (!container) {
      console.error('[KellyUnified] No container found for 2D avatar');
      return;
    }
    
    // Check if Kelly2DAvatar is available
    if (typeof window.Kelly2DAvatar !== 'undefined') {
      this.avatar2D = new window.Kelly2DAvatar(container, {});
      console.log('[KellyUnified] ✅ 2D avatar initialized');
    } else {
      console.warn('[KellyUnified] Kelly2DAvatar not loaded');
    }
  }
  
  /**
   * Initialize 3D Unity avatar
   */
  async _init3D() {
    if (this.isLoading) return;
    
    this.isLoading = true;
    console.log('[KellyUnified] Loading 3D avatar...');
    
    // Show loading UI
    this._showLoading('Preparing 3D Kelly...');
    
    try {
      // Create or get Unity loader
      if (!this.unityLoader && typeof window.UnityKellyLoader !== 'undefined') {
        this.unityLoader = new window.UnityKellyLoader({
          cdnUrl: 'https://unity-cdn.nicoletterankin.workers.dev'
        });
      } else if (window.unityKellyLoader) {
        this.unityLoader = window.unityKellyLoader;
      }
      
      if (!this.unityLoader) {
        throw new Error('Unity loader not available');
      }
      
      // Set timeout
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Unity load timeout')), this.options.unityTimeout);
      });
      
      // Load Unity with timeout
      const unityInstance = await Promise.race([
        this.unityLoader.load(),
        timeoutPromise
      ]);
      
      // Create bridge
      if (typeof window.UnityBridge !== 'undefined') {
        this.unityBridge = new window.UnityBridge(unityInstance);
      }
      
      // Switch to 3D mode
      this.currentMode = '3d';
      this._showUnity();
      this._hideLoading();
      this.isReady = true;
      
      // Sync current state to 3D
      this._syncStateTo3D();
      
      console.log('[KellyUnified] ✅ 3D avatar loaded successfully');
      this._notifyReady();
      
    } catch (error) {
      console.error('[KellyUnified] ❌ 3D load failed:', error.message);
      this._hideLoading();
      
      // Fallback to 2D
      this.currentMode = '2d';
      this._show2D();
      this.isReady = true;
      
      if (this.onError) {
        this.onError(error);
      }
      
      this._notifyReady();
    } finally {
      this.isLoading = false;
    }
  }
  
  /**
   * Show loading UI
   */
  _showLoading(message) {
    // Check for existing loading overlay or create one
    let overlay = document.getElementById('kelly-3d-loading');
    
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'kelly-3d-loading';
      overlay.className = 'kelly-3d-loading-overlay';
      overlay.innerHTML = `
        <div class="kelly-3d-loading-content">
          <div class="kelly-3d-spinner"></div>
          <div class="kelly-3d-loading-text">${message}</div>
          <div class="kelly-3d-progress-bar">
            <div class="kelly-3d-progress-fill" id="kelly-3d-progress"></div>
          </div>
          <button class="kelly-3d-cancel-btn" id="kelly-3d-cancel">Stay in 2D</button>
        </div>
      `;
      
      // Add styles if not present
      if (!document.getElementById('kelly-3d-loading-styles')) {
        const style = document.createElement('style');
        style.id = 'kelly-3d-loading-styles';
        style.textContent = `
          .kelly-3d-loading-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            backdrop-filter: blur(10px);
          }
          .kelly-3d-loading-content {
            text-align: center;
            color: white;
          }
          .kelly-3d-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255,255,255,0.2);
            border-top-color: #d97757;
            border-radius: 50%;
            animation: kelly-spin 1s linear infinite;
            margin: 0 auto 20px;
          }
          @keyframes kelly-spin {
            to { transform: rotate(360deg); }
          }
          .kelly-3d-loading-text {
            font-size: 16px;
            margin-bottom: 15px;
          }
          .kelly-3d-progress-bar {
            width: 200px;
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            overflow: hidden;
            margin: 0 auto 20px;
          }
          .kelly-3d-progress-fill {
            height: 100%;
            background: #d97757;
            width: 0%;
            transition: width 0.3s ease;
          }
          .kelly-3d-cancel-btn {
            background: transparent;
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
          }
          .kelly-3d-cancel-btn:hover {
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.5);
          }
        `;
        document.head.appendChild(style);
      }
      
      this.options.container?.appendChild(overlay);
      
      // Cancel button handler
      document.getElementById('kelly-3d-cancel')?.addEventListener('click', () => {
        this._cancel3DLoad();
      });
    }
    
    overlay.style.display = 'flex';
    const textEl = overlay.querySelector('.kelly-3d-loading-text');
    if (textEl) textEl.textContent = message;
  }
  
  /**
   * Hide loading UI
   */
  _hideLoading() {
    const overlay = document.getElementById('kelly-3d-loading');
    if (overlay) {
      overlay.style.display = 'none';
    }
  }
  
  /**
   * Cancel 3D loading and stay in 2D
   */
  _cancel3DLoad() {
    console.log('[KellyUnified] User cancelled 3D load');
    this._hideLoading();
    this.currentMode = '2d';
    this.targetMode = '2d';
    localStorage.setItem('kelly_avatar_mode', '2d');
    this._show2D();
    this._updateModeToggle();
  }
  
  /**
   * Show 2D avatar, hide 3D
   */
  _show2D() {
    // Show 2D image
    if (this.options.kellyImage) {
      this.options.kellyImage.style.display = 'block';
    }
    
    // Hide Unity
    if (this.options.unityContainer) {
      this.options.unityContainer.style.display = 'none';
    }
    if (this.options.unityIframe) {
      this.options.unityIframe.style.display = 'none';
    }
    
    // Make sure canvas is hidden
    const canvas = document.getElementById('unity-canvas');
    if (canvas) canvas.style.display = 'none';
    
    console.log('[KellyUnified] Showing 2D mode');
  }
  
  /**
   * Show 3D Unity, hide 2D
   */
  _showUnity() {
    // Hide 2D image
    if (this.options.kellyImage) {
      this.options.kellyImage.style.display = 'none';
    }
    
    // Show Unity container
    if (this.options.unityContainer) {
      this.options.unityContainer.style.display = 'block';
    }
    
    // Show canvas
    const canvas = document.getElementById('unity-canvas');
    if (canvas) canvas.style.display = 'block';
    
    console.log('[KellyUnified] Showing 3D mode');
  }
  
  /**
   * Create 2D/3D mode toggle button
   */
  _createModeToggle() {
    // Check if toggle already exists
    if (document.getElementById('kelly-mode-toggle')) return;
    
    const toggle = document.createElement('button');
    toggle.id = 'kelly-mode-toggle';
    toggle.className = 'kelly-mode-toggle';
    toggle.title = 'Switch between 2D and 3D Kelly';
    toggle.innerHTML = this.currentMode === '3d' ? '2D' : '3D';
    
    // Add styles
    if (!document.getElementById('kelly-mode-toggle-styles')) {
      const style = document.createElement('style');
      style.id = 'kelly-mode-toggle-styles';
      style.textContent = `
        .kelly-mode-toggle {
          position: absolute;
          bottom: 20px;
          left: 20px;
          z-index: 50;
          background: rgba(0, 0, 0, 0.6);
          color: white;
          border: 1px solid rgba(255,255,255,0.2);
          padding: 8px 16px;
          border-radius: 20px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 600;
          transition: all 0.2s;
          backdrop-filter: blur(10px);
        }
        .kelly-mode-toggle:hover {
          background: rgba(217, 119, 87, 0.8);
          border-color: #d97757;
        }
        .kelly-mode-toggle.loading {
          opacity: 0.5;
          pointer-events: none;
        }
        .kelly-mode-toggle.disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }
      `;
      document.head.appendChild(style);
    }
    
    // Add to container
    this.options.container?.appendChild(toggle);
    
    // Click handler
    toggle.addEventListener('click', () => this.toggleMode());
    
    // Disable if WebGL not supported
    if (!this.isWebGLSupported()) {
      toggle.classList.add('disabled');
      toggle.title = '3D mode not supported on this device';
    }
  }
  
  /**
   * Update mode toggle button
   */
  _updateModeToggle() {
    const toggle = document.getElementById('kelly-mode-toggle');
    if (toggle) {
      toggle.innerHTML = this.currentMode === '3d' ? '2D' : '3D';
      toggle.classList.toggle('loading', this.isLoading);
    }
  }
  
  /**
   * Sync current state to 3D Unity
   */
  _syncStateTo3D() {
    if (!this.unityBridge) return;
    
    this.unityBridge.setExpression(this.state.expression);
    this.unityBridge.setSpeaking(this.state.isSpeaking);
    
    if (this.state.phase) {
      this.unityBridge.setPhase(this.state.phase);
    }
  }
  
  /**
   * Notify ready callbacks
   */
  _notifyReady() {
    if (this.onReady) {
      this.onReady(this.currentMode);
    }
    
    if (this.onModeChange) {
      this.onModeChange(this.currentMode);
    }
    
    // Dispatch event
    document.dispatchEvent(new CustomEvent('kelly-avatar-ready', {
      detail: { mode: this.currentMode }
    }));
  }
  
  // ═══════════════════════════════════════════════════════════════════
  // PUBLIC API - Works in both 2D and 3D modes
  // ═══════════════════════════════════════════════════════════════════
  
  /**
   * Toggle between 2D and 3D modes
   */
  async toggleMode() {
    if (this.isLoading) return;
    
    if (this.currentMode === '3d') {
      // Switch to 2D
      this.currentMode = '2d';
      this.targetMode = '2d';
      this._show2D();
      
      // Sync state to 2D
      if (this.avatar2D) {
        this.avatar2D.setExpression(this.state.expression);
        this.avatar2D.setSpeaking(this.state.isSpeaking);
      }
    } else {
      // Switch to 3D
      if (!this.isWebGLSupported()) {
        console.warn('[KellyUnified] WebGL not supported');
        return;
      }
      
      this.targetMode = '3d';
      
      if (this.unityLoader?.isLoaded) {
        // Unity already loaded, just switch
        this.currentMode = '3d';
        this._showUnity();
        this._syncStateTo3D();
      } else {
        // Need to load Unity
        await this._init3D();
      }
    }
    
    localStorage.setItem('kelly_avatar_mode', this.currentMode);
    this._updateModeToggle();
    
    if (this.onModeChange) {
      this.onModeChange(this.currentMode);
    }
  }
  
  /**
   * Set Kelly's expression
   * @param {string} expression - Expression name (curious, explaining, listening, wisdom, celebrating)
   */
  setExpression(expression) {
    this.state.expression = expression;
    
    if (this.currentMode === '2d' && this.avatar2D) {
      this.avatar2D.setExpression(expression);
    } else if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.setExpression(expression);
    }
  }
  
  /**
   * Set Kelly's speaking state
   * @param {boolean} speaking - Whether Kelly is speaking
   */
  setSpeaking(speaking) {
    this.state.isSpeaking = speaking;
    
    if (this.currentMode === '2d' && this.avatar2D) {
      this.avatar2D.setSpeaking(speaking);
    } else if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.setSpeaking(speaking);
    }
  }
  
  /**
   * Set Kelly's phase
   * @param {string} phase - Phase name (welcome, hook, q1, q2, q3, wisdom, complete)
   * @param {string} choice - Optional choice made (a, b, c)
   */
  setPhase(phase, choice = null) {
    this.state.phase = phase;
    
    if (this.currentMode === '2d' && this.avatar2D) {
      this.avatar2D.setPhase(phase, choice);
    } else if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.setPhase(phase);
    }
  }
  
  /**
   * Load phase-specific visual (2D) or animation (3D)
   * @param {number} lessonId - Lesson day number
   * @param {string} phase - Phase name
   */
  loadPhaseVisual(lessonId, phase) {
    this.state.lessonId = lessonId;
    this.state.phase = phase;
    
    if (this.currentMode === '2d' && this.avatar2D) {
      this.avatar2D.loadPhaseVisual(lessonId, phase);
    } else if (this.currentMode === '3d' && this.unityBridge) {
      // 3D uses phase-based animations
      this.unityBridge.setPhase(phase);
    }
  }
  
  /**
   * Try to play HD video for phase (2D only)
   * @param {number} lessonId - Lesson day number
   * @param {string} phase - Phase name
   * @param {string} archetype - Archetype name
   */
  async playPhaseVideo(lessonId, phase, archetype) {
    if (this.currentMode === '2d' && this.avatar2D) {
      return await this.avatar2D.playPhaseVideo(lessonId, phase, archetype);
    }
    return false;
  }
  
  /**
   * Play animation (3D only)
   * @param {string} animationName - Animation to play
   */
  playAnimation(animationName) {
    if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.playAnimation(animationName);
    }
  }
  
  /**
   * Start lip sync (3D only)
   * @param {string} text - Text being spoken
   */
  startLipSync(text) {
    if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.startLipSync(text);
    }
  }
  
  /**
   * Stop lip sync (3D only)
   */
  stopLipSync() {
    if (this.currentMode === '3d' && this.unityBridge) {
      this.unityBridge.stopLipSync();
    }
  }
  
  /**
   * Get current mode
   * @returns {string} '2d' or '3d'
   */
  getMode() {
    return this.currentMode;
  }
  
  /**
   * Check if ready
   * @returns {boolean}
   */
  isAvatarReady() {
    return this.isReady;
  }
  
  /**
   * Destroy and clean up
   */
  destroy() {
    if (this.avatar2D) {
      this.avatar2D.destroy();
    }
    
    if (this.unityLoader) {
      this.unityLoader.unload();
    }
    
    // Remove UI elements
    document.getElementById('kelly-mode-toggle')?.remove();
    document.getElementById('kelly-3d-loading')?.remove();
    
    console.log('[KellyUnified] Destroyed');
  }
}

// Export globally
window.KellyUnifiedAvatar = KellyUnifiedAvatar;

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyUnifiedAvatar;
}

console.log('[KellyUnified] Module loaded');








