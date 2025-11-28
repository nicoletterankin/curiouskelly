/**
 * Kelly Avatar Controller
 * Unified controller for 2D and 3D Kelly avatar modes
 *
 * Features:
 * - Seamless 2D/3D mode switching
 * - State synchronization between modes
 * - Automatic fallback to 2D if 3D fails
 * - Progressive enhancement (2D default, 3D opt-in)
 * - Persistence of user preference
 */

class KellyAvatarController {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      defaultMode: options.defaultMode || '2D',
      unityBuildPath: options.unityBuildPath || '/unity/kelly/Build',
      allow3D: options.allow3D !== false,
      autoSavePreference: options.autoSavePreference !== false,
      onModeChange: options.onModeChange || null,
      onError: options.onError || null,
      ...options
    };

    this.state = {
      mode: null,
      expression: 'curious',
      phase: 'welcome',
      isSpeaking: false,
      isTransitioning: false
    };

    this.avatar2D = null;
    this.unityLoader = null;

    this.elements = {
      wrapper: null,
      layer2D: null,
      layer3D: null,
      loadingOverlay: null
    };

    this.init();
  }

  async init() {
    this.createDOM();

    // Initialize 2D avatar (always available)
    this.avatar2D = new Kelly2DAvatar(this.elements.layer2D, {
      preload: true,
      enableBreathing: true
    });

    // Always start with 2D (instant, no loading)
    // Only load 3D when user explicitly toggles
    await this.switchTo2D(false);

    console.log('[KellyController] Initialized in 2D mode');
  }

  createDOM() {
    this.container.innerHTML = `
            <div class="kelly-avatar-controller">
                <!-- 2D Layer (always present) -->
                <div class="kelly-layer kelly-2d-layer" id="kelly-2d-layer"></div>
                
                <!-- 3D Layer (Unity canvas) -->
                <div class="kelly-layer kelly-3d-layer" id="kelly-3d-layer" style="display: none;">
                    <canvas id="unity-canvas" style="width: 100%; height: 100%;"></canvas>
                </div>
                
                <!-- Loading overlay for 3D -->
                <div class="kelly-loading-overlay" id="kelly-loading-overlay" style="display: none;">
                    <div class="kelly-loading-content">
                        <div class="kelly-loading-spinner"></div>
                        <div class="kelly-loading-text">Loading 3D Kelly...</div>
                        <div class="kelly-loading-progress" id="kelly-loading-progress">0%</div>
                        <button class="kelly-loading-cancel" id="kelly-loading-cancel">
                            Stay in 2D
                        </button>
                    </div>
                </div>
            </div>
        `;

    this.elements.wrapper = this.container.querySelector('.kelly-avatar-controller');
    this.elements.layer2D = this.container.querySelector('#kelly-2d-layer');
    this.elements.layer3D = this.container.querySelector('#kelly-3d-layer');
    this.elements.loadingOverlay = this.container.querySelector('#kelly-loading-overlay');

    // Cancel button handler
    const cancelBtn = this.container.querySelector('#kelly-loading-cancel');
    if (cancelBtn) {
      cancelBtn.onclick = () => this.cancelUnityLoad();
    }

    // Inject styles
    this.injectStyles();
  }

  injectStyles() {
    if (document.getElementById('kelly-controller-styles')) return;

    const styles = document.createElement('style');
    styles.id = 'kelly-controller-styles';
    styles.textContent = `
            .kelly-avatar-controller {
                position: relative;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
            
            .kelly-layer {
                position: absolute;
                inset: 0;
                transition: opacity 0.4s ease;
            }
            
            .kelly-2d-layer {
                z-index: 1;
            }
            
            .kelly-3d-layer {
                z-index: 2;
                background: transparent;
            }
            
            .kelly-3d-layer canvas {
                display: block;
            }
            
            .kelly-loading-overlay {
                position: absolute;
                inset: 0;
                z-index: 10;
                background: rgba(0, 0, 0, 0.9);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .kelly-loading-content {
                text-align: center;
                color: white;
            }
            
            .kelly-loading-spinner {
                width: 48px;
                height: 48px;
                border: 3px solid rgba(255, 255, 255, 0.2);
                border-top-color: #3b82f6;
                border-radius: 50%;
                margin: 0 auto 16px;
                animation: kelly-spin 1s linear infinite;
            }
            
            @keyframes kelly-spin {
                to { transform: rotate(360deg); }
            }
            
            .kelly-loading-text {
                font-size: 16px;
                margin-bottom: 8px;
            }
            
            .kelly-loading-progress {
                font-size: 24px;
                font-weight: 700;
                color: #3b82f6;
                margin-bottom: 24px;
            }
            
            .kelly-loading-cancel {
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: rgba(255, 255, 255, 0.7);
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
            }
            
            .kelly-loading-cancel:hover {
                background: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.5);
                color: white;
            }
        `;
    document.head.appendChild(styles);
  }

  /**
   * Set avatar mode (2D or 3D)
   */
  async setMode(mode, animate = true) {
    if (mode === this.state.mode) return true;
    if (this.state.isTransitioning) return false;

    console.log(`[KellyController] Setting mode: ${this.state.mode} → ${mode}`);

    if (mode === '3D') {
      return this.switchTo3D(animate);
    } else {
      return this.switchTo2D(animate);
    }
  }

  /**
   * Switch to 2D mode
   */
  async switchTo2D(animate = true) {
    this.state.isTransitioning = true;

    // Hide 3D, show 2D
    if (animate) {
      this.elements.layer3D.style.opacity = '0';
      await this.wait(400);
    }

    this.elements.layer3D.style.display = 'none';
    this.elements.layer2D.style.display = 'block';

    if (animate) {
      this.elements.layer2D.style.opacity = '0';
      await this.wait(50);
      this.elements.layer2D.style.opacity = '1';
    }

    // Sync expression state to 2D
    if (this.avatar2D) {
      this.avatar2D.setExpression(this.state.expression);
      this.avatar2D.setSpeaking(this.state.isSpeaking);
    }

    this.state.mode = '2D';
    this.state.isTransitioning = false;

    // Save preference
    if (this.options.autoSavePreference) {
      localStorage.setItem('kelly_mode', '2D');
    }

    this.dispatchEvent('mode-changed', { mode: '2D' });

    if (this.options.onModeChange) {
      this.options.onModeChange('2D');
    }

    return true;
  }

  /**
   * Switch to 3D mode
   */
  async switchTo3D(animate = true) {
    // Check if 3D is allowed
    if (!this.options.allow3D) {
      console.warn('[KellyController] 3D mode not allowed');
      return false;
    }

    // Check device support
    if (!UnityKellyLoader.isSupported()) {
      console.warn('[KellyController] Device does not support 3D');
      this.dispatchEvent('3d-not-supported');
      if (this.options.onError) {
        this.options.onError('3d_not_supported', 'Device does not support Unity WebGL');
      }
      return false;
    }

    this.state.isTransitioning = true;

    // Show loading overlay
    this.elements.loadingOverlay.style.display = 'flex';

    // Initialize Unity loader if needed
    if (!this.unityLoader) {
      this.unityLoader = new UnityKellyLoader({
        canvasId: 'unity-canvas',
        buildPath: this.options.unityBuildPath,
        onProgress: (progress) => {
          const progressEl = document.getElementById('kelly-loading-progress');
          if (progressEl) {
            progressEl.textContent = `${Math.round(progress * 100)}%`;
          }
        },
        onError: (type, error) => {
          console.error('[KellyController] Unity load failed:', type, error);
          this.handleUnityLoadError(type, error);
        }
      });
    }

    // Load Unity
    try {
      if (!this.unityLoader.getIsLoaded()) {
        await this.unityLoader.load();
      }

      // Hide loading overlay
      this.elements.loadingOverlay.style.display = 'none';

      // Show 3D layer
      this.elements.layer3D.style.display = 'block';

      if (animate) {
        this.elements.layer2D.style.opacity = '0';
        this.elements.layer3D.style.opacity = '0';
        await this.wait(50);
        this.elements.layer3D.style.opacity = '1';
        await this.wait(400);
      }

      this.elements.layer2D.style.display = 'none';

      // Sync expression to Unity
      this.unityLoader.setExpression(this.state.expression);
      if (this.state.isSpeaking) {
        this.unityLoader.startLipSync('');
      }

      this.state.mode = '3D';
      this.state.isTransitioning = false;

      // Save preference
      if (this.options.autoSavePreference) {
        localStorage.setItem('kelly_mode', '3D');
      }

      this.dispatchEvent('mode-changed', { mode: '3D' });

      if (this.options.onModeChange) {
        this.options.onModeChange('3D');
      }

      return true;
    } catch (error) {
      this.handleUnityLoadError('load_exception', error);
      return false;
    }
  }

  /**
   * Handle Unity load error
   */
  handleUnityLoadError(type, error) {
    this.elements.loadingOverlay.style.display = 'none';
    this.state.isTransitioning = false;

    // Fall back to 2D
    this.switchTo2D(false);

    // Notify
    this.dispatchEvent('3d-load-failed', { type, error });

    if (this.options.onError) {
      this.options.onError(type, error);
    }
  }

  /**
   * Cancel Unity load
   */
  cancelUnityLoad() {
    console.log('[KellyController] Unity load cancelled by user');
    this.elements.loadingOverlay.style.display = 'none';
    this.state.isTransitioning = false;
    this.switchTo2D(false);
  }

  /**
   * Toggle between 2D and 3D
   */
  toggleMode() {
    const newMode = this.state.mode === '2D' ? '3D' : '2D';
    return this.setMode(newMode);
  }

  /**
   * Get current mode
   */
  getMode() {
    return this.state.mode;
  }

  /**
   * Set Kelly expression
   */
  setExpression(expression) {
    this.state.expression = expression;

    if (this.state.mode === '2D' && this.avatar2D) {
      this.avatar2D.setExpression(expression);
    } else if (this.state.mode === '3D' && this.unityLoader?.getIsLoaded()) {
      this.unityLoader.setExpression(expression);
    }
  }

  /**
   * Set expression based on lesson phase
   */
  setPhase(phase, choice = null) {
    this.state.phase = phase;

    if (this.state.mode === '2D' && this.avatar2D) {
      this.avatar2D.setPhase(phase, choice);
    } else if (this.state.mode === '3D' && this.unityLoader?.getIsLoaded()) {
      // Map phase to expression for Unity
      const expressionMap = {
        welcome: 'curious',
        q1: choice
          ? choice === 'a'
            ? 'explaining'
            : choice === 'b'
              ? 'celebrating'
              : 'wisdom'
          : 'curious',
        q2: choice
          ? choice === 'a'
            ? 'explaining'
            : choice === 'b'
              ? 'celebrating'
              : 'wisdom'
          : 'curious',
        q3: choice
          ? choice === 'a'
            ? 'explaining'
            : choice === 'b'
              ? 'celebrating'
              : 'wisdom'
          : 'listening',
        wisdom: 'wisdom',
        complete: 'celebrating'
      };

      const expression = expressionMap[phase] || 'curious';
      this.state.expression = expression;
      this.unityLoader.setExpression(expression);

      if (choice === 'b' || phase === 'complete') {
        this.unityLoader.playAnimation('celebrate');
      }
    }
  }

  /**
   * Set speaking state
   */
  setSpeaking(speaking, text = '') {
    this.state.isSpeaking = speaking;

    if (this.state.mode === '2D' && this.avatar2D) {
      this.avatar2D.setSpeaking(speaking);
    } else if (this.state.mode === '3D' && this.unityLoader?.getIsLoaded()) {
      if (speaking) {
        this.unityLoader.startLipSync(text);
      } else {
        this.unityLoader.stopLipSync();
      }
    }
  }

  /**
   * Get current expression
   */
  getExpression() {
    return this.state.expression;
  }

  /**
   * Check if currently speaking
   */
  isSpeaking() {
    return this.state.isSpeaking;
  }

  /**
   * Check if 3D is supported on this device
   */
  is3DSupported() {
    return UnityKellyLoader.isSupported();
  }

  /**
   * Check if 3D is currently loaded
   */
  is3DLoaded() {
    return this.unityLoader?.getIsLoaded() || false;
  }

  // Utility methods
  wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  dispatchEvent(name, detail = {}) {
    document.dispatchEvent(new CustomEvent(`kelly-avatar-${name}`, { detail }));
  }

  /**
   * Clean up
   */
  destroy() {
    if (this.avatar2D) {
      this.avatar2D.destroy();
    }
    if (this.unityLoader) {
      this.unityLoader.unload();
    }
    this.container.innerHTML = '';
    console.log('[KellyController] Destroyed');
  }
}

// Export for ES modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyAvatarController;
}

// Make available globally
window.KellyAvatarController = KellyAvatarController;
