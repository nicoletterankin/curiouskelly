/**
 * 🎛️ KELLY DUAL-MODE CONTROLLER
 * 
 * Manages switching between:
 * - 2D Video Mode (HeyGen pre-rendered videos)
 * - 3D Live Mode (Unity WebGL real-time)
 * 
 * Learners can choose their preferred experience in Settings.
 */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════
  // CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════════

  const CONFIG = {
    STORAGE_KEY: 'kelly_display_mode',
    UNITY_PATH: '/unity/kelly-live/index.html',
    VIDEO_BASE_PATH: '/kelly/2d/videos',
    POSES_PATH: '/kelly/poses',
    MODES: {
      AUTO: 'auto',
      VIDEO_2D: '2d',
      LIVE_3D: '3d',
      HYBRID: 'hybrid',
    },
  };

  // ═══════════════════════════════════════════════════════════════════════
  // MAIN CONTROLLER CLASS
  // ═══════════════════════════════════════════════════════════════════════

  class KellyModeController {
    constructor() {
      this.currentMode = this.loadPreference();
      this.effectiveMode = this.resolveEffectiveMode();
      this.videoPlayer = null;
      this.unityIframe = null;
      this.isUnityLoaded = false;
      this.isUnityLoading = false;
      this.container = null;
      this.unityContainer = null;
      this.callbacks = {};
      
      console.log(`[KellyMode] Initialized: preference=${this.currentMode}, effective=${this.effectiveMode}`);
    }

    // ═══════════════════════════════════════════════════════════════════
    // MODE DETECTION & PREFERENCES
    // ═══════════════════════════════════════════════════════════════════

    loadPreference() {
      try {
        return localStorage.getItem(CONFIG.STORAGE_KEY) || CONFIG.MODES.AUTO;
      } catch {
        return CONFIG.MODES.AUTO;
      }
    }

    savePreference(mode) {
      try {
        localStorage.setItem(CONFIG.STORAGE_KEY, mode);
        this.currentMode = mode;
        this.effectiveMode = this.resolveEffectiveMode();
        console.log(`[KellyMode] Preference saved: ${mode} → effective: ${this.effectiveMode}`);
        this.emit('modeChanged', { mode, effectiveMode: this.effectiveMode });
      } catch (e) {
        console.warn('[KellyMode] Could not save preference:', e);
      }
    }

    resolveEffectiveMode() {
      if (this.currentMode === CONFIG.MODES.AUTO) {
        return this.detectOptimalMode();
      }
      return this.currentMode;
    }

    detectOptimalMode() {
      // Mobile devices → 2D
      const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      if (isMobile) {
        console.log('[KellyMode] Mobile detected → 2D');
        return CONFIG.MODES.VIDEO_2D;
      }

      // Check for WebGL 2 support (required for Unity)
      try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2');
        if (!gl) {
          console.log('[KellyMode] No WebGL2 → 2D');
          return CONFIG.MODES.VIDEO_2D;
        }
      } catch {
        return CONFIG.MODES.VIDEO_2D;
      }

      // Check network speed
      const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      if (connection && connection.effectiveType !== '4g') {
        console.log('[KellyMode] Slow network → 2D');
        return CONFIG.MODES.VIDEO_2D;
      }

      // Check memory (if available)
      if (navigator.deviceMemory && navigator.deviceMemory < 4) {
        console.log('[KellyMode] Low memory → 2D');
        return CONFIG.MODES.VIDEO_2D;
      }

      // Good conditions → hybrid
      console.log('[KellyMode] Good conditions → hybrid');
      return CONFIG.MODES.HYBRID;
    }

    getMode() {
      return {
        preference: this.currentMode,
        effective: this.effectiveMode,
        is2D: this.effectiveMode === CONFIG.MODES.VIDEO_2D,
        is3D: this.effectiveMode === CONFIG.MODES.LIVE_3D,
        isHybrid: this.effectiveMode === CONFIG.MODES.HYBRID,
      };
    }

    // ═══════════════════════════════════════════════════════════════════
    // CONTAINER MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    setContainer(containerId) {
      this.container = document.getElementById(containerId);
      if (!this.container) {
        console.error(`[KellyMode] Container not found: ${containerId}`);
        return false;
      }
      
      // Create sub-containers
      this.container.innerHTML = `
        <div id="kelly-2d-container" class="kelly-mode-container kelly-2d"></div>
        <div id="kelly-3d-container" class="kelly-mode-container kelly-3d" style="display:none;"></div>
        <button id="kelly-mode-toggle" class="kelly-mode-toggle" style="display:none;">🎮 Switch to 3D</button>
      `;
      
      this.unityContainer = document.getElementById('kelly-3d-container');
      this.setupModeToggle();
      
      return true;
    }

    setupModeToggle() {
      const toggle = document.getElementById('kelly-mode-toggle');
      if (toggle) {
        toggle.addEventListener('click', () => this.toggleMode());
      }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2D VIDEO MODE (HeyGen)
    // ═══════════════════════════════════════════════════════════════════

    async play2DVideo(videoUrl, options = {}) {
      const container2D = document.getElementById('kelly-2d-container');
      if (!container2D) {
        console.error('[KellyMode] 2D container not found');
        return null;
      }

      // Hide 3D if showing
      this.hide3D();

      // Create or reuse video element
      if (!this.videoPlayer) {
        this.videoPlayer = document.createElement('video');
        this.videoPlayer.id = 'kelly-2d-player';
        this.videoPlayer.className = 'kelly-video-player';
        this.videoPlayer.playsInline = true;
        this.videoPlayer.preload = 'auto';
        
        // Events
        this.videoPlayer.addEventListener('ended', () => {
          this.emit('videoEnded', { src: this.videoPlayer.src });
        });
        this.videoPlayer.addEventListener('error', (e) => {
          this.emit('videoError', { error: e, src: this.videoPlayer.src });
        });
        
        container2D.appendChild(this.videoPlayer);
      }

      container2D.style.display = 'block';
      this.videoPlayer.style.display = 'block';
      this.videoPlayer.src = videoUrl;

      // Show hybrid toggle if in hybrid mode
      if (this.effectiveMode === CONFIG.MODES.HYBRID) {
        this.showModeToggle();
      }

      if (options.autoplay !== false) {
        try {
          await this.videoPlayer.play();
        } catch (e) {
          console.warn('[KellyMode] Autoplay blocked:', e);
          this.emit('autoplayBlocked', { videoUrl });
        }
      }

      return this.videoPlayer;
    }

    show2DStatic(imageUrl) {
      const container2D = document.getElementById('kelly-2d-container');
      if (!container2D) return;

      this.hide3D();

      let img = container2D.querySelector('.kelly-static-image');
      if (!img) {
        img = document.createElement('img');
        img.className = 'kelly-static-image';
        container2D.appendChild(img);
      }

      img.src = imageUrl;
      img.style.display = 'block';
      container2D.style.display = 'block';

      if (this.videoPlayer) {
        this.videoPlayer.pause();
        this.videoPlayer.style.display = 'none';
      }

      if (this.effectiveMode === CONFIG.MODES.HYBRID) {
        this.showModeToggle();
      }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3D LIVE MODE (Unity WebGL)
    // ═══════════════════════════════════════════════════════════════════

    async load3D() {
      if (this.isUnityLoaded) return Promise.resolve();
      if (this.isUnityLoading) {
        // Wait for existing load
        return new Promise((resolve) => {
          this.on('unityLoaded', resolve);
        });
      }

      this.isUnityLoading = true;
      console.log('[KellyMode] Loading Unity 3D...');

      return new Promise((resolve, reject) => {
        const container = this.unityContainer || document.getElementById('kelly-3d-container');
        if (!container) {
          this.isUnityLoading = false;
          reject(new Error('3D container not found'));
          return;
        }

        // Create iframe for Unity WebGL
        const iframe = document.createElement('iframe');
        iframe.id = 'kelly-unity-iframe';
        iframe.src = CONFIG.UNITY_PATH;
        iframe.style.cssText = 'width:100%;height:100%;border:none;';
        iframe.allow = 'autoplay; fullscreen';

        const timeout = setTimeout(() => {
          this.isUnityLoading = false;
          reject(new Error('Unity load timeout'));
        }, 30000);

        iframe.onload = () => {
          clearTimeout(timeout);
          this.isUnityLoaded = true;
          this.isUnityLoading = false;
          this.unityIframe = iframe;
          console.log('[KellyMode] Unity 3D loaded');
          this.emit('unityLoaded');
          resolve();
        };

        iframe.onerror = (e) => {
          clearTimeout(timeout);
          this.isUnityLoading = false;
          reject(e);
        };

        container.appendChild(iframe);
      });
    }

    show3D() {
      const container = this.unityContainer || document.getElementById('kelly-3d-container');
      if (container) {
        container.style.display = 'block';
      }

      // Hide 2D
      const container2D = document.getElementById('kelly-2d-container');
      if (container2D) {
        container2D.style.display = 'none';
      }

      this.updateToggleButton('2d');
    }

    hide3D() {
      const container = this.unityContainer || document.getElementById('kelly-3d-container');
      if (container) {
        container.style.display = 'none';
      }

      this.updateToggleButton('3d');
    }

    sendToUnity(command, data = {}) {
      if (!this.unityIframe || !this.unityIframe.contentWindow) {
        console.warn('[KellyMode] Unity not loaded, cannot send:', command);
        return false;
      }

      this.unityIframe.contentWindow.postMessage({
        type: 'kelly_command',
        command,
        data,
        timestamp: Date.now(),
      }, '*');

      console.log(`[KellyMode] Sent to Unity: ${command}`, data);
      return true;
    }

    // ═══════════════════════════════════════════════════════════════════
    // HYBRID MODE & TOGGLE
    // ═══════════════════════════════════════════════════════════════════

    showModeToggle() {
      const toggle = document.getElementById('kelly-mode-toggle');
      if (toggle) {
        toggle.style.display = 'block';
      }
    }

    hideModeToggle() {
      const toggle = document.getElementById('kelly-mode-toggle');
      if (toggle) {
        toggle.style.display = 'none';
      }
    }

    updateToggleButton(targetMode) {
      const toggle = document.getElementById('kelly-mode-toggle');
      if (toggle) {
        if (targetMode === '3d') {
          toggle.innerHTML = '🎮 Switch to 3D';
          toggle.setAttribute('data-target', '3d');
        } else {
          toggle.innerHTML = '📺 Switch to 2D';
          toggle.setAttribute('data-target', '2d');
        }
      }
    }

    async toggleMode() {
      const toggle = document.getElementById('kelly-mode-toggle');
      const targetMode = toggle?.getAttribute('data-target') || '3d';

      if (targetMode === '3d') {
        try {
          await this.load3D();
          this.show3D();
          this.emit('modeSwitched', { from: '2d', to: '3d' });
        } catch (e) {
          console.error('[KellyMode] Failed to load 3D:', e);
          this.emit('3dLoadError', { error: e });
        }
      } else {
        this.hide3D();
        const container2D = document.getElementById('kelly-2d-container');
        if (container2D) {
          container2D.style.display = 'block';
        }
        if (this.videoPlayer) {
          this.videoPlayer.style.display = 'block';
        }
        this.emit('modeSwitched', { from: '3d', to: '2d' });
      }
    }

    // ═══════════════════════════════════════════════════════════════════
    // LESSON INTEGRATION
    // ═══════════════════════════════════════════════════════════════════

    async playLessonPhase(phase, dayNumber, options = {}) {
      const dayStr = String(dayNumber).padStart(3, '0');
      
      if (this.effectiveMode === CONFIG.MODES.LIVE_3D) {
        // 3D mode: send to Unity
        await this.load3D();
        this.show3D();
        this.sendToUnity('play_phase', { phase, day: dayNumber });
      } else {
        // 2D mode: play HeyGen video
        const videoUrl = `${CONFIG.VIDEO_BASE_PATH}/day-${dayStr}/${phase}.mp4`;
        await this.play2DVideo(videoUrl, options);
      }
      
      this.emit('phaseStarted', { phase, day: dayNumber, mode: this.effectiveMode });
    }

    setExpression(expression) {
      if (this.effectiveMode === CONFIG.MODES.LIVE_3D && this.isUnityLoaded) {
        this.sendToUnity('set_expression', { expression });
      } else {
        const imageUrl = `${CONFIG.POSES_PATH}/kelly_${expression}.png`;
        this.show2DStatic(imageUrl);
      }
    }

    setPose(pose) {
      if (this.effectiveMode === CONFIG.MODES.LIVE_3D && this.isUnityLoaded) {
        this.sendToUnity('set_pose', { pose });
      } else {
        const imageUrl = `${CONFIG.POSES_PATH}/kelly_${pose}.png`;
        this.show2DStatic(imageUrl);
      }
    }

    // ═══════════════════════════════════════════════════════════════════
    // EVENT SYSTEM
    // ═══════════════════════════════════════════════════════════════════

    on(event, callback) {
      if (!this.callbacks[event]) {
        this.callbacks[event] = [];
      }
      this.callbacks[event].push(callback);
    }

    off(event, callback) {
      if (this.callbacks[event]) {
        this.callbacks[event] = this.callbacks[event].filter(cb => cb !== callback);
      }
    }

    emit(event, data = {}) {
      if (this.callbacks[event]) {
        this.callbacks[event].forEach(cb => {
          try {
            cb(data);
          } catch (e) {
            console.error(`[KellyMode] Event callback error (${event}):`, e);
          }
        });
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // UNITY MESSAGE HANDLER
  // ═══════════════════════════════════════════════════════════════════════

  window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'kelly_event') {
      console.log('[KellyMode] Event from Unity:', event.data);
      
      if (window.kellyMode) {
        window.kellyMode.emit('unityEvent', event.data);
        
        // Handle specific events
        if (event.data.event === 'animation_complete') {
          window.kellyMode.emit('animationComplete', event.data.data);
        }
      }
    }
  });

  // ═══════════════════════════════════════════════════════════════════════
  // GLOBAL INSTANCE
  // ═══════════════════════════════════════════════════════════════════════

  window.KellyModeController = KellyModeController;
  window.kellyMode = new KellyModeController();

  console.log('[KellyMode] Module loaded. Access via window.kellyMode');

})();
