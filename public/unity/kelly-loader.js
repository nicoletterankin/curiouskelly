/**
 * Kelly 3D Engine Loader — Progressive Enhancement
 * 
 * Detects browser capabilities and loads the best available Kelly build:
 * - WebGPU (when available & Unity supports it)
 * - WebGL 2.0 (current production)
 * - WebGL 1.0 (legacy fallback)
 * - Static fallback (last resort)
 * 
 * Created: December 16, 2025
 * See: docs/KELLY_3D_FUTURE_PROOFING.md
 */

(function() {
  'use strict';

  const KellyLoader = {
    // Build paths — update when new builds are available
    builds: {
      webgpu: '/unity/kelly-webgpu/index.html',   // Future
      webgl2: '/unity/kelly-chair/index.html',    // Current production
      webgl1: '/unity/kelly-v1/index.html',       // Legacy
      static: '/lessons/images/kelly-directors-chair-curious.png' // Last resort
    },

    // Detected capabilities (populated on init)
    capabilities: {
      webgpu: false,
      webgl2: false,
      webgl1: false,
      offscreenCanvas: false,
      sharedArrayBuffer: false,
      webAssembly: true
    },

    // Current state
    currentBuild: null,
    iframe: null,
    container: null,

    /**
     * Initialize and detect capabilities
     */
    async init() {
      this.detectCapabilities();
      console.log('[KellyLoader] Capabilities:', this.capabilities);
      return this.capabilities;
    },

    /**
     * Detect browser 3D capabilities
     */
    detectCapabilities() {
      // WebGPU detection
      this.capabilities.webgpu = 'gpu' in navigator;

      // WebGL 2.0 detection
      try {
        const canvas = document.createElement('canvas');
        this.capabilities.webgl2 = !!(
          canvas.getContext('webgl2') || 
          canvas.getContext('experimental-webgl2')
        );
      } catch (e) {
        this.capabilities.webgl2 = false;
      }

      // WebGL 1.0 detection
      try {
        const canvas = document.createElement('canvas');
        this.capabilities.webgl1 = !!(
          canvas.getContext('webgl') || 
          canvas.getContext('experimental-webgl')
        );
      } catch (e) {
        this.capabilities.webgl1 = false;
      }

      // OffscreenCanvas (helps with threading)
      this.capabilities.offscreenCanvas = 'OffscreenCanvas' in window;

      // SharedArrayBuffer (needed for Unity threading)
      this.capabilities.sharedArrayBuffer = 'SharedArrayBuffer' in window;

      // WebAssembly
      this.capabilities.webAssembly = 'WebAssembly' in window;
    },

    /**
     * Check if WebGPU is fully functional (not just present)
     */
    async checkWebGPU() {
      if (!this.capabilities.webgpu) return false;

      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return false;

        const device = await adapter.requestDevice();
        if (!device) return false;

        // WebGPU is fully functional
        console.log('[KellyLoader] WebGPU adapter:', adapter.name || 'available');
        return true;
      } catch (e) {
        console.log('[KellyLoader] WebGPU check failed:', e.message);
        return false;
      }
    },

    /**
     * Get the best available build for this browser
     */
    async getBestBuild() {
      // Future: Check for WebGPU when Unity supports it
      // const webgpuReady = await this.checkWebGPU();
      // if (webgpuReady && this.builds.webgpu) {
      //   return { type: 'webgpu', path: this.builds.webgpu };
      // }

      // Current: WebGL 2.0 is the production choice
      if (this.capabilities.webgl2) {
        return { type: 'webgl2', path: this.builds.webgl2 };
      }

      // Legacy: WebGL 1.0
      if (this.capabilities.webgl1) {
        return { type: 'webgl1', path: this.builds.webgl1 };
      }

      // Last resort: static image
      return { type: 'static', path: this.builds.static };
    },

    /**
     * Load Kelly into a container element
     * @param {string|HTMLElement} containerSelector - Container element or selector
     * @param {object} options - Loading options
     */
    async load(containerSelector, options = {}) {
      // Get container
      this.container = typeof containerSelector === 'string' 
        ? document.querySelector(containerSelector)
        : containerSelector;

      if (!this.container) {
        console.error('[KellyLoader] Container not found:', containerSelector);
        return null;
      }

      // Initialize if not done
      if (!this.capabilities.webgl1 && !this.capabilities.webgl2) {
        await this.init();
      }

      // Get best build
      const build = await this.getBestBuild();
      console.log('[KellyLoader] Selected build:', build.type, build.path);
      this.currentBuild = build;

      // Load based on type
      if (build.type === 'static') {
        return this.loadStaticFallback(build.path, options);
      } else {
        return this.loadUnityBuild(build.path, options);
      }
    },

    /**
     * Load Unity WebGL/WebGPU build in iframe
     */
    loadUnityBuild(path, options = {}) {
      // Remove existing iframe if present
      if (this.iframe) {
        this.iframe.remove();
      }

      // Create iframe
      this.iframe = document.createElement('iframe');
      this.iframe.id = options.id || 'kelly-unity-frame';
      this.iframe.src = path;
      this.iframe.allow = 'autoplay; fullscreen; xr-spatial-tracking';
      this.iframe.style.cssText = options.style || 'width:100%;height:100%;border:none;';
      
      // Add to container
      this.container.appendChild(this.iframe);

      // Setup message bridge
      this.setupBridge();

      return this.iframe;
    },

    /**
     * Load static image fallback
     */
    loadStaticFallback(path, options = {}) {
      const img = document.createElement('img');
      img.id = options.id || 'kelly-static-fallback';
      img.src = path;
      img.alt = 'Kelly';
      img.style.cssText = options.style || 'width:100%;height:auto;';
      
      this.container.appendChild(img);

      // Emit ready event for consistency
      window.dispatchEvent(new CustomEvent('kelly-ready', {
        detail: { type: 'static', element: img }
      }));

      return img;
    },

    /**
     * Setup postMessage bridge with Unity
     */
    setupBridge() {
      window.addEventListener('message', (event) => {
        if (!event.data || event.data.source !== 'kelly-webgl') return;

        const { type, status, lessonId, message } = event.data;

        // Re-emit as custom events for easier handling
        window.dispatchEvent(new CustomEvent(type, {
          detail: { status, lessonId, message, buildType: this.currentBuild?.type }
        }));

        console.log('[KellyLoader] Event:', type, event.data);
      });
    },

    /**
     * Send command to Kelly
     */
    sendCommand(type, payload = {}) {
      if (!this.iframe?.contentWindow) {
        console.warn('[KellyLoader] Cannot send command, no iframe loaded');
        return;
      }

      this.iframe.contentWindow.postMessage({
        destination: 'kelly-webgl',
        type,
        payload
      }, '*');
    },

    /**
     * Convenience methods for common commands
     */
    play(lessonId) {
      this.sendCommand('kelly-load', { lessonId });
    },

    stop() {
      this.sendCommand('kelly-stop', {});
    },

    ping() {
      this.sendCommand('kelly-ping', {});
    },

    /**
     * Get diagnostic info
     */
    getDiagnostics() {
      return {
        capabilities: this.capabilities,
        currentBuild: this.currentBuild,
        hasIframe: !!this.iframe,
        iframeSrc: this.iframe?.src || null,
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        timestamp: new Date().toISOString()
      };
    }
  };

  // Export to window
  window.KellyLoader = KellyLoader;

  // Auto-init on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyLoader.init());
  } else {
    KellyLoader.init();
  }
})();
