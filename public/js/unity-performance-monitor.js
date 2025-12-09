/**
 * Unity Performance Monitor
 * Tracks Unity WebGL load times and reports metrics
 * 
 * @usage
 * const monitor = new UnityPerformanceMonitor();
 * monitor.startTracking();
 * // ... Unity loads ...
 * monitor.recordLoaded();
 * console.log(monitor.getMetrics());
 */

class UnityPerformanceMonitor {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.reportEndpoint = options.reportEndpoint || null;
    this.debug = options.debug || false;
    
    this.metrics = {
      // Timing metrics
      startTime: null,
      loaderLoadedTime: null,
      firstProgressTime: null,
      fullyLoadedTime: null,
      
      // Progress tracking
      progressSteps: [],
      
      // Source tracking
      source: null,
      fallbackAttempts: 0,
      
      // Error tracking
      errors: [],
      
      // Device info
      deviceInfo: this._getDeviceInfo(),
    };
    
    this._log('UnityPerformanceMonitor initialized');
  }
  
  /**
   * Start tracking Unity load
   */
  startTracking() {
    if (!this.enabled) return;
    
    this.metrics.startTime = performance.now();
    this._log('Tracking started');
    
    // Track page visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this._recordEvent('page_hidden');
      } else {
        this._recordEvent('page_visible');
      }
    });
  }
  
  /**
   * Record when loader script is loaded
   */
  recordLoaderLoaded(source) {
    if (!this.enabled) return;
    
    this.metrics.loaderLoadedTime = performance.now();
    this.metrics.source = source;
    this._log(`Loader loaded from ${source} in ${this._elapsed(this.metrics.loaderLoadedTime)}ms`);
  }
  
  /**
   * Record Unity progress updates
   */
  recordProgress(progress) {
    if (!this.enabled) return;
    
    const now = performance.now();
    
    if (!this.metrics.firstProgressTime && progress > 0) {
      this.metrics.firstProgressTime = now;
      this._log(`First progress at ${this._elapsed(now)}ms`);
    }
    
    this.metrics.progressSteps.push({
      progress,
      time: now,
      elapsed: this._elapsed(now),
    });
    
    if (this.debug && progress % 0.1 < 0.01) {
      this._log(`Progress: ${Math.round(progress * 100)}% at ${this._elapsed(now)}ms`);
    }
  }
  
  /**
   * Record when Unity is fully loaded
   */
  recordLoaded() {
    if (!this.enabled) return;
    
    this.metrics.fullyLoadedTime = performance.now();
    const totalTime = this._elapsed(this.metrics.fullyLoadedTime);
    
    this._log(`Unity fully loaded in ${totalTime}ms`);
    
    // Send metrics if endpoint configured
    if (this.reportEndpoint) {
      this._sendMetrics();
    }
    
    return this.getMetrics();
  }
  
  /**
   * Record a fallback attempt
   */
  recordFallback(fromSource, toSource, error) {
    if (!this.enabled) return;
    
    this.metrics.fallbackAttempts++;
    this._recordEvent('fallback', {
      from: fromSource,
      to: toSource,
      error: error?.message || 'Unknown error',
    });
    
    this._log(`Fallback from ${fromSource} to ${toSource}: ${error?.message}`);
  }
  
  /**
   * Record an error
   */
  recordError(error, context = {}) {
    if (!this.enabled) return;
    
    this.metrics.errors.push({
      time: performance.now(),
      elapsed: this._elapsed(),
      message: error?.message || String(error),
      context,
    });
    
    this._log(`Error: ${error?.message}`, 'error');
  }
  
  /**
   * Get all metrics
   */
  getMetrics() {
    const now = performance.now();
    
    return {
      // Timing
      totalLoadTime: this.metrics.fullyLoadedTime 
        ? this._elapsed(this.metrics.fullyLoadedTime)
        : this._elapsed(),
      loaderTime: this.metrics.loaderLoadedTime 
        ? this._elapsed(this.metrics.loaderLoadedTime) 
        : null,
      initTime: this.metrics.firstProgressTime && this.metrics.loaderLoadedTime
        ? this.metrics.firstProgressTime - this.metrics.loaderLoadedTime
        : null,
      downloadTime: this.metrics.fullyLoadedTime && this.metrics.firstProgressTime
        ? this.metrics.fullyLoadedTime - this.metrics.firstProgressTime
        : null,
        
      // Source
      source: this.metrics.source,
      fallbackAttempts: this.metrics.fallbackAttempts,
      
      // Status
      isLoaded: !!this.metrics.fullyLoadedTime,
      hasErrors: this.metrics.errors.length > 0,
      errorCount: this.metrics.errors.length,
      
      // Progress details
      progressSteps: this.metrics.progressSteps.length,
      
      // Device
      ...this.metrics.deviceInfo,
      
      // Raw data for detailed analysis
      raw: this.enabled ? {
        startTime: this.metrics.startTime,
        loaderLoadedTime: this.metrics.loaderLoadedTime,
        firstProgressTime: this.metrics.firstProgressTime,
        fullyLoadedTime: this.metrics.fullyLoadedTime,
        errors: this.metrics.errors,
        progressSteps: this.metrics.progressSteps,
      } : null,
    };
  }
  
  /**
   * Get device info
   */
  _getDeviceInfo() {
    const ua = navigator.userAgent;
    const gl = this._getWebGLInfo();
    
    return {
      // Browser
      browser: this._detectBrowser(ua),
      isMobile: /iPhone|iPad|Android|Mobile/i.test(ua),
      
      // Screen
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
      devicePixelRatio: window.devicePixelRatio || 1,
      
      // WebGL
      webglVersion: gl.version,
      webglRenderer: gl.renderer,
      webglVendor: gl.vendor,
      
      // Memory (if available)
      deviceMemory: navigator.deviceMemory || null,
      hardwareConcurrency: navigator.hardwareConcurrency || null,
      
      // Connection (if available)
      connectionType: navigator.connection?.effectiveType || null,
      connectionDownlink: navigator.connection?.downlink || null,
    };
  }
  
  /**
   * Detect browser from UA
   */
  _detectBrowser(ua) {
    if (ua.includes('Chrome')) return 'Chrome';
    if (ua.includes('Firefox')) return 'Firefox';
    if (ua.includes('Safari')) return 'Safari';
    if (ua.includes('Edge')) return 'Edge';
    return 'Unknown';
  }
  
  /**
   * Get WebGL info
   */
  _getWebGLInfo() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!gl) {
        return { version: 'none', renderer: 'N/A', vendor: 'N/A' };
      }
      
      const version = gl.getParameter(gl.VERSION);
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      
      return {
        version: version?.includes('2.0') ? 'WebGL 2' : 'WebGL 1',
        renderer: debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'Unknown',
        vendor: debugInfo ? gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 'Unknown',
      };
    } catch (e) {
      return { version: 'error', renderer: 'N/A', vendor: 'N/A' };
    }
  }
  
  /**
   * Record an event with timestamp
   */
  _recordEvent(name, data = {}) {
    this.metrics.progressSteps.push({
      event: name,
      time: performance.now(),
      elapsed: this._elapsed(),
      ...data,
    });
  }
  
  /**
   * Calculate elapsed time
   */
  _elapsed(until = performance.now()) {
    if (!this.metrics.startTime) return 0;
    return Math.round(until - this.metrics.startTime);
  }
  
  /**
   * Log message
   */
  _log(message, level = 'log') {
    if (this.debug) {
      console[level](`[UnityPerf] ${message}`);
    }
  }
  
  /**
   * Send metrics to reporting endpoint
   */
  async _sendMetrics() {
    if (!this.reportEndpoint) return;
    
    try {
      const metrics = this.getMetrics();
      
      await fetch(this.reportEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: new Date().toISOString(),
          url: window.location.href,
          ...metrics,
        }),
        keepalive: true, // Ensure request completes even if page unloads
      });
      
      this._log('Metrics sent to reporting endpoint');
    } catch (error) {
      this._log(`Failed to send metrics: ${error.message}`, 'warn');
    }
  }
  
  /**
   * Generate a performance report for console
   */
  printReport() {
    const metrics = this.getMetrics();
    
    console.group('🎮 Unity Performance Report');
    console.log(`Total Load Time: ${metrics.totalLoadTime}ms`);
    console.log(`Source: ${metrics.source}`);
    console.log(`Fallback Attempts: ${metrics.fallbackAttempts}`);
    console.log(`Errors: ${metrics.errorCount}`);
    console.log('---');
    console.log(`Browser: ${metrics.browser}`);
    console.log(`WebGL: ${metrics.webglVersion}`);
    console.log(`GPU: ${metrics.webglRenderer}`);
    console.log(`Connection: ${metrics.connectionType || 'Unknown'}`);
    console.groupEnd();
    
    return metrics;
  }
}

// Export for use in other scripts
window.UnityPerformanceMonitor = UnityPerformanceMonitor;

// Create global instance
window.unityPerfMonitor = new UnityPerformanceMonitor({
  debug: window.location.hostname === 'localhost',
});

if (typeof module !== 'undefined' && module.exports) {
  module.exports = UnityPerformanceMonitor;
}



