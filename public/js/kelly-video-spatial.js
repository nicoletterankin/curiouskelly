/**
 * KELLY VIDEO SPATIAL INTELLIGENCE
 * 
 * For HD lipsync videos with pre-computed safe zone manifests.
 * Syncs safe zones to video playback in real-time.
 * 
 * Like iPhone lock screen: Kelly is the wallpaper, UI floats on top.
 */

class KellyVideoSpatial {
  constructor() {
    this.videoElement = null;
    this.manifest = null;
    this.currentSafeZones = [];
    this.currentBlockedZones = [];
    this.lastUpdateTime = -1;
    this.rafId = null;
    this.debugMode = false;
  }
  
  /**
   * Initialize with video element and manifest
   * @param {HTMLVideoElement} videoElement - The Kelly video
   * @param {string} manifestUrl - URL to safe zone manifest JSON
   */
  async init(videoElement, manifestUrl) {
    this.videoElement = videoElement;
    
    // Load manifest
    try {
      const response = await fetch(manifestUrl);
      this.manifest = await response.json();
      console.log('[KellyVideoSpatial] ✅ Manifest loaded:', this.manifest.video_id);
    } catch (error) {
      console.error('[KellyVideoSpatial] ❌ Failed to load manifest:', error);
      this.manifest = this.getDefaultManifest();
    }
    
    // Start syncing
    this.startSync();
    
    // Debug mode
    if (window.location.search.includes('debug=spatial')) {
      this.enableDebugMode();
    }
  }
  
  /**
   * Start syncing safe zones to video playback
   */
  startSync() {
    const update = () => {
      if (!this.videoElement || !this.manifest) return;
      
      const currentTime = this.videoElement.currentTime;
      
      // Only update if time changed (avoid redundant calculations)
      if (Math.abs(currentTime - this.lastUpdateTime) > 0.016) { // ~60 FPS
        this.updateSafeZones(currentTime);
        this.lastUpdateTime = currentTime;
        
        // Dispatch event for UI to reposition
        window.dispatchEvent(new CustomEvent('kelly-safe-zones-updated', {
          detail: {
            time: currentTime,
            safeZones: this.currentSafeZones,
            blockedZones: this.currentBlockedZones
          }
        }));
      }
      
      this.rafId = requestAnimationFrame(update);
    };
    
    this.rafId = requestAnimationFrame(update);
    console.log('[KellyVideoSpatial] 🎬 Sync started');
  }
  
  /**
   * Stop syncing
   */
  stopSync() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
      console.log('[KellyVideoSpatial] ⏸️ Sync stopped');
    }
  }
  
  /**
   * Update safe zones based on current video time
   * @param {number} currentTime - Video playback time in seconds
   */
  updateSafeZones(currentTime) {
    if (!this.manifest || !this.manifest.safe_zones) return;
    
    // Find the zone segment for current time
    const segment = this.manifest.safe_zones.find(zone => 
      currentTime >= zone.time_start && currentTime < zone.time_end
    );
    
    if (!segment) {
      // No segment found, use default
      this.currentSafeZones = this.getDefaultSafeZones();
      this.currentBlockedZones = this.getDefaultBlockedZones();
      return;
    }
    
    // Update blocked zones (face + hands)
    this.currentBlockedZones = [segment.kelly_face];
    if (segment.kelly_hands) {
      this.currentBlockedZones.push(...segment.kelly_hands);
    }
    
    // Update safe zones
    this.currentSafeZones = segment.safe_zones || [];
    
    // Update debug visualization if enabled
    if (this.debugMode) {
      this.visualizeSafeZones();
    }
  }
  
  /**
   * Get current safe zones
   */
  getSafeZones() {
    return this.currentSafeZones;
  }
  
  /**
   * Get current blocked zones
   */
  getBlockedZones() {
    return this.currentBlockedZones;
  }
  
  /**
   * Find best safe zone for a popover
   * @param {string} preferredPosition - Preferred zone name
   * @returns {object} - Best safe zone
   */
  findBestSafeZone(preferredPosition = 'top-right') {
    if (this.currentSafeZones.length === 0) {
      return this.getFallbackPosition();
    }
    
    // Try preferred position first
    let bestZone = this.currentSafeZones.find(zone => zone.name === preferredPosition);
    
    // If not available, find highest-scoring zone
    if (!bestZone) {
      bestZone = this.currentSafeZones.reduce((best, zone) => 
        zone.score > best.score ? zone : best
      );
    }
    
    return bestZone;
  }
  
  /**
   * Position a popover element in a safe zone
   * @param {HTMLElement} popover - The popover element
   * @param {string} preferredPosition - Preferred safe zone
   */
  positionPopover(popover, preferredPosition = 'top-right') {
    if (!popover) return;
    
    const container = this.videoElement?.parentElement;
    if (!container) return;
    
    const containerRect = container.getBoundingClientRect();
    const safeZone = this.findBestSafeZone(preferredPosition);
    
    // Convert normalized coordinates to pixels
    const left = containerRect.width * safeZone.x;
    const top = containerRect.height * safeZone.y;
    
    // Apply position
    popover.style.position = 'absolute';
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    popover.style.maxWidth = `${containerRect.width * safeZone.width}px`;
    popover.style.maxHeight = `${containerRect.height * safeZone.height}px`;
    
    // Store zone info
    popover.dataset.safeZone = safeZone.name;
    popover.dataset.safeZoneScore = safeZone.score;
  }
  
  /**
   * Get default manifest (fallback if load fails)
   */
  getDefaultManifest() {
    return {
      video_id: 'default',
      duration: 30,
      fps: 30,
      safe_zones: [
        {
          time_start: 0,
          time_end: 999,
          kelly_face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
          kelly_hands: [{ x: 0.50, y: 0.50, width: 0.08, height: 0.08 }],
          safe_zones: [
            { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
            { name: 'top-right', x: 0.65, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
            { name: 'bottom-left', x: 0.05, y: 0.75, width: 0.30, height: 0.20, score: 0.8 }
          ]
        }
      ]
    };
  }
  
  /**
   * Get default safe zones
   */
  getDefaultSafeZones() {
    return [
      { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
      { name: 'top-right', x: 0.65, y: 0.05, width: 0.30, height: 0.25, score: 0.9 }
    ];
  }
  
  /**
   * Get default blocked zones
   */
  getDefaultBlockedZones() {
    return [
      { x: 0.45, y: 0.15, width: 0.10, height: 0.15 } // Face
    ];
  }
  
  /**
   * Get fallback position
   */
  getFallbackPosition() {
    return {
      name: 'fallback-right',
      x: 0.75,
      y: 0.50,
      width: 0.20,
      height: 0.30,
      score: 0.5
    };
  }
  
  /**
   * Enable debug mode (visualize safe zones)
   */
  enableDebugMode() {
    this.debugMode = true;
    console.log('[KellyVideoSpatial] 🐛 Debug mode enabled');
    
    // Add toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = '👁️ Safe Zones';
    toggleBtn.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      z-index: 10000;
      padding: 8px 12px;
      background: #3B82F6;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    `;
    toggleBtn.addEventListener('click', () => {
      const existing = document.querySelectorAll('.kelly-video-spatial-debug');
      if (existing.length > 0) {
        existing.forEach(el => el.remove());
        this.debugMode = false;
      } else {
        this.debugMode = true;
        this.visualizeSafeZones();
      }
    });
    document.body.appendChild(toggleBtn);
  }
  
  /**
   * Visualize safe zones (debug mode)
   */
  visualizeSafeZones() {
    // Remove old visualizations
    document.querySelectorAll('.kelly-video-spatial-debug').forEach(el => el.remove());
    
    const container = this.videoElement?.parentElement;
    if (!container) return;
    
    // Visualize blocked zones (RED)
    this.currentBlockedZones.forEach((zone, index) => {
      const div = document.createElement('div');
      div.className = 'kelly-video-spatial-debug kelly-blocked-zone';
      div.style.cssText = `
        position: absolute;
        left: ${zone.x * 100}%;
        top: ${zone.y * 100}%;
        width: ${zone.width * 100}%;
        height: ${zone.height * 100}%;
        border: 2px solid #EF4444;
        background: rgba(239, 68, 68, 0.15);
        pointer-events: none;
        z-index: 9998;
        font-size: 10px;
        color: #EF4444;
        padding: 4px;
        font-weight: 700;
      `;
      div.textContent = index === 0 ? '❌ FACE' : `❌ HAND ${index}`;
      container.appendChild(div);
    });
    
    // Visualize safe zones (GREEN)
    this.currentSafeZones.forEach(zone => {
      const div = document.createElement('div');
      div.className = 'kelly-video-spatial-debug kelly-safe-zone';
      div.style.cssText = `
        position: absolute;
        left: ${zone.x * 100}%;
        top: ${zone.y * 100}%;
        width: ${zone.width * 100}%;
        height: ${zone.height * 100}%;
        border: 2px dashed #10B981;
        background: rgba(16, 185, 129, 0.1);
        pointer-events: none;
        z-index: 9999;
        font-size: 10px;
        color: #10B981;
        padding: 4px;
        font-weight: 700;
      `;
      div.innerHTML = `✅ ${zone.name}<br>Score: ${zone.score}`;
      container.appendChild(div);
    });
  }
  
  /**
   * Destroy instance
   */
  destroy() {
    this.stopSync();
    this.videoElement = null;
    this.manifest = null;
    this.currentSafeZones = [];
    this.currentBlockedZones = [];
  }
}

// Global instance
window.kellyVideoSpatial = new KellyVideoSpatial();

// Auto-initialize when video is ready
document.addEventListener('DOMContentLoaded', () => {
  const kellyVideo = document.getElementById('kelly-video');
  if (kellyVideo) {
    // Wait for video metadata
    kellyVideo.addEventListener('loadedmetadata', async () => {
      const videoId = kellyVideo.dataset.videoId || 'day-001-phase-01';
      const manifestUrl = `/kelly/videos/${videoId}-safe-zones.json`;
      await window.kellyVideoSpatial.init(kellyVideo, manifestUrl);
    });
  }
});







