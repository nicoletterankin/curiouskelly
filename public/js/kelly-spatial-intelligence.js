/**
 * KELLY SPATIAL INTELLIGENCE SYSTEM
 * 
 * Purpose: Know where Kelly is. Never cover her face or hands.
 * Philosophy: Clean, professional, invisible. No gimmicks.
 * 
 * Like walking into a light room where Kelly is ready to teach.
 * Confident. Calm. Cool. Prepared. Perfect.
 */

class KellySpatialIntelligence {
  constructor() {
    this.currentPose = 'kelly_hint';
    this.safeZones = [];
    this.blockedZones = [];
    this.debugMode = false;
    
    // Kelly's standard poses and their safe zones
    this.poseDefinitions = {
      'kelly_hint': {
        // Kelly thinking, hand on chin, pointing slightly
        name: 'Hint Pose',
        face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.55, y: 0.45, width: 0.08, height: 0.08, importance: 'high' }
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.20, score: 0.9 },
          { name: 'top-right', x: 0.75, y: 0.05, width: 0.20, height: 0.20, score: 0.8 },
          { name: 'left-mid', x: 0.05, y: 0.40, width: 0.25, height: 0.20, score: 0.7 },
          { name: 'bottom-left', x: 0.05, y: 0.75, width: 0.30, height: 0.20, score: 0.6 }
        ]
      },
      
      'kelly_welcome': {
        // Kelly centered, arms open, welcoming
        name: 'Welcome Pose',
        face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.25, y: 0.45, width: 0.08, height: 0.08, importance: 'high' },
          { x: 0.67, y: 0.45, width: 0.08, height: 0.08, importance: 'high' }
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.15, height: 0.20, score: 0.8 },
          { name: 'top-right', x: 0.80, y: 0.05, width: 0.15, height: 0.20, score: 0.8 },
          { name: 'bottom-center', x: 0.35, y: 0.80, width: 0.30, height: 0.15, score: 0.6 }
        ]
      },
      
      'kelly_choice_left': {
        // Kelly pointing left (learner's left)
        name: 'Pointing Left',
        face: { x: 0.50, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.25, y: 0.50, width: 0.10, height: 0.10, importance: 'critical' } // Pointing hand
        ],
        safeZones: [
          { name: 'top-right', x: 0.70, y: 0.05, width: 0.25, height: 0.30, score: 1.0 }, // Far from pointing
          { name: 'bottom-right', x: 0.70, y: 0.70, width: 0.25, height: 0.25, score: 0.9 }
        ]
      },
      
      'kelly_choice_right': {
        // Kelly pointing right (learner's right)
        name: 'Pointing Right',
        face: { x: 0.50, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.65, y: 0.50, width: 0.10, height: 0.10, importance: 'critical' } // Pointing hand
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.25, height: 0.30, score: 1.0 }, // Far from pointing
          { name: 'bottom-left', x: 0.05, y: 0.70, width: 0.25, height: 0.25, score: 0.9 }
        ]
      },
      
      'kelly_idle': {
        // Kelly standing, hands at sides
        name: 'Idle Pose',
        face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.35, y: 0.60, width: 0.06, height: 0.06, importance: 'low' },
          { x: 0.59, y: 0.60, width: 0.06, height: 0.06, importance: 'low' }
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
          { name: 'top-right', x: 0.65, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
          { name: 'left-mid', x: 0.05, y: 0.40, width: 0.20, height: 0.20, score: 0.8 },
          { name: 'right-mid', x: 0.75, y: 0.40, width: 0.20, height: 0.20, score: 0.8 }
        ]
      },
      
      'kelly_clasp': {
        // Kelly with hands clasped (thinking, listening)
        name: 'Clasp Pose',
        face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.45, y: 0.50, width: 0.10, height: 0.10, importance: 'medium' } // Clasped hands center
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
          { name: 'top-right', x: 0.65, y: 0.05, width: 0.30, height: 0.25, score: 0.9 },
          { name: 'left-mid', x: 0.05, y: 0.40, width: 0.25, height: 0.20, score: 0.7 },
          { name: 'right-mid', x: 0.70, y: 0.40, width: 0.25, height: 0.20, score: 0.7 }
        ]
      },
      
      'kelly_listening': {
        // Kelly in listening pose (hand to ear)
        name: 'Listening Pose',
        face: { x: 0.45, y: 0.15, width: 0.10, height: 0.15 },
        hands: [
          { x: 0.60, y: 0.20, width: 0.08, height: 0.08, importance: 'high' } // Hand near face
        ],
        safeZones: [
          { name: 'top-left', x: 0.05, y: 0.05, width: 0.30, height: 0.20, score: 0.9 },
          { name: 'left-mid', x: 0.05, y: 0.35, width: 0.25, height: 0.30, score: 0.8 },
          { name: 'bottom-left', x: 0.05, y: 0.70, width: 0.30, height: 0.25, score: 0.7 }
        ]
      }
    };
  }
  
  /**
   * Initialize the system
   */
  async init() {
    console.log('[KellySpatial] Initializing spatial intelligence...');
    
    // Detect current pose
    this.detectCurrentPose();
    
    // Calculate safe zones
    this.calculateSafeZones();
    
    // Listen for pose changes
    this.setupPoseChangeListener();
    
    // Debug mode from URL
    if (window.location.search.includes('debug=spatial')) {
      this.enableDebugMode();
    }
    
    console.log('[KellySpatial] ✅ Ready');
  }
  
  /**
   * Detect Kelly's current pose from the avatar image
   */
  detectCurrentPose() {
    const kellyAvatar = document.getElementById('kelly-avatar');
    if (!kellyAvatar) {
      console.warn('[KellySpatial] Kelly avatar not found, using default pose');
      this.currentPose = 'kelly_hint';
      return;
    }
    
    const src = kellyAvatar.src || '';
    
    // Extract pose from filename
    if (src.includes('kelly_choice_left')) {
      this.currentPose = 'kelly_choice_left';
    } else if (src.includes('kelly_choice_right')) {
      this.currentPose = 'kelly_choice_right';
    } else if (src.includes('kelly_welcome')) {
      this.currentPose = 'kelly_welcome';
    } else if (src.includes('kelly_idle')) {
      this.currentPose = 'kelly_idle';
    } else if (src.includes('kelly_clasp')) {
      this.currentPose = 'kelly_clasp';
    } else if (src.includes('kelly_listening')) {
      this.currentPose = 'kelly_listening';
    } else if (src.includes('kelly_hint')) {
      this.currentPose = 'kelly_hint';
    } else {
      this.currentPose = 'kelly_hint'; // Default
    }
    
    console.log(`[KellySpatial] Detected pose: ${this.currentPose}`);
  }
  
  /**
   * Calculate safe zones based on current pose
   */
  calculateSafeZones() {
    const pose = this.poseDefinitions[this.currentPose];
    if (!pose) {
      console.warn(`[KellySpatial] Unknown pose: ${this.currentPose}`);
      this.safeZones = [];
      this.blockedZones = [];
      return;
    }
    
    // Blocked zones = face + hands
    this.blockedZones = [pose.face, ...pose.hands];
    
    // Safe zones from definition
    this.safeZones = pose.safeZones.map(zone => ({
      ...zone,
      blocked: false
    }));
    
    console.log(`[KellySpatial] Calculated ${this.safeZones.length} safe zones for ${pose.name}`);
  }
  
  /**
   * Find the best safe zone for a popover
   * @param {string} preferredPosition - 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'left-mid', 'right-mid'
   * @param {object} popoverSize - { width, height } in pixels
   * @returns {object} - { x, y, name, score }
   */
  findBestSafeZone(preferredPosition = 'top-right', popoverSize = {}) {
    if (this.safeZones.length === 0) {
      console.warn('[KellySpatial] No safe zones available, using fallback');
      return this.getFallbackPosition();
    }
    
    // Try preferred position first
    let bestZone = this.safeZones.find(zone => zone.name === preferredPosition);
    
    // If not available, find highest-scoring zone
    if (!bestZone) {
      bestZone = this.safeZones.reduce((best, zone) => 
        zone.score > best.score ? zone : best
      );
    }
    
    return {
      x: bestZone.x,
      y: bestZone.y,
      width: bestZone.width,
      height: bestZone.height,
      name: bestZone.name,
      score: bestZone.score
    };
  }
  
  /**
   * Get fallback position (far right edge)
   */
  getFallbackPosition() {
    return {
      x: 0.75,
      y: 0.50,
      width: 0.20,
      height: 0.30,
      name: 'fallback-right',
      score: 0.5
    };
  }
  
  /**
   * Position a popover element in a safe zone
   * @param {HTMLElement} popover - The popover element
   * @param {string} preferredPosition - Preferred safe zone
   */
  positionPopover(popover, preferredPosition = 'top-right') {
    if (!popover) return;
    
    const kellyFrame = document.getElementById('kelly-frame');
    if (!kellyFrame) {
      console.warn('[KellySpatial] Kelly frame not found');
      return;
    }
    
    const frameRect = kellyFrame.getBoundingClientRect();
    const popoverSize = {
      width: popover.offsetWidth,
      height: popover.offsetHeight
    };
    
    // Find best safe zone
    const safeZone = this.findBestSafeZone(preferredPosition, popoverSize);
    
    // Convert normalized coordinates to pixels
    const left = frameRect.width * safeZone.x;
    const top = frameRect.height * safeZone.y;
    
    // Apply position
    popover.style.position = 'absolute';
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    popover.style.maxWidth = `${frameRect.width * safeZone.width}px`;
    popover.style.maxHeight = `${frameRect.height * safeZone.height}px`;
    
    // Store zone info
    popover.dataset.safeZone = safeZone.name;
    popover.dataset.safeZoneScore = safeZone.score;
    
    console.log(`[KellySpatial] Positioned popover in "${safeZone.name}" (score: ${safeZone.score})`);
  }
  
  /**
   * Check if a point is in a blocked zone (face or hands)
   * @param {number} x - Normalized x coordinate (0-1)
   * @param {number} y - Normalized y coordinate (0-1)
   * @returns {boolean}
   */
  isPointBlocked(x, y) {
    return this.blockedZones.some(zone => {
      return x >= zone.x && x <= (zone.x + zone.width) &&
             y >= zone.y && y <= (zone.y + zone.height);
    });
  }
  
  /**
   * Get blocked zones (for highlighting Kelly's face/hands)
   */
  getBlockedZones() {
    return this.blockedZones;
  }
  
  /**
   * Get safe zones (for popover positioning)
   */
  getSafeZones() {
    return this.safeZones;
  }
  
  /**
   * Setup listener for pose changes
   */
  setupPoseChangeListener() {
    // Listen for kelly-pose-changed event
    window.addEventListener('kelly-pose-changed', (event) => {
      const newPose = event.detail?.pose || 'kelly_hint';
      console.log(`[KellySpatial] Pose changed to: ${newPose}`);
      this.currentPose = newPose;
      this.calculateSafeZones();
      
      // Reposition any open popovers
      this.repositionOpenPopovers();
      
      // Update debug visualization if enabled
      if (this.debugMode) {
        this.visualizeSafeZones();
      }
    });
    
    // Also watch for avatar src changes
    const kellyAvatar = document.getElementById('kelly-avatar');
    if (kellyAvatar) {
      const observer = new MutationObserver(() => {
        const oldPose = this.currentPose;
        this.detectCurrentPose();
        if (oldPose !== this.currentPose) {
          this.calculateSafeZones();
          this.repositionOpenPopovers();
          if (this.debugMode) {
            this.visualizeSafeZones();
          }
        }
      });
      
      observer.observe(kellyAvatar, { attributes: true, attributeFilter: ['src'] });
    }
  }
  
  /**
   * Reposition any open popovers after pose change
   */
  repositionOpenPopovers() {
    document.querySelectorAll('.expand-panel.open, .overlay-panel.open').forEach(popover => {
      const preferredPosition = popover.dataset.preferredPosition || 'top-right';
      this.positionPopover(popover, preferredPosition);
    });
  }
  
  /**
   * Enable debug mode (visualize safe zones)
   */
  enableDebugMode() {
    this.debugMode = true;
    console.log('[KellySpatial] 🐛 Debug mode enabled');
    this.visualizeSafeZones();
    
    // Add toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = '👁️ Toggle Safe Zones';
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
      const existing = document.querySelectorAll('.kelly-spatial-debug');
      if (existing.length > 0) {
        existing.forEach(el => el.remove());
      } else {
        this.visualizeSafeZones();
      }
    });
    document.body.appendChild(toggleBtn);
  }
  
  /**
   * Visualize safe zones and blocked zones (debug mode)
   */
  visualizeSafeZones() {
    // Remove old visualizations
    document.querySelectorAll('.kelly-spatial-debug').forEach(el => el.remove());
    
    const kellyFrame = document.getElementById('kelly-frame');
    if (!kellyFrame) return;
    
    const frameRect = kellyFrame.getBoundingClientRect();
    
    // Visualize blocked zones (face + hands) in RED
    this.blockedZones.forEach((zone, index) => {
      const div = document.createElement('div');
      div.className = 'kelly-spatial-debug kelly-blocked-zone';
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
      kellyFrame.appendChild(div);
    });
    
    // Visualize safe zones in GREEN
    this.safeZones.forEach(zone => {
      const div = document.createElement('div');
      div.className = 'kelly-spatial-debug kelly-safe-zone';
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
      kellyFrame.appendChild(div);
    });
    
    console.log('[KellySpatial] 🐛 Visualized zones');
  }
}

// Global instance
window.kellySpatial = new KellySpatialIntelligence();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.kellySpatial.init();
  });
} else {
  window.kellySpatial.init();
}






