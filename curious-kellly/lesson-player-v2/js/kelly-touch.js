/**
 * Kelly Touch Module
 * 
 * Provides optimized touch interactions for mobile devices.
 * Handles gestures, touch targets, and haptic feedback.
 * 
 * Features:
 * - Swipe gestures (left/right for phase navigation)
 * - Tap to play/pause
 * - Long press for options
 * - Double tap to replay
 * - Touch target sizing
 * - Haptic feedback (where supported)
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const SWIPE_THRESHOLD = 80;  // Minimum distance for swipe
const SWIPE_VELOCITY = 0.3;  // Minimum velocity (px/ms)
const LONG_PRESS_DURATION = 500;  // ms
const DOUBLE_TAP_DELAY = 300;  // ms

// =============================================================================
// TOUCH HANDLER CLASS
// =============================================================================

class KellyTouch {
  constructor(options = {}) {
    this.options = {
      enableSwipe: true,
      enableLongPress: true,
      enableDoubleTap: true,
      enableHaptics: true,
      minTouchTargetSize: 44,  // WCAG minimum
      ...options
    };
    
    // State
    this.touchStart = null;
    this.touchStartTime = null;
    this.lastTap = null;
    this.longPressTimer = null;
    this.isSwiping = false;
    
    // Reference to KellyOS
    this.kellyOS = null;
    
    this.init();
  }
  
  // ---------------------------------------------------------------------------
  // INITIALIZATION
  // ---------------------------------------------------------------------------
  
  init() {
    this.setupTouchListeners();
    this.enforceTouchTargetSizes();
    this.injectStyles();
    
    console.log('[KellyTouch] Touch module initialized');
  }
  
  setKellyOS(kellyOS) {
    this.kellyOS = kellyOS;
  }
  
  setupTouchListeners() {
    // Main lesson area
    const lessonArea = document.getElementById('mode-lesson') || document.body;
    
    lessonArea.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: true });
    lessonArea.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
    lessonArea.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: true });
    lessonArea.addEventListener('touchcancel', () => this.resetTouch(), { passive: true });
    
    // Prevent pull-to-refresh
    document.body.addEventListener('touchmove', (e) => {
      if (document.body.classList.contains('ui-active')) {
        e.preventDefault();
      }
    }, { passive: false });
    
    // Avatar tap
    const avatarArea = document.getElementById('avatar-container');
    if (avatarArea) {
      avatarArea.addEventListener('touchend', (e) => this.handleAvatarTap(e));
    }
  }
  
  // ---------------------------------------------------------------------------
  // TOUCH HANDLERS
  // ---------------------------------------------------------------------------
  
  handleTouchStart(e) {
    const touch = e.touches[0];
    
    this.touchStart = {
      x: touch.clientX,
      y: touch.clientY
    };
    this.touchStartTime = Date.now();
    this.isSwiping = false;
    
    // Start long press timer
    if (this.options.enableLongPress) {
      this.longPressTimer = setTimeout(() => {
        this.handleLongPress(touch);
      }, LONG_PRESS_DURATION);
    }
  }
  
  handleTouchMove(e) {
    if (!this.touchStart) return;
    
    const touch = e.touches[0];
    const deltaX = touch.clientX - this.touchStart.x;
    const deltaY = touch.clientY - this.touchStart.y;
    
    // Cancel long press if moving
    if (Math.abs(deltaX) > 10 || Math.abs(deltaY) > 10) {
      this.cancelLongPress();
    }
    
    // Check if horizontal swipe
    if (this.options.enableSwipe && Math.abs(deltaX) > Math.abs(deltaY) * 1.5) {
      this.isSwiping = true;
      
      // Visual feedback
      const indicator = this.getSwipeIndicator();
      if (indicator) {
        indicator.style.opacity = Math.min(1, Math.abs(deltaX) / SWIPE_THRESHOLD);
        indicator.style.transform = deltaX > 0 ? 'translateX(-10px)' : 'translateX(10px)';
        indicator.textContent = deltaX > 0 ? '← Previous' : 'Next →';
      }
      
      // Prevent scroll while swiping horizontally
      e.preventDefault();
    }
  }
  
  handleTouchEnd(e) {
    this.cancelLongPress();
    
    if (!this.touchStart) return;
    
    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - this.touchStart.x;
    const deltaY = touch.clientY - this.touchStart.y;
    const duration = Date.now() - this.touchStartTime;
    const velocity = Math.abs(deltaX) / duration;
    
    // Hide swipe indicator
    const indicator = this.getSwipeIndicator();
    if (indicator) {
      indicator.style.opacity = '0';
    }
    
    // Check for swipe
    if (this.isSwiping && Math.abs(deltaX) > SWIPE_THRESHOLD && velocity > SWIPE_VELOCITY) {
      if (deltaX > 0) {
        this.onSwipeRight();
      } else {
        this.onSwipeLeft();
      }
      this.resetTouch();
      return;
    }
    
    // Check for tap (minimal movement)
    if (Math.abs(deltaX) < 10 && Math.abs(deltaY) < 10 && duration < 300) {
      // Check for double tap
      if (this.options.enableDoubleTap && this.lastTap && (Date.now() - this.lastTap) < DOUBLE_TAP_DELAY) {
        this.onDoubleTap(touch);
        this.lastTap = null;
      } else {
        this.lastTap = Date.now();
        // Single tap handled by default click
      }
    }
    
    this.resetTouch();
  }
  
  handleAvatarTap(e) {
    // Tap on Kelly to toggle play/pause
    if (!this.isSwiping) {
      this.kellyOS?.togglePlay?.();
      this.triggerHaptic('light');
    }
  }
  
  handleLongPress(touch) {
    this.triggerHaptic('medium');
    
    // Show context menu
    this.showTouchMenu(touch.clientX, touch.clientY);
  }
  
  // ---------------------------------------------------------------------------
  // GESTURE ACTIONS
  // ---------------------------------------------------------------------------
  
  onSwipeLeft() {
    // Next phase
    if (this.kellyOS?.advancePhase) {
      this.kellyOS.advancePhase();
      this.triggerHaptic('light');
      
      if (window.KellyA11y) {
        window.KellyA11y.announce('Next phase');
      }
    }
  }
  
  onSwipeRight() {
    // Previous phase (show message)
    this.triggerHaptic('light');
    
    if (window.KellyA11y) {
      window.KellyA11y.announce('Swipe forward to continue. Going back is not available.');
    }
    
    this.showToast('Swipe → to continue');
  }
  
  onDoubleTap(touch) {
    // Replay current audio
    const audio = document.getElementById('kelly-audio');
    if (audio && audio.src) {
      audio.currentTime = 0;
      audio.play();
      this.triggerHaptic('light');
      
      if (window.KellyA11y) {
        window.KellyA11y.announce('Replaying audio');
      }
    }
  }
  
  // ---------------------------------------------------------------------------
  // UI HELPERS
  // ---------------------------------------------------------------------------
  
  getSwipeIndicator() {
    let indicator = document.getElementById('swipe-indicator');
    if (!indicator) {
      indicator = document.createElement('div');
      indicator.id = 'swipe-indicator';
      indicator.className = 'swipe-indicator';
      document.body.appendChild(indicator);
    }
    return indicator;
  }
  
  showTouchMenu(x, y) {
    // Remove existing
    document.getElementById('touch-menu')?.remove();
    
    const menu = document.createElement('div');
    menu.id = 'touch-menu';
    menu.className = 'touch-menu';
    menu.innerHTML = `
      <button data-action="replay">🔄 Replay</button>
      <button data-action="settings">⚙️ Settings</button>
      <button data-action="share">📤 Share</button>
    `;
    
    // Position near touch
    menu.style.left = `${Math.min(x, window.innerWidth - 160)}px`;
    menu.style.top = `${Math.min(y, window.innerHeight - 150)}px`;
    
    document.body.appendChild(menu);
    
    // Handle actions
    menu.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const action = e.target.dataset.action;
        this.handleMenuAction(action);
        menu.remove();
      });
    });
    
    // Close on tap outside
    setTimeout(() => {
      document.addEventListener('touchstart', function handler(e) {
        if (!menu.contains(e.target)) {
          menu.remove();
          document.removeEventListener('touchstart', handler);
        }
      }, { once: true });
    }, 100);
  }
  
  handleMenuAction(action) {
    switch (action) {
      case 'replay':
        const audio = document.getElementById('kelly-audio');
        if (audio) {
          audio.currentTime = 0;
          audio.play();
        }
        break;
      case 'settings':
        window.KellySettings?.open?.();
        break;
      case 'share':
        window.EarnToLearn?.open?.();
        break;
    }
    this.triggerHaptic('light');
  }
  
  showToast(message) {
    const existing = document.querySelector('.touch-toast');
    if (existing) existing.remove();
    
    const toast = document.createElement('div');
    toast.className = 'touch-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.remove(), 2000);
  }
  
  // ---------------------------------------------------------------------------
  // HAPTIC FEEDBACK
  // ---------------------------------------------------------------------------
  
  triggerHaptic(intensity = 'light') {
    if (!this.options.enableHaptics) return;
    
    if ('vibrate' in navigator) {
      const patterns = {
        light: [10],
        medium: [20],
        heavy: [30, 10, 30]
      };
      navigator.vibrate(patterns[intensity] || patterns.light);
    }
  }
  
  // ---------------------------------------------------------------------------
  // TOUCH TARGET ENFORCEMENT
  // ---------------------------------------------------------------------------
  
  enforceTouchTargetSizes() {
    const minSize = this.options.minTouchTargetSize;
    
    // Observe DOM changes
    const observer = new MutationObserver(() => {
      this.checkTouchTargets();
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Initial check
    this.checkTouchTargets();
  }
  
  checkTouchTargets() {
    const interactiveElements = document.querySelectorAll(
      'button, a, .choice-card, [role="button"], input, select'
    );
    
    interactiveElements.forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.width < this.options.minTouchTargetSize || rect.height < this.options.minTouchTargetSize) {
        el.classList.add('touch-target-small');
      } else {
        el.classList.remove('touch-target-small');
      }
    });
  }
  
  // ---------------------------------------------------------------------------
  // STYLES
  // ---------------------------------------------------------------------------
  
  injectStyles() {
    if (document.getElementById('kelly-touch-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-touch-styles';
    styles.textContent = `
      /* Touch target minimum sizing */
      .touch-target-small {
        min-width: 44px !important;
        min-height: 44px !important;
      }
      
      /* Swipe indicator */
      .swipe-indicator {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 16px 28px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s;
        z-index: 9999;
      }
      
      /* Touch menu */
      .touch-menu {
        position: fixed;
        background: rgba(24, 24, 27, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 8px;
        z-index: 10000;
        min-width: 150px;
        animation: touchMenuIn 0.2s ease-out;
      }
      
      @keyframes touchMenuIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
      }
      
      .touch-menu button {
        display: block;
        width: 100%;
        padding: 14px 18px;
        background: transparent;
        border: none;
        color: #e4e4e7;
        font-size: 1rem;
        text-align: left;
        cursor: pointer;
        border-radius: 8px;
        transition: background 0.15s;
      }
      
      .touch-menu button:hover,
      .touch-menu button:active {
        background: rgba(255, 255, 255, 0.1);
      }
      
      /* Touch toast */
      .touch-toast {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.85);
        color: white;
        padding: 12px 24px;
        border-radius: 10px;
        font-size: 0.95rem;
        z-index: 9999;
        animation: toastFade 0.2s ease-out;
      }
      
      @keyframes toastFade {
        from { opacity: 0; transform: translate(-50%, 10px); }
        to { opacity: 1; transform: translate(-50%, 0); }
      }
      
      /* Prevent text selection during touch */
      .ui-active {
        -webkit-user-select: none;
        user-select: none;
        -webkit-touch-callout: none;
      }
      
      /* Improved tap highlighting */
      .choice-card {
        -webkit-tap-highlight-color: rgba(59, 130, 246, 0.3);
      }
      
      /* Mobile-specific button sizing */
      @media (max-width: 768px) {
        .choice-card {
          min-height: 56px;
          padding: 16px 20px;
          font-size: 1rem;
        }
        
        .share-btn-large {
          min-height: 52px;
          font-size: 1rem;
        }
      }
      
      /* Safe area handling for notched phones */
      @supports (padding: max(0px)) {
        .lesson-container,
        .mode-panel {
          padding-left: max(16px, env(safe-area-inset-left));
          padding-right: max(16px, env(safe-area-inset-right));
          padding-bottom: max(16px, env(safe-area-inset-bottom));
        }
      }
    `;
    document.head.appendChild(styles);
  }
  
  // ---------------------------------------------------------------------------
  // UTILITY
  // ---------------------------------------------------------------------------
  
  resetTouch() {
    this.touchStart = null;
    this.touchStartTime = null;
    this.isSwiping = false;
    this.cancelLongPress();
  }
  
  cancelLongPress() {
    if (this.longPressTimer) {
      clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
  }
  
  destroy() {
    document.getElementById('kelly-touch-styles')?.remove();
    document.getElementById('swipe-indicator')?.remove();
    document.getElementById('touch-menu')?.remove();
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

const kellyTouch = new KellyTouch();
window.KellyTouch = kellyTouch;

console.log('[KellyTouch] Touch module loaded - Swipe ←→ to navigate');






