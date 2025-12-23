/**
 * Kelly Autoplay Handler
 * 
 * Handles iOS Safari autoplay restrictions and cross-platform audio unlock.
 * iOS Safari blocks autoplay with sound - requires user interaction to unlock.
 * 
 * Usage:
 *   KellyAutoplayHandler.init();
 *   KellyAutoplayHandler.onUserInteraction(() => {
 *     // Audio can now play
 *   });
 */

(function() {
  'use strict';

  const KellyAutoplayHandler = {
    isInitialized: false,
    audioUnlocked: false,
    isIOS: false,
    isSafari: false,
    unlockCallbacks: [],
    unlockButton: null,

    /**
     * Detect iOS Safari
     */
    detectPlatform() {
      const ua = navigator.userAgent || navigator.vendor || window.opera;
      
      // iOS detection
      this.isIOS = /iPad|iPhone|iPod/.test(ua) && !window.MSStream;
      
      // Safari detection (including iOS Safari)
      this.isSafari = /^((?!chrome|android).)*safari/i.test(ua) || this.isIOS;
      
      return {
        isIOS: this.isIOS,
        isSafari: this.isSafari,
        needsUnlock: this.isIOS || this.isSafari
      };
    },

    /**
     * Initialize autoplay handler
     */
    init() {
      if (this.isInitialized) return this;
      
      const platform = this.detectPlatform();
      console.log('[KellyAutoplay] Platform:', platform);
      
      // Test if audio can autoplay
      this.testAutoplay().then((canAutoplay) => {
        if (!canAutoplay && platform.needsUnlock) {
          console.log('[KellyAutoplay] Autoplay blocked - showing unlock button');
          this.showUnlockButton();
        } else {
          this.audioUnlocked = true;
          this.notifyUnlock();
        }
      });

      // Listen for user interactions to unlock audio
      this.setupUnlockListeners();
      
      this.isInitialized = true;
      return this;
    },

    /**
     * Test if audio can autoplay
     */
    async testAutoplay() {
      try {
        const testAudio = new Audio();
        testAudio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OSfTgwOUKzn8LZjGwY4kdfyzHksBSR3x/DdkEAKFF606euoVRQKRp/g8r5sIQUrgc7y2Yk2CBtpvfDkn04MDlCs5/C2YxsGOJHX8sx5LAUkd8fw3ZBAC';
        testAudio.volume = 0.01; // Very quiet
        await testAudio.play();
        testAudio.pause();
        testAudio.remove();
        return true;
      } catch (e) {
        console.log('[KellyAutoplay] Autoplay test failed:', e.name);
        return false;
      }
    },

    /**
     * Setup listeners for user interaction to unlock audio
     */
    setupUnlockListeners() {
      const unlock = () => {
        if (!this.audioUnlocked) {
          this.audioUnlocked = true;
          this.hideUnlockButton();
          this.notifyUnlock();
        }
      };

      // Listen for any user interaction
      const events = ['touchstart', 'touchend', 'mousedown', 'keydown', 'click'];
      events.forEach(event => {
        document.addEventListener(event, unlock, { once: true, passive: true });
      });
    },

    /**
     * Show "Tap to start" button
     */
    showUnlockButton() {
      if (this.unlockButton) return;

      const button = document.createElement('button');
      button.id = 'kelly-autoplay-unlock';
      button.className = 'kelly-autoplay-unlock-btn';
      button.innerHTML = `
        <div class="kelly-autoplay-icon">🎤</div>
        <div class="kelly-autoplay-text">Tap to start Kelly's voice</div>
      `;
      
      button.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 10000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 24px 32px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        transition: transform 0.2s, box-shadow 0.2s;
      `;

      button.addEventListener('click', () => {
        this.audioUnlocked = true;
        this.hideUnlockButton();
        this.notifyUnlock();
      });

      button.addEventListener('touchstart', () => {
        button.style.transform = 'translate(-50%, -50%) scale(0.95)';
      });

      button.addEventListener('touchend', () => {
        button.style.transform = 'translate(-50%, -50%) scale(1)';
      });

      document.body.appendChild(button);
      this.unlockButton = button;
    },

    /**
     * Hide unlock button
     */
    hideUnlockButton() {
      if (this.unlockButton) {
        this.unlockButton.style.opacity = '0';
        this.unlockButton.style.transform = 'translate(-50%, -50%) scale(0.9)';
        setTimeout(() => {
          if (this.unlockButton && this.unlockButton.parentNode) {
            this.unlockButton.parentNode.removeChild(this.unlockButton);
          }
          this.unlockButton = null;
        }, 200);
      }
    },

    /**
     * Register callback for when audio is unlocked
     */
    onUnlock(callback) {
      if (this.audioUnlocked) {
        callback();
      } else {
        this.unlockCallbacks.push(callback);
      }
    },

    /**
     * Notify all unlock callbacks
     */
    notifyUnlock() {
      this.unlockCallbacks.forEach(cb => {
        try {
          cb();
        } catch (e) {
          console.warn('[KellyAutoplay] Unlock callback error:', e);
        }
      });
      this.unlockCallbacks = [];
    },

    /**
     * Check if audio is unlocked
     */
    isUnlocked() {
      return this.audioUnlocked;
    }
  };

  // Auto-init
  if (typeof window !== 'undefined') {
    window.KellyAutoplayHandler = KellyAutoplayHandler;
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => KellyAutoplayHandler.init());
    } else {
      KellyAutoplayHandler.init();
    }
  }
})();

