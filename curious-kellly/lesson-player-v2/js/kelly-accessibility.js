/**
 * Kelly Accessibility Module
 * 
 * Provides keyboard navigation, screen reader support, and accessibility features
 * for the Curious Kelly lesson player.
 * 
 * Features:
 * - Full keyboard navigation (Space, Arrows, Tab, Escape)
 * - Screen reader announcements
 * - Focus management
 * - High contrast mode support
 * - Reduced motion support
 * - Touch accessibility
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// KEYBOARD SHORTCUTS
// =============================================================================

const KEYBOARD_SHORTCUTS = {
  // Playback
  'Space': { action: 'togglePlay', description: 'Play/Pause audio' },
  'Enter': { action: 'selectFocused', description: 'Select focused option' },
  
  // Navigation
  'ArrowRight': { action: 'nextPhase', description: 'Skip to next phase' },
  'ArrowLeft': { action: 'previousPhase', description: 'Go to previous phase' },
  'ArrowUp': { action: 'focusPrevious', description: 'Focus previous option' },
  'ArrowDown': { action: 'focusNext', description: 'Focus next option' },
  
  // UI Controls
  'Escape': { action: 'closeModal', description: 'Close modal/menu' },
  'Tab': { action: 'cycleElements', description: 'Cycle through elements' },
  
  // Quick Actions
  'm': { action: 'toggleMute', description: 'Toggle mute' },
  's': { action: 'openSettings', description: 'Open settings' },
  'h': { action: 'showHelp', description: 'Show keyboard shortcuts' },
  '?': { action: 'showHelp', description: 'Show keyboard shortcuts' },
  
  // Age Adjustment
  '+': { action: 'increaseAge', description: 'Increase age by 1' },
  '=': { action: 'increaseAge', description: 'Increase age by 1' },
  '-': { action: 'decreaseAge', description: 'Decrease age by 1' },
  
  // Accessibility
  'r': { action: 'repeatAudio', description: 'Repeat current audio' },
  'c': { action: 'toggleCaptions', description: 'Toggle captions' }
};

// =============================================================================
// ACCESSIBILITY CLASS
// =============================================================================

class KellyAccessibility {
  constructor(options = {}) {
    this.options = {
      announcePhases: true,
      enableKeyboardNav: true,
      enableFocusManagement: true,
      highContrastMode: false,
      reducedMotion: this.prefersReducedMotion(),
      ...options
    };
    
    // State
    this.currentFocusIndex = -1;
    this.focusableElements = [];
    this.isHelpModalOpen = false;
    this.captionsEnabled = localStorage.getItem('kelly_captions') === 'true';
    
    // Live region for screen reader announcements
    this.liveRegion = null;
    
    // Reference to KellyOS
    this.kellyOS = null;
    
    this.init();
  }
  
  // ---------------------------------------------------------------------------
  // INITIALIZATION
  // ---------------------------------------------------------------------------
  
  init() {
    this.createLiveRegion();
    this.setupKeyboardNavigation();
    this.setupFocusManagement();
    this.checkSystemPreferences();
    this.injectStyles();
    
    console.log('[KellyA11y] Accessibility module initialized');
  }
  
  /**
   * Set reference to KellyOS for action callbacks
   */
  setKellyOS(kellyOS) {
    this.kellyOS = kellyOS;
  }
  
  /**
   * Create ARIA live region for screen reader announcements
   */
  createLiveRegion() {
    if (document.getElementById('kelly-live-region')) return;
    
    this.liveRegion = document.createElement('div');
    this.liveRegion.id = 'kelly-live-region';
    this.liveRegion.setAttribute('role', 'status');
    this.liveRegion.setAttribute('aria-live', 'polite');
    this.liveRegion.setAttribute('aria-atomic', 'true');
    this.liveRegion.className = 'sr-only';
    this.liveRegion.style.cssText = `
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    `;
    document.body.appendChild(this.liveRegion);
  }
  
  /**
   * Check system preferences for accessibility
   */
  checkSystemPreferences() {
    // Reduced motion preference
    if (this.prefersReducedMotion()) {
      document.body.classList.add('reduce-motion');
      this.options.reducedMotion = true;
    }
    
    // High contrast preference
    if (window.matchMedia('(prefers-contrast: high)').matches) {
      document.body.classList.add('high-contrast');
      this.options.highContrastMode = true;
    }
    
    // Listen for changes
    window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', (e) => {
      this.options.reducedMotion = e.matches;
      document.body.classList.toggle('reduce-motion', e.matches);
    });
  }
  
  /**
   * Check if user prefers reduced motion
   */
  prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  
  /**
   * Inject accessibility CSS
   */
  injectStyles() {
    if (document.getElementById('kelly-a11y-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-a11y-styles';
    styles.textContent = `
      /* Screen reader only content */
      .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }
      
      /* Focus indicators */
      .kelly-focusable:focus,
      .choice-card:focus,
      button:focus,
      [tabindex]:focus {
        outline: 3px solid #3b82f6;
        outline-offset: 2px;
      }
      
      /* Reduced motion */
      .reduce-motion * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
      
      /* High contrast mode */
      .high-contrast {
        --bg-primary: #000;
        --text-primary: #fff;
        --accent: #ffff00;
      }
      
      .high-contrast .choice-card {
        border: 2px solid #fff;
      }
      
      .high-contrast .choice-card:hover,
      .high-contrast .choice-card:focus {
        background: #333;
        border-color: #ffff00;
      }
      
      /* Skip link */
      .skip-link {
        position: absolute;
        top: -100px;
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 24px;
        background: #3b82f6;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        z-index: 10000;
        transition: top 0.2s;
      }
      
      .skip-link:focus {
        top: 10px;
      }
      
      /* Keyboard help modal */
      .keyboard-help-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        animation: fadeIn 0.2s ease-out;
      }
      
      .keyboard-help-content {
        background: #18181b;
        border-radius: 16px;
        padding: 32px;
        max-width: 500px;
        max-height: 80vh;
        overflow-y: auto;
        color: #f4f4f5;
      }
      
      .keyboard-help-content h2 {
        margin: 0 0 24px;
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .keyboard-shortcut {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #27272a;
      }
      
      .keyboard-shortcut:last-child {
        border-bottom: none;
      }
      
      .keyboard-key {
        display: inline-block;
        padding: 4px 10px;
        background: #27272a;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #a1a1aa;
        min-width: 32px;
        text-align: center;
      }
      
      .keyboard-action {
        color: #a1a1aa;
        font-size: 0.95rem;
      }
      
      /* Caption display */
      .kelly-captions {
        position: absolute;
        bottom: 120px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.85);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        max-width: 80%;
        text-align: center;
        font-size: 1.1rem;
        line-height: 1.5;
        z-index: 100;
        pointer-events: none;
      }
      
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
    `;
    document.head.appendChild(styles);
    
    // Add skip link
    this.addSkipLink();
  }
  
  /**
   * Add skip to main content link
   */
  addSkipLink() {
    if (document.getElementById('skip-link')) return;
    
    const skipLink = document.createElement('a');
    skipLink.id = 'skip-link';
    skipLink.className = 'skip-link';
    skipLink.href = '#question-text';
    skipLink.textContent = 'Skip to lesson content';
    document.body.insertBefore(skipLink, document.body.firstChild);
  }
  
  // ---------------------------------------------------------------------------
  // KEYBOARD NAVIGATION
  // ---------------------------------------------------------------------------
  
  setupKeyboardNavigation() {
    if (!this.options.enableKeyboardNav) return;
    
    document.addEventListener('keydown', (e) => this.handleKeyDown(e));
  }
  
  handleKeyDown(e) {
    // Don't handle if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    const key = e.key;
    const shortcut = KEYBOARD_SHORTCUTS[key];
    
    if (shortcut) {
      e.preventDefault();
      this.executeAction(shortcut.action);
      return;
    }
    
    // Number keys 1-9 for quick option selection
    if (/^[1-9]$/.test(key)) {
      e.preventDefault();
      this.selectOptionByNumber(parseInt(key));
    }
  }
  
  executeAction(action) {
    switch (action) {
      case 'togglePlay':
        this.kellyOS?.togglePlay?.();
        this.announce(this.kellyOS?.state?.isPlaying ? 'Paused' : 'Playing');
        break;
        
      case 'selectFocused':
        this.selectCurrentFocus();
        break;
        
      case 'nextPhase':
        if (this.kellyOS?.advancePhase) {
          this.kellyOS.advancePhase();
          this.announce('Next phase');
        }
        break;
        
      case 'previousPhase':
        this.announce('Going back is not available in this lesson');
        break;
        
      case 'focusPrevious':
        this.focusPreviousElement();
        break;
        
      case 'focusNext':
        this.focusNextElement();
        break;
        
      case 'closeModal':
        this.closeCurrentModal();
        break;
        
      case 'toggleMute':
        this.toggleMute();
        break;
        
      case 'openSettings':
        this.openSettings();
        break;
        
      case 'showHelp':
        this.toggleHelpModal();
        break;
        
      case 'increaseAge':
        this.adjustAge(1);
        break;
        
      case 'decreaseAge':
        this.adjustAge(-1);
        break;
        
      case 'repeatAudio':
        this.repeatCurrentAudio();
        break;
        
      case 'toggleCaptions':
        this.toggleCaptions();
        break;
        
      default:
        console.log(`[KellyA11y] Unknown action: ${action}`);
    }
  }
  
  // ---------------------------------------------------------------------------
  // FOCUS MANAGEMENT
  // ---------------------------------------------------------------------------
  
  setupFocusManagement() {
    if (!this.options.enableFocusManagement) return;
    
    // Update focusable elements when DOM changes
    const observer = new MutationObserver(() => this.updateFocusableElements());
    observer.observe(document.body, { childList: true, subtree: true });
    
    this.updateFocusableElements();
  }
  
  updateFocusableElements() {
    this.focusableElements = Array.from(document.querySelectorAll(
      '.choice-card, button:not([disabled]), [tabindex]:not([tabindex="-1"]), a[href]'
    )).filter(el => {
      const style = window.getComputedStyle(el);
      return style.display !== 'none' && style.visibility !== 'hidden';
    });
  }
  
  focusNextElement() {
    this.updateFocusableElements();
    if (this.focusableElements.length === 0) return;
    
    this.currentFocusIndex = (this.currentFocusIndex + 1) % this.focusableElements.length;
    this.focusableElements[this.currentFocusIndex].focus();
    this.announceFocusedElement();
  }
  
  focusPreviousElement() {
    this.updateFocusableElements();
    if (this.focusableElements.length === 0) return;
    
    this.currentFocusIndex = this.currentFocusIndex <= 0 
      ? this.focusableElements.length - 1 
      : this.currentFocusIndex - 1;
    this.focusableElements[this.currentFocusIndex].focus();
    this.announceFocusedElement();
  }
  
  selectCurrentFocus() {
    const focused = document.activeElement;
    if (focused && (focused.classList.contains('choice-card') || focused.tagName === 'BUTTON')) {
      focused.click();
    }
  }
  
  selectOptionByNumber(num) {
    const options = document.querySelectorAll('.choice-card');
    if (options[num - 1]) {
      options[num - 1].click();
      this.announce(`Selected option ${num}`);
    }
  }
  
  announceFocusedElement() {
    const focused = document.activeElement;
    if (focused) {
      const text = focused.textContent?.trim() || focused.getAttribute('aria-label') || '';
      this.announce(text);
    }
  }
  
  // ---------------------------------------------------------------------------
  // SCREEN READER ANNOUNCEMENTS
  // ---------------------------------------------------------------------------
  
  /**
   * Announce message to screen readers
   * @param {string} message - Message to announce
   * @param {string} priority - 'polite' or 'assertive'
   */
  announce(message, priority = 'polite') {
    if (!this.liveRegion) return;
    
    this.liveRegion.setAttribute('aria-live', priority);
    this.liveRegion.textContent = '';
    
    // Small delay to ensure announcement
    setTimeout(() => {
      this.liveRegion.textContent = message;
    }, 50);
  }
  
  /**
   * Announce phase change
   */
  announcePhase(phaseName, content) {
    if (!this.options.announcePhases) return;
    
    const phaseNames = {
      'welcome': 'Welcome',
      'Hook': 'Introduction',
      'Fact1': 'First question',
      'Fact2': 'Second question',
      'Fact3': 'Third question',
      'Wisdom': 'Today\'s wisdom',
      'complete': 'Lesson complete'
    };
    
    const name = phaseNames[phaseName] || phaseName;
    this.announce(`${name}. ${content || ''}`);
  }
  
  // ---------------------------------------------------------------------------
  // MODAL MANAGEMENT
  // ---------------------------------------------------------------------------
  
  toggleHelpModal() {
    if (this.isHelpModalOpen) {
      this.closeHelpModal();
    } else {
      this.showHelpModal();
    }
  }
  
  showHelpModal() {
    this.isHelpModalOpen = true;
    
    const modal = document.createElement('div');
    modal.id = 'keyboard-help-modal';
    modal.className = 'keyboard-help-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-label', 'Keyboard shortcuts');
    modal.onclick = (e) => { if (e.target === modal) this.closeHelpModal(); };
    
    const shortcutsHTML = Object.entries(KEYBOARD_SHORTCUTS)
      .filter(([key]) => !['=', '?'].includes(key)) // Hide duplicate keys
      .map(([key, info]) => `
        <div class="keyboard-shortcut">
          <span class="keyboard-key">${key === 'Space' ? '␣' : key}</span>
          <span class="keyboard-action">${info.description}</span>
        </div>
      `).join('');
    
    modal.innerHTML = `
      <div class="keyboard-help-content">
        <h2>⌨️ Keyboard Shortcuts</h2>
        ${shortcutsHTML}
        <div style="margin-top: 24px; text-align: center;">
          <button onclick="KellyA11y.closeHelpModal()" style="padding: 12px 24px; background: #3b82f6; border: none; color: white; border-radius: 8px; cursor: pointer; font-size: 1rem;">
            Close (Esc)
          </button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    modal.querySelector('button').focus();
    this.announce('Keyboard shortcuts dialog opened. Press Escape to close.');
  }
  
  closeHelpModal() {
    const modal = document.getElementById('keyboard-help-modal');
    if (modal) {
      modal.remove();
      this.isHelpModalOpen = false;
      this.announce('Keyboard shortcuts closed');
    }
  }
  
  closeCurrentModal() {
    if (this.isHelpModalOpen) {
      this.closeHelpModal();
      return;
    }
    
    // Close other modals
    const modal = document.querySelector('.earn-overlay.open, .modal.open, [role="dialog"]');
    if (modal) {
      // Try to find close button
      const closeBtn = modal.querySelector('.close, [aria-label="Close"], .btn-close');
      if (closeBtn) closeBtn.click();
      else modal.remove();
    }
    
    // Close drawer
    if (this.kellyOS?.closeDrawer) {
      this.kellyOS.closeDrawer();
    }
  }
  
  // ---------------------------------------------------------------------------
  // ACCESSIBILITY ACTIONS
  // ---------------------------------------------------------------------------
  
  toggleMute() {
    const audio = document.getElementById('kelly-audio');
    if (audio) {
      audio.muted = !audio.muted;
      this.announce(audio.muted ? 'Audio muted' : 'Audio unmuted');
    }
  }
  
  adjustAge(delta) {
    const slider = document.getElementById('age-slider');
    if (slider) {
      const newValue = parseInt(slider.value) + delta;
      if (newValue >= 2 && newValue <= 102) {
        slider.value = newValue;
        slider.dispatchEvent(new Event('input'));
        this.announce(`Age set to ${newValue}`);
      }
    }
  }
  
  repeatCurrentAudio() {
    const audio = document.getElementById('kelly-audio');
    if (audio && audio.src) {
      audio.currentTime = 0;
      audio.play();
      this.announce('Repeating audio');
    }
  }
  
  toggleCaptions() {
    this.captionsEnabled = !this.captionsEnabled;
    localStorage.setItem('kelly_captions', this.captionsEnabled.toString());
    
    if (this.captionsEnabled) {
      this.showCaptions();
    } else {
      this.hideCaptions();
    }
    
    this.announce(this.captionsEnabled ? 'Captions enabled' : 'Captions disabled');
  }
  
  showCaptions() {
    if (document.getElementById('kelly-captions')) return;
    
    const captions = document.createElement('div');
    captions.id = 'kelly-captions';
    captions.className = 'kelly-captions';
    captions.setAttribute('aria-live', 'polite');
    document.body.appendChild(captions);
  }
  
  hideCaptions() {
    document.getElementById('kelly-captions')?.remove();
  }
  
  updateCaptions(text) {
    if (!this.captionsEnabled) return;
    
    const captions = document.getElementById('kelly-captions');
    if (captions) {
      captions.textContent = text;
    }
  }
  
  openSettings() {
    // Trigger settings drawer
    const settingsBtn = document.querySelector('[data-action="open-settings"]');
    if (settingsBtn) settingsBtn.click();
    this.announce('Settings opened');
  }
  
  // ---------------------------------------------------------------------------
  // PUBLIC API
  // ---------------------------------------------------------------------------
  
  /**
   * Set high contrast mode
   */
  setHighContrast(enabled) {
    this.options.highContrastMode = enabled;
    document.body.classList.toggle('high-contrast', enabled);
    localStorage.setItem('kelly_high_contrast', enabled.toString());
    this.announce(enabled ? 'High contrast enabled' : 'High contrast disabled');
  }
  
  /**
   * Set reduced motion
   */
  setReducedMotion(enabled) {
    this.options.reducedMotion = enabled;
    document.body.classList.toggle('reduce-motion', enabled);
    localStorage.setItem('kelly_reduced_motion', enabled.toString());
    this.announce(enabled ? 'Reduced motion enabled' : 'Reduced motion disabled');
  }
  
  /**
   * Cleanup
   */
  destroy() {
    this.liveRegion?.remove();
    document.getElementById('kelly-a11y-styles')?.remove();
    document.getElementById('skip-link')?.remove();
    document.getElementById('keyboard-help-modal')?.remove();
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

const KellyA11y = new KellyAccessibility();
window.KellyA11y = KellyA11y;

console.log('[KellyA11y] Module loaded - Press ? for keyboard shortcuts');








