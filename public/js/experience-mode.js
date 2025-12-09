/**
 * ✨ Curious Kelly — Experience Mode Manager
 * 
 * Handles Focus Mode vs Explorer Mode switching with persistence.
 * 
 * Focus Mode: Clean, distraction-free learning (default)
 * Explorer Mode: All controls visible for power users
 * 
 * @version 1.0.0
 * @date December 2025
 */

(function() {
  'use strict';

  // Storage key
  const STORAGE_KEY = 'kelly_experience_mode';
  
  // Default mode
  const DEFAULT_MODE = 'focus';
  
  // Valid modes
  const MODES = ['focus', 'explorer'];

  /**
   * Experience Mode Manager
   */
  class ExperienceModeManager {
    constructor() {
      this.currentMode = this.loadMode();
      this.listeners = new Set();
      this.toggleElement = null;
      
      // Apply mode immediately
      this.applyMode(this.currentMode, false);
      
      // Initialize when DOM is ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.init());
      } else {
        this.init();
      }
    }

    /**
     * Initialize the mode manager
     */
    init() {
      this.createToggle();
      this.setupKeyboardShortcuts();
      this.announceMode();
      
      console.log(`✨ Kelly Experience Mode: ${this.currentMode.toUpperCase()}`);
    }

    /**
     * Load mode from localStorage
     */
    loadMode() {
      try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && MODES.includes(stored)) {
          return stored;
        }
      } catch (e) {
        console.warn('Could not load experience mode from storage:', e);
      }
      return DEFAULT_MODE;
    }

    /**
     * Save mode to localStorage
     */
    saveMode(mode) {
      try {
        localStorage.setItem(STORAGE_KEY, mode);
      } catch (e) {
        console.warn('Could not save experience mode to storage:', e);
      }
    }

    /**
     * Apply mode to the document
     */
    applyMode(mode, animate = true) {
      // Validate mode
      if (!MODES.includes(mode)) {
        console.warn(`Invalid mode: ${mode}. Using default.`);
        mode = DEFAULT_MODE;
      }

      const prevMode = this.currentMode;
      this.currentMode = mode;

      // Set data attribute on document
      document.documentElement.setAttribute('data-mode', mode);
      document.body.setAttribute('data-mode', mode);

      // Save to storage
      this.saveMode(mode);

      // Animate transition if requested
      if (animate && prevMode !== mode) {
        document.body.classList.add('mode-changed');
        setTimeout(() => {
          document.body.classList.remove('mode-changed');
        }, 300);
      }

      // Update toggle if it exists
      this.updateToggle();

      // Notify listeners
      this.notifyListeners(mode, prevMode);

      // Dispatch custom event
      const event = new CustomEvent('kelly-mode-change', {
        detail: { mode, previousMode: prevMode }
      });
      document.dispatchEvent(event);
    }

    /**
     * Toggle between modes
     */
    toggle() {
      const newMode = this.currentMode === 'focus' ? 'explorer' : 'focus';
      this.applyMode(newMode);
      this.announceMode();
    }

    /**
     * Get current mode
     */
    getMode() {
      return this.currentMode;
    }

    /**
     * Check if in focus mode
     */
    isFocusMode() {
      return this.currentMode === 'focus';
    }

    /**
     * Check if in explorer mode
     */
    isExplorerMode() {
      return this.currentMode === 'explorer';
    }

    /**
     * Create the mode toggle button
     */
    createToggle() {
      // Don't create if one already exists
      if (document.querySelector('.mode-toggle')) {
        this.toggleElement = document.querySelector('.mode-toggle');
        this.setupToggleEvents();
        return;
      }

      // Create toggle element
      const toggle = document.createElement('button');
      toggle.className = 'mode-toggle';
      toggle.setAttribute('aria-label', 'Switch experience mode');
      toggle.setAttribute('title', 'Switch to Explorer Mode for more controls');
      
      toggle.innerHTML = `
        <svg class="mode-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
        </svg>
        <span class="mode-toggle-label">${this.getModeLabel()}</span>
      `;

      document.body.appendChild(toggle);
      this.toggleElement = toggle;
      this.setupToggleEvents();
    }

    /**
     * Setup toggle button events
     */
    setupToggleEvents() {
      if (!this.toggleElement) return;
      
      this.toggleElement.addEventListener('click', () => this.toggle());
    }

    /**
     * Update toggle button appearance
     */
    updateToggle() {
      if (!this.toggleElement) return;
      
      const label = this.toggleElement.querySelector('.mode-toggle-label');
      if (label) {
        label.textContent = this.getModeLabel();
      }
      
      const title = this.currentMode === 'focus' 
        ? 'Switch to Explorer Mode for more controls'
        : 'Switch to Focus Mode for distraction-free learning';
      this.toggleElement.setAttribute('title', title);
    }

    /**
     * Get human-readable mode label
     */
    getModeLabel() {
      return this.currentMode === 'focus' ? '🎯 Focus' : '🔬 Explorer';
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
      document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Shift + M to toggle mode
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
          e.preventDefault();
          this.toggle();
        }
        
        // Escape in Explorer mode goes back to Focus
        if (e.key === 'Escape' && this.isExplorerMode()) {
          this.applyMode('focus');
        }
      });
    }

    /**
     * Announce mode change for screen readers
     */
    announceMode() {
      const announcement = this.currentMode === 'focus'
        ? 'Focus Mode: Distraction-free learning experience'
        : 'Explorer Mode: All customization options visible';
      
      // Create or update live region
      let liveRegion = document.getElementById('kelly-mode-announcer');
      if (!liveRegion) {
        liveRegion = document.createElement('div');
        liveRegion.id = 'kelly-mode-announcer';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        liveRegion.style.cssText = 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);';
        document.body.appendChild(liveRegion);
      }
      
      liveRegion.textContent = announcement;
    }

    /**
     * Add a mode change listener
     */
    onModeChange(callback) {
      this.listeners.add(callback);
      return () => this.listeners.delete(callback);
    }

    /**
     * Notify all listeners of mode change
     */
    notifyListeners(newMode, prevMode) {
      this.listeners.forEach(callback => {
        try {
          callback(newMode, prevMode);
        } catch (e) {
          console.error('Mode change listener error:', e);
        }
      });
    }
  }

  // Create singleton instance
  const modeManager = new ExperienceModeManager();

  // Expose to global scope
  window.KellyExperienceMode = {
    toggle: () => modeManager.toggle(),
    getMode: () => modeManager.getMode(),
    setMode: (mode) => modeManager.applyMode(mode),
    isFocus: () => modeManager.isFocusMode(),
    isExplorer: () => modeManager.isExplorerMode(),
    onChange: (callback) => modeManager.onModeChange(callback)
  };

  // Also expose instance for advanced use
  window._kellyModeManager = modeManager;

})();


