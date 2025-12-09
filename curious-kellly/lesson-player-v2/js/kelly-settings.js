/**
 * Kelly Settings Module
 * 
 * Provides a settings panel for the Curious Kelly lesson player.
 * Manages volume, language, accessibility, and display preferences.
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// DEFAULT SETTINGS
// =============================================================================

const DEFAULT_SETTINGS = {
  // Audio
  volume: 0.8,
  voiceSpeed: 1.0,
  autoPlay: true,
  
  // Display
  language: 'en',
  theme: 'dark',
  fontSize: 'medium',
  
  // Accessibility
  highContrast: false,
  reducedMotion: false,
  captions: false,
  screenReaderMode: false,
  
  // Avatar
  avatarEnabled: true,
  avatar3DPreferred: false,
  
  // Notifications
  streakReminders: true,
  dailyNotification: true,
  notificationTime: '08:00'
};

// =============================================================================
// SETTINGS CLASS
// =============================================================================

class KellySettings {
  constructor() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.isOpen = false;
    this.panel = null;
    this.kellyOS = null;
    
    this.loadSettings();
    this.init();
  }
  
  // ---------------------------------------------------------------------------
  // INITIALIZATION
  // ---------------------------------------------------------------------------
  
  init() {
    this.createPanel();
    this.bindEvents();
    this.applySettings();
    console.log('[KellySettings] Settings module initialized');
  }
  
  setKellyOS(kellyOS) {
    this.kellyOS = kellyOS;
  }
  
  loadSettings() {
    try {
      const stored = localStorage.getItem('kelly_settings');
      if (stored) {
        this.settings = { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
      }
    } catch (e) {
      console.warn('[KellySettings] Could not load settings:', e);
    }
  }
  
  saveSettings() {
    try {
      localStorage.setItem('kelly_settings', JSON.stringify(this.settings));
    } catch (e) {
      console.warn('[KellySettings] Could not save settings:', e);
    }
  }
  
  applySettings() {
    // Volume
    const audio = document.getElementById('kelly-audio');
    if (audio) {
      audio.volume = this.settings.volume;
    }
    
    // Accessibility
    if (this.settings.highContrast) {
      document.body.classList.add('high-contrast');
    }
    if (this.settings.reducedMotion) {
      document.body.classList.add('reduce-motion');
    }
    
    // Font size
    document.documentElement.style.setProperty('--kelly-font-scale', 
      this.settings.fontSize === 'small' ? '0.9' :
      this.settings.fontSize === 'large' ? '1.15' : '1'
    );
  }
  
  // ---------------------------------------------------------------------------
  // PANEL CREATION
  // ---------------------------------------------------------------------------
  
  createPanel() {
    if (this.panel) return;
    
    this.panel = document.createElement('div');
    this.panel.id = 'kelly-settings-panel';
    this.panel.className = 'settings-panel';
    this.panel.setAttribute('role', 'dialog');
    this.panel.setAttribute('aria-label', 'Settings');
    this.panel.innerHTML = this.getPanelHTML();
    
    this.injectStyles();
    document.body.appendChild(this.panel);
  }
  
  getPanelHTML() {
    return `
      <div class="settings-container">
        <div class="settings-header">
          <h2>⚙️ Settings</h2>
          <button class="settings-close" id="settings-close" aria-label="Close settings">✕</button>
        </div>
        
        <div class="settings-body">
          <!-- Audio Section -->
          <div class="settings-section">
            <h3>🔊 Audio</h3>
            
            <div class="settings-row">
              <label for="setting-volume">Volume</label>
              <div class="settings-slider-group">
                <input type="range" id="setting-volume" min="0" max="1" step="0.1" value="${this.settings.volume}">
                <span class="slider-value" id="volume-value">${Math.round(this.settings.volume * 100)}%</span>
              </div>
            </div>
            
            <div class="settings-row">
              <label for="setting-speed">Voice Speed</label>
              <div class="settings-slider-group">
                <input type="range" id="setting-speed" min="0.5" max="1.5" step="0.1" value="${this.settings.voiceSpeed}">
                <span class="slider-value" id="speed-value">${this.settings.voiceSpeed}x</span>
              </div>
            </div>
            
            <div class="settings-row">
              <label for="setting-autoplay">Auto-play Audio</label>
              <label class="toggle">
                <input type="checkbox" id="setting-autoplay" ${this.settings.autoPlay ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
          </div>
          
          <!-- Display Section -->
          <div class="settings-section">
            <h3>🎨 Display</h3>
            
            <div class="settings-row">
              <label for="setting-language">Language</label>
              <select id="setting-language">
                <option value="en" ${this.settings.language === 'en' ? 'selected' : ''}>English</option>
                <option value="es" ${this.settings.language === 'es' ? 'selected' : ''}>Español</option>
                <option value="fr" ${this.settings.language === 'fr' ? 'selected' : ''}>Français</option>
              </select>
            </div>
            
            <div class="settings-row">
              <label for="setting-fontsize">Text Size</label>
              <select id="setting-fontsize">
                <option value="small" ${this.settings.fontSize === 'small' ? 'selected' : ''}>Small</option>
                <option value="medium" ${this.settings.fontSize === 'medium' ? 'selected' : ''}>Medium</option>
                <option value="large" ${this.settings.fontSize === 'large' ? 'selected' : ''}>Large</option>
              </select>
            </div>
          </div>
          
          <!-- Accessibility Section -->
          <div class="settings-section">
            <h3>♿ Accessibility</h3>
            
            <div class="settings-row">
              <label for="setting-captions">Captions</label>
              <label class="toggle">
                <input type="checkbox" id="setting-captions" ${this.settings.captions ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
            
            <div class="settings-row">
              <label for="setting-highcontrast">High Contrast</label>
              <label class="toggle">
                <input type="checkbox" id="setting-highcontrast" ${this.settings.highContrast ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
            
            <div class="settings-row">
              <label for="setting-reducemotion">Reduce Motion</label>
              <label class="toggle">
                <input type="checkbox" id="setting-reducemotion" ${this.settings.reducedMotion ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
          </div>
          
          <!-- Avatar Section -->
          <div class="settings-section">
            <h3>🎭 Avatar</h3>
            
            <div class="settings-row">
              <label for="setting-avatar">Show Kelly</label>
              <label class="toggle">
                <input type="checkbox" id="setting-avatar" ${this.settings.avatarEnabled ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
            
            <div class="settings-row">
              <label for="setting-3d">Prefer 3D Avatar</label>
              <label class="toggle">
                <input type="checkbox" id="setting-3d" ${this.settings.avatar3DPreferred ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
          </div>
          
          <!-- Trust & Safety Section -->
          <div class="settings-section">
            <h3>🛡️ Trust & Safety</h3>
            
            <div class="settings-row">
              <label for="setting-simulated">Simulated Social Content</label>
              <label class="toggle">
                <input type="checkbox" id="setting-simulated">
                <span class="toggle-slider"></span>
              </label>
            </div>
            <p style="font-size: 0.8rem; color: #a1a1aa; margin-top: 4px;">
              Enables simulated learner comments marked with ✨.
            </p>
          </div>
          
          <!-- Notifications Section -->
          <div class="settings-section">
            <h3>🔔 Notifications</h3>
            
            <div class="settings-row">
              <label for="setting-streak">Streak Reminders</label>
              <label class="toggle">
                <input type="checkbox" id="setting-streak" ${this.settings.streakReminders ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
            
            <div class="settings-row">
              <label for="setting-daily">Daily Lesson Reminder</label>
              <label class="toggle">
                <input type="checkbox" id="setting-daily" ${this.settings.dailyNotification ? 'checked' : ''}>
                <span class="toggle-slider"></span>
              </label>
            </div>
            
            <div class="settings-row" id="notification-time-row" style="${this.settings.dailyNotification ? '' : 'opacity: 0.5; pointer-events: none;'}">
              <label for="setting-time">Reminder Time</label>
              <input type="time" id="setting-time" value="${this.settings.notificationTime}">
            </div>
          </div>
        </div>
        
        <div class="settings-footer">
          <button class="settings-btn-secondary" id="settings-reset">Reset to Defaults</button>
          <button class="settings-btn-primary" id="settings-done">Done</button>
        </div>
        
        <div class="settings-shortcuts">
          <p>Tip: Press <kbd>?</kbd> for keyboard shortcuts</p>
        </div>
      </div>
    `;
  }
  
  injectStyles() {
    if (document.getElementById('kelly-settings-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-settings-styles';
    styles.textContent = `
      .settings-panel {
        position: fixed;
        top: 0;
        right: -400px;
        width: 380px;
        height: 100%;
        background: rgba(15, 15, 17, 0.98);
        backdrop-filter: blur(30px);
        border-left: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 9500;
        transition: right 0.3s cubic-bezier(0.32, 0.72, 0, 1);
        display: flex;
        flex-direction: column;
      }
      
      .settings-panel.open {
        right: 0;
      }
      
      .settings-container {
        display: flex;
        flex-direction: column;
        height: 100%;
      }
      
      .settings-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }
      
      .settings-header h2 {
        margin: 0;
        font-size: 1.4rem;
        color: #fff;
      }
      
      .settings-close {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #fff;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .settings-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }
      
      .settings-body {
        flex: 1;
        overflow-y: auto;
        padding: 20px 24px;
      }
      
      .settings-section {
        margin-bottom: 28px;
      }
      
      .settings-section h3 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #71717a;
        margin: 0 0 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      }
      
      .settings-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0;
      }
      
      .settings-row label:first-child {
        color: #d4d4d8;
        font-size: 0.95rem;
      }
      
      .settings-slider-group {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .settings-slider-group input[type="range"] {
        width: 120px;
        height: 6px;
        -webkit-appearance: none;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        outline: none;
      }
      
      .settings-slider-group input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 18px;
        height: 18px;
        background: #3b82f6;
        border-radius: 50%;
        cursor: pointer;
        transition: transform 0.2s;
      }
      
      .settings-slider-group input[type="range"]::-webkit-slider-thumb:hover {
        transform: scale(1.1);
      }
      
      .slider-value {
        min-width: 45px;
        text-align: right;
        color: #a1a1aa;
        font-size: 0.9rem;
        font-family: monospace;
      }
      
      select {
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: #fff;
        font-size: 0.95rem;
        cursor: pointer;
      }
      
      select:focus {
        outline: none;
        border-color: #3b82f6;
      }
      
      input[type="time"] {
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: #fff;
        font-size: 0.95rem;
      }
      
      /* Toggle switch */
      .toggle {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 28px;
      }
      
      .toggle input {
        opacity: 0;
        width: 0;
        height: 0;
      }
      
      .toggle-slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        transition: all 0.3s;
      }
      
      .toggle-slider::before {
        position: absolute;
        content: "";
        height: 22px;
        width: 22px;
        left: 3px;
        bottom: 3px;
        background: white;
        border-radius: 50%;
        transition: transform 0.3s;
      }
      
      .toggle input:checked + .toggle-slider {
        background: #22c55e;
      }
      
      .toggle input:checked + .toggle-slider::before {
        transform: translateX(22px);
      }
      
      .toggle input:focus + .toggle-slider {
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
      }
      
      .settings-footer {
        display: flex;
        gap: 12px;
        padding: 20px 24px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }
      
      .settings-btn-primary,
      .settings-btn-secondary {
        flex: 1;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .settings-btn-primary {
        background: #3b82f6;
        border: none;
        color: white;
      }
      
      .settings-btn-primary:hover {
        background: #2563eb;
      }
      
      .settings-btn-secondary {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #a1a1aa;
      }
      
      .settings-btn-secondary:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #fff;
      }
      
      .settings-shortcuts {
        padding: 16px 24px;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
      }
      
      .settings-shortcuts p {
        color: #52525b;
        font-size: 0.85rem;
        margin: 0;
      }
      
      .settings-shortcuts kbd {
        display: inline-block;
        padding: 3px 8px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #a1a1aa;
      }
      
      /* Scrollbar */
      .settings-body::-webkit-scrollbar {
        width: 6px;
      }
      
      .settings-body::-webkit-scrollbar-track {
        background: transparent;
      }
      
      .settings-body::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
      }
      
      /* Mobile */
      @media (max-width: 768px) {
        .settings-panel {
          width: 100%;
          right: -100%;
        }
      }
    `;
    document.head.appendChild(styles);
  }
  
  // ---------------------------------------------------------------------------
  // EVENT BINDING
  // ---------------------------------------------------------------------------
  
  bindEvents() {
    // Close button
    document.getElementById('settings-close')?.addEventListener('click', () => this.close());
    
    // Done button
    document.getElementById('settings-done')?.addEventListener('click', () => this.close());
    
    // Reset button
    document.getElementById('settings-reset')?.addEventListener('click', () => this.resetToDefaults());
    
    // Volume slider
    document.getElementById('setting-volume')?.addEventListener('input', (e) => {
      this.settings.volume = parseFloat(e.target.value);
      document.getElementById('volume-value').textContent = `${Math.round(this.settings.volume * 100)}%`;
      const audio = document.getElementById('kelly-audio');
      if (audio) audio.volume = this.settings.volume;
      this.saveSettings();
    });
    
    // Speed slider
    document.getElementById('setting-speed')?.addEventListener('input', (e) => {
      this.settings.voiceSpeed = parseFloat(e.target.value);
      document.getElementById('speed-value').textContent = `${this.settings.voiceSpeed}x`;
      const audio = document.getElementById('kelly-audio');
      if (audio) audio.playbackRate = this.settings.voiceSpeed;
      this.saveSettings();
    });
    
    // Auto-play toggle
    document.getElementById('setting-autoplay')?.addEventListener('change', (e) => {
      this.settings.autoPlay = e.target.checked;
      this.saveSettings();
    });
    
    // Language select
    document.getElementById('setting-language')?.addEventListener('change', (e) => {
      this.settings.language = e.target.value;
      this.saveSettings();
      if (this.kellyOS) this.kellyOS.state.language = this.settings.language;
    });
    
    // Font size
    document.getElementById('setting-fontsize')?.addEventListener('change', (e) => {
      this.settings.fontSize = e.target.value;
      this.applySettings();
      this.saveSettings();
    });
    
    // Captions toggle
    document.getElementById('setting-captions')?.addEventListener('change', (e) => {
      this.settings.captions = e.target.checked;
      if (window.KellyA11y) {
        window.KellyA11y.captionsEnabled = e.target.checked;
        e.target.checked ? window.KellyA11y.showCaptions() : window.KellyA11y.hideCaptions();
      }
      this.saveSettings();
    });
    
    // High contrast toggle
    document.getElementById('setting-highcontrast')?.addEventListener('change', (e) => {
      this.settings.highContrast = e.target.checked;
      document.body.classList.toggle('high-contrast', e.target.checked);
      this.saveSettings();
    });
    
    // Reduced motion toggle
    document.getElementById('setting-reducemotion')?.addEventListener('change', (e) => {
      this.settings.reducedMotion = e.target.checked;
      document.body.classList.toggle('reduce-motion', e.target.checked);
      this.saveSettings();
    });
    
    // Avatar toggle
    document.getElementById('setting-avatar')?.addEventListener('change', (e) => {
      this.settings.avatarEnabled = e.target.checked;
      const avatarContainer = document.getElementById('avatar-container');
      if (avatarContainer) {
        avatarContainer.style.opacity = e.target.checked ? '1' : '0';
      }
      this.saveSettings();
    });
    
    // 3D avatar toggle
    document.getElementById('setting-3d')?.addEventListener('change', (e) => {
      this.settings.avatar3DPreferred = e.target.checked;
      this.saveSettings();
    });

    // Simulated Content toggle
    const simulatedToggle = document.getElementById('setting-simulated');
    if (simulatedToggle && window.KellySimulatedContent) {
        // Initialize state
        simulatedToggle.checked = window.KellySimulatedContent.getPrefs().enabled;
        
        // Bind change
        simulatedToggle.addEventListener('change', (e) => {
            window.KellySimulatedContent.toggle(e.target.checked);
        });
    }
    
    // Streak reminders
    document.getElementById('setting-streak')?.addEventListener('change', (e) => {
      this.settings.streakReminders = e.target.checked;
      this.saveSettings();
    });
    
    // Daily notification
    document.getElementById('setting-daily')?.addEventListener('change', (e) => {
      this.settings.dailyNotification = e.target.checked;
      const timeRow = document.getElementById('notification-time-row');
      if (timeRow) {
        timeRow.style.opacity = e.target.checked ? '1' : '0.5';
        timeRow.style.pointerEvents = e.target.checked ? 'auto' : 'none';
      }
      this.saveSettings();
    });
    
    // Notification time
    document.getElementById('setting-time')?.addEventListener('change', (e) => {
      this.settings.notificationTime = e.target.value;
      this.saveSettings();
    });
    
    // Click outside to close
    this.panel?.addEventListener('click', (e) => {
      if (e.target === this.panel) this.close();
    });
    
    // Escape to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) this.close();
    });
  }
  
  // ---------------------------------------------------------------------------
  // OPEN / CLOSE
  // ---------------------------------------------------------------------------
  
  open() {
    this.isOpen = true;
    this.panel?.classList.add('open');
    
    // Refresh simulated toggle state in case it changed elsewhere
    const simulatedToggle = document.getElementById('setting-simulated');
    if (simulatedToggle && window.KellySimulatedContent) {
        simulatedToggle.checked = window.KellySimulatedContent.getPrefs().enabled;
    }
    
    // Focus first interactive element
    setTimeout(() => {
      document.getElementById('setting-volume')?.focus();
    }, 300);
    
    if (window.KellyA11y) {
      window.KellyA11y.announce('Settings panel opened');
    }
  }
  
  close() {
    this.isOpen = false;
    this.panel?.classList.remove('open');
    
    if (window.KellyA11y) {
      window.KellyA11y.announce('Settings closed');
    }
  }
  
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
  
  // ---------------------------------------------------------------------------
  // ACTIONS
  // ---------------------------------------------------------------------------
  
  resetToDefaults() {
    if (!confirm('Reset all settings to defaults?')) return;
    
    this.settings = { ...DEFAULT_SETTINGS };
    this.saveSettings();
    this.applySettings();
    
    // Update UI
    this.panel.innerHTML = this.getPanelHTML();
    this.bindEvents();
    
    if (window.KellyA11y) {
      window.KellyA11y.announce('Settings reset to defaults');
    }
  }
  
  // ---------------------------------------------------------------------------
  // PUBLIC API
  // ---------------------------------------------------------------------------
  
  get(key) {
    return this.settings[key];
  }
  
  set(key, value) {
    this.settings[key] = value;
    this.saveSettings();
    this.applySettings();
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

const kellySettings = new KellySettings();
window.KellySettings = kellySettings;

console.log('[KellySettings] Settings module loaded');

