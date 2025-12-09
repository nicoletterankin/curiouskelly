/**
 * Kelly Avatar Controller
 * Manages Kelly's presence, modes, and visual states across the unified experience
 */

class KellyAvatarController {
  constructor() {
    this.mode = localStorage.getItem('kelly-mode') || '2d'; // 2d, 3d, audio, image, fullscreen
    this.socialMode = localStorage.getItem('kelly-social') || 'solo'; // solo, social
    this.state = 'idle'; // idle, teaching, listening, celebrating
    this.visible = true;
    
    this.expressions = {
      idle: '/images/expressions/curious-main.jpeg',
      teaching: '/images/expressions/explaining.jpeg',
      listening: '/images/expressions/curious-thinking.jpeg',
      celebrating: '/images/expressions/celebrating.jpeg',
      happy: '/images/expressions/happy-content.jpeg',
      surprised: '/images/expressions/surprised.jpeg',
      confused: '/images/expressions/confused.jpeg',
      peaceful: '/images/expressions/peaceful.jpeg'
    };
    
    this.init();
  }
  
  init() {
    this.createControlPanel();
    this.createAvatarContainer();
    this.applyMode();
    this.setState('idle');
  }
  
  createControlPanel() {
    const panel = document.createElement('div');
    panel.id = 'kelly-control-panel';
    panel.className = 'kelly-control-panel';
    panel.innerHTML = `
      <button class="control-toggle" aria-label="Kelly Controls">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 1v6m0 6v6m5.196-14.196l-4.242 4.242m0 6.364l4.242 4.242M23 12h-6m-6 0H1m14.196 5.196l-4.242-4.242m0-6.364l-4.242-4.242"/>
        </svg>
      </button>
      
      <div class="control-menu hidden">
        <div class="control-section">
          <h4>Display Mode</h4>
          <div class="control-buttons">
            <button class="control-btn" data-mode="2d" data-active="${this.mode === '2d'}">
              <span class="icon">🖼️</span>
              <span>2D</span>
            </button>
            <button class="control-btn" data-mode="3d" data-active="${this.mode === '3d'}">
              <span class="icon">🎭</span>
              <span>3D</span>
            </button>
            <button class="control-btn" data-mode="audio" data-active="${this.mode === 'audio'}">
              <span class="icon">🎧</span>
              <span>Audio</span>
            </button>
            <button class="control-btn" data-mode="image" data-active="${this.mode === 'image'}">
              <span class="icon">📸</span>
              <span>Image</span>
            </button>
            <button class="control-btn" data-mode="fullscreen" data-active="${this.mode === 'fullscreen'}">
              <span class="icon">⛶</span>
              <span>Full</span>
            </button>
          </div>
        </div>
        
        <div class="control-section">
          <h4>Experience</h4>
          <div class="control-buttons">
            <button class="control-btn" data-social="solo" data-active="${this.socialMode === 'solo'}">
              <span class="icon">👤</span>
              <span>Solo</span>
            </button>
            <button class="control-btn" data-social="social" data-active="${this.socialMode === 'social'}">
              <span class="icon">👥</span>
              <span>Social</span>
            </button>
          </div>
        </div>
        
        <div class="control-section">
          <button class="control-btn control-settings">
            <span class="icon">⚙️</span>
            <span>Settings</span>
          </button>
        </div>
      </div>
    `;
    
    document.body.appendChild(panel);
    this.attachControlListeners();
  }
  
  createAvatarContainer() {
    const container = document.createElement('div');
    container.id = 'kelly-avatar-container';
    container.className = 'kelly-avatar-container';
    container.innerHTML = `
      <div class="kelly-avatar">
        <img src="${this.expressions.idle}" alt="Kelly" class="kelly-image" />
        <div class="kelly-3d-canvas hidden"></div>
        <div class="kelly-audio-visualizer hidden">
          <div class="audio-bars">
            <span></span><span></span><span></span><span></span><span></span>
          </div>
        </div>
      </div>
    `;
    
    // Avatar is injected into sections as needed, not fixed
    // Each section can call kellyController.injectAvatar(targetElement)
  }
  
  attachControlListeners() {
    const toggle = document.querySelector('.control-toggle');
    const menu = document.querySelector('.control-menu');
    
    toggle.addEventListener('click', () => {
      menu.classList.toggle('hidden');
    });
    
    // Mode buttons
    document.querySelectorAll('[data-mode]').forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        this.setMode(mode);
      });
    });
    
    // Social buttons
    document.querySelectorAll('[data-social]').forEach(btn => {
      btn.addEventListener('click', () => {
        const social = btn.dataset.social;
        this.setSocialMode(social);
      });
    });
    
    // Settings
    document.querySelector('.control-settings')?.addEventListener('click', () => {
      window.location.href = '/settings.html';
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#kelly-control-panel')) {
        menu.classList.add('hidden');
      }
    });
  }
  
  setMode(mode) {
    this.mode = mode;
    localStorage.setItem('kelly-mode', mode);
    this.applyMode();
    this.updateControlButtons();
    this.dispatchEvent('modeChange', { mode });
  }
  
  setSocialMode(social) {
    this.socialMode = social;
    localStorage.setItem('kelly-social', social);
    this.updateControlButtons();
    this.dispatchEvent('socialModeChange', { socialMode: social });
  }
  
  setState(state) {
    this.state = state;
    this.updateExpression();
    this.dispatchEvent('stateChange', { state });
  }
  
  applyMode() {
    const image = document.querySelector('.kelly-image');
    const canvas = document.querySelector('.kelly-3d-canvas');
    const visualizer = document.querySelector('.kelly-audio-visualizer');
    
    if (!image) return;
    
    // Hide all
    image.classList.add('hidden');
    canvas?.classList.add('hidden');
    visualizer?.classList.add('hidden');
    
    switch (this.mode) {
      case '2d':
      case 'image':
        image.classList.remove('hidden');
        break;
      case '3d':
        canvas?.classList.remove('hidden');
        // TODO: Initialize Unity if not already loaded
        break;
      case 'audio':
        visualizer?.classList.remove('hidden');
        break;
      case 'fullscreen':
        image.classList.remove('hidden');
        this.enterFullscreen();
        break;
    }
  }
  
  updateExpression() {
    const image = document.querySelector('.kelly-image');
    if (image && this.expressions[this.state]) {
      image.src = this.expressions[this.state];
    }
  }
  
  updateControlButtons() {
    document.querySelectorAll('[data-mode]').forEach(btn => {
      btn.dataset.active = btn.dataset.mode === this.mode;
    });
    document.querySelectorAll('[data-social]').forEach(btn => {
      btn.dataset.active = btn.dataset.social === this.socialMode;
    });
  }
  
  enterFullscreen() {
    const container = document.querySelector('#kelly-avatar-container');
    if (container?.requestFullscreen) {
      container.requestFullscreen();
    }
  }
  
  injectAvatar(targetElement) {
    const container = document.querySelector('#kelly-avatar-container');
    if (container && targetElement) {
      targetElement.appendChild(container);
    }
  }
  
  dispatchEvent(eventName, detail) {
    window.dispatchEvent(new CustomEvent(`kelly:${eventName}`, { detail }));
  }
  
  // Public API
  show() {
    this.visible = true;
    document.querySelector('#kelly-avatar-container')?.classList.remove('hidden');
  }
  
  hide() {
    this.visible = false;
    document.querySelector('#kelly-avatar-container')?.classList.add('hidden');
  }
  
  speak(text) {
    this.setState('teaching');
    // TODO: Integrate with TTS
    setTimeout(() => this.setState('idle'), 3000);
  }
  
  listen() {
    this.setState('listening');
  }
  
  celebrate() {
    this.setState('celebrating');
    setTimeout(() => this.setState('happy'), 2000);
    setTimeout(() => this.setState('idle'), 4000);
  }
}

// Initialize globally
window.kellyController = new KellyAvatarController();

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyAvatarController;
}
