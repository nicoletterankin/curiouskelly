/**
 * VISUAL COMMONS CONTROLLER
 * 
 * Manages on-demand visual generation for lesson phases.
 * Checks cache first, generates if needed, saves for everyone.
 * 
 * @version 1.0.0
 * @created December 17, 2025
 */

class VisualCommonsController {
  constructor() {
    this.cache = new Map();
    this.currentGeneration = null;
    this.container = null;
    this.elements = {};
    this.context = {};
  }

  /**
   * Initialize the controller with a container element
   */
  async init(container) {
    this.container = container;
    this.elements = {
      loading: container.querySelector('.visual-loading'),
      cached: container.querySelector('.visual-cached'),
      generate: container.querySelector('.visual-generate'),
      generating: container.querySelector('.visual-generating'),
      complete: container.querySelector('.visual-complete'),
      error: container.querySelector('.visual-error')
    };

    // Bind event handlers
    const generateBtn = container.querySelector('.generate-cta');
    if (generateBtn) {
      generateBtn.addEventListener('click', () => this.generate());
    }

    const retryBtn = container.querySelector('.retry-button');
    if (retryBtn) {
      retryBtn.addEventListener('click', () => this.generate());
    }

    const expandBtn = container.querySelector('.visual-expand');
    if (expandBtn) {
      expandBtn.addEventListener('click', () => this.expand());
    }

    // Check for cached visual
    await this.check();
  }

  /**
   * Get context from container data attributes and global state
   */
  getContext() {
    return {
      dayNumber: parseInt(this.container.dataset.day) || window.currentDay || 1,
      phase: this.container.dataset.phase || 'hook',
      ageGroup: window.kellyState?.ageGroup || localStorage.getItem('kelly_age_group') || 'all',
      visualType: this.container.dataset.type || 'infographic'
    };
  }

  /**
   * Generate a simple client-side hash for local caching
   */
  generateLocalHash(context) {
    const str = JSON.stringify({
      d: context.dayNumber,
      p: context.phase,
      a: context.ageGroup,
      t: context.visualType
    });
    // Simple hash for client-side cache key
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  /**
   * Show a specific state and hide others
   */
  showState(state) {
    Object.entries(this.elements).forEach(([key, el]) => {
      if (el) {
        el.style.display = key === state ? 'flex' : 'none';
      }
    });
  }

  /**
   * Check if visual exists in cache
   */
  async check() {
    this.showState('loading');
    this.context = this.getContext();
    const localHash = this.generateLocalHash(this.context);

    // Check local memory cache first
    if (this.cache.has(localHash)) {
      this.showCached(this.cache.get(localHash));
      return;
    }

    try {
      const params = new URLSearchParams({
        day: this.context.dayNumber.toString(),
        phase: this.context.phase,
        age: this.context.ageGroup,
        type: this.context.visualType
      });

      const response = await fetch(`/api/visual/check?${params}`);
      const data = await response.json();

      if (data.exists && data.visual) {
        this.cache.set(localHash, data.visual);
        this.showCached(data.visual);
        // Track the view
        this.trackView(data.visual.id);
      } else {
        this.showGenerateOption(data);
      }
    } catch (error) {
      console.error('Visual check failed:', error);
      this.showGenerateOption({ canGenerate: true, keySource: 'platform' });
    }
  }

  /**
   * Generate a new visual
   */
  async generate() {
    if (this.currentGeneration) return;
    this.currentGeneration = true;

    this.showState('generating');
    this.context = this.getContext();

    try {
      const response = await fetch('/api/visual/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dayNumber: this.context.dayNumber,
          phase: this.context.phase,
          ageGroup: this.context.ageGroup,
          visualType: this.context.visualType,
          userApiKey: this.getUserApiKey()
        })
      });

      const data = await response.json();

      if (data.success) {
        const localHash = this.generateLocalHash(this.context);
        const visual = {
          id: data.visual.id,
          publicUrl: data.visual.publicUrl,
          generatedBy: { displayName: 'You', isAnonymous: false },
          helpedCount: 0,
          createdAt: new Date().toISOString()
        };
        this.cache.set(localHash, visual);
        this.showComplete(visual);
        this.trackContribution(visual);
      } else {
        this.showError(data.message || 'Generation failed');
      }
    } catch (error) {
      console.error('Visual generation failed:', error);
      this.showError('Generation failed. Please try again.');
    } finally {
      this.currentGeneration = null;
    }
  }

  /**
   * Get user's BYOK API key from localStorage
   */
  getUserApiKey() {
    return localStorage.getItem('kelly_google_api_key');
  }

  /**
   * Show cached visual
   */
  showCached(visual) {
    this.showState('cached');
    
    const img = this.elements.cached?.querySelector('.visual-image');
    if (img) {
      img.src = visual.publicUrl;
      img.alt = `Educational visual for ${this.context.phase}`;
      img.onerror = () => {
        img.src = '/kelly/placeholder-visual.svg';
      };
    }

    const contributor = this.elements.cached?.querySelector('.contributor strong');
    if (contributor) {
      contributor.textContent = visual.generatedBy?.displayName || 'A curious learner';
    }

    const impact = this.elements.cached?.querySelector('.impact');
    if (impact) {
      const count = visual.helpedCount || 0;
      impact.textContent = count === 1 
        ? '1 learner helped' 
        : `${count.toLocaleString()} learners helped`;
    }
  }

  /**
   * Show generate option
   */
  showGenerateOption(data) {
    this.showState('generate');
    
    const keyType = this.elements.generate?.querySelector('.key-type');
    if (keyType) {
      const hasUserKey = !!this.getUserApiKey();
      keyType.textContent = hasUserKey 
        ? 'Your Google AI credits' 
        : 'Curious Kelly credits';
    }
  }

  /**
   * Show completion celebration
   */
  showComplete(visual) {
    this.showState('complete');
    
    const img = this.elements.complete?.querySelector('.visual-image');
    if (img) {
      img.src = visual.publicUrl;
    }

    // Transition to cached view after celebration
    setTimeout(() => {
      this.showCached(visual);
    }, 3000);
  }

  /**
   * Show error state
   */
  showError(message) {
    this.showState('error');
    const msgEl = this.elements.error?.querySelector('.error-message');
    if (msgEl) {
      msgEl.textContent = message;
    }
  }

  /**
   * Expand visual in overlay
   */
  expand() {
    const img = this.elements.cached?.querySelector('.visual-image');
    if (!img) return;

    // Use existing infographic overlay
    const overlay = document.getElementById('overlay-infographic');
    const display = document.getElementById('infographic-image');
    
    if (overlay && display) {
      display.innerHTML = `<img src="${img.src}" style="max-width:100%; max-height:80vh; border-radius:12px;" alt="${img.alt}">`;
      
      if (typeof openOverlay === 'function') {
        openOverlay('overlay-infographic');
      } else {
        overlay.classList.add('open');
        overlay.setAttribute('aria-hidden', 'false');
      }
    }
  }

  /**
   * Track a view (called when cached visual is displayed)
   */
  async trackView(visualId) {
    try {
      // Fire and forget - don't block UI
      fetch('/api/visual/view', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ visualId })
      }).catch(() => {}); // Ignore errors
    } catch (e) {
      // Ignore tracking errors
    }
  }

  /**
   * Track contribution and show celebration
   */
  trackContribution(visual) {
    // Update local stats
    const stats = JSON.parse(localStorage.getItem('kelly_visual_stats') || '{}');
    stats.totalContributed = (stats.totalContributed || 0) + 1;
    localStorage.setItem('kelly_visual_stats', JSON.stringify(stats));

    // Show toast notification
    if (typeof showToast === 'function') {
      showToast('🎨 Visual saved to the Commons! Future learners will thank you.');
    }
  }
}

// =============================================================================
// VISUAL SLOT TEMPLATE
// =============================================================================

/**
 * Create a visual slot element for a phase
 */
function createVisualSlot(dayNumber, phase) {
  const slot = document.createElement('div');
  slot.className = 'visual-slot';
  slot.dataset.day = dayNumber;
  slot.dataset.phase = phase;
  slot.dataset.type = 'infographic';
  
  slot.innerHTML = `
    <!-- Loading state -->
    <div class="visual-loading" style="display: flex;">
      <div class="visual-shimmer"></div>
      <span>Checking for visual...</span>
    </div>
    
    <!-- Cached visual -->
    <div class="visual-cached" style="display: none;">
      <img class="visual-image" src="" alt="" loading="lazy">
      <div class="visual-attribution">
        <span class="contributor">Contributed by <strong>A curious learner</strong></span>
        <span class="impact">0 learners helped</span>
      </div>
      <button class="visual-expand" aria-label="View full size">⛶</button>
    </div>
    
    <!-- Generate option -->
    <div class="visual-generate" style="display: none;">
      <div class="visual-placeholder">
        <span class="placeholder-icon">📊</span>
        <span class="placeholder-text">No visual yet</span>
      </div>
      <button class="generate-cta">
        <span class="sparkle">✨</span>
        <span>Generate Visual</span>
        <span class="cta-subtext">Be the first to illuminate!</span>
      </button>
      <div class="key-source">
        Using: <span class="key-type">Curious Kelly credits</span>
      </div>
    </div>
    
    <!-- Generating state -->
    <div class="visual-generating" style="display: none;">
      <div class="generation-animation">
        <div class="kelly-sketching">🎨</div>
        <span>Kelly is creating your visual...</span>
      </div>
      <div class="generation-progress">
        <div class="progress-bar"></div>
      </div>
    </div>
    
    <!-- Complete celebration -->
    <div class="visual-complete" style="display: none;">
      <img class="visual-image" src="" alt="">
      <div class="completion-celebration">
        <span class="confetti">🎉</span>
        <span class="message">You illuminated this lesson!</span>
        <span class="impact-preview">Future learners will thank you!</span>
      </div>
    </div>
    
    <!-- Error state -->
    <div class="visual-error" style="display: none;">
      <span class="error-icon">⚠️</span>
      <span class="error-message">Generation unavailable</span>
      <button class="retry-button">Try Again</button>
    </div>
  `;
  
  return slot;
}

/**
 * Initialize a visual slot with its controller
 */
function initializeVisualSlot(container) {
  const controller = new VisualCommonsController();
  controller.init(container);
  return controller;
}

/**
 * Add visual slot to a phase container
 */
function addVisualSlotToPhase(phaseContainer, dayNumber, phase) {
  // Check if slot already exists
  if (phaseContainer.querySelector('.visual-slot')) {
    return;
  }
  
  const slot = createVisualSlot(dayNumber, phase);
  phaseContainer.appendChild(slot);
  return initializeVisualSlot(slot);
}

// =============================================================================
// CSS STYLES (inject if not already present)
// =============================================================================

function injectVisualCommonsStyles() {
  if (document.getElementById('visual-commons-styles')) return;
  
  const styles = document.createElement('style');
  styles.id = 'visual-commons-styles';
  styles.textContent = `
    .visual-slot {
      width: 100%;
      max-width: 400px;
      aspect-ratio: 16/9;
      border-radius: 16px;
      overflow: hidden;
      background: var(--surface-elevated, #18181b);
      border: 1px solid var(--border-default, #27272a);
      position: relative;
      margin: 16px auto;
    }

    .visual-slot > div {
      position: absolute;
      inset: 0;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
    }

    .visual-shimmer {
      width: 60%;
      height: 20px;
      background: linear-gradient(90deg, #27272a 25%, #3f3f46 50%, #27272a 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 4px;
    }

    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    .visual-image {
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.3s ease;
    }

    .visual-slot:hover .visual-image {
      transform: scale(1.02);
    }

    .visual-attribution {
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 12px;
      background: linear-gradient(transparent, rgba(0,0,0,0.85));
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: rgba(255,255,255,0.9);
    }

    .visual-expand {
      position: absolute;
      top: 12px;
      right: 12px;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.2);
      color: white;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s;
    }

    .visual-slot:hover .visual-expand {
      opacity: 1;
    }

    .visual-placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
    }

    .placeholder-icon {
      font-size: 48px;
      opacity: 0.4;
    }

    .placeholder-text {
      font-size: 14px;
      opacity: 0.6;
    }

    .generate-cta {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      padding: 12px 24px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border: none;
      border-radius: 12px;
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .generate-cta:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }

    .generate-cta .sparkle {
      animation: sparkle 1.5s ease-in-out infinite;
    }

    @keyframes sparkle {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.7; transform: scale(1.2); }
    }

    .cta-subtext {
      font-size: 11px;
      opacity: 0.8;
      font-weight: 400;
    }

    .key-source {
      margin-top: 12px;
      font-size: 11px;
      opacity: 0.5;
    }

    .generation-animation {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .kelly-sketching {
      font-size: 48px;
      animation: sketch 0.5s ease-in-out infinite alternate;
    }

    @keyframes sketch {
      from { transform: rotate(-5deg); }
      to { transform: rotate(5deg); }
    }

    .generation-progress {
      width: 200px;
      height: 4px;
      background: #27272a;
      border-radius: 2px;
      overflow: hidden;
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #8b5cf6);
      animation: progress 2s ease-in-out infinite;
    }

    @keyframes progress {
      0% { width: 0%; }
      50% { width: 80%; }
      100% { width: 100%; }
    }

    .completion-celebration {
      position: absolute;
      inset: 0;
      background: rgba(0,0,0,0.85);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      animation: celebrationFadeIn 0.5s ease-out;
    }

    @keyframes celebrationFadeIn {
      from { opacity: 0; transform: scale(0.9); }
      to { opacity: 1; transform: scale(1); }
    }

    .confetti {
      font-size: 48px;
      animation: confettiBounce 0.5s ease-out;
    }

    @keyframes confettiBounce {
      0% { transform: scale(0) rotate(-180deg); }
      60% { transform: scale(1.2) rotate(10deg); }
      100% { transform: scale(1) rotate(0); }
    }

    .completion-celebration .message {
      font-size: 18px;
      font-weight: 600;
      color: white;
    }

    .impact-preview {
      font-size: 14px;
      opacity: 0.7;
      color: white;
    }

    .visual-error {
      color: #f87171;
    }

    .retry-button {
      margin-top: 12px;
      padding: 8px 16px;
      background: transparent;
      border: 1px solid #f87171;
      border-radius: 8px;
      color: #f87171;
      cursor: pointer;
    }

    .retry-button:hover {
      background: rgba(248, 113, 113, 0.1);
    }
  `;
  
  document.head.appendChild(styles);
}

// Auto-inject styles when script loads
if (typeof document !== 'undefined') {
  injectVisualCommonsStyles();
}

// =============================================================================
// EXPORTS (for module usage)
// =============================================================================

if (typeof window !== 'undefined') {
  window.VisualCommonsController = VisualCommonsController;
  window.createVisualSlot = createVisualSlot;
  window.initializeVisualSlot = initializeVisualSlot;
  window.addVisualSlotToPhase = addVisualSlotToPhase;
}
