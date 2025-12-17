/**
 * Phase Visual V2 - Strategic UI Integration
 * 
 * Displays phase-aligned educational visuals with:
 * - Clean integration into lesson layout
 * - Responsive sizing with guardrails
 * - Smooth transitions between phases
 * - Fallback handling for missing visuals
 */

(function() {
  'use strict';

  // ============================================================================
  // CONFIGURATION
  // ============================================================================

  const CONFIG = {
    SUPABASE_URL: 'https://tvjalxxsyryjphkforjv.supabase.co',
    SUPABASE_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI',
    STYLE_VERSION: 'curious-kelly-v2', // Prefer V2 style
    FALLBACK_STYLES: ['curious-kelly-v2', 'default', 'artistic', 'minimal'],
    CACHE_TTL_MS: 5 * 60 * 1000, // 5 minute cache
  };

  // Phase mapping from UI to database
  const PHASE_MAP = {
    0: 'hook',
    1: 'cliff', 
    2: 'q1',
    3: 'q2',
    4: 'q3',
    5: 'wisdom',
    6: 'outro'
  };

  // ============================================================================
  // STATE
  // ============================================================================

  let currentDayNumber = null;
  let currentPhase = null;
  let visualCache = new Map(); // key: "day-phase" -> { url, timestamp }
  let containerElement = null;
  let isVisible = false;

  // ============================================================================
  // STYLES
  // ============================================================================

  const STYLES = `
    /* Phase Visual Container - Strategic Placement */
    .phase-visual-container {
      position: absolute;
      top: 80px;
      left: 50%;
      transform: translateX(-50%);
      width: calc(100% - 32px);
      max-width: 800px;
      z-index: 50;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.4s ease, transform 0.4s ease;
    }

    .phase-visual-container.visible {
      opacity: 1;
      pointer-events: auto;
    }

    .phase-visual-container.hidden {
      opacity: 0;
      transform: translateX(-50%) translateY(-10px);
    }

    /* The Visual Image */
    .phase-visual-image {
      width: 100%;
      max-height: 40vh;
      object-fit: cover;
      object-position: center left; /* Keep main content visible */
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }

    .phase-visual-image.loading {
      min-height: 200px;
      background: linear-gradient(90deg, #1a1a2e 0%, #2a2a4e 50%, #1a1a2e 100%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
    }

    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    /* Attribution badge */
    .phase-visual-badge {
      position: absolute;
      bottom: 8px;
      right: 8px;
      background: rgba(0, 0, 0, 0.6);
      color: white;
      font-size: 10px;
      padding: 4px 8px;
      border-radius: 4px;
      backdrop-filter: blur(4px);
    }

    /* Expand button */
    .phase-visual-expand {
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(0, 0, 0, 0.5);
      border: none;
      color: white;
      width: 32px;
      height: 32px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      transition: opacity 0.2s;
      pointer-events: auto;
    }

    .phase-visual-container:hover .phase-visual-expand {
      opacity: 1;
    }

    /* Mobile adjustments */
    @media (max-width: 768px) {
      .phase-visual-container {
        top: 70px;
        width: calc(100% - 16px);
      }

      .phase-visual-image {
        max-height: 30vh;
        border-radius: 12px;
      }
    }

    /* When in fullscreen/expanded mode */
    .phase-visual-fullscreen {
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: rgba(0, 0, 0, 0.95);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 40px;
    }

    .phase-visual-fullscreen img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border-radius: 8px;
    }

    .phase-visual-fullscreen-close {
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(255, 255, 255, 0.2);
      border: none;
      color: white;
      width: 48px;
      height: 48px;
      border-radius: 50%;
      font-size: 24px;
      cursor: pointer;
    }

    /* Hide visual when lesson detail panel is open */
    body.lesson-detail-open .phase-visual-container {
      opacity: 0 !important;
      pointer-events: none !important;
    }
  `;

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  function init() {
    injectStyles();
    createContainer();
    setupEventListeners();
    console.log('[PhaseVisual V2] Initialized');
  }

  function injectStyles() {
    if (document.getElementById('phase-visual-v2-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'phase-visual-v2-styles';
    style.textContent = STYLES;
    document.head.appendChild(style);
  }

  function createContainer() {
    if (containerElement) return;

    containerElement = document.createElement('div');
    containerElement.className = 'phase-visual-container hidden';
    containerElement.innerHTML = `
      <img class="phase-visual-image loading" alt="Phase illustration" />
      <button class="phase-visual-expand" title="Expand">⛶</button>
      <span class="phase-visual-badge">✨ Visual</span>
    `;

    // Find the lesson scene and insert
    const sceneLesson = document.getElementById('scene-lesson');
    if (sceneLesson) {
      sceneLesson.appendChild(containerElement);
    }

    // Setup expand button
    containerElement.querySelector('.phase-visual-expand').addEventListener('click', expandVisual);
  }

  function setupEventListeners() {
    // Listen for phase changes
    document.addEventListener('phaseChange', (e) => {
      const { phase, dayNumber } = e.detail || {};
      if (dayNumber && phase !== undefined) {
        showVisualForPhase(dayNumber, phase);
      }
    });

    // Listen for lesson load
    document.addEventListener('lessonLoaded', (e) => {
      const { dayNumber } = e.detail || {};
      if (dayNumber) {
        currentDayNumber = dayNumber;
        preloadVisualsForDay(dayNumber);
      }
    });

    // Listen for lesson detail panel toggle
    const detailPanel = document.getElementById('lesson-detail-panel');
    if (detailPanel) {
      const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          if (mutation.attributeName === 'class') {
            const isOpen = detailPanel.classList.contains('open');
            document.body.classList.toggle('lesson-detail-open', isOpen);
          }
        }
      });
      observer.observe(detailPanel, { attributes: true });
    }
  }

  // ============================================================================
  // VISUAL FETCHING
  // ============================================================================

  async function fetchVisualForPhase(dayNumber, phaseIndex) {
    const phaseName = PHASE_MAP[phaseIndex] || 'hook';
    const cacheKey = `${dayNumber}-${phaseName}`;

    // Check cache
    const cached = visualCache.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp) < CONFIG.CACHE_TTL_MS) {
      return cached.url;
    }

    // Fetch from Supabase
    try {
      // Try preferred style first, then fallbacks
      for (const style of CONFIG.FALLBACK_STYLES) {
        const response = await fetch(
          `${CONFIG.SUPABASE_URL}/rest/v1/visual_commons?` +
          `day_number=eq.${dayNumber}&phase=eq.${phaseName}&style=eq.${style}&status=eq.active&select=public_url&limit=1`,
          {
            headers: {
              'apikey': CONFIG.SUPABASE_KEY,
              'Authorization': `Bearer ${CONFIG.SUPABASE_KEY}`
            }
          }
        );

        if (response.ok) {
          const data = await response.json();
          if (data.length > 0 && data[0].public_url) {
            const url = data[0].public_url;
            visualCache.set(cacheKey, { url, timestamp: Date.now() });
            return url;
          }
        }
      }

      // No visual found in any style - try without style filter
      const fallbackResponse = await fetch(
        `${CONFIG.SUPABASE_URL}/rest/v1/visual_commons?` +
        `day_number=eq.${dayNumber}&phase=eq.${phaseName}&status=eq.active&select=public_url&limit=1`,
        {
          headers: {
            'apikey': CONFIG.SUPABASE_KEY,
            'Authorization': `Bearer ${CONFIG.SUPABASE_KEY}`
          }
        }
      );

      if (fallbackResponse.ok) {
        const data = await fallbackResponse.json();
        if (data.length > 0 && data[0].public_url) {
          const url = data[0].public_url;
          visualCache.set(cacheKey, { url, timestamp: Date.now() });
          return url;
        }
      }

      return null;
    } catch (error) {
      console.warn('[PhaseVisual V2] Fetch error:', error);
      return null;
    }
  }

  async function preloadVisualsForDay(dayNumber) {
    // Preload all phases in background
    const phases = Object.keys(PHASE_MAP);
    for (const phaseIndex of phases) {
      fetchVisualForPhase(dayNumber, parseInt(phaseIndex, 10));
    }
  }

  // ============================================================================
  // DISPLAY
  // ============================================================================

  async function showVisualForPhase(dayNumber, phaseIndex) {
    if (!containerElement) return;

    currentDayNumber = dayNumber;
    currentPhase = phaseIndex;

    const img = containerElement.querySelector('.phase-visual-image');
    
    // Show loading state
    img.classList.add('loading');
    containerElement.classList.remove('hidden');
    containerElement.classList.add('visible');

    // Fetch the visual
    const url = await fetchVisualForPhase(dayNumber, phaseIndex);

    if (url) {
      img.onload = () => {
        img.classList.remove('loading');
      };
      img.onerror = () => {
        hideVisual();
      };
      img.src = url;
      isVisible = true;
    } else {
      // No visual available - hide gracefully
      hideVisual();
    }
  }

  function hideVisual() {
    if (!containerElement) return;
    containerElement.classList.add('hidden');
    containerElement.classList.remove('visible');
    isVisible = false;
  }

  function expandVisual() {
    const img = containerElement.querySelector('.phase-visual-image');
    if (!img.src || img.classList.contains('loading')) return;

    const fullscreen = document.createElement('div');
    fullscreen.className = 'phase-visual-fullscreen';
    fullscreen.innerHTML = `
      <img src="${img.src}" alt="Expanded visual" />
      <button class="phase-visual-fullscreen-close">×</button>
    `;

    fullscreen.addEventListener('click', (e) => {
      if (e.target === fullscreen || e.target.classList.contains('phase-visual-fullscreen-close')) {
        fullscreen.remove();
      }
    });

    document.body.appendChild(fullscreen);
  }

  // ============================================================================
  // PUBLIC API
  // ============================================================================

  window.PhaseVisualV2 = {
    show: showVisualForPhase,
    hide: hideVisual,
    preload: preloadVisualsForDay,
    isVisible: () => isVisible
  };

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
