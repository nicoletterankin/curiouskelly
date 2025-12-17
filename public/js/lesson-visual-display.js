/**
 * LESSON VISUAL DISPLAY
 * 
 * Integrates Visual Commons into the lesson flow.
 * Shows phase-appropriate visuals during lessons automatically.
 * 
 * @created December 17, 2025
 */

(function() {
  'use strict';

  // Configuration
  const CONFIG = {
    SUPABASE_URL: window.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co',
    CACHE_TTL_MS: 1000 * 60 * 60, // 1 hour
    FADE_DURATION_MS: 400
  };

  // Visual cache
  const visualCache = new Map();

  /**
   * Fetch visuals for a specific day and phase
   */
  async function fetchVisualsForPhase(dayNumber, phase) {
    const cacheKey = `${dayNumber}-${phase}`;
    
    // Check cache
    const cached = visualCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_TTL_MS) {
      return cached.visuals;
    }

    try {
      // Query Supabase directly (no API needed for public read)
      const url = `${CONFIG.SUPABASE_URL}/rest/v1/visual_commons?day_number=eq.${dayNumber}&phase=eq.${phase}&status=eq.active&select=id,public_url,style,model_used,unique_learners_helped&order=unique_learners_helped.desc&limit=10`;
      
      const response = await fetch(url, {
        headers: {
          'apikey': window.SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI'
        }
      });

      if (!response.ok) {
        console.warn('Failed to fetch visuals:', response.status);
        return [];
      }

      const visuals = await response.json();
      
      // Cache the result
      visualCache.set(cacheKey, { visuals, timestamp: Date.now() });
      
      return visuals;
    } catch (error) {
      console.error('Error fetching visuals:', error);
      return [];
    }
  }

  /**
   * Get the best visual for display (prioritize by style preference)
   */
  function selectBestVisual(visuals, preferredStyle = 'artistic') {
    if (!visuals || visuals.length === 0) return null;
    
    // First try to find preferred style
    const preferred = visuals.find(v => v.style === preferredStyle);
    if (preferred) return preferred;
    
    // Otherwise return most popular
    return visuals[0];
  }

  /**
   * Create the visual display element
   */
  function createVisualDisplay() {
    // Check if already exists
    if (document.getElementById('lesson-phase-visual')) {
      return document.getElementById('lesson-phase-visual');
    }

    const container = document.createElement('div');
    container.id = 'lesson-phase-visual';
    container.className = 'lesson-phase-visual';
    container.innerHTML = `
      <div class="phase-visual-inner">
        <img class="phase-visual-img" alt="Phase visual" />
        <div class="phase-visual-badge">
          <span class="badge-style"></span>
          <span class="badge-helped"></span>
        </div>
        <button class="phase-visual-expand" aria-label="Expand visual">⛶</button>
        <button class="phase-visual-variants" aria-label="More styles">🎨</button>
      </div>
    `;

    // Add styles
    if (!document.getElementById('lesson-visual-display-styles')) {
      const style = document.createElement('style');
      style.id = 'lesson-visual-display-styles';
      style.textContent = `
        .lesson-phase-visual {
          position: fixed;
          bottom: 180px;
          right: 16px;
          width: 180px;
          height: 100px;
          z-index: 50;
          opacity: 0;
          transform: translateX(20px) scale(0.95);
          transition: opacity 0.4s ease, transform 0.4s ease;
          pointer-events: none;
        }
        
        .lesson-phase-visual.visible {
          opacity: 1;
          transform: translateX(0) scale(1);
          pointer-events: auto;
        }
        
        .lesson-phase-visual.expanded {
          bottom: auto;
          right: auto;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%) scale(1);
          width: 90vw;
          max-width: 800px;
          height: auto;
          max-height: 80vh;
          z-index: 1000;
        }
        
        .lesson-phase-visual.expanded .phase-visual-img {
          border-radius: 16px;
          width: 100%;
          height: auto;
        }
        
        .phase-visual-inner {
          position: relative;
          width: 100%;
          height: 100%;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1);
          cursor: pointer;
          background: rgba(15, 15, 20, 0.9);
        }
        
        .phase-visual-img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          border-radius: 12px;
          transition: opacity 0.3s ease;
        }
        
        .phase-visual-img[src=""] {
          opacity: 0;
        }
        
        .phase-visual-badge {
          position: absolute;
          bottom: 6px;
          left: 6px;
          display: flex;
          gap: 4px;
          font-size: 10px;
        }
        
        .phase-visual-badge span {
          background: rgba(0,0,0,0.6);
          color: white;
          padding: 2px 6px;
          border-radius: 4px;
          backdrop-filter: blur(4px);
        }
        
        .phase-visual-expand,
        .phase-visual-variants {
          position: absolute;
          top: 6px;
          background: rgba(0,0,0,0.5);
          color: white;
          border: none;
          width: 28px;
          height: 28px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          opacity: 0;
          transition: opacity 0.2s;
        }
        
        .phase-visual-expand { right: 6px; }
        .phase-visual-variants { right: 40px; }
        
        .phase-visual-inner:hover .phase-visual-expand,
        .phase-visual-inner:hover .phase-visual-variants {
          opacity: 1;
        }
        
        .lesson-phase-visual.expanded .phase-visual-expand::before {
          content: '✕';
        }
        
        /* Variant selector popup */
        .visual-variants-popup {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: rgba(20, 20, 30, 0.95);
          border-radius: 16px;
          padding: 20px;
          z-index: 1001;
          max-width: 90vw;
          max-height: 80vh;
          overflow-y: auto;
          display: none;
        }
        
        .visual-variants-popup.visible {
          display: block;
        }
        
        .visual-variants-popup h3 {
          margin: 0 0 16px;
          color: white;
          font-size: 18px;
        }
        
        .variants-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
          gap: 12px;
        }
        
        .variant-card {
          border-radius: 12px;
          overflow: hidden;
          cursor: pointer;
          transition: transform 0.2s, box-shadow 0.2s;
          background: rgba(255,255,255,0.05);
        }
        
        .variant-card:hover {
          transform: scale(1.03);
          box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }
        
        .variant-card.selected {
          outline: 3px solid #3b82f6;
        }
        
        .variant-card img {
          width: 100%;
          aspect-ratio: 16/9;
          object-fit: cover;
        }
        
        .variant-card-info {
          padding: 8px;
          font-size: 11px;
          color: rgba(255,255,255,0.8);
        }
        
        .variant-style-badge {
          display: inline-block;
          background: rgba(59, 130, 246, 0.3);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 10px;
          margin-bottom: 4px;
        }
        
        .variants-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.7);
          z-index: 1000;
          display: none;
        }
        
        .variants-backdrop.visible {
          display: block;
        }
        
        @media (max-width: 768px) {
          .lesson-phase-visual {
            bottom: 160px;
            right: 8px;
            width: 120px;
            height: 68px;
          }
          
          .phase-visual-badge {
            font-size: 8px;
          }
          
          .phase-visual-expand,
          .phase-visual-variants {
            width: 24px;
            height: 24px;
            font-size: 12px;
          }
          
          .variants-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `;
      document.head.appendChild(style);
    }

    // Insert into lesson scene
    const lessonScene = document.getElementById('scene-lesson');
    if (lessonScene) {
      lessonScene.appendChild(container);
    }

    // Event handlers
    const expandBtn = container.querySelector('.phase-visual-expand');
    const variantsBtn = container.querySelector('.phase-visual-variants');
    const inner = container.querySelector('.phase-visual-inner');

    expandBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      container.classList.toggle('expanded');
    });

    variantsBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      showVariantsPopup();
    });

    inner?.addEventListener('click', () => {
      container.classList.toggle('expanded');
    });

    return container;
  }

  /**
   * Show variants popup
   */
  let currentVisuals = [];
  let currentSelectedUrl = '';

  function showVariantsPopup() {
    if (currentVisuals.length <= 1) return;

    // Create backdrop
    let backdrop = document.querySelector('.variants-backdrop');
    if (!backdrop) {
      backdrop = document.createElement('div');
      backdrop.className = 'variants-backdrop';
      document.body.appendChild(backdrop);
      backdrop.addEventListener('click', hideVariantsPopup);
    }

    // Create popup
    let popup = document.querySelector('.visual-variants-popup');
    if (!popup) {
      popup = document.createElement('div');
      popup.className = 'visual-variants-popup';
      document.body.appendChild(popup);
    }

    popup.innerHTML = `
      <h3>🎨 Choose Your Visual Style</h3>
      <div class="variants-grid">
        ${currentVisuals.map(v => `
          <div class="variant-card ${v.public_url === currentSelectedUrl ? 'selected' : ''}" data-url="${v.public_url}">
            <img src="${v.public_url}" alt="${v.style}" loading="lazy" />
            <div class="variant-card-info">
              <span class="variant-style-badge">${v.style}</span>
              ${v.unique_learners_helped > 0 ? `<br>Helped ${v.unique_learners_helped} learners` : ''}
            </div>
          </div>
        `).join('')}
      </div>
    `;

    // Add click handlers
    popup.querySelectorAll('.variant-card').forEach(card => {
      card.addEventListener('click', () => {
        const url = card.dataset.url;
        selectVisual(url);
        hideVariantsPopup();
      });
    });

    backdrop.classList.add('visible');
    popup.classList.add('visible');
  }

  function hideVariantsPopup() {
    document.querySelector('.variants-backdrop')?.classList.remove('visible');
    document.querySelector('.visual-variants-popup')?.classList.remove('visible');
  }

  function selectVisual(url) {
    const visual = currentVisuals.find(v => v.public_url === url);
    if (visual) {
      currentSelectedUrl = url;
      updateVisualDisplay(visual);
      
      // Save preference
      localStorage.setItem('preferred_visual_style', visual.style);
    }
  }

  /**
   * Update the visual display with a specific visual
   */
  function updateVisualDisplay(visual) {
    const container = document.getElementById('lesson-phase-visual');
    if (!container) return;

    const img = container.querySelector('.phase-visual-img');
    const styleBadge = container.querySelector('.badge-style');
    const helpedBadge = container.querySelector('.badge-helped');

    if (img) {
      img.src = visual.public_url;
      img.alt = `${visual.style} visual`;
    }

    if (styleBadge) {
      styleBadge.textContent = visual.style;
    }

    if (helpedBadge) {
      helpedBadge.textContent = visual.unique_learners_helped > 0 
        ? `👁 ${visual.unique_learners_helped}` 
        : '';
    }
  }

  /**
   * Show visual for a specific phase
   */
  async function showPhaseVisual(dayNumber, phase) {
    const container = createVisualDisplay();
    
    // Get preferred style from localStorage
    const preferredStyle = localStorage.getItem('preferred_visual_style') || 'artistic';
    
    // Fetch visuals
    const visuals = await fetchVisualsForPhase(dayNumber, phase);
    currentVisuals = visuals;
    
    if (visuals.length === 0) {
      container.classList.remove('visible');
      return;
    }

    // Select best visual
    const best = selectBestVisual(visuals, preferredStyle);
    currentSelectedUrl = best.public_url;
    
    // Update display
    updateVisualDisplay(best);
    
    // Show with animation
    container.classList.remove('expanded');
    container.classList.add('visible');

    // Show variants button only if multiple exist
    const variantsBtn = container.querySelector('.phase-visual-variants');
    if (variantsBtn) {
      variantsBtn.style.display = visuals.length > 1 ? 'block' : 'none';
    }
  }

  /**
   * Hide visual display
   */
  function hidePhaseVisual() {
    const container = document.getElementById('lesson-phase-visual');
    if (container) {
      container.classList.remove('visible');
      container.classList.remove('expanded');
    }
  }

  /**
   * Hook into phase changes
   */
  function setupPhaseListener() {
    // Listen for phase change events (custom event dispatch)
    window.addEventListener('phaseChange', (e) => {
      const { dayNumber, phase } = e.detail || {};
      if (dayNumber && phase) {
        showPhaseVisual(dayNumber, phase);
      }
    });
    
    // Hide visual when lesson detail panel opens
    document.addEventListener('click', (e) => {
      if (e.target.closest('.phase-bar-top') || e.target.closest('#phase-label')) {
        // Lesson detail panel is toggling - hide visual temporarily
        const panel = document.getElementById('lesson-detail-panel');
        if (panel?.classList.contains('visible')) {
          hidePhaseVisual();
        }
      }
    });
    
    // Show visual again when panel closes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          const panel = document.getElementById('lesson-detail-panel');
          if (panel && !panel.classList.contains('visible')) {
            // Panel closed - show visual again
            const dayNumber = window.state?.currentDay || 1;
            const phases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'];
            const phase = phases[window.state?.currentPhase || 0] || 'hook';
            showPhaseVisual(dayNumber, phase);
          }
        }
      });
    });
    
    setTimeout(() => {
      const panel = document.getElementById('lesson-detail-panel');
      if (panel) {
        observer.observe(panel, { attributes: true });
      }
    }, 1000);
  }

  // Expose globally - learn.html calls these directly
  window.LessonVisualDisplay = {
    show: showPhaseVisual,
    hide: hidePhaseVisual,
    showVariants: showVariantsPopup
  };

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupPhaseListener);
  } else {
    setupPhaseListener();
  }

  console.log('✅ Lesson Visual Display loaded');
})();
