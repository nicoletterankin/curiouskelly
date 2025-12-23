/**
 * Universal Lesson Badge
 * 
 * Shows available languages, age groups, and tones for each lesson.
 * Clickable → opens adaptation demo modal.
 * 
 * Usage:
 *   UniversalBadge.render(lessonData, containerElement);
 */

(function() {
  'use strict';

  /**
   * Extract variant info from lesson data
   */
  function extractVariants(lessonData) {
    const languages = new Set(['en']); // English always available
    const ageBuckets = new Set(['adult']); // Default
    const archetypes = new Set(['explorer']); // Default
    
    // Check meta for available variants
    if (lessonData?.meta?.languages) {
      lessonData.meta.languages.forEach(lang => languages.add(lang));
    }
    
    if (lessonData?.meta?.ageBuckets) {
      lessonData.meta.ageBuckets.forEach(age => ageBuckets.add(age));
    }
    
    if (lessonData?.meta?.archetypes) {
      lessonData.meta.archetypes.forEach(arch => archetypes.add(arch));
    }
    
    // Check phases for language variants
    if (lessonData?.phases) {
      Object.values(lessonData.phases).forEach(phase => {
        if (phase?.kelly_says) {
          if (typeof phase.kelly_says === 'object') {
            Object.keys(phase.kelly_says).forEach(lang => languages.add(lang));
          }
        }
      });
    }
    
    return {
      languages: Array.from(languages),
      ageBuckets: Array.from(ageBuckets),
      archetypes: Array.from(archetypes),
    };
  }

  /**
   * Render universal badge
   */
  function render(lessonData, container) {
    if (!lessonData || !container) return;
    
    const variants = extractVariants(lessonData);
    const dayNumber = lessonData.day_number || lessonData.dayNumber || 1;
    
    const badge = document.createElement('div');
    badge.className = 'universal-lesson-badge';
    badge.innerHTML = `
      <div class="universal-badge-content">
        <span class="universal-badge-item">🌍 ${variants.languages.length} ${variants.languages.length === 1 ? 'language' : 'languages'}</span>
        <span class="universal-badge-item">👶 ${variants.ageBuckets.length} ${variants.ageBuckets.length === 1 ? 'age group' : 'age groups'}</span>
        <span class="universal-badge-item">🎭 ${variants.archetypes.length} ${variants.archetypes.length === 1 ? 'tone' : 'tones'}</span>
      </div>
      <button class="universal-badge-btn" onclick="UniversalBadge.showDemo(${dayNumber})" title="See how this lesson adapts">
        See how it adapts →
      </button>
    `;
    
    // Inject styles if not already present
    injectStyles();
    
    container.appendChild(badge);
  }

  /**
   * Show adaptation demo modal
   */
  async function showDemo(dayNumber) {
    // Close existing modal if open
    const existing = document.getElementById('adaptation-demo-modal');
    if (existing) existing.remove();
    
    // Create modal
    const modal = document.createElement('div');
    modal.id = 'adaptation-demo-modal';
    modal.className = 'adaptation-demo-modal';
    modal.innerHTML = `
      <div class="demo-modal-backdrop" onclick="UniversalBadge.closeDemo()"></div>
      <div class="demo-modal-content">
        <div class="demo-modal-header">
          <h2>How Day ${dayNumber} Adapts</h2>
          <button class="demo-modal-close" onclick="UniversalBadge.closeDemo()">×</button>
        </div>
        <div class="demo-modal-body" id="demo-modal-body">
          <div style="text-align: center; padding: 40px;">
            <div class="loading-spinner"></div>
            <div style="margin-top: 16px; color: rgba(255,255,255,0.6);">Loading lesson variants...</div>
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    injectStyles();
    
    // Load lesson data for all languages
    try {
      await loadDemoContent(dayNumber, modal);
    } catch (e) {
      console.error('[UniversalBadge] Demo load failed:', e);
      modal.querySelector('#demo-modal-body').innerHTML = `
        <div style="padding: 40px; text-align: center; color: rgba(255,255,255,0.6);">
          Failed to load adaptation demo. Please try again.
        </div>
      `;
    }
    
    // Escape key to close
    const escapeHandler = (e) => {
      if (e.key === 'Escape') {
        closeDemo();
        document.removeEventListener('keydown', escapeHandler);
      }
    };
    document.addEventListener('keydown', escapeHandler);
  }

  /**
   * Load demo content showing variants
   */
  async function loadDemoContent(dayNumber, modal) {
    const body = modal.querySelector('#demo-modal-body');
    
    // Try to load English lesson first
    let enLesson = null;
    try {
      const enRes = await fetch(`/lessons/day-${String(dayNumber).padStart(3, '0')}.json`);
      if (enRes.ok) {
        enLesson = await enRes.json();
      }
    } catch (e) {
      console.warn('[UniversalBadge] Failed to load English lesson:', e);
    }
    
    // Try Spanish
    let esLesson = null;
    try {
      const esRes = await fetch(`/lessons/es/day-${String(dayNumber).padStart(3, '0')}.json`);
      if (esRes.ok) {
        esLesson = await esRes.json();
      }
    } catch (e) {
      // Spanish not available, that's okay
    }
    
    // Try Portuguese
    let ptLesson = null;
    try {
      const ptRes = await fetch(`/lessons/pt/day-${String(dayNumber).padStart(3, '0')}.json`);
      if (ptRes.ok) {
        ptLesson = await ptRes.json();
      }
    } catch (e) {
      // Portuguese not available, that's okay
    }
    
    if (!enLesson) {
      body.innerHTML = `
        <div style="padding: 40px; text-align: center; color: rgba(255,255,255,0.6);">
          Lesson data not available.
        </div>
      `;
      return;
    }
    
    // Render comparison
    const hookPhase = enLesson.phases?.hook || enLesson.phases?.Hook;
    const esHook = esLesson?.phases?.hook || esLesson?.phases?.Hook;
    const ptHook = ptLesson?.phases?.hook || ptLesson?.phases?.Hook;
    
    body.innerHTML = `
      <div class="demo-tabs">
        <button class="demo-tab active" data-tab="language">🌍 Language</button>
        <button class="demo-tab" data-tab="age">👶 Age</button>
        <button class="demo-tab" data-tab="tone">🎭 Tone</button>
      </div>
      
      <div class="demo-content" id="demo-content">
        <!-- Language comparison -->
        <div class="demo-panel active" data-panel="language">
          <div class="demo-comparison-grid">
            <div class="demo-column">
              <div class="demo-column-header">
                <span class="demo-flag">🇺🇸</span>
                <strong>English</strong>
              </div>
              <div class="demo-text">
                ${typeof hookPhase?.kelly_says === 'string' ? hookPhase.kelly_says : (hookPhase?.kelly_says?.en || hookPhase?.script || 'Loading...')}
              </div>
            </div>
            <div class="demo-column">
              <div class="demo-column-header">
                <span class="demo-flag">🇪🇸</span>
                <strong>Español</strong>
                ${!esLesson ? '<span class="demo-badge-coming">Coming soon</span>' : ''}
              </div>
              <div class="demo-text">
                ${esLesson ? (typeof esHook?.kelly_says === 'string' ? esHook.kelly_says : (esHook?.kelly_says?.es || esHook?.script || 'Not available')) : 'Spanish translation coming soon!'}
              </div>
            </div>
            <div class="demo-column">
              <div class="demo-column-header">
                <span class="demo-flag">🇵🇹</span>
                <strong>Português</strong>
                ${!ptLesson ? '<span class="demo-badge-coming">Coming soon</span>' : ''}
              </div>
              <div class="demo-text">
                ${ptLesson ? (typeof ptHook?.kelly_says === 'string' ? ptHook.kelly_says : (ptHook?.kelly_says?.pt || ptHook?.script || 'Not available')) : 'Portuguese translation coming soon!'}
              </div>
            </div>
          </div>
        </div>
        
        <!-- Age comparison (placeholder) -->
        <div class="demo-panel" data-panel="age">
          <div style="padding: 40px; text-align: center; color: rgba(255,255,255,0.6);">
            Age adaptation demo coming soon!
          </div>
        </div>
        
        <!-- Tone comparison (placeholder) -->
        <div class="demo-panel" data-panel="tone">
          <div style="padding: 40px; text-align: center; color: rgba(255,255,255,0.6);">
            Tone adaptation demo coming soon!
          </div>
        </div>
      </div>
      
      <div class="demo-footer">
        <button class="demo-btn-primary" onclick="window.location.href='/learn.html?day=${dayNumber}'">
          Try This Lesson
        </button>
        <button class="demo-btn-secondary" onclick="UniversalBadge.closeDemo()">
          Close
        </button>
      </div>
    `;
    
    // Attach tab handlers
    body.querySelectorAll('.demo-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.getAttribute('data-tab');
        body.querySelectorAll('.demo-tab').forEach(t => t.classList.remove('active'));
        body.querySelectorAll('.demo-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        body.querySelector(`[data-panel="${tabName}"]`)?.classList.add('active');
      });
    });
  }

  /**
   * Close demo modal
   */
  function closeDemo() {
    const modal = document.getElementById('adaptation-demo-modal');
    if (modal) modal.remove();
  }

  /**
   * Inject styles
   */
  function injectStyles() {
    if (document.getElementById('universal-badge-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'universal-badge-styles';
    style.textContent = `
      .universal-lesson-badge {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        background: rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 8px;
        font-size: 12px;
        margin-top: 8px;
      }
      .universal-badge-content {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }
      .universal-badge-item {
        color: rgba(255,255,255,0.8);
      }
      .universal-badge-btn {
        padding: 4px 10px;
        background: var(--kelly-blue, #2563eb);
        border: none;
        border-radius: 6px;
        color: white;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s;
      }
      .universal-badge-btn:hover {
        background: var(--kelly-blue-light, #3b82f6);
      }
      
      /* Adaptation Demo Modal */
      .adaptation-demo-modal {
        position: fixed;
        inset: 0;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .demo-modal-backdrop {
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(4px);
      }
      .demo-modal-content {
        position: relative;
        background: var(--surface-base, #18181b);
        border-radius: 16px;
        max-width: 900px;
        max-height: 90vh;
        width: 90%;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
      }
      .demo-modal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 24px;
        border-bottom: 1px solid var(--border-default, #3f3f46);
      }
      .demo-modal-header h2 {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
      }
      .demo-modal-close {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        border: none;
        background: rgba(255,255,255,0.1);
        color: white;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.2s;
      }
      .demo-modal-close:hover {
        background: rgba(255,255,255,0.2);
      }
      .demo-modal-body {
        padding: 24px;
        overflow-y: auto;
        flex: 1;
      }
      .demo-tabs {
        display: flex;
        gap: 8px;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border-default, #3f3f46);
      }
      .demo-tab {
        padding: 10px 16px;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: rgba(255,255,255,0.6);
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
      }
      .demo-tab.active {
        color: white;
        border-bottom-color: var(--kelly-blue, #2563eb);
      }
      .demo-tab:hover {
        color: white;
      }
      .demo-panel {
        display: none;
      }
      .demo-panel.active {
        display: block;
      }
      .demo-comparison-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
      }
      .demo-column {
        padding: 16px;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
      }
      .demo-column-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        font-size: 14px;
        font-weight: 600;
      }
      .demo-flag {
        font-size: 20px;
      }
      .demo-badge-coming {
        margin-left: auto;
        padding: 2px 8px;
        background: rgba(245,158,11,0.2);
        color: var(--gold-wisdom, #f59e0b);
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
      }
      .demo-text {
        line-height: 1.6;
        color: rgba(255,255,255,0.9);
        font-size: 14px;
      }
      .demo-footer {
        display: flex;
        gap: 12px;
        padding: 20px 24px;
        border-top: 1px solid var(--border-default, #3f3f46);
      }
      .demo-btn-primary {
        flex: 1;
        padding: 12px 20px;
        background: var(--kelly-blue, #2563eb);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      .demo-btn-primary:hover {
        background: var(--kelly-blue-light, #3b82f6);
      }
      .demo-btn-secondary {
        padding: 12px 20px;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: white;
        cursor: pointer;
        transition: all 0.2s;
      }
      .demo-btn-secondary:hover {
        background: rgba(255,255,255,0.15);
      }
    `;
    document.head.appendChild(style);
  }

  // Expose API
  window.UniversalBadge = {
    render,
    showDemo,
    closeDemo,
    extractVariants,
  };

})();

