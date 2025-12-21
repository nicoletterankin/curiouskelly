/**
 * Language Selector Component (Enhanced with Translation Progress)
 * 
 * Provides a dropdown for language selection that integrates with i18n-core.
 * Shows translation progress and optional contribution CTAs for incomplete languages.
 * 
 * Usage:
 *   <div data-language-selector></div>
 *   
 *   Or programmatically:
 *   KellyLanguageSelector.create(containerElement);
 */

(function() {
  'use strict';

  // Language config with translation progress
  const LANGUAGE_CONFIG = {
    en: { name: 'English', flag: '🇺🇸', native: 'English', progress: 100, lessons: 365 },
    es: { name: 'Spanish', flag: '🇪🇸', native: 'Español', progress: 3, lessons: 10, cost: 27 },
    pt: { name: 'Portuguese', flag: '🇧🇷', native: 'Português', progress: 0, lessons: 0, cost: 30 },
    fr: { name: 'French', flag: '🇫🇷', native: 'Français', progress: 0, lessons: 0, cost: 30 },
    de: { name: 'German', flag: '🇩🇪', native: 'Deutsch', progress: 0, lessons: 0, cost: 30 },
    hi: { name: 'Hindi', flag: '🇮🇳', native: 'हिन्दी', progress: 0, lessons: 0, cost: 30 },
  };

  /**
   * Create a language selector dropdown with progress indicators
   */
  function create(container, options = {}) {
    const {
      showFlags = true,
      showNative = true,
      showProgress = true,
      compact = false,
      onChange = null,
    } = options;

    const wrapper = document.createElement('div');
    wrapper.className = 'kelly-language-selector';
    wrapper.style.cssText = `
      position: relative;
      display: inline-block;
      font-family: inherit;
    `;

    const currentLang = window.KellyI18n?.getLanguage() || 'en';
    const currentConfig = LANGUAGE_CONFIG[currentLang] || LANGUAGE_CONFIG.en;

    // Button
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'kelly-language-button';
    button.setAttribute('aria-haspopup', 'listbox');
    button.setAttribute('aria-expanded', 'false');
    button.style.cssText = `
      display: flex;
      align-items: center;
      gap: 8px;
      padding: ${compact ? '6px 10px' : '10px 16px'};
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 8px;
      background: rgba(255,255,255,0.1);
      color: inherit;
      font-size: ${compact ? '14px' : '16px'};
      cursor: pointer;
      transition: all 0.2s ease;
    `;
    button.innerHTML = `
      ${showFlags ? `<span class="flag">${currentConfig.flag}</span>` : ''}
      <span class="label">${showNative ? currentConfig.native : currentConfig.name}</span>
      <span class="arrow" style="font-size: 10px;">▼</span>
    `;

    // Dropdown
    const dropdown = document.createElement('ul');
    dropdown.className = 'kelly-language-dropdown';
    dropdown.setAttribute('role', 'listbox');
    dropdown.style.cssText = `
      position: absolute;
      top: 100%;
      left: 0;
      margin-top: 4px;
      padding: 8px 0;
      list-style: none;
      background: rgba(20, 20, 30, 0.98);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      z-index: 1000;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-8px);
      transition: all 0.2s ease;
      min-width: 280px;
      backdrop-filter: blur(12px);
    `;

    // Add language options
    const supportedLangs = Object.keys(LANGUAGE_CONFIG);
    supportedLangs.forEach(lang => {
      const config = LANGUAGE_CONFIG[lang];
      if (!config) return;

      const li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.setAttribute('data-lang', lang);
      li.setAttribute('aria-selected', lang === currentLang ? 'true' : 'false');
      li.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 12px 16px;
        cursor: pointer;
        transition: background 0.15s ease;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        ${lang === currentLang ? 'background: rgba(37, 99, 235, 0.15);' : ''}
      `;
      
      // Language row with flag, name, and status
      const isComplete = config.progress === 100;
      const isPartial = config.progress > 0 && config.progress < 100;
      
      let statusBadge = '';
      if (isComplete) {
        statusBadge = '<span style="color: #10b981; font-size: 12px;">✓ 100%</span>';
      } else if (isPartial) {
        statusBadge = `<span style="color: #f59e0b; font-size: 12px;">${config.progress}%</span>`;
      } else {
        statusBadge = '<span style="color: #94a3b8; font-size: 11px;">Open</span>';
      }
      
      li.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px; width: 100%;">
          ${showFlags ? `<span class="flag" style="font-size: 20px;">${config.flag}</span>` : ''}
          <span class="name" style="flex: 1;">${showNative ? config.native : config.name}</span>
          ${statusBadge}
          ${lang === currentLang ? '<span style="color: #2563eb;">✓</span>' : ''}
        </div>
        ${showProgress && !isComplete ? `
          <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
            <div style="flex: 1; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
              <div style="width: ${config.progress}%; height: 100%; background: ${isPartial ? '#f59e0b' : '#475569'}; border-radius: 2px;"></div>
            </div>
            <span style="font-size: 10px; color: #94a3b8;">${config.lessons}/365</span>
          </div>
        ` : ''}
      `;

      li.addEventListener('mouseenter', () => {
        li.style.background = lang === currentLang ? 'rgba(37, 99, 235, 0.2)' : 'rgba(255,255,255,0.05)';
      });
      li.addEventListener('mouseleave', () => {
        li.style.background = lang === currentLang ? 'rgba(37, 99, 235, 0.15)' : 'transparent';
      });

      li.addEventListener('click', async () => {
        // If language has content, switch to it
        if (config.progress > 0) {
          const success = await window.KellyI18n?.setLanguage(lang);
          if (success) {
            updateButton(lang);
            closeDropdown();
            if (onChange) onChange(lang);
            
            // Show fallback notice if not 100%
            if (config.progress < 100 && config.progress > 0) {
              showFallbackNotice(config);
            }
          }
        } else {
          // No content yet - show sponsor prompt
          closeDropdown();
          showSponsorModal(lang, config);
        }
      });

      dropdown.appendChild(li);
    });
    
    // Add "Help translate" link
    const helpLink = document.createElement('li');
    helpLink.style.cssText = `
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 16px;
      cursor: pointer;
      color: #6366f1;
      font-size: 13px;
      transition: background 0.15s ease;
    `;
    helpLink.innerHTML = `
      <span>✨</span>
      <span>Help translate Kelly</span>
      <span style="font-size: 10px;">→</span>
    `;
    helpLink.addEventListener('click', () => {
      window.location.href = '/languages';
    });
    helpLink.addEventListener('mouseenter', () => {
      helpLink.style.background = 'rgba(99, 102, 241, 0.1)';
    });
    helpLink.addEventListener('mouseleave', () => {
      helpLink.style.background = 'transparent';
    });
    dropdown.appendChild(helpLink);

    // Toggle dropdown
    let isOpen = false;
    
    function openDropdown() {
      isOpen = true;
      button.setAttribute('aria-expanded', 'true');
      dropdown.style.opacity = '1';
      dropdown.style.visibility = 'visible';
      dropdown.style.transform = 'translateY(0)';
    }

    function closeDropdown() {
      isOpen = false;
      button.setAttribute('aria-expanded', 'false');
      dropdown.style.opacity = '0';
      dropdown.style.visibility = 'hidden';
      dropdown.style.transform = 'translateY(-8px)';
    }

    button.addEventListener('click', (e) => {
      e.stopPropagation();
      if (isOpen) {
        closeDropdown();
      } else {
        openDropdown();
      }
    });

    // Close on outside click
    document.addEventListener('click', () => {
      if (isOpen) closeDropdown();
    });

    // Update button text when language changes
    function updateButton(lang) {
      const config = LANGUAGE_CONFIG[lang] || LANGUAGE_CONFIG.en;
      const flagEl = button.querySelector('.flag');
      if (flagEl) flagEl.textContent = showFlags ? config.flag : '';
      button.querySelector('.label').textContent = showNative ? config.native : config.name;
      
      // Update selection states
      dropdown.querySelectorAll('li[data-lang]').forEach(li => {
        const liLang = li.getAttribute('data-lang');
        li.setAttribute('aria-selected', liLang === lang ? 'true' : 'false');
        li.style.background = liLang === lang ? 'rgba(37, 99, 235, 0.15)' : 'transparent';
      });
    }

    // Listen for language changes from other sources
    window.addEventListener('languagechanged', (e) => {
      updateButton(e.detail.language);
    });

    wrapper.appendChild(button);
    wrapper.appendChild(dropdown);
    
    if (container) {
      container.appendChild(wrapper);
    }

    return wrapper;
  }

  /**
   * Show fallback notice when using a partially translated language
   */
  function showFallbackNotice(config) {
    // Check if notice already exists
    if (document.getElementById('lang-fallback-notice')) return;
    
    const notice = document.createElement('div');
    notice.id = 'lang-fallback-notice';
    notice.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: linear-gradient(135deg, #1e293b, #0f172a);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 12px;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      gap: 12px;
      z-index: 9999;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      animation: slideUp 0.3s ease;
    `;
    notice.innerHTML = `
      <span style="font-size: 20px;">${config.flag}</span>
      <div>
        <div style="font-size: 13px; color: #f8fafc;">${config.native}: ${config.lessons} of 365 lessons available</div>
        <div style="font-size: 11px; color: #94a3b8;">Missing lessons show in English</div>
      </div>
      <button onclick="this.parentElement.remove()" style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 18px; padding: 4px;">&times;</button>
    `;
    
    document.body.appendChild(notice);
    
    // Auto-remove after 5 seconds
    setTimeout(() => notice.remove(), 5000);
  }

  /**
   * Show contribution modal for untranslated languages
   */
  function showSponsorModal(lang, config) {
    // Check if modal already exists
    let modal = document.getElementById('lang-sponsor-modal');
    if (modal) modal.remove();
    
    modal = document.createElement('div');
    modal.id = 'lang-sponsor-modal';
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      animation: fadeIn 0.2s ease;
    `;
    modal.innerHTML = `
      <div style="
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 32px;
        max-width: 400px;
        width: 90%;
        text-align: center;
      ">
        <div style="font-size: 48px; margin-bottom: 16px;">${config.flag}</div>
        <h2 style="font-size: 24px; margin-bottom: 8px; color: #f8fafc;">${config.native}</h2>
        <p style="color: #94a3b8; margin-bottom: 24px; line-height: 1.5;">
          Contributing is learning. Want to help bring Kelly to ${config.name} speakers?
        </p>
        
        <div style="display: flex; flex-direction: column; gap: 12px;">
          <a href="/languages?sponsor=${lang}" style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 14px 24px;
            background: #6366f1;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.2s;
          ">
            💳 Contribute $${config.cost}
          </a>
          
          <a href="/languages?byok=${lang}" style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 14px 24px;
            background: rgba(255,255,255,0.1);
            color: #f8fafc;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: background 0.2s;
          ">
            🔑 Use My API Credits
          </a>
          
          <button onclick="this.closest('#lang-sponsor-modal').remove()" style="
            padding: 12px;
            background: none;
            border: none;
            color: #64748b;
            cursor: pointer;
            font-size: 14px;
          ">
            Maybe later
          </button>
        </div>
      </div>
    `;
    
    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
    
    document.body.appendChild(modal);
  }

  /**
   * Auto-initialize all [data-language-selector] elements
   */
  function initAll() {
    document.querySelectorAll('[data-language-selector]').forEach(el => {
      const compact = el.hasAttribute('data-compact');
      const showFlags = !el.hasAttribute('data-no-flags');
      const showProgress = !el.hasAttribute('data-no-progress');
      create(el, { compact, showFlags, showProgress });
    });
  }

  /**
   * Update translation progress (can be called with fresh data)
   */
  function updateProgress(langCode, lessons, total = 365) {
    if (LANGUAGE_CONFIG[langCode]) {
      LANGUAGE_CONFIG[langCode].lessons = lessons;
      LANGUAGE_CONFIG[langCode].progress = Math.round((lessons / total) * 100);
    }
  }

  // Expose API
  window.KellyLanguageSelector = {
    create,
    initAll,
    updateProgress,
    showSponsorModal,
    LANGUAGE_CONFIG,
  };

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
