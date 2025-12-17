/**
 * Language Selector Component
 * 
 * Provides a dropdown for language selection that integrates with i18n-core.
 * 
 * Usage:
 *   <div data-language-selector></div>
 *   
 *   Or programmatically:
 *   KellyLanguageSelector.create(containerElement);
 */

(function() {
  'use strict';

  const LANGUAGE_CONFIG = {
    en: { name: 'English', flag: '🇺🇸', native: 'English' },
    es: { name: 'Spanish', flag: '🇪🇸', native: 'Español' },
    pt: { name: 'Portuguese', flag: '🇧🇷', native: 'Português' },
    fr: { name: 'French', flag: '🇫🇷', native: 'Français' },
    de: { name: 'German', flag: '🇩🇪', native: 'Deutsch' },
    hi: { name: 'Hindi', flag: '🇮🇳', native: 'हिन्दी' },
  };

  /**
   * Create a language selector dropdown
   */
  function create(container, options = {}) {
    const {
      showFlags = true,
      showNative = true,
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
      right: 0;
      margin-top: 4px;
      padding: 4px 0;
      list-style: none;
      background: rgba(30, 30, 40, 0.98);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 1000;
      opacity: 0;
      visibility: hidden;
      transform: translateY(-8px);
      transition: all 0.2s ease;
      min-width: 160px;
    `;

    // Add language options
    const supportedLangs = window.KellyI18n?.getSupportedLanguages() || Object.keys(LANGUAGE_CONFIG);
    supportedLangs.forEach(lang => {
      const config = LANGUAGE_CONFIG[lang];
      if (!config) return;

      const li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.setAttribute('data-lang', lang);
      li.setAttribute('aria-selected', lang === currentLang ? 'true' : 'false');
      li.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        cursor: pointer;
        transition: background 0.15s ease;
        ${lang === currentLang ? 'background: rgba(255,255,255,0.1);' : ''}
      `;
      li.innerHTML = `
        ${showFlags ? `<span class="flag">${config.flag}</span>` : ''}
        <span class="name">${showNative ? config.native : config.name}</span>
        ${lang === currentLang ? '<span style="margin-left: auto;">✓</span>' : ''}
      `;

      li.addEventListener('mouseenter', () => {
        li.style.background = 'rgba(255,255,255,0.1)';
      });
      li.addEventListener('mouseleave', () => {
        li.style.background = lang === currentLang ? 'rgba(255,255,255,0.1)' : 'transparent';
      });

      li.addEventListener('click', async () => {
        const success = await window.KellyI18n?.setLanguage(lang);
        if (success) {
          updateButton(lang);
          closeDropdown();
          if (onChange) onChange(lang);
        }
      });

      dropdown.appendChild(li);
    });

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
      button.querySelector('.flag').textContent = showFlags ? config.flag : '';
      button.querySelector('.label').textContent = showNative ? config.native : config.name;
      
      // Update checkmarks
      dropdown.querySelectorAll('li').forEach(li => {
        const liLang = li.getAttribute('data-lang');
        li.setAttribute('aria-selected', liLang === lang ? 'true' : 'false');
        li.style.background = liLang === lang ? 'rgba(255,255,255,0.1)' : 'transparent';
        
        // Update checkmark
        const existingCheck = li.querySelector('.check');
        if (liLang === lang && !existingCheck) {
          li.innerHTML = li.innerHTML + '<span class="check" style="margin-left: auto;">✓</span>';
        } else if (liLang !== lang && existingCheck) {
          existingCheck.remove();
        }
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
   * Auto-initialize all [data-language-selector] elements
   */
  function initAll() {
    document.querySelectorAll('[data-language-selector]').forEach(el => {
      const compact = el.hasAttribute('data-compact');
      const showFlags = !el.hasAttribute('data-no-flags');
      create(el, { compact, showFlags });
    });
  }

  // Expose API
  window.KellyLanguageSelector = {
    create,
    initAll,
    LANGUAGE_CONFIG,
  };

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

})();
