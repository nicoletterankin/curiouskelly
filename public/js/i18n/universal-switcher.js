/**
 * Universal Switcher
 * 
 * Unified language + country/currency switcher that coordinates:
 * - Language switching (EN, ES, PT) via KellyI18n
 * - Country/currency switching via GeoPricing
 * - Instant UI updates
 * - Lesson content reloading
 * - Pricing updates
 * 
 * Usage:
 *   await UniversalSwitcher.init();
 *   await UniversalSwitcher.switchLanguage('es');
 *   await UniversalSwitcher.switchCountry('DE');
 */

(function() {
  'use strict';

  const STORAGE_LANG_KEY = 'kelly_language';
  const STORAGE_COUNTRY_KEY = 'kelly_country';

  // Supported countries with flags and currency info
  const SUPPORTED_COUNTRIES = [
    { code: 'US', name: 'United States', flag: '🇺🇸', currency: 'USD', symbol: '$' },
    { code: 'DE', name: 'Germany', flag: '🇪🇺', currency: 'EUR', symbol: '€' },
    { code: 'GB', name: 'United Kingdom', flag: '🇬🇧', currency: 'GBP', symbol: '£' },
    { code: 'CA', name: 'Canada', flag: '🇨🇦', currency: 'CAD', symbol: '$' },
    { code: 'AU', name: 'Australia', flag: '🇦🇺', currency: 'AUD', symbol: '$' },
    { code: 'IN', name: 'India', flag: '🇮🇳', currency: 'INR', symbol: '₹', ppp: true },
    { code: 'BR', name: 'Brazil', flag: '🇧🇷', currency: 'BRL', symbol: 'R$', ppp: true },
    { code: 'MX', name: 'Mexico', flag: '🇲🇽', currency: 'MXN', symbol: 'MX$', ppp: true },
    { code: 'PL', name: 'Poland', flag: '🇵🇱', currency: 'PLN', symbol: 'zł', ppp: true },
    { code: 'FR', name: 'France', flag: '🇫🇷', currency: 'EUR', symbol: '€' },
    { code: 'ES', name: 'Spain', flag: '🇪🇸', currency: 'EUR', symbol: '€' },
    { code: 'IT', name: 'Italy', flag: '🇮🇹', currency: 'EUR', symbol: '€' },
    { code: 'NL', name: 'Netherlands', flag: '🇳🇱', currency: 'EUR', symbol: '€' },
  ];

  let _currentLanguage = 'en';
  let _currentCountry = 'US';
  let _switcherElement = null;
  let _initialized = false;

  /**
   * Initialize the universal switcher
   */
  async function init() {
    if (_initialized) return;

    // Load saved preferences
    try {
      _currentLanguage = localStorage.getItem(STORAGE_LANG_KEY) || 'en';
      _currentCountry = localStorage.getItem(STORAGE_COUNTRY_KEY) || 'US';
    } catch (e) {
      console.warn('[UniversalSwitcher] Failed to load preferences:', e);
    }

    // Wait for dependencies
    if (!window.KellyI18n) {
      console.warn('[UniversalSwitcher] KellyI18n not loaded, waiting...');
      await new Promise(resolve => {
        const check = setInterval(() => {
          if (window.KellyI18n) {
            clearInterval(check);
            resolve();
          }
        }, 100);
        setTimeout(() => { clearInterval(check); resolve(); }, 5000);
      });
    }

    // Initialize i18n with saved language
    if (window.KellyI18n && typeof window.KellyI18n.setLanguage === 'function') {
      await window.KellyI18n.setLanguage(_currentLanguage);
    }

    // Load geo-pricing for saved country
    if (window.KellyGeoPricing) {
      await window.KellyGeoPricing.load();
      // Override country if saved
      if (_currentCountry !== 'US') {
        await setCountry(_currentCountry, false); // false = don't save (already saved)
      }
    }

    // Render switcher UI
    renderSwitcher();

    _initialized = true;
    console.log('[UniversalSwitcher] Initialized', { language: _currentLanguage, country: _currentCountry });
  }

  /**
   * Switch language
   */
  async function switchLanguage(lang) {
    if (!window.KellyI18n) {
      console.error('[UniversalSwitcher] KellyI18n not available');
      return false;
    }

    try {
      localStorage.setItem(STORAGE_LANG_KEY, lang);
      _currentLanguage = lang;

      // Update i18n
      await window.KellyI18n.setLanguage(lang);

      // Update switcher UI
      updateSwitcherUI();

      // Reload lesson content if on lesson page
      await updateLessonContent();

      // Show toast
      showToast(`🌍 Switched to ${getLanguageName(lang)}`);

      // Dispatch event
      window.dispatchEvent(new CustomEvent('universallanguagechanged', { 
        detail: { language: lang } 
      }));

      return true;
    } catch (e) {
      console.error('[UniversalSwitcher] Language switch failed:', e);
      showToast('Failed to switch language. Please try again.');
      return false;
    }
  }

  /**
   * Switch country/currency
   */
  async function setCountry(countryCode, save = true) {
    if (!window.KellyGeoPricing) {
      console.error('[UniversalSwitcher] GeoPricing not available');
      return false;
    }

    try {
      if (save) {
        localStorage.setItem(STORAGE_COUNTRY_KEY, countryCode);
      }
      _currentCountry = countryCode;

      // Fetch pricing for new country
      const response = await fetch(`/api/geo-pricing?force_country=${countryCode}`);
      if (response.ok) {
        const pricingData = await response.json();
        
        // Update geo-pricing cache
        try {
          localStorage.setItem('kelly_geo_pricing', JSON.stringify({
            data: pricingData,
            timestamp: Date.now(),
          }));
        } catch (e) {}

        // Trigger pricing update
        if (window.KellyGeoPricing.setPricingData) {
          window.KellyGeoPricing.setPricingData(pricingData);
        } else if (window.KellyGeoPricing.applyPricing) {
          // Fallback: try to update internal state if setPricingData not available
          window.KellyGeoPricing._pricingData = pricingData;
          window.KellyGeoPricing.applyPricing();
        }

        // Update all pricing displays
        await updatePricing(pricingData);

        // Update switcher UI
        updateSwitcherUI();

        // Show toast
        const country = SUPPORTED_COUNTRIES.find(c => c.code === countryCode);
        showToast(`💰 Switched to ${country?.flag || ''} ${country?.name || countryCode} (${country?.currency || 'USD'})`);

        // Dispatch event
        window.dispatchEvent(new CustomEvent('universalcountrychanged', { 
          detail: { country: countryCode, pricing: pricingData } 
        }));

        return true;
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (e) {
      console.error('[UniversalSwitcher] Country switch failed:', e);
      showToast('Failed to switch country. Using default pricing.');
      return false;
    }
  }

  /**
   * Update lesson content for new language
   */
  async function updateLessonContent() {
    // Check if we're on a lesson page
    const dayMatch = window.location.pathname.match(/day[=\/](\d+)/i) || 
                     new URLSearchParams(window.location.search).get('day');
    
    if (dayMatch) {
      const dayNumber = typeof dayMatch === 'string' ? dayMatch : dayMatch[1];
      // Trigger lesson reload with language param
      if (window.KellyLessonLoader && typeof window.KellyLessonLoader.loadLesson === 'function') {
        try {
          await window.KellyLessonLoader.loadLesson(parseInt(dayNumber), { 
            language: _currentLanguage 
          });
        } catch (e) {
          console.warn('[UniversalSwitcher] Lesson reload failed:', e);
        }
      }
    }
  }

  /**
   * Update all pricing displays
   */
  async function updatePricing(pricingData) {
    if (!pricingData?.prices) return;

    const prices = pricingData.prices;
    const symbol = pricingData.symbol || '$';

    // Update all elements with data-price attributes
    document.querySelectorAll('[data-price-monthly]').forEach(el => {
      el.textContent = prices.monthly || '$7.99';
    });
    document.querySelectorAll('[data-price-annual]').forEach(el => {
      el.textContent = prices.annual || '$49.99';
    });
    document.querySelectorAll('[data-price-family]').forEach(el => {
      el.textContent = prices.family || '$99.99';
    });
    document.querySelectorAll('[data-price-lifetime]').forEach(el => {
      el.textContent = prices.lifetime || '$199.99';
    });

    // Update currency symbols
    document.querySelectorAll('[data-currency-symbol]').forEach(el => {
      el.textContent = symbol;
    });

    // Update PPP badges
    if (pricingData.isPPP) {
      document.querySelectorAll('[data-ppp-badge]').forEach(el => {
        el.style.display = 'inline-flex';
        el.textContent = `${pricingData.pppDiscount || 50}% off for your region`;
      });
    } else {
      document.querySelectorAll('[data-ppp-badge]').forEach(el => {
        el.style.display = 'none';
      });
    }

    // Trigger custom pricing update event
    window.dispatchEvent(new CustomEvent('pricingupdated', { detail: pricingData }));
  }

  /**
   * Render switcher UI
   */
  function renderSwitcher() {
    // Find insertion point (top-right header area)
    const header = document.querySelector('header') || 
                   document.querySelector('.app-header') ||
                   document.querySelector('[data-header]');
    
    if (!header) {
      console.warn('[UniversalSwitcher] Header not found, creating floating switcher');
      createFloatingSwitcher();
      return;
    }

    // Check if switcher already exists
    if (document.getElementById('universal-switcher')) {
      updateSwitcherUI();
      return;
    }

    const switcher = document.createElement('div');
    switcher.id = 'universal-switcher';
    switcher.className = 'universal-switcher';
    switcher.innerHTML = `
      <div class="switcher-group">
        <label for="language-selector">🌍</label>
        <select id="language-selector" class="switcher-select" aria-label="Select language">
          <option value="en">🇺🇸 English</option>
          <option value="es">🇪🇸 Español</option>
          <option value="pt">🇵🇹 Português</option>
        </select>
      </div>
      <div class="switcher-group">
        <label for="country-selector">💰</label>
        <select id="country-selector" class="switcher-select" aria-label="Select country">
          ${SUPPORTED_COUNTRIES.map(c => `
            <option value="${c.code}">${c.flag} ${c.name} (${c.symbol})</option>
          `).join('')}
        </select>
      </div>
    `;

    // Insert before Settings button or at end of header
    const settingsBtn = header.querySelector('[data-settings]') || 
                        header.querySelector('.settings-btn') ||
                        header.querySelector('button:last-child');
    
    if (settingsBtn && settingsBtn.parentNode) {
      settingsBtn.parentNode.insertBefore(switcher, settingsBtn);
    } else {
      header.appendChild(switcher);
    }

    _switcherElement = switcher;
    injectSwitcherStyles();
    attachHandlers();
    updateSwitcherUI();
  }

  /**
   * Create floating switcher if header not found
   */
  function createFloatingSwitcher() {
    const switcher = document.createElement('div');
    switcher.id = 'universal-switcher';
    switcher.className = 'universal-switcher floating';
    switcher.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 10000;
      background: rgba(0, 0, 0, 0.9);
      padding: 12px;
      border-radius: 12px;
      backdrop-filter: blur(10px);
    `;
    switcher.innerHTML = `
      <div class="switcher-group">
        <label>🌍</label>
        <select id="language-selector" class="switcher-select">
          <option value="en">🇺🇸 English</option>
          <option value="es">🇪🇸 Español</option>
          <option value="pt">🇵🇹 Português</option>
        </select>
      </div>
      <div class="switcher-group">
        <label>💰</label>
        <select id="country-selector" class="switcher-select">
          ${SUPPORTED_COUNTRIES.map(c => `
            <option value="${c.code}">${c.flag} ${c.name} (${c.symbol})</option>
          `).join('')}
        </select>
      </div>
    `;
    document.body.appendChild(switcher);
    _switcherElement = switcher;
    injectSwitcherStyles();
    attachHandlers();
    updateSwitcherUI();
  }

  /**
   * Update switcher UI to reflect current state
   */
  function updateSwitcherUI() {
    const langSelect = document.getElementById('language-selector');
    const countrySelect = document.getElementById('country-selector');

    if (langSelect) {
      langSelect.value = _currentLanguage;
    }
    if (countrySelect) {
      countrySelect.value = _currentCountry;
    }
  }

  /**
   * Attach event handlers
   */
  function attachHandlers() {
    const langSelect = document.getElementById('language-selector');
    const countrySelect = document.getElementById('country-selector');

    if (langSelect) {
      langSelect.addEventListener('change', (e) => {
        switchLanguage(e.target.value);
      });
    }

    if (countrySelect) {
      countrySelect.addEventListener('change', (e) => {
        setCountry(e.target.value);
      });
    }
  }

  /**
   * Inject switcher styles
   */
  function injectSwitcherStyles() {
    if (document.getElementById('universal-switcher-styles')) return;

    const style = document.createElement('style');
    style.id = 'universal-switcher-styles';
    style.textContent = `
      .universal-switcher {
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      .switcher-group {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .switcher-group label {
        font-size: 18px;
        cursor: default;
      }
      .switcher-select {
        padding: 6px 10px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        color: white;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s;
        min-width: 140px;
      }
      .switcher-select:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: var(--kelly-blue, #2563eb);
      }
      .switcher-select:focus {
        outline: none;
        border-color: var(--kelly-blue, #2563eb);
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
      }
      .universal-switcher.floating {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      }
      @media (max-width: 768px) {
        .universal-switcher {
          flex-direction: column;
          gap: 8px;
        }
        .switcher-select {
          min-width: 120px;
        }
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Get language name
   */
  function getLanguageName(code) {
    const names = {
      en: 'English',
      es: 'Español',
      pt: 'Português',
      fr: 'Français',
      de: 'Deutsch',
    };
    return names[code] || code.toUpperCase();
  }

  /**
   * Show toast notification
   */
  function showToast(message) {
    // Use existing toast system if available
    if (typeof window.showToast === 'function') {
      window.showToast(message);
      return;
    }

    // Fallback toast
    const toast = document.createElement('div');
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      z-index: 10001;
      font-size: 14px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }

  /**
   * Get current language
   */
  function getLanguage() {
    return _currentLanguage;
  }

  /**
   * Get current country
   */
  function getCountry() {
    return _currentCountry;
  }

  /**
   * Get supported countries
   */
  function getSupportedCountries() {
    return [...SUPPORTED_COUNTRIES];
  }

  // Expose API
  window.UniversalSwitcher = {
    init,
    switchLanguage,
    setCountry,
    getLanguage,
    getCountry,
    getSupportedCountries,
    updatePricing,
  };

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

