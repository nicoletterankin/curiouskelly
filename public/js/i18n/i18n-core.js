/**
 * Kelly i18n Core Engine
 * 
 * Professional translation system with:
 * - Lazy loading of translation files
 * - Namespace support (common, lessons, settings, paywall, kelly, onboarding)
 * - Interpolation ({{variable}})
 * - Pluralization (_one, _few, _many, _other)
 * - Fallback to English
 * - Automatic DOM updates via data-i18n attributes
 * - Event-driven architecture
 * 
 * Usage:
 *   await KellyI18n.init();                           // Auto-detect language
 *   await KellyI18n.setLanguage('es');                // Switch language
 *   KellyI18n.t('common.greeting.morning');           // Get translation
 *   KellyI18n.t('lesson.day', { number: 42 });        // With interpolation
 *   KellyI18n.t('progress.streak', { count: 5 });     // With pluralization
 */

(function() {
  'use strict';

  // ============================================
  // CONFIGURATION
  // ============================================
  
  const SUPPORTED_LANGUAGES = ['en', 'es', 'pt', 'fr', 'de', 'hi'];
  const DEFAULT_LANGUAGE = 'en';
  const FALLBACK_LANGUAGE = 'en';
  const NAMESPACES = ['common', 'lessons', 'settings', 'paywall', 'kelly', 'onboarding'];
  const STORAGE_KEY = 'kelly_language';
  const CACHE_VERSION = '1'; // Bump to invalidate cache
  
  // ============================================
  // STATE
  // ============================================
  
  let _currentLanguage = DEFAULT_LANGUAGE;
  let _translations = {};
  let _loadedNamespaces = {};
  let _initialized = false;
  let _initPromise = null;
  
  // ============================================
  // LANGUAGE DETECTION
  // ============================================
  
  /**
   * Detect user's preferred language
   * Priority: saved preference > browser language > default
   */
  function detectLanguage() {
    // 1. Check saved preference
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && SUPPORTED_LANGUAGES.includes(saved)) {
        return saved;
      }
    } catch (e) {}
    
    // 2. Check browser languages
    const browserLangs = navigator.languages || [navigator.language];
    for (const lang of browserLangs) {
      const code = lang.split('-')[0].toLowerCase();
      if (SUPPORTED_LANGUAGES.includes(code)) {
        return code;
      }
    }
    
    // 3. Default
    return DEFAULT_LANGUAGE;
  }
  
  // ============================================
  // TRANSLATION LOADING
  // ============================================
  
  /**
   * Load a specific namespace for a language
   */
  async function loadNamespace(lang, namespace) {
    const cacheKey = `${lang}_${namespace}_v${CACHE_VERSION}`;
    
    // Check memory cache
    if (_translations[lang]?.[namespace]) {
      return _translations[lang][namespace];
    }
    
    // Check localStorage cache
    try {
      const cached = localStorage.getItem(`kelly_i18n_${cacheKey}`);
      if (cached) {
        const data = JSON.parse(cached);
        if (!_translations[lang]) _translations[lang] = {};
        _translations[lang][namespace] = data;
        return data;
      }
    } catch (e) {}
    
    // Fetch from server
    try {
      const response = await fetch(`/locales/${lang}/${namespace}.json`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      
      // Store in memory
      if (!_translations[lang]) _translations[lang] = {};
      _translations[lang][namespace] = data;
      
      // Store in localStorage
      try {
        localStorage.setItem(`kelly_i18n_${cacheKey}`, JSON.stringify(data));
      } catch (e) {}
      
      return data;
    } catch (error) {
      console.warn(`[i18n] Failed to load ${lang}/${namespace}:`, error);
      
      // Try fallback language
      if (lang !== FALLBACK_LANGUAGE) {
        return loadNamespace(FALLBACK_LANGUAGE, namespace);
      }
      
      return {};
    }
  }
  
  /**
   * Load all namespaces for a language
   */
  async function loadAllNamespaces(lang) {
    const promises = NAMESPACES.map(ns => loadNamespace(lang, ns));
    await Promise.all(promises);
    _loadedNamespaces[lang] = true;
  }
  
  // ============================================
  // TRANSLATION LOOKUP
  // ============================================
  
  /**
   * Get a translation by key
   * 
   * @param {string} key - Dot-notation key (e.g., 'common.greeting.morning')
   * @param {object} params - Interpolation parameters
   * @returns {string} - Translated string or key if not found
   */
  function t(key, params = {}) {
    // Parse key: can be 'namespace.path.to.key' or just 'path.to.key' (defaults to common)
    let namespace, path;
    const parts = key.split('.');
    
    if (NAMESPACES.includes(parts[0])) {
      namespace = parts[0];
      path = parts.slice(1);
    } else {
      namespace = 'common';
      path = parts;
    }
    
    // Look up in current language
    let value = _translations[_currentLanguage]?.[namespace];
    for (const p of path) {
      value = value?.[p];
      if (value === undefined) break;
    }
    
    // Handle pluralization
    if (value === undefined && params.count !== undefined) {
      // Try count-specific keys (_one, _few, _many, _other)
      const pluralKey = getPluralKey(params.count);
      const pluralPath = [...path.slice(0, -1), path[path.length - 1] + '_' + pluralKey];
      
      value = _translations[_currentLanguage]?.[namespace];
      for (const p of pluralPath) {
        value = value?.[p];
        if (value === undefined) break;
      }
    }
    
    // Fallback to English
    if (value === undefined && _currentLanguage !== FALLBACK_LANGUAGE) {
      value = _translations[FALLBACK_LANGUAGE]?.[namespace];
      for (const p of path) {
        value = value?.[p];
        if (value === undefined) break;
      }
    }
    
    // Still no value? Return key
    if (value === undefined || typeof value !== 'string') {
      console.warn(`[i18n] Missing translation: ${key}`);
      return key;
    }
    
    // Interpolate {{params}}
    return interpolate(value, params);
  }
  
  /**
   * Interpolate variables in a string
   */
  function interpolate(str, params) {
    return str.replace(/\{\{(\w+)\}\}/g, (match, key) => {
      const value = params[key];
      return value !== undefined ? String(value) : match;
    });
  }
  
  /**
   * Get plural key based on count
   * Simple English rules - expand for other languages
   */
  function getPluralKey(count) {
    if (count === 1) return 'one';
    return 'other';
  }
  
  // ============================================
  // DOM UPDATES
  // ============================================
  
  /**
   * Apply translations to all [data-i18n] elements in the DOM
   */
  function applyTranslations(root = document) {
    // Text content: data-i18n="key"
    root.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.getAttribute('data-i18n');
      const paramsStr = el.getAttribute('data-i18n-params');
      const params = paramsStr ? JSON.parse(paramsStr) : {};
      el.textContent = t(key, params);
    });
    
    // Placeholder: data-i18n-placeholder="key"
    root.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
      const key = el.getAttribute('data-i18n-placeholder');
      el.placeholder = t(key);
    });
    
    // Aria-label: data-i18n-aria="key"
    root.querySelectorAll('[data-i18n-aria]').forEach(el => {
      const key = el.getAttribute('data-i18n-aria');
      el.setAttribute('aria-label', t(key));
    });
    
    // Title: data-i18n-title="key"
    root.querySelectorAll('[data-i18n-title]').forEach(el => {
      const key = el.getAttribute('data-i18n-title');
      el.title = t(key);
    });
    
    // HTML content: data-i18n-html="key" (use with caution!)
    root.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.getAttribute('data-i18n-html');
      el.innerHTML = t(key);
    });
  }
  
  /**
   * Update document language attribute
   */
  function updateDocumentLang() {
    document.documentElement.lang = _currentLanguage;
    document.documentElement.dir = isRTL(_currentLanguage) ? 'rtl' : 'ltr';
  }
  
  /**
   * Check if language is RTL
   */
  function isRTL(lang) {
    return ['ar', 'he', 'fa', 'ur'].includes(lang);
  }
  
  // ============================================
  // PUBLIC API
  // ============================================
  
  /**
   * Initialize the i18n system
   */
  async function init() {
    if (_initialized) return _currentLanguage;
    if (_initPromise) return _initPromise;
    
    _initPromise = (async () => {
      // Detect and set language
      _currentLanguage = detectLanguage();
      
      // Load translations
      await loadAllNamespaces(_currentLanguage);
      
      // Always load English as fallback
      if (_currentLanguage !== FALLBACK_LANGUAGE) {
        await loadAllNamespaces(FALLBACK_LANGUAGE);
      }
      
      // Apply to DOM
      applyTranslations();
      updateDocumentLang();
      
      _initialized = true;
      
      // Dispatch event
      window.dispatchEvent(new CustomEvent('i18nready', { 
        detail: { language: _currentLanguage } 
      }));
      
      console.log(`[i18n] Initialized with language: ${_currentLanguage}`);
      return _currentLanguage;
    })();
    
    return _initPromise;
  }
  
  /**
   * Change the current language
   */
  async function setLanguage(lang) {
    if (!SUPPORTED_LANGUAGES.includes(lang)) {
      console.warn(`[i18n] Unsupported language: ${lang}`);
      return false;
    }
    
    if (lang === _currentLanguage) {
      return true;
    }
    
    // Load new language
    await loadAllNamespaces(lang);
    
    // Update state
    _currentLanguage = lang;
    
    // Save preference
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {}
    
    // Apply to DOM
    applyTranslations();
    updateDocumentLang();
    
    // Dispatch event
    window.dispatchEvent(new CustomEvent('languagechanged', { 
      detail: { language: lang } 
    }));
    
    console.log(`[i18n] Language changed to: ${lang}`);
    return true;
  }
  
  /**
   * Get current language
   */
  function getLanguage() {
    return _currentLanguage;
  }
  
  /**
   * Get list of supported languages
   */
  function getSupportedLanguages() {
    return [...SUPPORTED_LANGUAGES];
  }
  
  /**
   * Check if a language is supported
   */
  function isSupported(lang) {
    return SUPPORTED_LANGUAGES.includes(lang);
  }
  
  /**
   * Force reload all translations
   */
  async function reload() {
    // Clear caches
    _translations = {};
    _loadedNamespaces = {};
    
    // Clear localStorage cache
    try {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith('kelly_i18n_')) {
          localStorage.removeItem(key);
        }
      });
    } catch (e) {}
    
    // Reload
    await loadAllNamespaces(_currentLanguage);
    if (_currentLanguage !== FALLBACK_LANGUAGE) {
      await loadAllNamespaces(FALLBACK_LANGUAGE);
    }
    
    applyTranslations();
    return true;
  }
  
  /**
   * Get language display name
   */
  function getLanguageName(code) {
    const names = {
      en: 'English',
      es: 'Español',
      pt: 'Português',
      fr: 'Français',
      de: 'Deutsch',
      hi: 'हिन्दी',
      zh: '中文',
      ja: '日本語',
      ko: '한국어',
      ar: 'العربية'
    };
    return names[code] || code;
  }
  
  // ============================================
  // EXPOSE API
  // ============================================
  
  window.KellyI18n = {
    init,
    t,
    setLanguage,
    getLanguage,
    getSupportedLanguages,
    isSupported,
    applyTranslations,
    reload,
    getLanguageName,
    isRTL,
  };
  
  // Auto-initialize on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
})();
