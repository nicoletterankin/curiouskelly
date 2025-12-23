/**
 * Geo-Pricing Client
 * 
 * Fetches localized pricing based on user's country.
 * Updates all pricing displays across the app.
 * 
 * Usage:
 *   await window.KellyGeoPricing.load();
 *   const prices = window.KellyGeoPricing.getPrices();
 */

(function() {
  'use strict';

  const CACHE_KEY = 'kelly_geo_pricing';
  const CACHE_TTL = 60 * 60 * 1000; // 1 hour

  // Default USD prices (fallback)
  const DEFAULT_PRICES = {
    symbol: '$',
    monthly: '$7.99',
    annual: '$49.99',
    family: '$99.99',
    lifetime: '$199.99',
    gift_3mo: '$24.99',
    gift_6mo: '$39.99',
    gift_12mo: '$49.99',
    gift_lifetime: '$149.99',
    perDay: '$0.14',
    savings: '48%',
  };

  let _pricingData = null;
  let _loadPromise = null;

  /**
   * Load geo-pricing from API (with caching)
   */
  async function load() {
    // Return cached promise if already loading
    if (_loadPromise) return _loadPromise;

    // Check cache first
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (cached) {
        const { data, timestamp } = JSON.parse(cached);
        if (Date.now() - timestamp < CACHE_TTL) {
          _pricingData = data;
          applyPricing();
          return data;
        }
      }
    } catch (e) {
      console.warn('[GeoPricing] Cache read failed:', e);
    }

    // Fetch from API
    _loadPromise = fetchPricing();
    return _loadPromise;
  }

  async function fetchPricing(countryOverride = null) {
    try {
      const url = countryOverride 
        ? `/api/geo-pricing?force_country=${countryOverride}`
        : '/api/geo-pricing';
      const response = await fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      _pricingData = data;

      // Cache the result
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({
          data,
          timestamp: Date.now(),
        }));
      } catch (e) {
        console.warn('[GeoPricing] Cache write failed:', e);
      }

      applyPricing();
      return data;

    } catch (error) {
      console.warn('[GeoPricing] API fetch failed, using defaults:', error);
      _pricingData = {
        country: 'US',
        currency: 'USD',
        prices: DEFAULT_PRICES,
        isPPP: false,
      };
      return _pricingData;
    } finally {
      _loadPromise = null;
    }
  }

  /**
   * Apply pricing to all elements with data-price attributes
   */
  function applyPricing() {
    if (!_pricingData?.prices) return;

    const prices = _pricingData.prices;

    // Update all elements with data-price="monthly", data-price="annual", etc.
    const priceElements = document.querySelectorAll('[data-price]');
    priceElements.forEach(el => {
      const priceKey = el.getAttribute('data-price');
      if (prices[priceKey]) {
        el.textContent = prices[priceKey];
      }
    });

    // Update currency symbols
    const symbolElements = document.querySelectorAll('[data-currency-symbol]');
    symbolElements.forEach(el => {
      el.textContent = prices.symbol || '$';
    });

    // Show PPP badge if applicable
    if (_pricingData.isPPP) {
      const pppBadges = document.querySelectorAll('[data-ppp-badge]');
      pppBadges.forEach(el => {
        el.style.display = 'inline-flex';
        const discount = _pricingData.pppDiscount;
        if (discount) {
          el.textContent = `${discount}% off for your region`;
        }
      });
    }

    // Dispatch event for custom handlers
    window.dispatchEvent(new CustomEvent('geopricingloaded', { detail: _pricingData }));
  }

  /**
   * Get current pricing data
   */
  function getPrices() {
    return _pricingData?.prices || DEFAULT_PRICES;
  }

  /**
   * Get full pricing data including country, currency, etc.
   */
  function getData() {
    return _pricingData || {
      country: 'US',
      currency: 'USD',
      prices: DEFAULT_PRICES,
      isPPP: false,
    };
  }

  /**
   * Get the currency code
   */
  function getCurrency() {
    return _pricingData?.currency || 'USD';
  }

  /**
   * Check if this is a PPP country
   */
  function isPPP() {
    return _pricingData?.isPPP || false;
  }

  /**
   * Get the detected country
   */
  function getCountry() {
    return _pricingData?.country || 'US';
  }

  /**
   * Force refresh pricing (ignore cache)
   */
  async function refresh() {
    try {
      localStorage.removeItem(CACHE_KEY);
    } catch (e) {}
    _pricingData = null;
    _loadPromise = null;
    return load();
  }

  /**
   * Set country and reload pricing
   */
  async function setCountry(countryCode) {
    _loadPromise = null;
    const data = await fetchPricing(countryCode);
    _pricingData = data;
    applyPricing();
    return data;
  }

  // Expose API
  window.KellyGeoPricing = {
    load,
    refresh,
    setCountry,
    getPrices,
    getData,
    getCurrency,
    getCountry,
    isPPP,
    applyPricing,
    _pricingData: null, // Expose for UniversalSwitcher
  };

  // Auto-load on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    // DOM already loaded
    load();
  }

})();
