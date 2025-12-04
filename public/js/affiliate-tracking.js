/**
 * Curious Kelly Affiliate Tracking System
 * 
 * Tracks referral codes from URL parameters (?ref=CODE)
 * Sets cookies and localStorage for attribution
 * Provides utilities for checkout integration
 */

(function() {
    'use strict';
    
    const COOKIE_NAME = 'kelly_affiliate_ref';
    const COOKIE_DAYS = 30; // 30-day attribution window
    const STORAGE_KEY = 'kelly_affiliate';
    
    /**
     * Set a cookie with expiration
     */
    function setCookie(name, value, days) {
        const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toUTCString();
        document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax; Secure`;
    }
    
    /**
     * Get a cookie value by name
     */
    function getCookie(name) {
        const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
        return match ? decodeURIComponent(match[2]) : null;
    }
    
    /**
     * Store affiliate data in localStorage as backup
     */
    function storeAffiliateData(code, source) {
        const data = {
            code: code,
            source: source || 'direct',
            timestamp: Date.now(),
            expires: Date.now() + (COOKIE_DAYS * 24 * 60 * 60 * 1000),
            landingPage: window.location.pathname,
            utmSource: getUrlParam('utm_source'),
            utmMedium: getUrlParam('utm_medium'),
            utmCampaign: getUrlParam('utm_campaign')
        };
        
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
        } catch (e) {
            console.warn('Failed to store affiliate data:', e);
        }
    }
    
    /**
     * Get stored affiliate data from localStorage
     */
    function getStoredAffiliateData() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return null;
            
            const data = JSON.parse(stored);
            
            // Check if expired
            if (data.expires && Date.now() > data.expires) {
                localStorage.removeItem(STORAGE_KEY);
                return null;
            }
            
            return data;
        } catch (e) {
            return null;
        }
    }
    
    /**
     * Get URL parameter
     */
    function getUrlParam(param) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(param);
    }
    
    /**
     * Clean the URL by removing the ref parameter
     */
    function cleanUrl() {
        const url = new URL(window.location.href);
        if (url.searchParams.has('ref')) {
            url.searchParams.delete('ref');
            const newUrl = url.pathname + (url.search || '') + url.hash;
            window.history.replaceState({}, document.title, newUrl);
        }
    }
    
    /**
     * Get the current affiliate code from cookie or localStorage
     * This is the main function used by checkout
     */
    window.getAffiliateCode = function() {
        // Check cookie first (most reliable)
        const cookieCode = getCookie(COOKIE_NAME);
        if (cookieCode) return cookieCode;
        
        // Fallback to localStorage
        const storedData = getStoredAffiliateData();
        if (storedData && storedData.code) {
            return storedData.code;
        }
        
        return null;
    };
    
    /**
     * Get full affiliate tracking data for checkout metadata
     */
    window.getAffiliateTrackingData = function() {
        const code = window.getAffiliateCode();
        const storedData = getStoredAffiliateData() || {};
        
        return {
            affiliateCode: code,
            source: storedData.source || 'direct',
            landingPage: storedData.landingPage || window.location.pathname,
            utmSource: storedData.utmSource || getUrlParam('utm_source') || 'direct',
            utmMedium: storedData.utmMedium || getUrlParam('utm_medium') || 'none',
            utmCampaign: storedData.utmCampaign || getUrlParam('utm_campaign') || 'none'
        };
    };
    
    /**
     * Check if user has an active affiliate attribution
     */
    window.hasAffiliateAttribution = function() {
        return !!window.getAffiliateCode();
    };
    
    /**
     * Clear affiliate attribution (for testing)
     */
    window.clearAffiliateAttribution = function() {
        document.cookie = `${COOKIE_NAME}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;`;
        localStorage.removeItem(STORAGE_KEY);
        console.log('Affiliate attribution cleared');
    };
    
    /**
     * Initialize tracking on page load
     */
    function init() {
        // Check for ref parameter in URL
        const refCode = getUrlParam('ref');
        
        if (refCode) {
            // Validate code format (alphanumeric, 3-20 chars)
            if (/^[A-Za-z0-9_-]{3,20}$/.test(refCode)) {
                // Set cookie
                setCookie(COOKIE_NAME, refCode, COOKIE_DAYS);
                
                // Store in localStorage with metadata
                storeAffiliateData(refCode, 'referral_link');
                
                // Clean URL
                cleanUrl();
                
                // Log for debugging (remove in production)
                console.log(`[Affiliate] Tracked referral: ${refCode}`);
                
                // Fire tracking event
                if (window.gtag) {
                    gtag('event', 'affiliate_click', {
                        'event_category': 'affiliate',
                        'event_label': refCode
                    });
                }
            } else {
                console.warn('[Affiliate] Invalid referral code format:', refCode);
            }
        }
        
        // Log current attribution status (for debugging)
        const currentCode = window.getAffiliateCode();
        if (currentCode) {
            console.log(`[Affiliate] Active attribution: ${currentCode}`);
        }
    }
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();

