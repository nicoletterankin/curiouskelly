/**
 * Curious Kelly - Earn to Learn Referral Tracking System
 * 
 * LIFETIME ATTRIBUTION: Once you refer someone, you're credited FOREVER.
 * "You introduced them to Kelly. You deserve credit forever."
 * 
 * Tracks referral codes from URL parameters (?ref=CODE)
 * Stores in localStorage indefinitely (no expiration)
 * Sends tracking data to server for attribution
 */

(function() {
    'use strict';
    
    // Debug mode
    const __AFFILIATE_DEBUG = (
      (typeof location !== 'undefined' && location.search.includes('debug')) ||
      (typeof localStorage !== 'undefined' && localStorage.getItem('kellyDebug') === '1')
    );
    
    // ============================================================
    // CONFIGURATION
    // ============================================================
    
    const STORAGE_KEY = 'kelly_referral';
    const API_BASE = '/api/referral';
    
    // NO EXPIRATION - Lifetime attribution
    // This is the key philosophical difference from standard affiliate programs
    const ATTRIBUTION_EXPIRES = null; // NULL = NEVER expires
    
    // ============================================================
    // STORAGE UTILITIES
    // ============================================================
    
    /**
     * Store referral data in localStorage (LIFETIME - no expiration)
     */
    function storeReferralData(code, source, clickId) {
        const data = {
            code: code.toLowerCase(),
            source: source || 'direct',
            clickId: clickId || null,
            trackedAt: Date.now(),
            // LIFETIME ATTRIBUTION - No expires field!
            landingPage: window.location.pathname,
            fullUrl: window.location.href,
            utmSource: getUrlParam('utm_source'),
            utmMedium: getUrlParam('utm_medium'),
            utmCampaign: getUrlParam('utm_campaign'),
            utmContent: getUrlParam('utm_content'),
            utmTerm: getUrlParam('utm_term')
        };
        
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
            return true;
        } catch (e) {
            if (__AFFILIATE_DEBUG) console.warn('[Referral] Failed to store data:', e);
            return false;
        }
    }
    
    /**
     * Get stored referral data from localStorage
     * LIFETIME ATTRIBUTION - Never expires, just returns stored data
     */
    function getStoredReferralData() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return null;
            return JSON.parse(stored);
        } catch (e) {
            if (__AFFILIATE_DEBUG) console.warn('[Referral] Failed to read stored data:', e);
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
     * Clean the URL by removing the ref parameter (cosmetic only)
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
     * Generate a simple browser fingerprint for tracking
     * Note: This is not for security, just for connecting pre/post signup visits
     */
    function generateFingerprint() {
        const nav = window.navigator;
        const screen = window.screen;
        const components = [
            nav.userAgent,
            nav.language,
            screen.width + 'x' + screen.height,
            screen.colorDepth,
            new Date().getTimezoneOffset()
        ];
        return btoa(components.join('|')).substring(0, 32);
    }
    
    // ============================================================
    // API COMMUNICATION
    // ============================================================
    
    /**
     * Track referral click on server
     */
    async function trackClick(code, utmParams) {
        try {
            const response = await fetch(API_BASE + '/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    referralCode: code,
                    visitorFingerprint: generateFingerprint(),
                    sourceUrl: document.referrer || null,
                    landingPage: window.location.pathname,
                    utmSource: utmParams.source,
                    utmMedium: utmParams.medium,
                    utmCampaign: utmParams.campaign,
                    utmContent: utmParams.content,
                    utmTerm: utmParams.term
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (__AFFILIATE_DEBUG) console.log('[Referral] Click tracked:', code, '-> Click ID:', data.clickId);
                return data.clickId;
            } else {
                if (__AFFILIATE_DEBUG) console.warn('[Referral] Track failed:', data.message);
                return null;
            }
        } catch (error) {
            console.error('[Referral] Track error:', error);
            return null;
        }
    }
    
    /**
     * Look up referral code info
     */
    async function lookupCode(code) {
        try {
            const response = await fetch(API_BASE + '/lookup?code=' + encodeURIComponent(code));
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('[Referral] Lookup error:', error);
            return { valid: false };
        }
    }
    
    // ============================================================
    // PUBLIC API
    // ============================================================
    
    /**
     * Get the current referral code (LIFETIME attribution)
     * This is the main function used by checkout and signup flows
     */
    window.getReferralCode = function() {
        const stored = getStoredReferralData();
        return stored ? stored.code : null;
    };
    
    /**
     * Get full referral tracking data for checkout metadata
     */
    window.getReferralTrackingData = function() {
        const stored = getStoredReferralData() || {};
        return {
            referralCode: stored.code || null,
            clickId: stored.clickId || null,
            source: stored.source || 'direct',
            landingPage: stored.landingPage || window.location.pathname,
            utmSource: stored.utmSource || getUrlParam('utm_source') || 'direct',
            utmMedium: stored.utmMedium || getUrlParam('utm_medium') || 'none',
            utmCampaign: stored.utmCampaign || getUrlParam('utm_campaign') || 'none',
            trackedAt: stored.trackedAt || null
        };
    };
    
    /**
     * Check if user has an active referral attribution
     * LIFETIME ATTRIBUTION - If they have a code, it's always active
     */
    window.hasReferralAttribution = function() {
        return !!window.getReferralCode();
    };
    
    /**
     * Clear referral attribution (for testing only)
     */
    window.clearReferralAttribution = function() {
        localStorage.removeItem(STORAGE_KEY);
        if (__AFFILIATE_DEBUG) console.log('[Referral] Attribution cleared');
    };
    
    /**
     * Manually set a referral code (for testing or special flows)
     */
    window.setReferralCode = function(code) {
        if (code) {
            storeReferralData(code, 'manual', null);
            if (__AFFILIATE_DEBUG) console.log('[Referral] Manually set code:', code);
        }
    };
    
    /**
     * Look up referrer info for UI display
     * Returns { valid, referrer: { displayName, tier, lessonsCompleted } }
     */
    window.lookupReferrer = async function(code) {
        return await lookupCode(code || window.getReferralCode());
    };
    
    // Legacy compatibility with old affiliate tracking
    window.getAffiliateCode = window.getReferralCode;
    window.getAffiliateTrackingData = window.getReferralTrackingData;
    window.hasAffiliateAttribution = window.hasReferralAttribution;
    window.clearAffiliateAttribution = window.clearReferralAttribution;
    
    // ============================================================
    // INITIALIZATION
    // ============================================================
    
    /**
     * Initialize tracking on page load
     */
    async function init() {
        // Check for ref parameter in URL
        const refCode = getUrlParam('ref');
        
        if (refCode) {
            // Validate code format (alphanumeric, underscores, hyphens, 3-30 chars)
            if (/^[A-Za-z0-9_-]{3,30}$/.test(refCode)) {
                // Collect UTM params
                const utmParams = {
                    source: getUrlParam('utm_source'),
                    medium: getUrlParam('utm_medium'),
                    campaign: getUrlParam('utm_campaign'),
                    content: getUrlParam('utm_content'),
                    term: getUrlParam('utm_term')
                };
                
                // Track on server and get click ID
                const clickId = await trackClick(refCode, utmParams);
                
                // Store locally with LIFETIME attribution
                storeReferralData(refCode, 'referral_link', clickId);
                
                // Clean URL (cosmetic)
                cleanUrl();
                
                if (__AFFILIATE_DEBUG) console.log('[Referral] Tracked:', refCode, 'with LIFETIME attribution');
                
                // Fire analytics event if available
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'referral_click', {
                        event_category: 'referral',
                        event_label: refCode
                    });
                }
            } else {
                if (__AFFILIATE_DEBUG) console.warn('[Referral] Invalid code format:', refCode);
            }
        }
        
        // Log current attribution status
        const currentCode = window.getReferralCode();
        if (currentCode) {
            if (__AFFILIATE_DEBUG) console.log('[Referral] Active LIFETIME attribution:', currentCode);
        }
    }
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
})();
