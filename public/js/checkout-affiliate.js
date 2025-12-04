/**
 * Checkout Affiliate Integration
 * 
 * Enhances Stripe Payment Links with affiliate tracking data
 * Works with the affiliate-tracking.js script
 */

(function() {
    'use strict';
    
    /**
     * Enhance a Stripe payment link with affiliate tracking
     */
    function enhancePaymentLink(url) {
        if (!url || !url.includes('buy.stripe.com')) {
            return url;
        }
        
        try {
            const linkUrl = new URL(url);
            
            // Get affiliate code from tracking script
            const affiliateCode = window.getAffiliateCode ? window.getAffiliateCode() : null;
            
            if (affiliateCode) {
                // Stripe Payment Links support client_reference_id parameter
                // Format: affiliateCode_timestamp for uniqueness
                const refId = `aff_${affiliateCode}_${Date.now()}`;
                linkUrl.searchParams.set('client_reference_id', refId);
                
                console.log(`[Checkout] Enhanced with affiliate: ${affiliateCode}`);
            }
            
            // Also add prefilled email if available
            const savedEmail = localStorage.getItem('kelly_user_email');
            if (savedEmail) {
                linkUrl.searchParams.set('prefilled_email', savedEmail);
            }
            
            return linkUrl.toString();
        } catch (e) {
            console.warn('[Checkout] Failed to enhance payment link:', e);
            return url;
        }
    }
    
    /**
     * Handle click on payment links
     */
    function handlePaymentLinkClick(event) {
        const link = event.currentTarget;
        const originalUrl = link.href;
        
        if (originalUrl && originalUrl.includes('buy.stripe.com')) {
            event.preventDefault();
            
            const enhancedUrl = enhancePaymentLink(originalUrl);
            
            // Track the checkout attempt
            if (window.gtag) {
                const affiliateCode = window.getAffiliateCode ? window.getAffiliateCode() : null;
                gtag('event', 'begin_checkout', {
                    'event_category': 'ecommerce',
                    'event_label': affiliateCode || 'direct'
                });
            }
            
            // Open in same tab or new tab based on target attribute
            if (link.target === '_blank') {
                window.open(enhancedUrl, '_blank');
            } else {
                window.location.href = enhancedUrl;
            }
        }
    }
    
    /**
     * Initialize by finding and enhancing all payment links
     */
    function init() {
        // Find all links to Stripe payment pages
        const paymentLinks = document.querySelectorAll('a[href*="buy.stripe.com"]');
        
        paymentLinks.forEach(link => {
            // Add click handler to enhance URL dynamically
            link.addEventListener('click', handlePaymentLinkClick);
        });
        
        console.log(`[Checkout] Enhanced ${paymentLinks.length} payment links`);
    }
    
    // Also expose global function for dynamic links
    window.enhancePaymentLink = enhancePaymentLink;
    
    /**
     * Get checkout URL with affiliate tracking
     * Use this for programmatic checkout triggers
     */
    window.getCheckoutUrl = function(baseUrl) {
        return enhancePaymentLink(baseUrl);
    };
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Also run on dynamic content changes (for SPAs)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1) { // Element node
                    const newLinks = node.querySelectorAll ? 
                        node.querySelectorAll('a[href*="buy.stripe.com"]') : [];
                    newLinks.forEach(link => {
                        link.addEventListener('click', handlePaymentLinkClick);
                    });
                }
            });
        });
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
    
})();

