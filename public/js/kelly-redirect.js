/**
 * Kelly Redirect System
 * Redirects human visitors to learn.html while keeping SEO crawlers on static pages
 * 
 * Usage: Add to any page that should redirect to learn.html
 * <script src="/js/kelly-redirect.js" data-target="learn.html?tab=home&section=pricing"></script>
 */
(function() {
  'use strict';
  
  // Bot detection - keep crawlers on this page for SEO
  const botPatterns = [
    'googlebot', 'bingbot', 'slurp', 'duckduckbot', 'baiduspider',
    'yandexbot', 'facebot', 'twitterbot', 'linkedinbot', 'pinterest',
    'msnbot', 'applebot', 'semrushbot', 'ahrefsbot', 'screaming frog'
  ];
  
  const userAgent = (navigator.userAgent || '').toLowerCase();
  const isBot = botPatterns.some(bot => userAgent.includes(bot));
  
  // Don't redirect bots
  if (isBot) {
    console.log('[Kelly Redirect] Bot detected, staying on static page for SEO');
    return;
  }
  
  // Don't redirect if ?noredirect is in URL (for testing/debugging)
  if (location.search.includes('noredirect')) {
    console.log('[Kelly Redirect] noredirect flag detected, staying on page');
    return;
  }
  
  // Get redirect target from script tag data attribute
  const scriptTag = document.currentScript;
  const target = scriptTag?.dataset?.target;
  
  if (!target) {
    console.warn('[Kelly Redirect] No data-target specified');
    return;
  }
  
  // Perform redirect
  console.log('[Kelly Redirect] Redirecting to:', target);
  location.replace(target);
})();
