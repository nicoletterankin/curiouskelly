/**
 * Shared Footer Partial
 * Canonical footer for all marketing pages
 * Source: index.html
 */
(function() {
  'use strict';

  const footerHTML = `
    <footer class="site-footer">
        <div class="footer-inner">
            
            <!-- Brand section at top -->
            <div class="footer-brand">
                <a href="/" class="footer-logo">
                    <img src="/images/brand/kelly-mark-circle-64.png" alt="" width="36" height="36">
                    <span>Curious Kelly</span>
                </a>
                <p class="footer-tagline">Learn something new every day</p>
                <div class="footer-social">
                    <a href="https://twitter.com/curiouskelly" aria-label="Twitter" target="_blank" rel="noopener">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
                    </a>
                    <a href="https://instagram.com/curiouskelly" aria-label="Instagram" target="_blank" rel="noopener">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/></svg>
                    </a>
                    <a href="https://tiktok.com/@curiouskelly" aria-label="TikTok" target="_blank" rel="noopener">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/></svg>
                    </a>
                    <a href="https://youtube.com/@curiouskelly" aria-label="YouTube" target="_blank" rel="noopener">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
                    </a>
                </div>
            </div>
            
            <!-- 6-column footer grid -->
            <div class="footer-grid">
                
                <!-- Column 1: Learn -->
                <div class="footer-col">
                    <h4>Learn</h4>
                    <ul class="footer-links">
                        <li><a href="/learn.html">Today's Lesson</a></li>
                        <li><a href="/curriculum.html">365 Lessons</a></li>
                        <li><a href="/calendar.html">Calendar</a></li>
                        <li><a href="/perspectives.html">Perspectives</a></li>
                        <li><a href="/player.html">Lesson Player</a></li>
                    </ul>
                </div>
                
                <!-- Column 2: Teaching Styles -->
                <div class="footer-col">
                    <h4>Teaching Styles</h4>
                    <ul class="footer-links">
                        <li><a href="/learn.html?persona=scientist">The Scientist</a></li>
                        <li><a href="/learn.html?persona=explorer">The Explorer</a></li>
                        <li><a href="/learn.html?persona=storyteller">The Storyteller</a></li>
                        <li><a href="/learn.html?persona=rebel">The Rebel</a></li>
                        <li><a href="/learn.html?persona=empath">The Empath</a></li>
                        <li><a href="/learn.html?persona=architect">The Architect</a></li>
                    </ul>
                    <span class="footer-more">+ 6 more styles</span>
                </div>
                
                <!-- Column 3: Organizations -->
                <div class="footer-col">
                    <h4>Organizations</h4>
                    <ul class="footer-links">
                        <li><a href="/enterprise.html">Schools & Districts</a></li>
                        <li><a href="/affiliates.html">Affiliate Program</a></li>
                        <li><a href="/ambassador.html">Ambassadors</a></li>
                        <li><a href="/partner.html">Partners</a></li>
                        <li><a href="/gifts.html">Gift Cards</a></li>
                    </ul>
                </div>
                
                <!-- Column 4: Company -->
                <div class="footer-col">
                    <h4>Company</h4>
                    <ul class="footer-links">
                        <li><a href="/about.html">About</a></li>
                        <li><a href="/impact.html">Impact</a></li>
                        <li><a href="/newsroom.html">Newsroom</a></li>
                        <li><a href="/careers.html">Careers</a></li>
                        <li><a href="/compare-us.html">Why Curious Kelly</a></li>
                    </ul>
                </div>
                
                <!-- Column 5: Support -->
                <div class="footer-col">
                    <h4>Support</h4>
                    <ul class="footer-links">
                        <li><a href="/help.html">Help Center</a></li>
                        <li><a href="/contact.html">Contact</a></li>
                        <li><a href="/trust.html">Trust & Safety</a></li>
                        <li><a href="/accessibility.html">Accessibility</a></li>
                        <li><a href="/status.html">System Status</a></li>
                    </ul>
                </div>
                
                <!-- Column 6: Legal -->
                <div class="footer-col">
                    <h4>Legal</h4>
                    <ul class="footer-links">
                        <li><a href="/privacy.html">Privacy Policy</a></li>
                        <li><a href="/terms.html">Terms of Service</a></li>
                        <li><a href="/trust.html#coppa">COPPA Policy</a></li>
                        <li><a href="/api.html">API</a></li>
                    </ul>
                </div>
                
            </div>
            
            <!-- Bottom bar -->
            <div class="footer-bottom">
                <div class="footer-bottom-left">
                    <span>&copy; 2025 Lesson of the Day PBC</span>
                    <a href="mailto:hello@curiouskelly.com">hello@curiouskelly.com</a>
                </div>
                <div class="footer-bottom-right">
                    <span class="footer-badge">COPPA Compliant</span>
                    <span class="footer-badge">Family Safe</span>
                </div>
            </div>
            
            <!-- Language note -->
            <div class="footer-language">
                <p>More languages coming soon. <a href="/contact.html">Request a language</a></p>
            </div>
            
        </div>
    </footer>
  `;

  // Inject footer
  const target = document.getElementById('footer-partial');
  if (target) {
    target.innerHTML = footerHTML;
  }

})();
