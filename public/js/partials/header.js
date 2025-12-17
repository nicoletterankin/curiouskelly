/**
 * Shared Header Partial
 * Canonical header for all marketing pages
 * Source: index.html
 */
(function() {
  'use strict';

  const headerHTML = `
    <header class="site-header">
        <a href="/" class="logo">
            <img src="/images/brand/kelly-mark-circle-64.png" alt="Curious Kelly" class="logo-icon" onerror="this.style.display='none'" />
            Curious Kelly
        </a>
        
        <!-- Mobile hamburger button -->
        <button class="mobile-menu-btn" onclick="toggleMobileMenu()" aria-label="Menu">
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
            <span class="hamburger-line"></span>
        </button>
        
        <nav class="nav-links" id="nav-links">
            <a href="/curriculum.html" class="nav-link" onclick="closeMobileMenu()">Curriculum</a>
            <a href="/pricing.html" class="nav-link" onclick="closeMobileMenu()">Pricing</a>
            <a href="/calendar.html" class="nav-link" onclick="closeMobileMenu()">Calendar</a>
            <a href="/about.html" class="nav-link" onclick="closeMobileMenu()">About</a>
            <a href="/learn.html" class="btn btn-primary" onclick="closeMobileMenu()">Start Learning</a>
        </nav>
    </header>
  `;

  // Inject header
  const target = document.getElementById('header-partial');
  if (target) {
    target.innerHTML = headerHTML;
  }

  // Mobile menu functions (global scope for onclick handlers)
  window.toggleMobileMenu = function() {
    var btn = document.querySelector('.mobile-menu-btn');
    var nav = document.getElementById('nav-links');
    if (!btn || !nav) return;
    btn.classList.toggle('open');
    nav.classList.toggle('open');
    document.body.style.overflow = nav.classList.contains('open') ? 'hidden' : '';
  };

  window.closeMobileMenu = function() {
    var btn = document.querySelector('.mobile-menu-btn');
    var nav = document.getElementById('nav-links');
    if (!btn || !nav) return;
    btn.classList.remove('open');
    nav.classList.remove('open');
    document.body.style.overflow = '';
  };

})();
