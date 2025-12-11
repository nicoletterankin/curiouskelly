/**
 * ✨ KELLY ANIMATION SYSTEM ✨
 * Making the site feel ALIVE
 */

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        snowflakeCount: 15, // Reduced for calm
        sparkleTrailEnabled: false, // Too noisy
        activityFeedEnabled: false, // Sleazy
        particleCount: 8, // Reduced
        isChristmasSeason: false // Remove banner
    };

    // ========================================
    // 1. SPARKLE CURSOR TRAIL
    // ========================================
    function initSparkleTrail() {
        if (!CONFIG.sparkleTrailEnabled) return;
        
        let lastX = 0, lastY = 0;
        let throttle = false;
        
        document.addEventListener('mousemove', (e) => {
            if (throttle) return;
            
            // Only create sparkle if mouse moved enough
            const distance = Math.hypot(e.clientX - lastX, e.clientY - lastY);
            if (distance < 30) return;
            
            throttle = true;
            setTimeout(() => throttle = false, 50);
            
            lastX = e.clientX;
            lastY = e.clientY;
            
            createSparkle(e.clientX, e.clientY);
        });
    }

    function createSparkle(x, y) {
        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle-cursor';
        sparkle.style.left = (x - 4) + 'px';
        sparkle.style.top = (y - 4) + 'px';
        document.body.appendChild(sparkle);
        
        setTimeout(() => sparkle.remove(), 600);
    }

    // ========================================
    // 2. CHRISTMAS SNOWFALL
    // ========================================
    function initSnowfall() {
        if (!CONFIG.isChristmasSeason) return;
        
        const snowflakes = ['❄', '❅', '❆', '✦', '✧'];
        
        for (let i = 0; i < CONFIG.snowflakeCount; i++) {
            setTimeout(() => createSnowflake(snowflakes), i * 200);
        }
    }

    function createSnowflake(snowflakes) {
        const flake = document.createElement('div');
        flake.className = 'snowflake';
        flake.textContent = snowflakes[Math.floor(Math.random() * snowflakes.length)];
        flake.style.left = Math.random() * 100 + 'vw';
        flake.style.animationDuration = (Math.random() * 10 + 8) + 's';
        flake.style.opacity = Math.random() * 0.5 + 0.3;
        flake.style.fontSize = (Math.random() * 0.8 + 0.5) + 'rem';
        document.body.appendChild(flake);
        
        // Recycle snowflake after animation
        flake.addEventListener('animationend', () => {
            flake.style.left = Math.random() * 100 + 'vw';
            flake.style.animationDuration = (Math.random() * 10 + 8) + 's';
        });
    }

    // ========================================
    // 3. DAY COUNTER BADGE
    // ========================================
    function initDayCounter() {
        // Calculate day of year
        const now = new Date();
        const start = new Date(now.getFullYear(), 0, 0);
        const diff = now - start;
        const oneDay = 1000 * 60 * 60 * 24;
        const dayOfYear = Math.floor(diff / oneDay);
        
        // Create badge
        const badge = document.createElement('div');
        badge.className = 'day-counter-badge';
        badge.innerHTML = `
            <span class="sparkle">✨</span>
            <div class="counter-text">
                <span class="day-number">Day ${dayOfYear}</span>
                <span class="day-label">of 365</span>
            </div>
        `;
        badge.onclick = () => window.location.href = '/learn.html';
        badge.title = 'Start today\'s lesson';
        
        document.body.appendChild(badge);
        
        // Animate number on hover
        badge.addEventListener('mouseenter', () => {
            const numEl = badge.querySelector('.day-number');
            numEl.style.transform = 'scale(1.1)';
            numEl.style.transition = 'transform 0.2s';
        });
        badge.addEventListener('mouseleave', () => {
            const numEl = badge.querySelector('.day-number');
            numEl.style.transform = 'scale(1)';
        });
    }

    // ========================================
    // 4. LIVE ACTIVITY FEED
    // ========================================
    const CITIES = [
        'Tokyo', 'London', 'New York', 'Paris', 'Sydney', 
        'Berlin', 'Toronto', 'São Paulo', 'Mumbai', 'Singapore',
        'Los Angeles', 'Chicago', 'Mexico City', 'Dubai', 'Seoul',
        'Amsterdam', 'Stockholm', 'Dublin', 'Melbourne', 'Vancouver'
    ];
    
    const ACTIVITIES = [
        { icon: '📚', text: 'just started learning' },
        { icon: '🎯', text: 'completed today\'s lesson' },
        { icon: '🔥', text: 'is on a 7-day streak' },
        { icon: '✨', text: 'earned their first badge' },
        { icon: '🎓', text: 'finished this month\'s topics' }
    ];

    function initActivityFeed() {
        if (!CONFIG.activityFeedEnabled) return;
        
        // Create feed container
        const feed = document.createElement('div');
        feed.className = 'activity-feed';
        document.body.appendChild(feed);
        
        // Show initial activity
        setTimeout(() => showActivity(feed), 3000);
        
        // Show periodic activities
        setInterval(() => showActivity(feed), 12000);
    }

    function showActivity(feed) {
        // Only show on desktop
        if (window.innerWidth < 768) return;
        
        const city = CITIES[Math.floor(Math.random() * CITIES.length)];
        const activity = ACTIVITIES[Math.floor(Math.random() * ACTIVITIES.length)];
        
        const item = document.createElement('div');
        item.className = 'activity-item';
        item.innerHTML = `
            <span class="activity-icon">${activity.icon}</span>
            <span class="activity-text">Someone in <span class="activity-location">${city}</span> ${activity.text}</span>
        `;
        
        feed.appendChild(item);
        
        // Trigger animation
        requestAnimationFrame(() => {
            item.style.animation = 'activity-slide-in 0.5s ease-out forwards, activity-fade-out 0.5s ease-in 4.5s forwards';
        });
        
        // Remove after animation
        setTimeout(() => item.remove(), 5000);
        
        // Keep max 3 items
        while (feed.children.length > 3) {
            feed.firstChild.remove();
        }
    }

    // ========================================
    // 5. FLOATING PARTICLES BACKGROUND
    // ========================================
    function initParticles() {
        for (let i = 0; i < CONFIG.particleCount; i++) {
            setTimeout(() => createParticle(), i * 2000);
        }
    }

    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 100 + 50;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.animationDuration = (Math.random() * 20 + 15) + 's';
        particle.style.animationDelay = Math.random() * 5 + 's';
        
        document.body.appendChild(particle);
        
        // Recycle particle
        particle.addEventListener('animationend', () => {
            particle.style.left = Math.random() * 100 + 'vw';
            particle.style.animationDuration = (Math.random() * 20 + 15) + 's';
        });
    }

    // ========================================
    // 6. SECTION REVEAL ON SCROLL
    // ========================================
    function initScrollReveal() {
        const sections = document.querySelectorAll('section:not(.hero)');
        
        sections.forEach(section => {
            section.classList.add('section-reveal');
        });
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        });
        
        sections.forEach(section => observer.observe(section));
    }

    // ========================================
    // 7. SPARKLE HOVER EFFECTS ON BUTTONS
    // ========================================
    function initSparkleHovers() {
        const buttons = document.querySelectorAll('.btn-primary, .btn-accent, .today-cta');
        buttons.forEach(btn => btn.classList.add('sparkle-hover'));
    }

    // ========================================
    // 8. CHRISTMAS BANNER
    // ========================================
    function initChristmasBanner() {
        if (!CONFIG.isChristmasSeason) return;
        
        // Check if already exists
        if (document.querySelector('.christmas-banner')) return;
        
        const banner = document.createElement('div');
        banner.className = 'christmas-banner';
        banner.innerHTML = `Give the gift of learning! <a href="/gifts.html">Shop Holiday Gifts</a> — Perfect for everyone on your list`;
        
        // Insert before header
        document.body.insertBefore(banner, document.body.firstChild);
        
        // Adjust header position
        const header = document.querySelector('header');
        if (header) {
            header.style.top = '44px'; // Height of banner
        }
    }

    // ========================================
    // 9. KELLY'S GREETING (on first visit)
    // ========================================
    function initKellyGreeting() {
        // Check if first visit in session
        if (sessionStorage.getItem('kellyGreeted')) return;
        sessionStorage.setItem('kellyGreeted', 'true');
        
        // Create greeting overlay
        setTimeout(() => {
            const greeting = document.createElement('div');
            greeting.id = 'kelly-greeting';
            greeting.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 20px;
                padding: 20px 24px;
                max-width: 320px;
                z-index: 10000;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                animation: greeting-slide 0.5s ease-out;
            `;
            
            greeting.innerHTML = `
                <style>
                    @keyframes greeting-slide {
                        from { opacity: 0; transform: translateY(20px) scale(0.95); }
                        to { opacity: 1; transform: translateY(0) scale(1); }
                    }
                </style>
                <div style="display: flex; align-items: flex-start; gap: 14px;">
                    <img src="/images/brand/kelly-mark-circle-64.png" alt="Kelly" style="width: 48px; height: 48px; border-radius: 50%; border: 2px solid rgba(59,130,246,0.4);" onerror="this.style.display='none'">
                    <div>
                        <div style="font-weight: 600; margin-bottom: 6px; color: #f1f5f9;">Hey there! ✨</div>
                        <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.5;">I'm Kelly. I find something fascinating every day and can't wait to share it with you.</div>
                    </div>
                    <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: #64748b; cursor: pointer; padding: 4px; font-size: 1.2rem; line-height: 1;">&times;</button>
                </div>
                <a href="/learn.html" style="display: block; margin-top: 16px; padding: 12px 20px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; text-align: center; border-radius: 12px; text-decoration: none; font-weight: 600; font-size: 0.9rem;">Start Today's Lesson</a>
            `;
            
            document.body.appendChild(greeting);
            
            // Auto-dismiss after 10 seconds
            setTimeout(() => {
                if (greeting.parentNode) {
                    greeting.style.animation = 'greeting-slide 0.3s ease-in reverse';
                    setTimeout(() => greeting.remove(), 300);
                }
            }, 10000);
        }, 2000);
    }

    // ========================================
    // INITIALIZATION
    // ========================================
    function init() {
        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initAll);
        } else {
            initAll();
        }
    }

    function initAll() {
        console.log('✨ Kelly Animation System initializing...');
        
        // Core animations (Subtle only)
        // initSparkleTrail(); // Disabled
        initSnowfall();
        // initDayCounter(); // Disabled - "Day X of 365" is sleazy
        // initActivityFeed(); // Disabled - Fake social proof is sleazy
        initParticles();
        initScrollReveal();
        initSparkleHovers();
        
        // Seasonal
        // initChristmasBanner(); // Disabled
        
        // Kelly personality
        // initKellyGreeting(); // Disabled - Popups are annoying
        
        console.log('✨ Kelly Animation System ready!');
    }

    // Start!
    init();
})();

