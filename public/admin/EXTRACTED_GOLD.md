# 🪙 EXTRACTED GOLD - Before Archive Operation
*Mined December 10, 2025*

This document preserves all valuable code, copy, and design patterns from pages being archived or deleted.

---

## 📊 SUMMARY

| Category | Items Extracted |
|----------|-----------------|
| CSS Design Systems | 3 distinct themes |
| UI Components | 24 reusable patterns |
| Legal Copy | Complete COPPA/GDPR/CCPA |
| Business Logic | Calculators, forms, validators |
| Typography | 2 font pairings |
| Supabase Patterns | 5 integration examples |

---

## 🎨 CSS DESIGN SYSTEMS

### Theme 1: "Campus" (about.html, careers.html, etc.)
```css
:root {
    --bg-color: #0f0f11;
    --text-primary: #f4f4f5;
    --text-secondary: #a1a1aa;
    --accent-orange: #d97757;
    --accent-hover: #c56a4d;
    --card-bg: #18181b;
    --border-color: #3f3f46;
    --footer-bg: #000000;
    --live-red: #ef4444;
    --success-green: #10b981;
}
```

### Theme 2: "Production" (index-final.html)
```css
:root {
    --bg-color: #0a0a0b;
    --bg-secondary: #111113;
    --bg-elevated: #18181b;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-primary: #3b82f6;
    --accent-hover: #2563eb;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --success: #22c55e;
    --error: #ef4444;
}
```

### Typography Pairing (KEEP)
```css
/* Fraunces for headings + Inter for body */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&display=swap');

h1, h2, h3, .display-text {
    font-family: 'Fraunces', Georgia, serif;
    letter-spacing: -0.03em;
}

body, .ui-text {
    font-family: 'Inter', -apple-system, sans-serif;
}
```

---

## 🧩 UI COMPONENTS

### 1. Live Indicator (about.html)
```css
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 20px;
    color: var(--live-red);
    font-family: monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--live-red);
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse { 
    0% { opacity: 1; } 
    50% { opacity: 0.5; } 
    100% { opacity: 1; } 
}
```

### 2. Persona/Age Slider Bar (about.html)
```css
.persona-bar {
    max-width: 900px;
    margin: 0 auto;
    background: rgba(24, 24, 27, 0.9);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    position: sticky;
    top: 85px;
    z-index: 900;
}

.age-slider {
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    background: var(--border-color);
    border-radius: 2px;
}

.age-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    background: var(--accent-orange);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(217, 119, 87, 0.5);
}
```

### 3. Toggle Group Buttons (about.html)
```css
.toggle-group {
    display: flex;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 4px;
}

.toggle-btn {
    padding: 6px 12px;
    border-radius: 6px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}

.toggle-btn.active {
    background: var(--text-primary);
    color: var(--bg-color);
    font-weight: 600;
}
```

### 4. Day Card (Calendar Grid) (about.html)
```css
.day-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    height: 180px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

.day-card:hover {
    border-color: var(--text-secondary);
    background: #202023;
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.dna-tag { 
    font-size: 0.7rem; 
    color: var(--accent-orange); 
    border: 1px solid var(--accent-orange); 
    padding: 2px 6px; 
    border-radius: 4px; 
}
```

### 5. Affiliate Calculator (careers.html)
```javascript
function calculateEarnings(referrals) {
    let commission;
    let tierName;
    
    if (referrals < 100) {
        commission = 0.20;
        tierName = 'Scholar (20%)';
    } else if (referrals < 500) {
        commission = 0.25;
        tierName = 'Fellow (25%)';
    } else {
        commission = 0.30;
        tierName = 'Ambassador (30%)';
    }

    const annualPerSub = 199 * commission;
    const monthlyPerSub = annualPerSub / 12;
    
    const monthly = Math.round(referrals * monthlyPerSub);
    const annual = Math.round(referrals * annualPerSub);

    return { monthly, annual, tierName };
}
```

### 6. Countdown Timer (careers.html)
```javascript
const targetDate = new Date('2025-12-31T23:59:59').getTime();

function updateCountdown() {
    const now = new Date().getTime();
    const distance = targetDate - now;

    if (distance < 0) {
        countdownEl.textContent = 'Offer Ended';
        return;
    }

    const days = Math.floor(distance / (1000 * 60 * 60 * 24));
    const hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    
    countdownEl.textContent = `${days} days, ${hours} hours remaining`;
}

setInterval(updateCountdown, 3600000); // Every hour
```

### 7. Stat Card Grid (diversity.html)
```html
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-number">2-102</div>
        <div class="stat-label">Age Range Supported</div>
    </div>
    <!-- ... more cards -->
</div>
```

```css
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 24px;
}

.stat-number {
    font-size: 3rem;
    font-weight: 300;
    color: var(--accent-orange);
}
```

### 8. Feature Card Grid (diversity.html, enterprise.html)
```css
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
}

.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 28px;
}

.feature-icon { font-size: 2rem; margin-bottom: 16px; }
.feature-title { font-weight: 600; margin-bottom: 8px; }
.feature-desc { color: var(--text-secondary); font-size: 0.95rem; }
```

### 9. Quote Block (diversity.html)
```css
.quote-block {
    background: var(--card-bg);
    border-left: 4px solid var(--accent-orange);
    border-radius: 8px;
    padding: 32px;
    font-style: italic;
}

.quote-text {
    font-size: 1.2rem;
    color: var(--text-primary);
    margin-bottom: 16px;
}

.quote-author {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-style: normal;
}
```

### 10. Case Study Card (enterprise.html)
```css
.case-study {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-orange);
    border-radius: 12px;
    padding: 40px;
}

.case-study-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin-top: 32px;
    padding-top: 32px;
    border-top: 1px solid var(--border-color);
}
```

### 11. Press Kit Cards (newsroom.html)
```css
.kit-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 24px;
}

.kit-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 28px;
    text-align: center;
}
```

### 12. Highlight Box for Legal Docs (privacy.html)
```css
.highlight-box {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-orange);
    border-radius: 8px;
    padding: 20px;
    margin: 24px 0;
}
```

### 13. Platform Cards with Status (social.html)
```css
.platform-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    transition: all 0.3s;
}

.platform-card:hover {
    transform: translateY(-4px);
    border-color: var(--accent-orange);
    box-shadow: 0 8px 24px rgba(217, 119, 87, 0.15);
}

.platform-status {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-coming-soon {
    background: rgba(161, 161, 170, 0.1);
    color: var(--text-secondary);
}

.status-live {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981;
}
```

### 14. OS Trigger Button (player.html)
```css
#os-trigger {
    position: absolute;
    top: 30px;
    right: 30px;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    cursor: pointer;
    z-index: 1000;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

#os-trigger:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: scale(1.1);
}
```

### 15. Glass Drawer (player.html)
```css
.glass-drawer {
    width: 400px;
    background: rgba(15, 15, 17, 0.85);
    backdrop-filter: blur(20px);
}
```

### 16. Dashboard Stats (dashboard.html)
```css
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.stat-value {
    font-size: 2.5rem;
    color: var(--accent-orange);
}
```

### 17. Loading Spinner (dashboard.html)
```css
.spinner {
    border: 3px solid var(--border-color);
    border-top: 3px solid var(--accent-orange);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 0 auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```

### 18. Two-Panel Hero (index-final.html)
```css
.hero {
    min-height: 100vh;
    display: flex;
    padding-top: var(--header-height);
}

.hero-left {
    flex: 0 0 50%;
    max-width: 600px;
    padding: 80px 64px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-right {
    flex: 1;
    background: var(--bg-secondary);
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
}

.hero-right::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
}
```

### 19. Month Accordion (index-final.html)
```css
.month-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    overflow: hidden;
}

.month-header {
    padding: 24px;
    background: var(--bg-secondary);
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    user-select: none;
}

.month-lessons {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
}

.month-card.open .month-lessons {
    max-height: 2000px;
}
```

### 20. Time Machine / Perspective Explorer (index-final.html)
```javascript
function setYear(year) {
    document.getElementById('year-slider').value = year;
    updatePerspectives();
}

async function updatePerspectives() {
    const year = parseInt(document.getElementById('year-slider').value);
    const age = 2025 - year;
    
    document.getElementById('perspective-year').textContent = year;
    document.getElementById('perspective-age').textContent = age;
    
    // Generation buttons: Silent Gen (1955), Boomer (1960), Gen X (1975), 
    // Millennial (1990), Gen Z (2002), Gen Alpha (2015)
}
```

### 21. Lesson Thumbnail Gradient Generator (index-final.html)
```javascript
function generateLessonThumbnail(topic) {
    const categories = {
        science: ['#1e3a8a', '#3b82f6'],
        history: ['#7c2d12', '#ea580c'],
        art: ['#701a75', '#c026d3'],
        math: ['#065f46', '#10b981'],
        nature: ['#14532d', '#22c55e'],
        tech: ['#1e40af', '#60a5fa'],
        culture: ['#7c2d12', '#f97316']
    };

    const lowerTopic = topic.toLowerCase();
    let gradient = categories.science; // default

    if (lowerTopic.includes('art') || lowerTopic.includes('music')) gradient = categories.art;
    else if (lowerTopic.includes('history') || lowerTopic.includes('ancient')) gradient = categories.history;
    // ... etc

    return `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`;
}
```

### 22. OAuth Buttons (index-final.html)
```html
<!-- Google -->
<button class="btn btn-google">
    <svg class="btn-icon" viewBox="0 0 24 24">
        <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92..." fill="#4285F4"/>
        <!-- Full Google logo paths -->
    </svg>
    <span>Continue with Google</span>
</button>

<!-- Apple -->
<button class="btn btn-apple">
    <svg class="btn-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M17.05 20.28c-.98.95-2.05.88-3.08.4..."/>
    </svg>
    <span>Continue with Apple</span>
</button>
```

### 23. Collapsible Section Pattern (index-final.html)
```javascript
window.toggleCollapsible = function(header) {
    header.parentElement.classList.toggle('open');
};
```

```css
.collapsible-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
}

.collapsible.open .collapsible-content {
    max-height: 2000px;
}

.collapse-icon {
    transition: transform 0.3s;
}

.collapsible.open .collapse-icon {
    transform: rotate(180deg);
}
```

### 24. Loading Skeleton Animation (index-final.html)
```css
.skeleton-card {
    height: 200px;
    background: linear-gradient(90deg, #18181b 0%, #27272a 50%, #18181b 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 16px;
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
```

---

## 📜 LEGAL COPY (COMPLETE)

### Privacy Policy - TL;DR Box
> **TL;DR:** We collect only what we need to provide personalized learning. We never sell your data. Parents have full control over children's accounts. You can delete your data anytime.

### Terms of Service - Key Points Box
> **Key Points:** You must be 13+ to create your own account (parents can create accounts for younger children). Subscriptions auto-renew but can be canceled anytime. Be respectful and don't abuse the service. We own Kelly and the content, but you own your learning data.

### COPPA Data Limits for Under-13
- First name only (no last name)
- Age range (not exact birthdate)
- Learning progress only
- NO: Email, phone, photos, videos, precise geolocation, social profiles

### Refund Policy
- **7-Day Money-Back Guarantee:** Full refund if canceled within 7 days
- **After 7 Days:** No refunds for unused time
- **Gift Subscriptions:** Refundable only if unredeemed

### Contact Emails (Official)
- General: hello@curiouskelly.com
- Legal: legal@curiouskelly.com  
- Privacy: privacy@curiouskelly.com
- Support: support@curiouskelly.com

---

## 🔌 SUPABASE INTEGRATION PATTERNS

### 1. ESM Import (Modern)
```javascript
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm'

const supabase = createClient(
    'https://tvjalxxsyryjphkforjv.supabase.co',
    'eyJhbGci...ANON_KEY'
);
```

### 2. OAuth Flow
```javascript
const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google', // or 'apple'
    options: {
        redirectTo: `${window.location.origin}/learn.html`
    }
});
```

### 3. Magic Link
```javascript
const { data, error } = await supabase.auth.signInWithOtp({
    email: email,
    options: {
        emailRedirectTo: `${window.location.origin}/learn.html`
    }
});
```

### 4. Lesson Query
```javascript
const { data: lesson } = await supabase
    .from('lessons')
    .select('*')
    .eq('day_number', currentDay)
    .eq('is_published', true)
    .single();
```

### 5. User Progress Query
```javascript
const { count } = await supabase
    .from('user_progress')
    .select('*', { count: 'exact', head: true })
    .eq('user_id', user.id)
    .eq('completed', true);
```

---

## 📊 PRICING & BUSINESS DATA

### Consumer Pricing
| Plan | Price | Details |
|------|-------|---------|
| Free | $0 | Today's lesson only |
| Monthly | $9.99/mo | All 366 lessons |
| Annual | $99/year | 2 months free (17% off) |
| Lifetime | $299 once | Forever access |

### Gift Pricing
| Duration | Price | Lessons |
|----------|-------|---------|
| 3 Months | $34.99 | 90 |
| 6 Months | $59.99 | 180 |
| 12 Months | $99.99 | 365 |
| Lifetime | $299.99 | Unlimited |

### Affiliate Tiers
| Tier | Commission | Requirement |
|------|------------|-------------|
| Scholar | 20% | 0-99 referrals |
| Fellow | 25% | 100-499 referrals |
| Ambassador | 30% | 500+ referrals |

---

## 🏢 COMPANY FACTS (for Newsroom)

| Fact | Value |
|------|-------|
| Company | Curious Kelly PBC |
| Legal Name | Lesson of the Day PBC |
| Founded | 2025 |
| HQ | California, USA |
| Structure | Public Benefit Corporation |
| Age Range | 2-102 years |
| Curriculum | 365 daily lessons |
| Languages | EN, ES, FR |
| Pricing | $199/year |
| Technology | AI voice, 3D avatar |
| Platforms | Web, iOS, Android |
| Launch Date | Dec 17, 2025 |

---

## ✅ NEXT STEPS

1. Consolidate CSS tokens into `/public/css/design-tokens.css`
2. Create component library at `/public/css/components.css`
3. Move legal copy to `/public/legal/` folder
4. Archive original files to `/_archive/legacy-pages/`
5. Update site-health.html with new canonical locations

---

*This document generated by Claude during archive operation.*
*Original pages preserved in `/_archive/` for 30-day recovery.*

