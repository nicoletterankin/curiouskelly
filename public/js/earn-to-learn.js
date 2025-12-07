/**
 * Earn to Learn - Share & Earn System
 * 
 * PHILOSOPHY: Every learner is an affiliate from Day 1.
 * LIFETIME ATTRIBUTION: Once you refer someone, you're credited FOREVER.
 * 
 * "Learn to Earn. Earn to Learn. Teach to Grow."
 */

const EarnToLearn = {
  isOpen: false,
  overlay: null,
  userData: null,
  supabase: null,
  eligibility: null, // COPPA/age compliance data
  
  // Commission tier names and rates
  TIERS: {
    'new_learner': { name: 'New Learner', rate: 10, emoji: '🌱', lessons: 0 },
    'active_learner': { name: 'Active Learner', rate: 15, emoji: '📚', lessons: 7 },
    'committed_learner': { name: 'Committed Learner', rate: 20, emoji: '🎯', lessons: 30 },
    'dedicated_learner': { name: 'Dedicated Learner', rate: 25, emoji: '⭐', lessons: 100 },
    'complete_learner': { name: 'Complete Learner', rate: 30, emoji: '👑', lessons: 365 },
    'legendary_learner': { name: 'Legendary Learner', rate: 35, emoji: '🏆', lessons: 1000 }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init() {
    this.initSupabase();
    this.createPanel();
    this.bindEvents();
    console.log('[EarnToLearn] Initialized');
  },
  
  initSupabase() {
    if (window.supabase) {
      this.supabase = window.supabase;
    } else if (window.supabaseClient) {
      this.supabase = window.supabaseClient;
    }
  },
  
  createPanel() {
    if (this.overlay) return;
    
    this.overlay = document.createElement('div');
    this.overlay.id = 'earn-overlay';
    this.overlay.className = 'earn-overlay';
    this.overlay.innerHTML = `
      <div class="earn-container">
        
        <!-- Header -->
        <div class="earn-header">
          <div class="earn-title">
            <span class="earn-icon">💰</span>
            <h2>Share & Earn</h2>
          </div>
          <button class="earn-close" id="earn-close" aria-label="Close">✕</button>
        </div>
        
        <!-- Earnings Summary -->
        <div class="earn-summary" id="earn-summary">
          <div class="earn-stat-row">
            <div class="earn-stat">
              <span class="earn-stat-value" id="available-earnings">$0.00</span>
              <span class="earn-stat-label">Available</span>
            </div>
            <div class="earn-stat">
              <span class="earn-stat-value" id="pending-earnings">$0.00</span>
              <span class="earn-stat-label">Pending</span>
            </div>
            <div class="earn-stat">
              <span class="earn-stat-value" id="lifetime-earnings">$0.00</span>
              <span class="earn-stat-label">Lifetime</span>
            </div>
          </div>
        </div>
        
        <!-- Current Tier -->
        <div class="earn-tier-section" id="earn-tier">
          <div class="tier-current">
            <span class="tier-emoji" id="tier-emoji">🌱</span>
            <div class="tier-info">
              <span class="tier-name" id="tier-name">New Learner</span>
              <span class="tier-rate" id="tier-rate">10% commission</span>
            </div>
          </div>
          <div class="tier-progress">
            <div class="tier-bar">
              <div class="tier-bar-fill" id="tier-bar-fill" style="width: 0%"></div>
            </div>
            <span class="tier-next" id="tier-next">7 more lessons for Active Learner (15%)</span>
          </div>
        </div>
        
        <!-- Referral Link -->
        <div class="earn-link-section">
          <label class="earn-label">Your Referral Link (LIFETIME attribution)</label>
          <div class="earn-link-row">
            <input type="text" class="earn-link-input" id="referral-link" readonly value="Loading...">
            <button class="earn-link-copy" id="copy-link-btn">
              <span class="copy-icon">📋</span>
              <span class="copy-text">Copy</span>
            </button>
          </div>
          <p class="earn-link-note">
            When someone uses your link, you're credited <strong>forever</strong> — even if they sign up years later.
          </p>
        </div>
        
        <!-- Quick Stats -->
        <div class="earn-quick-stats" id="quick-stats">
          <div class="quick-stat">
            <span class="quick-stat-value" id="total-clicks">0</span>
            <span class="quick-stat-label">Clicks</span>
          </div>
          <div class="quick-stat">
            <span class="quick-stat-value" id="total-signups">0</span>
            <span class="quick-stat-label">Sign-ups</span>
          </div>
          <div class="quick-stat">
            <span class="quick-stat-value" id="total-active">0</span>
            <span class="quick-stat-label">Active</span>
          </div>
        </div>
        
        <!-- Share Buttons -->
        <div class="earn-share-section">
          <label class="earn-label">Share & Start Earning</label>
          <div class="share-button-grid">
            <button class="share-btn twitter" data-platform="twitter">
              <span class="share-icon">𝕏</span>
              <span class="share-name">Twitter</span>
            </button>
            <button class="share-btn facebook" data-platform="facebook">
              <span class="share-icon">f</span>
              <span class="share-name">Facebook</span>
            </button>
            <button class="share-btn whatsapp" data-platform="whatsapp">
              <span class="share-icon">💬</span>
              <span class="share-name">WhatsApp</span>
            </button>
            <button class="share-btn email" data-platform="email">
              <span class="share-icon">✉️</span>
              <span class="share-name">Email</span>
            </button>
            <button class="share-btn linkedin" data-platform="linkedin">
              <span class="share-icon">in</span>
              <span class="share-name">LinkedIn</span>
            </button>
            <button class="share-btn native" id="native-share-btn">
              <span class="share-icon">📤</span>
              <span class="share-name">Share...</span>
            </button>
          </div>
        </div>
        
        <!-- Footer Actions -->
        <div class="earn-footer">
          <a href="/earnings.html" class="earn-footer-link">
            📊 View Full Dashboard
          </a>
          <a href="#" class="earn-footer-link" id="request-payout-link" style="display: none;">
            💸 Request Payout ($50 min)
          </a>
        </div>
        
        <!-- Not Logged In State -->
        <div class="earn-not-logged-in" id="earn-not-logged-in" style="display: none;">
          <div class="not-logged-icon">🔐</div>
          <h3>Sign in to Share & Earn</h3>
          <p>Every learner gets a referral link. Create an account to start earning!</p>
          <button class="earn-signin-btn" id="earn-signin-btn">Sign In / Sign Up</button>
        </div>
        
        <!-- Under 13 State (COPPA) -->
        <div class="earn-under-13" id="earn-under-13" style="display: none;">
          <div class="not-logged-icon">👶</div>
          <h3>Ask a Parent to Help!</h3>
          <p>Kids under 13 need a parent or guardian to set up Share & Earn. It's the law to keep you safe!</p>
          <p class="earn-parent-note">Your parent can link your account to theirs, and they'll manage any earnings until you're older.</p>
          <a href="mailto:hello@curiouskelly.com?subject=Family%20Account%20Setup" class="earn-signin-btn" style="text-decoration: none; display: inline-block;">
            📧 Email Us for Help
          </a>
        </div>
        
        <!-- Minor State (13-17) -->
        <div class="earn-minor" id="earn-minor" style="display: none;">
          <div class="minor-notice">
            <span class="minor-icon">🔒</span>
            <div class="minor-text">
              <strong>Your earnings are being saved!</strong>
              <p>As you're under 18, your earnings are held until your 18th birthday. 
              If a parent is linked to your account, they can claim them earlier.</p>
            </div>
          </div>
          <div class="held-earnings-display">
            <span class="held-label">Held Earnings</span>
            <span class="held-amount" id="held-earnings-amount">$0.00</span>
          </div>
        </div>
        
      </div>
    `;
    
    this.addStyles();
    document.body.appendChild(this.overlay);
  },
  
  addStyles() {
    if (document.getElementById('earn-to-learn-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'earn-to-learn-styles';
    styles.textContent = `
      .earn-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        z-index: 9500;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
      }
      
      .earn-overlay.open {
        opacity: 1;
        pointer-events: auto;
      }
      
      .earn-container {
        width: 100%;
        max-width: 480px;
        max-height: 90vh;
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 28px 28px 0 0;
        padding: 24px 24px 32px;
        overflow-y: auto;
        transform: translateY(100%);
        transition: transform 0.4s cubic-bezier(0.32, 0.72, 0, 1);
      }
      
      .earn-overlay.open .earn-container {
        transform: translateY(0);
      }
      
      @media (min-width: 768px) {
        .earn-overlay {
          align-items: center;
        }
        
        .earn-container {
          border-radius: 24px;
          max-height: 85vh;
          transform: translateY(20px) scale(0.95);
          opacity: 0;
        }
        
        .earn-overlay.open .earn-container {
          transform: translateY(0) scale(1);
          opacity: 1;
        }
      }
      
      .earn-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
      }
      
      .earn-title {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .earn-icon {
        font-size: 1.8rem;
      }
      
      .earn-header h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
      }
      
      .earn-close {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #fff;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-size: 1.2rem;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .earn-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }
      
      /* Earnings Summary */
      .earn-summary {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
      }
      
      .earn-stat-row {
        display: flex;
        justify-content: space-around;
      }
      
      .earn-stat {
        text-align: center;
      }
      
      .earn-stat-value {
        display: block;
        font-size: 1.6rem;
        font-weight: 700;
        color: #22c55e;
      }
      
      .earn-stat-label {
        font-size: 0.8rem;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      
      /* Tier Section */
      .earn-tier-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
      }
      
      .tier-current {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
      }
      
      .tier-emoji {
        font-size: 2.5rem;
      }
      
      .tier-info {
        display: flex;
        flex-direction: column;
      }
      
      .tier-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fff;
      }
      
      .tier-rate {
        font-size: 0.95rem;
        color: #3b82f6;
        font-weight: 500;
      }
      
      .tier-progress {
        margin-top: 8px;
      }
      
      .tier-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
      }
      
      .tier-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
        transition: width 0.5s ease;
      }
      
      .tier-next {
        display: block;
        font-size: 0.8rem;
        color: #71717a;
        margin-top: 8px;
        text-align: center;
      }
      
      /* Referral Link */
      .earn-link-section {
        margin-bottom: 20px;
      }
      
      .earn-label {
        display: block;
        font-size: 0.9rem;
        font-weight: 600;
        color: #d1d5db;
        margin-bottom: 10px;
      }
      
      .earn-link-row {
        display: flex;
        gap: 8px;
      }
      
      .earn-link-input {
        flex: 1;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #fff;
        font-size: 0.95rem;
        font-family: monospace;
      }
      
      .earn-link-input:focus {
        outline: none;
        border-color: #3b82f6;
      }
      
      .earn-link-copy {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 14px 20px;
        background: #3b82f6;
        border: none;
        border-radius: 12px;
        color: #fff;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .earn-link-copy:hover {
        background: #2563eb;
        transform: scale(1.02);
      }
      
      .earn-link-copy.copied {
        background: #22c55e;
      }
      
      .earn-link-note {
        font-size: 0.8rem;
        color: #71717a;
        margin-top: 10px;
        line-height: 1.5;
      }
      
      .earn-link-note strong {
        color: #3b82f6;
      }
      
      /* Quick Stats */
      .earn-quick-stats {
        display: flex;
        justify-content: space-around;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
      }
      
      .quick-stat {
        text-align: center;
      }
      
      .quick-stat-value {
        display: block;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
      }
      
      .quick-stat-label {
        font-size: 0.75rem;
        color: #71717a;
        text-transform: uppercase;
      }
      
      /* Share Buttons */
      .earn-share-section {
        margin-bottom: 20px;
      }
      
      .share-button-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
      }
      
      .share-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 16px 12px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        color: #fff;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .share-btn:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
      }
      
      .share-icon {
        font-size: 1.5rem;
      }
      
      .share-name {
        font-size: 0.75rem;
        color: #a1a1aa;
      }
      
      .share-btn.twitter:hover { background: rgba(29, 161, 242, 0.2); border-color: #1da1f2; }
      .share-btn.facebook:hover { background: rgba(24, 119, 242, 0.2); border-color: #1877f2; }
      .share-btn.whatsapp:hover { background: rgba(37, 211, 102, 0.2); border-color: #25d366; }
      .share-btn.linkedin:hover { background: rgba(10, 102, 194, 0.2); border-color: #0a66c2; }
      .share-btn.email:hover { background: rgba(234, 67, 53, 0.2); border-color: #ea4335; }
      .share-btn.native:hover { background: rgba(59, 130, 246, 0.2); border-color: #3b82f6; }
      
      /* Footer */
      .earn-footer {
        display: flex;
        justify-content: center;
        gap: 20px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
      }
      
      .earn-footer-link {
        color: #71717a;
        font-size: 0.9rem;
        text-decoration: none;
        transition: color 0.2s;
      }
      
      .earn-footer-link:hover {
        color: #3b82f6;
      }
      
      /* Not Logged In */
      .earn-not-logged-in {
        text-align: center;
        padding: 40px 20px;
      }
      
      .not-logged-icon {
        font-size: 3rem;
        margin-bottom: 16px;
      }
      
      .earn-not-logged-in h3 {
        font-size: 1.3rem;
        margin-bottom: 8px;
        color: #fff;
      }
      
      .earn-not-logged-in p {
        color: #a1a1aa;
        margin-bottom: 24px;
      }
      
      .earn-signin-btn {
        padding: 14px 32px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border: none;
        border-radius: 12px;
        color: #fff;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .earn-signin-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
      }
      
      /* Toast */
      .earn-toast {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: #22c55e;
        color: white;
        padding: 14px 28px;
        border-radius: 14px;
        font-size: 0.95rem;
        font-weight: 600;
        z-index: 10001;
        animation: earnToastIn 0.3s ease-out;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4);
      }
      
      @keyframes earnToastIn {
        from { opacity: 0; transform: translate(-50%, 20px); }
        to { opacity: 1; transform: translate(-50%, 0); }
      }
      
      /* Under 13 State (COPPA) */
      .earn-under-13 {
        text-align: center;
        padding: 40px 20px;
      }
      
      .earn-under-13 h3 {
        font-size: 1.3rem;
        margin-bottom: 12px;
        color: #fff;
      }
      
      .earn-under-13 p {
        color: #a1a1aa;
        margin-bottom: 16px;
        line-height: 1.6;
      }
      
      .earn-parent-note {
        font-size: 0.85rem;
        color: #71717a;
        background: rgba(255, 255, 255, 0.05);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 20px 0;
      }
      
      /* Minor State (13-17) */
      .earn-minor {
        margin-bottom: 20px;
      }
      
      .minor-notice {
        display: flex;
        gap: 14px;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
      }
      
      .minor-icon {
        font-size: 1.5rem;
      }
      
      .minor-text strong {
        display: block;
        color: #fbbf24;
        margin-bottom: 4px;
      }
      
      .minor-text p {
        color: #a1a1aa;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.5;
      }
      
      .held-earnings-display {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(251, 191, 36, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
      }
      
      .held-label {
        color: #a1a1aa;
        font-size: 0.9rem;
      }
      
      .held-amount {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fbbf24;
      }
    `;
    document.head.appendChild(styles);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // EVENT BINDING
  // ═══════════════════════════════════════════════════════════════════
  
  bindEvents() {
    // Close button
    document.getElementById('earn-close')?.addEventListener('click', () => this.close());
    
    // Click outside
    this.overlay?.addEventListener('click', (e) => {
      if (e.target === this.overlay) this.close();
    });
    
    // Copy link
    document.getElementById('copy-link-btn')?.addEventListener('click', () => this.copyLink());
    
    // Share buttons
    document.querySelectorAll('.share-btn[data-platform]').forEach(btn => {
      btn.addEventListener('click', () => this.shareTo(btn.dataset.platform));
    });
    
    // Native share
    document.getElementById('native-share-btn')?.addEventListener('click', () => this.nativeShare());
    
    // Sign in button
    document.getElementById('earn-signin-btn')?.addEventListener('click', () => {
      this.close();
      // Trigger auth modal if available
      if (window.AuthModal) {
        window.AuthModal.show();
      } else {
        window.location.href = '/learn.html#auth';
      }
    });
    
    // Keyboard escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) this.close();
    });
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // OPEN / CLOSE
  // ═══════════════════════════════════════════════════════════════════
  
  open() {
    if (!this.overlay) this.init();
    
    this.isOpen = true;
    this.overlay.classList.add('open');
    this.loadUserData();
    console.log('[EarnToLearn] Opened');
  },
  
  close() {
    this.isOpen = false;
    this.overlay?.classList.remove('open');
    console.log('[EarnToLearn] Closed');
  },
  
  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // DATA LOADING
  // ═══════════════════════════════════════════════════════════════════
  
  async loadUserData() {
    if (!this.supabase) {
      this.showNotLoggedIn();
      return;
    }
    
    try {
      const { data: { session } } = await this.supabase.auth.getSession();
      const { data: { user } } = await this.supabase.auth.getUser();
      
      if (!user) {
        this.showNotLoggedIn();
        return;
      }
      
      // COMPLIANCE: Check eligibility based on age
      await this.checkEligibility(session?.access_token);
      
      // Get user data with referral info
      const { data: userData, error } = await this.supabase
        .from('users')
        .select(`
          id,
          display_name,
          email,
          referral_code,
          commission_tier,
          commission_rate,
          total_referrals,
          active_referrals,
          pending_earnings,
          available_earnings,
          lifetime_earnings,
          total_lessons_completed,
          unique_lessons_completed,
          age,
          birthday,
          birth_year,
          parent_account_id
        `)
        .eq('id', user.id)
        .single();
      
      if (error || !userData) {
        console.error('[EarnToLearn] Failed to load user data:', error);
        this.showNotLoggedIn();
        return;
      }
      
      this.userData = userData;
      
      // Show appropriate UI based on eligibility
      if (this.eligibility && !this.eligibility.canSeeReferralLink) {
        // Under 13 without parental consent
        this.showUnder13();
        return;
      }
      
      this.showLoggedIn();
      this.updateUI();
      
      // Show minor notice if 13-17
      if (this.eligibility?.isMinor && this.eligibility?.canSeeReferralLink) {
        this.showMinorNotice();
      }
      
      // Load click stats
      await this.loadClickStats();
      
    } catch (e) {
      console.error('[EarnToLearn] Error loading data:', e);
      this.showNotLoggedIn();
    }
  },
  
  async checkEligibility(token) {
    if (!token) return;
    
    try {
      const response = await fetch('/api/referral/eligibility', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        this.eligibility = data.eligibility;
        console.log('[EarnToLearn] Eligibility:', this.eligibility);
      }
    } catch (e) {
      console.log('[EarnToLearn] Could not check eligibility, assuming adult');
      // Default to adult behavior if API fails
      this.eligibility = {
        canSeeReferralLink: true,
        canShare: true,
        canAccumulateEarnings: true,
        canRequestPayout: true,
        earningsDestination: 'self',
        isMinor: false
      };
    }
  },
  
  async loadClickStats() {
    if (!this.supabase || !this.userData?.id) return;
    
    try {
      const { count: clicks } = await this.supabase
        .from('referral_clicks')
        .select('*', { count: 'exact', head: true })
        .eq('referrer_id', this.userData.id);
      
      document.getElementById('total-clicks').textContent = clicks || 0;
    } catch (e) {
      console.log('[EarnToLearn] Could not load click stats');
    }
  },
  
  showNotLoggedIn() {
    document.getElementById('earn-summary').style.display = 'none';
    document.getElementById('earn-tier').style.display = 'none';
    document.querySelector('.earn-link-section').style.display = 'none';
    document.getElementById('quick-stats').style.display = 'none';
    document.querySelector('.earn-share-section').style.display = 'none';
    document.querySelector('.earn-footer').style.display = 'none';
    document.getElementById('earn-not-logged-in').style.display = 'block';
    document.getElementById('earn-under-13').style.display = 'none';
    document.getElementById('earn-minor').style.display = 'none';
  },
  
  showLoggedIn() {
    document.getElementById('earn-summary').style.display = 'block';
    document.getElementById('earn-tier').style.display = 'block';
    document.querySelector('.earn-link-section').style.display = 'block';
    document.getElementById('quick-stats').style.display = 'flex';
    document.querySelector('.earn-share-section').style.display = 'block';
    document.querySelector('.earn-footer').style.display = 'flex';
    document.getElementById('earn-not-logged-in').style.display = 'none';
    document.getElementById('earn-under-13').style.display = 'none';
  },
  
  showUnder13() {
    // COPPA: Hide everything except the under-13 message
    document.getElementById('earn-summary').style.display = 'none';
    document.getElementById('earn-tier').style.display = 'none';
    document.querySelector('.earn-link-section').style.display = 'none';
    document.getElementById('quick-stats').style.display = 'none';
    document.querySelector('.earn-share-section').style.display = 'none';
    document.querySelector('.earn-footer').style.display = 'none';
    document.getElementById('earn-not-logged-in').style.display = 'none';
    document.getElementById('earn-minor').style.display = 'none';
    document.getElementById('earn-under-13').style.display = 'block';
    console.log('[EarnToLearn] Showing Under 13 state (COPPA compliance)');
  },
  
  showMinorNotice() {
    // Show the minor notice (13-17) but still show full functionality
    const minorNotice = document.getElementById('earn-minor');
    minorNotice.style.display = 'block';
    
    // Update held earnings amount if available
    if (this.eligibility?.heldEarnings) {
      document.getElementById('held-earnings-amount').textContent = 
        this.formatMoney(this.eligibility.heldEarnings);
    }
    
    // Hide payout link for minors
    const payoutLink = document.getElementById('request-payout-link');
    if (payoutLink) {
      payoutLink.style.display = 'none';
      payoutLink.title = 'Payouts available when you turn 18';
    }
    
    console.log('[EarnToLearn] Showing Minor notice (13-17)');
  },
  
  updateUI() {
    if (!this.userData) return;
    
    const data = this.userData;
    
    // Earnings
    document.getElementById('available-earnings').textContent = this.formatMoney(data.available_earnings || 0);
    document.getElementById('pending-earnings').textContent = this.formatMoney(data.pending_earnings || 0);
    document.getElementById('lifetime-earnings').textContent = this.formatMoney(data.lifetime_earnings || 0);
    
    // Show payout link if available >= $50 AND user can request payout (not a minor)
    const payoutLink = document.getElementById('request-payout-link');
    const canPayout = this.eligibility?.canRequestPayout !== false;
    if ((data.available_earnings || 0) >= 50 && canPayout) {
      payoutLink.style.display = 'inline';
      payoutLink.href = '/earnings.html#payout';
    } else if (!canPayout) {
      payoutLink.style.display = 'none';
    }
    
    // Tier
    const tier = data.commission_tier || 'new_learner';
    const tierInfo = this.TIERS[tier] || this.TIERS.new_learner;
    
    document.getElementById('tier-emoji').textContent = tierInfo.emoji;
    document.getElementById('tier-name').textContent = tierInfo.name;
    document.getElementById('tier-rate').textContent = `${Math.round((data.commission_rate || 0.1) * 100)}% commission`;
    
    // Tier progress
    const lessonsCompleted = data.total_lessons_completed || 0;
    const nextTier = this.getNextTier(tier, lessonsCompleted);
    
    if (nextTier) {
      const progress = this.calculateTierProgress(lessonsCompleted, tierInfo.lessons, nextTier.lessons);
      document.getElementById('tier-bar-fill').style.width = `${progress}%`;
      document.getElementById('tier-next').textContent = 
        `${nextTier.lessons - lessonsCompleted} more lessons for ${nextTier.name} (${nextTier.rate}%)`;
    } else {
      document.getElementById('tier-bar-fill').style.width = '100%';
      document.getElementById('tier-next').textContent = '🏆 Maximum tier achieved!';
    }
    
    // Referral link
    const code = data.referral_code || 'loading';
    const linkInput = document.getElementById('referral-link');
    linkInput.value = `curiouskelly.com/?ref=${code}`;
    
    // Stats
    document.getElementById('total-signups').textContent = data.total_referrals || 0;
    document.getElementById('total-active').textContent = data.active_referrals || 0;
  },
  
  getNextTier(currentTier, lessonsCompleted) {
    const tiers = Object.entries(this.TIERS).sort((a, b) => a[1].lessons - b[1].lessons);
    const currentIndex = tiers.findIndex(([key]) => key === currentTier);
    
    for (let i = currentIndex + 1; i < tiers.length; i++) {
      if (tiers[i][1].lessons > lessonsCompleted) {
        return tiers[i][1];
      }
    }
    return null;
  },
  
  calculateTierProgress(completed, currentTierLessons, nextTierLessons) {
    const range = nextTierLessons - currentTierLessons;
    const progress = completed - currentTierLessons;
    return Math.min(100, Math.max(0, (progress / range) * 100));
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // SHARE ACTIONS
  // ═══════════════════════════════════════════════════════════════════
  
  getReferralUrl() {
    const code = this.userData?.referral_code || '';
    return `https://curiouskelly.com/?ref=${code}`;
  },
  
  getShareText() {
    return "I'm learning something new every day with Curious Kelly! 🌟 Join me — we both earn when you sign up through my link.";
  },
  
  copyLink() {
    const url = this.getReferralUrl();
    navigator.clipboard.writeText(url).then(() => {
      const btn = document.getElementById('copy-link-btn');
      btn.classList.add('copied');
      btn.querySelector('.copy-text').textContent = 'Copied!';
      
      this.showToast('📋 Link copied!');
      
      setTimeout(() => {
        btn.classList.remove('copied');
        btn.querySelector('.copy-text').textContent = 'Copy';
      }, 2000);
    });
  },
  
  shareTo(platform) {
    const url = encodeURIComponent(this.getReferralUrl());
    const text = encodeURIComponent(this.getShareText());
    
    const urls = {
      twitter: `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${url}`,
      whatsapp: `https://wa.me/?text=${text}%20${url}`,
      linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${url}`,
      email: `mailto:?subject=${encodeURIComponent('Join me on Curious Kelly!')}&body=${text}%0A%0A${url}%0A%0A(I may earn a commission if you subscribe.)`
    };
    
    const shareUrl = urls[platform];
    if (shareUrl) {
      window.open(shareUrl, '_blank', 'width=600,height=500');
    }
  },
  
  async nativeShare() {
    if (!navigator.share) {
      this.copyLink();
      return;
    }
    
    try {
      await navigator.share({
        title: 'Join me on Curious Kelly',
        text: this.getShareText(),
        url: this.getReferralUrl()
      });
    } catch (e) {
      if (e.name !== 'AbortError') {
        this.copyLink();
      }
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // UTILITIES
  // ═══════════════════════════════════════════════════════════════════
  
  formatMoney(amount) {
    return '$' + parseFloat(amount || 0).toFixed(2);
  },
  
  showToast(message) {
    document.querySelector('.earn-toast')?.remove();
    
    const toast = document.createElement('div');
    toast.className = 'earn-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.remove(), 2500);
  }
};

// Auto-initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  EarnToLearn.init();
});

// Export
window.EarnToLearn = EarnToLearn;

console.log('[EarnToLearn] ✅ Loaded - Share & Earn System');

