/**
 * Share Hub v1.0
 * Share/Perspectives overlay with 4 sections:
 * - Global Perspectives (stats)
 * - My Learning Groups
 * - Invite Someone
 * - Ambassador Program
 * 
 * Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
 */

const ShareHub = {
  isOpen: false,
  overlay: null,
  currentUserId: null,
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init() {
    this.createOverlay();
    this.bindEvents();
    this.loadUserGroups();
    console.log('[ShareHub] Initialized');
  },
  
  createOverlay() {
    if (this.overlay) return;
    
    this.overlay = document.createElement('div');
    this.overlay.id = 'share-overlay';
    this.overlay.className = 'share-overlay';
    this.overlay.innerHTML = `
      <div class="share-container">
        
        <!-- Header -->
        <div class="share-header">
          <h2>Share & Connect</h2>
          <button class="share-close" id="share-close" aria-label="Close">✕</button>
        </div>
        
        <!-- Global Perspectives -->
        <section class="share-section" id="section-perspectives">
          <div class="section-header">
            <span class="section-icon">🌍</span>
            <h3>Global Perspectives</h3>
          </div>
          <div class="perspectives-preview">
            <div class="perspective-stat">
              <span class="stat-number" id="global-learners">-</span>
              <span class="stat-label">learners today</span>
            </div>
            <div class="perspective-stat">
              <span class="stat-number" id="global-countries">-</span>
              <span class="stat-label">countries</span>
            </div>
          </div>
          <button class="section-action" id="btn-view-perspectives">
            View Insights →
          </button>
        </section>
        
        <!-- Learning Groups -->
        <section class="share-section" id="section-groups">
          <div class="section-header">
            <span class="section-icon">👥</span>
            <h3>My Learning Groups</h3>
          </div>
          <div class="groups-grid" id="groups-grid">
            <div class="group-card add-group" id="add-group-card">
              <div class="group-emoji">+</div>
              <div class="group-name">Create Group</div>
            </div>
          </div>
          <p class="section-hint">Learn together with friends & family</p>
        </section>
        
        <!-- Invite -->
        <section class="share-section" id="section-invite">
          <div class="section-header">
            <span class="section-icon">💌</span>
            <h3>Invite Someone</h3>
          </div>
          <p class="section-desc">Share the gift of daily learning</p>
          <div class="invite-actions">
            <button class="invite-btn" id="btn-copy-link">
              🔗 Copy Link
            </button>
            <button class="invite-btn" id="btn-share-native">
              📤 Share...
            </button>
          </div>
          <div class="share-platforms" id="share-platforms">
            <button class="platform-btn" data-platform="twitter" title="Twitter/X">𝕏</button>
            <button class="platform-btn" data-platform="facebook" title="Facebook">f</button>
            <button class="platform-btn" data-platform="whatsapp" title="WhatsApp">💬</button>
            <button class="platform-btn" data-platform="linkedin" title="LinkedIn">in</button>
            <button class="platform-btn" data-platform="email" title="Email">✉️</button>
          </div>
        </section>
        
        <!-- Ambassador -->
        <section class="share-section highlight" id="section-ambassador">
          <div class="section-header">
            <span class="section-icon">💼</span>
            <h3>Become an Ambassador</h3>
          </div>
          <p class="section-desc">Earn by spreading curiosity</p>
          <button class="section-action ambassador" id="btn-ambassador">
            Learn More →
          </button>
        </section>
        
      </div>
    `;
    
    // Add styles
    this.addStyles();
    
    document.body.appendChild(this.overlay);
  },
  
  addStyles() {
    if (document.getElementById('share-hub-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'share-hub-styles';
    styles.textContent = `
      .share-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        z-index: 9000;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
      }
      
      .share-overlay.open {
        opacity: 1;
        pointer-events: auto;
      }
      
      .share-container {
        width: 100%;
        max-width: 500px;
        max-height: 85vh;
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 24px 24px 0 0;
        padding: 24px;
        overflow-y: auto;
        transform: translateY(100%);
        transition: transform 0.35s cubic-bezier(0.32, 0.72, 0, 1);
      }
      
      .share-overlay.open .share-container {
        transform: translateY(0);
      }
      
      @media (min-width: 768px) {
        .share-overlay {
          align-items: center;
        }
        
        .share-container {
          border-radius: 24px;
          max-height: 90vh;
          transform: translateY(20px) scale(0.95);
          opacity: 0;
        }
        
        .share-overlay.open .share-container {
          transform: translateY(0) scale(1);
          opacity: 1;
        }
      }
      
      .share-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .share-header h2 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin: 0;
      }
      
      .share-close {
        background: rgba(255, 255, 255, 0.1);
        border: none;
        color: #fff;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        font-size: 1.2rem;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .share-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }
      
      .share-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
      }
      
      .share-section.highlight {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
        border-color: rgba(59, 130, 246, 0.3);
      }
      
      .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }
      
      .section-icon {
        font-size: 1.5rem;
      }
      
      .section-header h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        margin: 0;
      }
      
      .section-desc, .section-hint {
        color: #a1a1aa;
        font-size: 0.9rem;
        margin: 0 0 16px;
      }
      
      .section-hint {
        font-size: 0.8rem;
        margin-top: 12px;
        margin-bottom: 0;
        font-style: italic;
      }
      
      /* Perspectives Stats */
      .perspectives-preview {
        display: flex;
        gap: 24px;
        margin-bottom: 16px;
      }
      
      .perspective-stat {
        display: flex;
        flex-direction: column;
      }
      
      .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
      }
      
      .stat-label {
        font-size: 0.8rem;
        color: #71717a;
      }
      
      /* Groups Grid */
      .groups-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        gap: 12px;
        margin-bottom: 8px;
      }
      
      .group-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .group-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
      }
      
      .group-card.add-group {
        border-style: dashed;
      }
      
      .group-emoji {
        font-size: 1.8rem;
        margin-bottom: 8px;
      }
      
      .group-name {
        font-size: 0.85rem;
        color: #d1d5db;
        font-weight: 500;
      }
      
      .group-members {
        font-size: 0.75rem;
        color: #71717a;
        margin-top: 4px;
      }
      
      /* Invite Actions */
      .invite-actions {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
      }
      
      .invite-btn {
        flex: 1;
        padding: 12px 16px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #fff;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .invite-btn:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: #3b82f6;
      }
      
      .share-platforms {
        display: flex;
        justify-content: center;
        gap: 12px;
      }
      
      .platform-btn {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #fff;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      
      .platform-btn:hover {
        background: #3b82f6;
        border-color: #3b82f6;
        transform: scale(1.1);
      }
      
      /* Section Action Button */
      .section-action {
        width: 100%;
        padding: 12px 20px;
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: #fff;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .section-action:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #3b82f6;
      }
      
      .section-action.ambassador {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border: none;
      }
      
      .section-action.ambassador:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
      }
      
      /* Toast */
      .share-toast {
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: #3b82f6;
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 500;
        z-index: 10001;
        animation: toastIn 0.3s ease-out;
      }
      
      @keyframes toastIn {
        from { opacity: 0; transform: translate(-50%, 20px); }
        to { opacity: 1; transform: translate(-50%, 0); }
      }
    `;
    document.head.appendChild(styles);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // EVENT BINDING
  // ═══════════════════════════════════════════════════════════════════
  
  bindEvents() {
    // Close button
    document.getElementById('share-close')?.addEventListener('click', () => this.close());
    
    // Click outside to close
    this.overlay?.addEventListener('click', (e) => {
      if (e.target === this.overlay) this.close();
    });
    
    // Copy link
    document.getElementById('btn-copy-link')?.addEventListener('click', () => this.copyLink());
    
    // Native share
    document.getElementById('btn-share-native')?.addEventListener('click', () => this.nativeShare());
    
    // View perspectives
    document.getElementById('btn-view-perspectives')?.addEventListener('click', () => this.openPerspectives());
    
    // Create group
    document.getElementById('add-group-card')?.addEventListener('click', () => this.createGroup());
    
    // Ambassador
    document.getElementById('btn-ambassador')?.addEventListener('click', () => this.openAmbassador());
    
    // Platform buttons
    document.querySelectorAll('.platform-btn').forEach(btn => {
      btn.addEventListener('click', () => this.shareToPlattform(btn.dataset.platform));
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
    this.loadGlobalStats();
    console.log('[ShareHub] Opened');
  },
  
  close() {
    this.isOpen = false;
    this.overlay?.classList.remove('open');
    console.log('[ShareHub] Closed');
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
  
  async loadGlobalStats() {
    // Simulated stats for now (replace with Supabase query)
    const baseViewers = 847000 + Math.floor(Math.random() * 100000);
    const countries = 142 + Math.floor(Math.random() * 10);
    
    const viewersEl = document.getElementById('global-learners');
    const countriesEl = document.getElementById('global-countries');
    
    if (viewersEl) {
      viewersEl.textContent = this.formatNumber(baseViewers);
    }
    if (countriesEl) {
      countriesEl.textContent = countries;
    }
    
    // TODO: Fetch from Supabase
    // const { data } = await supabase
    //   .from('daily_stats')
    //   .select('learners_count, countries_count')
    //   .eq('lesson_day', getCurrentDay())
    //   .single();
  },
  
  async loadUserGroups() {
    // TODO: Fetch from Supabase
    // const { data: groups } = await supabase
    //   .from('learning_groups')
    //   .select('*')
    //   .contains('member_ids', [this.currentUserId]);
    
    // For now, show empty state
    const grid = document.getElementById('groups-grid');
    if (grid) {
      // Keep just the "add" card
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // SHARE ACTIONS
  // ═══════════════════════════════════════════════════════════════════
  
  getCurrentDay() {
    return window.state?.dayNumber || new Date().getDay() || 1;
  },
  
  getCurrentLesson() {
    return window.state?.lesson || { topic: 'Today\'s Lesson', hook: '' };
  },
  
  getShareUrl() {
    const day = this.getCurrentDay();
    return `https://curiouskelly.com/learn.html?day=${day}`;
  },
  
  getShareText() {
    const lesson = this.getCurrentLesson();
    return `Today I learned about "${lesson.topic}" with Curious Kelly! 🌟`;
  },
  
  copyLink() {
    const url = this.getShareUrl();
    navigator.clipboard.writeText(url).then(() => {
      this.showToast('📋 Link copied!');
    }).catch(() => {
      this.showToast('Failed to copy');
    });
  },
  
  async nativeShare() {
    const lesson = this.getCurrentLesson();
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Day ${this.getCurrentDay()}: ${lesson.topic}`,
          text: this.getShareText(),
          url: this.getShareUrl()
        });
      } catch (e) {
        // User cancelled or error
        if (e.name !== 'AbortError') {
          this.copyLink();
        }
      }
    } else {
      this.copyLink();
    }
  },
  
  shareToPlattform(platform) {
    const url = encodeURIComponent(this.getShareUrl());
    const text = encodeURIComponent(this.getShareText());
    const lesson = this.getCurrentLesson();
    const day = this.getCurrentDay();
    
    const urls = {
      twitter: `https://twitter.com/intent/tweet?text=${text}&url=${url}&hashtags=CuriousKelly,DailyLesson`,
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${url}&quote=${text}`,
      whatsapp: `https://wa.me/?text=${text}%20${url}`,
      linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${url}`,
      email: `mailto:?subject=${encodeURIComponent(`Day ${day}: ${lesson.topic} - Curious Kelly`)}&body=${text}%0A%0A${url}`
    };
    
    const shareUrl = urls[platform];
    if (shareUrl) {
      window.open(shareUrl, '_blank', 'width=600,height=400');
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // ADDITIONAL FEATURES
  // ═══════════════════════════════════════════════════════════════════
  
  openPerspectives() {
    // TODO: Open perspectives/insights view
    this.showToast('🌍 Coming soon!');
  },
  
  createGroup() {
    // TODO: Open group creation flow
    this.showToast('👥 Group feature coming soon!');
  },
  
  openAmbassador() {
    window.open('https://curiouskelly.com/ambassador', '_blank');
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // UTILITIES
  // ═══════════════════════════════════════════════════════════════════
  
  formatNumber(num) {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return Math.round(num / 1000) + 'K';
    }
    return num.toString();
  },
  
  showToast(message) {
    // Remove existing toast
    document.querySelector('.share-toast')?.remove();
    
    const toast = document.createElement('div');
    toast.className = 'share-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.remove(), 2500);
  }
};

// Auto-initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  ShareHub.init();
});

// Export
window.ShareHub = ShareHub;

console.log('[ShareHub] ✅ Loaded');



