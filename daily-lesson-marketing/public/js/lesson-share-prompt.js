/**
 * Lesson Share Prompt
 * 
 * Shows after lesson completion to encourage sharing with referral link.
 * "Someone in your life would love this too."
 * 
 * Features:
 * - Pre-written share messages
 * - Referral link automatically included
 * - Track shares for analytics
 * - Beautiful celebration UI
 */

const LessonSharePrompt = {
  isOpen: false,
  overlay: null,
  stats: null,
  lessonData: null,
  supabase: null,
  
  // Share message templates
  MESSAGES: {
    general: "I just learned about \"{topic}\" with Curious Kelly! 🌟 Join me for free daily lessons.",
    short: "Today I learned: {topic}. Mind = blown. 🤯",
    invitation: "Someone I know needs to learn this! \"{topic}\" — join me on Curious Kelly.",
    streak: "Day {streak} of learning with Curious Kelly! 🔥 Today: {topic}",
    celebrate: "Just finished Day {day}/365 with Curious Kelly! 🎓 Topic: {topic}"
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init() {
    this.initSupabase();
    console.log('[LessonSharePrompt] Initialized');
  },
  
  initSupabase() {
    if (window.supabase) {
      this.supabase = window.supabase;
    } else if (window.supabaseClient) {
      this.supabase = window.supabaseClient;
    }
  },
  
  createOverlay() {
    if (this.overlay) {
      this.overlay.remove();
    }
    
    const topic = this.lessonData?.topic || 'today\'s lesson';
    const dayNumber = this.stats?.lessonDay || this.lessonData?.dayNumber || 1;
    const streak = this.stats?.streak || 1;
    
    // Pick a random message variant
    const messageVariants = [
      this.MESSAGES.general.replace('{topic}', topic),
      this.MESSAGES.short.replace('{topic}', topic),
      this.MESSAGES.celebrate.replace('{day}', dayNumber).replace('{topic}', topic)
    ];
    const shareMessage = messageVariants[Math.floor(Math.random() * messageVariants.length)];
    
    // Get referral code
    const referralCode = window.getReferralTrackingData?.()?.referralCode || 
                         this.userData?.referral_code || '';
    const shareUrl = referralCode ? 
      `https://curiouskelly.com/?ref=${referralCode}` : 
      'https://curiouskelly.com';
    
    this.overlay = document.createElement('div');
    this.overlay.id = 'share-prompt-overlay';
    this.overlay.className = 'share-prompt-overlay';
    this.overlay.innerHTML = `
      <div class="share-prompt-container">
        
        <!-- Celebration Header -->
        <div class="share-prompt-header">
          <div class="celebration-icon">🎉</div>
          <h2>Lesson Complete!</h2>
          <p class="lesson-title">${this.escapeHtml(topic)}</p>
        </div>
        
        <!-- Share CTA -->
        <div class="share-prompt-cta">
          <p class="share-prompt-text">
            Someone in your life would love this too.
          </p>
          
          <!-- Pre-written message -->
          <div class="share-message-box">
            <textarea class="share-message-input" id="share-message-text" readonly>${this.escapeHtml(shareMessage)}</textarea>
            <button class="message-copy-btn" id="copy-message-btn">📋</button>
          </div>
          
          <p class="share-link-display">
            Your link: <strong>${shareUrl}</strong>
            ${referralCode ? '<span class="earn-badge">You earn 💰</span>' : ''}
          </p>
        </div>
        
        <!-- Share Buttons -->
        <div class="share-buttons-row">
          <button class="share-prompt-btn twitter" data-platform="twitter">
            <span>𝕏</span>
          </button>
          <button class="share-prompt-btn facebook" data-platform="facebook">
            <span>f</span>
          </button>
          <button class="share-prompt-btn whatsapp" data-platform="whatsapp">
            <span>💬</span>
          </button>
          <button class="share-prompt-btn email" data-platform="email">
            <span>✉️</span>
          </button>
          <button class="share-prompt-btn copy" id="copy-link-btn">
            <span>🔗</span>
          </button>
        </div>
        
        <!-- Commission Note -->
        ${referralCode ? `
        <div class="commission-note">
          <span class="commission-icon">💰</span>
          <span>You earn commission when friends subscribe!</span>
          <button class="learn-more-btn" id="learn-more-earn">Learn more</button>
        </div>
        ` : ''}
        
        <!-- Actions -->
        <div class="share-prompt-actions">
          <button class="action-btn secondary" id="skip-share-btn">
            Maybe Later
          </button>
          <button class="action-btn primary" id="continue-btn">
            Continue Learning →
          </button>
        </div>
        
      </div>
    `;
    
    this.addStyles();
    document.body.appendChild(this.overlay);
    this.bindEvents();
  },
  
  addStyles() {
    if (document.getElementById('share-prompt-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'share-prompt-styles';
    styles.textContent = `
      .share-prompt-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.95);
        backdrop-filter: blur(20px);
        z-index: 9800;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.4s ease;
      }
      
      .share-prompt-overlay.open {
        opacity: 1;
        pointer-events: auto;
      }
      
      .share-prompt-container {
        width: 92%;
        max-width: 420px;
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 28px;
        padding: 32px 24px;
        text-align: center;
        transform: translateY(30px) scale(0.95);
        opacity: 0;
        transition: all 0.4s cubic-bezier(0.32, 0.72, 0, 1);
      }
      
      .share-prompt-overlay.open .share-prompt-container {
        transform: translateY(0) scale(1);
        opacity: 1;
      }
      
      /* Header */
      .share-prompt-header {
        margin-bottom: 24px;
      }
      
      .celebration-icon {
        font-size: 3.5rem;
        margin-bottom: 12px;
        animation: bounce 0.6s ease infinite alternate;
      }
      
      @keyframes bounce {
        from { transform: translateY(0); }
        to { transform: translateY(-8px); }
      }
      
      .share-prompt-header h2 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
        margin: 0 0 8px;
      }
      
      .lesson-title {
        font-size: 1rem;
        color: #3b82f6;
        margin: 0;
        font-weight: 500;
      }
      
      /* CTA Section */
      .share-prompt-cta {
        margin-bottom: 20px;
      }
      
      .share-prompt-text {
        font-size: 1.1rem;
        color: #d1d5db;
        margin: 0 0 16px;
        line-height: 1.5;
      }
      
      .share-message-box {
        position: relative;
        margin-bottom: 12px;
      }
      
      .share-message-input {
        width: 100%;
        min-height: 80px;
        padding: 14px 50px 14px 14px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        color: #e5e5e5;
        font-size: 0.95rem;
        resize: none;
        line-height: 1.5;
      }
      
      .share-message-input:focus {
        outline: none;
        border-color: #3b82f6;
      }
      
      .message-copy-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 36px;
        height: 36px;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .message-copy-btn:hover {
        background: rgba(59, 130, 246, 0.4);
      }
      
      .share-link-display {
        font-size: 0.85rem;
        color: #71717a;
        margin: 0;
      }
      
      .share-link-display strong {
        color: #a1a1aa;
      }
      
      .earn-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2));
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #22c55e;
        margin-left: 8px;
      }
      
      /* Share Buttons */
      .share-buttons-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
      }
      
      .share-prompt-btn {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #fff;
        font-size: 1.3rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      
      .share-prompt-btn:hover {
        transform: scale(1.1);
      }
      
      .share-prompt-btn.twitter:hover { background: rgba(29, 161, 242, 0.3); border-color: #1da1f2; }
      .share-prompt-btn.facebook:hover { background: rgba(24, 119, 242, 0.3); border-color: #1877f2; }
      .share-prompt-btn.whatsapp:hover { background: rgba(37, 211, 102, 0.3); border-color: #25d366; }
      .share-prompt-btn.email:hover { background: rgba(234, 67, 53, 0.3); border-color: #ea4335; }
      .share-prompt-btn.copy:hover { background: rgba(59, 130, 246, 0.3); border-color: #3b82f6; }
      
      /* Commission Note */
      .commission-note {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #a1a1aa;
      }
      
      .commission-icon {
        font-size: 1.2rem;
      }
      
      .learn-more-btn {
        background: none;
        border: none;
        color: #3b82f6;
        font-size: 0.85rem;
        cursor: pointer;
        text-decoration: underline;
        padding: 0;
      }
      
      /* Actions */
      .share-prompt-actions {
        display: flex;
        gap: 12px;
      }
      
      .action-btn {
        flex: 1;
        padding: 14px 20px;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .action-btn.secondary {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #a1a1aa;
      }
      
      .action-btn.secondary:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #fff;
      }
      
      .action-btn.primary {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        border: none;
        color: #fff;
      }
      
      .action-btn.primary:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
      }
      
      /* Toast */
      .share-prompt-toast {
        position: fixed;
        bottom: 120px;
        left: 50%;
        transform: translateX(-50%);
        background: #22c55e;
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 600;
        z-index: 10001;
        animation: toastSlide 0.3s ease;
      }
      
      @keyframes toastSlide {
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
    // Share buttons
    this.overlay.querySelectorAll('.share-prompt-btn[data-platform]').forEach(btn => {
      btn.addEventListener('click', () => {
        this.shareTo(btn.dataset.platform);
        this.trackShare(btn.dataset.platform);
      });
    });
    
    // Copy buttons
    document.getElementById('copy-link-btn')?.addEventListener('click', () => {
      this.copyLink();
      this.trackShare('copy_link');
    });
    
    document.getElementById('copy-message-btn')?.addEventListener('click', () => {
      this.copyMessage();
      this.trackShare('copy_message');
    });
    
    // Actions
    document.getElementById('skip-share-btn')?.addEventListener('click', () => {
      this.close();
    });
    
    document.getElementById('continue-btn')?.addEventListener('click', () => {
      this.close();
      this.goToNextLesson();
    });
    
    // Learn more
    document.getElementById('learn-more-earn')?.addEventListener('click', () => {
      this.close();
      if (window.EarnToLearn) {
        setTimeout(() => EarnToLearn.open(), 300);
      }
    });
    
    // Close on escape
    this.escapeHandler = (e) => {
      if (e.key === 'Escape' && this.isOpen) {
        this.close();
      }
    };
    document.addEventListener('keydown', this.escapeHandler);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // OPEN / CLOSE
  // ═══════════════════════════════════════════════════════════════════
  
  show(stats = {}, lessonData = {}) {
    this.stats = stats;
    this.lessonData = lessonData;
    
    // Get lesson data from global state if not provided
    if (!lessonData.topic && window.state?.lesson) {
      this.lessonData = {
        ...lessonData,
        topic: window.state.lesson.topic || window.state.lesson.title,
        dayNumber: window.state.dayNumber
      };
    }
    
    // Get user data for referral code
    this.loadUserData().then(() => {
      this.createOverlay();
      
      requestAnimationFrame(() => {
        this.isOpen = true;
        this.overlay.classList.add('open');
      });
      
      console.log('[LessonSharePrompt] Showing for:', this.lessonData.topic);
    });
  },
  
  async loadUserData() {
    if (!this.supabase) return;
    
    try {
      const { data: { user } } = await this.supabase.auth.getUser();
      if (!user) return;
      
      const { data } = await this.supabase
        .from('users')
        .select('referral_code, commission_rate, commission_tier')
        .eq('id', user.id)
        .single();
      
      if (data) {
        this.userData = data;
      }
    } catch (e) {
      console.log('[LessonSharePrompt] Could not load user data');
    }
  },
  
  close() {
    this.isOpen = false;
    if (this.overlay) {
      this.overlay.classList.remove('open');
      setTimeout(() => {
        this.overlay?.remove();
        this.overlay = null;
      }, 400);
    }
    
    if (this.escapeHandler) {
      document.removeEventListener('keydown', this.escapeHandler);
    }
    
    console.log('[LessonSharePrompt] Closed');
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // SHARE ACTIONS
  // ═══════════════════════════════════════════════════════════════════
  
  getShareUrl() {
    const code = this.userData?.referral_code || '';
    const day = this.lessonData?.dayNumber || '';
    
    let url = 'https://curiouskelly.com';
    const params = new URLSearchParams();
    
    if (code) params.set('ref', code);
    if (day) params.set('day', day);
    
    const queryString = params.toString();
    return queryString ? `${url}?${queryString}` : url;
  },
  
  getShareMessage() {
    return document.getElementById('share-message-text')?.value || '';
  },
  
  shareTo(platform) {
    const url = encodeURIComponent(this.getShareUrl());
    const text = encodeURIComponent(this.getShareMessage());
    const topic = encodeURIComponent(this.lessonData?.topic || 'Today\'s Lesson');
    
    const urls = {
      twitter: `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${url}&quote=${text}`,
      whatsapp: `https://wa.me/?text=${text}%20${url}`,
      email: `mailto:?subject=${encodeURIComponent(`I learned about ${this.lessonData?.topic || 'something amazing'} today!`)}&body=${text}%0A%0A${url}%0A%0A(I may earn a commission if you subscribe.)`
    };
    
    const shareUrl = urls[platform];
    if (shareUrl) {
      window.open(shareUrl, '_blank', 'width=600,height=500');
    }
  },
  
  copyLink() {
    const url = this.getShareUrl();
    navigator.clipboard.writeText(url).then(() => {
      this.showToast('🔗 Link copied!');
    });
  },
  
  copyMessage() {
    const message = this.getShareMessage() + '\n\n' + this.getShareUrl();
    navigator.clipboard.writeText(message).then(() => {
      this.showToast('📋 Message copied!');
    });
  },
  
  goToNextLesson() {
    const currentDay = this.lessonData?.dayNumber || window.state?.dayNumber || 1;
    const nextDay = currentDay < 365 ? currentDay + 1 : 1;
    
    if (window.goToLesson) {
      window.goToLesson(nextDay);
    } else {
      window.location.href = `/learn.html?day=${nextDay}`;
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // ANALYTICS
  // ═══════════════════════════════════════════════════════════════════
  
  async trackShare(platform) {
    console.log(`[LessonSharePrompt] Share tracked: ${platform}`);
    
    // Track in Supabase analytics if available
    if (this.supabase) {
      try {
        const { data: { user } } = await this.supabase.auth.getUser();
        
        await this.supabase.from('analytics_events').insert({
          user_id: user?.id || null,
          event_type: 'lesson_share',
          event_data: {
            platform,
            day_number: this.lessonData?.dayNumber,
            topic: this.lessonData?.topic,
            has_referral_code: !!this.userData?.referral_code
          }
        });
      } catch (e) {
        // Silent fail - analytics shouldn't break the experience
      }
    }
    
    // Fire Google Analytics event if available
    if (typeof gtag !== 'undefined') {
      gtag('event', 'share', {
        method: platform,
        content_type: 'lesson',
        item_id: `day_${this.lessonData?.dayNumber}`
      });
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // UTILITIES
  // ═══════════════════════════════════════════════════════════════════
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },
  
  showToast(message) {
    document.querySelector('.share-prompt-toast')?.remove();
    
    const toast = document.createElement('div');
    toast.className = 'share-prompt-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => toast.remove(), 2500);
  }
};

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
  LessonSharePrompt.init();
});

// Export
window.LessonSharePrompt = LessonSharePrompt;

console.log('[LessonSharePrompt] ✅ Loaded');

