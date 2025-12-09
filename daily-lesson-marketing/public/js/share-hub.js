/**
 * Share Hub v2.0
 * Share/Perspectives overlay with 4 sections:
 * - Global Perspectives (stats) - REAL DATA from Supabase
 * - My Learning Groups - Fully wired
 * - Invite Someone - Working share links
 * - Ambassador Program - Links to affiliate program
 * 
 * Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
 * Trust & Safety: No fake metrics. Real data or honest disclosure.
 */

const ShareHub = {
  isOpen: false,
  overlay: null,
  currentUserId: null,
  supabase: null,
  userGroups: [],
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init() {
    this.initSupabase();
    this.createOverlay();
    this.bindEvents();
    this.loadUserGroups();
    console.log('[ShareHub] v2.0 Initialized');
  },
  
  initSupabase() {
    // Check if supabase client exists globally
    if (window.supabase) {
      this.supabase = window.supabase;
    } else if (window.supabaseClient) {
      this.supabase = window.supabaseClient;
    }
    // If no supabase, we'll show honest placeholder data
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
          <div class="perspectives-preview" id="perspectives-preview">
            <div class="perspective-stat">
              <span class="stat-number" id="global-learners">–</span>
              <span class="stat-label" id="global-learners-label">learners today</span>
            </div>
            <div class="perspective-stat">
              <span class="stat-number" id="global-countries">–</span>
              <span class="stat-label" id="global-countries-label">countries</span>
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
    const viewersEl = document.getElementById('global-learners');
    const countriesEl = document.getElementById('global-countries');
    const viewersLabel = document.getElementById('global-learners-label');
    const countriesLabel = document.getElementById('global-countries-label');
    
    // Try to get REAL data from Supabase
    if (this.supabase) {
      try {
        // Get today's actual lesson stats
        const today = new Date().toISOString().split('T')[0];
        
        // Query daily_lesson_stats for real counts
        const { data: stats, error } = await this.supabase
          .from('daily_lesson_stats')
          .select('learners_count, countries_count')
          .eq('stat_date', today)
          .single();
        
        if (stats && !error) {
          // REAL DATA - show without disclosure
          if (viewersEl) viewersEl.textContent = this.formatNumber(stats.learners_count);
          if (countriesEl) countriesEl.textContent = stats.countries_count;
          return;
        }
        
        // Fallback: count actual users who completed lessons today
        const { count: todayLearners } = await this.supabase
          .from('user_progress')
          .select('*', { count: 'exact', head: true })
          .gte('updated_at', today);
        
        if (todayLearners !== null) {
          if (viewersEl) viewersEl.textContent = this.formatNumber(todayLearners);
          if (viewersLabel) viewersLabel.textContent = 'active today';
          if (countriesEl) countriesEl.textContent = '–';
          if (countriesLabel) countriesLabel.textContent = 'tracking soon';
          return;
        }
      } catch (e) {
        console.log('[ShareHub] Stats query failed, showing launch state:', e);
      }
    }
    
    // NO REAL DATA - Show honest "launching" state
    // Per Trust & Safety: Never show fake metrics as real
    if (viewersEl) {
      viewersEl.textContent = '✨';
      viewersEl.classList.add('launching');
    }
    if (viewersLabel) {
      viewersLabel.textContent = 'Launching Soon';
    }
    if (countriesEl) {
      countriesEl.textContent = '🌍';
      countriesEl.classList.add('launching');
    }
    if (countriesLabel) {
      countriesLabel.textContent = 'Join us!';
    }
    
    // Add launching indicator to section
    const preview = document.getElementById('perspectives-preview');
    if (preview && !preview.querySelector('.launching-badge')) {
      const badge = document.createElement('div');
      badge.className = 'launching-badge';
      badge.innerHTML = '✨ <span>Building our global community</span>';
      preview.appendChild(badge);
    }
  },
  
  async loadUserGroups() {
    const grid = document.getElementById('groups-grid');
    if (!grid) return;
    
    // Get current user ID
    this.currentUserId = this.supabase ? 
      (await this.supabase.auth.getUser())?.data?.user?.id : null;
    
    if (!this.currentUserId || !this.supabase) {
      // Not logged in - show sign in prompt
      this.renderGroupsEmpty(grid, 'Sign in to create learning groups');
      return;
    }
    
    try {
      // Fetch user's groups from Supabase
      const { data: memberships, error } = await this.supabase
        .from('group_members')
        .select(`
          group_id,
          learning_groups (
            id,
            name,
            emoji,
            invite_code,
            created_by
          )
        `)
        .eq('user_id', this.currentUserId);
      
      if (error) throw error;
      
      if (memberships && memberships.length > 0) {
        this.userGroups = memberships.map(m => m.learning_groups).filter(Boolean);
        this.renderGroups(grid);
      } else {
        this.renderGroupsEmpty(grid, 'Learn together with friends & family');
      }
    } catch (e) {
      console.log('[ShareHub] Groups fetch failed:', e);
      this.renderGroupsEmpty(grid, 'Learn together with friends & family');
    }
  },
  
  renderGroups(grid) {
    // Clear existing cards except add button
    const addCard = grid.querySelector('.add-group');
    grid.innerHTML = '';
    
    // Render user's groups
    this.userGroups.forEach(group => {
      const card = document.createElement('div');
      card.className = 'group-card';
      card.dataset.groupId = group.id;
      card.innerHTML = `
        <div class="group-emoji">${group.emoji || '👥'}</div>
        <div class="group-name">${this.escapeHtml(group.name)}</div>
      `;
      card.addEventListener('click', () => this.openGroup(group));
      grid.appendChild(card);
    });
    
    // Add the "create" card back
    if (addCard) {
      grid.appendChild(addCard);
    } else {
      const newAddCard = document.createElement('div');
      newAddCard.className = 'group-card add-group';
      newAddCard.id = 'add-group-card';
      newAddCard.innerHTML = `
        <div class="group-emoji">+</div>
        <div class="group-name">Create Group</div>
      `;
      newAddCard.addEventListener('click', () => this.createGroup());
      grid.appendChild(newAddCard);
    }
  },
  
  renderGroupsEmpty(grid, message) {
    const hint = document.querySelector('#section-groups .section-hint');
    if (hint) hint.textContent = message;
  },
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
    // Navigate to Global Perspectives / Insights page
    window.location.href = '/perspectives.html';
  },
  
  openGroup(group) {
    // Open group detail view
    if (group && group.id) {
      window.location.href = `/group.html?id=${group.id}`;
    }
  },
  
  async createGroup() {
    // Check if user is logged in
    if (!this.currentUserId) {
      this.showToast('🔐 Sign in to create a group');
      setTimeout(() => {
        window.location.href = '/learn.html#auth';
      }, 1500);
      return;
    }
    
    // Show create group modal
    this.showCreateGroupModal();
  },
  
  showCreateGroupModal() {
    // Remove existing modal
    document.querySelector('.group-modal')?.remove();
    
    const modal = document.createElement('div');
    modal.className = 'group-modal';
    modal.innerHTML = `
      <div class="group-modal-content">
        <h3>Create Learning Group</h3>
        <p>Learn together with friends & family</p>
        
        <div class="group-form">
          <div class="form-group">
            <label>Group Name</label>
            <input type="text" id="group-name-input" placeholder="e.g., Smith Family" maxlength="50" />
          </div>
          
          <div class="form-group">
            <label>Choose an Emoji</label>
            <div class="emoji-picker" id="emoji-picker">
              <button type="button" class="emoji-option selected" data-emoji="👨‍👩‍👧‍👦">👨‍👩‍👧‍👦</button>
              <button type="button" class="emoji-option" data-emoji="🏠">🏠</button>
              <button type="button" class="emoji-option" data-emoji="📚">📚</button>
              <button type="button" class="emoji-option" data-emoji="🌟">🌟</button>
              <button type="button" class="emoji-option" data-emoji="🎓">🎓</button>
              <button type="button" class="emoji-option" data-emoji="🧠">🧠</button>
              <button type="button" class="emoji-option" data-emoji="🌍">🌍</button>
              <button type="button" class="emoji-option" data-emoji="💡">💡</button>
            </div>
          </div>
        </div>
        
        <div class="modal-actions">
          <button class="btn-cancel" id="cancel-group-btn">Cancel</button>
          <button class="btn-create" id="confirm-group-btn">Create Group</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    this.addGroupModalStyles();
    
    // Focus input
    setTimeout(() => document.getElementById('group-name-input')?.focus(), 100);
    
    // Bind events
    modal.querySelector('#cancel-group-btn').addEventListener('click', () => modal.remove());
    modal.querySelector('#confirm-group-btn').addEventListener('click', () => this.confirmCreateGroup(modal));
    
    // Emoji selection
    modal.querySelectorAll('.emoji-option').forEach(btn => {
      btn.addEventListener('click', () => {
        modal.querySelectorAll('.emoji-option').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
      });
    });
    
    // Close on outside click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
  },
  
  async confirmCreateGroup(modal) {
    const nameInput = modal.querySelector('#group-name-input');
    const selectedEmoji = modal.querySelector('.emoji-option.selected');
    
    const name = nameInput?.value?.trim();
    const emoji = selectedEmoji?.dataset?.emoji || '👨‍👩‍👧‍👦';
    
    if (!name) {
      nameInput?.classList.add('error');
      this.showToast('Please enter a group name');
      return;
    }
    
    if (!this.supabase) {
      this.showToast('Unable to create group - please try again');
      return;
    }
    
    // Generate invite code
    const inviteCode = this.generateInviteCode();
    
    try {
      // Create the group
      const { data: group, error: groupError } = await this.supabase
        .from('learning_groups')
        .insert({
          name: name,
          emoji: emoji,
          invite_code: inviteCode,
          created_by: this.currentUserId
        })
        .select()
        .single();
      
      if (groupError) throw groupError;
      
      // Add creator as member
      const { error: memberError } = await this.supabase
        .from('group_members')
        .insert({
          group_id: group.id,
          user_id: this.currentUserId,
          role: 'owner'
        });
      
      if (memberError) throw memberError;
      
      // Success!
      modal.remove();
      this.showToast(`✨ "${name}" created!`);
      
      // Refresh groups list
      await this.loadUserGroups();
      
      // Show invite code
      setTimeout(() => {
        this.showInviteCode(group);
      }, 1000);
      
    } catch (e) {
      console.error('[ShareHub] Create group failed:', e);
      this.showToast('Failed to create group - please try again');
    }
  },
  
  generateInviteCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  },
  
  showInviteCode(group) {
    const inviteUrl = `https://curiouskelly.com/join?code=${group.invite_code}`;
    
    const modal = document.createElement('div');
    modal.className = 'group-modal';
    modal.innerHTML = `
      <div class="group-modal-content">
        <div class="success-icon">🎉</div>
        <h3>Group Created!</h3>
        <p>Share this code to invite others:</p>
        
        <div class="invite-code-display">
          <span class="invite-code">${group.invite_code}</span>
          <button class="copy-code-btn" id="copy-invite-code">Copy</button>
        </div>
        
        <p class="invite-url">${inviteUrl}</p>
        
        <div class="modal-actions">
          <button class="btn-create" id="done-invite-btn">Done</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
    
    modal.querySelector('#copy-invite-code').addEventListener('click', () => {
      navigator.clipboard.writeText(inviteUrl).then(() => {
        this.showToast('📋 Invite link copied!');
      });
    });
    
    modal.querySelector('#done-invite-btn').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
  },
  
  addGroupModalStyles() {
    if (document.getElementById('group-modal-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'group-modal-styles';
    styles.textContent = `
      .group-modal {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        animation: fadeIn 0.2s ease;
      }
      
      @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
      }
      
      .group-modal-content {
        background: #1a1a2e;
        border-radius: 20px;
        padding: 32px;
        width: 90%;
        max-width: 400px;
        text-align: center;
      }
      
      .group-modal-content h3 {
        font-size: 1.5rem;
        margin-bottom: 8px;
        color: #fff;
      }
      
      .group-modal-content p {
        color: #a1a1aa;
        margin-bottom: 24px;
      }
      
      .success-icon {
        font-size: 3rem;
        margin-bottom: 16px;
      }
      
      .group-form {
        text-align: left;
      }
      
      .form-group {
        margin-bottom: 20px;
      }
      
      .form-group label {
        display: block;
        font-size: 0.9rem;
        color: #a1a1aa;
        margin-bottom: 8px;
      }
      
      .form-group input {
        width: 100%;
        padding: 14px 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #fff;
        font-size: 1rem;
        outline: none;
        transition: border-color 0.2s;
      }
      
      .form-group input:focus {
        border-color: #3b82f6;
      }
      
      .form-group input.error {
        border-color: #ef4444;
      }
      
      .emoji-picker {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      
      .emoji-option {
        width: 48px;
        height: 48px;
        font-size: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 2px solid transparent;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .emoji-option:hover {
        background: rgba(255, 255, 255, 0.1);
      }
      
      .emoji-option.selected {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.2);
      }
      
      .modal-actions {
        display: flex;
        gap: 12px;
        margin-top: 24px;
      }
      
      .btn-cancel, .btn-create {
        flex: 1;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
      }
      
      .btn-cancel {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #a1a1aa;
      }
      
      .btn-cancel:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #fff;
      }
      
      .btn-create {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border: none;
        color: #fff;
      }
      
      .btn-create:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
      }
      
      .invite-code-display {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 20px 0;
      }
      
      .invite-code {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 4px;
        color: #3b82f6;
        font-family: monospace;
      }
      
      .copy-code-btn {
        padding: 8px 16px;
        background: #3b82f6;
        border: none;
        border-radius: 8px;
        color: #fff;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s;
      }
      
      .copy-code-btn:hover {
        background: #2563eb;
      }
      
      .invite-url {
        font-size: 0.85rem !important;
        color: #71717a !important;
        word-break: break-all;
      }
      
      .launching-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 16px;
        font-size: 0.9rem;
        color: #a1a1aa;
      }
      
      .launching-badge span {
        color: #d1d5db;
      }
      
      .stat-number.launching {
        font-size: 2rem;
        animation: pulse 2s infinite;
      }
      
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
      }
    `;
    document.head.appendChild(styles);
  },
  
  openAmbassador() {
    // Go to Ambassador program page
    window.location.href = '/ambassador.html';
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




