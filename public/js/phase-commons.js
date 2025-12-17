/**
 * Phase Commons - In-Lesson Community Governance
 * 
 * Allows learners to interact with Learner Commons directly within a lesson:
 * - View proposals for the current phase
 * - Vote on proposals
 * - Add community notes
 * - See version history
 * - Suggest improvements
 * 
 * Usage:
 *   PhaseCommons.init();
 *   PhaseCommons.setCurrentPhase(dayNumber, phaseName);
 */

const PhaseCommons = {
  // Current lesson context
  currentDay: null,
  currentPhase: null,
  currentContent: null,
  
  // DOM elements (created on init)
  trigger: null,
  overlay: null,
  
  // State
  isOpen: false,
  proposals: [],
  notes: [],
  history: [],
  userVotes: {},
  
  // Config
  API_BASE: '/api',
  
  // ═══════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════
  
  init() {
    this.createTriggerButton();
    this.createOverlay();
    this.bindEvents();
    console.log('✅ PhaseCommons initialized');
  },
  
  createTriggerButton() {
    const trigger = document.createElement('div');
    trigger.className = 'phase-commons-trigger';
    trigger.innerHTML = `
      <span class="icon">💬</span>
      <span class="label">Commons</span>
      <span class="badge" style="display: none;">0</span>
    `;
    document.body.appendChild(trigger);
    this.trigger = trigger;
    
    trigger.addEventListener('click', () => this.open());
  },
  
  createOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'phase-commons-overlay';
    overlay.innerHTML = `
      <div class="phase-commons-container">
        <!-- Header -->
        <div class="phase-commons-header">
          <div class="phase-commons-title">
            <h2>
              <span class="phase-icon">💬</span>
              Phase Commons
              <span class="phase-badge">Hook</span>
            </h2>
            <span class="subtitle">Day <span class="day-number">1</span> • Shape how Kelly teaches</span>
          </div>
          <button class="phase-commons-close">×</button>
        </div>
        
        <!-- Tabs -->
        <div class="phase-commons-tabs">
          <button class="phase-commons-tab active" data-tab="proposals">
            📝 Proposals <span class="count">0</span>
          </button>
          <button class="phase-commons-tab" data-tab="notes">
            💡 Notes <span class="count">0</span>
          </button>
          <button class="phase-commons-tab" data-tab="history">
            📜 History
          </button>
        </div>
        
        <!-- Content -->
        <div class="phase-commons-content">
          <!-- Current Content Preview -->
          <div class="current-content-preview">
            <div class="label">Current Script</div>
            <div class="content">"Loading..."</div>
            <div class="version-badge">
              <span class="icon">📌</span> Version <span class="version">1</span>
            </div>
          </div>
          
          <!-- Proposals Section -->
          <div class="phase-commons-section active" data-section="proposals">
            <div class="proposals-list">
              <!-- Populated dynamically -->
            </div>
            <div class="empty-state" style="display: none;">
              <div class="icon">📝</div>
              <h3>No proposals yet</h3>
              <p>Be the first to suggest an improvement for this phase!</p>
            </div>
          </div>
          
          <!-- Notes Section -->
          <div class="phase-commons-section" data-section="notes">
            <div class="notes-list">
              <!-- Populated dynamically -->
            </div>
            <div class="empty-state" style="display: none;">
              <div class="icon">💡</div>
              <h3>No community notes</h3>
              <p>Share your expertise or add helpful context!</p>
            </div>
          </div>
          
          <!-- History Section -->
          <div class="phase-commons-section" data-section="history">
            <div class="version-history">
              <!-- Populated dynamically -->
            </div>
          </div>
        </div>
        
        <!-- Footer with action button -->
        <div class="phase-commons-footer">
          <button class="suggest-btn">
            <span>✨</span> Suggest Improvement
          </button>
          
          <div class="suggest-form">
            <div class="suggest-form-group">
              <label>Type of change</label>
              <select id="suggest-type">
                <option value="enhance">Enhance - Make it better</option>
                <option value="correct">Correct - Fix an error</option>
                <option value="simplify">Simplify - Make easier</option>
                <option value="expand">Expand - Add more detail</option>
              </select>
            </div>
            <div class="suggest-form-group">
              <label>Your proposed change</label>
              <textarea id="suggest-content" placeholder="Write your improved version of this content..."></textarea>
            </div>
            <div class="suggest-form-group">
              <label>Why this change? (optional)</label>
              <textarea id="suggest-rationale" placeholder="Explain why this would be an improvement..."></textarea>
            </div>
            <div class="suggest-form-actions">
              <button type="button" class="cancel-btn">Cancel</button>
              <button type="submit" class="submit-btn" disabled>Submit Proposal</button>
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    this.overlay = overlay;
  },
  
  bindEvents() {
    // Close button
    this.overlay.querySelector('.phase-commons-close').addEventListener('click', () => this.close());
    
    // Background click to close
    this.overlay.addEventListener('click', (e) => {
      if (e.target === this.overlay) this.close();
    });
    
    // Tab switching
    this.overlay.querySelectorAll('.phase-commons-tab').forEach(tab => {
      tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
    });
    
    // Suggest button
    this.overlay.querySelector('.suggest-btn').addEventListener('click', () => this.showSuggestForm());
    
    // Cancel suggest
    this.overlay.querySelector('.cancel-btn').addEventListener('click', () => this.hideSuggestForm());
    
    // Submit proposal
    this.overlay.querySelector('.submit-btn').addEventListener('click', () => this.submitProposal());
    
    // Enable submit when content is entered
    this.overlay.querySelector('#suggest-content').addEventListener('input', (e) => {
      this.overlay.querySelector('.submit-btn').disabled = !e.target.value.trim();
    });
    
    // Escape key to close
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen) this.close();
    });
  },
  
  // ═══════════════════════════════════════════════════════════════
  // PUBLIC API
  // ═══════════════════════════════════════════════════════════════
  
  /**
   * Set the current phase context and show/hide trigger
   */
  setCurrentPhase(dayNumber, phaseName, content = null) {
    this.currentDay = dayNumber;
    this.currentPhase = phaseName;
    this.currentContent = content;
    
    // Show trigger button
    this.trigger.classList.add('visible');
    
    // Update badge (fetch proposal count)
    this.fetchProposalCount();
    
    // Update overlay if open
    if (this.isOpen) {
      this.updateOverlayContent();
    }
  },
  
  /**
   * Hide the commons trigger (e.g., on non-lesson screens)
   */
  hide() {
    this.trigger.classList.remove('visible');
    if (this.isOpen) this.close();
  },
  
  /**
   * Open the commons overlay
   */
  open() {
    if (!this.currentDay || !this.currentPhase) {
      console.warn('PhaseCommons: No phase context set');
      return;
    }
    
    this.isOpen = true;
    this.overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
    
    this.updateOverlayContent();
    this.loadData();
  },
  
  /**
   * Close the commons overlay
   */
  close() {
    this.isOpen = false;
    this.overlay.classList.remove('open');
    document.body.style.overflow = '';
    this.hideSuggestForm();
  },
  
  // ═══════════════════════════════════════════════════════════════
  // DATA LOADING
  // ═══════════════════════════════════════════════════════════════
  
  async fetchProposalCount() {
    try {
      const address = this.getContentAddress();
      // TODO: Replace with real API call
      const count = await this.fetchProposalCount(address);
      
      const badge = this.trigger.querySelector('.badge');
      if (count > 0) {
        badge.textContent = count;
        badge.style.display = 'block';
        if (count >= 5) badge.classList.add('hot');
      } else {
        badge.style.display = 'none';
      }
    } catch (e) {
      console.error('Failed to fetch proposal count:', e);
    }
  },
  
  async loadData() {
    // Show loading state
    this.showLoadingState();
    
    try {
      const address = this.getContentAddress();
      
      // Load in parallel
      const [proposals, notes, history] = await Promise.all([
        this.loadProposals(address),
        this.loadNotes(address),
        this.loadHistory(address)
      ]);
      
      this.proposals = proposals;
      this.notes = notes;
      this.history = history;
      
      this.renderProposals();
      this.renderNotes();
      this.renderHistory();
      this.updateCounts();
      
    } catch (e) {
      console.error('Failed to load commons data:', e);
      this.showErrorState();
    }
  },
  
  getContentAddress() {
    const paddedDay = String(this.currentDay).padStart(3, '0');
    const phase = this.currentPhase.toLowerCase();
    return `${paddedDay}.${phase}.talk`;
  },
  
  // ═══════════════════════════════════════════════════════════════
  // RENDERING
  // ═══════════════════════════════════════════════════════════════
  
  updateOverlayContent() {
    // Update header
    const phaseBadge = this.overlay.querySelector('.phase-badge');
    const dayNumber = this.overlay.querySelector('.day-number');
    const phaseIcon = this.overlay.querySelector('.phase-icon');
    
    const icons = {
      hook: '🪝', cliff: '🧗', fact1: '💡', fact2: '🧠', 
      fact3: '✨', wisdom: '🦉', outro: '👋'
    };
    
    phaseBadge.textContent = this.formatPhaseName(this.currentPhase);
    dayNumber.textContent = this.currentDay;
    phaseIcon.textContent = icons[this.currentPhase.toLowerCase()] || '📖';
    
    // Update current content preview
    if (this.currentContent) {
      const preview = this.overlay.querySelector('.current-content-preview .content');
      preview.textContent = `"${this.truncate(this.currentContent, 200)}"`;
    }
  },
  
  renderProposals() {
    const container = this.overlay.querySelector('.proposals-list');
    const emptyState = this.overlay.querySelector('[data-section="proposals"] .empty-state');
    
    if (this.proposals.length === 0) {
      container.innerHTML = '';
      emptyState.style.display = 'block';
      return;
    }
    
    emptyState.style.display = 'none';
    container.innerHTML = this.proposals.map(p => this.renderProposalCard(p)).join('');
    
    // Bind vote handlers
    container.querySelectorAll('.vote-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const proposalId = e.target.closest('.proposal-card').dataset.id;
        const voteType = btn.classList.contains('upvote') ? 'up' : 'down';
        this.vote(proposalId, voteType);
      });
    });
  },
  
  renderProposalCard(proposal) {
    const userVote = this.userVotes[proposal.id];
    const score = proposal.upvotes - proposal.downvotes;
    const scoreClass = score > 0 ? 'positive' : score < 0 ? 'negative' : '';
    
    return `
      <div class="proposal-card" data-id="${proposal.id}">
        <div class="proposal-header">
          <div class="proposal-votes">
            <button class="vote-btn upvote ${userVote === 'up' ? 'active' : ''}">▲</button>
            <span class="vote-count ${scoreClass}">${score}</span>
            <button class="vote-btn downvote ${userVote === 'down' ? 'active' : ''}">▼</button>
          </div>
          <div class="proposal-info">
            <span class="proposal-type ${proposal.type}">${proposal.type}</span>
            <div class="proposal-title">${proposal.title}</div>
            <div class="proposal-author">by @${proposal.author} • ${this.timeAgo(proposal.createdAt)}</div>
          </div>
        </div>
        ${proposal.diff ? `
          <div class="proposal-diff">
            <span class="old">${proposal.diff.old}</span>
            <span class="new">${proposal.diff.new}</span>
          </div>
        ` : ''}
        <div class="proposal-actions">
          <button onclick="PhaseCommons.viewProposal('${proposal.id}')">View Details</button>
          <button onclick="PhaseCommons.discussProposal('${proposal.id}')">💬 Discuss</button>
        </div>
      </div>
    `;
  },
  
  renderNotes() {
    const container = this.overlay.querySelector('.notes-list');
    const emptyState = this.overlay.querySelector('[data-section="notes"] .empty-state');
    
    if (this.notes.length === 0) {
      container.innerHTML = '';
      emptyState.style.display = 'block';
      return;
    }
    
    emptyState.style.display = 'none';
    container.innerHTML = this.notes.map(n => this.renderNoteCard(n)).join('');
  },
  
  renderNoteCard(note) {
    const typeIcons = {
      expert_context: '🎓',
      historical_note: '📜',
      source_citation: '📚',
      teaching_tip: '💡',
      common_misconception: '⚠️',
      real_world_example: '🌍'
    };
    
    return `
      <div class="note-card ${note.isVerified ? 'verified' : ''} ${note.isFeatured ? 'featured' : ''}">
        <div class="note-header">
          <div class="note-type-icon">${typeIcons[note.type] || '📝'}</div>
          <div class="note-meta">
            <div class="note-type-label">${this.formatNoteType(note.type)}</div>
            <div class="note-author">
              by @${note.author}
              ${note.isVerified ? '<span class="verified-badge">✓</span>' : ''}
            </div>
          </div>
        </div>
        <div class="note-content">${note.content}</div>
        ${note.sources?.length ? `
          <div class="note-sources">
            <div class="label">Sources</div>
            ${note.sources.map(s => `<a href="${s}" target="_blank">${this.formatUrl(s)}</a>`).join(', ')}
          </div>
        ` : ''}
        <div class="note-reactions">
          <span class="note-reaction" data-reaction="helpful">
            👍 ${note.reactions?.helpful || 0} Helpful
          </span>
          <span class="note-reaction" data-reaction="insightful">
            💡 ${note.reactions?.insightful || 0} Insightful
          </span>
        </div>
      </div>
    `;
  },
  
  renderHistory() {
    const container = this.overlay.querySelector('.version-history');
    
    if (this.history.length === 0) {
      // Show at least v1
      this.history = [{
        version: 1,
        source: 'initial_seed',
        content: this.currentContent || 'Original content',
        createdAt: new Date('2025-12-17'),
        isCurrent: true
      }];
    }
    
    container.innerHTML = this.history.map((v, i) => `
      <div class="version-item ${i === 0 ? 'current' : ''}">
        <div class="version-header">
          <span class="version-number ${i === 0 ? 'current' : ''}">
            Version ${v.version} ${i === 0 ? '(Current)' : ''}
          </span>
          <span class="version-date">${this.formatDate(v.createdAt)}</span>
        </div>
        <div class="version-source">
          <span class="source-badge ${v.source === 'commons_proposal' ? 'commons' : 'seed'}">
            ${v.source === 'commons_proposal' ? '💬 Commons' : '🌱 Initial'}
          </span>
          ${v.reason || ''}
        </div>
        <div class="version-content">"${this.truncate(v.content, 150)}"</div>
        ${i > 0 ? `
          <div class="version-actions">
            <button onclick="PhaseCommons.compareVersions(${v.version})">Compare</button>
          </div>
        ` : ''}
      </div>
    `).join('');
  },
  
  updateCounts() {
    this.overlay.querySelector('[data-tab="proposals"] .count').textContent = this.proposals.length;
    this.overlay.querySelector('[data-tab="notes"] .count').textContent = this.notes.length;
  },
  
  // ═══════════════════════════════════════════════════════════════
  // TAB SWITCHING
  // ═══════════════════════════════════════════════════════════════
  
  switchTab(tabName) {
    // Update tab buttons
    this.overlay.querySelectorAll('.phase-commons-tab').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update sections
    this.overlay.querySelectorAll('.phase-commons-section').forEach(section => {
      section.classList.toggle('active', section.dataset.section === tabName);
    });
  },
  
  // ═══════════════════════════════════════════════════════════════
  // VOTING
  // ═══════════════════════════════════════════════════════════════
  
  async vote(proposalId, voteType) {
    const previousVote = this.userVotes[proposalId];
    
    // Optimistic update
    if (previousVote === voteType) {
      // Remove vote
      delete this.userVotes[proposalId];
    } else {
      // Add/change vote
      this.userVotes[proposalId] = voteType;
    }
    
    // Update proposal in list
    const proposal = this.proposals.find(p => p.id === proposalId);
    if (proposal) {
      if (previousVote === 'up') proposal.upvotes--;
      if (previousVote === 'down') proposal.downvotes--;
      if (this.userVotes[proposalId] === 'up') proposal.upvotes++;
      if (this.userVotes[proposalId] === 'down') proposal.downvotes++;
    }
    
    // Re-render
    this.renderProposals();
    
    // Animate vote button
    const card = this.overlay.querySelector(`[data-id="${proposalId}"]`);
    if (card) {
      const btn = card.querySelector(`.vote-btn.${voteType}`);
      btn.classList.add('voted');
      setTimeout(() => btn.classList.remove('voted'), 300);
    }
    
    // TODO: Send to API
    try {
      await this.sendVote(proposalId, this.userVotes[proposalId] || null);
    } catch (e) {
      console.error('Failed to send vote:', e);
      // Revert on error
      this.userVotes[proposalId] = previousVote;
      this.renderProposals();
    }
  },
  
  // ═══════════════════════════════════════════════════════════════
  // SUGGEST IMPROVEMENT
  // ═══════════════════════════════════════════════════════════════
  
  showSuggestForm() {
    this.overlay.querySelector('.suggest-btn').style.display = 'none';
    this.overlay.querySelector('.suggest-form').classList.add('active');
    this.overlay.querySelector('#suggest-content').focus();
  },
  
  hideSuggestForm() {
    this.overlay.querySelector('.suggest-btn').style.display = 'flex';
    this.overlay.querySelector('.suggest-form').classList.remove('active');
    this.overlay.querySelector('#suggest-content').value = '';
    this.overlay.querySelector('#suggest-rationale').value = '';
  },
  
  async submitProposal() {
    const type = this.overlay.querySelector('#suggest-type').value;
    const content = this.overlay.querySelector('#suggest-content').value;
    const rationale = this.overlay.querySelector('#suggest-rationale').value;
    
    if (!content.trim()) return;
    
    const submitBtn = this.overlay.querySelector('.submit-btn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';
    
    try {
      const address = this.getContentAddress();
      
      await this.submitProposalToAPI({
        targetAtoms: [address],
        type,
        title: `${this.formatProposalType(type)} for ${this.formatPhaseName(this.currentPhase)}`,
        proposedChanges: {
          [address]: {
            current: this.currentContent,
            proposed: content
          }
        },
        rationale
      });
      
      // Success!
      this.hideSuggestForm();
      
      // Show success message
      alert('🎉 Your proposal has been submitted! The community will vote on it.');
      
      // Reload proposals
      this.loadData();
      
    } catch (e) {
      console.error('Failed to submit proposal:', e);
      alert('Failed to submit proposal. Please try again.');
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Proposal';
    }
  },
  
  // ═══════════════════════════════════════════════════════════════
  // REAL API CALLS
  // ═══════════════════════════════════════════════════════════════
  
  async getAuthToken() {
    try {
      const supabase = window.getSupabase?.();
      if (supabase) {
        const { data } = await supabase.auth.getSession();
        return data?.session?.access_token || null;
      }
    } catch (e) {
      console.warn('[PhaseCommons] Could not get auth token:', e);
    }
    return null;
  },
  
  async fetchProposalCount(address) {
    try {
      const response = await fetch(`${this.API_BASE}/commons/proposals?address=${encodeURIComponent(address)}&status=open`);
      const data = await response.json();
      return data.count || data.proposals?.length || 0;
    } catch (e) {
      console.warn('[PhaseCommons] Failed to fetch proposal count:', e);
      return 0;
    }
  },
  
  async loadProposals(address) {
    try {
      const response = await fetch(`${this.API_BASE}/commons/proposals?address=${encodeURIComponent(address)}`);
      const data = await response.json();
      
      // Transform API response to UI format
      return (data.proposals || []).map(p => ({
        id: p.id,
        type: p.type,
        title: p.title,
        author: p.author || 'Anonymous',
        createdAt: p.createdAt,
        upvotes: p.upvotes || 0,
        downvotes: p.downvotes || 0,
        diff: p.proposedChanges?.[address] ? {
          old: p.proposedChanges[address].current || '',
          new: p.proposedChanges[address].proposed || ''
        } : null
      }));
    } catch (e) {
      console.error('[PhaseCommons] Failed to load proposals:', e);
      return [];
    }
  },
  
  async loadNotes(address) {
    try {
      const response = await fetch(`${this.API_BASE}/commons/notes?address=${encodeURIComponent(address)}`);
      const data = await response.json();
      return data.notes || [];
    } catch (e) {
      console.error('[PhaseCommons] Failed to load notes:', e);
      return [];
    }
  },
  
  async loadHistory(address) {
    try {
      const response = await fetch(`${this.API_BASE}/commons/history?address=${encodeURIComponent(address)}`);
      const data = await response.json();
      return data.history || [];
    } catch (e) {
      console.error('[PhaseCommons] Failed to load history:', e);
      // Return default history with current content
      return [{
        version: 1,
        source: 'initial_seed',
        content: this.currentContent || 'Original content',
        createdAt: new Date('2025-12-17'),
        isCurrent: true
      }];
    }
  },
  
  async sendVote(proposalId, voteType) {
    const token = await this.getAuthToken();
    if (!token) {
      throw new Error('Must be signed in to vote');
    }
    
    const response = await fetch(`${this.API_BASE}/commons/votes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ proposalId, vote: voteType })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to vote');
    }
    
    return response.json();
  },
  
  async submitProposalToAPI(data) {
    const token = await this.getAuthToken();
    if (!token) {
      throw new Error('Must be signed in to submit proposals');
    }
    
    const response = await fetch(`${this.API_BASE}/commons/proposals`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to submit proposal');
    }
    
    return response.json();
  },
  
  // ═══════════════════════════════════════════════════════════════
  // UTILITIES
  // ═══════════════════════════════════════════════════════════════
  
  formatPhaseName(phase) {
    const names = {
      hook: 'Hook', cliff: 'Cliff', fact1: 'Fact 1', fact2: 'Fact 2',
      fact3: 'Fact 3', wisdom: 'Wisdom', outro: 'Outro'
    };
    return names[phase.toLowerCase()] || phase;
  },
  
  formatNoteType(type) {
    const names = {
      expert_context: 'Expert Context',
      historical_note: 'Historical Note',
      source_citation: 'Source Citation',
      teaching_tip: 'Teaching Tip',
      common_misconception: 'Common Misconception',
      real_world_example: 'Real World Example'
    };
    return names[type] || type;
  },
  
  formatProposalType(type) {
    const names = {
      enhance: 'Enhancement',
      correct: 'Correction',
      simplify: 'Simplification',
      expand: 'Expansion'
    };
    return names[type] || type;
  },
  
  truncate(str, length) {
    if (!str) return '';
    return str.length > length ? str.slice(0, length) + '...' : str;
  },
  
  timeAgo(date) {
    const seconds = Math.floor((new Date() - new Date(date)) / 1000);
    const intervals = [
      { label: 'year', seconds: 31536000 },
      { label: 'month', seconds: 2592000 },
      { label: 'week', seconds: 604800 },
      { label: 'day', seconds: 86400 },
      { label: 'hour', seconds: 3600 },
      { label: 'minute', seconds: 60 }
    ];
    
    for (const { label, seconds: s } of intervals) {
      const count = Math.floor(seconds / s);
      if (count >= 1) {
        return `${count} ${label}${count > 1 ? 's' : ''} ago`;
      }
    }
    return 'just now';
  },
  
  formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  },
  
  formatUrl(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return url;
    }
  },
  
  showLoadingState() {
    // Could add loading spinners
  },
  
  showErrorState() {
    // Could show error message
  },
  
  // Actions (to be implemented)
  viewProposal(id) {
    console.log('View proposal:', id);
    // Could open detailed view
  },
  
  discussProposal(id) {
    console.log('Discuss proposal:', id);
    // Could open discussion thread
  },
  
  compareVersions(version) {
    console.log('Compare version:', version);
    // Could show diff view
  }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => PhaseCommons.init());
} else {
  PhaseCommons.init();
}

// Export for module systems
if (typeof module !== 'undefined') {
  module.exports = PhaseCommons;
}
