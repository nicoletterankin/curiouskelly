/**
 * ✨ Chat Overlay - TikTok-Style Social Learning Comments
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * Displays simulated and real comments during lessons for social learning experience.
 * 
 * TRUST & SAFETY COMPLIANCE:
 * - All simulated content marked with ✨ indicator
 * - Respects SimulatedContentManager preferences
 * - Never claims simulated users are real people
 * - Transparent disclosure on hover
 * 
 * @module ChatOverlay
 */

// Diverse, thoughtful messages from around the world
const CHAT_MESSAGES_DB = {
  // Topic-aware messages (keyed by common topics)
  default: [
    { user: "Maria", flag: "🇧🇷", text: "Oh wow, this explains so much!", category: "insightful" },
    { user: "James", flag: "🇬🇧", text: "I never thought about it that way before", category: "insightful" },
    { user: "Yuki", flag: "🇯🇵", text: "This connects to what we learned yesterday!", category: "insightful" },
    { user: "Ahmed", flag: "🇪🇬", text: "Now the whole concept makes sense", category: "insightful" },
    { user: "Sofia", flag: "🇲🇽", text: "This is exactly what I needed to understand", category: "insightful" },
    { user: "Li Wei", flag: "🇨🇳", text: "The way Kelly explains this is brilliant", category: "insightful" },
    { user: "Priya", flag: "🇮🇳", text: "I can apply this to my work immediately", category: "insightful" },
    { user: "Hans", flag: "🇩🇪", text: "This changes how I see the world", category: "insightful" },
    { user: "Emma", flag: "🇺🇸", text: "Mind blown! 🤯", category: "excited" },
    { user: "Anya", flag: "🇷🇺", text: "Kelly is the best teacher ever! ✨", category: "excited" },
    { user: "Kofi", flag: "🇬🇭", text: "This is so cool!", category: "excited" },
    { user: "Mei", flag: "🇹🇼", text: "Wow wow wow! 🌟", category: "excited" },
    { user: "Lucas", flag: "🇫🇷", text: "C'est magnifique! 🙌", category: "excited" },
    { user: "Isabella", flag: "🇮🇹", text: "Incredibile!", category: "excited" },
    { user: "Aisha", flag: "🇰🇪", text: "Good morning from Nairobi! 🌅", category: "social" },
    { user: "Diego", flag: "🇨🇱", text: "Showing my kids right now 👨‍👩‍👧‍👦", category: "social" },
    { user: "Nina", flag: "🇳🇴", text: "Learning together across the world 🌍", category: "social" },
    { user: "Omar", flag: "🇦🇪", text: "My morning ritual with coffee ☕", category: "social" },
    { user: "Zara", flag: "🇿🇦", text: "Sharing with my students!", category: "social" },
    { user: "Kai", flag: "🇹🇭", text: "Hello from Bangkok! 👋", category: "social" },
    { user: "Miguel", flag: "🇪🇸", text: "Wait, so does that mean...? 🤔", category: "question" },
    { user: "Sarah", flag: "🇨🇦", text: "How does this connect to the last topic?", category: "question" }
  ],
  
  // Phase-specific messages
  welcome: [
    { user: "Raj", flag: "🇮🇳", text: "Ready to learn! 📚", category: "excited" },
    { user: "Elena", flag: "🇷🇴", text: "Good timing, just got my coffee ☕", category: "social" },
    { user: "Chen", flag: "🇸🇬", text: "Let's gooo!", category: "excited" }
  ],
  question: [
    { user: "Sven", flag: "🇸🇪", text: "Ooh, good question!", category: "insightful" },
    { user: "Fatima", flag: "🇲🇦", text: "I think I know this one...", category: "question" },
    { user: "Tom", flag: "🇦🇺", text: "This made me think twice", category: "insightful" }
  ],
  wisdom: [
    { user: "Yuki", flag: "🇯🇵", text: "That was beautiful 💫", category: "excited" },
    { user: "Grace", flag: "🇳🇬", text: "I'm going to remember this forever", category: "insightful" },
    { user: "Marco", flag: "🇧🇷", text: "Kelly always knows how to end strong", category: "social" }
  ]
};

class ChatOverlay {
  constructor(options = {}) {
    this.options = {
      topic: options.topic || 'Today\'s Lesson',
      phase: options.phase || 'welcome',
      lessonDay: options.lessonDay || 1,
      useDatabase: options.useDatabase !== false,
      minInterval: options.minInterval || 3000,
      maxInterval: options.maxInterval || 7000,
      containerId: options.containerId || 'chat-messages-container',
      floatContainerId: options.floatContainerId || 'comments-float',
      maxMessages: options.maxMessages || 20,
      ...options
    };
    
    this.container = document.getElementById(this.options.containerId);
    this.floatContainer = document.getElementById(this.options.floatContainerId);
    this.isActive = false;
    this.messageQueue = [];
    this.messageIndex = 0;
    this.scheduledTimeout = null;
    this.dbComments = [];
    this.dbCommentsLoaded = false;
    
    // Check SimulatedContentManager preferences
    this.simulatedEnabled = this._checkSimulatedEnabled();
    
    // Listen for preference changes
    window.addEventListener('simulated-content-changed', (e) => {
      this.simulatedEnabled = e.detail?.enabled !== false;
      if (!this.simulatedEnabled && this.isActive) {
        this.stop();
      }
    });
    
    console.log(`[ChatOverlay] ✨ Initialized for topic: ${this.options.topic}, day: ${this.options.lessonDay}`);
  }
  
  /**
   * Check if simulated content is enabled
   */
  _checkSimulatedEnabled() {
    if (window.SimulatedContent) {
      return window.SimulatedContent.isEnabled();
    }
    // Default to enabled if manager not loaded
    return true;
  }
  
  /**
   * Start showing chat messages
   */
  async start() {
    if (this.isActive) return;
    
    // Check if simulated content is disabled
    if (!this.simulatedEnabled) {
      console.log('[ChatOverlay] Simulated content disabled by user preference');
      return;
    }
    
    this.isActive = true;
    
    // Try to load real comments from database first
    if (this.options.useDatabase) {
      await this._loadDatabaseComments();
    }
    
    // Build message queue (mix real + simulated)
    this._buildMessageQueue();
    
    // Show first message immediately
    this._showNextMessage();
    
    // Schedule subsequent messages
    this._scheduleNext();
    
    console.log('[ChatOverlay] ✨ Started with', this.messageQueue.length, 'messages');
  }
  
  /**
   * Stop showing messages
   */
  stop() {
    this.isActive = false;
    if (this.scheduledTimeout) {
      clearTimeout(this.scheduledTimeout);
      this.scheduledTimeout = null;
    }
    console.log('[ChatOverlay] Stopped');
  }
  
  /**
   * Update the current phase (affects message selection)
   */
  setPhase(phase) {
    this.options.phase = phase;
    // Rebuild queue with phase-appropriate messages
    this._buildMessageQueue();
    console.log(`[ChatOverlay] Phase updated to: ${phase}`);
  }
  
  /**
   * Load real comments from Supabase
   */
  async _loadDatabaseComments() {
    if (!window.supabase || !window.CONFIG) {
      console.log('[ChatOverlay] Supabase not available, using simulated comments');
      return;
    }
    
    try {
      const supabase = window.supabase.createClient 
        ? window.supabase.createClient(window.CONFIG.SUPABASE_URL, window.CONFIG.SUPABASE_ANON_KEY)
        : window.supabase;
      
      // Try to load real comments for this lesson day
      const { data, error } = await supabase
        .from('lesson_comments')
        .select('user_name, country_code, comment_text, created_at')
        .eq('day_number', this.options.lessonDay)
        .eq('is_approved', true)
        .order('created_at', { ascending: false })
        .limit(50);
      
      if (data && !error && data.length > 0) {
        this.dbComments = data.map(c => ({
          user: c.user_name || 'Learner',
          flag: this._countryToFlag(c.country_code),
          text: c.comment_text,
          isReal: true,
          timestamp: c.created_at
        }));
        this.dbCommentsLoaded = true;
        console.log(`[ChatOverlay] Loaded ${this.dbComments.length} real comments from database`);
      }
    } catch (e) {
      console.warn('[ChatOverlay] Database load error:', e.message);
    }
  }
  
  /**
   * Convert country code to flag emoji
   */
  _countryToFlag(countryCode) {
    if (!countryCode || countryCode.length !== 2) return '🌍';
    const codePoints = countryCode
      .toUpperCase()
      .split('')
      .map(char => 127397 + char.charCodeAt(0));
    return String.fromCodePoint(...codePoints);
  }
  
  /**
   * Build the message queue from real + simulated comments
   */
  _buildMessageQueue() {
    const queue = [];
    
    // Add real database comments first (they don't need ✨)
    if (this.dbComments.length > 0) {
      queue.push(...this.dbComments);
    }
    
    // Add phase-specific simulated messages
    const phaseMessages = CHAT_MESSAGES_DB[this.options.phase] || [];
    queue.push(...phaseMessages.map(m => ({ ...m, isSimulated: true })));
    
    // Add default messages
    queue.push(...CHAT_MESSAGES_DB.default.map(m => ({ ...m, isSimulated: true })));
    
    // Shuffle the queue
    this.messageQueue = queue.sort(() => Math.random() - 0.5);
    this.messageIndex = 0;
  }
  
  /**
   * Schedule the next message
   */
  _scheduleNext() {
    if (!this.isActive) return;
    
    const delay = this.options.minInterval + 
      Math.random() * (this.options.maxInterval - this.options.minInterval);
    
    this.scheduledTimeout = setTimeout(() => {
      this._showNextMessage();
      this._scheduleNext();
    }, delay);
  }
  
  /**
   * Show the next message in the queue
   */
  _showNextMessage() {
    if (!this.container && !this.floatContainer) return;
    
    const msg = this.messageQueue[this.messageIndex % this.messageQueue.length];
    this.messageIndex++;
    
    // Create message element
    const messageEl = this._createMessageElement(msg);
    
    // Add to main container
    if (this.container) {
      this.container.appendChild(messageEl.cloneNode(true));
      this.container.scrollTop = this.container.scrollHeight;
      
      // Limit messages
      while (this.container.children.length > this.options.maxMessages) {
        this.container.removeChild(this.container.firstChild);
      }
    }
    
    // Add to float container (if exists and different)
    if (this.floatContainer && this.floatContainer !== this.container) {
      const floatMsg = messageEl.cloneNode(true);
      floatMsg.classList.add('float-comment');
      this.floatContainer.appendChild(floatMsg);
      
      // Auto-remove float messages after 5 seconds
      setTimeout(() => {
        floatMsg.classList.add('fade-out');
        setTimeout(() => floatMsg.remove(), 500);
      }, 5000);
      
      // Limit float messages
      while (this.floatContainer.children.length > 5) {
        this.floatContainer.removeChild(this.floatContainer.firstChild);
      }
    }
    
    // Use global addChatMessage if available (for unified styling)
    if (window.addChatMessage && msg.isReal) {
      window.addChatMessage(msg.user, msg.text, msg.flag);
    }
  }
  
  /**
   * Create a styled message element
   * TRUST & SAFETY: Simulated messages get ✨ indicator
   */
  _createMessageElement(msg) {
    const el = document.createElement('div');
    el.className = 'chat-comment' + (msg.isSimulated ? ' simulated-content' : ' real-content');
    
    // Build indicator - ✨ for simulated, nothing for real
    const indicator = msg.isSimulated 
      ? '<span class="simulated-indicator" title="Simulated comment to illustrate global learning community">✨</span>' 
      : '';
    
    el.innerHTML = `
      <span class="avatar">${msg.flag}</span>
      <span class="name">${msg.user}${indicator}</span>
      <span class="text">${msg.text}</span>
    `;
    
    // Add animation
    el.style.animation = 'chatSlideIn 0.3s ease-out';
    
    return el;
  }
  
  /**
   * Add a user-submitted comment (real, not simulated)
   */
  addUserComment(text, userName = 'You') {
    const msg = {
      user: userName,
      flag: '👤',
      text: text,
      isReal: true
    };
    
    // Show immediately
    const messageEl = this._createMessageElement(msg);
    if (this.container) {
      this.container.appendChild(messageEl);
      this.container.scrollTop = this.container.scrollHeight;
    }
    
    // Optionally save to database
    this._saveCommentToDatabase(text);
  }
  
  /**
   * Save a user comment to the database
   */
  async _saveCommentToDatabase(text) {
    if (!window.supabase || !window.CONFIG) return;
    
    try {
      const supabase = window.supabase.createClient 
        ? window.supabase.createClient(window.CONFIG.SUPABASE_URL, window.CONFIG.SUPABASE_ANON_KEY)
        : window.supabase;
      
      await supabase.from('lesson_comments').insert({
        day_number: this.options.lessonDay,
        comment_text: text,
        is_approved: false // Requires moderation
      });
      
      console.log('[ChatOverlay] Comment submitted for moderation');
    } catch (e) {
      console.warn('[ChatOverlay] Failed to save comment:', e.message);
    }
  }
}

// Add required CSS animations
const chatOverlayStyles = document.createElement('style');
chatOverlayStyles.textContent = `
  @keyframes chatSlideIn {
    from {
      opacity: 0;
      transform: translateX(20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  
  .chat-comment {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 12px;
    backdrop-filter: blur(8px);
    transition: opacity 0.3s ease;
  }
  
  .chat-comment .avatar {
    font-size: 16px;
    flex-shrink: 0;
  }
  
  .chat-comment .name {
    font-weight: 600;
    color: var(--tiktok-accent, #3b82f6);
    font-size: 0.85rem;
    margin-right: 6px;
  }
  
  .chat-comment .text {
    color: rgba(255, 255, 255, 0.95);
    font-size: 0.9rem;
    line-height: 1.4;
  }
  
  .chat-comment .simulated-indicator {
    font-size: 0.7rem;
    margin-left: 4px;
    cursor: help;
    opacity: 0.8;
  }
  
  .chat-comment.simulated-content .simulated-indicator {
    display: inline;
  }
  
  /* Hide simulated content when disabled */
  .simulated-content-disabled .chat-comment.simulated-content {
    display: none !important;
  }
  
  /* Float comment styles */
  .float-comment {
    position: relative;
    margin-bottom: 8px;
  }
  
  .float-comment.fade-out {
    opacity: 0;
    transform: translateY(-10px);
    transition: all 0.5s ease-out;
  }
`;
document.head.appendChild(chatOverlayStyles);

// Export globally
window.ChatOverlay = ChatOverlay;

console.log('[ChatOverlay] ✨ Module loaded - Trust & Safety compliant');
