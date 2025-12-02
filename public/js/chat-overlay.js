/**
 * Chat Overlay v2 - TikTok-Style Social Learning
 * 
 * The "dirty secret": Curated comments that trigger social learning psychology
 * - Social proof ("I finally get it!")
 * - Engagement hooks ("Wait, so that means...")
 * - Kelly appreciation ("Best teacher ever")
 * - Learning moments ("Mind = blown")
 * - Safe, simulated social environment
 */

// ═══════════════════════════════════════════════════════════════════
// COMMENT BANKS - Organized by psychological trigger
// ═══════════════════════════════════════════════════════════════════

const COMMENT_BANKS = {
  // 💡 BREAKTHROUGH MOMENTS - "I finally understand!"
  breakthrough: [
    { user: "Maria", flag: "🇧🇷", text: "OMG this finally makes sense 🤯", verified: false },
    { user: "James", flag: "🇬🇧", text: "Wait... I've been thinking about this wrong my whole life", verified: false },
    { user: "Yuki", flag: "🇯🇵", text: "なるほど! I finally get it ✨", verified: false },
    { user: "Ahmed", flag: "🇪🇬", text: "This just clicked for me 💡", verified: false },
    { user: "Sofia", flag: "🇲🇽", text: "My brain just leveled up", verified: false },
    { user: "Priya", flag: "🇮🇳", text: "I wish I learned this years ago!", verified: false },
    { user: "Hans", flag: "🇩🇪", text: "Mind = completely blown 🤯", verified: false },
    { user: "Nina", flag: "🇳🇴", text: "This is the explanation I needed", verified: false },
    { user: "Chen Wei", flag: "🇨🇳", text: "终于明白了! Finally!", verified: false },
    { user: "Isabella", flag: "🇮🇹", text: "OHHHH so THAT'S why!", verified: false },
  ],

  // 🤔 ENGAGEMENT HOOKS - Questions that show active learning
  engagement: [
    { user: "Emma", flag: "🇺🇸", text: "Wait so does that mean...? 🤔", verified: false },
    { user: "Lucas", flag: "🇫🇷", text: "But what about when...?", verified: false },
    { user: "Kofi", flag: "🇬🇭", text: "Can someone explain the second part?", verified: false },
    { user: "Aisha", flag: "🇰🇪", text: "I never thought of it that way!", verified: false },
    { user: "Diego", flag: "🇨🇱", text: "This connects to what we learned yesterday!", verified: false },
    { user: "Mei", flag: "🇹🇼", text: "Real question: why don't they teach this in school?", verified: false },
    { user: "Omar", flag: "🇦🇪", text: "Hold on, let me think about this... 🧠", verified: false },
    { user: "Zara", flag: "🇿🇦", text: "So the key insight is...?", verified: false },
  ],

  // ❤️ KELLY APPRECIATION - Why they love learning with Kelly
  kelly_love: [
    { user: "Sarah", flag: "🇨🇦", text: "Kelly's voice is so calming 💙", verified: false },
    { user: "Jin", flag: "🇰🇷", text: "Best AI teacher EVER", verified: false },
    { user: "Anya", flag: "🇷🇺", text: "I'm literally addicted to learning now", verified: false },
    { user: "Carlos", flag: "🇦🇷", text: "Kelly > my college professors 💯", verified: false },
    { user: "Lena", flag: "🇸🇪", text: "365 days of this? Yes please! 🙌", verified: false },
    { user: "Raj", flag: "🇮🇳", text: "Kelly makes everything interesting", verified: false },
    { user: "Fatima", flag: "🇲🇦", text: "I look forward to this every day now", verified: false },
    { user: "Tomoko", flag: "🇯🇵", text: "Kelly-sensei! 🌸", verified: false },
    { user: "Sven", flag: "🇩🇰", text: "This is the future of education", verified: false },
    { user: "Ana", flag: "🇵🇹", text: "Kelly's explanations hit different ✨", verified: false },
  ],

  // 👨‍👩‍👧 SOCIAL CONNECTION - Learning together
  social: [
    { user: "Michael", flag: "🇺🇸", text: "Watching with my daughter 👨‍👧", verified: true },
    { user: "Lisa", flag: "🇦🇺", text: "Our whole family does this together!", verified: false },
    { user: "Kenji", flag: "🇯🇵", text: "Morning ritual with coffee ☕", verified: false },
    { user: "Maria", flag: "🇪🇸", text: "My kids are hooked on this", verified: false },
    { user: "David", flag: "🇮🇱", text: "Study group checking in! 👋", verified: false },
    { user: "Olga", flag: "🇺🇦", text: "Hello from Kyiv! 💙💛", verified: false },
    { user: "Adebayo", flag: "🇳🇬", text: "Lagos in the building! 🏙️", verified: false },
    { user: "Camila", flag: "🇧🇷", text: "Brazil loves Kelly! 💚💛", verified: false },
    { user: "Pierre", flag: "🇫🇷", text: "Bonjour from Paris! 🗼", verified: false },
    { user: "Ling", flag: "🇸🇬", text: "Singapore checking in!", verified: false },
  ],

  // 🔥 REACTIONS - Quick emotional responses (TikTok style)
  reactions: [
    { user: "User", flag: "🇺🇸", text: "🔥🔥🔥", verified: false },
    { user: "User", flag: "🇬🇧", text: "👏👏👏", verified: false },
    { user: "User", flag: "🇯🇵", text: "❤️", verified: false },
    { user: "User", flag: "🇧🇷", text: "💯", verified: false },
    { user: "User", flag: "🇮🇳", text: "🙏", verified: false },
    { user: "User", flag: "🇩🇪", text: "💡💡💡", verified: false },
    { user: "User", flag: "🇫🇷", text: "🤯", verified: false },
    { user: "User", flag: "🇪🇸", text: "✨", verified: false },
    { user: "User", flag: "🇲🇽", text: "🎯", verified: false },
    { user: "User", flag: "🇨🇦", text: "💪", verified: false },
  ],

  // 📝 CHOICE PHASE - Comments during question phases
  choice_phase: [
    { user: "Alex", flag: "🇺🇸", text: "I'm going with A!", verified: false },
    { user: "Sophie", flag: "🇫🇷", text: "B for sure 🤔", verified: false },
    { user: "Hiroshi", flag: "🇯🇵", text: "This one's tricky...", verified: false },
    { user: "Eva", flag: "🇩🇪", text: "Wait let me think about this", verified: false },
    { user: "Marco", flag: "🇮🇹", text: "I changed my answer 3 times 😅", verified: false },
    { user: "Aaliya", flag: "🇵🇰", text: "Going with my gut on this one", verified: false },
    { user: "Tom", flag: "🇬🇧", text: "I said A but now I'm not sure", verified: false },
    { user: "Maya", flag: "🇮🇳", text: "Both seem right to me? 🤷‍♀️", verified: false },
  ],

  // 🎓 WISDOM PHASE - Comments during conclusion/wisdom
  wisdom_phase: [
    { user: "Rebecca", flag: "🇺🇸", text: "This is so deep 🌊", verified: false },
    { user: "Takeshi", flag: "🇯🇵", text: "Words to live by 🙏", verified: false },
    { user: "Elena", flag: "🇷🇺", text: "Screenshotting this", verified: false },
    { user: "Paulo", flag: "🇧🇷", text: "Wisdom dropped 💎", verified: false },
    { user: "Kim", flag: "🇰🇷", text: "I'm putting this in my journal", verified: false },
    { user: "Fatou", flag: "🇸🇳", text: "This lesson changed my perspective", verified: false },
    { user: "Henrik", flag: "🇳🇴", text: "Beautiful way to end", verified: false },
    { user: "Nadia", flag: "🇲🇦", text: "Sharing this with everyone 📲", verified: false },
  ],

  // 📚 TOPIC-SPECIFIC TEMPLATES (filled in dynamically)
  topic_specific: [
    { user: "Student", flag: "🌍", text: "I never knew {topic} was so fascinating!", verified: false },
    { user: "Learner", flag: "🌍", text: "Can we get more lessons on {topic}?", verified: false },
    { user: "Curious", flag: "🌍", text: "{topic} just became my new obsession", verified: false },
    { user: "Mind", flag: "🌍", text: "The {topic} episode is elite 🔥", verified: false },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// GLOBAL NAMES POOL - Authentic diverse names
// ═══════════════════════════════════════════════════════════════════
const GLOBAL_NAMES = [
  { name: "Emma", flag: "🇺🇸" }, { name: "Liam", flag: "🇺🇸" }, { name: "Olivia", flag: "🇺🇸" },
  { name: "James", flag: "🇬🇧" }, { name: "Charlotte", flag: "🇬🇧" }, { name: "Harry", flag: "🇬🇧" },
  { name: "Marie", flag: "🇫🇷" }, { name: "Lucas", flag: "🇫🇷" }, { name: "Léa", flag: "🇫🇷" },
  { name: "Hans", flag: "🇩🇪" }, { name: "Lena", flag: "🇩🇪" }, { name: "Finn", flag: "🇩🇪" },
  { name: "Yuki", flag: "🇯🇵" }, { name: "Haruto", flag: "🇯🇵" }, { name: "Sakura", flag: "🇯🇵" },
  { name: "Priya", flag: "🇮🇳" }, { name: "Arjun", flag: "🇮🇳" }, { name: "Ananya", flag: "🇮🇳" },
  { name: "Wei", flag: "🇨🇳" }, { name: "Mei", flag: "🇨🇳" }, { name: "Chen", flag: "🇨🇳" },
  { name: "Sofia", flag: "🇲🇽" }, { name: "Diego", flag: "🇲🇽" }, { name: "Valentina", flag: "🇲🇽" },
  { name: "Camila", flag: "🇧🇷" }, { name: "Pedro", flag: "🇧🇷" }, { name: "Julia", flag: "🇧🇷" },
  { name: "Fatima", flag: "🇪🇬" }, { name: "Ahmed", flag: "🇪🇬" }, { name: "Nour", flag: "🇪🇬" },
  { name: "Aarav", flag: "🇮🇳" }, { name: "Zara", flag: "🇵🇰" }, { name: "Hassan", flag: "🇵🇰" },
  { name: "Olga", flag: "🇷🇺" }, { name: "Ivan", flag: "🇷🇺" }, { name: "Anastasia", flag: "🇷🇺" },
  { name: "Jin", flag: "🇰🇷" }, { name: "Soo-yeon", flag: "🇰🇷" }, { name: "Min-jun", flag: "🇰🇷" },
  { name: "Isabella", flag: "🇮🇹" }, { name: "Marco", flag: "🇮🇹" }, { name: "Giulia", flag: "🇮🇹" },
  { name: "Kofi", flag: "🇬🇭" }, { name: "Ama", flag: "🇬🇭" }, { name: "Kwame", flag: "🇬🇭" },
  { name: "Thabo", flag: "🇿🇦" }, { name: "Naledi", flag: "🇿🇦" }, { name: "Sipho", flag: "🇿🇦" },
  { name: "Omar", flag: "🇦🇪" }, { name: "Layla", flag: "🇦🇪" }, { name: "Amir", flag: "🇦🇪" },
  { name: "Sven", flag: "🇸🇪" }, { name: "Elsa", flag: "🇸🇪" }, { name: "Lars", flag: "🇸🇪" },
  { name: "Carlos", flag: "🇦🇷" }, { name: "Luciana", flag: "🇦🇷" }, { name: "Mateo", flag: "🇦🇷" },
  { name: "Nguyen", flag: "🇻🇳" }, { name: "Linh", flag: "🇻🇳" }, { name: "Minh", flag: "🇻🇳" },
];

// ═══════════════════════════════════════════════════════════════════
// CHAT OVERLAY CLASS
// ═══════════════════════════════════════════════════════════════════

class ChatOverlay {
  constructor(options = {}) {
    this.container = null;
    this.statsBar = null;
    this.isActive = false;
    this.messageQueue = [];
    this.currentTopic = options.topic || 'learning';
    this.currentPhase = options.phase || 'welcome';
    this.viewerCount = 847000 + Math.floor(Math.random() * 400000);
    this.countriesCount = 142 + Math.floor(Math.random() * 10);
    this.likesCount = 0;
    this.commentsCount = 0;
    
    // Comment timing
    this.minInterval = options.minInterval || 1500;
    this.maxInterval = options.maxInterval || 4000;
    
    // Phase-aware comment distribution
    this.phaseWeights = {
      welcome: { breakthrough: 0.3, kelly_love: 0.3, social: 0.3, reactions: 0.1 },
      question: { engagement: 0.3, choice_phase: 0.4, reactions: 0.2, breakthrough: 0.1 },
      wisdom: { wisdom_phase: 0.4, breakthrough: 0.3, kelly_love: 0.2, reactions: 0.1 },
    };
    
    this.init();
  }
  
  init() {
    // Create overlay container with TikTok-style design
    this.container = document.createElement('div');
    this.container.id = 'chat-overlay';
    this.container.innerHTML = `
      <style>
        #chat-overlay {
          position: fixed;
          bottom: 220px;
          left: 16px;
          width: 260px;
          max-height: 200px;
          pointer-events: none;
          z-index: 500;
          overflow: hidden;
          display: flex;
          flex-direction: column-reverse;
          gap: 6px;
        }
        
        @media (max-width: 768px) {
          #chat-overlay {
            width: 220px;
            bottom: 260px;
            left: 12px;
            max-height: 160px;
          }
        }
        
        @media (max-width: 375px) {
          #chat-overlay {
            width: 180px;
            max-height: 140px;
          }
        }
        
        /* TikTok-style comment bubble */
        .chat-comment {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          border-radius: 16px;
          animation: commentSlideIn 0.25s cubic-bezier(0.4, 0, 0.2, 1);
          max-width: fit-content;
          transform-origin: left center;
        }
        
        .chat-comment.fading {
          animation: commentFadeOut 0.4s ease-out forwards;
        }
        
        .chat-comment .avatar {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: linear-gradient(135deg, #3b82f6, #8b5cf6);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          flex-shrink: 0;
        }
        
        .chat-comment .content {
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 0;
        }
        
        .chat-comment .header {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .chat-comment .username {
          font-size: 12px;
          font-weight: 600;
          color: #fff;
          opacity: 0.9;
        }
        
        .chat-comment .verified {
          font-size: 10px;
        }
        
        .chat-comment .text {
          font-size: 13px;
          color: #fff;
          line-height: 1.35;
          word-wrap: break-word;
        }
        
        @keyframes commentSlideIn {
          from {
            opacity: 0;
            transform: translateY(10px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        
        @keyframes commentFadeOut {
          to {
            opacity: 0;
            transform: translateX(-20px);
          }
        }
        
        /* Live stats badge (TikTok style) */
        #live-badge {
          position: fixed;
          top: calc(env(safe-area-inset-top, 44px) + 60px);
          left: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          border-radius: 20px;
          z-index: 500;
        }
        
        @media (max-width: 768px) {
          #live-badge {
            left: 12px;
            padding: 5px 10px;
            gap: 6px;
          }
        }
        
        .live-dot {
          width: 8px;
          height: 8px;
          background: #ef4444;
          border-radius: 50%;
          animation: livePulse 1.5s ease-in-out infinite;
        }
        
        @keyframes livePulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.85); }
        }
        
        .live-text {
          font-size: 11px;
          font-weight: 700;
          color: #fff;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        
        .live-viewers {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.8);
        }
        
        /* Engagement counter (bottom right, like TikTok) */
        #engagement-counter {
          position: fixed;
          bottom: 220px;
          right: 70px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
          z-index: 500;
          pointer-events: none;
        }
        
        @media (max-width: 768px) {
          #engagement-counter {
            bottom: 260px;
            right: 60px;
            gap: 12px;
          }
        }
        
        .engagement-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
        }
        
        .engagement-icon {
          width: 40px;
          height: 40px;
          background: rgba(0, 0, 0, 0.4);
          backdrop-filter: blur(4px);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 20px;
        }
        
        .engagement-count {
          font-size: 11px;
          font-weight: 600;
          color: #fff;
        }
      </style>
    `;
    
    // Create live badge
    this.liveBadge = document.createElement('div');
    this.liveBadge.id = 'live-badge';
    this.updateLiveBadge();
    
    document.body.appendChild(this.container);
    document.body.appendChild(this.liveBadge);
  }
  
  start() {
    if (this.isActive) return;
    this.isActive = true;
    
    // Initial burst of comments (feels like joining a live stream)
    setTimeout(() => this.addComment(), 500);
    setTimeout(() => this.addComment(), 1200);
    setTimeout(() => this.addComment(), 2000);
    
    // Continue with regular schedule
    this.scheduleNext();
    
    // Update stats periodically
    this.statsInterval = setInterval(() => this.updateLiveBadge(), 3000);
    
    console.log('[ChatOverlay v2] 🚀 Started - TikTok-style social learning');
  }
  
  stop() {
    this.isActive = false;
    if (this.statsInterval) clearInterval(this.statsInterval);
    console.log('[ChatOverlay v2] Stopped');
  }
  
  setPhase(phase) {
    this.currentPhase = phase;
    console.log(`[ChatOverlay] Phase changed to: ${phase}`);
  }
  
  setTopic(topic) {
    this.currentTopic = topic;
    console.log(`[ChatOverlay] Topic set to: ${topic}`);
  }
  
  scheduleNext() {
    if (!this.isActive) return;
    
    const delay = this.minInterval + Math.random() * (this.maxInterval - this.minInterval);
    setTimeout(() => {
      this.addComment();
      this.scheduleNext();
    }, delay);
  }
  
  selectCommentBank() {
    const weights = this.phaseWeights[this.currentPhase] || this.phaseWeights.welcome;
    const random = Math.random();
    let cumulative = 0;
    
    for (const [bank, weight] of Object.entries(weights)) {
      cumulative += weight;
      if (random <= cumulative) {
        return COMMENT_BANKS[bank] || COMMENT_BANKS.reactions;
      }
    }
    
    return COMMENT_BANKS.reactions;
  }
  
  getRandomComment() {
    const bank = this.selectCommentBank();
    const comment = bank[Math.floor(Math.random() * bank.length)];
    
    // If it's a topic-specific template, fill it in
    if (comment.text.includes('{topic}')) {
      return {
        ...comment,
        text: comment.text.replace('{topic}', this.currentTopic),
        user: GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)].name,
        flag: GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)].flag,
      };
    }
    
    // Randomize flag for reaction-only comments
    if (comment.user === 'User') {
      const person = GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)];
      return { ...comment, user: person.name, flag: person.flag };
    }
    
    return comment;
  }
  
  addComment() {
    const comment = this.getRandomComment();
    this.commentsCount++;
    
    const el = document.createElement('div');
    el.className = 'chat-comment';
    el.innerHTML = `
      <div class="avatar">${comment.flag}</div>
      <div class="content">
        <div class="header">
          <span class="username">${comment.user}</span>
          ${comment.verified ? '<span class="verified">✓</span>' : ''}
        </div>
        <span class="text">${comment.text}</span>
      </div>
    `;
    
    // Add to top (newest at bottom, flexbox column-reverse)
    this.container.appendChild(el);
    
    // Fade out after 6 seconds
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), 400);
    }, 6000);
    
    // Keep max 5 comments visible
    while (this.container.querySelectorAll('.chat-comment:not(.fading)').length > 5) {
      const oldest = this.container.querySelector('.chat-comment:not(.fading)');
      if (oldest) {
        oldest.classList.add('fading');
        setTimeout(() => oldest.remove(), 400);
      }
    }
    
    // Random like increments
    if (Math.random() > 0.3) {
      this.likesCount += Math.floor(Math.random() * 50) + 10;
    }
  }
  
  // Add a specific comment (for phase-triggered comments)
  addSpecificComment(text, options = {}) {
    const person = GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)];
    
    const el = document.createElement('div');
    el.className = 'chat-comment';
    el.innerHTML = `
      <div class="avatar">${options.flag || person.flag}</div>
      <div class="content">
        <div class="header">
          <span class="username">${options.user || person.name}</span>
        </div>
        <span class="text">${text}</span>
      </div>
    `;
    
    this.container.appendChild(el);
    
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), 400);
    }, 6000);
  }
  
  updateLiveBadge() {
    // Simulate realistic viewer fluctuations
    this.viewerCount += Math.floor(Math.random() * 1000) - 400;
    this.viewerCount = Math.max(750000, this.viewerCount);
    
    const viewers = this.viewerCount > 1000000 
      ? (this.viewerCount / 1000000).toFixed(1) + 'M'
      : (this.viewerCount / 1000).toFixed(0) + 'K';
    
    this.liveBadge.innerHTML = `
      <div class="live-dot"></div>
      <span class="live-text">LIVE</span>
      <span class="live-viewers">${viewers} learning</span>
    `;
  }
  
  // Trigger a burst of reactions (e.g., after a big reveal)
  triggerReactionBurst(count = 5) {
    for (let i = 0; i < count; i++) {
      setTimeout(() => {
        const reaction = COMMENT_BANKS.reactions[Math.floor(Math.random() * COMMENT_BANKS.reactions.length)];
        const person = GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)];
        this.addSpecificComment(reaction.text, { user: person.name, flag: person.flag });
      }, i * 300);
    }
  }
  
  // Trigger breakthrough comments (after a learner "gets it")
  triggerBreakthrough() {
    const breakthrough = COMMENT_BANKS.breakthrough[Math.floor(Math.random() * COMMENT_BANKS.breakthrough.length)];
    this.addSpecificComment(breakthrough.text, { user: breakthrough.user, flag: breakthrough.flag });
  }
  
  destroy() {
    this.stop();
    if (this.container) this.container.remove();
    if (this.liveBadge) this.liveBadge.remove();
  }
}

// ═══════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════

window.ChatOverlay = ChatOverlay;
window.COMMENT_BANKS = COMMENT_BANKS;
