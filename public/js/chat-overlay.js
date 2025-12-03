/**
 * Chat Overlay v3 - TikTok-Style Social Learning with Trust & Safety Disclosure
 * 
 * TRUST & SAFETY DISCLOSURE:
 * All comments shown in this overlay are AI-generated simulated social content.
 * They are designed to provide the psychological benefits of social learning
 * (belonging, normalized struggle, shared experience) while being fully transparent.
 * 
 * Every comment is marked with ✨ to indicate it's simulated.
 * Users can disable this feature in Settings.
 * 
 * Why we simulate social learning:
 * - Humans are inherently social learners
 * - Social media hijacked this need with harmful, addictive patterns
 * - Kelly provides the social mirror learners need—safely and predictably
 * 
 * How we're different from social media:
 * - No variable rewards (predictable, not addictive)
 * - No engagement optimization (learning-focused)
 * - Full disclosure (every comment marked ✨)
 * - Full user control (can be disabled)
 * 
 * See: /trust for full Trust & Safety documentation
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
    this.currentLessonDay = options.lessonDay || null;
    this.viewerCount = 847000 + Math.floor(Math.random() * 400000);
    this.countriesCount = 142 + Math.floor(Math.random() * 10);
    this.likesCount = 0;
    this.commentsCount = 0;
    
    // Supabase integration
    this.supabase = options.supabase || window.supabase?.createClient?.(
      window.CONFIG?.SUPABASE_URL,
      window.CONFIG?.SUPABASE_ANON_KEY
    ) || null;
    this.dbComments = new Map(); // Cache: phase -> comments[]
    this.useDatabase = options.useDatabase !== false; // Default true
    
    // Comment timing
    this.minInterval = options.minInterval || 1500;
    this.maxInterval = options.maxInterval || 4000;
    
    // Phase-aware comment distribution
    this.phaseWeights = {
      welcome: { breakthrough: 0.3, kelly_love: 0.3, social: 0.3, reactions: 0.1 },
      question: { engagement: 0.3, choice_phase: 0.4, reactions: 0.2, breakthrough: 0.1 },
      wisdom: { wisdom_phase: 0.4, breakthrough: 0.3, kelly_love: 0.2, reactions: 0.1 },
      // Map new phase IDs to weights
      q1: { engagement: 0.3, choice_phase: 0.4, reactions: 0.2, breakthrough: 0.1 },
      q2: { engagement: 0.3, choice_phase: 0.4, reactions: 0.2, breakthrough: 0.1 },
      q3: { engagement: 0.3, choice_phase: 0.4, reactions: 0.2, breakthrough: 0.1 },
      hook: { wisdom_phase: 0.4, breakthrough: 0.3, kelly_love: 0.2, reactions: 0.1 },
      complete: { kelly_love: 0.4, social: 0.3, reactions: 0.3 },
    };
    
    this.init();
  }
  
  init() {
    // Check user preference for simulated content
    this.simulatedEnabled = this.getSimulatedContentPref();
    
    // Create overlay container with TikTok-style design + Trust & Safety disclosure
    this.container = document.createElement('div');
    this.container.id = 'chat-overlay';
    this.container.innerHTML = `
      <style>
        #chat-overlay {
          position: fixed;
          bottom: 220px;
          left: 16px;
          width: 280px;
          max-height: 200px;
          pointer-events: none;
          z-index: 500;
          overflow: hidden;
          display: flex;
          flex-direction: column-reverse;
          gap: 6px;
        }
        
        #chat-overlay.simulated-hidden {
          display: none !important;
        }
        
        @media (max-width: 768px) {
          #chat-overlay {
            width: 240px;
            bottom: 260px;
            left: 12px;
            max-height: 160px;
          }
        }
        
        @media (max-width: 375px) {
          #chat-overlay {
            width: 200px;
            max-height: 140px;
          }
        }
        
        /* TikTok-style comment bubble with ✨ disclosure */
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
          position: relative;
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
        
        /* ✨ TRUST & SAFETY: Simulated content indicator */
        .chat-comment .simulated-indicator {
          font-size: 10px;
          opacity: 0.7;
          margin-left: 2px;
          cursor: help;
          pointer-events: auto;
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
        
        /* Live stats badge - UPDATED: Shows "Simulated" disclosure */
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
          pointer-events: auto;
          cursor: pointer;
        }
        
        #live-badge.simulated-hidden {
          display: none !important;
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
          background: #f59e0b; /* Amber for "simulated" vs red for "live" */
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
        
        .simulated-label {
          font-size: 9px;
          color: rgba(255, 255, 255, 0.6);
          margin-left: 4px;
        }
        
        /* Disclosure tooltip */
        #simulated-tooltip {
          position: fixed;
          top: calc(env(safe-area-inset-top, 44px) + 100px);
          left: 16px;
          background: rgba(0, 0, 0, 0.9);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 16px;
          width: 280px;
          z-index: 600;
          display: none;
          pointer-events: auto;
        }
        
        #simulated-tooltip.show {
          display: block;
          animation: tooltipFadeIn 0.2s ease-out;
        }
        
        @keyframes tooltipFadeIn {
          from { opacity: 0; transform: translateY(-8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        #simulated-tooltip h4 {
          color: #f59e0b;
          font-size: 14px;
          font-weight: 600;
          margin: 0 0 8px 0;
        }
        
        #simulated-tooltip p {
          color: rgba(255, 255, 255, 0.85);
          font-size: 13px;
          line-height: 1.5;
          margin: 0 0 12px 0;
        }
        
        #simulated-tooltip .tooltip-actions {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        
        #simulated-tooltip .tooltip-btn {
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 12px;
          font-weight: 500;
          cursor: pointer;
          border: none;
          transition: all 0.2s;
        }
        
        #simulated-tooltip .tooltip-btn.primary {
          background: #3b82f6;
          color: white;
        }
        
        #simulated-tooltip .tooltip-btn.secondary {
          background: rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.9);
        }
        
        #simulated-tooltip .tooltip-btn:hover {
          transform: scale(1.02);
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
    
    // Create live badge with simulated disclosure
    this.liveBadge = document.createElement('div');
    this.liveBadge.id = 'live-badge';
    this.updateLiveBadge();
    
    // Create disclosure tooltip
    this.tooltip = document.createElement('div');
    this.tooltip.id = 'simulated-tooltip';
    this.tooltip.innerHTML = `
      <h4>✨ Simulated Learning Community</h4>
      <p>These comments are AI-generated to create a supportive social learning experience. They're designed to make you feel less alone while learning—without the harmful effects of social media.</p>
      <p style="font-size: 11px; color: rgba(255,255,255,0.6); margin-bottom: 12px;">Every comment is marked with ✨</p>
      <div class="tooltip-actions">
        <button class="tooltip-btn primary" onclick="window.chatOverlay?.hideTooltip()">Got it</button>
        <button class="tooltip-btn secondary" onclick="window.chatOverlay?.toggleSimulated()">Turn off</button>
        <a href="/trust" class="tooltip-btn secondary" style="text-decoration: none;">Learn more</a>
      </div>
    `;
    
    document.body.appendChild(this.container);
    document.body.appendChild(this.liveBadge);
    document.body.appendChild(this.tooltip);
    
    // Click handler for disclosure
    this.liveBadge.addEventListener('click', () => this.showTooltip());
    
    // Close tooltip on outside click
    document.addEventListener('click', (e) => {
      if (!this.tooltip.contains(e.target) && !this.liveBadge.contains(e.target)) {
        this.hideTooltip();
      }
    });
    
    // Apply user preference
    if (!this.simulatedEnabled) {
      this.container.classList.add('simulated-hidden');
      this.liveBadge.classList.add('simulated-hidden');
    }
  }
  
  // ═══════════════════════════════════════════════════════════════════
  // TRUST & SAFETY: User Preference Management
  // ═══════════════════════════════════════════════════════════════════
  
  getSimulatedContentPref() {
    try {
      const prefs = JSON.parse(localStorage.getItem('kellySimulatedContentPrefs') || '{}');
      return prefs.enabled !== false; // Default to true
    } catch {
      return true;
    }
  }
  
  setSimulatedContentPref(enabled) {
    try {
      const prefs = JSON.parse(localStorage.getItem('kellySimulatedContentPrefs') || '{}');
      prefs.enabled = enabled;
      localStorage.setItem('kellySimulatedContentPrefs', JSON.stringify(prefs));
    } catch (e) {
      console.warn('[ChatOverlay] Could not save preference:', e);
    }
  }
  
  toggleSimulated() {
    this.simulatedEnabled = !this.simulatedEnabled;
    this.setSimulatedContentPref(this.simulatedEnabled);
    
    if (this.simulatedEnabled) {
      this.container.classList.remove('simulated-hidden');
      this.liveBadge.classList.remove('simulated-hidden');
      console.log('[ChatOverlay] Simulated content enabled');
    } else {
      this.container.classList.add('simulated-hidden');
      this.liveBadge.classList.add('simulated-hidden');
      console.log('[ChatOverlay] Simulated content disabled by user');
    }
    
    this.hideTooltip();
  }
  
  showTooltip() {
    this.tooltip.classList.add('show');
  }
  
  hideTooltip() {
    this.tooltip.classList.remove('show');
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
    
    // Preload comments from database for this phase
    if (this.useDatabase && this.currentLessonDay) {
      this.loadCommentsFromDB(this.currentLessonDay, phase);
    }
  }
  
  setTopic(topic) {
    this.currentTopic = topic;
    console.log(`[ChatOverlay] Topic set to: ${topic}`);
  }
  
  setLessonDay(dayNumber) {
    this.currentLessonDay = dayNumber;
    console.log(`[ChatOverlay] Lesson day set to: ${dayNumber}`);
    // Clear cache when lesson changes
    this.dbComments.clear();
  }
  
  // ═══════════════════════════════════════════════════════════════════
  // SUPABASE INTEGRATION
  // ═══════════════════════════════════════════════════════════════════
  
  async loadCommentsFromDB(lessonDay, phase) {
    if (!this.supabase || !lessonDay) return;
    
    const cacheKey = `${lessonDay}-${phase}`;
    if (this.dbComments.has(cacheKey)) {
      return this.dbComments.get(cacheKey);
    }
    
    try {
      const { data, error } = await this.supabase
        .from('lesson_comments')
        .select('*')
        .eq('lesson_day', lessonDay)
        .eq('phase', phase)
        .is('option_context', null);
      
      if (error) {
        console.warn('[ChatOverlay] DB fetch error:', error.message);
        return null;
      }
      
      if (data && data.length > 0) {
        this.dbComments.set(cacheKey, data);
        console.log(`[ChatOverlay] Loaded ${data.length} comments from DB for ${phase}`);
        return data;
      }
    } catch (e) {
      console.warn('[ChatOverlay] DB error:', e.message);
    }
    
    return null;
  }
  
  async loadOptionComments(lessonDay, phase, optionLetter) {
    if (!this.supabase || !lessonDay) return null;
    
    const cacheKey = `${lessonDay}-${phase}-${optionLetter}`;
    if (this.dbComments.has(cacheKey)) {
      return this.dbComments.get(cacheKey);
    }
    
    try {
      const { data, error } = await this.supabase
        .from('lesson_comments')
        .select('*')
        .eq('lesson_day', lessonDay)
        .eq('phase', phase)
        .eq('option_context', optionLetter);
      
      if (!error && data && data.length > 0) {
        this.dbComments.set(cacheKey, data);
        return data;
      }
    } catch (e) {
      console.warn('[ChatOverlay] Option comment fetch error:', e.message);
    }
    
    return null;
  }
  
  getDBComment() {
    // Try to get a comment from the database cache
    const cacheKey = `${this.currentLessonDay}-${this.currentPhase}`;
    const dbComments = this.dbComments.get(cacheKey);
    
    if (dbComments && dbComments.length > 0) {
      const comment = dbComments[Math.floor(Math.random() * dbComments.length)];
      return {
        user: comment.persona_name,
        flag: comment.persona_flag,
        text: comment.comment_text,
        verified: false
      };
    }
    
    return null;
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
    // Try database first (50% of the time if available)
    if (this.useDatabase && this.currentLessonDay && Math.random() > 0.5) {
      const dbComment = this.getDBComment();
      if (dbComment) {
        return dbComment;
      }
    }
    
    // Fallback to hardcoded banks
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
    if (!this.simulatedEnabled) return; // Respect user preference
    
    const comment = this.getRandomComment();
    this.commentsCount++;
    
    const el = document.createElement('div');
    el.className = 'chat-comment';
    el.setAttribute('data-simulated', 'true');
    el.setAttribute('aria-label', `Simulated learner ${comment.user} says: ${comment.text}`);
    el.innerHTML = `
      <div class="avatar">${comment.flag}</div>
      <div class="content">
        <div class="header">
          <span class="username">${comment.user}</span>
          <span class="simulated-indicator" title="Simulated learner - tap badge above to learn more">✨</span>
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
    if (!this.simulatedEnabled) return; // Respect user preference
    
    const person = GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)];
    const userName = options.user || person.name;
    
    const el = document.createElement('div');
    el.className = 'chat-comment';
    el.setAttribute('data-simulated', 'true');
    el.setAttribute('aria-label', `Simulated learner ${userName} says: ${text}`);
    el.innerHTML = `
      <div class="avatar">${options.flag || person.flag}</div>
      <div class="content">
        <div class="header">
          <span class="username">${userName}</span>
          <span class="simulated-indicator" title="Simulated learner - tap badge above to learn more">✨</span>
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
    // TRUST & SAFETY: Updated to show "Simulated" not "LIVE"
    // We don't show fake viewer counts - that would be deceptive
    // Instead, we show this is a simulated social experience
    
    this.liveBadge.innerHTML = `
      <div class="live-dot"></div>
      <span class="live-text">✨ Social</span>
      <span class="live-viewers">Tap to learn more</span>
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
  
  // Show option-specific comments when user hovers/selects an option
  async showOptionComments(optionLetter) {
    if (!this.useDatabase || !this.currentLessonDay) {
      // Fallback to choice_phase comments
      const comment = COMMENT_BANKS.choice_phase[Math.floor(Math.random() * COMMENT_BANKS.choice_phase.length)];
      this.addSpecificComment(comment.text, { user: comment.user, flag: comment.flag });
      return;
    }
    
    const comments = await this.loadOptionComments(
      this.currentLessonDay, 
      this.currentPhase, 
      optionLetter.toUpperCase()
    );
    
    if (comments && comments.length > 0) {
      const comment = comments[Math.floor(Math.random() * comments.length)];
      this.addSpecificComment(comment.comment_text, {
        user: comment.persona_name,
        flag: comment.persona_flag
      });
    } else {
      // Fallback
      const fallback = COMMENT_BANKS.choice_phase[Math.floor(Math.random() * COMMENT_BANKS.choice_phase.length)];
      this.addSpecificComment(fallback.text, { user: fallback.user, flag: fallback.flag });
    }
  }
  
  // Trigger completion celebration comments
  triggerCompletion() {
    const count = 4;
    for (let i = 0; i < count; i++) {
      setTimeout(() => {
        const dbComment = this.getDBComment();
        if (dbComment) {
          this.addSpecificComment(dbComment.text, { user: dbComment.user, flag: dbComment.flag });
        } else {
          const comment = COMMENT_BANKS.social[Math.floor(Math.random() * COMMENT_BANKS.social.length)];
          this.addSpecificComment(comment.text, { user: comment.user, flag: comment.flag });
        }
      }, i * 400);
    }
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
