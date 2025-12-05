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
// COMMENT BANKS v4 - Humble, Growth-Mindset, No Hyperbole
// ═══════════════════════════════════════════════════════════════════
// Philosophy: "The social experience is simulated. The learning is real."
// NO: "Mind blown", "Best ever", "I'm addicted"
// YES: "I had to replay that", "Not sure I follow", "Good takeaway"

const COMMENT_BANKS = {
  // 💡 BREAKTHROUGH - Genuine understanding (not hyperbolic)
  breakthrough: [
    { user: "Maria", flag: "🇧🇷", text: "That makes sense now", verified: false },
    { user: "James", flag: "🇬🇧", text: "I see the connection", verified: false },
    { user: "Yuki", flag: "🇯🇵", text: "I get it now", verified: false },
    { user: "Ahmed", flag: "🇪🇬", text: "This clicked for me", verified: false },
    { user: "Sofia", flag: "🇲🇽", text: "I understand the pattern", verified: false },
    { user: "Priya", flag: "🇮🇳", text: "Good explanation", verified: false },
    { user: "Hans", flag: "🇩🇪", text: "Now I see why", verified: false },
    { user: "Nina", flag: "🇳🇴", text: "That clarifies things", verified: false },
    { user: "Chen Wei", flag: "🇨🇳", text: "This helps", verified: false },
    { user: "Isabella", flag: "🇮🇹", text: "Makes more sense now", verified: false },
  ],

  // 🤔 ENGAGEMENT - Thoughtful questions (normalizes asking)
  engagement: [
    { user: "Emma", flag: "🇺🇸", text: "Wait, can you explain that again?", verified: false },
    { user: "Lucas", flag: "🇫🇷", text: "But what about when...?", verified: false },
    { user: "Kofi", flag: "🇬🇭", text: "I'm not sure I follow", verified: false },
    { user: "Aisha", flag: "🇰🇪", text: "I never thought of it that way", verified: false },
    { user: "Diego", flag: "🇨🇱", text: "This connects to yesterday's lesson", verified: false },
    { user: "Mei", flag: "🇹🇼", text: "Is that always true?", verified: false },
    { user: "Omar", flag: "🇦🇪", text: "Let me think about this", verified: false },
    { user: "Zara", flag: "🇿🇦", text: "Interesting approach", verified: false },
  ],

  // 👍 APPRECIATION - Humble, genuine (no superlatives)
  kelly_love: [
    { user: "Sarah", flag: "🇨🇦", text: "Good lesson today", verified: false },
    { user: "Jin", flag: "🇰🇷", text: "Clear explanation", verified: false },
    { user: "Anya", flag: "🇷🇺", text: "This is helpful", verified: false },
    { user: "Carlos", flag: "🇦🇷", text: "I appreciate this format", verified: false },
    { user: "Lena", flag: "🇸🇪", text: "Thanks for the lesson", verified: false },
    { user: "Raj", flag: "🇮🇳", text: "Easy to follow", verified: false },
    { user: "Fatima", flag: "🇲🇦", text: "Good pace", verified: false },
    { user: "Tomoko", flag: "🇯🇵", text: "Nice teaching style", verified: false },
    { user: "Sven", flag: "🇩🇰", text: "Well structured", verified: false },
    { user: "Ana", flag: "🇵🇹", text: "Learned something new", verified: false },
  ],

  // 👨‍👩‍👧 SOCIAL - Learning together (authentic)
  social: [
    { user: "Michael", flag: "🇺🇸", text: "Watching with my daughter", verified: true },
    { user: "Lisa", flag: "🇦🇺", text: "Family learning time", verified: false },
    { user: "Kenji", flag: "🇯🇵", text: "Morning routine ☕", verified: false },
    { user: "Maria", flag: "🇪🇸", text: "Here with my kids", verified: false },
    { user: "David", flag: "🇮🇱", text: "Study group here 👋", verified: false },
    { user: "Olga", flag: "🇺🇦", text: "Hello from Kyiv", verified: false },
    { user: "Adebayo", flag: "🇳🇬", text: "Lagos checking in", verified: false },
    { user: "Camila", flag: "🇧🇷", text: "Good morning from Brazil", verified: false },
    { user: "Pierre", flag: "🇫🇷", text: "Bonjour", verified: false },
    { user: "Ling", flag: "🇸🇬", text: "Singapore here", verified: false },
  ],

  // 💬 REACTIONS - Simple, authentic
  reactions: [
    { user: "User", flag: "🇺🇸", text: "👍", verified: false },
    { user: "User", flag: "🇬🇧", text: "Interesting", verified: false },
    { user: "User", flag: "🇯🇵", text: "📝", verified: false },
    { user: "User", flag: "🇧🇷", text: "True", verified: false },
    { user: "User", flag: "🇮🇳", text: "Good point", verified: false },
    { user: "User", flag: "🇩🇪", text: "💡", verified: false },
    { user: "User", flag: "🇫🇷", text: "Hmm", verified: false },
    { user: "User", flag: "🇪🇸", text: "I see", verified: false },
    { user: "User", flag: "🇲🇽", text: "Noted", verified: false },
    { user: "User", flag: "🇨🇦", text: "Makes sense", verified: false },
  ],

  // 📝 CHOICE PHASE - Honest uncertainty (normalizes not knowing)
  choice_phase: [
    { user: "Alex", flag: "🇺🇸", text: "I think A", verified: false },
    { user: "Sophie", flag: "🇫🇷", text: "Going with B", verified: false },
    { user: "Hiroshi", flag: "🇯🇵", text: "This one's tricky", verified: false },
    { user: "Eva", flag: "🇩🇪", text: "Not sure about this", verified: false },
    { user: "Marco", flag: "🇮🇹", text: "Changed my answer", verified: false },
    { user: "Aaliya", flag: "🇵🇰", text: "Going with my gut", verified: false },
    { user: "Tom", flag: "🇬🇧", text: "I'm unsure", verified: false },
    { user: "Maya", flag: "🇮🇳", text: "Both seem possible", verified: false },
  ],

  // 🎓 WISDOM PHASE - Thoughtful, not hyperbolic
  wisdom_phase: [
    { user: "Rebecca", flag: "🇺🇸", text: "Good takeaway", verified: false },
    { user: "Takeshi", flag: "🇯🇵", text: "I'll remember that", verified: false },
    { user: "Elena", flag: "🇷🇺", text: "Worth thinking about", verified: false },
    { user: "Paulo", flag: "🇧🇷", text: "Helpful insight", verified: false },
    { user: "Kim", flag: "🇰🇷", text: "Adding to my notes", verified: false },
    { user: "Fatou", flag: "🇸🇳", text: "Makes me think", verified: false },
    { user: "Henrik", flag: "🇳🇴", text: "Good ending", verified: false },
    { user: "Nadia", flag: "🇲🇦", text: "I'll share this", verified: false },
  ],

  // 🤔 STRUGGLE - Normalize confusion (IMPORTANT for growth mindset)
  struggle: [
    { user: "Emma", flag: "🇺🇸", text: "I had to replay that", verified: false },
    { user: "Lucas", flag: "🇫🇷", text: "Still processing", verified: false },
    { user: "Priya", flag: "🇮🇳", text: "This is new to me", verified: false },
    { user: "Hans", flag: "🇩🇪", text: "I don't fully get it yet", verified: false },
    { user: "Yuki", flag: "🇯🇵", text: "Confused but curious", verified: false },
    { user: "Ahmed", flag: "🇪🇬", text: "Need to think about this", verified: false },
    { user: "Sofia", flag: "🇲🇽", text: "Can someone explain?", verified: false },
    { user: "Jin", flag: "🇰🇷", text: "Third time watching", verified: false },
  ],

  // 📚 TOPIC-SPECIFIC TEMPLATES (humble versions)
  topic_specific: [
    { user: "Student", flag: "🌍", text: "Interesting topic today", verified: false },
    { user: "Learner", flag: "🌍", text: "Learning about {topic}", verified: false },
    { user: "Curious", flag: "🌍", text: "{topic} is interesting", verified: false },
    { user: "Mind", flag: "🌍", text: "Good lesson on {topic}", verified: false },
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
    
    // Comment timing - More frequent for lively feel
    this.minInterval = options.minInterval || 800;  // Was 1500
    this.maxInterval = options.maxInterval || 2500; // Was 4000
    
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
    
    // V3: Try to use existing HTML container first
    this.container = document.getElementById('live-comments');
    if (this.container) {
      console.log('[ChatOverlay] V3: Using existing #live-comments container from HTML');
      // Skip creating old-style container, just setup badge and tooltip
      this.setupBadgeAndTooltip();
      return;
    }
    
    // Fallback: Create overlay container with TikTok-style design + Trust & Safety disclosure
    console.log('[ChatOverlay] Fallback: Creating #chat-overlay container');
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
  // V3: Simplified setup when using existing HTML containers
  // ═══════════════════════════════════════════════════════════════════
  
  setupBadgeAndTooltip() {
    // Use existing live badge from HTML (#live-badge-v2) if available
    this.liveBadge = document.getElementById('live-badge-v2');
    if (!this.liveBadge) {
      // Fallback badge creation
      this.liveBadge = document.createElement('div');
      this.liveBadge.id = 'live-badge';
      this.liveBadge.className = 'live-badge-v2';
      this.liveBadge.innerHTML = `
        <span class="live-dot" style="background: #f59e0b;"></span>
        <span class="live-text">✨ Social</span>
      `;
      this.liveBadge.style.cssText = 'position:fixed;top:8px;left:50%;transform:translateX(-50%);background:rgba(239,68,68,0.9);color:white;padding:4px 12px;border-radius:16px;font-size:12px;font-weight:600;z-index:100;display:flex;align-items:center;gap:6px;cursor:pointer;';
      document.body.appendChild(this.liveBadge);
    }
    
    // Create tooltip for disclosure
    this.tooltip = document.getElementById('simulated-tooltip');
    if (!this.tooltip) {
      this.tooltip = document.createElement('div');
      this.tooltip.id = 'simulated-tooltip';
      this.tooltip.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,0.95);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:20px;width:min(320px,90vw);z-index:1000;display:none;';
      this.tooltip.innerHTML = `
        <h4 style="color:#f59e0b;font-size:16px;font-weight:600;margin:0 0 12px 0;">✨ Simulated Learning Community</h4>
        <p style="color:rgba(255,255,255,0.85);font-size:14px;line-height:1.5;margin:0 0 16px 0;">These comments are AI-generated to create a supportive social learning experience. They help you feel less alone while learning—without the harmful effects of social media.</p>
        <p style="font-size:12px;color:rgba(255,255,255,0.6);margin-bottom:16px;">Every simulated comment is marked with ✨</p>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button onclick="window.chatOverlay?.hideTooltip()" style="padding:10px 16px;border-radius:10px;font-size:13px;font-weight:500;cursor:pointer;border:none;background:#3b82f6;color:white;">Got it</button>
          <button onclick="window.chatOverlay?.toggleSimulated()" style="padding:10px 16px;border-radius:10px;font-size:13px;font-weight:500;cursor:pointer;border:none;background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.9);">Turn off</button>
          <a href="/trust" style="padding:10px 16px;border-radius:10px;font-size:13px;font-weight:500;cursor:pointer;border:none;background:rgba(255,255,255,0.1);color:rgba(255,255,255,0.9);text-decoration:none;">Learn more</a>
        </div>
      `;
      document.body.appendChild(this.tooltip);
    }
    
    // Click handler for disclosure
    if (this.liveBadge) {
      this.liveBadge.style.cursor = 'pointer';
      this.liveBadge.addEventListener('click', () => this.showTooltip());
    }
    
    // Close tooltip on outside click
    document.addEventListener('click', (e) => {
      if (this.tooltip && !this.tooltip.contains(e.target) && 
          this.liveBadge && !this.liveBadge.contains(e.target)) {
        this.hideTooltip();
      }
    });
    
    // Apply user preference
    if (!this.simulatedEnabled) {
      if (this.container) this.container.classList.add('simulated-hidden');
      if (this.liveBadge) this.liveBadge.classList.add('simulated-hidden');
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
    if (!this.simulatedEnabled) return;
    
    const comment = this.getRandomComment();
    this.commentsCount++;
    
    // Use the new minimal live-comments container if available
    const container = document.getElementById('live-comments') || this.container;
    
    const el = document.createElement('div');
    el.className = 'live-comment';
    el.setAttribute('data-simulated', 'true');
    el.innerHTML = `
      <span class="comment-user">${comment.user}</span>
      <span class="comment-badge">✨</span>
      <span class="comment-text">${comment.text}</span>
    `;
    
    container.appendChild(el);
    
    // Fade out after 8 seconds (longer visible time)
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), 500);
    }, 8000);
    
    // Keep max 10 comments visible (more lively)
    const comments = container.querySelectorAll('.live-comment:not(.fading)');
    if (comments.length > 10) {
      const oldest = comments[0];
      oldest.classList.add('fading');
      setTimeout(() => oldest.remove(), 500);
    }
  }
  
  // Add a specific comment (for phase-triggered comments)
  addSpecificComment(text, options = {}) {
    if (!this.simulatedEnabled) return;
    
    const person = GLOBAL_NAMES[Math.floor(Math.random() * GLOBAL_NAMES.length)];
    const userName = options.user || person.name;
    
    const container = document.getElementById('live-comments') || this.container;
    
    const el = document.createElement('div');
    el.className = 'live-comment';
    el.setAttribute('data-simulated', 'true');
    el.innerHTML = `
      <span class="comment-user">${userName}</span>
      <span class="comment-badge">✨</span>
      <span class="comment-text">${text}</span>
    `;
    
    container.appendChild(el);
    
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), 500);
    }, 5000);
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
  
  // ═══════════════════════════════════════════════════════════════════
  // KELLY HOST MESSAGES - Kelly participates as the host
  // ═══════════════════════════════════════════════════════════════════
  
  /**
   * Add a Kelly host message to the chat
   * Kelly appears inline with special styling
   */
  addKellyMessage(text, options = {}) {
    if (!this.simulatedEnabled) return;
    
    const container = document.getElementById('live-comments') || this.container;
    
    const el = document.createElement('div');
    el.className = 'live-comment kelly-comment';
    el.setAttribute('aria-label', `Kelly (Host) says: ${text}`);
    el.innerHTML = `
      <span class="comment-user">Kelly</span>
      <span class="host-badge">HOST</span>
      <span class="comment-text">${text}</span>
    `;
    
    container.appendChild(el);
    
    // Kelly messages stay longer (10 seconds)
    const duration = options.duration || 10000;
    setTimeout(() => {
      el.classList.add('fading');
      setTimeout(() => el.remove(), 500);
    }, duration);
  }
  
  /**
   * Kelly's contextual messages for each phase
   */
  getKellyPhaseMessage(phase, topic) {
    const messages = {
      welcome: [
        `Welcome everyone! Today we're exploring ${topic}. Let's learn together 💙`,
        `Great to see you all! Ready to dive into ${topic}?`,
        `Hello, learners! Today's topic: ${topic}. Let's go!`,
      ],
      hook: [
        `Here's something interesting about ${topic}...`,
        `Think about this for a moment...`,
        `Let me explain why this matters...`,
      ],
      q1: [
        `Take your time with this question`,
        `What do you think? There's no wrong answer for learning`,
        `Think about what we just discussed...`,
      ],
      q2: [
        `This one builds on the first question`,
        `Consider the connection here...`,
        `Good thinking, everyone!`,
      ],
      q3: [
        `Last question — let's bring it together`,
        `Think about the bigger picture here`,
        `You've got this!`,
      ],
      wisdom: [
        `Here's the key takeaway...`,
        `This is what I want you to remember`,
        `Let this sink in for a moment`,
      ],
      complete: [
        `Great work today, everyone! See you tomorrow 🌟`,
        `You did it! Another day of learning complete`,
        `Thanks for learning with me today 💙`,
      ],
    };
    
    const phaseMessages = messages[phase] || messages.welcome;
    return phaseMessages[Math.floor(Math.random() * phaseMessages.length)];
  }
  
  /**
   * Trigger Kelly's phase-specific message
   */
  triggerKellyPhaseMessage(phase, topic) {
    const message = this.getKellyPhaseMessage(phase, topic || this.currentTopic);
    this.addKellyMessage(message);
  }
  
  /**
   * Kelly responds to a learner's comment
   */
  kellyRespondTo(learnerComment, responseType = 'acknowledge') {
    const responses = {
      acknowledge: [
        `Great observation!`,
        `I see what you mean`,
        `Good thinking!`,
        `That's a thoughtful point`,
      ],
      encourage: [
        `You're on the right track!`,
        `Keep going, you've got this`,
        `That's the spirit!`,
      ],
      clarify: [
        `Let me explain that a bit more...`,
        `Good question — here's the key...`,
        `Think of it this way...`,
      ],
    };
    
    const pool = responses[responseType] || responses.acknowledge;
    const message = pool[Math.floor(Math.random() * pool.length)];
    
    // Delay Kelly's response slightly
    setTimeout(() => {
      this.addKellyMessage(message, { duration: 8000 });
    }, 1500);
  }
}

// ═══════════════════════════════════════════════════════════════════
// KELLY HOST COMMENTS - Pre-defined contextual messages
// ═══════════════════════════════════════════════════════════════════

const KELLY_COMMENTS = {
  // Welcome/intro phase
  welcome: [
    "Welcome to today's lesson! 💙",
    "Great to see everyone here",
    "Let's learn something new together",
    "Ready? Let's dive in!",
  ],
  // During questions
  thinking: [
    "Take your time with this one",
    "No pressure — think it through",
    "What's your gut telling you?",
    "Trust your instincts here",
  ],
  // Encouragement
  encourage: [
    "You're doing great!",
    "Good thinking, everyone",
    "I love seeing you engage with this",
    "Keep those great questions coming",
  ],
  // Wrap up
  complete: [
    "Great work today! See you tomorrow 🌟",
    "You did it! Another lesson complete",
    "Thanks for learning with me 💙",
    "Until next time, stay curious!",
  ],
};

window.KELLY_COMMENTS = KELLY_COMMENTS;

// ═══════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════

window.ChatOverlay = ChatOverlay;
window.COMMENT_BANKS = COMMENT_BANKS;
