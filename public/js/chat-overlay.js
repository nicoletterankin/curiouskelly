/**
 * Chat Overlay - YouTube Live / Movie Style
 * Transparent overlay with floating messages
 */

const CHAT_MESSAGES = [
  // Insightful
  { user: "Maria", flag: "🇧🇷", text: "This makes so much sense now! 💡" },
  { user: "James", flag: "🇬🇧", text: "Never thought about it that way" },
  { user: "Yuki", flag: "🇯🇵", text: "Kelly explains this so well ✨" },
  { user: "Ahmed", flag: "🇪🇬", text: "Now I understand!" },
  { user: "Sofia", flag: "🇲🇽", text: "This is exactly what I needed" },
  { user: "Li Wei", flag: "🇨🇳", text: "Brilliant explanation 🌟" },
  { user: "Priya", flag: "🇮🇳", text: "I can use this today!" },
  { user: "Hans", flag: "🇩🇪", text: "Mind = blown 🤯" },
  
  // Excited
  { user: "Emma", flag: "🇺🇸", text: "WOW! 🔥" },
  { user: "Anya", flag: "🇷🇺", text: "Kelly is the best! ❤️" },
  { user: "Kofi", flag: "🇬🇭", text: "So cool! 🙌" },
  { user: "Mei", flag: "🇹🇼", text: "Amazing! ✨" },
  { user: "Lucas", flag: "🇫🇷", text: "Magnifique! 💫" },
  { user: "Isabella", flag: "🇮🇹", text: "Incredibile! 🎉" },
  
  // Social
  { user: "Aisha", flag: "🇰🇪", text: "Good morning! ☀️" },
  { user: "Diego", flag: "🇨🇱", text: "Watching with my kids 👨‍👩‍👧" },
  { user: "Nina", flag: "🇳🇴", text: "Learning together 🌍" },
  { user: "Omar", flag: "🇦🇪", text: "Daily ritual ☕" },
  { user: "Zara", flag: "🇿🇦", text: "Sharing this!" },
  
  // Reactions
  { user: "Jin", flag: "🇰🇷", text: "👏👏👏" },
  { user: "Carlos", flag: "🇦🇷", text: "🎯" },
  { user: "Sarah", flag: "🇨🇦", text: "❤️❤️❤️" },
  { user: "Raj", flag: "🇮🇳", text: "🙏" },
  { user: "Sven", flag: "🇸🇪", text: "💯" }
];

class ChatOverlay {
  constructor() {
    this.container = null;
    this.statsBar = null;
    this.isActive = false;
    this.messageIndex = 0;
    this.viewerCount = 1247832;
    this.countriesCount = 147;
    this.shuffledMessages = [];
    
    this.init();
  }
  
  init() {
    // Create overlay container
    this.container = document.createElement('div');
    this.container.id = 'chat-overlay';
    this.container.innerHTML = `
      <style>
        #chat-overlay {
          position: fixed;
          bottom: 100px;
          left: 16px;
          width: 320px;
          max-height: 200px;
          pointer-events: none;
          z-index: 1000;
          overflow: hidden;
        }
        
        @media (max-width: 768px) {
          #chat-overlay {
            width: calc(100% - 80px);
            bottom: 80px;
            left: 12px;
          }
        }
        
        .chat-message-overlay {
          display: flex;
          align-items: flex-start;
          gap: 8px;
          padding: 8px 12px;
          margin-bottom: 6px;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(8px);
          border-radius: 20px;
          animation: chatSlideIn 0.3s ease-out, chatFadeOut 0.5s ease-in 4.5s forwards;
          max-width: fit-content;
        }
        
        .chat-message-overlay .flag {
          font-size: 1rem;
        }
        
        .chat-message-overlay .content {
          display: flex;
          flex-direction: column;
        }
        
        .chat-message-overlay .user {
          font-size: 0.75rem;
          font-weight: 600;
          color: #3b82f6;
        }
        
        .chat-message-overlay .text {
          font-size: 0.875rem;
          color: white;
          line-height: 1.3;
        }
        
        @keyframes chatSlideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes chatFadeOut {
          to {
            opacity: 0;
          }
        }
        
        /* Stats bar at bottom */
        #live-stats-bar {
          position: fixed;
          bottom: 60px;
          left: 16px;
          display: flex;
          gap: 16px;
          padding: 8px 16px;
          background: rgba(0, 0, 0, 0.7);
          backdrop-filter: blur(8px);
          border-radius: 20px;
          z-index: 999;
        }
        
        @media (max-width: 768px) {
          #live-stats-bar {
            left: 12px;
            bottom: 70px;
            gap: 12px;
            padding: 6px 12px;
          }
        }
        
        .stat-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 0.8rem;
          color: white;
        }
        
        .stat-item .value {
          font-weight: 600;
          color: #3b82f6;
        }
      </style>
    `;
    
    // Create stats bar
    this.statsBar = document.createElement('div');
    this.statsBar.id = 'live-stats-bar';
    this.updateStats();
    
    document.body.appendChild(this.container);
    document.body.appendChild(this.statsBar);
  }
  
  start() {
    if (this.isActive) return;
    this.isActive = true;
    
    // Shuffle messages
    this.shuffledMessages = [...CHAT_MESSAGES].sort(() => Math.random() - 0.5);
    this.messageIndex = 0;
    
    // Add first message
    this.addMessage();
    
    // Schedule more messages
    this.scheduleNext();
    
    // Update stats periodically
    this.statsInterval = setInterval(() => this.updateStats(), 5000);
    
    console.log('[ChatOverlay] Started');
  }
  
  stop() {
    this.isActive = false;
    if (this.statsInterval) clearInterval(this.statsInterval);
    console.log('[ChatOverlay] Stopped');
  }
  
  scheduleNext() {
    if (!this.isActive) return;
    
    const delay = 2000 + Math.random() * 4000; // 2-6 seconds
    setTimeout(() => {
      this.addMessage();
      this.scheduleNext();
    }, delay);
  }
  
  addMessage() {
    const msg = this.shuffledMessages[this.messageIndex % this.shuffledMessages.length];
    this.messageIndex++;
    
    const el = document.createElement('div');
    el.className = 'chat-message-overlay';
    el.innerHTML = `
      <span class="flag">${msg.flag}</span>
      <div class="content">
        <span class="user">${msg.user}</span>
        <span class="text">${msg.text}</span>
      </div>
    `;
    
    this.container.appendChild(el);
    
    // Remove after animation
    setTimeout(() => {
      if (el.parentNode) el.remove();
    }, 5000);
    
    // Keep max 5 messages
    while (this.container.children.length > 6) { // +1 for style tag
      const first = this.container.querySelector('.chat-message-overlay');
      if (first) first.remove();
    }
  }
  
  updateStats() {
    // Simulate changes
    this.viewerCount += Math.floor(Math.random() * 200) - 80;
    this.viewerCount = Math.max(1200000, this.viewerCount);
    
    if (Math.random() > 0.8) {
      this.countriesCount = Math.min(195, this.countriesCount + 1);
    }
    
    const viewers = (this.viewerCount / 1000000).toFixed(2) + 'M';
    
    this.statsBar.innerHTML = `
      <div class="stat-item">
        <span>🌍</span>
        <span class="value">${this.countriesCount}</span>
        <span>countries</span>
      </div>
      <div class="stat-item">
        <span>👥</span>
        <span class="value">${viewers}</span>
        <span>watching</span>
      </div>
    `;
  }
  
  destroy() {
    this.stop();
    if (this.container) this.container.remove();
    if (this.statsBar) this.statsBar.remove();
  }
}

// Auto-initialize
window.ChatOverlay = ChatOverlay;

