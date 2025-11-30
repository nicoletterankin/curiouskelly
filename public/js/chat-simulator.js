/**
 * Chat Simulator for Social Learning Experience
 * Simulates global learners engaging with lessons in real-time
 */

// Diverse, thoughtful messages from around the world
const CHAT_MESSAGES = [
  // Insightful (40%)
  { user: "Maria", flag: "🇧🇷", country: "Brazil", text: "Oh wow, this explains why I see this every day!", category: "insightful" },
  { user: "James", flag: "🇬🇧", country: "UK", text: "I never thought about it that way before", category: "insightful" },
  { user: "Yuki", flag: "🇯🇵", country: "Japan", text: "This connects to yesterday's lesson about energy!", category: "insightful" },
  { user: "Ahmed", flag: "🇪🇬", country: "Egypt", text: "Now the whole concept makes sense", category: "insightful" },
  { user: "Sofia", flag: "🇲🇽", country: "Mexico", text: "This is exactly what I needed to understand", category: "insightful" },
  { user: "Li Wei", flag: "🇨🇳", country: "China", text: "The way Kelly explains this is brilliant", category: "insightful" },
  { user: "Priya", flag: "🇮🇳", country: "India", text: "I can apply this to my work immediately", category: "insightful" },
  { user: "Hans", flag: "🇩🇪", country: "Germany", text: "This changes how I see the world", category: "insightful" },
  { user: "Fatima", flag: "🇸🇦", country: "Saudi Arabia", text: "Beautiful way to explain a complex idea", category: "insightful" },
  { user: "Carlos", flag: "🇦🇷", country: "Argentina", text: "This is the missing piece I was looking for", category: "insightful" },
  
  // Excited (30%)
  { user: "Emma", flag: "🇺🇸", country: "USA", text: "Mind blown! 🤯", category: "excited" },
  { user: "Anya", flag: "🇷🇺", country: "Russia", text: "Kelly is the best teacher ever! ✨", category: "excited" },
  { user: "Kofi", flag: "🇬🇭", country: "Ghana", text: "This is so cool!", category: "excited" },
  { user: "Mei", flag: "🇹🇼", country: "Taiwan", text: "Wow wow wow! 🌟", category: "excited" },
  { user: "Lucas", flag: "🇫🇷", country: "France", text: "C'est magnifique! 🙌", category: "excited" },
  { user: "Isabella", flag: "🇮🇹", country: "Italy", text: "Incredibile!", category: "excited" },
  { user: "Raj", flag: "🇮🇳", country: "India", text: "This is amazing! 💫", category: "excited" },
  { user: "Sven", flag: "🇸🇪", country: "Sweden", text: "Fantastic explanation!", category: "excited" },
  
  // Social (20%)
  { user: "Aisha", flag: "🇰🇪", country: "Kenya", text: "Good morning from Nairobi! 🌅", category: "social" },
  { user: "Diego", flag: "🇨🇱", country: "Chile", text: "Showing my kids right now 👨‍👩‍👧‍👦", category: "social" },
  { user: "Nina", flag: "🇳🇴", country: "Norway", text: "Learning together across the world 🌍", category: "social" },
  { user: "Omar", flag: "🇦🇪", country: "UAE", text: "My morning ritual with coffee ☕", category: "social" },
  { user: "Zara", flag: "🇿🇦", country: "South Africa", text: "Sharing with my students!", category: "social" },
  { user: "Kai", flag: "🇹🇭", country: "Thailand", text: "Hello from Bangkok! 👋", category: "social" },
  { user: "Lena", flag: "🇵🇱", country: "Poland", text: "My whole family watches together", category: "social" },
  
  // Questions (10%)
  { user: "Miguel", flag: "🇪🇸", country: "Spain", text: "Wait, so does that mean...? 🤔", category: "question" },
  { user: "Sarah", flag: "🇨🇦", country: "Canada", text: "How does this work with what we learned last week?", category: "question" },
  { user: "Jin", flag: "🇰🇷", country: "South Korea", text: "Can someone explain the part about...?", category: "question" },
  { user: "Ana", flag: "🇵🇹", country: "Portugal", text: "Is this related to the previous topic?", category: "question" }
];

// Live stats simulation
let viewerCount = 1247832;
let countriesCount = 147;
let reactionsCount = 89234;

class ChatSimulator {
  constructor(containerId, statsContainerId) {
    this.container = document.getElementById(containerId);
    this.statsContainer = document.getElementById(statsContainerId);
    this.messageQueue = [];
    this.isActive = false;
    this.messageIndex = 0;
  }
  
  start() {
    if (this.isActive) return;
    this.isActive = true;
    
    // Shuffle messages
    this.messageQueue = [...CHAT_MESSAGES].sort(() => Math.random() - 0.5);
    
    // Add first message immediately
    this.addMessage();
    
    // Then add messages at intervals
    this.scheduleNextMessage();
    
    // Update stats periodically
    this.updateStats();
    setInterval(() => this.updateStats(), 5000);
  }
  
  stop() {
    this.isActive = false;
  }
  
  scheduleNextMessage() {
    if (!this.isActive) return;
    
    // Random interval between 3-8 seconds
    const delay = 3000 + Math.random() * 5000;
    
    setTimeout(() => {
      this.addMessage();
      this.scheduleNextMessage();
    }, delay);
  }
  
  addMessage() {
    if (!this.container) return;
    
    const msg = this.messageQueue[this.messageIndex % this.messageQueue.length];
    this.messageIndex++;
    
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message';
    messageEl.style.cssText = `
      margin-bottom: 16px;
      animation: slideInRight 0.3s ease-out;
      opacity: 0;
      animation-fill-mode: forwards;
    `;
    
    messageEl.innerHTML = `
      <div style="
        font-size: 0.8rem;
        color: var(--tiktok-accent);
        margin-bottom: 4px;
        font-weight: 600;
      ">
        <span style="margin-right: 4px;">${msg.flag}</span>${msg.user} • ${msg.country}
      </div>
      <div style="
        font-size: 0.9rem;
        color: var(--tiktok-text);
        line-height: 1.4;
      ">${msg.text}</div>
    `;
    
    this.container.appendChild(messageEl);
    
    // Scroll to bottom
    this.container.scrollTop = this.container.scrollHeight;
    
    // Remove old messages (keep last 20)
    while (this.container.children.length > 20) {
      this.container.removeChild(this.container.firstChild);
    }
    
    // Update reactions count
    reactionsCount += Math.floor(Math.random() * 50) + 10;
  }
  
  updateStats() {
    if (!this.statsContainer) return;
    
    // Simulate viewer count changes
    viewerCount += Math.floor(Math.random() * 200) - 80;
    viewerCount = Math.max(1200000, viewerCount);
    
    // Update countries occasionally
    if (Math.random() > 0.7) {
      countriesCount = Math.min(195, countriesCount + 1);
    }
    
    this.statsContainer.innerHTML = `
      <div style="
        display: flex;
        justify-content: space-around;
        text-align: center;
        padding: 16px;
        background: rgba(59, 130, 246, 0.1);
        border-top: 1px solid rgba(59, 130, 246, 0.2);
      ">
        <div>
          <div style="font-size: 1.25rem; font-weight: 700; color: var(--tiktok-accent);">
            ${countriesCount}
          </div>
          <div style="font-size: 0.7rem; color: var(--tiktok-text-muted); text-transform: uppercase;">
            Countries
          </div>
        </div>
        <div>
          <div style="font-size: 1.25rem; font-weight: 700; color: var(--tiktok-accent);">
            ${(reactionsCount / 1000).toFixed(0)}K
          </div>
          <div style="font-size: 0.7rem; color: var(--tiktok-text-muted); text-transform: uppercase;">
            Reactions
          </div>
        </div>
        <div>
          <div style="font-size: 1.25rem; font-weight: 700; color: var(--tiktok-accent);">
            ${(viewerCount / 1000000).toFixed(2)}M
          </div>
          <div style="font-size: 0.7rem; color: var(--tiktok-text-muted); text-transform: uppercase;">
            Watching
          </div>
        </div>
      </div>
    `;
  }
}

// Add animation styles
const chatStyles = document.createElement('style');
chatStyles.textContent = `
  @keyframes slideInRight {
    from {
      opacity: 0;
      transform: translateX(20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  
  .chat-message {
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }
  
  .chat-message:last-child {
    border-bottom: none;
  }
`;
document.head.appendChild(chatStyles);

// Export
window.ChatSimulator = ChatSimulator;

