/**
 * Kelly Conversational AI v1.0
 * ElevenLabs Conversational AI Integration
 * 
 * Enables real-time voice conversation with Kelly during lessons.
 * Kelly stays on-topic and relates everything back to the current lesson.
 * 
 * Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
 */

// ═══════════════════════════════════════════════════════════════════
// KELLY CONVERSATION SYSTEM
// ═══════════════════════════════════════════════════════════════════

const KellyConversation = {
  isActive: false,
  isListening: false,
  isSpeaking: false,
  conversationHistory: [],
  lessonContext: null,
  
  // ElevenLabs configuration
  config: {
    agentId: null,           // Set via init() or environment
    apiKey: null,            // Optional: for signed URLs
    voiceId: 'wAdymQH5YucAkXwmrdL0', // Kelly's trained voice
  },
  
  // WebSocket connection
  ws: null,
  audioContext: null,
  mediaRecorder: null,
  audioQueue: [],
  
  // UI elements
  micButton: null,
  transcriptContainer: null,
  
  // ═══════════════════════════════════════════════════════════════════
  // SYSTEM PROMPT
  // ═══════════════════════════════════════════════════════════════════
  
  getSystemPrompt() {
    const ctx = this.lessonContext || {};
    
    return `You are Kelly, a warm and curious educator who helps people learn something new every day. You're talking to a learner during their daily lesson.

TODAY'S LESSON: ${ctx.topic || 'General Learning'}
LESSON CONTENT: ${ctx.summary || 'Exploring new ideas together'}
CURRENT PHASE: ${ctx.currentPhase || 'welcome'}
${ctx.currentQuestion ? `CURRENT QUESTION: ${ctx.currentQuestion}` : ''}

YOUR PERSONALITY:
- Warm, encouraging, genuinely curious
- You celebrate small wins enthusiastically
- You make complex topics feel approachable
- You use analogies and real-world examples
- You're playful but never condescending
- You ask follow-up questions to deepen understanding
- You have a slight sparkle of wonder in your voice

YOUR GOAL:
- Help the learner engage with today's lesson topic
- If they ask off-topic questions, gently relate it back to today's topic
- Encourage them to think about the questions, don't give away answers
- Build their confidence and curiosity
- Keep responses conversational and natural

IMPORTANT RULES:
- Keep responses concise (2-3 sentences usually, max 4)
- Always be supportive, never judgmental
- If they're stuck, give hints not answers
- Relate EVERYTHING back to today's topic: "${ctx.topic || 'learning'}"
- Sound natural, like a friendly teacher, not a robot
- Use "we" and "us" to create togetherness
- Express genuine excitement about learning

EXAMPLE INTERACTIONS:
User: "What should I have for dinner?"
Kelly: "Ooh, dinner! You know, choosing what to eat is actually a great example of what we're exploring today. What 'rules' do you usually follow when picking dinner?"

User: "I don't understand"
Kelly: "That's totally okay! Let's break it down together. What part feels tricky? Sometimes it helps to think about a real example from your own life."

User: "This is boring"
Kelly: "I hear you! But here's the thing - ${ctx.topic || 'this topic'} actually shows up everywhere in life. Can you think of a time when...?"`;
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init(options = {}) {
    this.config.agentId = options.agentId || window.ELEVENLABS_AGENT_ID || null;
    this.config.apiKey = options.apiKey || window.ELEVENLABS_API_KEY || null;
    this.config.voiceId = options.voiceId || window.ELEVENLABS_VOICE_ID || this.config.voiceId;
    
    this.createMicButton();
    this.createTranscriptUI();
    this.bindEvents();
    
    console.log('[KellyConversation] Initialized', {
      hasAgentId: !!this.config.agentId,
      voiceId: this.config.voiceId
    });
    
    return this;
  },
  
  setLessonContext(lesson, phase) {
    this.lessonContext = {
      topic: lesson?.topic || 'Today\'s Lesson',
      summary: lesson?.universal_truth || lesson?.marketing_headline || '',
      currentPhase: phase?.name || phase?.type || 'welcome',
      currentQuestion: phase?.question || null,
      dayNumber: lesson?.day_number || null
    };
    
    console.log('[KellyConversation] Context updated:', this.lessonContext.topic);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // MIC BUTTON UI
  // ═══════════════════════════════════════════════════════════════════
  
  createMicButton() {
    // Don't create if already exists
    if (document.getElementById('kelly-mic-btn')) {
      this.micButton = document.getElementById('kelly-mic-btn');
      return;
    }
    
    // Create button
    this.micButton = document.createElement('button');
    this.micButton.id = 'kelly-mic-btn';
    this.micButton.className = 'kelly-mic-btn';
    this.micButton.setAttribute('aria-label', 'Talk to Kelly');
    this.micButton.innerHTML = `
      <svg class="mic-icon" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
      </svg>
      <span class="mic-pulse"></span>
      <span class="mic-waves">
        <span class="wave"></span>
        <span class="wave"></span>
        <span class="wave"></span>
      </span>
    `;
    
    // Add styles
    this.addMicStyles();
    
    document.body.appendChild(this.micButton);
  },
  
  addMicStyles() {
    if (document.getElementById('kelly-mic-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-mic-styles';
    styles.textContent = `
      .kelly-mic-btn {
        position: fixed;
        bottom: 100px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border: none;
        color: white;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        z-index: 1000;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }
      
      .kelly-mic-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(59, 130, 246, 0.5);
      }
      
      .kelly-mic-btn:active {
        transform: scale(0.95);
      }
      
      .kelly-mic-btn .mic-icon {
        width: 28px;
        height: 28px;
        z-index: 2;
        transition: transform 0.3s ease;
      }
      
      .kelly-mic-btn .mic-pulse {
        display: none;
        position: absolute;
        inset: -8px;
        border-radius: 50%;
        border: 3px solid rgba(239, 68, 68, 0.5);
        animation: micPulse 1.5s infinite;
      }
      
      .kelly-mic-btn .mic-waves {
        display: none;
        position: absolute;
        inset: 0;
        align-items: center;
        justify-content: center;
      }
      
      .kelly-mic-btn .wave {
        position: absolute;
        width: 4px;
        height: 20px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 2px;
        animation: micWave 0.5s ease-in-out infinite;
      }
      
      .kelly-mic-btn .wave:nth-child(1) { left: 14px; animation-delay: 0s; }
      .kelly-mic-btn .wave:nth-child(2) { left: 28px; animation-delay: 0.1s; }
      .kelly-mic-btn .wave:nth-child(3) { left: 42px; animation-delay: 0.2s; }
      
      /* Listening state */
      .kelly-mic-btn.listening {
        background: linear-gradient(135deg, #ef4444, #f97316);
        animation: micGlow 1.5s ease-in-out infinite;
      }
      
      .kelly-mic-btn.listening .mic-pulse {
        display: block;
      }
      
      .kelly-mic-btn.listening .mic-icon {
        transform: scale(0.9);
      }
      
      /* Speaking state */
      .kelly-mic-btn.speaking {
        background: linear-gradient(135deg, #10b981, #3b82f6);
      }
      
      .kelly-mic-btn.speaking .mic-icon {
        display: none;
      }
      
      .kelly-mic-btn.speaking .mic-waves {
        display: flex;
      }
      
      /* Disabled state */
      .kelly-mic-btn.disabled {
        opacity: 0.5;
        cursor: not-allowed;
        pointer-events: none;
      }
      
      /* Tooltip */
      .kelly-mic-btn::after {
        content: 'Talk to Kelly';
        position: absolute;
        bottom: 100%;
        right: 0;
        margin-bottom: 8px;
        padding: 8px 12px;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        font-size: 0.85rem;
        border-radius: 8px;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s;
      }
      
      .kelly-mic-btn:hover::after {
        opacity: 1;
      }
      
      .kelly-mic-btn.listening::after {
        content: 'Listening...';
      }
      
      .kelly-mic-btn.speaking::after {
        content: 'Kelly is speaking';
      }
      
      @keyframes micPulse {
        0% { transform: scale(1); opacity: 1; }
        100% { transform: scale(1.4); opacity: 0; }
      }
      
      @keyframes micGlow {
        0%, 100% { box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 4px 40px rgba(239, 68, 68, 0.7); }
      }
      
      @keyframes micWave {
        0%, 100% { height: 10px; }
        50% { height: 30px; }
      }
      
      /* Transcript UI */
      .kelly-transcript {
        position: fixed;
        bottom: 180px;
        right: 20px;
        width: 300px;
        max-height: 200px;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 16px;
        z-index: 999;
        opacity: 0;
        transform: translateY(10px);
        pointer-events: none;
        transition: all 0.3s ease;
        overflow-y: auto;
      }
      
      .kelly-transcript.visible {
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
      }
      
      .kelly-transcript .message {
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .kelly-transcript .message:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
      }
      
      .kelly-transcript .message-role {
        font-size: 0.75rem;
        font-weight: 600;
        color: #3b82f6;
        margin-bottom: 4px;
        text-transform: uppercase;
      }
      
      .kelly-transcript .message-role.user {
        color: #f97316;
      }
      
      .kelly-transcript .message-text {
        font-size: 0.9rem;
        color: #e5e5e5;
        line-height: 1.4;
      }
      
      /* Mobile adjustments */
      @media (max-width: 768px) {
        .kelly-mic-btn {
          bottom: 140px;
          right: 16px;
          width: 56px;
          height: 56px;
        }
        
        .kelly-transcript {
          bottom: 210px;
          right: 16px;
          left: 16px;
          width: auto;
        }
      }
    `;
    
    document.head.appendChild(styles);
  },
  
  createTranscriptUI() {
    if (document.getElementById('kelly-transcript')) {
      this.transcriptContainer = document.getElementById('kelly-transcript');
      return;
    }
    
    this.transcriptContainer = document.createElement('div');
    this.transcriptContainer.id = 'kelly-transcript';
    this.transcriptContainer.className = 'kelly-transcript';
    
    document.body.appendChild(this.transcriptContainer);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // EVENT BINDING
  // ═══════════════════════════════════════════════════════════════════
  
  bindEvents() {
    if (!this.micButton) return;
    
    // Click to toggle conversation
    this.micButton.addEventListener('click', () => {
      if (this.isActive) {
        this.endConversation();
      } else {
        this.startConversation();
      }
    });
    
    // Keyboard shortcut (hold space to talk)
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        if (!this.isActive && !e.repeat) {
          e.preventDefault();
          this.startConversation();
        }
      }
    });
    
    document.addEventListener('keyup', (e) => {
      if (e.code === 'Space' && this.isActive) {
        // Optional: end on space release for push-to-talk mode
        // this.endConversation();
      }
    });
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // CONVERSATION FLOW
  // ═══════════════════════════════════════════════════════════════════
  
  async startConversation() {
    if (this.isActive) return;
    
    // Check for agent ID
    if (!this.config.agentId) {
      this.showNotConfiguredMessage();
      return;
    }
    
    console.log('[KellyConversation] Starting conversation...');
    
    this.isActive = true;
    this.isListening = true;
    this.conversationHistory = [];
    
    // Update UI
    this.micButton.classList.add('listening');
    this.showTranscript();
    
    // Kelly's initial greeting
    this.addToTranscript('kelly', "I'm here! What's on your mind about today's lesson?");
    
    try {
      // Initialize audio context
      await this.initAudio();
      
      // Connect to ElevenLabs
      await this.connectToElevenLabs();
      
      // Start recording
      await this.startRecording();
      
    } catch (error) {
      console.error('[KellyConversation] Error starting:', error);
      this.showError('Could not start conversation. Please check microphone permissions.');
      this.endConversation();
    }
  },
  
  endConversation() {
    console.log('[KellyConversation] Ending conversation');
    
    this.isActive = false;
    this.isListening = false;
    this.isSpeaking = false;
    
    // Stop recording
    this.stopRecording();
    
    // Close WebSocket
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    // Update UI
    this.micButton.classList.remove('listening', 'speaking');
    
    // Hide transcript after delay
    setTimeout(() => {
      if (!this.isActive) {
        this.hideTranscript();
      }
    }, 3000);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // AUDIO HANDLING
  // ═══════════════════════════════════════════════════════════════════
  
  async initAudio() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  },
  
  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.ws && this.ws.readyState === WebSocket.OPEN) {
          // Convert to base64 and send
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            this.ws.send(JSON.stringify({
              type: 'audio',
              audio: base64Audio
            }));
          };
          reader.readAsDataURL(event.data);
        }
      };
      
      // Record in 100ms chunks for low latency
      this.mediaRecorder.start(100);
      
      console.log('[KellyConversation] Recording started');
      
    } catch (error) {
      console.error('[KellyConversation] Microphone error:', error);
      throw new Error('Microphone access denied');
    }
  },
  
  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      this.mediaRecorder = null;
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // ELEVENLABS WEBSOCKET
  // ═══════════════════════════════════════════════════════════════════
  
  async connectToElevenLabs() {
    return new Promise((resolve, reject) => {
      const wsUrl = `wss://api.elevenlabs.io/v1/convai/conversation?agent_id=${this.config.agentId}`;
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('[KellyConversation] WebSocket connected');
        
        // Send initial configuration
        this.ws.send(JSON.stringify({
          type: 'conversation_initiation_client_data',
          conversation_config_override: {
            agent: {
              prompt: {
                prompt: this.getSystemPrompt()
              },
              first_message: "I'm here! What's on your mind about today's lesson?",
              language: 'en'
            },
            tts: {
              voice_id: this.config.voiceId
            }
          }
        }));
        
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        this.handleWSMessage(event);
      };
      
      this.ws.onerror = (error) => {
        console.error('[KellyConversation] WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onclose = () => {
        console.log('[KellyConversation] WebSocket closed');
        if (this.isActive) {
          this.endConversation();
        }
      };
    });
  },
  
  handleWSMessage(event) {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'user_transcript':
          // User's speech transcribed
          if (data.user_transcript_event?.user_transcript) {
            this.addToTranscript('user', data.user_transcript_event.user_transcript);
          }
          break;
          
        case 'agent_response':
          // Kelly's response text
          if (data.agent_response_event?.agent_response) {
            this.addToTranscript('kelly', data.agent_response_event.agent_response);
          }
          break;
          
        case 'audio':
          // Kelly's voice audio
          this.playAudio(data.audio_event?.audio_base_64);
          this.isSpeaking = true;
          this.micButton.classList.add('speaking');
          this.micButton.classList.remove('listening');
          
          // Update Kelly's visual state
          if (window.KellyPoseManager) {
            KellyPoseManager.setPose('explaining');
            KellyPoseManager.setMouthState('speaking');
          }
          break;
          
        case 'audio_done':
          // Kelly finished speaking
          this.isSpeaking = false;
          this.micButton.classList.remove('speaking');
          if (this.isActive) {
            this.micButton.classList.add('listening');
          }
          
          // Update Kelly's visual state
          if (window.KellyPoseManager) {
            KellyPoseManager.setPose('listening');
            KellyPoseManager.setMouthState('idle');
          }
          break;
          
        case 'interruption':
          // User interrupted Kelly
          this.stopCurrentAudio();
          break;
          
        case 'ping':
          // Keep-alive
          this.ws.send(JSON.stringify({ type: 'pong' }));
          break;
          
        case 'error':
          console.error('[KellyConversation] Server error:', data);
          this.showError(data.message || 'Connection error');
          break;
      }
    } catch (e) {
      console.warn('[KellyConversation] Message parse error:', e);
    }
  },
  
  async playAudio(base64Audio) {
    if (!base64Audio || !this.audioContext) return;
    
    try {
      const audioData = atob(base64Audio);
      const arrayBuffer = new ArrayBuffer(audioData.length);
      const view = new Uint8Array(arrayBuffer);
      
      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i);
      }
      
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);
      source.start();
      
      this.currentAudioSource = source;
    } catch (e) {
      console.warn('[KellyConversation] Audio playback error:', e);
    }
  },
  
  stopCurrentAudio() {
    if (this.currentAudioSource) {
      try {
        this.currentAudioSource.stop();
      } catch (e) {}
      this.currentAudioSource = null;
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // TRANSCRIPT UI
  // ═══════════════════════════════════════════════════════════════════
  
  addToTranscript(role, text) {
    if (!text) return;
    
    this.conversationHistory.push({ role, text, timestamp: Date.now() });
    
    if (this.transcriptContainer) {
      const messageEl = document.createElement('div');
      messageEl.className = 'message';
      messageEl.innerHTML = `
        <div class="message-role ${role}">${role === 'kelly' ? 'Kelly' : 'You'}</div>
        <div class="message-text">${text}</div>
      `;
      
      this.transcriptContainer.appendChild(messageEl);
      this.transcriptContainer.scrollTop = this.transcriptContainer.scrollHeight;
    }
  },
  
  showTranscript() {
    if (this.transcriptContainer) {
      this.transcriptContainer.innerHTML = '';
      this.transcriptContainer.classList.add('visible');
    }
  },
  
  hideTranscript() {
    if (this.transcriptContainer) {
      this.transcriptContainer.classList.remove('visible');
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // ERROR HANDLING
  // ═══════════════════════════════════════════════════════════════════
  
  showNotConfiguredMessage() {
    const toast = document.createElement('div');
    toast.className = 'kelly-conversation-toast';
    toast.innerHTML = `
      <strong>Voice Chat Not Configured</strong>
      <p>Add ELEVENLABS_AGENT_ID to enable voice conversations with Kelly.</p>
    `;
    toast.style.cssText = `
      position: fixed;
      bottom: 180px;
      right: 20px;
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 16px 20px;
      border-radius: 12px;
      font-size: 0.9rem;
      z-index: 10000;
      max-width: 300px;
      animation: fadeIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  },
  
  showError(message) {
    const toast = document.createElement('div');
    toast.className = 'kelly-conversation-toast error';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 180px;
      right: 20px;
      background: rgba(239, 68, 68, 0.95);
      color: white;
      padding: 12px 20px;
      border-radius: 12px;
      font-size: 0.9rem;
      z-index: 10000;
      animation: fadeIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }
};

// ═══════════════════════════════════════════════════════════════════
// FALLBACK: Text-based conversation (when voice not available)
// ═══════════════════════════════════════════════════════════════════

const KellyTextChat = {
  isOpen: false,
  chatContainer: null,
  
  init() {
    // Only init if voice conversation not available
    if (window.ELEVENLABS_AGENT_ID) return;
    
    this.createChatUI();
    console.log('[KellyTextChat] Text fallback initialized');
  },
  
  createChatUI() {
    // Text chat UI as fallback - simplified for now
    // Full implementation would include input field and message display
  },
  
  async sendMessage(text) {
    // Use Claude/OpenAI for text responses when voice not available
    // This is a fallback for when ElevenLabs is not configured
  }
};

// ═══════════════════════════════════════════════════════════════════
// AUTO-INITIALIZE
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // Initialize conversation system
  KellyConversation.init();
  
  // Update context when lesson loads
  const checkLesson = () => {
    if (window.state?.lesson) {
      const phase = window.state.lesson.phases?.[window.state.currentPhase - 1];
      KellyConversation.setLessonContext(window.state.lesson, phase);
    } else {
      setTimeout(checkLesson, 500);
    }
  };
  
  setTimeout(checkLesson, 1000);
});

// ═══════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════

window.KellyConversation = KellyConversation;
window.KellyTextChat = KellyTextChat;

console.log('[KellyConversation] ✅ Loaded - Voice conversation system ready');



