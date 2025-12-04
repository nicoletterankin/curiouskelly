/**
 * Kelly Conversational AI v2.0
 * ElevenLabs Conversational AI Integration (Fixed)
 * 
 * Enables real-time voice conversation with Kelly during lessons.
 * Kelly stays on-topic and relates everything back to the current lesson.
 * 
 * FIXES in v2.0:
 * - Proper audio format handling (PCM 16-bit for input)
 * - Support for both public agents and signed URLs
 * - Expression bridge to Kelly avatar
 * - Correct message type handling
 */

// ═══════════════════════════════════════════════════════════════════
// KELLY CONVERSATION SYSTEM v2
// ═══════════════════════════════════════════════════════════════════

const KellyConversation = {
  isActive: false,
  isListening: false,
  isSpeaking: false,
  conversationHistory: [],
  lessonContext: null,
  conversationId: null,
  
  // ElevenLabs configuration
  config: {
    agentId: null,           // Set via init() or environment
    voiceId: 'wAdymQH5YucAkXwmrdL0', // Kelly's trained voice
    signedUrl: null,         // For private agents
    isPublic: true,          // Whether agent is public
  },
  
  // Audio configuration
  audio: {
    context: null,
    mediaStream: null,
    audioWorklet: null,
    gainNode: null,
    sampleRate: 16000,       // ElevenLabs expects 16kHz
  },
  
  // WebSocket connection
  ws: null,
  
  // Audio playback queue
  audioQueue: [],
  isPlayingAudio: false,
  
  // Lip-sync integration
  lipSyncEnabled: true,
  
  // UI elements
  talkButton: null,
  transcriptContainer: null,
  
  // Expression callbacks
  onExpression: null,
  onSpeakingStart: null,
  onSpeakingEnd: null,
  onListeningStart: null,
  onListeningEnd: null,
  
  // ═══════════════════════════════════════════════════════════════════
  // SYSTEM PROMPT (Sent via conversation_config_override)
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
- Express genuine excitement about learning`;
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  init(options = {}) {
    this.config.agentId = options.agentId || window.ELEVENLABS_AGENT_ID || null;
    this.config.voiceId = options.voiceId || window.ELEVENLABS_VOICE_ID || this.config.voiceId;
    
    // Set up expression callbacks
    this.onExpression = options.onExpression || this.defaultExpressionHandler.bind(this);
    this.onSpeakingStart = options.onSpeakingStart || null;
    this.onSpeakingEnd = options.onSpeakingEnd || null;
    this.onListeningStart = options.onListeningStart || null;
    this.onListeningEnd = options.onListeningEnd || null;
    
    // Lip-sync configuration
    this.lipSyncEnabled = options.lipSyncEnabled !== false;
    
    // Initialize lip-sync system if available
    if (this.lipSyncEnabled && window.KellyLipSync) {
      window.KellyLipSync.init({
        sendToUnity: true,
        sendTo2D: true,
        sensitivity: 1.6,
        smoothing: 0.55,
      });
      console.log('[KellyConversation v2] Lip-sync system initialized');
    }
    
    // Create UI elements
    this.createTranscriptUI();
    this.addStyles();
    
    // Wire up the talk button if it exists
    this.talkButton = document.getElementById('talk-to-kelly-btn');
    
    console.log('[KellyConversation v2] Initialized', {
      hasAgentId: !!this.config.agentId,
      agentId: this.config.agentId,
      voiceId: this.config.voiceId,
      lipSyncEnabled: this.lipSyncEnabled
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
    
    console.log('[KellyConversation v2] Context updated:', this.lessonContext.topic);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // DEFAULT EXPRESSION HANDLER
  // Connects voice events to Kelly's visual state
  // ═══════════════════════════════════════════════════════════════════
  
  defaultExpressionHandler(expression, data = {}) {
    console.log('[KellyConversation v2] Expression:', expression, data);
    
    // Update KellyPoseManager if available (2D avatar)
    if (window.KellyPoseManager) {
      const poseMap = {
        'listening': 'listening',
        'thinking': 'curious',
        'speaking': 'explaining',
        'celebrating': 'celebrating',
        'idle': 'hello',
        'curious': 'curious',
        'explaining': 'explaining',
        'wisdom': 'wisdom'
      };
      
      const pose = poseMap[expression] || 'hello';
      KellyPoseManager.setPose(pose);
      
      // Handle mouth state
      if (expression === 'speaking') {
        KellyPoseManager.setMouthState?.('speaking');
      } else {
        KellyPoseManager.setMouthState?.('idle');
      }
    }
    
    // Update Kelly 2D Avatar if available
    if (window.kellyAssets) {
      const stateMap = {
        'listening': 'listening',
        'thinking': 'thinking',
        'speaking': 'hello',
        'celebrating': 'hello',
        'idle': 'hello'
      };
      kellyAssets.setState?.(stateMap[expression] || 'thinking');
    }
    
    // Send to Unity WebGL if available
    if (window.unityInstance) {
      try {
        window.unityInstance.SendMessage('kelly_fbx_v4', 'SetExpression', expression);
        
        if (expression === 'speaking' && data.text) {
          window.unityInstance.SendMessage('kelly_fbx_v4', 'StartLipSync', data.text);
        } else if (expression !== 'speaking') {
          window.unityInstance.SendMessage('kelly_fbx_v4', 'StopLipSync');
        }
      } catch (e) {
        // Unity not loaded, ignore
      }
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // CONVERSATION FLOW
  // ═══════════════════════════════════════════════════════════════════
  
  async startConversation() {
    if (this.isActive) {
      console.log('[KellyConversation v2] Already active');
      return;
    }
    
    // Check for agent ID
    if (!this.config.agentId) {
      this.showNotConfiguredMessage();
      return;
    }
    
    console.log('[KellyConversation v2] Starting conversation...');
    
    this.isActive = true;
    this.conversationHistory = [];
    this.conversationId = null;
    
    // Update UI
    this.updateTalkButton('connecting');
    this.showTranscript();
    
    try {
      // Initialize audio
      await this.initAudio();
      
      // Start lip-sync streaming mode
      if (this.lipSyncEnabled && window.KellyLipSync) {
        await window.KellyLipSync.resume();
        window.KellyLipSync.startStreaming();
        console.log('[KellyConversation v2] Lip-sync streaming started');
      }
      
      // Try to get signed URL first, fall back to public agent
      await this.getSignedUrlIfNeeded();
      
      // Connect to ElevenLabs
      await this.connectToElevenLabs();
      
      // Start listening
      this.startListening();
      
    } catch (error) {
      console.error('[KellyConversation v2] Error starting:', error);
      this.showError('Could not start conversation. ' + (error.message || 'Please check microphone permissions.'));
      this.endConversation();
    }
  },
  
  endConversation() {
    console.log('[KellyConversation v2] Ending conversation');
    
    this.isActive = false;
    this.isListening = false;
    this.isSpeaking = false;
    
    // Stop audio
    this.stopListening();
    this.stopAudioPlayback();
    
    // Stop lip-sync
    if (this.lipSyncEnabled && window.KellyLipSync) {
      window.KellyLipSync.stop();
      console.log('[KellyConversation v2] Lip-sync stopped');
    }
    
    // Close WebSocket
    if (this.ws) {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close(1000, 'Conversation ended by user');
      }
      this.ws = null;
    }
    
    // Update UI
    this.updateTalkButton('idle');
    this.onExpression?.('idle');
    
    // Hide transcript after delay
    setTimeout(() => {
      if (!this.isActive) {
        this.hideTranscript();
      }
    }, 3000);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // SIGNED URL HANDLING
  // ═══════════════════════════════════════════════════════════════════
  
  async getSignedUrlIfNeeded() {
    try {
      const response = await fetch('/api/elevenlabs-signed-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.signedUrl) {
          this.config.signedUrl = data.signedUrl;
          this.config.isPublic = false;
          console.log('[KellyConversation v2] Using signed URL (private agent)');
        } else {
          this.config.isPublic = true;
          console.log('[KellyConversation v2] Using public agent connection');
        }
      }
    } catch (e) {
      // If we can't get signed URL, try public connection
      console.log('[KellyConversation v2] Signed URL not available, trying public connection');
      this.config.isPublic = true;
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // AUDIO INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  
  async initAudio() {
    // Create AudioContext
    if (!this.audio.context) {
      this.audio.context = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.audio.sampleRate
      });
    }
    
    // Resume if suspended (browser autoplay policy)
    if (this.audio.context.state === 'suspended') {
      await this.audio.context.resume();
    }
    
    // Create gain node for output
    this.audio.gainNode = this.audio.context.createGain();
    this.audio.gainNode.gain.value = 1.0;
    this.audio.gainNode.connect(this.audio.context.destination);
    
    console.log('[KellyConversation v2] Audio initialized, sample rate:', this.audio.context.sampleRate);
  },
  
  async startListening() {
    try {
      // Get microphone stream
      this.audio.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: this.audio.sampleRate,
          channelCount: 1
        }
      });
      
      // Create media stream source
      const source = this.audio.context.createMediaStreamSource(this.audio.mediaStream);
      
      // Create script processor for capturing audio
      // Note: ScriptProcessorNode is deprecated but widely supported
      // For production, consider using AudioWorklet
      const bufferSize = 4096;
      const processor = this.audio.context.createScriptProcessor(bufferSize, 1, 1);
      
      processor.onaudioprocess = (e) => {
        if (!this.isActive || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.isSpeaking) return; // Don't send audio while Kelly is speaking
        
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert Float32 to Int16 PCM
        const pcm16 = this.float32ToInt16(inputData);
        
        // Convert to base64
        const base64 = this.arrayBufferToBase64(pcm16.buffer);
        
        // Send to ElevenLabs
        this.ws.send(JSON.stringify({
          user_audio_chunk: base64
        }));
      };
      
      source.connect(processor);
      processor.connect(this.audio.context.destination);
      
      this.audio.processor = processor;
      this.audio.source = source;
      
      this.isListening = true;
      this.updateTalkButton('listening');
      this.onExpression?.('listening');
      this.onListeningStart?.();
      
      console.log('[KellyConversation v2] Listening started');
      
    } catch (error) {
      console.error('[KellyConversation v2] Microphone error:', error);
      throw new Error('Microphone access denied. Please allow microphone access.');
    }
  },
  
  stopListening() {
    // Stop media stream
    if (this.audio.mediaStream) {
      this.audio.mediaStream.getTracks().forEach(track => track.stop());
      this.audio.mediaStream = null;
    }
    
    // Disconnect processor
    if (this.audio.processor) {
      this.audio.processor.disconnect();
      this.audio.processor = null;
    }
    
    if (this.audio.source) {
      this.audio.source.disconnect();
      this.audio.source = null;
    }
    
    this.isListening = false;
    this.onListeningEnd?.();
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // WEBSOCKET CONNECTION
  // ═══════════════════════════════════════════════════════════════════
  
  async connectToElevenLabs() {
    return new Promise((resolve, reject) => {
      // Build WebSocket URL
      let wsUrl;
      if (this.config.signedUrl) {
        wsUrl = this.config.signedUrl;
      } else {
        wsUrl = `wss://api.elevenlabs.io/v1/convai/conversation?agent_id=${this.config.agentId}`;
      }
      
      console.log('[KellyConversation v2] Connecting to:', wsUrl.substring(0, 60) + '...');
      
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('[KellyConversation v2] WebSocket connected');
        
        // Send initial configuration
        const initMessage = {
          type: 'conversation_initiation_client_data',
          conversation_config_override: {
            agent: {
              prompt: {
                prompt: this.getSystemPrompt()
              },
              first_message: "Hey there! What's on your mind about today's lesson?",
              language: 'en'
            },
            tts: {
              voice_id: this.config.voiceId
            }
          },
          custom_llm_extra_body: {
            lesson_context: this.lessonContext
          }
        };
        
        this.ws.send(JSON.stringify(initMessage));
        
        // Add initial greeting to transcript
        this.addToTranscript('kelly', "Hey there! What's on your mind about today's lesson?");
        
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        this.handleWSMessage(event);
      };
      
      this.ws.onerror = (error) => {
        console.error('[KellyConversation v2] WebSocket error:', error);
        reject(new Error('Connection failed. The agent may not be available.'));
      };
      
      this.ws.onclose = (event) => {
        console.log('[KellyConversation v2] WebSocket closed:', event.code, event.reason);
        if (this.isActive) {
          // Unexpected close
          if (event.code !== 1000) {
            this.showError('Connection lost. Please try again.');
          }
          this.endConversation();
        }
      };
    });
  },
  
  handleWSMessage(event) {
    try {
      const data = JSON.parse(event.data);
      
      // Log message type
      const logTypes = ['audio', 'ping'];
      if (!logTypes.includes(data.type)) {
        console.log('[KellyConversation v2] Message:', data.type, data);
      }
      
      switch (data.type) {
        case 'conversation_initiation_metadata':
          // Conversation started, store ID
          this.conversationId = data.conversation_id;
          console.log('[KellyConversation v2] Conversation ID:', this.conversationId);
          break;
          
        case 'user_transcript':
          // User's speech transcribed
          const userText = data.user_transcription_event?.user_transcript || 
                          data.user_transcript;
          if (userText) {
            this.addToTranscript('user', userText);
          }
          break;
          
        case 'agent_response':
          // Kelly's response text (before audio)
          const agentText = data.agent_response_event?.agent_response || 
                           data.agent_response;
          if (agentText) {
            this.addToTranscript('kelly', agentText);
            this.onExpression?.('speaking', { text: agentText });
          }
          break;
          
        case 'audio':
          // Kelly's voice audio chunk
          if (data.audio_event?.audio_base_64 || data.audio) {
            const audioData = data.audio_event?.audio_base_64 || data.audio;
            this.queueAudio(audioData);
            
            if (!this.isSpeaking) {
              this.isSpeaking = true;
              this.updateTalkButton('speaking');
              this.onSpeakingStart?.();
            }
          }
          break;
          
        case 'audio_end':
        case 'agent_response_end':
          // Kelly finished speaking
          this.finishAudioPlayback();
          break;
          
        case 'interruption':
          // User interrupted Kelly
          console.log('[KellyConversation v2] User interrupted');
          this.stopAudioPlayback();
          this.onExpression?.('listening');
          break;
          
        case 'ping':
          // Keep-alive
          this.ws.send(JSON.stringify({ type: 'pong' }));
          break;
          
        case 'error':
          console.error('[KellyConversation v2] Server error:', data);
          this.showError(data.message || data.error || 'Connection error');
          break;
          
        case 'internal_tentative_agent_response':
          // Agent is formulating response
          this.onExpression?.('thinking');
          break;
      }
    } catch (e) {
      console.warn('[KellyConversation v2] Message parse error:', e, event.data);
    }
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // AUDIO PLAYBACK
  // ═══════════════════════════════════════════════════════════════════
  
  queueAudio(base64Audio) {
    this.audioQueue.push(base64Audio);
    
    // Feed audio to lip-sync system for real-time mouth animation
    if (this.lipSyncEnabled && window.KellyLipSync) {
      window.KellyLipSync.addAudioChunk(base64Audio);
    }
    
    if (!this.isPlayingAudio) {
      this.playNextAudio();
    }
  },
  
  async playNextAudio() {
    if (this.audioQueue.length === 0) {
      this.isPlayingAudio = false;
      return;
    }
    
    this.isPlayingAudio = true;
    const base64Audio = this.audioQueue.shift();
    
    try {
      // Decode base64 to ArrayBuffer
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // ElevenLabs sends MP3 audio
      const audioBuffer = await this.audio.context.decodeAudioData(bytes.buffer.slice(0));
      
      // Create buffer source
      const source = this.audio.context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audio.gainNode);
      
      source.onended = () => {
        this.playNextAudio();
      };
      
      source.start();
      
    } catch (e) {
      console.warn('[KellyConversation v2] Audio playback error:', e);
      // Try next chunk
      this.playNextAudio();
    }
  },
  
  stopAudioPlayback() {
    this.audioQueue = [];
    this.isPlayingAudio = false;
    this.isSpeaking = false;
    this.onSpeakingEnd?.();
  },
  
  finishAudioPlayback() {
    // Wait for queue to empty
    const checkQueue = () => {
      if (this.audioQueue.length === 0 && !this.isPlayingAudio) {
        this.isSpeaking = false;
        this.updateTalkButton('listening');
        this.onExpression?.('listening');
        this.onSpeakingEnd?.();
      } else {
        setTimeout(checkQueue, 100);
      }
    };
    setTimeout(checkQueue, 100);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // UTILITY FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════
  
  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16Array;
  },
  
  arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // UI MANAGEMENT
  // ═══════════════════════════════════════════════════════════════════
  
  updateTalkButton(state) {
    if (!this.talkButton) return;
    
    // Remove all state classes
    this.talkButton.classList.remove('connecting', 'listening', 'speaking', 'idle');
    
    // Add current state class
    this.talkButton.classList.add(state);
    
    // Update label
    const label = this.talkButton.querySelector('.talk-btn-label');
    const hint = this.talkButton.querySelector('.talk-btn-hint');
    
    if (label) {
      const labels = {
        'idle': 'Talk to Kelly',
        'connecting': 'Connecting...',
        'listening': 'Listening...',
        'speaking': 'Kelly is speaking'
      };
      label.textContent = labels[state] || 'Talk to Kelly';
    }
    
    if (hint) {
      const hints = {
        'idle': 'Ask anything about today\'s lesson',
        'connecting': 'Please wait...',
        'listening': 'Go ahead, I\'m listening!',
        'speaking': 'Tap to end conversation'
      };
      hint.textContent = hints[state] || '';
    }
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
  
  addToTranscript(role, text) {
    if (!text) return;
    
    this.conversationHistory.push({ role, text, timestamp: Date.now() });
    
    if (this.transcriptContainer) {
      const messageEl = document.createElement('div');
      messageEl.className = 'message';
      messageEl.innerHTML = `
        <div class="message-role ${role}">${role === 'kelly' ? '✨ Kelly' : 'You'}</div>
        <div class="message-text">${this.escapeHtml(text)}</div>
      `;
      
      this.transcriptContainer.appendChild(messageEl);
      this.transcriptContainer.scrollTop = this.transcriptContainer.scrollHeight;
    }
  },
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
  // STYLES
  // ═══════════════════════════════════════════════════════════════════
  
  addStyles() {
    if (document.getElementById('kelly-conversation-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-conversation-styles';
    styles.textContent = `
      /* Transcript Container */
      .kelly-transcript {
        position: fixed;
        bottom: 180px;
        right: 20px;
        width: 320px;
        max-height: 250px;
        background: rgba(0, 0, 0, 0.9);
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
        border: 1px solid rgba(255, 255, 255, 0.1);
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
        letter-spacing: 0.5px;
      }
      
      .kelly-transcript .message-role.user {
        color: #f97316;
      }
      
      .kelly-transcript .message-text {
        font-size: 0.9rem;
        color: #e5e5e5;
        line-height: 1.5;
      }
      
      /* Talk Button States */
      .talk-to-kelly-btn.connecting {
        animation: pulse-connecting 1.5s ease-in-out infinite;
      }
      
      .talk-to-kelly-btn.listening {
        background: linear-gradient(135deg, #ef4444, #f97316) !important;
        animation: pulse-listening 1.5s ease-in-out infinite;
      }
      
      .talk-to-kelly-btn.speaking {
        background: linear-gradient(135deg, #10b981, #3b82f6) !important;
      }
      
      @keyframes pulse-connecting {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
      }
      
      @keyframes pulse-listening {
        0%, 100% { box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 4px 40px rgba(239, 68, 68, 0.7); }
      }
      
      /* Toast Notifications */
      .kelly-conversation-toast {
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
        animation: fadeInUp 0.3s ease;
      }
      
      .kelly-conversation-toast strong {
        display: block;
        margin-bottom: 4px;
      }
      
      .kelly-conversation-toast p {
        margin: 0;
        opacity: 0.9;
      }
      
      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      /* Mobile adjustments */
      @media (max-width: 768px) {
        .kelly-transcript {
          bottom: 200px;
          right: 16px;
          left: 16px;
          width: auto;
        }
      }
    `;
    
    document.head.appendChild(styles);
  },
  
  // ═══════════════════════════════════════════════════════════════════
  // ERROR HANDLING
  // ═══════════════════════════════════════════════════════════════════
  
  showNotConfiguredMessage() {
    const toast = document.createElement('div');
    toast.className = 'kelly-conversation-toast';
    toast.innerHTML = `
      <strong>Voice Chat Not Available</strong>
      <p>Kelly's voice chat isn't configured yet. Please check back soon!</p>
    `;
    
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  },
  
  showError(message) {
    const toast = document.createElement('div');
    toast.className = 'kelly-conversation-toast';
    toast.innerHTML = `
      <strong>Oops!</strong>
      <p>${this.escapeHtml(message)}</p>
    `;
    
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  }
};

// ═══════════════════════════════════════════════════════════════════
// AUTO-INITIALIZE
// ═══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // Initialize conversation system
  KellyConversation.init();
  
  // Wire up the Talk to Kelly button
  const talkBtn = document.getElementById('talk-to-kelly-btn');
  if (talkBtn) {
    talkBtn.addEventListener('click', () => {
      if (KellyConversation.isActive) {
        KellyConversation.endConversation();
      } else {
        KellyConversation.startConversation();
      }
    });
    console.log('[KellyConversation v2] Talk button wired up');
  }
  
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

console.log('[KellyConversation v2] ✅ Loaded - Voice conversation system ready');
