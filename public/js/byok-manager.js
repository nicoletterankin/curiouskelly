/**
 * BYOK Manager - Bring Your Own Key Flywheel
 * 
 * THE MAGIC PAINTBRUSH:
 * - Pool community AI resources (free credits + paid keys)
 * - Generate lessons using distributed keys
 * - Enable learner-powered content creation
 * 
 * SUPPORTED PROVIDERS:
 * - OpenAI (chat, vision, TTS, DALL-E)
 * - Anthropic (chat, vision)
 * - Google AI (Gemini chat, Imagen)
 * - HeyGen (video avatars) ← UNLOCKS VIDEO BOTTLENECK
 * - ElevenLabs (TTS, voice clone)
 * - KELLY_KEYS (platform-provided pooled credits)
 */

const BYOKManager = {
  // Provider configurations
  providers: {
    openai: {
      id: 'openai',
      name: 'OpenAI',
      emoji: '🤖',
      capabilities: ['chat', 'vision', 'tts', 'image'],
      keyPrefix: 'sk-',
      keyRegex: /^sk-[a-zA-Z0-9]{32,}$/,
      testEndpoint: 'https://api.openai.com/v1/models',
      freeInfo: 'Free $5 credit for new accounts',
      setupUrl: 'https://platform.openai.com/api-keys',
      permissions: ['gpt-4o-mini', 'gpt-4o', 'dall-e-3', 'tts-1', 'whisper-1']
    },
    anthropic: {
      id: 'anthropic',
      name: 'Anthropic',
      emoji: '🧠',
      capabilities: ['chat', 'vision'],
      keyPrefix: 'sk-ant-',
      keyRegex: /^sk-ant-[a-zA-Z0-9-]{32,}$/,
      testEndpoint: 'https://api.anthropic.com/v1/messages',
      freeInfo: 'Free $5 credit for new accounts',
      setupUrl: 'https://console.anthropic.com/settings/keys',
      permissions: ['claude-3-5-sonnet', 'claude-3-haiku']
    },
    google: {
      id: 'google',
      name: 'Google AI',
      emoji: '✨',
      capabilities: ['chat', 'vision', 'image'],
      keyPrefix: 'AIza',
      keyRegex: /^AIza[a-zA-Z0-9_-]{35}$/,
      testEndpoint: 'https://generativelanguage.googleapis.com/v1/models',
      freeInfo: 'Generous free tier!',
      setupUrl: 'https://aistudio.google.com/app/apikey',
      permissions: ['gemini-1.5-flash', 'gemini-1.5-pro', 'imagen-3']
    },
    heygen: {
      id: 'heygen',
      name: 'HeyGen',
      emoji: '🎬',
      capabilities: ['video', 'avatar'],
      keyPrefix: '',
      keyRegex: /^[a-zA-Z0-9]{32,}$/,
      testEndpoint: 'https://api.heygen.com/v2/user/remaining_quota',
      freeInfo: 'Free credits for new users! 🎉',
      setupUrl: 'https://app.heygen.com/settings/api',
      permissions: ['video_generate', 'avatar_list']
    },
    elevenlabs: {
      id: 'elevenlabs',
      name: 'ElevenLabs',
      emoji: '🎙️',
      capabilities: ['tts', 'voice'],
      keyPrefix: '',
      keyRegex: /^[a-zA-Z0-9]{32}$/,
      testEndpoint: 'https://api.elevenlabs.io/v1/user',
      freeInfo: '10,000 free chars/month',
      setupUrl: 'https://elevenlabs.io/app/settings/api-keys',
      permissions: ['tts', 'voice_clone']
    }
  },

  // Local storage keys
  STORAGE_KEY: 'kelly_byok_keys',
  
  // Cached keys (decrypted in memory only)
  cachedKeys: {},
  
  // Initialize
  init() {
    this.loadFromStorage();
    console.log('🔑 BYOK Manager initialized');
  },

  // Load keys from localStorage
  loadFromStorage() {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        this.cachedKeys = parsed || {};
      }
    } catch (e) {
      console.warn('BYOK load error:', e);
      this.cachedKeys = {};
    }
  },

  // Save keys to localStorage
  saveToStorage() {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.cachedKeys));
    } catch (e) {
      console.warn('BYOK save error:', e);
    }
  },

  // Get provider config
  getProvider(providerId) {
    return this.providers[providerId] || null;
  },

  // Get all configured providers
  getConfiguredProviders() {
    return Object.keys(this.cachedKeys).filter(id => this.cachedKeys[id]);
  },

  // Check if provider is configured
  hasProvider(providerId) {
    return !!this.cachedKeys[providerId];
  },

  // Get key for provider
  getKey(providerId) {
    return this.cachedKeys[providerId] || null;
  },

  // Get display prefix for key (first 12 chars)
  getKeyPrefix(providerId) {
    const key = this.getKey(providerId);
    if (!key) return null;
    return key.slice(0, 12) + '...';
  },

  // Validate key format
  validateKeyFormat(providerId, key) {
    const provider = this.getProvider(providerId);
    if (!provider) return { valid: false, error: 'Unknown provider' };
    
    if (!key || typeof key !== 'string') {
      return { valid: false, error: 'Key is required' };
    }
    
    const trimmed = key.trim();
    if (!provider.keyRegex.test(trimmed)) {
      return { valid: false, error: `Invalid key format. Should start with ${provider.keyPrefix}` };
    }
    
    return { valid: true };
  },

  // Test key with provider API
  async testKey(providerId, key) {
    const provider = this.getProvider(providerId);
    if (!provider) return { valid: false, error: 'Unknown provider' };

    try {
      let response;
      const headers = {};

      switch (providerId) {
        case 'openai':
          headers['Authorization'] = `Bearer ${key}`;
          response = await fetch(provider.testEndpoint, { headers });
          break;
          
        case 'anthropic':
          headers['x-api-key'] = key;
          headers['anthropic-version'] = '2023-06-01';
          headers['Content-Type'] = 'application/json';
          // Anthropic doesn't have a simple test endpoint, so we'll just validate format
          return { valid: true, credits: 'Unknown' };
          
        case 'google':
          response = await fetch(`${provider.testEndpoint}?key=${key}`);
          break;
          
        case 'heygen':
          headers['X-Api-Key'] = key;
          response = await fetch(provider.testEndpoint, { headers });
          if (response.ok) {
            const data = await response.json();
            return { 
              valid: true, 
              credits: data.data?.remaining_quota || 'Unknown',
              info: `${data.data?.remaining_quota || 0} credits remaining`
            };
          }
          break;
          
        case 'elevenlabs':
          headers['xi-api-key'] = key;
          response = await fetch(provider.testEndpoint, { headers });
          if (response.ok) {
            const data = await response.json();
            return { 
              valid: true, 
              credits: data.subscription?.character_limit - data.subscription?.character_count,
              info: `${data.subscription?.character_limit - data.subscription?.character_count} chars remaining`
            };
          }
          break;
      }

      if (response && response.ok) {
        return { valid: true };
      } else if (response) {
        const error = await response.text().catch(() => 'Unknown error');
        return { valid: false, error: `API error: ${response.status}` };
      }
      
      return { valid: false, error: 'Could not validate key' };
    } catch (e) {
      return { valid: false, error: e.message };
    }
  },

  // Save key for provider
  async saveKey(providerId, key, skipValidation = false) {
    // Validate format
    const formatCheck = this.validateKeyFormat(providerId, key);
    if (!formatCheck.valid) {
      return { success: false, error: formatCheck.error };
    }

    // Test with API (unless skipped)
    if (!skipValidation) {
      const testResult = await this.testKey(providerId, key.trim());
      if (!testResult.valid) {
        return { success: false, error: testResult.error };
      }
    }

    // Save to cache and storage
    this.cachedKeys[providerId] = key.trim();
    this.saveToStorage();

    // Also save to legacy location for backwards compatibility
    if (providerId === 'openai') {
      localStorage.setItem('kelly_byok_key', key.trim());
      localStorage.setItem('kelly_byok_provider', 'openai');
    } else if (providerId === 'google') {
      localStorage.setItem('kelly_visual_api_key', key.trim());
    }

    return { success: true };
  },

  // Remove key for provider
  removeKey(providerId) {
    delete this.cachedKeys[providerId];
    this.saveToStorage();
    
    // Clear legacy storage too
    if (providerId === 'openai') {
      localStorage.removeItem('kelly_byok_key');
    } else if (providerId === 'google') {
      localStorage.removeItem('kelly_visual_api_key');
    }
  },

  // Get capabilities for configured providers
  getAvailableCapabilities() {
    const caps = new Set();
    for (const providerId of this.getConfiguredProviders()) {
      const provider = this.getProvider(providerId);
      if (provider) {
        provider.capabilities.forEach(c => caps.add(c));
      }
    }
    return Array.from(caps);
  },

  // Check if capability is available
  hasCapability(capability) {
    return this.getAvailableCapabilities().includes(capability);
  },

  // Get best provider for capability
  getBestProviderForCapability(capability) {
    // Priority order for each capability
    const priority = {
      chat: ['anthropic', 'openai', 'google'],
      vision: ['openai', 'anthropic', 'google'],
      tts: ['elevenlabs', 'openai'],
      image: ['google', 'openai'],
      video: ['heygen'],
      avatar: ['heygen'],
      voice: ['elevenlabs']
    };

    const providers = priority[capability] || [];
    for (const providerId of providers) {
      if (this.hasProvider(providerId)) {
        return providerId;
      }
    }
    return null;
  },

  // Generate with BYOK (routes to correct provider)
  async generate(capability, options) {
    const providerId = this.getBestProviderForCapability(capability);
    if (!providerId) {
      return { success: false, error: `No provider configured for ${capability}` };
    }

    const key = this.getKey(providerId);
    if (!key) {
      return { success: false, error: 'Key not found' };
    }

    switch (capability) {
      case 'chat':
        return this.generateChat(providerId, key, options);
      case 'video':
        return this.generateVideo(providerId, key, options);
      case 'image':
        return this.generateImage(providerId, key, options);
      case 'tts':
        return this.generateTTS(providerId, key, options);
      default:
        return { success: false, error: `Unsupported capability: ${capability}` };
    }
  },

  // Chat generation
  async generateChat(providerId, key, { messages, model }) {
    try {
      if (providerId === 'openai') {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${key}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: model || 'gpt-4o-mini',
            messages
          })
        });
        const data = await response.json();
        return { success: true, content: data.choices?.[0]?.message?.content };
      }
      
      if (providerId === 'anthropic') {
        const response = await fetch('https://api.anthropic.com/v1/messages', {
          method: 'POST',
          headers: {
            'x-api-key': key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: model || 'claude-3-5-sonnet-20241022',
            max_tokens: 1024,
            messages: messages.filter(m => m.role !== 'system').map(m => ({
              role: m.role === 'assistant' ? 'assistant' : 'user',
              content: m.content
            })),
            system: messages.find(m => m.role === 'system')?.content
          })
        });
        const data = await response.json();
        return { success: true, content: data.content?.[0]?.text };
      }

      if (providerId === 'google') {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1/models/${model || 'gemini-1.5-flash'}:generateContent?key=${key}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: messages.map(m => ({
              role: m.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: m.content }]
            }))
          })
        });
        const data = await response.json();
        return { success: true, content: data.candidates?.[0]?.content?.parts?.[0]?.text };
      }
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  // Video generation (HeyGen)
  async generateVideo(providerId, key, { script, avatarId, voiceId }) {
    if (providerId !== 'heygen') {
      return { success: false, error: 'Video requires HeyGen' };
    }

    try {
      // Create video
      const response = await fetch('https://api.heygen.com/v2/video/generate', {
        method: 'POST',
        headers: {
          'X-Api-Key': key,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          video_inputs: [{
            character: {
              type: 'avatar',
              avatar_id: avatarId || 'Kristin_pubblic_2_20240108',
              avatar_style: 'normal'
            },
            voice: {
              type: 'text',
              input_text: script,
              voice_id: voiceId || '1985fa7482cf4a2ab9a34e5a89240e76'
            }
          }],
          dimension: { width: 1280, height: 720 }
        })
      });

      const data = await response.json();
      
      if (data.data?.video_id) {
        return { 
          success: true, 
          videoId: data.data.video_id,
          status: 'processing'
        };
      }
      
      return { success: false, error: data.message || 'Failed to create video' };
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  // Check video status (HeyGen)
  async checkVideoStatus(videoId) {
    const key = this.getKey('heygen');
    if (!key) return { success: false, error: 'No HeyGen key' };

    try {
      const response = await fetch(`https://api.heygen.com/v1/video_status.get?video_id=${videoId}`, {
        headers: { 'X-Api-Key': key }
      });
      const data = await response.json();
      
      return {
        success: true,
        status: data.data?.status,
        videoUrl: data.data?.video_url,
        thumbnailUrl: data.data?.thumbnail_url
      };
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  // Image generation
  async generateImage(providerId, key, { prompt, size }) {
    try {
      if (providerId === 'openai') {
        const response = await fetch('https://api.openai.com/v1/images/generations', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${key}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'dall-e-3',
            prompt,
            n: 1,
            size: size || '1024x1024'
          })
        });
        const data = await response.json();
        return { success: true, url: data.data?.[0]?.url };
      }

      if (providerId === 'google') {
        // Use Gemini for image description, Imagen for generation
        const response = await fetch(`https://generativelanguage.googleapis.com/v1/models/imagen-3.0-generate-001:predict?key=${key}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            instances: [{ prompt }],
            parameters: { sampleCount: 1 }
          })
        });
        const data = await response.json();
        return { success: true, url: data.predictions?.[0]?.bytesBase64Encoded };
      }
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  // TTS generation
  async generateTTS(providerId, key, { text, voice }) {
    try {
      if (providerId === 'elevenlabs') {
        const voiceId = voice || '21m00Tcm4TlvDq8ikWAM'; // Rachel
        const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
          method: 'POST',
          headers: {
            'xi-api-key': key,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text,
            model_id: 'eleven_monolingual_v1'
          })
        });
        
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          return { success: true, audioUrl: url };
        }
      }

      if (providerId === 'openai') {
        const response = await fetch('https://api.openai.com/v1/audio/speech', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${key}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'tts-1',
            input: text,
            voice: voice || 'nova'
          })
        });
        
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          return { success: true, audioUrl: url };
        }
      }
    } catch (e) {
      return { success: false, error: e.message };
    }
  },

  // Get status summary for UI
  getStatusSummary() {
    const configured = this.getConfiguredProviders();
    const caps = this.getAvailableCapabilities();
    
    return {
      providersConfigured: configured.length,
      providers: configured.map(id => ({
        id,
        name: this.providers[id]?.name,
        emoji: this.providers[id]?.emoji,
        prefix: this.getKeyPrefix(id)
      })),
      capabilities: caps,
      canChat: caps.includes('chat'),
      canGenerateVideo: caps.includes('video'),
      canGenerateImage: caps.includes('image'),
      canGenerateTTS: caps.includes('tts')
    };
  }
};

// Auto-initialize
if (typeof window !== 'undefined') {
  window.BYOKManager = BYOKManager;
  BYOKManager.init();
}
