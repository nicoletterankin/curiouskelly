/**
 * Voice Service - OpenAI Realtime API Integration
 * Handles voice-to-voice conversations with Kelly
 * 
 * NOTE: Requires OpenAI Realtime API beta access
 * Current implementation uses text-based fallback until access granted
 */

const OpenAI = require('openai');

class VoiceService {
  constructor() {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY not found in environment variables');
    }

    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    this.realtimeModel = process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview-2024-10-01';
    
    // Voice characteristics for different Kelly ages
    this.voiceConfigs = {
      3: {  // Toddler Kelly
        pitch: 1.3,
        speed: 0.9,
        voice: 'alloy',  // OpenAI voice
        characteristics: 'playful, high-pitched, enthusiastic'
      },
      9: {  // Kid Kelly
        pitch: 1.15,
        speed: 1.0,
        voice: 'shimmer',
        characteristics: 'curious, energetic, friendly'
      },
      15: {  // Teen Kelly
        pitch: 1.05,
        speed: 1.1,
        voice: 'nova',
        characteristics: 'enthusiastic, relatable, modern'
      },
      27: {  // Adult Kelly
        pitch: 1.0,
        speed: 1.0,
        voice: 'nova',
        characteristics: 'professional, warm, knowledgeable'
      },
      48: {  // Mentor Kelly
        pitch: 0.95,
        speed: 0.95,
        voice: 'onyx',
        characteristics: 'wise, patient, experienced'
      },
      82: {  // Elder Kelly
        pitch: 0.9,
        speed: 0.85,
        voice: 'echo',
        characteristics: 'reflective, gentle, thoughtful'
      }
    };
  }

  /**
   * Get voice configuration for Kelly's age
   */
  getVoiceConfig(kellyAge) {
    return this.voiceConfigs[kellyAge] || this.voiceConfigs[27];
  }

  /**
   * Create system prompt for Kelly
   */
  createKellyPrompt(kellyAge, kellyPersona, topic, learnerAge) {
    const voiceConfig = this.getVoiceConfig(kellyAge);

    return `You are Kelly, a ${kellyAge}-year-old teacher who is teaching about "${topic}".

Your Persona: ${kellyPersona}
Voice Characteristics: ${voiceConfig.characteristics}
Your Student's Age: ${learnerAge}

Speaking Style:
- Speak as a ${kellyAge}-year-old would naturally speak
- Match the energy and vocabulary of someone your age
- Be enthusiastic about learning and teaching
- Keep responses conversational and under 150 words
- Use age-appropriate examples and analogies

Your Goal:
- Make learning fun and engaging
- Adapt explanations to the student's age
- Ask questions to keep them involved
- Celebrate their curiosity
- Never talk down to them - meet them where they are

Remember: You're ${kellyAge} years old, so speak authentically as someone that age!`;
  }

  /**
   * REALTIME API: Start voice session
   * (This will use WebRTC/WebSocket when Realtime API access granted)
   */
  async startRealtimeSession(sessionConfig) {
    const {
      age,
      topic,
      kellyAge,
      kellyPersona,
      sessionId
    } = sessionConfig;

    // In production with Realtime API access, this would:
    // 1. Create WebRTC/WebSocket connection
    // 2. Set up audio streaming
    // 3. Configure Kelly's voice parameters
    // 4. Return connection details

    // For now, return session configuration
    const voiceConfig = this.getVoiceConfig(kellyAge);
    const systemPrompt = this.createKellyPrompt(kellyAge, kellyPersona, topic, age);

    return {
      sessionId,
      type: 'realtime',
      status: 'ready',
      config: {
        model: this.realtimeModel,
        voice: voiceConfig.voice,
        pitch: voiceConfig.pitch,
        speed: voiceConfig.speed,
        systemPrompt,
        turnDetection: {
          type: 'server_vad',  // Voice Activity Detection
          threshold: 0.5,
          silence_duration_ms: 500
        },
        interruption: {
          enabled: true,  // Barge-in support
          type: 'truncate'
        }
      },
      instructions: {
        client: 'Connect via WebRTC or WebSocket to begin voice conversation',
        endpoint: '/api/voice/realtime/connect',
        protocol: 'WebRTC'
      }
    };
  }

  /**
   * TEXT FALLBACK: Get Kelly's text response
   * (Used until Realtime API access granted)
   */
  async getTextResponse(sessionConfig, userMessage, metadata = {}) {
    const startedAt = metadata.requestStartedAt
      ? new Date(metadata.requestStartedAt)
      : new Date();
    const startedAtIso = startedAt.toISOString();
    const startedAtMs = startedAt.getTime();

    const {
      age,
      topic,
      kellyAge,
      kellyPersona
    } = sessionConfig;

    const systemPrompt = this.createKellyPrompt(kellyAge, kellyPersona, topic, age);

    try {
      const completion = await this.client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage }
        ],
        max_tokens: 200,
        temperature: 0.8
      });

      const textResponse = completion.choices[0].message.content;
      const completedAt = new Date();
      const latencyMs = completedAt.getTime() - startedAtMs;

      await logVoiceLatency({
        sessionId: metadata.sessionId,
        source: metadata.source || 'voice_text_fallback',
        topic,
        learnerAge: age,
        ageBucket: this.getAgeBucket(age),
        kellyAge,
        kellyPersona,
        requestStartedAt: startedAtIso,
        responseSentAt: completedAt.toISOString(),
        latencyMs,
        status: 'ok'
      });

      return {
        success: true,
        type: 'text',
        response: textResponse,
        kellyAge,
        kellyPersona,
        voiceConfig: this.getVoiceConfig(kellyAge),
        note: 'Text fallback - upgrade to Realtime API for voice',
        timestamp: completedAt.toISOString(),
        latencyMs
      };
    } catch (error) {
      const failedAt = new Date();
      const latencyMs = failedAt.getTime() - startedAtMs;

      await logVoiceLatency({
        sessionId: metadata.sessionId,
        source: metadata.source || 'voice_text_fallback',
        topic,
        learnerAge: age,
        ageBucket: this.getAgeBucket(age),
        kellyAge,
        kellyPersona,
        requestStartedAt: startedAtIso,
        responseSentAt: failedAt.toISOString(),
        latencyMs,
        status: 'error'
      });

      return {
        success: false,
        error: error.message,
        timestamp: failedAt.toISOString(),
        latencyMs
      };
    }
  }

  /**
   * Map learner age to age bucket for analytics.
   */
  getAgeBucket(learnerAge) {
    if (learnerAge >= 2 && learnerAge <= 5) return '2-5';
    if (learnerAge >= 6 && learnerAge <= 12) return '6-12';
    if (learnerAge >= 13 && learnerAge <= 17) return '13-17';
    if (learnerAge >= 18 && learnerAge <= 35) return '18-35';
    if (learnerAge >= 36 && learnerAge <= 60) return '36-60';
    if (learnerAge >= 61 && learnerAge <= 102) return '61-102';
    return 'unknown';
  }

  /**
   * ELEVENLABS FALLBACK: Generate speech from text
   * (Used until Realtime API access granted)
   */
  async generateSpeech(text, kellyAge) {
    const voiceConfig = this.getVoiceConfig(kellyAge);

    if (!process.env.ELEVENLABS_API_KEY) {
      throw new Error('ELEVENLABS_API_KEY is missing. Add it to backend/.env');
    }

    const voiceId = process.env.ELEVENLABS_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: 'POST',
        headers: {
          Accept: 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': process.env.ELEVENLABS_API_KEY
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.55,
            similarity_boost: 0.7,
            style: 0.1,
            use_speaker_boost: true
          }
        })
      }
    );

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`ElevenLabs error ${response.status}: ${errorBody}`);
    }

    const audioBuffer = await response.arrayBuffer();

    return {
      success: true,
      mimeType: 'audio/mpeg',
      audio: Buffer.from(audioBuffer),
      voiceConfig
    };
  }

  /**
   * Measure latency for voice round-trip
   */
  async measureLatency(testConfig) {
    const start = Date.now();

    // Simulate realtime interaction
    await this.getTextResponse(testConfig, 'Hello Kelly!', {
      source: 'voice_measure_latency',
      requestStartedAt: new Date(start).toISOString()
    });

    const latency = Date.now() - start;

    return {
      latency,
      target: 600,
      status: latency < 600 ? 'pass' : 'fail',
      note: 'Text-based measurement. Realtime API will be faster.'
    };
  }

  /**
   * WebRTC connection handler (for future implementation)
   */
  async handleWebRTCConnection(sessionId, offer) {
    // This will be implemented when Realtime API access is granted
    // For now, return placeholder
    
    return {
      status: 'not_implemented',
      message: 'WebRTC connection requires OpenAI Realtime API beta access',
      sessionId,
      nextSteps: [
        '1. Request Realtime API beta access from OpenAI',
        '2. Update this handler with WebRTC implementation',
        '3. Configure audio streaming infrastructure',
        '4. Test with mobile clients'
      ]
    };
  }

  /**
   * Handle barge-in (user interrupts Kelly)
   */
  async handleInterruption(sessionId) {
    // In Realtime API, this would:
    // 1. Detect user speech during Kelly's response
    // 2. Truncate Kelly's output
    // 3. Switch to listening mode
    // 4. Process new user input

    return {
      action: 'truncate',
      sessionId,
      status: 'ready_for_input',
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = VoiceService;


