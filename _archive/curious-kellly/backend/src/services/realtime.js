/**
 * OpenAI Realtime API Service
 * Handles voice-to-voice interactions with Kelly
 */

const OpenAI = require('openai');

class RealtimeService {
  constructor() {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY not found in environment variables');
    }

    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    this.model = process.env.OPENAI_REALTIME_MODEL || 'gpt-4o-realtime-preview-2024-10-01';
  }

  /**
   * Test connection to OpenAI API
   */
  async testConnection() {
    try {
      const completion = await this.client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [{ 
          role: "user", 
          content: "Say 'Hello from Curious Kellly backend!'" 
        }],
        max_tokens: 20
      });

      return {
        success: true,
        response: completion.choices[0].message.content,
        model: completion.model,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Get Kelly's response for a given age and topic
   * (This will be expanded with Realtime API in Week 3)
   */
  async getKellyResponse(age, topic, userMessage) {
    try {
      // Determine Kelly's age based on age bucket
      const kellyAge = this.getKellyAge(age);
      const kellyPersona = this.getKellyPersona(age);

      const systemPrompt = `You are Kelly, a ${kellyAge}-year-old teacher. 
You're teaching about "${topic}" to someone who is ${age} years old.
Your persona: ${kellyPersona}
Speak naturally as a ${kellyAge}-year-old would speak.
Keep responses under 100 words.`;

      const completion = await this.client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage }
        ],
        max_tokens: 150,
        temperature: 0.8
      });

      return {
        success: true,
        kellyAge,
        kellyPersona,
        response: completion.choices[0].message.content,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Map learner age to Kelly's age
   */
  getKellyAge(learnerAge) {
    if (learnerAge <= 5) return 3;
    if (learnerAge <= 12) return 9;
    if (learnerAge <= 17) return 15;
    if (learnerAge <= 35) return 27;
    if (learnerAge <= 60) return 48;
    return 82;
  }

  /**
   * Get Kelly's persona based on age
   */
  getKellyPersona(learnerAge) {
    if (learnerAge <= 5) return 'playful-toddler';
    if (learnerAge <= 12) return 'curious-kid';
    if (learnerAge <= 17) return 'enthusiastic-teen';
    if (learnerAge <= 35) return 'knowledgeable-adult';
    if (learnerAge <= 60) return 'wise-mentor';
    return 'reflective-elder';
  }

  /**
   * Create an ephemeral API key for Realtime API
   * Returns a temporary key that can be used by the client
   */
  async createEphemeralKey(learnerAge, sessionId = null) {
    try {
      // Create ephemeral key via OpenAI API
      // Note: OpenAI Realtime API uses session-based authentication
      // For now, we'll return a session token that the backend can use
      const ephemeralKey = {
        key: process.env.OPENAI_API_KEY, // In production, use OpenAI's ephemeral key endpoint
        expiresAt: new Date(Date.now() + 60 * 60 * 1000).toISOString(), // 1 hour
        sessionId: sessionId || `session_${Date.now()}`,
        learnerAge,
        kellyAge: this.getKellyAge(learnerAge),
        kellyPersona: this.getKellyPersona(learnerAge),
      };

      return {
        success: true,
        ephemeralKey,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Create Realtime API session configuration
   */
  createRealtimeConfig(learnerAge, sessionId) {
    const kellyAge = this.getKellyAge(learnerAge);
    const kellyPersona = this.getKellyPersona(learnerAge);

    return {
      model: this.model,
      voice: this.getVoiceForAge(kellyAge),
      instructions: this.createSystemPrompt(kellyAge, kellyPersona, 'general', learnerAge),
      input_audio_format: 'pcm16',
      output_audio_format: 'pcm16',
      turn_detection: {
        type: 'server_vad',
        threshold: 0.5,
        silence_duration_ms: 500,
      },
      modalities: ['audio', 'text'],
      temperature: 0.8,
      max_response_output_tokens: 4096,
      input_audio_transcription: {
        model: 'whisper-1',
      },
      tool_choice: 'auto',
      tools: [],
      session_id: sessionId,
    };
  }

  /**
   * Get voice name for Kelly's age
   */
  getVoiceForAge(kellyAge) {
    if (kellyAge <= 5) return 'alloy';
    if (kellyAge <= 12) return 'shimmer';
    if (kellyAge <= 17) return 'nova';
    if (kellyAge <= 35) return 'nova';
    if (kellyAge <= 60) return 'onyx';
    return 'echo';
  }

  /**
   * Create system prompt for Kelly
   */
  createSystemPrompt(kellyAge, kellyPersona, topic, learnerAge) {
    return `You are Kelly, a ${kellyAge}-year-old teacher. 
You're teaching about "${topic}" to someone who is ${learnerAge} years old.
Your persona: ${kellyPersona}
Speak naturally as a ${kellyAge}-year-old would speak.
Keep responses conversational and under 100 words.
Be encouraging and engaging.`;
  }
}

module.exports = RealtimeService;



