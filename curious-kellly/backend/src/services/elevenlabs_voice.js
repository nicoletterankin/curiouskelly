/**
 * ElevenLabs Voice Service
 * Real-time voice synthesis using ElevenLabs API
 * Week 3-4 Implementation
 */

const fetch = require('node-fetch');

class ElevenLabsVoiceService {
  constructor() {
    this.apiKey = process.env.ELEVENLABS_API_KEY;
    this.voiceId = 'wAdymQH5YucAkXwmrdL0'; // Kelly voice
    this.modelId = 'eleven_multilingual_v2';

    if (!this.apiKey) {
      console.warn('⚠️  ELEVENLABS_API_KEY not set - voice synthesis disabled');
    }
  }

  /**
   * Voice settings optimized for different Kelly ages
   */
  getVoiceSettings(kellyAge) {
    const configs = {
      3: {  // Toddler Kelly
        stability: 0.5,
        similarity_boost: 0.8,
        style: 0.2,
        use_speaker_boost: true
      },
      9: {  // Kid Kelly
        stability: 0.6,
        similarity_boost: 0.8,
        style: 0.1,
        use_speaker_boost: true
      },
      15: {  // Teen Kelly
        stability: 0.65,
        similarity_boost: 0.8,
        style: 0.15,
        use_speaker_boost: true
      },
      27: {  // Adult Kelly (default)
        stability: 0.6,
        similarity_boost: 0.8,
        style: 0.0,
        use_speaker_boost: true
      },
      48: {  // Mentor Kelly
        stability: 0.7,
        similarity_boost: 0.75,
        style: 0.0,
        use_speaker_boost: true
      },
      82: {  // Elder Kelly
        stability: 0.75,
        similarity_boost: 0.7,
        style: 0.0,
        use_speaker_boost: true
      }
    };

    return configs[kellyAge] || configs[27];
  }

  /**
   * Generate speech audio from text
   * @param {string} text - Text to convert to speech
   * @param {number} kellyAge - Kelly's age (determines voice characteristics)
   * @param {string} language - Language code (en, es, fr)
   * @returns {Promise<Buffer>} Audio buffer (MP3)
   */
  async generateSpeech(text, kellyAge = 27, language = 'en') {
    if (!this.apiKey) {
      throw new Error('ElevenLabs API key not configured');
    }

    const url = `https://api.elevenlabs.io/v1/text-to-speech/${this.voiceId}`;
    const voiceSettings = this.getVoiceSettings(kellyAge);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': this.apiKey
      },
      body: JSON.stringify({
        text,
        model_id: this.modelId,
        voice_settings: voiceSettings
      })
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`ElevenLabs API error (${response.status}): ${error}`);
    }

    return await response.buffer();
  }

  /**
   * Stream speech audio (for realtime applications)
   * @param {string} text - Text to convert
   * @param {number} kellyAge - Kelly's age
   * @returns {Promise<ReadableStream>} Audio stream
   */
  async streamSpeech(text, kellyAge = 27) {
    if (!this.apiKey) {
      throw new Error('ElevenLabs API key not configured');
    }

    const url = `https://api.elevenlabs.io/v1/text-to-speech/${this.voiceId}/stream`;
    const voiceSettings = this.getVoiceSettings(kellyAge);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': this.apiKey
      },
      body: JSON.stringify({
        text,
        model_id: this.modelId,
        voice_settings: voiceSettings
      })
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`ElevenLabs API error (${response.status}): ${error}`);
    }

    return response.body;
  }

  /**
   * Get voice metadata and available voices
   */
  async getVoiceInfo() {
    if (!this.apiKey) {
      throw new Error('ElevenLabs API key not configured');
    }

    const url = `https://api.elevenlabs.io/v1/voices/${this.voiceId}`;

    const response = await fetch(url, {
      headers: {
        'xi-api-key': this.apiKey
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to get voice info: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Test if API key is valid
   */
  async testConnection() {
    try {
      await this.getVoiceInfo();
      return { success: true, message: 'ElevenLabs API connected successfully' };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }
}

module.exports = ElevenLabsVoiceService;
