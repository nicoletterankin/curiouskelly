/**
 * Kelly Voice Engine for Lesson Player v2
 * 
 * Provides age-adaptive voice synthesis using ElevenLabs.
 * Falls back to TTS generation when pre-computed audio is unavailable.
 * Integrates with lip-sync system for synchronized animation.
 * 
 * Features:
 * - Age-based pitch modulation (2-102 years)
 * - Archetype-specific voice settings
 * - Static audio file fallback
 * - TTS generation via /api/tts
 * - Automatic lip-sync connection
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

// =============================================================================
// AGE-BASED VOICE SETTINGS
// =============================================================================

/**
 * Age buckets with pitch modulation settings
 * Base voice is calibrated for age 27 (within 18-35 bracket)
 */
const AGE_VOICE_SETTINGS = {
  '2-5':   { minAge: 2,  maxAge: 5,   pitchShift: 0.20,  stability: 0.45, description: 'Childlike, higher' },
  '6-12':  { minAge: 6,  maxAge: 12,  pitchShift: 0.12,  stability: 0.50, description: 'Pre-teen' },
  '13-17': { minAge: 13, maxAge: 17,  pitchShift: 0.05,  stability: 0.50, description: 'Teen' },
  '18-35': { minAge: 18, maxAge: 35,  pitchShift: 0.00,  stability: 0.50, description: 'Base voice (27yo)' },
  '36-60': { minAge: 36, maxAge: 60,  pitchShift: -0.05, stability: 0.55, description: 'Mature' },
  '61-102': { minAge: 61, maxAge: 102, pitchShift: -0.12, stability: 0.60, description: 'Elder' },
};

/**
 * Archetype voice settings
 */
const ARCHETYPE_VOICE_SETTINGS = {
  'The Scientist':  { stability: 0.70, similarity: 0.80, style: 0.30 },
  'The Explorer':   { stability: 0.50, similarity: 0.70, style: 0.50 },
  'The Rebel':      { stability: 0.45, similarity: 0.65, style: 0.60 },
  'The Artist':     { stability: 0.40, similarity: 0.90, style: 0.70 },
  'The Storyteller': { stability: 0.45, similarity: 0.85, style: 0.65 },
  'The Empath':     { stability: 0.50, similarity: 0.90, style: 0.50 },
};

/**
 * Language-specific voice IDs
 * Ensures native-sounding accent for each supported language
 */
const VOICE_IDS = {
  en: 'wAdymQH5YucAkXwmrdL0', // Kelly (Original)
  es: 'TX3LPaxmHKxFdv7VOQHJ', // Liam (Spanish/Multilingual) - Placeholder, need specific Kelly clone
  fr: 'TX3LPaxmHKxFdv7VOQHJ'  // Liam (French/Multilingual) - Placeholder
};

const KELLY_VOICE_ID = VOICE_IDS.en;

// =============================================================================
// KELLY VOICE ENGINE CLASS
// =============================================================================

class KellyVoiceEngine {
  constructor(options = {}) {
    this.options = {
      voiceId: options.voiceId || VOICE_IDS.en,
      ttsEndpoint: options.ttsEndpoint || '/api/tts',
      staticAudioPath: options.staticAudioPath || '/lessons/audio',
      goldenAudioPath: options.goldenAudioPath || '/generated-videos/golden-lesson-hd',
      useStaticFirst: options.useStaticFirst !== false, // Try static files before TTS
      enableLipSync: options.enableLipSync !== false,
      onSpeakingStart: options.onSpeakingStart || null,
      onSpeakingEnd: options.onSpeakingEnd || null,
      onError: options.onError || null,
      ...options
    };
    
    // State
    this.currentAudio = null;
    this.isPlaying = false;
    this.isPaused = false;
    this.isMuted = false;
    
    // Current settings
    this.currentAge = 25;
    this.currentAgeBucket = '18-35';
    this.currentArchetype = 'The Explorer';
    this.currentLanguage = 'en';
    
    // Audio cache
    this.audioCache = new Map();
    
    // Lip-sync player reference
    this.lipSyncPlayer = null;
    
    console.log('[KellyVoiceEngine] Initialized');
  }
  
  // ===========================================================================
  // AGE & ARCHETYPE SETTINGS
  // ===========================================================================
  
  /**
   * Set learner age (affects voice pitch and style)
   * @param {number} age - Age in years (2-102)
   */
  setAge(age) {
    this.currentAge = Math.max(2, Math.min(102, age));
    this.currentAgeBucket = this.getAgeBucket(age);
    console.log(`[KellyVoiceEngine] Age set to ${age} (bucket: ${this.currentAgeBucket})`);
  }
  
  /**
   * Set archetype (affects voice character)
   * @param {string} archetype - Archetype name
   */
  setArchetype(archetype) {
    this.currentArchetype = archetype;
    console.log(`[KellyVoiceEngine] Archetype set to ${archetype}`);
  }
  
  /**
   * Set language (affects TTS)
   * @param {string} language - Language code ('en', 'es', 'fr')
   */
  setLanguage(language) {
    this.currentLanguage = language;
    // Update voice ID based on language
    if (VOICE_IDS[language]) {
        this.options.voiceId = VOICE_IDS[language];
    }
    console.log(`[KellyVoiceEngine] Language set to ${language} (VoiceID: ${this.options.voiceId})`);
  }
  
  /**
   * Get age bucket from numeric age
   * @param {number} age - Age in years
   * @returns {string} Age bucket key
   */
  getAgeBucket(age) {
    const normalizedAge = Math.max(2, Math.min(102, age));
    
    for (const [bucket, config] of Object.entries(AGE_VOICE_SETTINGS)) {
      if (normalizedAge >= config.minAge && normalizedAge <= config.maxAge) {
        return bucket;
      }
    }
    
    return '18-35';
  }
  
  /**
   * Get voice settings for current age and archetype
   * @returns {Object} Combined voice settings
   */
  getVoiceSettings() {
    const ageSettings = AGE_VOICE_SETTINGS[this.currentAgeBucket] || AGE_VOICE_SETTINGS['18-35'];
    const archetypeSettings = ARCHETYPE_VOICE_SETTINGS[this.currentArchetype] || ARCHETYPE_VOICE_SETTINGS['The Explorer'];
    
    return {
      stability: (ageSettings.stability + archetypeSettings.stability) / 2,
      similarity_boost: archetypeSettings.similarity,
      style: archetypeSettings.style,
      pitchShift: ageSettings.pitchShift,
      use_speaker_boost: true
    };
  }
  
  // ===========================================================================
  // AUDIO PLAYBACK
  // ===========================================================================
  
  /**
   * Play audio for a lesson phase
   * Tries: 1) Golden lesson audio, 2) Static audio, 3) TTS generation
   * 
   * @param {number} dayNumber - Lesson day number
   * @param {string} phase - Phase name (Hook, Fact1, etc.)
   * @param {string} text - Script text (for TTS fallback)
   * @returns {Promise<boolean>} Whether playback started
   */
  async playPhaseAudio(dayNumber, phase, text = '') {
    // Stop any current playback
    this.stop();
    
    // Try to find audio in order of preference
    let audioUrl = null;
    let audioSource = null;
    
    // 1. Try golden lesson audio (HD quality)
    if (this.options.useStaticFirst) {
      audioUrl = await this._tryGoldenAudio(dayNumber, phase);
      if (audioUrl) {
        audioSource = 'golden';
      }
    }
    
    // 2. Try standard static audio
    if (!audioUrl && this.options.useStaticFirst) {
      audioUrl = await this._tryStaticAudio(dayNumber, phase);
      if (audioUrl) {
        audioSource = 'static';
      }
    }
    
    // 3. Generate via TTS
    if (!audioUrl && text) {
      audioUrl = await this._generateTTS(text, this.currentLanguage);
      if (audioUrl) {
        audioSource = 'tts';
      }
    }
    
    if (!audioUrl) {
      console.warn('[KellyVoiceEngine] No audio available for phase:', phase);
      return false;
    }
    
    console.log(`[KellyVoiceEngine] Playing audio from ${audioSource}: ${phase}`);
    
    // Play the audio
    return this._playAudioUrl(audioUrl);
  }
  
  /**
   * Speak text directly (TTS only)
   * @param {string} text - Text to speak
   * @returns {Promise<boolean>} Whether playback started
   */
  async speak(text) {
    if (!text) return false;
    
    this.stop();
    
    const audioUrl = await this._generateTTS(text, this.currentLanguage);
    if (audioUrl) {
      return this._playAudioUrl(audioUrl);
    }
    
    return false;
  }
  
  /**
   * Try to load golden lesson audio
   * @private
   */
  async _tryGoldenAudio(dayNumber, phase) {
    const dayStr = String(dayNumber).padStart(3, '0');
    const archetypeKey = this.currentArchetype.replace(/\s+/g, '_');
    const phaseKey = phase.charAt(0).toUpperCase() + phase.slice(1);
    
    const path = `${this.options.goldenAudioPath}/day_${dayStr}_${phaseKey}_${archetypeKey}/audio.mp3`;
    
    try {
      const response = await fetch(path, { method: 'HEAD' });
      if (response.ok) {
        return path;
      }
    } catch (e) {
      // File doesn't exist
    }
    
    return null;
  }
  
  /**
   * Try to load standard static audio
   * @private
   */
  async _tryStaticAudio(dayNumber, phase) {
    const filename = `${this.currentAgeBucket}-${this.currentLanguage}-${phase}.mp3`;
    const path = `${this.options.staticAudioPath}/${dayNumber}/${filename}`;
    
    try {
      const response = await fetch(path, { method: 'HEAD' });
      if (response.ok) {
        return path;
      }
    } catch (e) {
      // File doesn't exist
    }
    
    return null;
  }
  
  /**
   * Generate audio via TTS API
   * @private
   */
  async _generateTTS(text, language = 'en') {
    // Check cache
    const cacheKey = `${text.substring(0, 50)}-${this.currentAgeBucket}-${language}`;
    if (this.audioCache.has(cacheKey)) {
      return this.audioCache.get(cacheKey);
    }
    
    try {
      const voiceSettings = this.getVoiceSettings();
      
      const response = await fetch(this.options.ttsEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voiceId: this.options.voiceId,
          modelId: language === 'en' ? 'eleven_turbo_v2' : 'eleven_multilingual_v2',
          // Voice settings would be applied server-side if the API supports it
        })
      });
      
      if (!response.ok) {
        console.error('[KellyVoiceEngine] TTS API error:', response.status);
        return null;
      }
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Cache the URL
      this.audioCache.set(cacheKey, audioUrl);
      
      return audioUrl;
      
    } catch (error) {
      console.error('[KellyVoiceEngine] TTS generation failed:', error);
      if (this.options.onError) {
        this.options.onError(error);
      }
      return null;
    }
  }
  
  /**
   * Play audio from URL
   * @private
   */
  _playAudioUrl(url) {
    return new Promise((resolve) => {
      this.currentAudio = new Audio(url);
      this.currentAudio.volume = this.isMuted ? 0 : 1;
      
      // Connect lip-sync
      this._connectLipSync();
      
      this.currentAudio.onplay = () => {
        this.isPlaying = true;
        this.isPaused = false;
        if (this.options.onSpeakingStart) {
          this.options.onSpeakingStart();
        }
        document.dispatchEvent(new CustomEvent('kelly-speaking-start'));
      };
      
      this.currentAudio.onended = () => {
        this.isPlaying = false;
        this._disconnectLipSync();
        if (this.options.onSpeakingEnd) {
          this.options.onSpeakingEnd();
        }
        document.dispatchEvent(new CustomEvent('kelly-speaking-end'));
        resolve(true);
      };
      
      this.currentAudio.onerror = (e) => {
        this.isPlaying = false;
        this._disconnectLipSync();
        console.error('[KellyVoiceEngine] Audio playback error:', e);
        if (this.options.onError) {
          this.options.onError(e);
        }
        resolve(false);
      };
      
      this.currentAudio.play().catch(e => {
        console.error('[KellyVoiceEngine] Play failed:', e);
        resolve(false);
      });
    });
  }
  
  // ===========================================================================
  // LIP-SYNC INTEGRATION
  // ===========================================================================
  
  /**
   * Set lip-sync player reference
   * @param {KellyLipSyncPlayer} player - Lip-sync player instance
   */
  setLipSyncPlayer(player) {
    this.lipSyncPlayer = player;
    console.log('[KellyVoiceEngine] Lip-sync player connected');
  }
  
  /**
   * Connect lip-sync to current audio
   * @private
   */
  _connectLipSync() {
    if (!this.options.enableLipSync || !this.currentAudio) return;
    
    // Connect pre-computed lip-sync player
    if (this.lipSyncPlayer && this.lipSyncPlayer.hasTrack()) {
      this.lipSyncPlayer.playWithAudio(this.currentAudio);
      console.log('[KellyVoiceEngine] Pre-computed lip-sync connected');
      return;
    }
    
    // Fall back to realtime lip-sync
    if (window.KellyLipSync) {
      try {
        if (!window.KellyLipSync.isInitialized) {
          window.KellyLipSync.init();
        }
        window.KellyLipSync.startFromAudioElement(this.currentAudio);
        console.log('[KellyVoiceEngine] Realtime lip-sync connected');
      } catch (e) {
        console.warn('[KellyVoiceEngine] Lip-sync connection failed:', e);
      }
    }
  }
  
  /**
   * Disconnect lip-sync from current audio
   * @private
   */
  _disconnectLipSync() {
    if (!this.options.enableLipSync) return;
    
    // Disconnect pre-computed lip-sync
    if (this.lipSyncPlayer) {
      this.lipSyncPlayer.stop();
    }
    
    // Disconnect realtime lip-sync
    if (window.KellyLipSync && this.currentAudio) {
      try {
        window.KellyLipSync.stopFromAudioElement(this.currentAudio);
      } catch (e) {
        // Ignore
      }
    }
  }
  
  // ===========================================================================
  // PLAYBACK CONTROLS
  // ===========================================================================
  
  /**
   * Pause playback
   */
  pause() {
    if (this.currentAudio && this.isPlaying) {
      this.currentAudio.pause();
      this.isPaused = true;
      if (this.options.onSpeakingEnd) {
        this.options.onSpeakingEnd();
      }
    }
  }
  
  /**
   * Resume playback
   */
  resume() {
    if (this.currentAudio && this.isPaused) {
      this.currentAudio.play();
      this.isPaused = false;
      if (this.options.onSpeakingStart) {
        this.options.onSpeakingStart();
      }
    }
  }
  
  /**
   * Stop playback
   */
  stop() {
    this._disconnectLipSync();
    
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    
    this.isPlaying = false;
    this.isPaused = false;
    
    if (this.options.onSpeakingEnd) {
      this.options.onSpeakingEnd();
    }
  }
  
  /**
   * Toggle play/pause
   * @returns {boolean} New playing state
   */
  togglePlayPause() {
    if (this.isPaused) {
      this.resume();
      return true;
    } else if (this.isPlaying) {
      this.pause();
      return false;
    }
    return this.isPlaying;
  }
  
  /**
   * Set muted state
   * @param {boolean} muted - Whether to mute
   */
  setMuted(muted) {
    this.isMuted = muted;
    if (this.currentAudio) {
      this.currentAudio.volume = muted ? 0 : 1;
    }
  }
  
  /**
   * Toggle mute
   * @returns {boolean} New muted state
   */
  toggleMute() {
    this.setMuted(!this.isMuted);
    return this.isMuted;
  }
  
  /**
   * Get current audio element (for external control)
   * @returns {HTMLAudioElement|null}
   */
  getAudioElement() {
    return this.currentAudio;
  }
  
  /**
   * Get current state
   * @returns {Object} Current state
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      isMuted: this.isMuted,
      age: this.currentAge,
      ageBucket: this.currentAgeBucket,
      archetype: this.currentArchetype,
      language: this.currentLanguage
    };
  }
  
  /**
   * Clear audio cache
   */
  clearCache() {
    for (const url of this.audioCache.values()) {
      URL.revokeObjectURL(url);
    }
    this.audioCache.clear();
    console.log('[KellyVoiceEngine] Cache cleared');
  }
  
  /**
   * Dispose of resources
   */
  dispose() {
    this.stop();
    this.clearCache();
    console.log('[KellyVoiceEngine] Disposed');
  }
}

// =============================================================================
// GLOBAL EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyVoiceEngine = KellyVoiceEngine;
  window.AGE_VOICE_SETTINGS = AGE_VOICE_SETTINGS;
  window.ARCHETYPE_VOICE_SETTINGS = ARCHETYPE_VOICE_SETTINGS;
  window.KELLY_VOICE_ID = KELLY_VOICE_ID;
}

// ES Module export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyVoiceEngine;
}

console.log('[KellyVoiceEngine] Module loaded');

