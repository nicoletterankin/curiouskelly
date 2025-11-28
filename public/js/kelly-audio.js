/**
 * Kelly Audio System
 *
 * Handles Kelly's voice playback with:
 * - ElevenLabs TTS ONLY (browser TTS is PROHIBITED)
 * - Pre-generated audio file support
 * - Play/pause/mute controls
 * - Lip-sync signaling
 *
 * ⚠️ IMPORTANT: Browser TTS is NEVER used.
 * If no ElevenLabs key, Kelly shows text only (silent mode).
 */

class KellyAudio {
  constructor(options = {}) {
    this.options = {
      elevenLabsApiKey: options.elevenLabsApiKey || null,
      kellyVoiceId: options.kellyVoiceId || 'EXAVITQu4vr4xnSDxMaL', // Kelly's trained voice
      audioBasePath: options.audioBasePath || '/audio/lessons/', // Pre-generated audio
      onSpeakingStart: options.onSpeakingStart || null,
      onSpeakingEnd: options.onSpeakingEnd || null,
      onWordSpoken: options.onWordSpoken || null,
      ...options
    };

    this.currentAudio = null;
    this.isMuted = false;
    this.isPlaying = false;
    this.isPaused = false;
    this.currentText = '';
    this.audioCache = new Map();

    // ⚠️ NO BROWSER TTS - EVER
    // If no ElevenLabs key, we run in silent mode (text only)
    this.hasVoice = !!this.options.elevenLabsApiKey;

    console.log('[KellyAudio] Initialized', {
      elevenLabs: this.hasVoice,
      mode: this.hasVoice ? 'VOICE' : 'SILENT (no ElevenLabs key)'
    });

    if (!this.hasVoice) {
      console.warn(
        '[KellyAudio] ⚠️ No ElevenLabs API key - running in SILENT mode. Browser TTS is PROHIBITED.'
      );
    }
  }

  /**
   * Speak text using Kelly's voice (ElevenLabs ONLY)
   *
   * ⚠️ BROWSER TTS IS PROHIBITED - NEVER USED
   */
  async speak(text, options = {}) {
    if (!text) {
      this._notifySpeakingEnd();
      return;
    }

    // Stop any current speech
    this.stop();

    this.currentText = text;
    this.isPlaying = true;
    this.isPaused = false;

    // Notify speaking started (for avatar lip-sync)
    this._notifySpeakingStart();

    try {
      // Check for pre-generated audio file first
      const preGenerated = await this._tryPreGeneratedAudio(text, options);
      if (preGenerated) {
        return;
      }

      // Use ElevenLabs if API key available
      if (this.options.elevenLabsApiKey && !this.isMuted) {
        await this._speakWithElevenLabs(text, options);
      } else {
        // SILENT MODE - No voice, just timing for lip-sync animation
        // ⚠️ NO BROWSER TTS FALLBACK - THIS IS INTENTIONAL
        await this._silentMode(text);
      }
    } catch (error) {
      console.warn('[KellyAudio] ElevenLabs error:', error);
      // Fall to silent mode - NEVER to browser TTS
      await this._silentMode(text);
    }
  }

  /**
   * Try to load pre-generated audio file
   */
  async _tryPreGeneratedAudio(text, options) {
    // Check if we have a pre-generated audio URL in options
    if (options.audioUrl) {
      try {
        await this._playAudioUrl(options.audioUrl);
        return true;
      } catch (e) {
        console.warn('[KellyAudio] Pre-generated audio failed:', e);
        return false;
      }
    }
    return false;
  }

  /**
   * Play audio from URL
   */
  _playAudioUrl(url) {
    return new Promise((resolve, reject) => {
      this.currentAudio = new Audio(url);
      this.currentAudio.volume = this.isMuted ? 0 : 1;

      this.currentAudio.onended = () => {
        this.isPlaying = false;
        this._notifySpeakingEnd();
        resolve();
      };

      this.currentAudio.onerror = (e) => {
        this.isPlaying = false;
        this._notifySpeakingEnd();
        reject(e);
      };

      this.currentAudio.play().catch(reject);
    });
  }

  /**
   * Silent mode - animates lip-sync without audio
   * Used when no ElevenLabs key is available
   * ⚠️ THIS IS NOT BROWSER TTS - IT'S COMPLETELY SILENT
   */
  async _silentMode(text) {
    // Estimate speaking duration (~150 words per minute)
    const words = text.split(/\s+/).length;
    const duration = Math.max(2000, (words / 150) * 60000);

    // Simulate word timing for lip-sync animation
    const wordList = text.split(/\s+/);
    const wordInterval = duration / wordList.length;

    let wordIndex = 0;
    const interval = setInterval(() => {
      if (wordIndex < wordList.length && this.isPlaying && !this.isPaused) {
        if (this.options.onWordSpoken) {
          this.options.onWordSpoken(wordList[wordIndex], wordIndex);
        }
        wordIndex++;
      }
    }, wordInterval);

    return new Promise((resolve) => {
      setTimeout(() => {
        clearInterval(interval);
        if (this.isPlaying) {
          this.isPlaying = false;
          this._notifySpeakingEnd();
        }
        resolve();
      }, duration);
    });
  }

  /**
   * ElevenLabs TTS
   */
  async _speakWithElevenLabs(text, options) {
    // Check cache first
    const cacheKey = `${text}-${options.language || 'en'}`;
    if (this.audioCache.has(cacheKey)) {
      return this._playAudioBuffer(this.audioCache.get(cacheKey));
    }

    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${this.options.kellyVoiceId}`,
      {
        method: 'POST',
        headers: {
          Accept: 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': this.options.elevenLabsApiKey
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_monolingual_v1',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.5,
            use_speaker_boost: true
          }
        })
      }
    );

    if (!response.ok) {
      throw new Error(`ElevenLabs API error: ${response.status}`);
    }

    const audioBuffer = await response.arrayBuffer();

    // Cache the audio
    this.audioCache.set(cacheKey, audioBuffer);

    return this._playAudioBuffer(audioBuffer);
  }

  /**
   * Play audio buffer
   */
  _playAudioBuffer(audioBuffer) {
    return new Promise((resolve, reject) => {
      const blob = new Blob([audioBuffer], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);

      this.currentAudio = new Audio(url);
      this.currentAudio.volume = this.isMuted ? 0 : 1;

      this.currentAudio.onended = () => {
        URL.revokeObjectURL(url);
        this.isPlaying = false;
        this._notifySpeakingEnd();
        resolve();
      };

      this.currentAudio.onerror = (e) => {
        URL.revokeObjectURL(url);
        this.isPlaying = false;
        this._notifySpeakingEnd();
        reject(e);
      };

      this.currentAudio.play().catch(reject);
    });
  }

  // ═══════════════════════════════════════════════════════════════════
  // ⚠️ BROWSER TTS METHODS REMOVED - PROHIBITED BY CLAUDE.MD
  // Only ElevenLabs or silent mode allowed
  // ═══════════════════════════════════════════════════════════════════

  /**
   * Pause speech (ElevenLabs audio only)
   */
  pause() {
    this.isPaused = true;

    if (this.currentAudio) {
      this.currentAudio.pause();
    }

    this._notifySpeakingEnd();
  }

  /**
   * Resume speech (ElevenLabs audio only)
   */
  resume() {
    if (!this.isPaused) return;

    this.isPaused = false;

    if (this.currentAudio) {
      this.currentAudio.play();
      this._notifySpeakingStart();
    }
  }

  /**
   * Stop speech (ElevenLabs audio only)
   */
  stop() {
    this.isPlaying = false;
    this.isPaused = false;

    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }

    this._notifySpeakingEnd();
  }

  /**
   * Toggle mute (ElevenLabs audio only)
   */
  toggleMute() {
    this.isMuted = !this.isMuted;

    if (this.currentAudio) {
      this.currentAudio.volume = this.isMuted ? 0 : 1;
    }

    return this.isMuted;
  }

  /**
   * Set mute state
   */
  setMuted(muted) {
    this.isMuted = muted;

    if (this.currentAudio) {
      this.currentAudio.volume = muted ? 0 : 1;
    }
  }

  /**
   * Check if voice is available (ElevenLabs configured)
   */
  hasVoiceEnabled() {
    return this.hasVoice;
  }

  /**
   * Toggle play/pause
   */
  togglePlayPause() {
    if (this.isPaused) {
      this.resume();
      return true; // playing
    } else if (this.isPlaying) {
      this.pause();
      return false; // paused
    }
    return this.isPlaying;
  }

  /**
   * Get current state
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      isMuted: this.isMuted,
      currentText: this.currentText
    };
  }

  _notifySpeakingStart() {
    if (this.options.onSpeakingStart) {
      this.options.onSpeakingStart();
    }
    document.dispatchEvent(new CustomEvent('kelly-speaking-start'));
  }

  _notifySpeakingEnd() {
    if (this.options.onSpeakingEnd) {
      this.options.onSpeakingEnd();
    }
    document.dispatchEvent(new CustomEvent('kelly-speaking-end'));
  }
}

// TikTok-style interaction handler
class TikTokInteractions {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      onTap: options.onTap || null, // Single tap - play/pause
      onDoubleTap: options.onDoubleTap || null, // Double tap - like/heart
      onHold: options.onHold || null, // Long press - pause speech
      onHoldRelease: options.onHoldRelease || null,
      onSwipeUp: options.onSwipeUp || null, // Swipe up - next lesson
      onSwipeDown: options.onSwipeDown || null, // Swipe down - previous lesson
      onSwipeLeft: options.onSwipeLeft || null,
      onSwipeRight: options.onSwipeRight || null,
      holdDuration: 500,
      swipeThreshold: 50,
      ...options
    };

    this.touchStartX = 0;
    this.touchStartY = 0;
    this.touchStartTime = 0;
    this.holdTimer = null;
    this.isHolding = false;
    this.lastTapTime = 0;
    this.tapTimeout = null;

    this._setupEventListeners();

    console.log('[TikTokInteractions] Initialized');
  }

  _setupEventListeners() {
    // Touch events
    this.container.addEventListener('touchstart', this._onTouchStart.bind(this), { passive: true });
    this.container.addEventListener('touchmove', this._onTouchMove.bind(this), { passive: true });
    this.container.addEventListener('touchend', this._onTouchEnd.bind(this), { passive: true });
    this.container.addEventListener('touchcancel', this._onTouchCancel.bind(this));

    // Mouse events for desktop
    this.container.addEventListener('mousedown', this._onMouseDown.bind(this));
    this.container.addEventListener('mouseup', this._onMouseUp.bind(this));
    this.container.addEventListener('mouseleave', this._onMouseLeave.bind(this));

    // Keyboard for accessibility
    this.container.addEventListener('keydown', this._onKeyDown.bind(this));
  }

  _onTouchStart(e) {
    const touch = e.touches[0];
    this.touchStartX = touch.clientX;
    this.touchStartY = touch.clientY;
    this.touchStartTime = Date.now();

    // Start hold timer
    this.holdTimer = setTimeout(() => {
      this.isHolding = true;
      if (this.options.onHold) {
        this.options.onHold();
      }
    }, this.options.holdDuration);
  }

  _onTouchMove(e) {
    if (this.holdTimer) {
      const touch = e.touches[0];
      const deltaX = Math.abs(touch.clientX - this.touchStartX);
      const deltaY = Math.abs(touch.clientY - this.touchStartY);

      // Cancel hold if finger moved
      if (deltaX > 10 || deltaY > 10) {
        clearTimeout(this.holdTimer);
        this.holdTimer = null;
      }
    }
  }

  _onTouchEnd(e) {
    clearTimeout(this.holdTimer);
    this.holdTimer = null;

    const touch = e.changedTouches[0];
    const deltaX = touch.clientX - this.touchStartX;
    const deltaY = touch.clientY - this.touchStartY;
    const duration = Date.now() - this.touchStartTime;

    // Release hold
    if (this.isHolding) {
      this.isHolding = false;
      if (this.options.onHoldRelease) {
        this.options.onHoldRelease();
      }
      return;
    }

    // Check for swipe
    if (Math.abs(deltaY) > this.options.swipeThreshold && Math.abs(deltaY) > Math.abs(deltaX)) {
      if (deltaY < 0 && this.options.onSwipeUp) {
        this.options.onSwipeUp();
        return;
      } else if (deltaY > 0 && this.options.onSwipeDown) {
        this.options.onSwipeDown();
        return;
      }
    }

    if (Math.abs(deltaX) > this.options.swipeThreshold && Math.abs(deltaX) > Math.abs(deltaY)) {
      if (deltaX < 0 && this.options.onSwipeLeft) {
        this.options.onSwipeLeft();
        return;
      } else if (deltaX > 0 && this.options.onSwipeRight) {
        this.options.onSwipeRight();
        return;
      }
    }

    // Check for tap (quick touch without movement)
    if (duration < 300 && Math.abs(deltaX) < 10 && Math.abs(deltaY) < 10) {
      this._handleTap(touch.clientX, touch.clientY);
    }
  }

  _onTouchCancel() {
    clearTimeout(this.holdTimer);
    this.holdTimer = null;
    this.isHolding = false;
  }

  _onMouseDown(e) {
    this.touchStartX = e.clientX;
    this.touchStartY = e.clientY;
    this.touchStartTime = Date.now();

    this.holdTimer = setTimeout(() => {
      this.isHolding = true;
      if (this.options.onHold) {
        this.options.onHold();
      }
    }, this.options.holdDuration);
  }

  _onMouseUp(e) {
    clearTimeout(this.holdTimer);

    if (this.isHolding) {
      this.isHolding = false;
      if (this.options.onHoldRelease) {
        this.options.onHoldRelease();
      }
      return;
    }

    const duration = Date.now() - this.touchStartTime;
    if (duration < 300) {
      this._handleTap(e.clientX, e.clientY);
    }
  }

  _onMouseLeave() {
    clearTimeout(this.holdTimer);
    if (this.isHolding) {
      this.isHolding = false;
      if (this.options.onHoldRelease) {
        this.options.onHoldRelease();
      }
    }
  }

  _handleTap(x, y) {
    const now = Date.now();

    // Check for double tap
    if (now - this.lastTapTime < 300) {
      clearTimeout(this.tapTimeout);
      this.lastTapTime = 0;
      if (this.options.onDoubleTap) {
        this.options.onDoubleTap(x, y);
      }
    } else {
      this.lastTapTime = now;

      // Delay single tap to wait for potential double tap
      this.tapTimeout = setTimeout(() => {
        if (this.options.onTap) {
          this.options.onTap(x, y);
        }
        this.lastTapTime = 0;
      }, 300);
    }
  }

  _onKeyDown(e) {
    switch (e.key) {
      case ' ':
      case 'Enter':
        e.preventDefault();
        if (this.options.onTap) this.options.onTap();
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (this.options.onSwipeUp) this.options.onSwipeUp();
        break;
      case 'ArrowDown':
        e.preventDefault();
        if (this.options.onSwipeDown) this.options.onSwipeDown();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (this.options.onSwipeLeft) this.options.onSwipeLeft();
        break;
      case 'ArrowRight':
        e.preventDefault();
        if (this.options.onSwipeRight) this.options.onSwipeRight();
        break;
    }
  }

  destroy() {
    clearTimeout(this.holdTimer);
    clearTimeout(this.tapTimeout);
  }
}

// Export
if (typeof window !== 'undefined') {
  window.KellyAudio = KellyAudio;
  window.TikTokInteractions = TikTokInteractions;
}
