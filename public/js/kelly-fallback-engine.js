/**
 * Kelly Fallback Engine
 * 
 * BULLETPROOF MEDIA DELIVERY - Kelly ALWAYS appears, content ALWAYS plays
 * 
 * Cascade Priority:
 *   1. Video (kelly_motion_library) - Full talking head animation
 *   2. Static Image + Audio - Kelly photo with ElevenLabs audio
 *   3. Static Image + Text - Kelly photo with captions only
 *   4. Generic Kelly + Text - Default Kelly image with content
 * 
 * For millions of users daily - no single point of failure
 */

(function() {
  'use strict';

  // CDN Base URLs
  const SUPABASE_STORAGE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public';
  const KELLY_VIDEOS_BUCKET = `${SUPABASE_STORAGE}/kelly-videos`;
  const KELLY_TEMPLATES_BUCKET = `${SUPABASE_STORAGE}/kelly-templates`;

  // Core 60 Kelly Heads - Maps persona/age to image URL
  const KELLY_HEAD_URLS = {};
  const AGES = ['kid', 'teen', 'adult', 'elder', 'super_elder'];
  const PERSONAS = [
    'scientist', 'explorer', 'rebel', 'architect', 'diplomat', 'empath',
    'macgyver', 'mystic', 'provider', 'storyteller', 'strategist', 'survivor'
  ];

  // Pre-compute all Kelly head URLs
  AGES.forEach(age => {
    KELLY_HEAD_URLS[age] = {};
    PERSONAS.forEach(persona => {
      const path = age === 'adult' 
        ? `heygen/archetypes-head-only/kelly_${persona}_head.png`
        : `heygen/archetypes-head-only/age/${age}/kelly_${persona}_head.png`;
      KELLY_HEAD_URLS[age][persona] = `${KELLY_TEMPLATES_BUCKET}/${path}`;
    });
  });

  // Default fallback image when nothing else works
  const DEFAULT_KELLY_IMAGE = `${KELLY_TEMPLATES_BUCKET}/heygen/archetypes-head-only/kelly_storyteller_head.png`;

  // Playback state (persisted in localStorage)
  const PLAYBACK_STATE_KEY = 'kelly_playback_state';
  
  function getPlaybackState() {
    try {
      const saved = localStorage.getItem(PLAYBACK_STATE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) { /* ignore */ }
    return {
      muted: false,
      volume: 1.0,
      autoplay: true,
      preferVideo: true
    };
  }

  function savePlaybackState(state) {
    try {
      localStorage.setItem(PLAYBACK_STATE_KEY, JSON.stringify(state));
    } catch (e) { /* ignore */ }
  }

  /**
   * Get Kelly head image URL for a persona/age combination
   */
  function getKellyHeadUrl(persona, ageBucket) {
    const normalizedAge = ageBucket || 'adult';
    const normalizedPersona = persona || 'storyteller';
    
    if (KELLY_HEAD_URLS[normalizedAge] && KELLY_HEAD_URLS[normalizedAge][normalizedPersona]) {
      return KELLY_HEAD_URLS[normalizedAge][normalizedPersona];
    }
    
    // Fallback: try adult version of persona
    if (KELLY_HEAD_URLS['adult'] && KELLY_HEAD_URLS['adult'][normalizedPersona]) {
      return KELLY_HEAD_URLS['adult'][normalizedPersona];
    }
    
    return DEFAULT_KELLY_IMAGE;
  }

  /**
   * Get video URL from motion library format
   */
  function getVideoUrl(persona, ageBucket, phase) {
    return `${KELLY_VIDEOS_BUCKET}/motion/${persona}/${ageBucket}/${phase}.mp4`;
  }

  /**
   * Check if a URL is accessible (HEAD request with timeout)
   */
  async function checkUrlAccessible(url, timeoutMs = 3000) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      
      const response = await fetch(url, {
        method: 'HEAD',
        signal: controller.signal
      });
      
      clearTimeout(timeout);
      return response.ok;
    } catch (e) {
      return false;
    }
  }

  /**
   * Get best available media for a lesson phase
   * Returns: { type: 'video'|'image'|'fallback', url: string, audioUrl?: string }
   */
  async function getBestMedia(options) {
    const { persona, ageBucket, phase, audioUrl } = options;
    
    // Try video first
    const videoUrl = getVideoUrl(persona, ageBucket, phase);
    if (await checkUrlAccessible(videoUrl)) {
      return { type: 'video', url: videoUrl };
    }
    
    // Fall back to image + audio
    const imageUrl = getKellyHeadUrl(persona, ageBucket);
    if (await checkUrlAccessible(imageUrl)) {
      return { 
        type: 'image', 
        url: imageUrl, 
        audioUrl: audioUrl || null,
        fallbackReason: 'video_unavailable'
      };
    }
    
    // Final fallback
    return {
      type: 'fallback',
      url: DEFAULT_KELLY_IMAGE,
      audioUrl: audioUrl || null,
      fallbackReason: 'all_media_unavailable'
    };
  }

  /**
   * Fallback Player - Handles image + audio playback
   */
  class FallbackPlayer {
    constructor(containerElement) {
      this.container = containerElement;
      this.audioElement = null;
      this.imageElement = null;
      this.state = getPlaybackState();
      this.isPlaying = false;
    }

    setup(imageUrl, audioUrl) {
      // Clear container
      this.container.innerHTML = '';
      
      // Create image element
      this.imageElement = document.createElement('img');
      this.imageElement.src = imageUrl;
      this.imageElement.alt = 'Kelly';
      this.imageElement.style.cssText = `
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 12px;
      `;
      
      // Kelly never floats or hovers - no animation on static images
      
      this.container.appendChild(this.imageElement);
      
      // Create audio element if we have audio
      if (audioUrl) {
        this.audioElement = document.createElement('audio');
        this.audioElement.src = audioUrl;
        this.audioElement.preload = 'auto';
        this.audioElement.muted = this.state.muted;
        this.audioElement.volume = this.state.volume;
        
        this.audioElement.addEventListener('ended', () => {
          this.isPlaying = false;
          this.onEnded?.();
        });
        
        this.audioElement.addEventListener('error', (e) => {
          console.warn('[FallbackPlayer] Audio error:', e);
          // Continue without audio
          this.onEnded?.();
        });
      }
      
      return this;
    }

    async play() {
      if (this.audioElement && !this.state.muted) {
        try {
          await this.audioElement.play();
          this.isPlaying = true;
        } catch (e) {
          console.warn('[FallbackPlayer] Audio play failed:', e);
          // Autoplay might be blocked - continue silently
        }
      }
      return this;
    }

    pause() {
      if (this.audioElement) {
        this.audioElement.pause();
        this.isPlaying = false;
      }
      return this;
    }

    setMuted(muted) {
      this.state.muted = muted;
      if (this.audioElement) {
        this.audioElement.muted = muted;
      }
      savePlaybackState(this.state);
      return this;
    }

    setVolume(volume) {
      this.state.volume = Math.max(0, Math.min(1, volume));
      if (this.audioElement) {
        this.audioElement.volume = this.state.volume;
      }
      savePlaybackState(this.state);
      return this;
    }

    destroy() {
      if (this.audioElement) {
        this.audioElement.pause();
        this.audioElement.src = '';
        this.audioElement = null;
      }
      if (this.container) {
        this.container.innerHTML = '';
      }
    }
  }

  /**
   * Global Playback Controller
   * Manages all media playback with unified controls
   */
  class PlaybackController {
    constructor() {
      this.currentPlayer = null;
      this.videoElement = null;
      this.fallbackPlayer = null;
      this.state = getPlaybackState();
      this.listeners = [];
    }

    /**
     * Initialize with video and fallback container elements
     */
    init(videoElement, fallbackContainer) {
      this.videoElement = videoElement;
      this.fallbackContainer = fallbackContainer;
      
      // Apply saved state to video element
      if (this.videoElement) {
        this.videoElement.muted = this.state.muted;
        this.videoElement.volume = this.state.volume;
      }
      
      return this;
    }

    /**
     * Load media for a lesson phase
     */
    async loadPhase(options) {
      const media = await getBestMedia(options);
      
      if (media.type === 'video') {
        // Use video player
        this.showVideo(media.url);
        return { type: 'video', url: media.url };
      } else {
        // Use fallback player (image + audio)
        this.showFallback(media.url, media.audioUrl);
        return media;
      }
    }

    showVideo(url) {
      // Hide fallback, show video
      if (this.fallbackContainer) {
        this.fallbackContainer.style.display = 'none';
      }
      if (this.videoElement) {
        this.videoElement.style.display = 'block';
        this.videoElement.src = url;
        this.videoElement.muted = this.state.muted;
        this.videoElement.volume = this.state.volume;
        this.currentPlayer = 'video';
      }
    }

    showFallback(imageUrl, audioUrl) {
      // Hide video, show fallback
      if (this.videoElement) {
        this.videoElement.style.display = 'none';
        this.videoElement.pause();
        this.videoElement.src = '';
      }
      if (this.fallbackContainer) {
        this.fallbackContainer.style.display = 'block';
        
        if (!this.fallbackPlayer) {
          this.fallbackPlayer = new FallbackPlayer(this.fallbackContainer);
        }
        
        this.fallbackPlayer.setup(imageUrl, audioUrl);
        this.currentPlayer = 'fallback';
      }
    }

    async play() {
      if (this.currentPlayer === 'video' && this.videoElement) {
        try {
          await this.videoElement.play();
        } catch (e) {
          console.warn('[PlaybackController] Video play failed:', e);
        }
      } else if (this.currentPlayer === 'fallback' && this.fallbackPlayer) {
        await this.fallbackPlayer.play();
      }
      this.emit('play');
    }

    pause() {
      if (this.currentPlayer === 'video' && this.videoElement) {
        this.videoElement.pause();
      } else if (this.currentPlayer === 'fallback' && this.fallbackPlayer) {
        this.fallbackPlayer.pause();
      }
      this.emit('pause');
    }

    toggleMute() {
      this.state.muted = !this.state.muted;
      
      if (this.videoElement) {
        this.videoElement.muted = this.state.muted;
      }
      if (this.fallbackPlayer) {
        this.fallbackPlayer.setMuted(this.state.muted);
      }
      
      savePlaybackState(this.state);
      this.emit('muteChange', this.state.muted);
      return this.state.muted;
    }

    setMuted(muted) {
      this.state.muted = muted;
      
      if (this.videoElement) {
        this.videoElement.muted = muted;
      }
      if (this.fallbackPlayer) {
        this.fallbackPlayer.setMuted(muted);
      }
      
      savePlaybackState(this.state);
      this.emit('muteChange', muted);
    }

    setVolume(volume) {
      this.state.volume = Math.max(0, Math.min(1, volume));
      
      if (this.videoElement) {
        this.videoElement.volume = this.state.volume;
      }
      if (this.fallbackPlayer) {
        this.fallbackPlayer.setVolume(this.state.volume);
      }
      
      savePlaybackState(this.state);
      this.emit('volumeChange', this.state.volume);
    }

    getState() {
      return { ...this.state };
    }

    on(event, callback) {
      this.listeners.push({ event, callback });
      return this;
    }

    emit(event, data) {
      this.listeners
        .filter(l => l.event === event)
        .forEach(l => l.callback(data));
    }
  }

  // Add CSS for static image fallback (no floating/breathing - Kelly is grounded)
  const style = document.createElement('style');
  style.textContent = `
    
    .kelly-fallback-container {
      position: relative;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #0a0a0b;
      border-radius: 12px;
      overflow: hidden;
    }
    
    .kelly-fallback-indicator {
      position: absolute;
      bottom: 12px;
      right: 12px;
      background: rgba(0, 0, 0, 0.7);
      color: white;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      opacity: 0.7;
    }
  `;
  document.head.appendChild(style);

  // Export to global namespace
  window.KellyFallbackEngine = {
    getKellyHeadUrl,
    getVideoUrl,
    getBestMedia,
    checkUrlAccessible,
    FallbackPlayer,
    PlaybackController,
    KELLY_HEAD_URLS,
    DEFAULT_KELLY_IMAGE,
    AGES,
    PERSONAS
  };

  if (typeof location !== 'undefined' && location.search.includes('debug')) {
    console.log('🛡️ Kelly Fallback Engine initialized - bulletproof media delivery ready');
  }
})();
