/**
 * Kelly Video Player
 * 
 * Handles playback of ElevenLabs Omnihuman 1.5 generated lip-sync videos.
 * Provides seamless integration with the lesson player and fallback to
 * image+audio when video is not available.
 * 
 * @module KellyVideoPlayer
 */

export class KellyVideoPlayer {
  /**
   * Create a new Kelly Video Player instance
   * @param {string|HTMLElement} containerSelector - Container element or selector
   * @param {Object} options - Configuration options
   */
  constructor(containerSelector, options = {}) {
    this.container = typeof containerSelector === 'string' 
      ? document.querySelector(containerSelector)
      : containerSelector;

    if (!this.container) {
      throw new Error(`KellyVideoPlayer: Container not found: ${containerSelector}`);
    }

    this.options = {
      autoplay: true,
      loop: false,
      muted: false,
      preload: 'auto',
      showControls: false,
      aspectRatio: '9:16', // Kelly's default portrait aspect ratio
      maxHeight: '70vh',
      borderRadius: '16px',
      onError: null, // Called when video fails to load
      onFallback: null, // Called when falling back to image+audio
      ...options
    };

    // State
    this.videoElement = null;
    this.posterElement = null;
    this.loadingElement = null;
    this.isPlaying = false;
    this.isPaused = false;
    this.isLoading = false;
    this.currentUrl = null;
    this.hasError = false;

    // Callbacks
    this.callbacks = {
      onStart: options.onStart || null,
      onEnd: options.onEnd || null,
      onPause: options.onPause || null,
      onResume: options.onResume || null,
      onTimeUpdate: options.onTimeUpdate || null,
      onProgress: options.onProgress || null,
      onError: options.onError || null,
      onReady: options.onReady || null,
    };

    // Initialize
    this.init();
  }

  /**
   * Initialize the video player DOM elements
   */
  init() {
    // Create wrapper
    this.wrapper = document.createElement('div');
    this.wrapper.className = 'kelly-video-wrapper';
    this.wrapper.style.cssText = `
      position: relative;
      width: 100%;
      max-height: ${this.options.maxHeight};
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
      border-radius: ${this.options.borderRadius};
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    `;

    // Create video element
    this.videoElement = document.createElement('video');
    this.videoElement.className = 'kelly-video-player';
    this.videoElement.playsInline = true;
    this.videoElement.preload = this.options.preload;
    this.videoElement.loop = this.options.loop;
    this.videoElement.muted = this.options.muted;
    this.videoElement.controls = this.options.showControls;
    
    this.videoElement.style.cssText = `
      width: 100%;
      height: auto;
      max-height: ${this.options.maxHeight};
      object-fit: contain;
      border-radius: ${this.options.borderRadius};
      display: none;
    `;

    // Create poster/fallback image element
    this.posterElement = document.createElement('img');
    this.posterElement.className = 'kelly-video-poster';
    this.posterElement.style.cssText = `
      width: 100%;
      height: auto;
      max-height: ${this.options.maxHeight};
      object-fit: contain;
      border-radius: ${this.options.borderRadius};
      display: none;
    `;

    // Create loading indicator
    this.loadingElement = document.createElement('div');
    this.loadingElement.className = 'kelly-video-loading';
    this.loadingElement.innerHTML = `
      <div class="kelly-loading-spinner">
        <svg viewBox="0 0 50 50" class="spinner">
          <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" stroke-width="4" stroke-linecap="round">
            <animate attributeName="stroke-dasharray" values="1, 150; 90, 150; 90, 150" dur="1.5s" repeatCount="indefinite"/>
            <animate attributeName="stroke-dashoffset" values="0; -35; -124" dur="1.5s" repeatCount="indefinite"/>
          </circle>
        </svg>
      </div>
      <div class="kelly-loading-text">Loading Kelly...</div>
    `;
    this.loadingElement.style.cssText = `
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      color: white;
      display: none;
    `;

    // Add spinner styles
    const style = document.createElement('style');
    style.textContent = `
      .kelly-loading-spinner .spinner {
        width: 50px;
        height: 50px;
        animation: rotate 2s linear infinite;
      }
      @keyframes rotate {
        100% { transform: rotate(360deg); }
      }
      .kelly-loading-text {
        margin-top: 12px;
        font-size: 14px;
        opacity: 0.8;
      }
    `;
    this.wrapper.appendChild(style);

    // Add elements to wrapper
    this.wrapper.appendChild(this.videoElement);
    this.wrapper.appendChild(this.posterElement);
    this.wrapper.appendChild(this.loadingElement);

    // Add wrapper to container
    this.container.appendChild(this.wrapper);

    // Setup event listeners
    this.setupEventListeners();

    console.log('[KellyVideoPlayer] Initialized');
  }

  /**
   * Setup video element event listeners
   */
  setupEventListeners() {
    // Playback events
    this.videoElement.addEventListener('play', () => {
      this.isPlaying = true;
      this.isPaused = false;
      this.posterElement.style.display = 'none';
      this.videoElement.style.display = 'block';
      this.callbacks.onStart?.();
      this.dispatchEvent('kelly-video-start');
    });

    this.videoElement.addEventListener('pause', () => {
      if (!this.videoElement.ended) {
        this.isPaused = true;
        this.callbacks.onPause?.();
        this.dispatchEvent('kelly-video-pause');
      }
    });

    this.videoElement.addEventListener('ended', () => {
      this.isPlaying = false;
      this.isPaused = false;
      this.callbacks.onEnd?.();
      this.dispatchEvent('kelly-video-end');
    });

    // Time updates
    this.videoElement.addEventListener('timeupdate', () => {
      const currentTime = this.videoElement.currentTime;
      const duration = this.videoElement.duration;
      const progress = duration > 0 ? currentTime / duration : 0;
      
      this.callbacks.onTimeUpdate?.(currentTime, duration);
      this.callbacks.onProgress?.(progress);
      this.dispatchEvent('kelly-video-progress', { currentTime, duration, progress });
    });

    // Loading events
    this.videoElement.addEventListener('loadstart', () => {
      this.isLoading = true;
      this.showLoading();
    });

    this.videoElement.addEventListener('canplay', () => {
      this.isLoading = false;
      this.hideLoading();
      this.callbacks.onReady?.();
      this.dispatchEvent('kelly-video-ready');
    });

    this.videoElement.addEventListener('waiting', () => {
      this.showLoading();
    });

    this.videoElement.addEventListener('playing', () => {
      this.hideLoading();
    });

    // Error handling
    this.videoElement.addEventListener('error', (e) => {
      this.hasError = true;
      this.isLoading = false;
      this.hideLoading();
      
      const error = this.videoElement.error;
      const errorMessage = error ? `${error.code}: ${error.message}` : 'Unknown error';
      
      console.error('[KellyVideoPlayer] Video error:', errorMessage);
      
      this.callbacks.onError?.(errorMessage);
      this.dispatchEvent('kelly-video-error', { error: errorMessage });
    });
  }

  /**
   * Dispatch custom event
   * @param {string} eventName - Event name
   * @param {Object} detail - Event detail data
   */
  dispatchEvent(eventName, detail = {}) {
    const event = new CustomEvent(eventName, {
      detail: { ...detail, player: this },
      bubbles: true
    });
    this.container.dispatchEvent(event);
    document.dispatchEvent(event);
  }

  /**
   * Load and play a video
   * @param {string} videoUrl - URL of the video to play
   * @param {Object} options - Playback options
   * @returns {Promise<void>}
   */
  async play(videoUrl, options = {}) {
    if (!videoUrl) {
      console.warn('[KellyVideoPlayer] No video URL provided');
      return;
    }

    this.currentUrl = videoUrl;
    this.hasError = false;
    this.showLoading();

    // Set poster if provided
    if (options.poster) {
      this.posterElement.src = options.poster;
      this.posterElement.style.display = 'block';
      this.videoElement.poster = options.poster;
    }

    // Load video
    this.videoElement.src = videoUrl;
    this.videoElement.load();

    try {
      // Wait for video to be ready
      await new Promise((resolve, reject) => {
        const onCanPlay = () => {
          this.videoElement.removeEventListener('canplay', onCanPlay);
          this.videoElement.removeEventListener('error', onError);
          resolve();
        };
        const onError = () => {
          this.videoElement.removeEventListener('canplay', onCanPlay);
          this.videoElement.removeEventListener('error', onError);
          reject(new Error('Video failed to load'));
        };
        this.videoElement.addEventListener('canplay', onCanPlay);
        this.videoElement.addEventListener('error', onError);
      });

      // Start playback if autoplay is enabled
      if (this.options.autoplay !== false && options.autoplay !== false) {
        await this.videoElement.play();
      }

      this.hideLoading();
      this.posterElement.style.display = 'none';
      this.videoElement.style.display = 'block';

    } catch (error) {
      console.error('[KellyVideoPlayer] Playback failed:', error);
      this.hideLoading();
      
      // Trigger fallback callback
      if (this.options.onFallback) {
        this.options.onFallback(videoUrl, error);
      }
      
      throw error;
    }
  }

  /**
   * Pause video playback
   */
  pause() {
    if (this.isPlaying && !this.isPaused) {
      this.videoElement.pause();
    }
  }

  /**
   * Resume video playback
   */
  resume() {
    if (this.isPaused) {
      this.videoElement.play();
      this.isPaused = false;
    }
  }

  /**
   * Stop video playback and reset
   */
  stop() {
    this.videoElement.pause();
    this.videoElement.currentTime = 0;
    this.isPlaying = false;
    this.isPaused = false;
    this.dispatchEvent('kelly-video-stop');
  }

  /**
   * Toggle play/pause
   * @returns {boolean} - New playing state
   */
  togglePlayPause() {
    if (this.isPlaying && !this.isPaused) {
      this.pause();
      return false;
    } else {
      this.resume();
      return true;
    }
  }

  /**
   * Set volume level
   * @param {number} level - Volume level (0-1)
   */
  setVolume(level) {
    this.videoElement.volume = Math.max(0, Math.min(1, level));
  }

  /**
   * Get current volume level
   * @returns {number}
   */
  getVolume() {
    return this.videoElement.volume;
  }

  /**
   * Mute audio
   */
  mute() {
    this.videoElement.muted = true;
  }

  /**
   * Unmute audio
   */
  unmute() {
    this.videoElement.muted = false;
  }

  /**
   * Toggle mute state
   * @returns {boolean} - New muted state
   */
  toggleMute() {
    this.videoElement.muted = !this.videoElement.muted;
    return this.videoElement.muted;
  }

  /**
   * Seek to a specific time
   * @param {number} time - Time in seconds
   */
  seek(time) {
    if (this.videoElement.duration) {
      this.videoElement.currentTime = Math.max(0, Math.min(time, this.videoElement.duration));
    }
  }

  /**
   * Seek by percentage
   * @param {number} percent - Percentage (0-1)
   */
  seekPercent(percent) {
    if (this.videoElement.duration) {
      this.seek(this.videoElement.duration * percent);
    }
  }

  /**
   * Get current playback state
   * @returns {Object}
   */
  getState() {
    return {
      isPlaying: this.isPlaying,
      isPaused: this.isPaused,
      isLoading: this.isLoading,
      hasError: this.hasError,
      currentTime: this.videoElement.currentTime,
      duration: this.videoElement.duration,
      progress: this.videoElement.duration > 0 
        ? this.videoElement.currentTime / this.videoElement.duration 
        : 0,
      volume: this.videoElement.volume,
      muted: this.videoElement.muted,
      currentUrl: this.currentUrl
    };
  }

  /**
   * Show loading indicator
   */
  showLoading() {
    this.loadingElement.style.display = 'block';
  }

  /**
   * Hide loading indicator
   */
  hideLoading() {
    this.loadingElement.style.display = 'none';
  }

  /**
   * Show poster/fallback image
   * @param {string} imageUrl - Image URL
   */
  showPoster(imageUrl) {
    this.posterElement.src = imageUrl;
    this.posterElement.style.display = 'block';
    this.videoElement.style.display = 'none';
  }

  /**
   * Register event callback
   * @param {string} event - Event name (start, end, pause, resume, timeUpdate, progress, error, ready)
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    const callbackName = `on${event.charAt(0).toUpperCase() + event.slice(1)}`;
    if (callbackName in this.callbacks) {
      this.callbacks[callbackName] = callback;
    }
  }

  /**
   * Remove event callback
   * @param {string} event - Event name
   */
  off(event) {
    const callbackName = `on${event.charAt(0).toUpperCase() + event.slice(1)}`;
    if (callbackName in this.callbacks) {
      this.callbacks[callbackName] = null;
    }
  }

  /**
   * Destroy the player and clean up
   */
  destroy() {
    this.stop();
    this.videoElement.src = '';
    this.wrapper.remove();
    console.log('[KellyVideoPlayer] Destroyed');
  }
}

/**
 * Factory function to create a KellyVideoPlayer
 * @param {string|HTMLElement} container - Container element or selector
 * @param {Object} options - Configuration options
 * @returns {KellyVideoPlayer}
 */
export function createKellyVideoPlayer(container, options = {}) {
  return new KellyVideoPlayer(container, options);
}

// Export for browser global use
if (typeof window !== 'undefined') {
  window.KellyVideoPlayer = KellyVideoPlayer;
  window.createKellyVideoPlayer = createKellyVideoPlayer;
}

export default KellyVideoPlayer;

