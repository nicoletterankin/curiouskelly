/**
 * Kelly Video Player
 * 
 * Plays pre-generated Kelly lip-sync videos in the lesson player.
 * Falls back to static images + audio if video not available.
 * 
 * Usage:
 *   KellyVideoPlayer.play({
 *     videoUrl: '/kelly/videos/lesson-1-welcome.mp4',
 *     fallbackImage: '/kelly/poses/kelly_welcome.png',
 *     fallbackAudio: '/kelly/audio/lesson-1-welcome.mp3',
 *   });
 */

const KellyVideoPlayer = {
  // State
  isPlaying: false,
  currentVideo: null,
  videoContainer: null,
  fallbackMode: false,
  
  // Configuration
  config: {
    containerId: 'kelly-video-container',
    videoBasePath: '/kelly/videos/',
    imageBasePath: '/kelly/poses/',
    audioBasePath: '/kelly/audio/',
    autoplay: true,
    muted: false,
    loop: false,
  },
  
  // Callbacks
  onStart: null,
  onEnd: null,
  onError: null,
  
  // ===========================================================================
  // INITIALIZATION
  // ===========================================================================
  
  /**
   * Initialize the video player
   * @param {Object} options - Configuration options
   */
  init(options = {}) {
    Object.assign(this.config, options);
    
    // Find or create container
    this.videoContainer = document.getElementById(this.config.containerId);
    if (!this.videoContainer) {
      this.videoContainer = document.createElement('div');
      this.videoContainer.id = this.config.containerId;
      this.videoContainer.className = 'kelly-video-container';
    }
    
    // Add styles
    this.addStyles();
    
    console.log('[KellyVideoPlayer] Initialized');
    return this;
  },
  
  /**
   * Add required CSS styles
   */
  addStyles() {
    if (document.getElementById('kelly-video-styles')) return;
    
    const styles = document.createElement('style');
    styles.id = 'kelly-video-styles';
    styles.textContent = `
      .kelly-video-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        aspect-ratio: 16/9;
        background: #1a1a2e;
        border-radius: 16px;
        overflow: hidden;
      }
      
      .kelly-video-container video,
      .kelly-video-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
      
      .kelly-video-container .loading-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(26, 26, 46, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10;
      }
      
      .kelly-video-container .loading-spinner {
        width: 48px;
        height: 48px;
        border: 4px solid rgba(255, 255, 255, 0.2);
        border-top-color: #00d4aa;
        border-radius: 50%;
        animation: kelly-spin 1s linear infinite;
      }
      
      @keyframes kelly-spin {
        to { transform: rotate(360deg); }
      }
      
      .kelly-video-container.playing .loading-overlay {
        display: none;
      }
      
      .kelly-video-controls {
        position: absolute;
        bottom: 12px;
        right: 12px;
        display: flex;
        gap: 8px;
      }
      
      .kelly-video-controls button {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.6);
        border: none;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
      }
      
      .kelly-video-controls button:hover {
        background: rgba(0, 212, 170, 0.8);
      }
    `;
    document.head.appendChild(styles);
  },
  
  // ===========================================================================
  // PLAYBACK
  // ===========================================================================
  
  /**
   * Play a Kelly video
   * @param {Object} options - Playback options
   */
  async play(options) {
    const {
      videoUrl,
      fallbackImage,
      fallbackAudio,
      onStart,
      onEnd,
      onError,
    } = options;
    
    this.onStart = onStart;
    this.onEnd = onEnd;
    this.onError = onError;
    
    // Stop any current playback
    this.stop();
    
    // Show loading state
    this.showLoading();
    
    // Try video first
    if (videoUrl) {
      const videoAvailable = await this.checkVideoAvailable(videoUrl);
      
      if (videoAvailable) {
        this.playVideo(videoUrl);
        return;
      }
    }
    
    // Fall back to image + audio
    if (fallbackImage) {
      this.fallbackMode = true;
      this.playFallback(fallbackImage, fallbackAudio);
    } else {
      this.handleError('No video or fallback image provided');
    }
  },
  
  /**
   * Check if a video URL is available
   * @param {string} url - Video URL
   * @returns {Promise<boolean>}
   */
  async checkVideoAvailable(url) {
    try {
      const response = await fetch(url, { method: 'HEAD' });
      return response.ok;
    } catch {
      return false;
    }
  },
  
  /**
   * Play a video file
   * @param {string} url - Video URL
   */
  playVideo(url) {
    // Create video element
    const video = document.createElement('video');
    video.src = url;
    video.autoplay = this.config.autoplay;
    video.muted = this.config.muted;
    video.loop = this.config.loop;
    video.playsInline = true;
    
    // Event handlers
    video.oncanplay = () => {
      this.hideLoading();
      this.videoContainer.classList.add('playing');
      this.isPlaying = true;
      this.onStart?.();
    };
    
    video.onended = () => {
      this.isPlaying = false;
      this.videoContainer.classList.remove('playing');
      this.onEnd?.();
    };
    
    video.onerror = () => {
      this.handleError('Video playback error');
    };
    
    // Clear container and add video
    this.videoContainer.innerHTML = '';
    this.videoContainer.appendChild(video);
    this.currentVideo = video;
    
    // Add controls
    this.addControls(video);
    
    // Add loading overlay
    this.addLoadingOverlay();
  },
  
  /**
   * Play fallback (static image + audio)
   * @param {string} imageUrl - Image URL
   * @param {string} audioUrl - Audio URL (optional)
   */
  playFallback(imageUrl, audioUrl) {
    // Create image
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = 'Kelly';
    
    img.onload = () => {
      this.hideLoading();
      this.videoContainer.classList.add('playing');
      this.onStart?.();
      
      // Play audio if available
      if (audioUrl) {
        const audio = new Audio(audioUrl);
        audio.play();
        
        audio.onended = () => {
          this.isPlaying = false;
          this.onEnd?.();
        };
        
        this.currentVideo = { pause: () => audio.pause() };
      } else {
        this.isPlaying = true;
      }
    };
    
    img.onerror = () => {
      this.handleError('Image load error');
    };
    
    // Clear container and add image
    this.videoContainer.innerHTML = '';
    this.videoContainer.appendChild(img);
    
    // Add loading overlay
    this.addLoadingOverlay();
  },
  
  // ===========================================================================
  // CONTROLS
  // ===========================================================================
  
  /**
   * Add video controls
   * @param {HTMLVideoElement} video
   */
  addControls(video) {
    const controls = document.createElement('div');
    controls.className = 'kelly-video-controls';
    
    // Play/Pause button
    const playBtn = document.createElement('button');
    playBtn.innerHTML = '⏸';
    playBtn.onclick = () => {
      if (video.paused) {
        video.play();
        playBtn.innerHTML = '⏸';
      } else {
        video.pause();
        playBtn.innerHTML = '▶';
      }
    };
    
    // Mute button
    const muteBtn = document.createElement('button');
    muteBtn.innerHTML = video.muted ? '🔇' : '🔊';
    muteBtn.onclick = () => {
      video.muted = !video.muted;
      muteBtn.innerHTML = video.muted ? '🔇' : '🔊';
    };
    
    controls.appendChild(playBtn);
    controls.appendChild(muteBtn);
    this.videoContainer.appendChild(controls);
  },
  
  /**
   * Add loading overlay
   */
  addLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = '<div class="loading-spinner"></div>';
    this.videoContainer.appendChild(overlay);
  },
  
  /**
   * Show loading state
   */
  showLoading() {
    this.videoContainer.classList.remove('playing');
  },
  
  /**
   * Hide loading state
   */
  hideLoading() {
    this.videoContainer.classList.add('playing');
  },
  
  /**
   * Handle error
   * @param {string} message
   */
  handleError(message) {
    console.error('[KellyVideoPlayer]', message);
    this.hideLoading();
    this.onError?.(message);
  },
  
  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================
  
  /**
   * Stop playback
   */
  stop() {
    if (this.currentVideo) {
      this.currentVideo.pause?.();
      this.currentVideo = null;
    }
    this.isPlaying = false;
    this.fallbackMode = false;
    this.videoContainer?.classList.remove('playing');
  },
  
  /**
   * Pause playback
   */
  pause() {
    this.currentVideo?.pause?.();
  },
  
  /**
   * Resume playback
   */
  resume() {
    this.currentVideo?.play?.();
  },
  
  // ===========================================================================
  // HELPERS
  // ===========================================================================
  
  /**
   * Get video URL for a lesson phase
   * @param {number} dayNumber - Lesson day number
   * @param {string} phase - Phase name
   * @returns {string}
   */
  getLessonVideoUrl(dayNumber, phase = 'welcome') {
    const dayStr = dayNumber.toString().padStart(3, '0');
    return `${this.config.videoBasePath}lesson-${dayStr}-${phase}.mp4`;
  },
  
  /**
   * Get fallback image URL for a lesson phase
   * @param {number} dayNumber - Lesson day number
   * @param {string} phase - Phase name
   * @returns {string}
   */
  getLessonImageUrl(dayNumber, phase = 'hero') {
    const dayStr = dayNumber.toString().padStart(3, '0');
    return `/kelly/lessons/${dayStr}/lesson-${dayNumber}-${phase}.png`;
  },
  
  /**
   * Play lesson phase with automatic fallback
   * @param {number} dayNumber - Lesson day number
   * @param {string} phase - Phase name
   * @param {Object} options - Additional options
   */
  playLessonPhase(dayNumber, phase, options = {}) {
    const videoUrl = this.getLessonVideoUrl(dayNumber, phase);
    const fallbackImage = this.getLessonImageUrl(dayNumber, options.imageType || 'hero');
    
    this.play({
      videoUrl,
      fallbackImage,
      fallbackAudio: options.audioUrl,
      ...options,
    });
  },
};

// =============================================================================
// EXPORT
// =============================================================================

if (typeof window !== 'undefined') {
  window.KellyVideoPlayer = KellyVideoPlayer;
}

// Auto-init on DOM ready
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyVideoPlayer.init());
  } else {
    KellyVideoPlayer.init();
  }
}

