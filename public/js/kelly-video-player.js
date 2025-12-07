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
      this.videoContainer.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;z-index:10;display:none;';
      
      // Try to append to kelly-frame if it exists
      const kellyFrame = document.getElementById('kelly-frame');
      if (kellyFrame) {
        kellyFrame.appendChild(this.videoContainer);
      }
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
  
  // ===========================================================================
  // GOLDEN LESSON VIDEOS (Day 1 - Premium Quality)
  // ===========================================================================
  
  /**
   * Supabase storage URL for Kelly templates
   */
  supabaseStorageUrl: 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates',
  
  /**
   * Map phase names to storage format
   */
  phaseMap: {
    'welcome': 'Hook',
    'hook': 'Hook',
    'Hook': 'Hook',
    'fact1': 'Fact1',
    'Fact1': 'Fact1',
    'q1': 'Fact1',
    'fact2': 'Fact2',
    'Fact2': 'Fact2',
    'q2': 'Fact2',
    'fact3': 'Fact3',
    'Fact3': 'Fact3',
    'q3': 'Fact3',
    'wisdom': 'Wisdom',
    'Wisdom': 'Wisdom',
  },
  
  /**
   * Get Golden Lesson video URL (Day 1 with premium Sync Labs lipsync)
   * @param {string} archetype - User archetype (e.g., 'The Explorer', 'The Rebel', 'The Scientist')
   * @param {string} phase - Phase name (Hook, Fact1, Fact2, Fact3, Wisdom)
   * @returns {string} Full Supabase public URL
   */
  getGoldenLessonVideoUrl(archetype, phase) {
    const normalizedPhase = this.phaseMap[phase] || phase;
    const normalizedArchetype = archetype.replace(/\s+/g, '_');
    return `${this.supabaseStorageUrl}/production/videos/day_001_${normalizedPhase}_${normalizedArchetype}.mp4`;
  },
  
  /**
   * Check if Golden Lesson video exists for this archetype/phase
   * @param {string} archetype 
   * @param {string} phase 
   * @returns {Promise<boolean>}
   */
  async hasGoldenLessonVideo(archetype, phase) {
    const url = this.getGoldenLessonVideoUrl(archetype, phase);
    return await this.checkVideoAvailable(url);
  },
  
  /**
   * Play Golden Lesson video with automatic fallback
   * @param {string} archetype - User archetype
   * @param {string} phase - Phase name  
   * @param {Object} options - Additional options
   */
  async playGoldenLesson(archetype, phase, options = {}) {
    const videoUrl = this.getGoldenLessonVideoUrl(archetype, phase);
    
    console.log(`[KellyVideoPlayer] 🏆 Playing Golden Lesson: ${archetype} - ${phase}`);
    console.log(`[KellyVideoPlayer] Video URL: ${videoUrl}`);
    
    // Ensure container is visible
    if (this.videoContainer) {
      this.videoContainer.style.display = 'block';
    }
    
    await this.play({
      videoUrl,
      fallbackImage: options.fallbackImage || '/kelly/poses/kelly_welcome.png',
      fallbackAudio: options.audioUrl,
      onStart: () => {
        console.log(`[KellyVideoPlayer] 🎬 Golden Lesson started: ${archetype} - ${phase}`);
        if (this.videoContainer) {
          this.videoContainer.style.display = 'block';
        }
        options.onStart?.();
      },
      onEnd: () => {
        console.log(`[KellyVideoPlayer] ✅ Golden Lesson ended: ${archetype} - ${phase}`);
        // Keep video visible - let the lesson player handle hiding
        options.onEnd?.();
      },
      onError: (error) => {
        console.warn(`[KellyVideoPlayer] ⚠️ Golden Lesson error, using fallback: ${error}`);
        // Hide video container on error, show static avatar
        if (this.videoContainer) {
          this.videoContainer.style.display = 'none';
        }
        options.onError?.(error);
      },
    });
  },
  
  /**
   * Get all available Golden Lesson videos
   * @returns {Object} Map of archetype -> phase -> videoUrl
   */
  getGoldenLessonVideoMap() {
    const archetypes = ['The Explorer', 'The Rebel', 'The Scientist'];
    const phases = ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'];
    
    const map = {};
    for (const archetype of archetypes) {
      map[archetype] = {};
      for (const phase of phases) {
        map[archetype][phase] = this.getGoldenLessonVideoUrl(archetype, phase);
      }
    }
    return map;
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


