/**
 * Kelly Video Assets System v1.0
 * 
 * GOLDEN LESSON: Day 1 "Starting Fresh"
 * 
 * This system delivers pre-computed Kelly videos for lessons.
 * 75 videos generated for Day 1 (5 phases × 15 archetypes)
 * 
 * Integration:
 *   1. Detects current day/phase/archetype
 *   2. Fetches matching video from Supabase
 *   3. Replaces static Kelly image with talking video
 *   4. Falls back to images if video unavailable
 */

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2ZvcmciLCJyb2xlIjoiYW5vbiIsImlhdCI6MTcyOTU0NzE2NywiZXhwIjoyMDQ1MTIzMTY3fQ.z-8XCGB0VgW2V3zC-Jb2WeNX6NklPIQRrYhujO-l86I';

// Day 1 "Starting Fresh" lesson structure
const DAY_1_STRUCTURE = {
  day: 1,
  title: 'Starting Fresh',
  topic: 'The science of fresh starts and new beginnings',
  phases: ['hook', 'q1', 'q2', 'q3', 'wisdom'],
  archetypes: [
    'Mystic', 'Scientist', 'Survivor',
    'The Architect', 'The Diplomat', 'The Empath',
    'The Explorer', 'The MacGyver', 'The Mystic',
    'The Provider', 'The Rebel', 'The Scientist',
    'The Storyteller', 'The Strategist', 'The Survivor'
  ]
};

// Phase display names
const PHASE_NAMES = {
  hook: 'Hook',
  q1: 'Fact 1',
  q2: 'Fact 2',
  q3: 'Fact 3',
  wisdom: 'Wisdom'
};

// Phase to emotion mapping for video selection
const PHASE_EMOTIONS = {
  hook: 'excited',
  q1: 'curious',
  q2: 'explaining',
  q3: 'thoughtful',
  wisdom: 'heartfelt'
};

class KellyVideoManager {
  constructor(options = {}) {
    this.dayNumber = options.day || 1;
    this.currentPhase = options.phase || 'hook';
    this.archetype = options.archetype || 'The Explorer';
    this.videoCache = new Map();
    this.audioCache = new Map();
    this.videoElement = null;
    this.imageElement = null;
    this.containerElement = null;
    this.isPlaying = false;
    this.onPhaseComplete = options.onPhaseComplete || (() => {});
    this.onVideoStart = options.onVideoStart || (() => {});
    this.onVideoEnd = options.onVideoEnd || (() => {});
    
    console.log('[KellyVideo] Initialized for Day', this.dayNumber);
  }
  
  setContainer(containerId) {
    this.containerElement = document.getElementById(containerId);
    if (!this.containerElement) {
      console.error('[KellyVideo] Container not found:', containerId);
      return;
    }
    
    // Find existing image element
    this.imageElement = this.containerElement.querySelector('img');
    
    // Create video element
    this.videoElement = document.createElement('video');
    this.videoElement.id = 'kelly-video';
    this.videoElement.playsInline = true;
    this.videoElement.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center 20%;
      display: none;
      z-index: 10;
    `;
    
    this.videoElement.addEventListener('ended', () => this.handleVideoEnd());
    this.videoElement.addEventListener('error', (e) => this.handleVideoError(e));
    
    this.containerElement.appendChild(this.videoElement);
    console.log('[KellyVideo] Container set:', containerId);
  }
  
  async fetchVideoUrl(phase, archetype) {
    const cacheKey = `${this.dayNumber}-${phase}-${archetype}`;
    
    if (this.videoCache.has(cacheKey)) {
      return this.videoCache.get(cacheKey);
    }
    
    try {
      // Normalize archetype for query
      const archetypeQuery = archetype.replace(/\s+/g, '_');
      
      const response = await fetch(
        `${SUPABASE_URL}/rest/v1/kelly_video_assets?` +
        `day_number=eq.${this.dayNumber}&` +
        `phase=eq.${phase}&` +
        `age_bucket=eq.${archetype}&` +
        `asset_type=eq.video&` +
        `select=public_url`,
        {
          headers: {
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
          }
        }
      );
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const url = data[0].public_url;
        this.videoCache.set(cacheKey, url);
        return url;
      }
      
      console.warn('[KellyVideo] No video found for:', cacheKey);
      return null;
      
    } catch (error) {
      console.error('[KellyVideo] Fetch error:', error);
      return null;
    }
  }
  
  async fetchAudioUrl(phase, archetype) {
    const cacheKey = `${this.dayNumber}-${phase}-${archetype}-audio`;
    
    if (this.audioCache.has(cacheKey)) {
      return this.audioCache.get(cacheKey);
    }
    
    try {
      const response = await fetch(
        `${SUPABASE_URL}/rest/v1/kelly_video_assets?` +
        `day_number=eq.${this.dayNumber}&` +
        `phase=eq.${phase}&` +
        `age_bucket=eq.${archetype}&` +
        `asset_type=eq.audio&` +
        `select=public_url`,
        {
          headers: {
            'apikey': SUPABASE_ANON_KEY,
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
          }
        }
      );
      
      const data = await response.json();
      
      if (data && data.length > 0) {
        const url = data[0].public_url;
        this.audioCache.set(cacheKey, url);
        return url;
      }
      
      return null;
      
    } catch (error) {
      console.error('[KellyVideo] Audio fetch error:', error);
      return null;
    }
  }
  
  async playPhase(phase, archetype = this.archetype) {
    if (this.isPlaying) {
      this.stop();
    }
    
    this.currentPhase = phase;
    this.archetype = archetype;
    
    const videoUrl = await this.fetchVideoUrl(phase, archetype);
    
    if (!videoUrl) {
      console.warn('[KellyVideo] No video available, using static image');
      this.showImage();
      return false;
    }
    
    try {
      this.videoElement.src = videoUrl;
      this.showVideo();
      
      await this.videoElement.play();
      this.isPlaying = true;
      this.onVideoStart(phase, archetype);
      
      console.log('[KellyVideo] Playing:', phase, archetype);
      return true;
      
    } catch (error) {
      console.error('[KellyVideo] Play error:', error);
      this.showImage();
      return false;
    }
  }
  
  showVideo() {
    if (this.videoElement) {
      this.videoElement.style.display = 'block';
    }
    if (this.imageElement) {
      this.imageElement.style.opacity = '0';
    }
  }
  
  showImage() {
    if (this.videoElement) {
      this.videoElement.style.display = 'none';
    }
    if (this.imageElement) {
      this.imageElement.style.opacity = '1';
    }
  }
  
  stop() {
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.currentTime = 0;
    }
    this.isPlaying = false;
    this.showImage();
  }
  
  pause() {
    if (this.videoElement && this.isPlaying) {
      this.videoElement.pause();
    }
  }
  
  resume() {
    if (this.videoElement && this.isPlaying) {
      this.videoElement.play();
    }
  }
  
  handleVideoEnd() {
    this.isPlaying = false;
    this.onVideoEnd(this.currentPhase, this.archetype);
    this.onPhaseComplete(this.currentPhase);
    
    // Keep showing last frame briefly, then transition
    setTimeout(() => {
      this.showImage();
    }, 500);
  }
  
  handleVideoError(error) {
    console.error('[KellyVideo] Video error:', error);
    this.isPlaying = false;
    this.showImage();
  }
  
  async preloadPhase(phase, archetype = this.archetype) {
    const url = await this.fetchVideoUrl(phase, archetype);
    if (url) {
      // Create hidden video element for preloading
      const preloadVideo = document.createElement('video');
      preloadVideo.preload = 'auto';
      preloadVideo.src = url;
      console.log('[KellyVideo] Preloaded:', phase);
    }
  }
  
  async preloadLesson() {
    console.log('[KellyVideo] Preloading Day', this.dayNumber, 'for archetype:', this.archetype);
    
    for (const phase of DAY_1_STRUCTURE.phases) {
      await this.preloadPhase(phase, this.archetype);
    }
    
    console.log('[KellyVideo] ✅ Lesson preloaded');
  }
  
  setArchetype(archetype) {
    this.archetype = archetype;
    console.log('[KellyVideo] Archetype set:', archetype);
  }
  
  getAvailableArchetypes() {
    return DAY_1_STRUCTURE.archetypes;
  }
  
  getCurrentState() {
    return {
      day: this.dayNumber,
      phase: this.currentPhase,
      archetype: this.archetype,
      isPlaying: this.isPlaying
    };
  }
}

// Golden Lesson Controller - Manages Day 1 "Starting Fresh"
class GoldenLessonController {
  constructor() {
    this.videoManager = null;
    this.currentPhaseIndex = 0;
    this.phases = DAY_1_STRUCTURE.phases;
    this.selectedArchetype = 'The Explorer';
    this.autoAdvance = true;
    this.onLessonComplete = () => {};
  }
  
  async init(containerId, archetype = 'The Explorer') {
    this.selectedArchetype = archetype;
    
    this.videoManager = new KellyVideoManager({
      day: 1,
      archetype: archetype,
      onPhaseComplete: (phase) => this.handlePhaseComplete(phase),
      onVideoStart: (phase) => this.updateUI('playing', phase),
      onVideoEnd: (phase) => this.updateUI('ended', phase)
    });
    
    this.videoManager.setContainer(containerId);
    
    // Preload first phase
    await this.videoManager.preloadPhase(this.phases[0], archetype);
    
    console.log('[GoldenLesson] Initialized - Day 1: Starting Fresh');
    return this;
  }
  
  async start() {
    this.currentPhaseIndex = 0;
    await this.playCurrentPhase();
  }
  
  async playCurrentPhase() {
    const phase = this.phases[this.currentPhaseIndex];
    const success = await this.videoManager.playPhase(phase, this.selectedArchetype);
    
    // Preload next phase
    if (this.currentPhaseIndex < this.phases.length - 1) {
      this.videoManager.preloadPhase(
        this.phases[this.currentPhaseIndex + 1],
        this.selectedArchetype
      );
    }
    
    return success;
  }
  
  handlePhaseComplete(phase) {
    console.log('[GoldenLesson] Phase complete:', phase);
    
    if (this.autoAdvance && this.currentPhaseIndex < this.phases.length - 1) {
      this.currentPhaseIndex++;
      setTimeout(() => this.playCurrentPhase(), 1000);
    } else if (this.currentPhaseIndex >= this.phases.length - 1) {
      console.log('[GoldenLesson] ✅ Lesson complete!');
      this.onLessonComplete();
    }
  }
  
  updateUI(state, phase) {
    // Dispatch custom event for UI updates
    window.dispatchEvent(new CustomEvent('kelly-video-state', {
      detail: { state, phase, archetype: this.selectedArchetype }
    }));
  }
  
  next() {
    if (this.currentPhaseIndex < this.phases.length - 1) {
      this.currentPhaseIndex++;
      this.playCurrentPhase();
    }
  }
  
  previous() {
    if (this.currentPhaseIndex > 0) {
      this.currentPhaseIndex--;
      this.playCurrentPhase();
    }
  }
  
  goToPhase(phaseIndex) {
    if (phaseIndex >= 0 && phaseIndex < this.phases.length) {
      this.currentPhaseIndex = phaseIndex;
      this.playCurrentPhase();
    }
  }
  
  setArchetype(archetype) {
    this.selectedArchetype = archetype;
    if (this.videoManager) {
      this.videoManager.setArchetype(archetype);
    }
  }
  
  getProgress() {
    return {
      current: this.currentPhaseIndex + 1,
      total: this.phases.length,
      phase: this.phases[this.currentPhaseIndex],
      phaseName: PHASE_NAMES[this.phases[this.currentPhaseIndex]],
      percent: ((this.currentPhaseIndex + 1) / this.phases.length) * 100
    };
  }
}

// Export
window.KellyVideoManager = KellyVideoManager;
window.GoldenLessonController = GoldenLessonController;
window.DAY_1_STRUCTURE = DAY_1_STRUCTURE;
window.PHASE_NAMES = PHASE_NAMES;

console.log('[KellyVideo] ✅ v1.0 loaded - Golden Lesson: Day 1 "Starting Fresh"');

