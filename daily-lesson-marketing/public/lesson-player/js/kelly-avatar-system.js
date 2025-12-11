/**
 * Kelly Avatar System - Primary Visual Experience
 * 
 * A playful, reactive avatar system for the 5-phase learning journey.
 * Think "Hot or Not" meets educational delight.
 * 
 * Phase Flow:
 * 1. WELCOME - Hook moment, Kelly introduces herself
 * 2. Q1 (Fact 1) - First "hot or not" choice + reaction
 * 3. Q2 (Fact 2) - Second choice + reaction
 * 4. Q3 (Fact 3) - Third choice + reaction
 * 5. WISDOM - Kelly delivers the wisdom moment
 * 
 * Lesson Structure (PhaseDNA):
 * - 2 options per question
 * - 2 teaching moments (after choices)
 * - Single topic
 * - Endless variants for daily delight
 */

export class KellyAvatarSystem {
  constructor(containerElement) {
    this.container = containerElement;
    this.currentPhase = 'welcome';
    this.currentAge = 27; // Default adult Kelly
    this.currentPose = 'curious';
    this.currentLanguage = 'en';
    this.isAnimating = false;
    this.isSpeaking = false;
    
    // Avatar state
    this.state = {
      pose: 'curious',        // curious, explaining, celebrating, listening, wisdom
      age: 27,                // 3, 9, 15, 27, 48, 82
      emotion: 'neutral',     // neutral, excited, thoughtful, celebratory
      breathing: true,
      blinking: true,
      eyeGaze: 'center'       // center, left, right, down, up
    };
    
    // Phase definitions
    this.phases = {
      welcome: {
        defaultPose: 'curious',
        transitions: ['q1'],
        teachingMoment: false
      },
      q1: {
        defaultPose: 'curious',
        transitions: ['q1_reaction_a', 'q1_reaction_b'],
        teachingMoment: true
      },
      q1_reaction_a: {
        defaultPose: 'explaining',
        transitions: ['q2'],
        teachingMoment: true,
        duration: 3000 // Auto-advance after teaching
      },
      q1_reaction_b: {
        defaultPose: 'celebrating',
        transitions: ['q2'],
        teachingMoment: true,
        duration: 3000
      },
      q2: {
        defaultPose: 'curious',
        transitions: ['q2_reaction_a', 'q2_reaction_b'],
        teachingMoment: true
      },
      q2_reaction_a: {
        defaultPose: 'explaining',
        transitions: ['q3'],
        teachingMoment: true,
        duration: 3000
      },
      q2_reaction_b: {
        defaultPose: 'celebrating',
        transitions: ['q3'],
        teachingMoment: true,
        duration: 3000
      },
      q3: {
        defaultPose: 'curious',
        transitions: ['q3_reaction_a', 'q3_reaction_b'],
        teachingMoment: true
      },
      q3_reaction_a: {
        defaultPose: 'explaining',
        transitions: ['wisdom'],
        teachingMoment: true,
        duration: 3000
      },
      q3_reaction_b: {
        defaultPose: 'celebrating',
        transitions: ['wisdom'],
        teachingMoment: true,
        duration: 3000
      },
      wisdom: {
        defaultPose: 'wisdom',
        transitions: ['complete'],
        teachingMoment: false
      }
    };
    
    this.init();
  }
  
  init() {
    this.createAvatarContainer();
    this.setupEventListeners();
    this.startIdleAnimations();
    console.log('[KellyAvatar] Initialized - Ready for 5-phase journey');
  }
  
  createAvatarContainer() {
    // Create main avatar structure
    const avatarHTML = `
      <div class="kelly-avatar-wrapper" data-phase="${this.currentPhase}">
        
        <!-- Base Image Layer -->
        <div class="kelly-image-layer">
          <img 
            id="kelly-base-image" 
            class="kelly-base-image" 
            src="/lessons/images/kelly-directors-chair-curious.jpeg"
            alt="Kelly"
            data-pose="curious"
            data-age="27"
          />
        </div>
        
        <!-- SVG Effects Overlay -->
        <svg class="kelly-effects-overlay" viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <!-- Glow effect for excitement -->
            <filter id="kelly-glow">
              <feGaussianBlur stdDeviation="10" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
            
            <!-- Shimmer effect for transitions -->
            <linearGradient id="kelly-shimmer" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:rgba(255,255,255,0);stop-opacity:0" />
              <stop offset="50%" style="stop-color:rgba(255,255,255,0.3);stop-opacity:1" />
              <stop offset="100%" style="stop-color:rgba(255,255,255,0);stop-opacity:0" />
              <animate attributeName="x1" values="-100%;200%" dur="2s" repeatCount="indefinite" />
              <animate attributeName="x2" values="0%;300%" dur="2s" repeatCount="indefinite" />
            </linearGradient>
            
            <!-- Sparkle effect for celebration -->
            <radialGradient id="kelly-sparkle">
              <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#FFD700;stop-opacity:0" />
            </radialGradient>
          </defs>
          
          <!-- Breathing Aura (Subtle pulse around Kelly) -->
          <ellipse 
            class="breathing-aura" 
            cx="960" 
            cy="800" 
            rx="300" 
            ry="150"
            fill="rgba(217, 119, 87, 0.05)"
            opacity="0.5"
          >
            <animate attributeName="ry" values="150;165;150" dur="4s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="4s" repeatCount="indefinite"/>
          </ellipse>
          
          <!-- Eye Blink Overlays (positioned over eyes) -->
          <g class="blink-overlay" opacity="0">
            <!-- Left eye -->
            <ellipse cx="820" cy="420" rx="30" ry="5" fill="rgba(0,0,0,0.3)"/>
            <!-- Right eye -->
            <ellipse cx="1100" cy="420" rx="30" ry="5" fill="rgba(0,0,0,0.3)"/>
          </g>
          
          <!-- Speaking Indicator (animated ring around mouth area) -->
          <g class="speaking-indicator" opacity="0">
            <circle cx="960" cy="580" r="40" fill="none" stroke="rgba(217, 119, 87, 0.4)" stroke-width="3">
              <animate attributeName="r" values="40;50;40" dur="0.6s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.8;0.3;0.8" dur="0.6s" repeatCount="indefinite"/>
            </circle>
          </g>
          
          <!-- Celebration Sparkles -->
          <g class="celebration-sparkles" opacity="0">
            <circle class="sparkle" cx="700" cy="300" r="8" fill="url(#kelly-sparkle)">
              <animate attributeName="opacity" values="0;1;0" dur="1s" begin="0s" repeatCount="indefinite"/>
            </circle>
            <circle class="sparkle" cx="1220" cy="280" r="10" fill="url(#kelly-sparkle)">
              <animate attributeName="opacity" values="0;1;0" dur="1s" begin="0.3s" repeatCount="indefinite"/>
            </circle>
            <circle class="sparkle" cx="850" cy="250" r="6" fill="url(#kelly-sparkle)">
              <animate attributeName="opacity" values="0;1;0" dur="1s" begin="0.6s" repeatCount="indefinite"/>
            </circle>
            <circle class="sparkle" cx="1050" cy="320" r="9" fill="url(#kelly-sparkle)">
              <animate attributeName="opacity" values="0;1;0" dur="1s" begin="0.9s" repeatCount="indefinite"/>
            </circle>
          </g>
          
          <!-- Thinking Dots (for contemplative moments) -->
          <g class="thinking-dots" opacity="0" transform="translate(1100, 350)">
            <circle cx="0" cy="0" r="8" fill="rgba(74, 144, 226, 0.6)">
              <animate attributeName="cy" values="0;-15;0" dur="1.2s" begin="0s" repeatCount="indefinite"/>
            </circle>
            <circle cx="25" cy="0" r="8" fill="rgba(74, 144, 226, 0.6)">
              <animate attributeName="cy" values="0;-15;0" dur="1.2s" begin="0.2s" repeatCount="indefinite"/>
            </circle>
            <circle cx="50" cy="0" r="8" fill="rgba(74, 144, 226, 0.6)">
              <animate attributeName="cy" values="0;-15;0" dur="1.2s" begin="0.4s" repeatCount="indefinite"/>
            </circle>
          </g>
          
          <!-- Age Transition Shimmer -->
          <rect 
            class="age-transition-shimmer" 
            x="0" 
            y="0" 
            width="1920" 
            height="1080" 
            fill="url(#kelly-shimmer)"
            opacity="0"
            pointer-events="none"
          />
          
        </svg>
        
        <!-- Emotion Label (for debugging/testing) -->
        <div class="kelly-state-debug" style="display: none;">
          <span class="state-phase">Phase: <strong>${this.currentPhase}</strong></span>
          <span class="state-pose">Pose: <strong>${this.currentPose}</strong></span>
          <span class="state-age">Age: <strong>${this.currentAge}</strong></span>
        </div>
        
      </div>
    `;
    
    this.container.innerHTML = avatarHTML;
    
    // Cache DOM references
    this.elements = {
      wrapper: this.container.querySelector('.kelly-avatar-wrapper'),
      baseImage: this.container.querySelector('#kelly-base-image'),
      breathingAura: this.container.querySelector('.breathing-aura'),
      blinkOverlay: this.container.querySelector('.blink-overlay'),
      speakingIndicator: this.container.querySelector('.speaking-indicator'),
      celebrationSparkles: this.container.querySelector('.celebration-sparkles'),
      thinkingDots: this.container.querySelector('.thinking-dots'),
      ageShimmer: this.container.querySelector('.age-transition-shimmer')
    };

    // --- VIDEO LAYER INTEGRATION ---
    // We add a video element dynamically if it doesn't exist
    this.videoElement = document.createElement('video');
    this.videoElement.id = 'kelly-avatar-video';
    this.videoElement.className = 'kelly-avatar-video';
    this.videoElement.style.position = 'absolute';
    this.videoElement.style.top = '0';
    this.videoElement.style.left = '0';
    this.videoElement.style.width = '100%';
    this.videoElement.style.height = '100%';
    this.videoElement.style.objectFit = 'cover';
    this.videoElement.style.opacity = '0'; // Hidden by default
    this.videoElement.style.transition = 'opacity 0.5s ease';
    this.videoElement.muted = true; // Audio handled separately
    this.videoElement.playsInline = true;
    
    // Insert video before the SVG overlay so effects sit on top
    const imageLayer = this.container.querySelector('.kelly-image-layer');
    if (imageLayer) {
        imageLayer.appendChild(this.videoElement);
    }
  }
  
  setupEventListeners() {
    // Listen for audio events (if audio player exists)
    document.addEventListener('kelly-audio-playing', () => this.setSpeaking(true));
    document.addEventListener('kelly-audio-paused', () => this.setSpeaking(false));
    document.addEventListener('kelly-audio-ended', () => this.setSpeaking(false));
    
    // Listen for phase changes
    document.addEventListener('kelly-phase-change', (e) => {
      this.setPhase(e.detail.phase, e.detail.choice);
    });
    
    // Listen for age changes
    document.addEventListener('kelly-age-change', (e) => {
      this.setAge(e.detail.age);
    });
  }
  
  /**
   * Start idle animations (breathing, blinking)
   */
  startIdleAnimations() {
    // Breathing is handled by CSS/SVG animations
    
    // Blink every 3-6 seconds randomly
    this.blinkInterval = setInterval(() => {
      const delay = 3000 + Math.random() * 3000; // 3-6 seconds
      setTimeout(() => this.blink(), delay);
    }, 100);
    
    console.log('[KellyAvatar] Idle animations started');
  }
  
  /**
   * Perform a blink animation
   */
  blink() {
    if (!this.state.blinking || this.isAnimating) return;
    
    const blinkOverlay = this.elements.blinkOverlay;
    
    // Quick blink animation
    blinkOverlay.style.transition = 'opacity 0.1s ease-in-out';
    blinkOverlay.setAttribute('opacity', '1');
    
    setTimeout(() => {
      blinkOverlay.setAttribute('opacity', '0');
    }, 150);
  }
  
  /**
   * Set Kelly's speaking state
   */
  setSpeaking(isSpeaking) {
    this.isSpeaking = isSpeaking;
    
    const indicator = this.elements.speakingIndicator;
    indicator.style.transition = 'opacity 0.3s ease';
    indicator.setAttribute('opacity', isSpeaking ? '1' : '0');
    
    // Dispatch event for other systems
    document.dispatchEvent(new CustomEvent('kelly-speaking-change', {
      detail: { isSpeaking }
    }));
  }
  
  /**
   * Set the current lesson phase
   * @param {string} phase - Phase name (welcome, q1, q2, q3, wisdom)
   * @param {string} choice - User choice (a or b) for reaction phases
   * @param {string} videoUrl - Optional video URL for this phase
   */
  async setPhase(phase, choice = null, videoUrl = null) {
    console.log(`[KellyAvatar] Phase transition: ${this.currentPhase} → ${phase}`, { choice, videoUrl });
    
    // Handle reaction phases
    if (choice && phase.startsWith('q')) {
      const reactionPhase = `${phase}_reaction_${choice}`;
      phase = reactionPhase;
    }
    
    const phaseConfig = this.phases[phase];
    if (!phaseConfig) {
      console.warn(`[KellyAvatar] Unknown phase: ${phase}`);
      return;
    }
    
    this.currentPhase = phase;
    this.elements.wrapper.setAttribute('data-phase', phase);
    
    // VIDEO HANDOFF: If we have a video URL, play it
    if (videoUrl) {
       this.playVideo(videoUrl);
    } else {
       // Fallback to static pose if no video
       this.stopVideo();
       await this.setPose(phaseConfig.defaultPose);
    }
    
    // Show appropriate effects
    this.showPhaseEffects(phase);
    
    // Auto-advance if configured
    if (phaseConfig.duration) {
      setTimeout(() => {
        const nextPhase = phaseConfig.transitions[0];
        if (nextPhase) {
          // Note: Auto-advance won't know the next video URL without a look-ahead mechanism
          // Ideally, the 'app.js' controller handles timing via audio-ended events
          // So we might remove internal auto-advance if app.js is driving
          // this.setPhase(nextPhase); 
        }
      }, phaseConfig.duration);
    }
    
    // Dispatch event
    document.dispatchEvent(new CustomEvent('kelly-phase-changed', {
      detail: { phase, pose: phaseConfig.defaultPose }
    }));
  }

  playVideo(url) {
    if (!this.videoElement) return;
    
    console.log(`[KellyAvatar] Playing video: ${url}`);
    this.videoElement.src = url;
    this.videoElement.style.opacity = '1';
    this.elements.baseImage.style.opacity = '0'; // Hide static image
    
    this.videoElement.play().catch(e => console.warn('Video play failed (autoplay policy?):', e));
    
    // Ensure loop if it's a loopable phase, or play once?
    // Usually these are talking heads matching audio
    this.videoElement.loop = false; 
    
    // Sync with audio? The audio is played by 'app.js' via <audio> tag.
    // Ideally, we should use the video's audio track if it has one (HeyGen does).
    // BUT 'app.js' is currently designed to play separate MP3s.
    // For now, let's mute the video and let 'app.js' drive the audio to keep logic simple.
    this.videoElement.muted = true;
  }

  stopVideo() {
    if (!this.videoElement) return;
    this.videoElement.pause();
    this.videoElement.style.opacity = '0';
    this.elements.baseImage.style.opacity = '1'; // Show static image
  }
  
  /**
   * Show visual effects appropriate to the phase
   */
  showPhaseEffects(phase) {
    // Hide all effect layers first
    this.elements.celebrationSparkles.setAttribute('opacity', '0');
    this.elements.thinkingDots.setAttribute('opacity', '0');
    
    // Show phase-specific effects
    if (phase.includes('reaction_b') || phase === 'wisdom') {
      // Celebration effects
      this.elements.celebrationSparkles.setAttribute('opacity', '1');
      setTimeout(() => {
        this.elements.celebrationSparkles.setAttribute('opacity', '0');
      }, 3000);
    } else if (phase.includes('reaction_a')) {
      // Thinking/explaining effects
      this.elements.thinkingDots.setAttribute('opacity', '1');
      setTimeout(() => {
        this.elements.thinkingDots.setAttribute('opacity', '0');
      }, 2000);
    }
  }
  
  /**
   * Change Kelly's pose (visual state)
   * @param {string} pose - Pose name (curious, explaining, celebrating, listening, wisdom)
   */
  async setPose(pose) {
    if (this.currentPose === pose) return;
    
    const validPoses = ['curious', 'explaining', 'celebrating', 'listening', 'wisdom'];
    if (!validPoses.includes(pose)) {
      console.warn(`[KellyAvatar] Invalid pose: ${pose}`);
      return;
    }
    
    console.log(`[KellyAvatar] Pose change: ${this.currentPose} → ${pose}`);
    
    this.isAnimating = true;
    
    // Fade out current image
    this.elements.baseImage.style.transition = 'opacity 0.3s ease-out';
    this.elements.baseImage.style.opacity = '0';
    
    await this.wait(300);
    
    // Change image source
    const imagePath = `/lessons/images/kelly-directors-chair-${pose}.png`;
    this.elements.baseImage.src = imagePath;
    this.elements.baseImage.setAttribute('data-pose', pose);
    
    // Fade in new image
    this.elements.baseImage.style.opacity = '1';
    
    await this.wait(300);
    
    this.currentPose = pose;
    this.state.pose = pose;
    this.isAnimating = false;
    
    // Dispatch event
    document.dispatchEvent(new CustomEvent('kelly-pose-changed', {
      detail: { pose }
    }));
  }
  
  /**
   * Change Kelly's age with visual transition
   * @param {number} age - Target age (3, 9, 15, 27, 48, 82)
   */
  async setAge(age) {
    const validAges = [3, 9, 15, 27, 48, 82];
    if (!validAges.includes(age)) {
      console.warn(`[KellyAvatar] Invalid age: ${age}. Using closest valid age.`);
      // Find closest valid age
      age = validAges.reduce((prev, curr) => 
        Math.abs(curr - age) < Math.abs(prev - age) ? curr : prev
      );
    }
    
    if (this.currentAge === age) return;
    
    console.log(`[KellyAvatar] Age change: ${this.currentAge} → ${age}`);
    
    this.isAnimating = true;
    
    // Show shimmer effect
    this.elements.ageShimmer.style.transition = 'opacity 0.5s ease';
    this.elements.ageShimmer.setAttribute('opacity', '0.7');
    
    await this.wait(500);
    
    // Fade out current image
    this.elements.baseImage.style.opacity = '0';
    
    await this.wait(300);
    
    // Change to age-specific image
    // Use upperbody pose for age variants (or closest available)
    const imagePath = `/images/kelly/kelly-age${age}-upperbody-16x9.png`;
    this.elements.baseImage.src = imagePath;
    this.elements.baseImage.setAttribute('data-age', age);
    
    // Fade in new image
    this.elements.baseImage.style.opacity = '1';
    
    await this.wait(300);
    
    // Hide shimmer
    this.elements.ageShimmer.setAttribute('opacity', '0');
    
    this.currentAge = age;
    this.state.age = age;
    this.isAnimating = false;
    
    // Dispatch event
    document.dispatchEvent(new CustomEvent('kelly-age-changed', {
      detail: { age }
    }));
  }
  
  /**
   * Play a reaction animation (for hot-or-not choices)
   * @param {string} reaction - Reaction type (celebrate, explain, curious)
   */
  async playReaction(reaction) {
    console.log(`[KellyAvatar] Playing reaction: ${reaction}`);
    
    switch (reaction) {
      case 'celebrate':
        await this.setPose('celebrating');
        this.elements.celebrationSparkles.setAttribute('opacity', '1');
        await this.wait(2000);
        this.elements.celebrationSparkles.setAttribute('opacity', '0');
        break;
        
      case 'explain':
        await this.setPose('explaining');
        this.elements.thinkingDots.setAttribute('opacity', '1');
        await this.wait(1500);
        this.elements.thinkingDots.setAttribute('opacity', '0');
        break;
        
      case 'curious':
        await this.setPose('curious');
        break;
        
      case 'listening':
        await this.setPose('listening');
        break;
        
      case 'wisdom':
        await this.setPose('wisdom');
        break;
    }
  }
  
  /**
   * Quick visual "pop" for delightful feedback
   */
  async pop() {
    this.elements.wrapper.style.transition = 'transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
    this.elements.wrapper.style.transform = 'scale(1.05)';
    
    await this.wait(200);
    
    this.elements.wrapper.style.transform = 'scale(1)';
  }
  
  /**
   * Utility: Wait for specified milliseconds
   */
  wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  /**
   * Cleanup
   */
  destroy() {
    if (this.blinkInterval) {
      clearInterval(this.blinkInterval);
    }
    console.log('[KellyAvatar] Destroyed');
  }
}

// Export for module usage
export default KellyAvatarSystem;







