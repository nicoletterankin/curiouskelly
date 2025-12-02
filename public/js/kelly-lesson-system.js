/**
 * Kelly Lesson System v1.0
 * Complete interactive lesson experience with:
 * - Phase progression (Welcome → Q1 → Q2 → Q3 → Hook → Complete)
 * - Kelly pose state machine
 * - Auto-advance timing
 * - Comments integration
 * 
 * Per CURIOUS-KELLY-COMPLETE-SYSTEM-SPEC.md
 */

// ═══════════════════════════════════════════════════════════════════
// LESSON PHASES CONSTANTS
// ═══════════════════════════════════════════════════════════════════

const LESSON_PHASES = {
  WELCOME: {
    id: 'welcome',
    order: 0,
    name: 'Welcome',
    kellyPose: 'welcome',
    kellyGreeting: true,
    autoAdvance: 5000,    // Auto-advance after 5s or user tap
    requiresAnswer: false,
  },
  Q1: {
    id: 'q1',
    order: 1,
    name: 'Question 1',
    kellyPose: 'thinking',
    hasOptions: true,
    requiresAnswer: true,
  },
  Q2: {
    id: 'q2', 
    order: 2,
    name: 'Question 2',
    kellyPose: 'thinking',
    hasOptions: true,
    requiresAnswer: true,
  },
  Q3: {
    id: 'q3',
    order: 3,
    name: 'Question 3',
    kellyPose: 'thinking',
    hasOptions: true,
    requiresAnswer: true,
  },
  HOOK: {
    id: 'hook',
    order: 4,
    name: 'The Hook',
    kellyPose: 'excited',
    kellyReveal: true,    // Kelly delivers the insight
    autoAdvance: 8000,
    requiresAnswer: false,
  },
  COMPLETE: {
    id: 'complete',
    order: 5,
    name: 'Complete',
    kellyPose: 'celebrating',
    showCompletion: true,
    showShare: true,
    requiresAnswer: false,
  }
};

// Phase array for iteration
const PHASES_ARRAY = Object.values(LESSON_PHASES).sort((a, b) => a.order - b.order);

// ═══════════════════════════════════════════════════════════════════
// KELLY POSE STATE MACHINE
// ═══════════════════════════════════════════════════════════════════

const KellyPoseManager = {
  currentPose: 'idle',
  currentMouthState: 'idle',
  container: null,
  
  // Map pose names to actual asset files
  // Based on /kelly/poses/ inventory:
  // - kelly_welcome.png   → Standing, hand on arm (welcoming)
  // - kelly_idle.png      → Chair, hand gesturing (presenting/explaining)
  // - kelly_hint.png      → Hand on chin (thinking/pondering)
  // - kelly_clasp.png     → Hands clasped in lap (attentive/encouraging)
  // - kelly_choice_left.png  → Pointing left (Option A)
  // - kelly_choice_right.png → Pointing right (Option B)
  // - kelly_listening.png → Hands clasped, leaning in (conversation mode)
  poseToAsset: {
    'welcome': 'kelly_welcome.png',
    'idle': 'kelly_idle.png',
    'thinking': 'kelly_hint.png',           // Hand on chin = thinking pose
    'explaining': 'kelly_idle.png',         // Animated gesture = explaining
    'listening': 'kelly_listening.png',
    'pointing-left': 'kelly_choice_left.png',
    'pointing-right': 'kelly_choice_right.png',
    'hint': 'kelly_hint.png',
    'excited': 'kelly_idle.png',            // Animated gesture = excited
    'celebrating': 'kelly_welcome.png',     // Warm welcome = celebrating
    'encouraging': 'kelly_clasp.png',       // Hands clasped = supportive
    'proud': 'kelly_welcome.png',           // Same as celebrating
    'attentive': 'kelly_clasp.png',         // Hands clasped = attentive
  },
  
  init(containerOrImg) {
    if (typeof containerOrImg === 'string') {
      this.container = document.getElementById(containerOrImg);
    } else {
      this.container = containerOrImg;
    }
    
    if (!this.container) {
      console.warn('[KellyPose] Container not found');
      return false;
    }
    
    // Find or create the Kelly image
    this.kellyImg = this.container.tagName === 'IMG' 
      ? this.container 
      : this.container.querySelector('img') || this.container.querySelector('.kelly-avatar');
    
    console.log('[KellyPose] Initialized with container:', this.container?.id);
    return true;
  },
  
  setPose(pose) {
    if (!this.kellyImg) {
      console.warn('[KellyPose] No Kelly image found');
      return;
    }
    
    const asset = this.poseToAsset[pose] || this.poseToAsset['idle'];
    const newSrc = `/kelly/poses/${asset}`;
    
    if (this.kellyImg.src !== newSrc) {
      this.currentPose = pose;
      this.kellyImg.src = newSrc;
      this.kellyImg.alt = `Kelly - ${pose}`;
      console.log(`[KellyPose] Pose changed: ${this.currentPose} → ${pose}`);
    }
    
    // Also dispatch event for other systems
    document.dispatchEvent(new CustomEvent('kelly-pose-change', { 
      detail: { pose, asset } 
    }));
  },
  
  setMouthState(state) {
    this.currentMouthState = state;
    // Future: animate mouth for lip-sync
    document.dispatchEvent(new CustomEvent('kelly-mouth-state', { 
      detail: { state } 
    }));
  },
  
  // Convenience methods for common transitions
  greet() { this.setPose('welcome'); },
  think() { this.setPose('thinking'); },
  explain() { this.setPose('explaining'); },
  listen() { this.setPose('listening'); },
  celebrate() { this.setPose('celebrating'); },
  encourage() { this.setPose('encouraging'); },
  pointLeft() { this.setPose('pointing-left'); },
  pointRight() { this.setPose('pointing-right'); },
  showExcitement() { this.setPose('excited'); },
  
  // Get pose for phase
  getPoseForPhase(phaseId) {
    const phase = Object.values(LESSON_PHASES).find(p => p.id === phaseId);
    return phase?.kellyPose || 'idle';
  },
  
  // Get pose for feedback
  getPoseForFeedback(isCorrect) {
    return isCorrect ? 'celebrating' : 'encouraging';
  }
};

// ═══════════════════════════════════════════════════════════════════
// LESSON CONTROLLER
// ═══════════════════════════════════════════════════════════════════

class LessonController {
  constructor(lessonData, options = {}) {
    this.lesson = lessonData;
    this.currentPhase = LESSON_PHASES.WELCOME;
    this.currentPhaseIndex = 0;
    this.responses = {};
    this.startTime = Date.now();
    this.autoAdvanceTimer = null;
    
    // Options
    this.options = {
      onPhaseChange: options.onPhaseChange || null,
      onComplete: options.onComplete || null,
      onChoiceSelected: options.onChoiceSelected || null,
      kellyAudio: options.kellyAudio || null,
      commentsSystem: options.commentsSystem || null,
      ...options
    };
    
    console.log('[LessonController] Initialized for lesson:', this.lesson?.topic);
  }
  
  async start() {
    console.log('[LessonController] Starting lesson...');
    await this.startPhase(LESSON_PHASES.WELCOME);
  }
  
  async startPhase(phase) {
    // Clear any existing auto-advance timer
    if (this.autoAdvanceTimer) {
      clearTimeout(this.autoAdvanceTimer);
      this.autoAdvanceTimer = null;
    }
    
    this.currentPhase = phase;
    this.currentPhaseIndex = phase.order;
    
    console.log(`[LessonController] Starting phase: ${phase.name} (${phase.id})`);
    
    // Update Kelly's pose
    KellyPoseManager.setPose(phase.kellyPose);
    
    // Load phase-specific comments
    if (this.options.commentsSystem) {
      this.options.commentsSystem.setPhase(phase.id);
    }
    
    // Notify listeners
    if (this.options.onPhaseChange) {
      this.options.onPhaseChange(phase, this);
    }
    
    // Kelly speaks if needed
    if ((phase.kellyGreeting || phase.kellyReveal) && this.options.kellyAudio) {
      const script = this.getPhaseScript(phase);
      if (script) {
        KellyPoseManager.setMouthState('speaking');
        await this.options.kellyAudio.speak(script);
        KellyPoseManager.setMouthState('idle');
      }
    }
    
    // Auto-advance if configured and doesn't require answer
    if (phase.autoAdvance && !phase.requiresAnswer) {
      this.autoAdvanceTimer = setTimeout(() => {
        this.nextPhase();
      }, phase.autoAdvance);
    }
    
    // Handle completion phase
    if (phase.showCompletion) {
      await this.handleCompletion();
    }
  }
  
  getPhaseScript(phase) {
    if (!this.lesson) return null;
    
    // Map phase to lesson data
    const phaseData = this.lesson.phases?.find(p => {
      const phaseType = p.type?.toLowerCase();
      return phaseType === phase.id || 
             (phase.id === 'q1' && phaseType === 'question' && p.order === 1) ||
             (phase.id === 'q2' && phaseType === 'question' && p.order === 2) ||
             (phase.id === 'q3' && phaseType === 'question' && p.order === 3) ||
             (phase.id === 'hook' && phaseType === 'wisdom') ||
             (phase.id === 'welcome' && phaseType === 'welcome');
    });
    
    if (phaseData) {
      // Get text from variants or default
      return phaseData.content?.text || 
             phaseData.text ||
             phaseData.content?.default?.text ||
             '';
    }
    
    return null;
  }
  
  handleOptionSelect(option, isCorrect) {
    // Record response
    this.responses[this.currentPhase.id] = {
      selected: option,
      correct: isCorrect,
      timestamp: Date.now()
    };
    
    // Kelly reacts
    KellyPoseManager.setPose(KellyPoseManager.getPoseForFeedback(isCorrect));
    
    // Notify listener
    if (this.options.onChoiceSelected) {
      this.options.onChoiceSelected(option, isCorrect, this.currentPhase);
    }
    
    // Speak feedback if audio available
    if (this.options.kellyAudio && option.response) {
      KellyPoseManager.setMouthState('speaking');
      this.options.kellyAudio.speak(option.response).then(() => {
        KellyPoseManager.setMouthState('idle');
        // Advance after feedback
        setTimeout(() => this.nextPhase(), 500);
      });
    } else {
      // Advance after delay
      setTimeout(() => this.nextPhase(), 2000);
    }
  }
  
  nextPhase() {
    const currentIndex = PHASES_ARRAY.findIndex(p => p.id === this.currentPhase.id);
    
    if (currentIndex < PHASES_ARRAY.length - 1) {
      this.startPhase(PHASES_ARRAY[currentIndex + 1]);
    } else {
      this.completeLesson();
    }
  }
  
  previousPhase() {
    const currentIndex = PHASES_ARRAY.findIndex(p => p.id === this.currentPhase.id);
    
    if (currentIndex > 0) {
      this.startPhase(PHASES_ARRAY[currentIndex - 1]);
    }
  }
  
  goToPhase(phaseId) {
    const phase = Object.values(LESSON_PHASES).find(p => p.id === phaseId);
    if (phase) {
      this.startPhase(phase);
    }
  }
  
  async handleCompletion() {
    console.log('[LessonController] Lesson complete!');
    
    // Kelly celebrates
    KellyPoseManager.celebrate();
    
    // Speak celebration
    if (this.options.kellyAudio) {
      KellyPoseManager.setMouthState('speaking');
      await this.options.kellyAudio.speak("You did amazing today! I'm so proud of you.");
      KellyPoseManager.setMouthState('idle');
    }
  }
  
  async completeLesson() {
    // Calculate stats
    const duration = Date.now() - this.startTime;
    const correctCount = Object.values(this.responses).filter(r => r.correct).length;
    
    const completionData = {
      lessonDay: this.lesson?.day_number,
      responses: this.responses,
      duration,
      correctCount,
      totalQuestions: Object.keys(this.responses).length
    };
    
    console.log('[LessonController] Lesson completed:', completionData);
    
    // Notify listener
    if (this.options.onComplete) {
      await this.options.onComplete(completionData);
    }
  }
  
  // Tap to advance (for auto-advance phases)
  tap() {
    if (this.autoAdvanceTimer && !this.currentPhase.requiresAnswer) {
      clearTimeout(this.autoAdvanceTimer);
      this.autoAdvanceTimer = null;
      this.nextPhase();
    }
  }
  
  // Pause/resume
  pause() {
    if (this.autoAdvanceTimer) {
      clearTimeout(this.autoAdvanceTimer);
      this.autoAdvanceTimer = null;
    }
  }
  
  resume() {
    if (this.currentPhase.autoAdvance && !this.currentPhase.requiresAnswer) {
      this.autoAdvanceTimer = setTimeout(() => {
        this.nextPhase();
      }, 2000); // Shorter delay on resume
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// PHASE-AWARE PROGRESS INDICATOR
// ═══════════════════════════════════════════════════════════════════

const PhaseProgressUI = {
  container: null,
  
  init(containerId = 'phase-progress') {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      // Create if doesn't exist
      this.container = document.createElement('div');
      this.container.id = containerId;
      this.container.className = 'phase-progress-container';
    }
    return this;
  },
  
  render(currentPhaseId) {
    if (!this.container) return;
    
    const phases = ['welcome', 'q1', 'q2', 'q3', 'hook', 'complete'];
    const currentIndex = phases.indexOf(currentPhaseId);
    
    this.container.innerHTML = phases.map((phaseId, i) => {
      let status = '';
      if (i < currentIndex) status = 'completed';
      else if (i === currentIndex) status = 'active';
      
      const phase = Object.values(LESSON_PHASES).find(p => p.id === phaseId);
      const label = phase?.name || phaseId;
      
      return `
        <div class="phase-dot ${status}" data-phase="${phaseId}" title="${label}">
          ${status === 'completed' ? '✓' : ''}
        </div>
        ${i < phases.length - 1 ? '<div class="phase-connector ' + (i < currentIndex ? 'completed' : '') + '"></div>' : ''}
      `;
    }).join('');
  },
  
  update(phaseId) {
    this.render(phaseId);
  }
};

// ═══════════════════════════════════════════════════════════════════
// COMPLETION OVERLAY
// ═══════════════════════════════════════════════════════════════════

const CompletionOverlay = {
  overlay: null,
  
  show(stats) {
    if (!this.overlay) {
      this.createOverlay();
    }
    
    const { lessonDay, duration, correctCount, totalQuestions } = stats;
    const minutes = Math.floor(duration / 60000);
    const accuracy = totalQuestions > 0 ? Math.round((correctCount / totalQuestions) * 100) : 100;
    
    this.overlay.innerHTML = `
      <div class="completion-modal">
        <div class="completion-kelly">
          <img src="/kelly/poses/kelly_welcome.png" alt="Kelly celebrating" />
        </div>
        <h2>🎉 Lesson Complete!</h2>
        <p class="completion-day">Day ${lessonDay}</p>
        
        <div class="completion-stats">
          <div class="stat">
            <span class="stat-value">${minutes || '<1'}</span>
            <span class="stat-label">minutes</span>
          </div>
          <div class="stat">
            <span class="stat-value">${accuracy}%</span>
            <span class="stat-label">accuracy</span>
          </div>
          <div class="stat">
            <span class="stat-value">${correctCount}/${totalQuestions}</span>
            <span class="stat-label">correct</span>
          </div>
        </div>
        
        <div class="completion-actions">
          <button class="btn-share" onclick="ShareHub.open()">
            📤 Share & Connect
          </button>
          <button class="btn-next" onclick="CompletionOverlay.hide(); navigateLesson(1);">
            Next Lesson →
          </button>
        </div>
        
        <button class="btn-close" onclick="CompletionOverlay.hide()">✕</button>
      </div>
    `;
    
    this.overlay.classList.add('visible');
  },
  
  hide() {
    if (this.overlay) {
      this.overlay.classList.remove('visible');
    }
  },
  
  createOverlay() {
    this.overlay = document.createElement('div');
    this.overlay.id = 'completion-overlay';
    this.overlay.className = 'completion-overlay';
    
    // Add styles if not present
    if (!document.getElementById('completion-overlay-styles')) {
      const styles = document.createElement('style');
      styles.id = 'completion-overlay-styles';
      styles.textContent = `
        .completion-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.9);
          backdrop-filter: blur(20px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 10000;
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.3s ease;
        }
        
        .completion-overlay.visible {
          opacity: 1;
          pointer-events: auto;
        }
        
        .completion-modal {
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
          border-radius: 24px;
          padding: 40px;
          max-width: 400px;
          text-align: center;
          position: relative;
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }
        
        .completion-kelly img {
          width: 120px;
          height: auto;
          margin-bottom: 20px;
        }
        
        .completion-modal h2 {
          font-size: 2rem;
          margin: 0 0 8px;
          color: #fff;
        }
        
        .completion-day {
          color: #3b82f6;
          font-size: 1.1rem;
          margin-bottom: 24px;
        }
        
        .completion-stats {
          display: flex;
          justify-content: center;
          gap: 24px;
          margin-bottom: 32px;
        }
        
        .completion-stats .stat {
          display: flex;
          flex-direction: column;
        }
        
        .completion-stats .stat-value {
          font-size: 1.8rem;
          font-weight: 700;
          color: #fff;
        }
        
        .completion-stats .stat-label {
          font-size: 0.85rem;
          color: #a1a1aa;
        }
        
        .completion-actions {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        
        .completion-actions button {
          padding: 14px 24px;
          border-radius: 12px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .btn-share {
          background: linear-gradient(135deg, #3b82f6, #8b5cf6);
          color: white;
          border: none;
        }
        
        .btn-share:hover {
          transform: scale(1.02);
          box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        }
        
        .btn-next {
          background: transparent;
          color: #fff;
          border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .btn-next:hover {
          background: rgba(255, 255, 255, 0.1);
        }
        
        .btn-close {
          position: absolute;
          top: 16px;
          right: 16px;
          background: transparent;
          border: none;
          color: #71717a;
          font-size: 1.2rem;
          cursor: pointer;
          padding: 8px;
        }
        
        .btn-close:hover {
          color: #fff;
        }
      `;
      document.head.appendChild(styles);
    }
    
    document.body.appendChild(this.overlay);
  }
};

// ═══════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════

window.LESSON_PHASES = LESSON_PHASES;
window.PHASES_ARRAY = PHASES_ARRAY;
window.KellyPoseManager = KellyPoseManager;
window.LessonController = LessonController;
window.PhaseProgressUI = PhaseProgressUI;
window.CompletionOverlay = CompletionOverlay;

console.log('[KellyLessonSystem] ✅ Loaded - Phase system, Kelly poses, completion overlay');

