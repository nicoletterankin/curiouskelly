/**
 * Conversational Lesson System
 * Makes Kelly narrate everything, understand visuals, and guide learners naturally
 * 
 * Features:
 * - Pre-choice narration (Kelly describes options before buttons appear)
 * - Visual awareness (Kelly references visuals on screen)
 * - Content-rich phase names (not generic "Hook", "Cliff")
 * - Unified Learn + Grow track structure
 * - Smooth conversational flow
 */

(function() {
  'use strict';

  const ConversationalLesson = {
    currentTrack: 'learn', // 'learn' or 'grow'
    currentPhaseIndex: 0,
    lessonData: null,
    visualDisplay: null,
    
    /**
     * Initialize the conversational system
     */
    init() {
      // Detect if visual display system is available
      this.visualDisplay = window.LessonVisualDisplay || null;
      
      // Hook into lesson loading
      if (window.KellyLessonLoader) {
        const originalLoad = window.KellyLessonLoader.loadLesson;
        window.KellyLessonLoader.loadLesson = async (dayNumber, options) => {
          const result = await originalLoad.call(window.KellyLessonLoader, dayNumber, options);
          this.onLessonLoaded(dayNumber, result);
          return result;
        };
      }
      
      console.log('[ConversationalLesson] Initialized');
    },
    
    /**
     * Called when a lesson is loaded
     */
    onLessonLoaded(dayNumber, lessonPayload) {
      // Check if unified format is available
      const unified = window.CURIOUS_KELLY?.LOCAL_PACKS?.[dayNumber];
      if (unified?.is_unified) {
        this.lessonData = unified;
        console.log('[ConversationalLesson] Unified lesson data loaded for Day', dayNumber);
      }
    },
    
    /**
     * Get current track data (learn or grow)
     */
    getCurrentTrackData() {
      if (!this.lessonData) return null;
      return this.currentTrack === 'learn' 
        ? this.lessonData.learn 
        : this.lessonData.grow;
    },
    
    /**
     * Get current phase data
     */
    getCurrentPhase() {
      const trackData = this.getCurrentTrackData();
      if (!trackData || !trackData.phases) return null;
      return trackData.phases[this.currentPhaseIndex] || null;
    },
    
    /**
     * Render phase with conversational narration
     */
    async renderPhase(phaseIndex, track = 'learn') {
      this.currentPhaseIndex = phaseIndex;
      this.currentTrack = track;
      
      const phase = this.getCurrentPhase();
      if (!phase) {
        console.warn('[ConversationalLesson] No phase data found');
        return;
      }
      
      // Update phase title (use actual content name)
      this.updatePhaseTitle(phase.title);
      
      // Display visual if available
      if (phase.visual_url) {
        await this.displayVisual(phase.visual_url, phase.visual_description);
      }
      
      // Play narration with visual references
      await this.playNarration(phase);
      
      // Handle choices with pre-narration
      if (phase.has_choice && phase.options) {
        await this.handleChoicesWithNarration(phase);
      }
    },
    
    /**
     * Update phase title to use actual content name
     */
    updatePhaseTitle(title) {
      const phaseLabel = document.getElementById('phase-label');
      const captionPhase = document.getElementById('caption-phase');
      
      if (phaseLabel) {
        phaseLabel.textContent = title.toUpperCase();
      }
      
      if (captionPhase) {
        captionPhase.innerHTML = `<span class="phase-name">${title.toUpperCase()}</span>`;
      }
      
      // Update in lesson guide panel
      const guidePhaseTitle = document.getElementById('guide-phase-title');
      if (guidePhaseTitle) {
        guidePhaseTitle.textContent = title;
      }
    },
    
    /**
     * Display visual and make it available for reference
     */
    async displayVisual(visualUrl, description) {
      if (!visualUrl) return;
      
      // Use visual display system if available
      if (this.visualDisplay) {
        const dayNumber = this.lessonData?.meta?.day_number || 1;
        const phaseKey = this.getCurrentPhase()?.phase_key || 'welcome';
        await this.visualDisplay.show(dayNumber, phaseKey);
      }
      
      // Also update lesson visual element if it exists
      const lessonVisual = document.getElementById('lesson-visual');
      if (lessonVisual) {
        lessonVisual.src = visualUrl;
        lessonVisual.alt = description || 'Lesson visual';
        lessonVisual.style.display = 'block';
      }
    },
    
    /**
     * Play narration with visual references
     */
    async playNarration(phase) {
      const script = phase.script || '';
      const captionText = document.getElementById('caption-text');
      
      // Typewriter effect for caption
      if (captionText) {
        this.typewriterEffect(captionText, script);
      }
      
      // Play audio/video with script
      if (window.playPhaseMedia) {
        await window.playPhaseMedia({
          dbPhase: phase.phase_key || phase.id,
          script: script,
          videoUrl: phase.video_url
        });
      }
    },
    
    /**
     * Handle choices with pre-narration
     */
    async handleChoicesWithNarration(phase) {
      const options = phase.options || [];
      
      // Step 1: Play choice intro
      if (phase.choice_intro) {
        await this.playNarration({
          script: phase.choice_intro,
          phase_key: phase.phase_key
        });
        
        // Wait for intro to finish
        await this.waitForAudio(2000);
      }
      
      // Step 2: Play choice narration (describes options BEFORE they appear)
      if (phase.choice_narration) {
        await this.playNarration({
          script: phase.choice_narration,
          phase_key: phase.phase_key
        });
        
        // Wait for narration to finish
        await this.waitForAudio(3000);
      }
      
      // Step 3: Animate buttons appearing
      await this.showChoiceButtons(phase, options);
    },
    
    /**
     * Show choice buttons with animation
     */
    async showChoiceButtons(phase, options) {
      const container = document.getElementById('cliff-container');
      if (!container) {
        console.warn('[ConversationalLesson] Choice container not found');
        return;
      }
      
      // Update button labels
      const labelA = document.getElementById('cliff-label-a');
      const labelB = document.getElementById('cliff-label-b');
      const labelC = document.getElementById('cliff-label-c'); // For 3-option choices
      
      const btnA = document.getElementById('cliff-choice-a');
      const btnB = document.getElementById('cliff-choice-b');
      const btnC = document.getElementById('cliff-choice-c');
      
      // Update option A
      if (options[0] && labelA && btnA) {
        labelA.textContent = options[0].title;
        const descA = document.getElementById('cliff-desc-a');
        if (descA) descA.textContent = options[0].description || '';
        
        // Add icon if available
        if (options[0].icon) {
          const iconEl = btnA.querySelector('.cliff-icon') || document.createElement('span');
          iconEl.className = 'cliff-icon';
          iconEl.textContent = options[0].icon;
          if (!btnA.querySelector('.cliff-icon')) {
            btnA.insertBefore(iconEl, btnA.firstChild);
          }
        }
        
        // Load visual if available
        if (options[0].visual_url) {
          const imgA = document.getElementById('cliff-visual-a');
          if (imgA) {
            imgA.src = options[0].visual_url;
            imgA.style.display = 'block';
          }
        }
      }
      
      // Update option B
      if (options[1] && labelB && btnB) {
        labelB.textContent = options[1].title;
        const descB = document.getElementById('cliff-desc-b');
        if (descB) descB.textContent = options[1].description || '';
        
        if (options[1].icon) {
          const iconEl = btnB.querySelector('.cliff-icon') || document.createElement('span');
          iconEl.className = 'cliff-icon';
          iconEl.textContent = options[1].icon;
          if (!btnB.querySelector('.cliff-icon')) {
            btnB.insertBefore(iconEl, btnB.firstChild);
          }
        }
        
        if (options[1].visual_url) {
          const imgB = document.getElementById('cliff-visual-b');
          if (imgB) {
            imgB.src = options[1].visual_url;
            imgB.style.display = 'block';
          }
        }
      }
      
      // Update option C (if exists)
      if (options[2] && labelC && btnC) {
        labelC.textContent = options[2].title;
        const descC = document.getElementById('cliff-desc-c');
        if (descC) descC.textContent = options[2].description || '';
        
        if (options[2].icon) {
          const iconEl = btnC.querySelector('.cliff-icon') || document.createElement('span');
          iconEl.className = 'cliff-icon';
          iconEl.textContent = options[2].icon;
          if (!btnC.querySelector('.cliff-icon')) {
            btnC.insertBefore(iconEl, btnC.firstChild);
          }
        }
        
        // Show third button
        btnC.style.display = 'flex';
      }
      
      // Show container with animation
      container.hidden = false;
      container.classList.remove('hidden');
      container.classList.add('conversational-choice-appear');
      
      // Store phase data for choice handler
      window.__conversationalPhase = phase;
    },
    
    /**
     * Handle choice selection with conversational response
     */
    async handleChoice(choiceId) {
      const phase = window.__conversationalPhase;
      if (!phase || !phase.options) return;
      
      const selectedOption = phase.options.find(opt => opt.id === choiceId);
      if (!selectedOption) return;
      
      // Visual feedback
      const btnA = document.getElementById('cliff-choice-a');
      const btnB = document.getElementById('cliff-choice-b');
      const btnC = document.getElementById('cliff-choice-c');
      
      [btnA, btnB, btnC].forEach(btn => {
        if (btn) btn.classList.remove('selected');
      });
      
      const selectedBtn = choiceId === 'option_a' ? btnA : (choiceId === 'option_b' ? btnB : btnC);
      if (selectedBtn) {
        selectedBtn.classList.add('selected');
      }
      
      // Play Kelly's response
      const response = selectedOption.kelly_response || selectedOption.success_response || "Great choice!";
      
      // Update visual if available
      if (selectedOption.visual_url) {
        await this.displayVisual(selectedOption.visual_url, selectedOption.description);
      }
      
      // Play response narration
      await this.playNarration({
        script: response,
        phase_key: phase.phase_key
      });
      
      // Wait for response to finish
      await this.waitForAudio(3000);
      
      // Advance to next phase
      return true;
    },
    
    /**
     * Typewriter effect for text
     */
    typewriterEffect(element, text, speed = 30) {
      if (!element) return;
      
      element.textContent = '';
      let index = 0;
      
      const type = () => {
        if (index < text.length) {
          element.textContent += text.charAt(index);
          index++;
          setTimeout(type, speed);
        }
      };
      
      type();
    },
    
    /**
     * Wait for audio to finish (approximate)
     */
    waitForAudio(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    },
    
    /**
     * Get phase count for current track
     */
    getPhaseCount() {
      const trackData = this.getCurrentTrackData();
      return trackData?.phases?.length || 0;
    },
    
    /**
     * Switch between Learn and Grow tracks
     */
    switchTrack(track) {
      if (track !== 'learn' && track !== 'grow') return;
      this.currentTrack = track;
      this.currentPhaseIndex = 0;
      
      // Re-render current phase
      if (this.lessonData) {
        this.renderPhase(0, track);
      }
    }
  };
  
  // Initialize on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ConversationalLesson.init());
  } else {
    ConversationalLesson.init();
  }
  
  // Expose globally
  window.ConversationalLesson = ConversationalLesson;
})();





