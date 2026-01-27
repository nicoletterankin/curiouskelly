/**
 * Kelly Curriculum Integration
 * 
 * Lightweight integration layer that connects the curriculum knowledge base
 * with the lesson player without interfering with existing functionality.
 * 
 * Features:
 * - Tracks lesson access when lessons load
 * - Tracks phase completion when phases advance
 * - Updates learning stats in UI
 * - Provides "Ask Kelly" functionality
 */

(function() {
  'use strict';

  const KellyCurriculumIntegration = {
    initialized: false,
    currentDay: null,
    currentPhase: null,

    /**
     * Initialize integration (called after DOM ready)
     */
    init() {
      if (this.initialized) return;
      
      // Wait for knowledge base to be available
      if (!window.KellyCurriculumKB) {
        setTimeout(() => this.init(), 500);
        return;
      }

      this.initialized = true;
      this.setupTracking();
      this.updateLearningStats();
      
      console.log('[KellyCurriculumIntegration] Initialized');
    },

    /**
     * Setup tracking hooks
     */
    setupTracking() {
      // Track lesson access when lesson loads
      this.trackLessonLoad();
      
      // Track phase completion when phases advance
      this.trackPhaseCompletion();
      
      // Update stats periodically
      setInterval(() => this.updateLearningStats(), 30000); // Every 30 seconds
    },

    /**
     * Track lesson load by intercepting applyLoadedLesson
     */
    trackLessonLoad() {
      // Find the applyLoadedLesson function in global scope
      const originalApplyLoadedLesson = window.applyLoadedLesson;
      
      if (originalApplyLoadedLesson) {
        // Wrap it to add tracking
        window.applyLoadedLesson = (dayNumber, payload) => {
          const result = originalApplyLoadedLesson(dayNumber, payload);
          
          // Track in knowledge base
          if (window.KellyCurriculumKB && dayNumber) {
            window.KellyCurriculumKB.trackLessonAccess(dayNumber);
            this.currentDay = dayNumber;
          }
          
          // Update stats
          this.updateLearningStats();
          
          return result;
        };
      } else {
        // Fallback: watch for lesson changes via state
        this.watchStateChanges();
      }
    },

    /**
     * Watch state changes for lesson/phase tracking
     */
    watchStateChanges() {
      let lastDay = null;
      let lastPhase = null;
      
      setInterval(() => {
        // Try to access state from learn.html
        if (typeof window.state !== 'undefined') {
          const currentDay = window.state?.currentDay;
          const currentPhase = window.state?.currentPhase;
          
          // Track day change
          if (currentDay && currentDay !== lastDay) {
            if (window.KellyCurriculumKB) {
              window.KellyCurriculumKB.trackLessonAccess(currentDay);
            }
            lastDay = currentDay;
            this.currentDay = currentDay;
            this.updateLearningStats();
          }
          
          // Track phase change
          if (currentPhase !== null && currentPhase !== lastPhase && lastPhase !== null) {
            // Phase advanced - track completion
            if (window.KellyCurriculumKB && this.currentDay) {
              const phaseNames = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
              const phaseName = phaseNames[lastPhase];
              if (phaseName) {
                window.KellyCurriculumKB.trackLessonAccess(this.currentDay, phaseName);
              }
            }
            lastPhase = currentPhase;
            this.currentPhase = currentPhase;
          } else if (currentPhase !== null) {
            lastPhase = currentPhase;
            this.currentPhase = currentPhase;
          }
        }
      }, 1000); // Check every second
    },

    /**
     * Track phase completion
     */
    trackPhaseCompletion() {
      // Listen for phase advance events
      document.addEventListener('click', (e) => {
        // Check if it's a phase navigation button
        const phaseBtn = e.target.closest('[data-phase]');
        if (phaseBtn && this.currentDay) {
          const phaseIndex = parseInt(phaseBtn.dataset.phase);
          const phaseNames = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
          const phaseName = phaseNames[phaseIndex];
          
          if (phaseName && window.KellyCurriculumKB) {
            window.KellyCurriculumKB.trackLessonAccess(this.currentDay, phaseName);
          }
        }
      });
    },

    /**
     * Update learning stats in UI
     */
    updateLearningStats() {
      if (!window.KellyCurriculumKB) return;
      
      try {
        const stats = window.KellyCurriculumKB.getStats();
        
        // Update journey lessons count
        const journeyCount = document.getElementById('journey-lessons-count');
        if (journeyCount) {
          journeyCount.textContent = `${stats.learningHistory.lessonsSeen}/365`;
        }
        
        // Update streak if element exists
        const streakElement = document.getElementById('learning-streak');
        if (streakElement) {
          streakElement.textContent = `${stats.learningHistory.currentStreak} days`;
        }
      } catch (e) {
        // Silently fail - stats update is non-critical
      }
    },

    /**
     * Open "Ask Kelly" panel
     */
    openAskKellyPanel() {
      // Open settings panel first
      if (typeof window.openPanel === 'function') {
        window.openPanel('settings');
      }
      
      // Then show BYOK section
      setTimeout(() => {
        const byokSection = document.getElementById('byok-prompt-section');
        if (byokSection) {
          byokSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else if (window.KellyBYOKPromptGenerator) {
          // Initialize BYOK UI if not already done
          window.KellyBYOKPromptGenerator.init();
          setTimeout(() => {
            const byokSection = document.getElementById('byok-prompt-section');
            if (byokSection) {
              byokSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }, 100);
        }
      }, 300);
    }
  };

  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      KellyCurriculumIntegration.init();
    });
  } else {
    KellyCurriculumIntegration.init();
  }

  // Expose globally
  window.KellyCurriculumIntegration = KellyCurriculumIntegration;
  
  // Expose openAskKellyPanel globally for onclick handler
  window.openAskKellyPanel = () => KellyCurriculumIntegration.openAskKellyPanel();
})();





