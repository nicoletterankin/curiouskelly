/**
 * ═══════════════════════════════════════════════════════════════════════════
 * LESSON DIRECTOR INTEGRATION v1.0
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Wires together the Intelligent Director with the existing lesson player.
 * This script enhances the learn.html experience with automatic emotional
 * expression direction based on lesson content.
 * 
 * What it does:
 * 1. Hooks into phase transitions
 * 2. Analyzes lesson content in real-time
 * 3. Directs Kelly's expressions intelligently
 * 4. Coordinates with lip-sync and audio
 * 5. Reacts to user interactions
 * 
 * @author Curious Kelly Team
 * @version 1.0.0
 */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════════
  // CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════════════

  const CONFIG = {
    // Enable/disable features
    autoDirectExpression: true,
    logPerformance: true,
    showExpressionBadge: true,
    
    // Timing
    expressionTransitionMs: 400,
    minExpressionDurationMs: 1500,
    
    // Debug
    debug: false,
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // INTEGRATION STATE
  // ═══════════════════════════════════════════════════════════════════════════

  const LessonDirectorIntegration = {
    isInitialized: false,
    isEnabled: true,
    currentExpression: 'neutral',
    
    // References
    director: null,
    performance: null,
    lessonSystem: null,
    videoPlayer: null,
    
    // UI elements
    expressionBadge: null,
    
    // Stats
    stats: {
      phasesDirected: 0,
      expressionChanges: 0,
      userReactions: 0,
      sessionStart: null,
    },

    // ═════════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═════════════════════════════════════════════════════════════════════════

    init() {
      if (this.isInitialized) return;

      console.log('[LessonDirector] 🎬 Initializing lesson director integration...');

      // Get system references
      this.director = window.KellyDirector;
      this.performance = window.KellyPerformance;
      this.lessonSystem = window.kellyLessonSystem;
      this.videoPlayer = window.KellyVideoPlayer;

      // Verify systems
      if (!this.director) {
        console.warn('[LessonDirector] ⚠️ KellyDirector not found');
      }
      if (!this.performance) {
        console.warn('[LessonDirector] ⚠️ KellyPerformance not found');
      }

      // Initialize systems
      if (this.director && !this.director.isInitialized) {
        this.director.init();
      }
      if (this.performance && !this.performance.isInitialized) {
        this.performance.init();
      }

      // Hook into video player if available
      this.hookVideoPlayer();
      
      // Hook into chat/conversation if available
      this.hookConversation();
      
      // Create expression badge UI
      if (CONFIG.showExpressionBadge) {
        this.createExpressionBadge();
      }
      
      // Hook into global events
      this.setupGlobalHooks();

      // Start stats tracking
      this.stats.sessionStart = Date.now();

      this.isInitialized = true;
      console.log('[LessonDirector] ✅ Integration complete');
      
      // Log system status
      this.logSystemStatus();
    },

    // ═════════════════════════════════════════════════════════════════════════
    // VIDEO PLAYER INTEGRATION
    // ═════════════════════════════════════════════════════════════════════════

    hookVideoPlayer() {
      // Hook into KellyVideoPlayer if available
      if (!window.KellyVideoPlayer) {
        // Check again after a delay
        setTimeout(() => {
          if (window.KellyVideoPlayer) {
            this.hookVideoPlayer();
          }
        }, 1000);
        return;
      }

      const player = window.KellyVideoPlayer;

      // Wrap phase transition
      const originalTransitionToPhase = player.transitionToPhase?.bind(player);
      if (originalTransitionToPhase) {
        player.transitionToPhase = async (phaseIndex) => {
          // Get phase info before transition
          const phase = player.phases?.[phaseIndex];
          
          // Direct the expression based on phase
          if (phase && this.director) {
            this.directPhase(phase, phaseIndex);
          }
          
          // Call original
          return originalTransitionToPhase(phaseIndex);
        };
      }

      // Wrap showScript
      const originalShowScript = player.showScript?.bind(player);
      if (originalShowScript) {
        player.showScript = (text, options) => {
          // Analyze text for expression
          if (text && this.director && CONFIG.autoDirectExpression) {
            this.director.analyzeAndDirect(text);
          }
          
          return originalShowScript(text, options);
        };
      }

      console.log('[LessonDirector] ✅ Video player hooks installed');
    },

    // ═════════════════════════════════════════════════════════════════════════
    // CONVERSATION INTEGRATION
    // ═════════════════════════════════════════════════════════════════════════

    hookConversation() {
      // Hook into KellyConversation if available
      if (!window.KellyConversation) {
        setTimeout(() => {
          if (window.KellyConversation) {
            this.hookConversation();
          }
        }, 1000);
        return;
      }

      const conv = window.KellyConversation;

      // Wrap speak method
      const originalSpeak = conv.speak?.bind(conv);
      if (originalSpeak) {
        conv.speak = (text, options) => {
          // Direct expression based on what Kelly is saying
          if (text && this.director && CONFIG.autoDirectExpression) {
            this.director.analyzeAndDirect(text);
          }
          
          return originalSpeak(text, options);
        };
      }

      console.log('[LessonDirector] ✅ Conversation hooks installed');
    },

    // ═════════════════════════════════════════════════════════════════════════
    // GLOBAL HOOKS
    // ═════════════════════════════════════════════════════════════════════════

    setupGlobalHooks() {
      // Hook into answer validation (correct/incorrect)
      document.addEventListener('kelly:answer', (e) => {
        const isCorrect = e.detail?.correct;
        this.reactToAnswer(isCorrect);
      });

      // Hook into phase changes
      document.addEventListener('kelly:phaseChange', (e) => {
        const phase = e.detail?.phase;
        if (phase && this.director) {
          this.directPhase(phase);
        }
      });

      // Hook into lesson complete
      document.addEventListener('kelly:lessonComplete', () => {
        this.reactToLessonComplete();
      });

      // Hook into any chat message
      document.addEventListener('kelly:message', (e) => {
        const text = e.detail?.text;
        if (text && this.director && CONFIG.autoDirectExpression) {
          this.director.analyzeAndDirect(text);
        }
      });

      // Observe DOM for dynamically added content
      this.observeDynamicContent();
    },

    observeDynamicContent() {
      // Watch for Kelly's script text being updated
      const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          if (mutation.type === 'childList' || mutation.type === 'characterData') {
            const target = mutation.target;
            
            // Check if this is Kelly's script area
            if (target.classList?.contains('kelly-script') || 
                target.closest?.('.kelly-script') ||
                target.id === 'kelly-script') {
              const text = target.textContent?.trim();
              if (text && text.length > 10 && this.director && CONFIG.autoDirectExpression) {
                this.director.analyzeAndDirect(text);
              }
            }
          }
        }
      });

      // Observe the body for changes
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    },

    // ═════════════════════════════════════════════════════════════════════════
    // DIRECTION METHODS
    // ═════════════════════════════════════════════════════════════════════════

    directPhase(phase, phaseIndex = 0) {
      if (!this.director) return;

      const phaseType = phase.type || phase.visualPhase || 'welcome';
      const text = phase.text || phase.script || '';

      // Direct based on phase type and text content
      this.director.directPhase(phaseType, text);
      
      // Update stats
      this.stats.phasesDirected++;
      
      // Update badge
      this.updateExpressionBadge();
      
      if (CONFIG.logPerformance) {
        console.log(`[LessonDirector] 📖 Phase ${phaseIndex + 1}: ${phaseType} → ${this.director.currentExpression}`);
      }
    },

    reactToAnswer(isCorrect) {
      if (!this.director) return;

      if (isCorrect) {
        this.director.applyExpression('celebrating', { confidence: 0.95, force: true });
      } else {
        this.director.applyExpression('encouraging', { confidence: 0.9, force: true });
      }
      
      this.stats.userReactions++;
      this.updateExpressionBadge();
    },

    reactToLessonComplete() {
      if (!this.director) return;

      this.director.applyExpression('celebrating', { confidence: 1.0, force: true });
      this.stats.userReactions++;
      this.updateExpressionBadge();
      
      console.log('[LessonDirector] 🎉 Lesson complete!');
    },

    // ═════════════════════════════════════════════════════════════════════════
    // EXPRESSION BADGE UI
    // ═════════════════════════════════════════════════════════════════════════

    createExpressionBadge() {
      // Create floating badge showing current expression
      const badge = document.createElement('div');
      badge.id = 'kelly-expression-badge';
      badge.innerHTML = `
        <div class="expression-badge-content">
          <span class="expression-icon">🎭</span>
          <span class="expression-name">neutral</span>
          <span class="expression-confidence"></span>
        </div>
      `;
      badge.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        z-index: 9999;
        opacity: 0;
        transform: translateY(20px);
        transition: all 0.3s ease;
        pointer-events: none;
      `;

      document.body.appendChild(badge);
      this.expressionBadge = badge;

      // Show badge after a delay
      setTimeout(() => {
        badge.style.opacity = '1';
        badge.style.transform = 'translateY(0)';
      }, 1000);
    },

    updateExpressionBadge() {
      if (!this.expressionBadge || !this.director) return;

      const expression = this.director.currentExpression;
      const stats = this.director.getStats?.() || {};
      
      const iconMap = {
        happy: '😊',
        curious: '🤔',
        wisdom: '🦉',
        thinking: '💭',
        excited: '🤩',
        explaining: '📚',
        encouraging: '💪',
        listening: '👂',
        celebrating: '🎉',
        empathetic: '💝',
        neutral: '🙂',
      };
      
      const icon = iconMap[expression] || '🎭';
      const confidence = stats.averageConfidence 
        ? `${(stats.averageConfidence * 100).toFixed(0)}%`
        : '';

      const badge = this.expressionBadge;
      badge.querySelector('.expression-icon').textContent = icon;
      badge.querySelector('.expression-name').textContent = expression;
      badge.querySelector('.expression-confidence').textContent = confidence;
      
      // Flash animation
      badge.style.transform = 'translateY(0) scale(1.1)';
      setTimeout(() => {
        badge.style.transform = 'translateY(0) scale(1)';
      }, 200);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // UTILITY METHODS
    // ═════════════════════════════════════════════════════════════════════════

    logSystemStatus() {
      const status = {
        director: !!this.director?.isInitialized,
        performance: !!this.performance?.isInitialized,
        lessonSystem: !!this.lessonSystem,
        videoPlayer: !!window.KellyVideoPlayer,
        unityBridge: !!window.unityBridge?.ready,
        expressionBridge: !!window.KellyExpressionBridge?.isInitialized,
        lipSync: !!window.KellyLipSync?.isInitialized,
      };

      console.log('[LessonDirector] System status:', status);
      
      // Count ready systems
      const readyCount = Object.values(status).filter(Boolean).length;
      const totalCount = Object.keys(status).length;
      
      console.log(`[LessonDirector] ${readyCount}/${totalCount} systems ready`);
    },

    getStats() {
      return {
        ...this.stats,
        sessionDuration: Date.now() - this.stats.sessionStart,
        directorStats: this.director?.getStats?.() || {},
      };
    },

    enable() {
      this.isEnabled = true;
      CONFIG.autoDirectExpression = true;
      console.log('[LessonDirector] ✅ Enabled');
    },

    disable() {
      this.isEnabled = false;
      CONFIG.autoDirectExpression = false;
      console.log('[LessonDirector] ⏸️ Disabled');
    },

    toggle() {
      if (this.isEnabled) {
        this.disable();
      } else {
        this.enable();
      }
    },
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // AUTO-INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════════════

  function initWhenReady() {
    // Wait for dependencies
    const checkDeps = () => {
      const hasDirector = window.KellyDirector;
      const hasExpressionBridge = window.KellyExpressionBridge;
      
      if (hasDirector || hasExpressionBridge) {
        LessonDirectorIntegration.init();
      } else {
        // Retry
        setTimeout(checkDeps, 500);
      }
    };

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        setTimeout(checkDeps, 100);
      });
    } else {
      setTimeout(checkDeps, 100);
    }
  }

  // Start initialization
  initWhenReady();

  // Export globally
  window.LessonDirectorIntegration = LessonDirectorIntegration;

})();

