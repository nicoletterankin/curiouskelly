/**
 * ═══════════════════════════════════════════════════════════════════════════
 * KELLY UNIVERSAL ACCESS SYSTEM v1.0
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Ensures Kelly serves EVERYONE, EVERY DAY regardless of:
 * - Connection speed (2G to fiber)
 * - Device capability (old phones to gaming PCs)
 * - Accessibility needs (blind, deaf, cognitive)
 * - Language (with graceful degradation)
 * - Time zone (global reach)
 * 
 * Features:
 * - Connection-aware progressive loading
 * - Streak tracking for habit formation
 * - Accessibility announcements (ARIA live)
 * - Offline lesson support
 * - Learning outcome tracking
 * - Graceful error recovery
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
    // Storage keys
    STORAGE_PREFIX: 'kelly_',
    STREAK_KEY: 'kelly_streak_data',
    LEARNING_KEY: 'kelly_learning_progress',
    PREFERENCES_KEY: 'kelly_preferences',
    OFFLINE_QUEUE_KEY: 'kelly_offline_queue',
    
    // Connection thresholds (Mbps)
    CONNECTION: {
      SLOW: 0.5,      // 2G-like
      MEDIUM: 2,      // 3G-like
      FAST: 10,       // 4G/WiFi
    },
    
    // Streak settings
    STREAK: {
      GRACE_HOURS: 36,  // Hours before streak breaks
      PROTECTION_DAYS: 1,  // Free streak protection per week
    },
    
    // A11y settings
    ARIA_LIVE_DELAY: 100,  // ms before announcing
    REDUCED_MOTION_QUERY: '(prefers-reduced-motion: reduce)',
    HIGH_CONTRAST_QUERY: '(prefers-contrast: more)',
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // UNIVERSAL ACCESS SYSTEM
  // ═══════════════════════════════════════════════════════════════════════════

  const KellyUniversal = {
    // State
    isInitialized: false,
    connectionTier: 'fast',  // slow | medium | fast
    isOffline: false,
    prefersReducedMotion: false,
    prefersHighContrast: false,
    
    // References
    ariaLiveRegion: null,
    
    // ═════════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═════════════════════════════════════════════════════════════════════════

    init() {
      if (this.isInitialized) return this;

      console.log('[KellyUniversal] 🌍 Initializing Universal Access System...');

      // Detect connection quality
      this.detectConnection();

      // Setup offline/online handlers
      this.setupNetworkHandlers();

      // Setup accessibility features
      this.setupAccessibility();

      // Initialize streak tracking
      this.initStreakTracking();

      // Setup learning outcome tracking
      this.initLearningTracking();

      // Create ARIA live region
      this.createAriaLiveRegion();

      // Setup error recovery
      this.setupErrorRecovery();

      // Log capabilities
      this.logCapabilities();

      this.isInitialized = true;
      console.log('[KellyUniversal] ✅ Universal Access System ready');
      
      return this;
    },

    // ═════════════════════════════════════════════════════════════════════════
    // CONNECTION DETECTION & PROGRESSIVE LOADING
    // ═════════════════════════════════════════════════════════════════════════

    detectConnection() {
      // Check Network Information API
      const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      
      if (connection) {
        const effectiveType = connection.effectiveType;
        const downlink = connection.downlink;
        
        if (effectiveType === '2g' || effectiveType === 'slow-2g' || downlink < CONFIG.CONNECTION.SLOW) {
          this.connectionTier = 'slow';
        } else if (effectiveType === '3g' || downlink < CONFIG.CONNECTION.MEDIUM) {
          this.connectionTier = 'medium';
        } else {
          this.connectionTier = 'fast';
        }

        // Listen for changes
        connection.addEventListener('change', () => {
          this.detectConnection();
          this.adaptToConnection();
        });
      } else {
        // Fallback: measure actual download speed
        this.measureConnectionSpeed();
      }

      console.log(`[KellyUniversal] 📶 Connection tier: ${this.connectionTier}`);
    },

    async measureConnectionSpeed() {
      try {
        const startTime = performance.now();
        const response = await fetch('/images/brand/kelly-mark-circle-64.png', {
          method: 'HEAD',
          cache: 'no-store'
        });
        const endTime = performance.now();
        
        const latency = endTime - startTime;
        
        if (latency > 3000) {
          this.connectionTier = 'slow';
        } else if (latency > 1000) {
          this.connectionTier = 'medium';
        } else {
          this.connectionTier = 'fast';
        }
      } catch (e) {
        // Assume slow if can't measure
        this.connectionTier = 'slow';
      }
    },

    adaptToConnection() {
      const tier = this.connectionTier;
      
      // Dispatch event for other systems
      window.dispatchEvent(new CustomEvent('kelly:connectionChange', {
        detail: { tier }
      }));

      // Apply tier-specific optimizations
      if (tier === 'slow') {
        this.enableLowBandwidthMode();
      } else if (tier === 'medium') {
        this.enableMediumBandwidthMode();
      } else {
        this.enableFullExperience();
      }
    },

    enableLowBandwidthMode() {
      console.log('[KellyUniversal] 🐌 Low bandwidth mode enabled');
      
      // Disable video
      document.querySelectorAll('video').forEach(v => {
        v.pause();
        v.style.display = 'none';
      });
      
      // Use low-res images
      document.querySelectorAll('img[data-src-lowres]').forEach(img => {
        img.src = img.dataset.srcLowres;
      });
      
      // Disable animations
      document.body.classList.add('low-bandwidth');
      
      // Announce to user
      this.announce('Low bandwidth mode activated. Videos disabled for faster loading.');
    },

    enableMediumBandwidthMode() {
      console.log('[KellyUniversal] 📱 Medium bandwidth mode enabled');
      
      // Allow audio, compressed images
      document.body.classList.add('medium-bandwidth');
      document.body.classList.remove('low-bandwidth');
    },

    enableFullExperience() {
      console.log('[KellyUniversal] 🚀 Full experience enabled');
      
      document.body.classList.remove('low-bandwidth', 'medium-bandwidth');
    },

    // ═════════════════════════════════════════════════════════════════════════
    // OFFLINE/ONLINE HANDLING
    // ═════════════════════════════════════════════════════════════════════════

    setupNetworkHandlers() {
      this.isOffline = !navigator.onLine;

      window.addEventListener('online', () => {
        this.isOffline = false;
        this.onOnline();
      });

      window.addEventListener('offline', () => {
        this.isOffline = true;
        this.onOffline();
      });

      if (this.isOffline) {
        this.onOffline();
      }
    },

    onOffline() {
      console.log('[KellyUniversal] 📴 Offline mode');
      
      document.body.classList.add('offline-mode');
      this.announce('You are offline. Some features may be limited, but you can still learn!');
      
      // Show offline indicator
      this.showOfflineIndicator(true);
    },

    onOnline() {
      console.log('[KellyUniversal] 📶 Back online');
      
      document.body.classList.remove('offline-mode');
      this.announce('You are back online!');
      
      // Hide offline indicator
      this.showOfflineIndicator(false);
      
      // Sync queued actions
      this.syncOfflineQueue();
    },

    showOfflineIndicator(show) {
      let indicator = document.getElementById('kelly-offline-indicator');
      
      if (show && !indicator) {
        indicator = document.createElement('div');
        indicator.id = 'kelly-offline-indicator';
        indicator.setAttribute('role', 'status');
        indicator.setAttribute('aria-live', 'polite');
        indicator.innerHTML = `
          <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(90deg, #f59e0b, #d97706);
            color: white;
            padding: 8px 16px;
            text-align: center;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
          ">
            <span>📴</span>
            <span>You're offline - Lesson available from cache</span>
          </div>
        `;
        document.body.prepend(indicator);
      } else if (!show && indicator) {
        indicator.remove();
      }
    },

    queueOfflineAction(action) {
      const queue = this.getOfflineQueue();
      queue.push({
        ...action,
        timestamp: Date.now()
      });
      localStorage.setItem(CONFIG.OFFLINE_QUEUE_KEY, JSON.stringify(queue));
    },

    getOfflineQueue() {
      try {
        return JSON.parse(localStorage.getItem(CONFIG.OFFLINE_QUEUE_KEY)) || [];
      } catch (e) {
        return [];
      }
    },

    async syncOfflineQueue() {
      const queue = this.getOfflineQueue();
      if (queue.length === 0) return;

      console.log(`[KellyUniversal] 🔄 Syncing ${queue.length} offline actions...`);

      for (const action of queue) {
        try {
          // Process based on action type
          if (action.type === 'lesson_complete') {
            await this.syncLessonComplete(action);
          } else if (action.type === 'answer') {
            await this.syncAnswer(action);
          }
        } catch (e) {
          console.warn('[KellyUniversal] Failed to sync action:', e);
        }
      }

      // Clear queue
      localStorage.removeItem(CONFIG.OFFLINE_QUEUE_KEY);
      console.log('[KellyUniversal] ✅ Offline queue synced');
    },

    async syncLessonComplete(action) {
      // Implement based on your backend
      console.log('[KellyUniversal] Syncing lesson completion:', action);
    },

    async syncAnswer(action) {
      // Implement based on your backend
      console.log('[KellyUniversal] Syncing answer:', action);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ACCESSIBILITY
    // ═════════════════════════════════════════════════════════════════════════

    setupAccessibility() {
      // Check motion preference
      const motionQuery = window.matchMedia(CONFIG.REDUCED_MOTION_QUERY);
      this.prefersReducedMotion = motionQuery.matches;
      motionQuery.addEventListener('change', (e) => {
        this.prefersReducedMotion = e.matches;
        this.applyMotionPreference();
      });
      this.applyMotionPreference();

      // Check contrast preference
      const contrastQuery = window.matchMedia(CONFIG.HIGH_CONTRAST_QUERY);
      this.prefersHighContrast = contrastQuery.matches;
      contrastQuery.addEventListener('change', (e) => {
        this.prefersHighContrast = e.matches;
        this.applyContrastPreference();
      });
      this.applyContrastPreference();

      // Setup keyboard navigation
      this.setupKeyboardNavigation();
    },

    applyMotionPreference() {
      if (this.prefersReducedMotion) {
        document.body.classList.add('reduce-motion');
        console.log('[KellyUniversal] ♿ Reduced motion enabled');
      } else {
        document.body.classList.remove('reduce-motion');
      }
    },

    applyContrastPreference() {
      if (this.prefersHighContrast) {
        document.body.classList.add('high-contrast');
        console.log('[KellyUniversal] ♿ High contrast enabled');
      } else {
        document.body.classList.remove('high-contrast');
      }
    },

    setupKeyboardNavigation() {
      // Skip link
      this.createSkipLink();

      // Focus trapping for modals
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          // Close any open modals/panels
          document.querySelectorAll('[data-kelly-modal]').forEach(modal => {
            if (modal.style.display !== 'none') {
              modal.style.display = 'none';
              this.announce('Dialog closed');
            }
          });
        }
      });
    },

    createSkipLink() {
      if (document.getElementById('kelly-skip-link')) return;

      const skipLink = document.createElement('a');
      skipLink.id = 'kelly-skip-link';
      skipLink.href = '#main-content';
      skipLink.textContent = 'Skip to main content';
      skipLink.style.cssText = `
        position: fixed;
        top: -100px;
        left: 50%;
        transform: translateX(-50%);
        background: #3b82f6;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        z-index: 10001;
        transition: top 0.2s;
        font-weight: 600;
      `;
      
      skipLink.addEventListener('focus', () => {
        skipLink.style.top = '10px';
      });
      
      skipLink.addEventListener('blur', () => {
        skipLink.style.top = '-100px';
      });

      document.body.prepend(skipLink);
    },

    createAriaLiveRegion() {
      if (document.getElementById('kelly-aria-live')) return;

      const region = document.createElement('div');
      region.id = 'kelly-aria-live';
      region.setAttribute('role', 'status');
      region.setAttribute('aria-live', 'polite');
      region.setAttribute('aria-atomic', 'true');
      region.className = 'sr-only';
      region.style.cssText = `
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      `;

      document.body.appendChild(region);
      this.ariaLiveRegion = region;
    },

    /**
     * Announce text to screen readers
     * @param {string} message - Message to announce
     * @param {string} priority - 'polite' or 'assertive'
     */
    announce(message, priority = 'polite') {
      if (!this.ariaLiveRegion) return;

      this.ariaLiveRegion.setAttribute('aria-live', priority);
      
      // Clear and set after delay to ensure announcement
      this.ariaLiveRegion.textContent = '';
      
      setTimeout(() => {
        this.ariaLiveRegion.textContent = message;
        console.log(`[KellyUniversal] 📢 Announced: "${message}"`);
      }, CONFIG.ARIA_LIVE_DELAY);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // STREAK TRACKING (Habit Formation)
    // ═════════════════════════════════════════════════════════════════════════

    initStreakTracking() {
      this.checkAndUpdateStreak();
    },

    getStreakData() {
      try {
        const data = localStorage.getItem(CONFIG.STREAK_KEY);
        return data ? JSON.parse(data) : {
          currentStreak: 0,
          longestStreak: 0,
          lastLessonDate: null,
          protectionsUsed: 0,
          protectionResetWeek: null,
          totalLessons: 0,
          lessonsThisWeek: 0,
        };
      } catch (e) {
        return { currentStreak: 0, longestStreak: 0, lastLessonDate: null };
      }
    },

    saveStreakData(data) {
      localStorage.setItem(CONFIG.STREAK_KEY, JSON.stringify(data));
    },

    checkAndUpdateStreak() {
      const data = this.getStreakData();
      const now = new Date();
      const today = this.getDateString(now);
      
      if (!data.lastLessonDate) {
        // First time user
        return data;
      }

      const lastDate = new Date(data.lastLessonDate);
      const hoursSince = (now - lastDate) / (1000 * 60 * 60);
      
      // Reset weekly protections if new week
      const currentWeek = this.getWeekNumber(now);
      if (data.protectionResetWeek !== currentWeek) {
        data.protectionsUsed = 0;
        data.protectionResetWeek = currentWeek;
        data.lessonsThisWeek = 0;
      }

      // Check if streak should break
      if (hoursSince > CONFIG.STREAK.GRACE_HOURS) {
        // Check for protection
        if (data.protectionsUsed < CONFIG.STREAK.PROTECTION_DAYS) {
          data.protectionsUsed++;
          console.log('[KellyUniversal] 🛡️ Streak protection used!');
          this.announce('Streak protection activated! Your streak is safe.');
        } else {
          // Streak breaks
          console.log('[KellyUniversal] 💔 Streak broken');
          data.currentStreak = 0;
        }
      }

      this.saveStreakData(data);
      return data;
    },

    recordLessonComplete(dayNumber) {
      const data = this.getStreakData();
      const today = this.getDateString(new Date());
      
      // Don't count same day twice
      if (data.lastLessonDate === today) {
        return data;
      }

      const yesterday = this.getDateString(new Date(Date.now() - 86400000));
      
      if (data.lastLessonDate === yesterday || !data.lastLessonDate) {
        // Continue streak
        data.currentStreak++;
      } else {
        // Start new streak
        data.currentStreak = 1;
      }

      // Update stats
      data.lastLessonDate = today;
      data.totalLessons++;
      data.lessonsThisWeek++;
      
      if (data.currentStreak > data.longestStreak) {
        data.longestStreak = data.currentStreak;
      }

      this.saveStreakData(data);
      
      // Announce milestone streaks
      if (data.currentStreak === 7) {
        this.announce('Amazing! You have a 7-day learning streak! 🔥');
      } else if (data.currentStreak === 30) {
        this.announce('Incredible! 30-day learning streak! You are unstoppable! 🏆');
      } else if (data.currentStreak === 100) {
        this.announce('Legendary! 100-day learning streak! 🌟');
      }

      // Dispatch event
      window.dispatchEvent(new CustomEvent('kelly:streakUpdate', {
        detail: data
      }));

      console.log(`[KellyUniversal] 🔥 Streak: ${data.currentStreak} days`);
      return data;
    },

    getDateString(date) {
      return date.toISOString().split('T')[0];
    },

    getWeekNumber(date) {
      const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
      const dayNum = d.getUTCDay() || 7;
      d.setUTCDate(d.getUTCDate() + 4 - dayNum);
      const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
      return Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // LEARNING OUTCOME TRACKING
    // ═════════════════════════════════════════════════════════════════════════

    initLearningTracking() {
      // Initialize learning progress storage
      this.learningProgress = this.getLearningProgress();
    },

    getLearningProgress() {
      try {
        return JSON.parse(localStorage.getItem(CONFIG.LEARNING_KEY)) || {
          lessonsCompleted: [],
          correctAnswers: 0,
          totalAnswers: 0,
          topicsLearned: [],
          comprehensionScores: {},
        };
      } catch (e) {
        return { lessonsCompleted: [], correctAnswers: 0, totalAnswers: 0 };
      }
    },

    saveLearningProgress(data) {
      localStorage.setItem(CONFIG.LEARNING_KEY, JSON.stringify(data));
    },

    recordAnswer(dayNumber, phaseIndex, isCorrect, topic = null) {
      const progress = this.getLearningProgress();
      
      progress.totalAnswers++;
      if (isCorrect) {
        progress.correctAnswers++;
      }

      // Track comprehension per lesson
      const lessonKey = `day_${dayNumber}`;
      if (!progress.comprehensionScores[lessonKey]) {
        progress.comprehensionScores[lessonKey] = {
          correct: 0,
          total: 0,
          topic: topic,
        };
      }
      progress.comprehensionScores[lessonKey].total++;
      if (isCorrect) {
        progress.comprehensionScores[lessonKey].correct++;
      }

      this.saveLearningProgress(progress);

      // Queue for offline sync if needed
      if (this.isOffline) {
        this.queueOfflineAction({
          type: 'answer',
          dayNumber,
          phaseIndex,
          isCorrect,
          timestamp: Date.now()
        });
      }

      return progress;
    },

    recordLessonCompleteWithLearning(dayNumber, topic) {
      const progress = this.getLearningProgress();
      
      if (!progress.lessonsCompleted.includes(dayNumber)) {
        progress.lessonsCompleted.push(dayNumber);
      }

      if (topic && !progress.topicsLearned.includes(topic)) {
        progress.topicsLearned.push(topic);
      }

      this.saveLearningProgress(progress);

      // Also record streak
      this.recordLessonComplete(dayNumber);

      // Queue for offline sync
      if (this.isOffline) {
        this.queueOfflineAction({
          type: 'lesson_complete',
          dayNumber,
          topic,
          timestamp: Date.now()
        });
      }

      return progress;
    },

    getComprehensionRate() {
      const progress = this.getLearningProgress();
      if (progress.totalAnswers === 0) return 0;
      return Math.round((progress.correctAnswers / progress.totalAnswers) * 100);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // ERROR RECOVERY
    // ═════════════════════════════════════════════════════════════════════════

    setupErrorRecovery() {
      // Global error handler
      window.addEventListener('error', (event) => {
        console.error('[KellyUniversal] Error caught:', event.error);
        this.handleError(event.error);
      });

      // Unhandled promise rejection handler
      window.addEventListener('unhandledrejection', (event) => {
        console.error('[KellyUniversal] Unhandled rejection:', event.reason);
        this.handleError(event.reason);
      });
    },

    handleError(error) {
      // Don't show errors for minor issues
      const ignoredErrors = [
        'ResizeObserver loop',
        'Script error',
        'Network request failed',
      ];

      const errorMessage = error?.message || String(error);
      
      if (ignoredErrors.some(ie => errorMessage.includes(ie))) {
        return;
      }

      // Show friendly error message
      this.showErrorMessage();
    },

    showErrorMessage() {
      // Only show once per session
      if (this.errorShown) return;
      this.errorShown = true;

      const existing = document.getElementById('kelly-error-toast');
      if (existing) existing.remove();

      const toast = document.createElement('div');
      toast.id = 'kelly-error-toast';
      toast.setAttribute('role', 'alert');
      toast.innerHTML = `
        <div style="
          position: fixed;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: #1f2937;
          color: white;
          padding: 16px 24px;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          gap: 12px;
          z-index: 10000;
          max-width: 400px;
        ">
          <span style="font-size: 24px;">🤔</span>
          <div>
            <div style="font-weight: 600;">Something went wrong</div>
            <div style="font-size: 14px; opacity: 0.8;">But don't worry - Kelly's still here to help!</div>
          </div>
          <button onclick="this.parentElement.parentElement.remove()" style="
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 20px;
            padding: 4px;
          ">×</button>
        </div>
      `;
      document.body.appendChild(toast);

      // Auto-remove after 5 seconds
      setTimeout(() => {
        toast.remove();
        this.errorShown = false;
      }, 5000);
    },

    // ═════════════════════════════════════════════════════════════════════════
    // UTILITY METHODS
    // ═════════════════════════════════════════════════════════════════════════

    logCapabilities() {
      console.log('[KellyUniversal] 📊 Capabilities:', {
        connectionTier: this.connectionTier,
        isOffline: this.isOffline,
        prefersReducedMotion: this.prefersReducedMotion,
        prefersHighContrast: this.prefersHighContrast,
        streak: this.getStreakData().currentStreak,
        comprehensionRate: this.getComprehensionRate() + '%',
      });
    },

    /**
     * Get current user context for analytics/debugging
     */
    getContext() {
      return {
        connectionTier: this.connectionTier,
        isOffline: this.isOffline,
        prefersReducedMotion: this.prefersReducedMotion,
        prefersHighContrast: this.prefersHighContrast,
        streak: this.getStreakData(),
        learning: this.getLearningProgress(),
        deviceMemory: navigator.deviceMemory || 'unknown',
        hardwareConcurrency: navigator.hardwareConcurrency || 'unknown',
        platform: navigator.platform,
        language: navigator.language,
      };
    },

    /**
     * Check if user should see simplified experience
     */
    shouldSimplify() {
      return this.connectionTier === 'slow' || 
             this.prefersReducedMotion || 
             this.isOffline;
    },
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // CSS INJECTION
  // ═══════════════════════════════════════════════════════════════════════════

  const styles = document.createElement('style');
  styles.textContent = `
    /* Low bandwidth mode */
    .low-bandwidth video,
    .low-bandwidth iframe[src*="youtube"],
    .low-bandwidth .heavy-animation {
      display: none !important;
    }

    .low-bandwidth img {
      filter: grayscale(50%);
    }

    /* Medium bandwidth mode */
    .medium-bandwidth video {
      display: none !important;
    }

    /* Reduced motion */
    .reduce-motion *,
    .reduce-motion *::before,
    .reduce-motion *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
    }

    /* High contrast mode */
    .high-contrast {
      --text-primary: #ffffff !important;
      --text-secondary: #e0e0e0 !important;
      --bg-color: #000000 !important;
      --accent-primary: #ffff00 !important;
    }

    .high-contrast a {
      text-decoration: underline !important;
    }

    .high-contrast button,
    .high-contrast [role="button"] {
      border: 2px solid currentColor !important;
    }

    /* Offline mode */
    .offline-mode [data-requires-network] {
      opacity: 0.5;
      pointer-events: none;
    }

    /* Screen reader only */
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    /* Focus visible for keyboard users */
    :focus-visible {
      outline: 3px solid #3b82f6 !important;
      outline-offset: 2px !important;
    }
  `;
  document.head.appendChild(styles);

  // ═══════════════════════════════════════════════════════════════════════════
  // AUTO-INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════════════

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => KellyUniversal.init());
  } else {
    KellyUniversal.init();
  }

  // Export globally
  window.KellyUniversal = KellyUniversal;

})();

