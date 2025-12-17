/**
 * Kelly Personality Layer
 * 
 * Makes Kelly aware of:
 * - Time of day (morning/afternoon/evening/night)
 * - Day of week (weekday vs weekend)
 * - Season (hemisphere-aware!)
 * - User's streak and progress
 * - Special occasions and holidays
 * 
 * Kelly isn't just translated — she's localized and contextual.
 * 
 * Usage:
 *   await KellyPersonality.init();
 *   const greeting = KellyPersonality.getGreeting('Sarah');
 *   const comment = KellyPersonality.getSeasonalComment();
 */

(function() {
  'use strict';

  // ============================================
  // STATE
  // ============================================
  
  let _context = null;
  let _contextPromise = null;
  let _userName = null;
  
  // ============================================
  // CONTEXT LOADING
  // ============================================
  
  /**
   * Load geo-context from API
   */
  async function loadContext() {
    if (_context) return _context;
    if (_contextPromise) return _contextPromise;
    
    _contextPromise = (async () => {
      try {
        const response = await fetch('/api/geo-context');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        _context = await response.json();
        console.log('[KellyPersonality] Context loaded:', _context.timeOfDay, _context.season, _context.country);
        return _context;
      } catch (error) {
        console.warn('[KellyPersonality] Context fetch failed:', error);
        // Use sensible defaults
        _context = getDefaultContext();
        return _context;
      } finally {
        _contextPromise = null;
      }
    })();
    
    return _contextPromise;
  }
  
  /**
   * Get default context (fallback)
   */
  function getDefaultContext() {
    const now = new Date();
    const hour = now.getHours();
    const month = now.getMonth();
    const dayOfWeek = now.getDay();
    
    return {
      timeOfDay: getTimeOfDay(hour),
      hour,
      dayOfWeek: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][dayOfWeek],
      isWeekend: dayOfWeek === 0 || dayOfWeek === 6,
      season: getSeason(month, 'northern'),
      hemisphere: 'northern',
      country: 'US',
    };
  }
  
  function getTimeOfDay(hour) {
    if (hour >= 5 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 21) return 'evening';
    return 'night';
  }
  
  function getSeason(month, hemisphere) {
    const northernSeasons = {
      0: 'winter', 1: 'winter', 2: 'spring',
      3: 'spring', 4: 'spring', 5: 'summer',
      6: 'summer', 7: 'summer', 8: 'autumn',
      9: 'autumn', 10: 'autumn', 11: 'winter'
    };
    
    const season = northernSeasons[month];
    
    if (hemisphere === 'southern') {
      const flip = { spring: 'autumn', summer: 'winter', autumn: 'spring', winter: 'summer' };
      return flip[season];
    }
    
    return season;
  }
  
  // ============================================
  // GREETINGS
  // ============================================
  
  /**
   * Get a personalized greeting
   */
  function getGreeting(userName) {
    const t = window.KellyI18n?.t || ((key) => key);
    const ctx = _context || getDefaultContext();
    
    // Base time greeting
    let greetingKey = `kelly.greetings.${ctx.timeOfDay}.default`;
    
    // Check for special variants
    if (ctx.isWeekend) {
      greetingKey = `kelly.greetings.${ctx.timeOfDay}.weekend`;
    } else if (ctx.dayOfWeek === 'Monday') {
      greetingKey = `kelly.greetings.morning.monday`;
    } else if (ctx.dayOfWeek === 'Friday') {
      greetingKey = `kelly.greetings.morning.friday`;
    }
    
    let greeting = t(greetingKey);
    
    // Fallback if specific key not found
    if (greeting === greetingKey) {
      greeting = t(`kelly.greetings.${ctx.timeOfDay}.default`);
    }
    
    // Still fallback? Use common greeting
    if (greeting.includes('.')) {
      greeting = t(`common.greeting.${ctx.timeOfDay}`);
    }
    
    // Add name if provided
    if (userName) {
      greeting = t('kelly.greetings.withName', { greeting, name: userName });
      // Fallback format
      if (greeting.includes('{{')) {
        greeting = `${t(`common.greeting.${ctx.timeOfDay}`)}, ${userName}!`;
      }
    }
    
    return greeting;
  }
  
  /**
   * Get a short greeting (just the time-based part)
   */
  function getShortGreeting() {
    const t = window.KellyI18n?.t || ((key) => key);
    const ctx = _context || getDefaultContext();
    return t(`common.greeting.${ctx.timeOfDay}`);
  }
  
  // ============================================
  // SEASONAL & CONTEXTUAL COMMENTS
  // ============================================
  
  /**
   * Get a seasonal comment
   */
  function getSeasonalComment() {
    const t = window.KellyI18n?.t || ((key) => key);
    const ctx = _context || getDefaultContext();
    return t(`kelly.seasonal.${ctx.season}.comment`);
  }
  
  /**
   * Get time-appropriate encouragement
   */
  function getEncouragement(type = 'start') {
    const t = window.KellyI18n?.t || ((key) => key);
    const ctx = _context || getDefaultContext();
    
    // Get array of encouragements
    const key = `kelly.encouragement.${type}`;
    const value = t(key);
    
    // If it's an array (from JSON), pick random
    if (Array.isArray(value)) {
      return value[Math.floor(Math.random() * value.length)];
    }
    
    // Otherwise return as-is or generate default
    if (value === key) {
      const defaults = {
        start: "Let's explore!",
        midLesson: "You're doing great!",
        complete: "You did it!"
      };
      return defaults[type] || "Let's learn!";
    }
    
    return value;
  }
  
  /**
   * Get streak-based encouragement
   */
  function getStreakMessage(streakCount) {
    const t = window.KellyI18n?.t || ((key) => key);
    
    const milestones = [365, 100, 30, 14, 7, 3, 1];
    for (const milestone of milestones) {
      if (streakCount >= milestone) {
        const key = `kelly.encouragement.streak.day${milestone}`;
        const message = t(key);
        if (message !== key) return message;
      }
    }
    
    return t('kelly.encouragement.streak.day1');
  }
  
  /**
   * Get return welcome message (for users coming back after absence)
   */
  function getReturnMessage(daysAway) {
    const t = window.KellyI18n?.t || ((key) => key);
    
    if (daysAway <= 1) {
      return t('kelly.encouragement.return.welcome_back');
    } else if (daysAway < 7) {
      return t('kelly.encouragement.return.missed_days', { days: daysAway });
    } else {
      return t('kelly.encouragement.return.no_pressure');
    }
  }
  
  // ============================================
  // LESSON COMMENTARY
  // ============================================
  
  /**
   * Get lesson introduction
   */
  function getLessonIntro(topic, isToday = true) {
    const t = window.KellyI18n?.t || ((key) => key);
    const key = isToday ? 'kelly.lessonIntro.today' : 'kelly.lessonIntro.past';
    return t(key, { topic });
  }
  
  /**
   * Get phase transition text
   */
  function getPhaseTransition(phase) {
    const t = window.KellyI18n?.t || ((key) => key);
    return t(`kelly.lessonPhases.${phase}.transition`);
  }
  
  /**
   * Get reaction to quiz answer
   */
  function getQuizReaction(isCorrect) {
    const t = window.KellyI18n?.t || ((key) => key);
    const key = isCorrect ? 'kelly.reactions.correct' : 'kelly.reactions.incorrect';
    const value = t(key);
    
    if (Array.isArray(value)) {
      return value[Math.floor(Math.random() * value.length)];
    }
    
    return value === key ? (isCorrect ? "Correct!" : "Not quite!") : value;
  }
  
  /**
   * Get closing message
   */
  function getClosingMessage(options = {}) {
    const t = window.KellyI18n?.t || ((key) => key);
    const ctx = _context || getDefaultContext();
    
    if (options.streakCount && options.streakCount > 1) {
      return t('kelly.closings.streak');
    }
    
    if (options.lessonsCompleted && options.lessonsCompleted % 10 === 0) {
      return t('kelly.closings.milestone', { count: options.lessonsCompleted });
    }
    
    if (ctx.timeOfDay === 'evening' || ctx.timeOfDay === 'night') {
      return t('kelly.closings.evening');
    }
    
    if (ctx.isWeekend) {
      return t('kelly.closings.weekend');
    }
    
    return t('kelly.closings.default');
  }
  
  // ============================================
  // ERROR MESSAGES
  // ============================================
  
  /**
   * Get friendly error message
   */
  function getErrorMessage(errorType) {
    const t = window.KellyI18n?.t || ((key) => key);
    const key = `kelly.errors.${errorType}`;
    const message = t(key);
    return message === key ? t('kelly.errors.generic') : message;
  }
  
  // ============================================
  // INITIALIZATION
  // ============================================
  
  /**
   * Initialize Kelly's personality layer
   */
  async function init(userName = null) {
    _userName = userName;
    await loadContext();
    console.log('[KellyPersonality] Initialized');
    return true;
  }
  
  /**
   * Set the user's name
   */
  function setUserName(name) {
    _userName = name;
  }
  
  /**
   * Refresh context (e.g., after timezone change)
   */
  async function refreshContext() {
    _context = null;
    return loadContext();
  }
  
  // ============================================
  // EXPOSE API
  // ============================================
  
  window.KellyPersonality = {
    init,
    loadContext,
    refreshContext,
    setUserName,
    
    // Greetings
    getGreeting,
    getShortGreeting,
    
    // Context-aware messages
    getSeasonalComment,
    getEncouragement,
    getStreakMessage,
    getReturnMessage,
    
    // Lesson commentary
    getLessonIntro,
    getPhaseTransition,
    getQuizReaction,
    getClosingMessage,
    
    // Errors
    getErrorMessage,
    
    // Direct context access
    getContext: () => _context || getDefaultContext(),
  };
  
  // Auto-initialize after i18n is ready
  window.addEventListener('i18nready', () => {
    init();
  });
  
  // Also try to init if i18n already loaded
  if (window.KellyI18n) {
    init();
  }
  
})();
