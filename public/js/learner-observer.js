/**
 * STEALTH LEARNER OBSERVATION SYSTEM
 * 
 * Collects behavioral signals invisibly during normal lesson flow.
 * This data powers "Your Learning Journey" insights in Settings.
 * 
 * Philosophy: Assessment that feels like play, not testing.
 * 
 * @version 1.0.0
 * @created December 16, 2025
 */

class LearnerObserver {
  constructor() {
    this.sessionId = this.generateSessionId();
    this.lessonStartTime = Date.now();
    this.enabled = this.checkEnabled();
    
    // Core observation data
    this.observations = {
      // Response quality
      optionQualities: [],      // ['best', 'good', 'redirect', 'best']
      hintsUsed: 0,
      redirectsCount: 0,
      redirectRecoveries: 0,
      
      // Timing
      phaseTimings: {},         // {welcome: {start, end, duration}, q1: {...}}
      choiceTimings: [],        // [3200, 4100, 2800] ms per choice
      
      // Engagement
      audioReplays: 0,
      videoReplays: 0,
      pausesCount: 0,
      
      // Session state
      currentPhase: null,
      optionsShownTime: null,
      lastRedirect: false      // Track if previous choice was redirect
    };
    
    // Context
    this.context = {
      dayNumber: null,
      lessonId: null,
      archetype: null,
      ageSetting: null,
      language: 'en',
      deviceType: this.detectDeviceType()
    };
  }
  
  // ==========================================
  // INITIALIZATION & CONFIG
  // ==========================================
  
  generateSessionId() {
    // Use crypto.randomUUID if available, fallback to timestamp-based
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  checkEnabled() {
    // Check if user has opted out
    const preference = localStorage.getItem('kelly_observation_enabled');
    return preference !== 'false';
  }
  
  setEnabled(enabled) {
    localStorage.setItem('kelly_observation_enabled', enabled ? 'true' : 'false');
    this.enabled = enabled;
  }
  
  detectDeviceType() {
    const ua = navigator.userAgent;
    if (/tablet|ipad|playbook|silk/i.test(ua)) {
      return 'tablet';
    }
    if (/mobile|iphone|ipod|android|blackberry|opera mini|iemobile/i.test(ua)) {
      return 'mobile';
    }
    return 'desktop';
  }
  
  // ==========================================
  // LESSON LIFECYCLE
  // ==========================================
  
  /**
   * Call when a new lesson starts
   */
  startLesson(dayNumber, lessonId, archetype, ageSetting) {
    if (!this.enabled) return;
    
    this.sessionId = this.generateSessionId();
    this.lessonStartTime = Date.now();
    
    // Reset observations
    this.observations = {
      optionQualities: [],
      hintsUsed: 0,
      redirectsCount: 0,
      redirectRecoveries: 0,
      phaseTimings: {},
      choiceTimings: [],
      audioReplays: 0,
      videoReplays: 0,
      pausesCount: 0,
      currentPhase: null,
      optionsShownTime: null,
      lastRedirect: false
    };
    
    // Set context
    this.context = {
      dayNumber,
      lessonId,
      archetype,
      ageSetting,
      language: document.documentElement.lang || 'en',
      deviceType: this.detectDeviceType()
    };
    
    console.debug('[Observer] Lesson started:', this.sessionId);
  }
  
  // ==========================================
  // PHASE TRACKING
  // ==========================================
  
  /**
   * Call when a phase renders
   */
  onPhaseStart(phaseName) {
    if (!this.enabled) return;
    
    // End previous phase if exists
    if (this.observations.currentPhase) {
      this.onPhaseEnd(this.observations.currentPhase);
    }
    
    this.observations.currentPhase = phaseName;
    this.observations.phaseTimings[phaseName] = {
      startedAt: Date.now(),
      endedAt: null,
      duration: null
    };
    
    console.debug('[Observer] Phase started:', phaseName);
  }
  
  /**
   * Call when options/choices appear
   */
  onOptionsShown() {
    if (!this.enabled) return;
    
    this.observations.optionsShownTime = Date.now();
    console.debug('[Observer] Options shown');
  }
  
  /**
   * Call when a phase ends (before next phase starts)
   */
  onPhaseEnd(phaseName) {
    if (!this.enabled) return;
    
    const timing = this.observations.phaseTimings[phaseName];
    if (timing && !timing.endedAt) {
      timing.endedAt = Date.now();
      timing.duration = timing.endedAt - timing.startedAt;
    }
    
    console.debug('[Observer] Phase ended:', phaseName, timing?.duration, 'ms');
  }
  
  // ==========================================
  // CHOICE TRACKING
  // ==========================================
  
  /**
   * Call when user makes a choice
   * @param {string} quality - 'best', 'good', or 'redirect'
   */
  onChoice(quality) {
    if (!this.enabled) return;
    
    // Calculate time to choose
    let choiceTime = 0;
    if (this.observations.optionsShownTime) {
      choiceTime = Date.now() - this.observations.optionsShownTime;
      this.observations.choiceTimings.push(choiceTime);
    }
    
    // Track quality
    this.observations.optionQualities.push(quality);
    
    // Track redirects and recoveries
    if (quality === 'redirect') {
      this.observations.redirectsCount++;
      this.observations.lastRedirect = true;
    } else {
      // If previous was redirect and this is good/best, it's a recovery
      if (this.observations.lastRedirect) {
        this.observations.redirectRecoveries++;
      }
      this.observations.lastRedirect = false;
    }
    
    // Reset options timer
    this.observations.optionsShownTime = null;
    
    console.debug('[Observer] Choice:', quality, 'in', choiceTime, 'ms');
  }
  
  /**
   * Call when hint is shown (user stuck)
   */
  onHintShown() {
    if (!this.enabled) return;
    
    this.observations.hintsUsed++;
    console.debug('[Observer] Hint shown, total:', this.observations.hintsUsed);
  }
  
  // ==========================================
  // ENGAGEMENT TRACKING
  // ==========================================
  
  /**
   * Call when audio is replayed
   */
  onAudioReplay() {
    if (!this.enabled) return;
    
    this.observations.audioReplays++;
    console.debug('[Observer] Audio replay, total:', this.observations.audioReplays);
  }
  
  /**
   * Call when video is replayed
   */
  onVideoReplay() {
    if (!this.enabled) return;
    
    this.observations.videoReplays++;
    console.debug('[Observer] Video replay, total:', this.observations.videoReplays);
  }
  
  /**
   * Call when user pauses content
   */
  onPause() {
    if (!this.enabled) return;
    
    this.observations.pausesCount++;
    console.debug('[Observer] Pause, total:', this.observations.pausesCount);
  }
  
  // ==========================================
  // METRICS COMPUTATION
  // ==========================================
  
  /**
   * Calculate first-try accuracy (% of best choices)
   */
  getFirstTryAccuracy() {
    const qualities = this.observations.optionQualities;
    if (qualities.length === 0) return 0;
    
    const bestCount = qualities.filter(q => q === 'best').length;
    return bestCount / qualities.length;
  }
  
  /**
   * Calculate redirect recovery rate
   */
  getRedirectRecoveryRate() {
    if (this.observations.redirectsCount === 0) return 1;
    return this.observations.redirectRecoveries / this.observations.redirectsCount;
  }
  
  /**
   * Calculate average choice time
   */
  getAvgChoiceTime() {
    const timings = this.observations.choiceTimings;
    if (timings.length === 0) return 0;
    
    return timings.reduce((a, b) => a + b, 0) / timings.length;
  }
  
  /**
   * Count rushed choices (< 2 seconds)
   */
  getRushedChoicesCount() {
    return this.observations.choiceTimings.filter(t => t < 2000).length;
  }
  
  /**
   * Count deliberate choices (5-25 seconds)
   */
  getDeliberateChoicesCount() {
    return this.observations.choiceTimings.filter(t => t >= 5000 && t <= 25000).length;
  }
  
  /**
   * Detect if user is rushing through
   */
  isRushing() {
    const rushed = this.getRushedChoicesCount();
    const total = this.observations.choiceTimings.length;
    return total >= 3 && rushed / total > 0.5;
  }
  
  /**
   * Detect if user is exploring thoughtfully
   */
  isExploring() {
    const deliberate = this.getDeliberateChoicesCount();
    return deliberate >= 2;
  }
  
  // ==========================================
  // SUMMARY & SAVE
  // ==========================================
  
  /**
   * Get complete observation summary for saving
   */
  getSummary(completed = true, abandonedAtPhase = null) {
    // End current phase
    if (this.observations.currentPhase) {
      this.onPhaseEnd(this.observations.currentPhase);
    }
    
    const totalDuration = Date.now() - this.lessonStartTime;
    
    return {
      // Identifiers
      sessionId: this.sessionId,
      dayNumber: this.context.dayNumber,
      lessonId: this.context.lessonId,
      
      // Response quality
      firstTryCorrect: this.getFirstTryAccuracy() >= 0.5,
      optionQualitySequence: this.observations.optionQualities,
      hintsUsed: this.observations.hintsUsed,
      redirectsCount: this.observations.redirectsCount,
      redirectRecoveries: this.observations.redirectRecoveries,
      
      // Timing
      phaseDurations: this.observations.phaseTimings,
      choiceTimings: this.observations.choiceTimings,
      avgChoiceTime: Math.round(this.getAvgChoiceTime()),
      rushedChoicesCount: this.getRushedChoicesCount(),
      deliberateChoicesCount: this.getDeliberateChoicesCount(),
      
      // Engagement
      audioReplays: this.observations.audioReplays,
      videoReplays: this.observations.videoReplays,
      pausesCount: this.observations.pausesCount,
      totalSessionDuration: totalDuration,
      
      // Status
      completed,
      abandonedAtPhase,
      
      // Context
      archetype: this.context.archetype,
      ageSetting: this.context.ageSetting,
      language: this.context.language,
      deviceType: this.context.deviceType,
      
      // Computed flags
      rushingDetected: this.isRushing(),
      exploringDetected: this.isExploring()
    };
  }
  
  /**
   * Save observation to Supabase
   * @param {object} supabaseClient - Supabase client instance
   * @param {string} userId - User's UUID
   * @param {boolean} completed - Was lesson completed?
   * @param {string|null} abandonedAtPhase - Phase name if abandoned
   */
  async save(supabaseClient, userId, completed = true, abandonedAtPhase = null) {
    if (!this.enabled) {
      console.debug('[Observer] Disabled, not saving');
      return { success: false, reason: 'disabled' };
    }
    
    if (!userId) {
      console.debug('[Observer] No user ID, not saving');
      return { success: false, reason: 'no_user' };
    }
    
    const summary = this.getSummary(completed, abandonedAtPhase);
    
    const observation = {
      user_id: userId,
      lesson_id: summary.lessonId || null,
      day_number: summary.dayNumber,
      session_id: summary.sessionId,
      
      first_try_correct: summary.firstTryCorrect,
      option_quality_sequence: summary.optionQualitySequence,
      hints_used: summary.hintsUsed,
      redirects_count: summary.redirectsCount,
      redirect_recoveries: summary.redirectRecoveries,
      
      phase_durations: summary.phaseDurations,
      choice_timings: summary.choiceTimings,
      avg_choice_time: summary.avgChoiceTime,
      rushed_choices_count: summary.rushedChoicesCount,
      deliberate_choices_count: summary.deliberateChoicesCount,
      
      audio_replays: summary.audioReplays,
      video_replays: summary.videoReplays,
      pauses_count: summary.pausesCount,
      total_session_duration: summary.totalSessionDuration,
      
      completed: summary.completed,
      abandoned_at_phase: summary.abandonedAtPhase,
      
      archetype: summary.archetype,
      age_setting: summary.ageSetting,
      language: summary.language,
      device_type: summary.deviceType,
      
      completed_at: completed ? new Date().toISOString() : null
    };
    
    try {
      const { data, error } = await supabaseClient
        .from('learner_observations')
        .upsert(observation, { onConflict: 'user_id,session_id' });
      
      if (error) {
        console.error('[Observer] Save error:', error);
        return { success: false, reason: 'db_error', error };
      }
      
      console.debug('[Observer] Saved successfully');
      return { success: true, sessionId: summary.sessionId };
    } catch (err) {
      console.error('[Observer] Save exception:', err);
      return { success: false, reason: 'exception', error: err };
    }
  }
}

// ==========================================
// INSIGHT FETCHER
// ==========================================

class LearnerInsights {
  constructor(supabaseClient) {
    this.supabase = supabaseClient;
    this.cached = null;
    this.cacheTime = null;
  }
  
  /**
   * Fetch user's learning journey insights
   */
  async fetch(userId, forceRefresh = false) {
    // Return cached if recent (5 min)
    if (!forceRefresh && this.cached && this.cacheTime) {
      const age = Date.now() - this.cacheTime;
      if (age < 5 * 60 * 1000) {
        return this.cached;
      }
    }
    
    try {
      const { data, error } = await this.supabase
        .from('user_learning_journey')
        .select('*')
        .eq('user_id', userId)
        .single();
      
      if (error && error.code !== 'PGRST116') {
        // PGRST116 = not found, which is okay for new users
        console.error('[Insights] Fetch error:', error);
        return null;
      }
      
      this.cached = data;
      this.cacheTime = Date.now();
      return data;
    } catch (err) {
      console.error('[Insights] Fetch exception:', err);
      return null;
    }
  }
  
  /**
   * Export all learning data (for user privacy)
   */
  async exportData(userId) {
    try {
      const { data, error } = await this.supabase
        .rpc('export_my_learning_data', { target_user_id: userId });
      
      if (error) throw error;
      return data;
    } catch (err) {
      console.error('[Insights] Export error:', err);
      return null;
    }
  }
  
  /**
   * Delete all learning history (for user privacy)
   */
  async deleteHistory(userId) {
    try {
      const { data, error } = await this.supabase
        .rpc('delete_my_learning_history', { target_user_id: userId });
      
      if (error) throw error;
      return data;
    } catch (err) {
      console.error('[Insights] Delete error:', err);
      return null;
    }
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LearnerObserver, LearnerInsights };
}

// Also attach to window for script tag usage
if (typeof window !== 'undefined') {
  window.LearnerObserver = LearnerObserver;
  window.LearnerInsights = LearnerInsights;
}
