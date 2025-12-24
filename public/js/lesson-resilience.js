/**
 * Lesson Resilience Layer - BULLETPROOF LESSON LOADING
 * 
 * API-First Architecture (Cascading Fallback System):
 *   1. Standard API (/api/lessons/[dayNumber]) - Primary data source
 *   2. Cloudflare D1 (Mirror) - 3s timeout
 *   3. Static JSON (Pre-exported) - 2s timeout
 *   4. Emergency Fallback (Hardcoded) - instant
 * 
 * THE LESSON ALWAYS PLAYS.
 * 
 * NOTE: Direct Supabase access removed - all lesson data goes through API layer.
 */

// Debug mode - only log when ?debug=true or localStorage.kellyDebug=1
const __RESILIENCE_DEBUG = (
  (typeof location !== 'undefined' && location.search.includes('debug')) ||
  (typeof localStorage !== 'undefined' && localStorage.getItem('kellyDebug') === '1')
);

const LessonResilience = {
  // Configuration
  API_TIMEOUT: 5000,        // 5 seconds (Standard API)
  D1_TIMEOUT: 3000,         // 3 seconds
  STATIC_TIMEOUT: 2000,     // 2 seconds
  
  // API endpoints
  API_ENDPOINT: '/api/lessons',  // Standard Vercel API
  D1_ENDPOINT: '/api/lessons',   // Cloudflare D1 Mirror (or local API fallback)
  
  // Metrics
  metrics: {
    apiHits: 0,
    apiTimeouts: 0,
    d1Hits: 0,
    d1Timeouts: 0,
    staticHits: 0,
    staticTimeouts: 0,
    emergencyHits: 0,
    lastSource: null
  },
  
  /**
   * Fetch with timeout - never hangs forever
   */
  async fetchWithTimeout(promise, timeoutMs, label) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
      const result = await Promise.race([
        promise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error(`${label} timeout after ${timeoutMs}ms`)), timeoutMs)
        )
      ]);
      clearTimeout(timeoutId);
      return result;
    } catch (error) {
      clearTimeout(timeoutId);
      if (__RESILIENCE_DEBUG) console.warn(`⏱️ ${label} failed:`, error.message);
      throw error;
    }
  },
  
  /**
   * Layer 1: Standard API (Primary)
   * ARCHITECTURE FIX: Replaced direct Supabase access with API endpoint
   */
  async fromAPI(dayNumber, options = {}) {
    const { archetype = 'The Scientist', ageBucket = 'adult', track = 'learn' } = options;
    
    if (__RESILIENCE_DEBUG) console.log(`🔍 [L1] Standard API: Day ${dayNumber}, track=${track}`);
    
    const fetchPromise = (async () => {
      const response = await fetch(`${this.API_ENDPOINT}/${dayNumber}?archetype=${encodeURIComponent(archetype)}&track=${encodeURIComponent(track)}&ageBucket=${encodeURIComponent(ageBucket)}`);
      
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data?.lesson) {
        throw new Error(`Lesson ${dayNumber} not found in API response`);
      }
      
      return {
        source: data.source || 'api',
        lesson: data.lesson,
        atoms: data.atoms || [],
        shards: data.shards || [],
        dayNumber,
        archetype,
        ageBucket
      };
    })();
    
    const result = await this.fetchWithTimeout(fetchPromise, this.API_TIMEOUT, 'Standard API');
    this.metrics.apiHits++;
    this.metrics.lastSource = 'api';
    return result;
  },
  
  /**
   * Layer 2: Cloudflare D1 (Mirror)
   */
  async fromD1(dayNumber, options = {}) {
    const { archetype = 'The Scientist', ageBucket = 'adult' } = options;
    
    if (__RESILIENCE_DEBUG) console.log(`🔍 [L2] D1 Mirror: Day ${dayNumber}`);
    
    const fetchPromise = (async () => {
      const response = await fetch(`${this.D1_ENDPOINT}/${dayNumber}?archetype=${encodeURIComponent(archetype)}&ageBucket=${encodeURIComponent(ageBucket)}`);
      
      if (!response.ok) {
        throw new Error(`D1 returned ${response.status}`);
      }
      
      const data = await response.json();
      
      if (!data.lesson) {
        throw new Error('D1 returned empty lesson');
      }
      
      return {
        source: 'd1',
        ...data
      };
    })();
    
    const result = await this.fetchWithTimeout(fetchPromise, this.D1_TIMEOUT, 'D1 Mirror');
    this.metrics.d1Hits++;
    this.metrics.lastSource = 'd1';
    return result;
  },
  
  /**
   * Layer 3: Static JSON (Pre-exported)
   */
  async fromStaticJSON(dayNumber, options = {}) {
    const { ageBucket = '18-35' } = options;
    
    if (__RESILIENCE_DEBUG) console.log(`🔍 [L3] Static JSON: Day ${dayNumber}`);
    
    const paddedDay = String(dayNumber).padStart(3, '0');
    const jsonUrl = `/generated/lessons/day-${paddedDay}.json`;
    
    const fetchPromise = (async () => {
      const response = await fetch(jsonUrl);
      
      if (!response.ok) {
        throw new Error(`Static JSON ${jsonUrl} returned ${response.status}`);
      }
      
      const data = await response.json();
      
      // Map static JSON format to expected format
      const ageVariant = data.ageVariants?.[ageBucket] || data.ageVariants?.['18-35'] || Object.values(data.ageVariants || {})[0];
      
      return {
        source: 'static',
        lesson: {
          id: `static-${dayNumber}`,
          day_number: dayNumber,
          topic: data.meta?.topic || data.topic || 'Loading...',
          universal_truth: data.meta?.universalTruth || '',
          marketing_headline: data.meta?.topic || data.topic || '',
          marketing_tagline: ''
        },
        atoms: this.buildAtomsFromStatic(ageVariant, dayNumber),
        shards: [],
        dayNumber,
        ageVariant
      };
    })();
    
    const result = await this.fetchWithTimeout(fetchPromise, this.STATIC_TIMEOUT, 'Static JSON');
    this.metrics.staticHits++;
    this.metrics.lastSource = 'static';
    return result;
  },
  
  /**
   * Convert static JSON phases to atoms format
   */
  buildAtomsFromStatic(ageVariant, dayNumber) {
    if (!ageVariant?.phases) return [];
    
    const phaseMap = {
      hook: 'Hook',
      q1: 'Fact1',
      q2: 'Fact2',
      q3: 'Fact3',
      wisdom: 'Wisdom'
    };
    
    return Object.entries(ageVariant.phases).map(([key, content]) => ({
      phase: phaseMap[key] || key,
      content: {
        script: typeof content === 'string' ? content : (content?.en || content?.script || ''),
        text: typeof content === 'string' ? content : (content?.en || content?.text || '')
      },
      dayNumber
    }));
  },
  
  /**
   * Layer 4: Emergency Fallback (Hardcoded)
   */
  fromEmergency(dayNumber) {
    if (__RESILIENCE_DEBUG) console.log(`🔍 [L4] Emergency Fallback: Day ${dayNumber}`);
    
    // Use window.SUPPORT_LESSONS if available (loaded from support-lessons.js)
    const emergency = window.EMERGENCY_LESSONS?.[dayNumber] || this.HARDCODED_LESSONS[dayNumber % 7 + 1] || this.HARDCODED_LESSONS[1];
    
    this.metrics.emergencyHits++;
    this.metrics.lastSource = 'emergency';
    
    return {
      source: 'emergency',
      lesson: {
        id: emergency.id || `emergency-${dayNumber}`,
        day_number: dayNumber,
        topic: emergency.topic || 'Daily Discovery',
        universal_truth: emergency.universal_truth || emergency.script || 'Every day brings something new to learn.',
        marketing_headline: emergency.marketing_headline || emergency.topic || 'Discover something amazing',
        marketing_tagline: emergency.marketing_tagline || ''
      },
      atoms: [
        { phase: 'Hook', content: { script: emergency.greeting || emergency.script || 'Welcome to today\'s lesson!' } },
        { phase: 'Fact1', content: { script: emergency.script || 'Let\'s explore together.' } },
        { phase: 'Fact2', content: { script: 'Here\'s something interesting...' } },
        { phase: 'Fact3', content: { script: 'And one more thing...' } },
        { phase: 'Wisdom', content: { script: emergency.universal_truth || 'Knowledge is power.' } }
      ],
      shards: [],
      dayNumber,
      greeting: emergency.greeting || 'Let\'s learn something new!',
      script: emergency.script || ''
    };
  },
  
  // Extended hardcoded lessons (7 days cycling)
  HARDCODED_LESSONS: {
    1: {
      id: 'hardcoded-1',
      topic: 'Start Strong',
      greeting: 'Welcome back! Let\'s grab an easy win together.',
      script: 'Today is about momentum: pick one tiny action and finish it. Progress beats perfection.',
      universal_truth: 'Small, consistent steps compound faster than perfect starts.'
    },
    2: {
      id: 'hardcoded-2',
      topic: 'Stay Curious',
      greeting: 'What surprised you yesterday? Let\'s explore it.',
      script: 'List two questions about something you use daily. Explore one answer—share what changed.',
      universal_truth: 'Curiosity turns obstacles into experiments.'
    },
    3: {
      id: 'hardcoded-3',
      topic: 'One Percent Better',
      greeting: 'Choose a habit and give it a 1% lift.',
      script: 'Pick a habit. Make it 1% easier: shorten, stage, or simplify. Log the tweak.',
      universal_truth: 'A 1% daily gain doubles you in weeks.'
    },
    4: {
      id: 'hardcoded-4',
      topic: 'Reduce Friction',
      greeting: 'What\'s slowing you down? Let\'s remove it.',
      script: 'Identify one friction point. Remove or automate it. Notice how energy frees up.',
      universal_truth: 'Removing drag beats adding willpower.'
    },
    5: {
      id: 'hardcoded-5',
      topic: 'Share the Learning',
      greeting: 'Explain a concept like you would to a friend.',
      script: 'Pick a concept you learned this week. Explain it in 3 sentences. Refine until clear.',
      universal_truth: 'Teaching cements understanding.'
    },
    6: {
      id: 'hardcoded-6',
      topic: 'Recover Quickly',
      greeting: 'Missed a day? Reset now—no guilt.',
      script: 'Name one slip. Design a 5-minute reset ritual (water, stretch, inbox zero). Do it now.',
      universal_truth: 'The recovery is more important than the stumble.'
    },
    7: {
      id: 'hardcoded-7',
      topic: 'Celebrate Micro-Wins',
      greeting: 'What tiny win can you celebrate today?',
      script: 'Write down one win from this week. Reward it: share it, log it, or take a deep breath and smile.',
      universal_truth: 'Celebrated wins become repeatable habits.'
    }
  },
  
  /**
   * MAIN ENTRY POINT: Get lesson with cascading fallback
   * THE LESSON ALWAYS PLAYS.
   */
  async getLesson(dayNumber, options = {}) {
    const startTime = Date.now();
    let lastError = null;
    
    // Layer 1: Standard API (Primary)
    try {
      const result = await this.fromAPI(dayNumber, options);
      if (__RESILIENCE_DEBUG) console.log(`✅ [L1] Standard API success in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      this.metrics.apiTimeouts++;
      lastError = error;
      if (__RESILIENCE_DEBUG) console.warn(`⚠️ [L1] Standard API failed, trying D1...`);
    }
    
    // Layer 2: D1 Mirror
    try {
      const result = await this.fromD1(dayNumber, options);
      if (__RESILIENCE_DEBUG) console.log(`✅ [L2] D1 success in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      this.metrics.d1Timeouts++;
      lastError = error;
      if (__RESILIENCE_DEBUG) console.warn(`⚠️ [L2] D1 failed, trying Static JSON...`);
    }
    
    // Layer 3: Static JSON
    try {
      const result = await this.fromStaticJSON(dayNumber, options);
      if (__RESILIENCE_DEBUG) console.log(`✅ [L3] Static JSON success in ${Date.now() - startTime}ms`);
      return result;
    } catch (error) {
      this.metrics.staticTimeouts++;
      lastError = error;
      if (__RESILIENCE_DEBUG) console.warn(`⚠️ [L3] Static JSON failed, using Emergency Fallback...`);
    }
    
    // Layer 4: Emergency Fallback (NEVER FAILS)
    const result = this.fromEmergency(dayNumber);
    if (__RESILIENCE_DEBUG) console.log(`✅ [L4] Emergency Fallback in ${Date.now() - startTime}ms`);
    return result;
  },
  
  /**
   * Get metrics for debugging
   */
  getMetrics() {
    return {
      ...this.metrics,
      successRate: {
        api: this.metrics.apiHits / (this.metrics.apiHits + this.metrics.apiTimeouts) || 0,
        d1: this.metrics.d1Hits / (this.metrics.d1Hits + this.metrics.d1Timeouts) || 0,
        static: this.metrics.staticHits / (this.metrics.staticHits + this.metrics.staticTimeouts) || 0
      }
    };
  },
  
  /**
   * Reset metrics
   */
  resetMetrics() {
    this.metrics = {
      apiHits: 0,
      apiTimeouts: 0,
      d1Hits: 0,
      d1Timeouts: 0,
      staticHits: 0,
      staticTimeouts: 0,
      emergencyHits: 0,
      lastSource: null
    };
  }
};

// Make available globally
window.LessonResilience = LessonResilience;

if (__RESILIENCE_DEBUG) console.log('🛡️ Lesson Resilience Layer ready - THE LESSON ALWAYS PLAYS');





