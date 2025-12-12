/**
 * Kelly Lesson Loader - Unified Supabase Data Layer
 * 
 * Fetches lessons from:
 *   - core_lessons (365 rows) - base lesson data
 *   - lesson_atoms (21,915 rows) - archetype-specific dialog
 *   - lesson_shards (38,700 rows) - age/region personalized content
 * 
 * Usage:
 *   KellyLessonLoader.init(supabaseClient);
 *   const lesson = await KellyLessonLoader.getToday({ archetype: 'The Scientist', region: 'adult' });
 */

const KellyLessonLoader = {
  supabase: null,
  cache: new Map(),
  preloadQueue: new Set(),
  emergencyLessonsPromise: null,
  
  // Archetype mapping (id -> database name)
  ARCHETYPES: {
    'scientist': 'The Scientist',
    'explorer': 'The Explorer',
    'rebel': 'The Rebel',
    'architect': 'The Architect',
    'diplomat': 'The Diplomat',
    'empath': 'The Empath',
    'macgyver': 'The MacGyver',
    'mystic': 'The Mystic',
    'provider': 'The Provider',
    'storyteller': 'The Storyteller',
    'strategist': 'The Strategist',
    'survivor': 'The Survivor'
  },
  
  // Age to region mapping
  AGE_REGIONS: {
    'kid': { min: 2, max: 12 },
    'teen': { min: 13, max: 19 },
    'adult': { min: 20, max: 59 },
    'mature': { min: 60, max: 79 },
    'elder': { min: 80, max: 102 }
  },
  
  /**
   * Initialize with Supabase client
   * Falls back to global window.supabaseClient or creates one from config
   */
  init(supabaseClient) {
    if (supabaseClient) {
      this.supabase = supabaseClient;
    } else if (window.supabaseClient) {
      // Try global client first
      this.supabase = window.supabaseClient;
      console.log('📚 KellyLessonLoader using global supabaseClient');
    } else if (window.supabase?.createClient && window.SUPABASE_URL && window.SUPABASE_ANON_KEY) {
      // Create from global config
      this.supabase = window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY);
      console.log('📚 KellyLessonLoader created Supabase client from config');
    } else if (window.supabase?.createClient && window.KELLY_CONFIG?.supabaseUrl && window.KELLY_CONFIG?.supabaseKey) {
      // Create from KELLY_CONFIG
      this.supabase = window.supabase.createClient(window.KELLY_CONFIG.supabaseUrl, window.KELLY_CONFIG.supabaseKey);
      console.log('📚 KellyLessonLoader created Supabase client from KELLY_CONFIG');
    } else {
      console.error('❌ KellyLessonLoader: No Supabase client available');
      return this;
    }
    
    console.log('📚 KellyLessonLoader initialized');
    return this;
  },
  
  /**
   * Convert age to region bucket
   */
  ageToRegion(age) {
    const numAge = parseInt(age) || 30;
    for (const [region, range] of Object.entries(this.AGE_REGIONS)) {
      if (numAge >= range.min && numAge <= range.max) {
        return region;
      }
    }
    return 'adult';
  },
  
  /**
   * Normalize archetype input to database format
   * Accepts: 'scientist', 'The Scientist', 'SCIENTIST'
   */
  normalizeArchetype(input) {
    if (!input) return 'The Scientist';
    
    const lower = input.toLowerCase().replace(/^the\s+/, '');
    
    // Check if it's an id
    if (this.ARCHETYPES[lower]) {
      return this.ARCHETYPES[lower];
    }
    
    // Check if it's already a full name
    const fullNames = Object.values(this.ARCHETYPES);
    const match = fullNames.find(n => n.toLowerCase() === input.toLowerCase());
    if (match) return match;
    
    // Default
    return 'The Scientist';
  },
  
  /**
   * Get complete lesson for a day number (1-365)
   * Returns: { lesson, atoms, shards, title, greeting, script, imageUrl, audioUrl }
   */
  async getLesson(dayNumber, options = {}) {
    const {
      archetype = 'The Scientist',
      age = 30,
      region = null,
      useCache = true
    } = options;
    
    const normalizedArchetype = this.normalizeArchetype(archetype);
    const targetRegion = region || this.ageToRegion(age);
    const dayNum = Math.max(1, Math.min(365, parseInt(dayNumber) || 1));
    
    const cacheKey = `${dayNum}-${normalizedArchetype}-${targetRegion}`;
    
    // Check cache
    if (useCache && this.cache.has(cacheKey)) {
      console.log(`📦 Cache hit: Day ${dayNum}`);
      return this.cache.get(cacheKey);
    }
    
    console.log(`🔍 Loading Day ${dayNum} for ${normalizedArchetype} (${targetRegion})`);
    
    if (!this.supabase) {
      console.error('❌ Supabase not initialized');
      return await this.getFallback(dayNum);
    }
    
    try {
      // Fetch base lesson from core_lessons
      const { data: lesson, error: lessonError } = await this.supabase
        .from('core_lessons')
        .select('*')
        .eq('day_number', dayNum)
        .single();
      
      if (lessonError || !lesson) {
        console.error('Lesson not found:', dayNum, lessonError);
        return await this.getFallback(dayNum);
      }
      
      // Fetch atoms (dialog) for this archetype
      const { data: atoms, error: atomsError } = await this.supabase
        .from('lesson_atoms')
        .select('*')
        .eq('core_lesson_id', lesson.id)
        .eq('archetype', normalizedArchetype)
        .order('phase');
      
      if (atomsError) {
        console.warn('Atoms fetch warning:', atomsError);
      }
      
      // Fetch shards (content variations) for this archetype + region
      const { data: shards, error: shardsError } = await this.supabase
        .from('lesson_shards')
        .select('*')
        .eq('core_lesson_id', lesson.id)
        .eq('archetype', normalizedArchetype);
      
      if (shardsError) {
        console.warn('Shards fetch warning:', shardsError);
      }
      
      // Filter shards by region (may be in 'region' column or need age match)
      let matchedShards = shards || [];
      if (targetRegion && shards) {
        const regionFiltered = shards.filter(s => 
          s.region === targetRegion || 
          s.region === 'en' || // Language fallback
          this.shardMatchesAge(s, options.age)
        );
        if (regionFiltered.length > 0) {
          matchedShards = regionFiltered;
        }
      }
      
      // Build the result object
      const result = this.buildResult(lesson, atoms || [], matchedShards, {
        dayNumber: dayNum,
        archetype: normalizedArchetype,
        region: targetRegion
      });
      
      // Cache it
      this.cache.set(cacheKey, result);
      
      // Preload adjacent lessons (fire and forget)
      this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
      
      return result;
      
    } catch (error) {
      console.error('❌ Lesson fetch error:', error);
      return await this.getFallback(dayNum);
    }
  },

  /**
   * Load emergency lessons bundle once
   */
  async ensureEmergencyLessons() {
    if (this.emergencyLessonsPromise) return this.emergencyLessonsPromise;
    
    this.emergencyLessonsPromise = new Promise((resolve) => {
      if (typeof window === 'undefined' || typeof document === 'undefined') {
        resolve({});
        return;
      }
      
      if (window.EMERGENCY_LESSONS) {
        resolve(window.EMERGENCY_LESSONS);
        return;
      }
      
      const script = document.createElement('script');
      script.src = '/data/emergency-lessons.js';
      script.async = true;
      script.onload = () => resolve(window.EMERGENCY_LESSONS || {});
      script.onerror = () => {
        console.warn('⚠️ Emergency lessons script failed to load');
        resolve({});
      };
      document.head.appendChild(script);
    });
    
    return this.emergencyLessonsPromise;
  },
  
  /**
   * Check if shard matches user age
   */
  shardMatchesAge(shard, userAge) {
    if (!shard.age || !userAge) return false;
    return Math.abs(shard.age - userAge) <= 5; // Within 5 years
  },
  
  /**
   * Build a structured result from raw data
   */
  buildResult(lesson, atoms, shards, meta) {
    const { dayNumber, archetype, region } = meta;
    
    // Extract greeting from atoms
    const greetingAtom = atoms.find(a => 
      a.phase?.toLowerCase().includes('welcome') || 
      a.phase?.toLowerCase().includes('intro')
    );
    
    // Extract script from shards or atoms
    let script = '';
    if (shards.length > 0) {
      // Combine shard content
      script = shards.map(s => {
        if (typeof s.script_content === 'string') return s.script_content;
        if (s.script_content?.text) return s.script_content.text;
        if (s.script_content?.script) return s.script_content.script;
        return '';
      }).filter(Boolean).join('\n\n');
    }
    
    // If no shard script, use atoms content
    if (!script && atoms.length > 0) {
      script = atoms.map(a => {
        if (typeof a.content === 'string') return a.content;
        if (a.content?.script) return a.content.script;
        if (a.content?.text) return a.content.text;
        return '';
      }).filter(Boolean).join('\n\n');
    }
    
    // Build greeting
    let greeting = '';
    if (greetingAtom?.content) {
      greeting = greetingAtom.content.script || greetingAtom.content.text || '';
    }
    if (!greeting) {
      greeting = `Let's learn about ${lesson.topic || 'something new'}!`;
    }
    
    return {
      // Raw data
      lesson,
      atoms,
      shards,
      
      // Metadata
      dayNumber,
      archetype,
      region,
      
      // Convenience getters
      get id() { return lesson.id; },
      get title() { return lesson.topic || lesson.title || 'Daily Discovery'; },
      get subtitle() { return lesson.marketing_tagline || lesson.marketing_headline || ''; },
      get topic() { return lesson.topic; },
      get universalTruth() { return lesson.universal_truth || ''; },
      get marketingHeadline() { return lesson.marketing_headline || ''; },
      get marketingPitch() { return lesson.marketing_pitch || ''; },
      
      get greeting() { return greeting; },
      get script() { return script || lesson.universal_truth || ''; },
      
      get imageUrl() {
        // Priority: lesson image > generated > fallback
        if (lesson.hero_image_url) return lesson.hero_image_url;
        if (lesson.thumbnail_url) return lesson.thumbnail_url;
        // Try generated assets path
        const paddedDay = String(dayNumber).padStart(3, '0');
        return `/generated-assets/day-${paddedDay}/infographic.png`;
      },
      
      get audioUrl() {
        if (lesson.audio_url) return lesson.audio_url;
        return null;
      },
      
      get quickQuiz() {
        return lesson.quick_quiz_questions || [];
      },
      
      get reflectionPrompts() {
        return lesson.reflection_prompts || [];
      },
      
      get masteryCriteria() {
        return lesson.mastery_criteria || '';
      },
      
      // Get content for a specific phase
      getPhase(phaseName) {
        return atoms.find(a => 
          a.phase?.toLowerCase() === phaseName.toLowerCase() ||
          a.phase?.toLowerCase().includes(phaseName.toLowerCase())
        );
      },
      
      // Get all phases in order
      getPhases() {
        const phaseOrder = ['welcome', 'fact1', 'fact2', 'fact3', 'wisdom'];
        return phaseOrder.map(p => this.getPhase(p)).filter(Boolean);
      }
    };
  },
  
  /**
   * Get today's lesson based on day of year
   */
  async getToday(options = {}) {
    const today = new Date();
    const startOfYear = new Date(today.getFullYear(), 0, 0);
    const diff = today - startOfYear;
    const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    // Clamp to 1-365
    const dayNumber = Math.max(1, Math.min(365, dayOfYear));
    
    console.log(`📅 Today is Day ${dayNumber} of the year`);
    return this.getLesson(dayNumber, options);
  },
  
  /**
   * Get lesson by calendar date (month/day)
   */
  async getByDate(month, day, options = {}) {
    // Convert month/day to day of year (using 2025 as reference)
    const date = new Date(2025, month - 1, day);
    const startOfYear = new Date(2025, 0, 0);
    const diff = date - startOfYear;
    const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    return this.getLesson(dayOfYear, options);
  },
  
  /**
   * Preload next/previous lessons for smooth navigation
   */
  async preloadAdjacent(dayNumber, archetype, region) {
    const prev = Math.max(1, dayNumber - 1);
    const next = Math.min(365, dayNumber + 1);
    
    // Avoid duplicate preloads
    const prevKey = `${prev}-${archetype}-${region}`;
    const nextKey = `${next}-${archetype}-${region}`;
    
    if (!this.cache.has(prevKey) && !this.preloadQueue.has(prevKey)) {
      this.preloadQueue.add(prevKey);
      // Fire and forget - don't await
      this.getLesson(prev, { archetype, region, useCache: true })
        .then(() => this.preloadQueue.delete(prevKey))
        .catch(() => this.preloadQueue.delete(prevKey));
    }
    
    if (!this.cache.has(nextKey) && !this.preloadQueue.has(nextKey)) {
      this.preloadQueue.add(nextKey);
      this.getLesson(next, { archetype, region, useCache: true })
        .then(() => this.preloadQueue.delete(nextKey))
        .catch(() => this.preloadQueue.delete(nextKey));
    }
  },
  
  /**
   * Get list of all Kelly archetypes
   */
  async getKellys() {
    // Try to fetch from database first
    if (this.supabase) {
      try {
        const { data } = await this.supabase
          .from('kellys')
          .select('*')
          .order('display_order');
        
        if (data && data.length > 0) {
          return data;
        }
      } catch (e) {
        console.warn('Kellys table not found, using defaults');
      }
    }
    
    // Fallback to hardcoded list
    return Object.entries(this.ARCHETYPES).map(([id, name], index) => ({
      id,
      name,
      tagline: this.getDefaultTagline(id),
      display_order: index + 1,
      color: this.getDefaultColor(id)
    }));
  },
  
  /**
   * Default taglines for archetypes
   */
  getDefaultTagline(id) {
    const taglines = {
      'scientist': 'Data-driven precision',
      'explorer': 'Wonder and discovery',
      'rebel': 'Bold challenging spirit',
      'architect': 'Methodical structure',
      'diplomat': 'Inclusive harmony',
      'empath': 'Nurturing warmth',
      'macgyver': 'Resourceful ingenuity',
      'mystic': 'Spiritual wisdom',
      'provider': 'Caring support',
      'storyteller': 'Narrative magic',
      'strategist': 'Tactical thinking',
      'survivor': 'Resilient strength'
    };
    return taglines[id] || 'Unique perspective';
  },
  
  /**
   * Default colors for archetypes
   */
  getDefaultColor(id) {
    const colors = {
      'scientist': '#3b82f6',
      'explorer': '#eab308',
      'rebel': '#ef4444',
      'architect': '#6b7280',
      'diplomat': '#22c55e',
      'empath': '#ec4899',
      'macgyver': '#f97316',
      'mystic': '#8b5cf6',
      'provider': '#14b8a6',
      'storyteller': '#f59e0b',
      'strategist': '#6366f1',
      'survivor': '#84cc16'
    };
    return colors[id] || '#3b82f6';
  },
  
  /**
   * Try loading from static JSON files (pre-exported)
   */
  async tryStaticFiles(dayNumber, archetype, region) {
    try {
      const paddedDay = String(dayNumber).padStart(3, '0');
      const response = await fetch(`/data/lessons/day-${paddedDay}.json`);
      
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ Loaded day ${dayNumber} from static files`);
        
        const lesson = data.lesson || data;
        const atoms = data.atoms || [];
        const shards = data.shards || [];
        
        // Filter atoms/shards for archetype if available
        const filteredAtoms = atoms.filter(a => 
          !a.archetype || a.archetype === archetype
        );
        const filteredShards = shards.filter(s => 
          (!s.archetype || s.archetype === archetype) &&
          (!s.region || s.region === region || s.region === 'en')
        );
        
        return this.buildResult(lesson, filteredAtoms, filteredShards, {
          dayNumber,
          archetype,
          region,
          source: 'static'
        });
      }
    } catch (error) {
      console.warn(`⚠️ Static files failed for day ${dayNumber}:`, error.message);
    }
    return null;
  },

  /**
   * Fallback for missing lessons - tries static files first, then emergency
   */
  async getFallback(dayNumber) {
    console.warn('⚠️ Primary data source failed, trying fallbacks...');
    
    // Try static files first
    const staticResult = await this.tryStaticFiles(dayNumber, 'The Scientist', 'adult');
    if (staticResult) {
      return staticResult;
    }
    
    // Use emergency lessons as last resort
    console.log(`🆘 Using emergency fallback for day ${dayNumber}`);
    
    // Try global getEmergencyLesson if available
    if (typeof window !== 'undefined' && typeof window.getEmergencyLesson === 'function') {
      const emergency = window.getEmergencyLesson(dayNumber);
      return this.formatEmergencyLesson(emergency, dayNumber);
    }
    
    // Otherwise load emergency lessons
    const emergencyLessons = await this.ensureEmergencyLessons();
    const entry = (emergencyLessons && emergencyLessons[dayNumber]) || {};
    
    return this.formatEmergencyLesson(entry, dayNumber);
  },

  /**
   * Format emergency lesson data into standard result format
   */
  formatEmergencyLesson(entry, dayNumber) {
    const lesson = {
      id: entry.id || `fallback-${dayNumber}`,
      day_number: entry.day_number || dayNumber,
      topic: entry.topic || entry.title || 'Daily Discovery',
      universal_truth: entry.universal_truth || entry.content || 'Every day brings something new to learn.',
      marketing_headline: entry.marketing_headline || entry.marketing_hook || entry.title || 'Discover something amazing today',
      marketing_tagline: entry.marketing_tagline || entry.subtitle || 'Learning never stops',
      hero_image_url: entry.hero_image_url || entry.imageUrl || entry.thumbnail_url,
      thumbnail_url: entry.thumbnail_url
    };
    
    const fallbackGreeting = 'Every day brings something new to learn!';
    const script = entry.content || entry.script || entry.universal_truth || lesson.universal_truth;
    const greeting = entry.greeting || fallbackGreeting;
    const imageUrl = lesson.hero_image_url || '/images/fallback-lesson.png';
    const atoms = entry.atoms || [];
    const shards = entry.shards || [];
    const source = entry.source || 'emergency';

    return {
      lesson,
      atoms,
      shards,
      dayNumber,
      archetype: 'The Scientist',
      region: 'adult',
      source,
      
      get id() { return lesson.id; },
      get title() { return lesson.topic; },
      get subtitle() { return lesson.marketing_tagline; },
      get topic() { return lesson.topic; },
      get universalTruth() { return lesson.universal_truth; },
      get greeting() { return greeting; },
      get script() { return script; },
      get imageUrl() { return imageUrl; },
      get audioUrl() { return null; },
      get quickQuiz() { return []; },
      get reflectionPrompts() { return []; },
      get masteryCriteria() { return ''; },
      getPhase() { return null; },
      getPhases() { return []},
      
      // Add helper to check source
      get isEmergencyFallback() { return source === 'emergency' || source === 'generic-fallback'; }
    };
  },
  
  /**
   * Clear all cached data
   */
  clearCache() {
    this.cache.clear();
    this.preloadQueue.clear();
    console.log('🗑️ Lesson cache cleared');
  },
  
  /**
   * Get cache stats
   */
  getCacheStats() {
    return {
      cachedLessons: this.cache.size,
      pendingPreloads: this.preloadQueue.size
    };
  }
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = KellyLessonLoader;
}

// Make available globally
window.KellyLessonLoader = KellyLessonLoader;

console.log('📚 Kelly Lesson Loader ready');
