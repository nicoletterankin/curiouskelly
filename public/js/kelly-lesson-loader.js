/**
 * Kelly Lesson Loader - Unified Data Layer with Cascading Fallbacks
 * 
 * BULLETPROOF LESSON LOADING:
 *   1. Supabase (Primary) - 5s timeout
 *   2. Cloudflare D1 (Mirror) - 3s timeout  
 *   3. Static JSON (Pre-exported) - 2s timeout
 *   4. Emergency Fallback (Hardcoded) - instant
 * 
 * THE LESSON ALWAYS PLAYS.
 * 
 * Tables:
 *   - core_lessons (365 rows) - base lesson data
 *   - lesson_atoms (21,915 rows) - archetype-specific dialog
 *   - lesson_shards (38,700 rows) - age/region personalized content
 * 
 * Usage:
 *   KellyLessonLoader.init(supabaseClient);
 *   const lesson = await KellyLessonLoader.getToday({ archetype: 'The Scientist', region: 'adult' });
 */

const __kellyLoaderParams = (typeof location !== 'undefined' && location.search)
  ? new URLSearchParams(location.search)
  : new URLSearchParams('');
const __KELLY_LOADER_DEBUG =
  __kellyLoaderParams.has('debug') ||
  __kellyLoaderParams.has('audit') ||
  (typeof localStorage !== 'undefined' && localStorage.getItem('kellyDebug') === '1') ||
  (typeof window !== 'undefined' && window.KELLY_DEBUG === true);

function __kellyLoaderDebugLog(...args) {
  if (__KELLY_LOADER_DEBUG) console.log(...args);
}

function __kellyLoaderDebugWarn(...args) {
  if (__KELLY_LOADER_DEBUG) console.warn(...args);
}

const KellyLessonLoader = {
  supabase: null,
  cache: new Map(),
  preloadQueue: new Set(),
  emergencyLessonsPromise: null,
  
  // Timeout configuration (never hang forever)
  SUPABASE_TIMEOUT: 5000,  // 5 seconds
  D1_TIMEOUT: 3000,        // 3 seconds
  STATIC_TIMEOUT: 2000,    // 2 seconds
  
  // D1 Mirror endpoint
  D1_ENDPOINT: '/api/lessons',
  
  // Cloudflare D1 API endpoint (set after deployment)
  // Update this after deploying the worker
  D1_API_URL: 'https://curiouskelly-lessons.pages.dev',

  // Local API fallback (Vercel functions) — always available in this repo
  LOCAL_API_ENDPOINT: '/api/lessons',
  
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
    } else if (typeof window.getSupabase === 'function') {
      // Preferred: single shared instance (prevents multiple GoTrue clients)
      this.supabase = window.getSupabase();
      if (this.supabase) {
        __kellyLoaderDebugLog('📚 KellyLessonLoader using getSupabase() singleton');
      }
    } else if (window.supabaseClient) {
      // Try global client first
      this.supabase = window.supabaseClient;
      __kellyLoaderDebugLog('📚 KellyLessonLoader using global supabaseClient');
    } else {
      __kellyLoaderDebugWarn('⚠️ KellyLessonLoader: No Supabase client available, will use D1 mirror');
    }
    
    __kellyLoaderDebugLog('📚 KellyLessonLoader initialized');
    __kellyLoaderDebugLog(`   Supabase: ${this.supabase ? 'connected' : 'not connected'}`);
    __kellyLoaderDebugLog(`   D1 Mirror: ${this.D1_API_URL}`);
    return this;
  },
  
  /**
   * Configure D1 API URL
   */
  setD1ApiUrl(url) {
    this.D1_API_URL = url;
    __kellyLoaderDebugLog(`📡 D1 API URL set to: ${url}`);
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
   * Canonical runtime entrypoint for lesson payloads.
   * This is the ONLY place that should decide where lesson data comes from.
   *
   * Returns a simple payload shape:
   *   { lesson, atoms, shards, source }
   */
  async loadLesson(dayNumber, options = {}) {
    const {
      archetype = 'The Scientist',
      age = 30,
      region = null,
    } = options;

    const normalizedArchetype = this.normalizeArchetype(archetype);
    const targetRegion = region || this.ageToRegion(age);
    const dayNum = Math.max(1, Math.min(365, parseInt(dayNumber) || 1));

    const paddedDay = String(dayNum).padStart(3, '0');
    const packKey = `day-${paddedDay}`;

    // Priority 1: Local Pack (deterministic, offline-ready)
    try {
      const localPack = window?.CURIOUS_KELLY?.LOCAL_PACKS?.[packKey];
      if (localPack && (localPack.lesson || localPack.atoms)) {
        const rawAtoms = Array.isArray(localPack.atoms) ? localPack.atoms : [];
        const atoms = rawAtoms.filter((a) => !a?.archetype || a.archetype === normalizedArchetype);
        __kellyLoaderDebugLog(`[Loader] Using local pack for day ${dayNum}`);
        return {
          lesson: localPack.lesson || null,
          atoms,
          shards: [],
          source: 'local_pack',
        };
      }
    } catch (_) {
      // Non-fatal: fall through to normal loader logic.
    }

    // Priority 2+: Existing cascading loader logic
    const result = await this.getLesson(dayNum, { archetype: normalizedArchetype, age, region: targetRegion });
    return {
      lesson: result?.lesson || null,
      atoms: result?.atoms || [],
      shards: result?.shards || [],
      source: result?.isEmergencyFallback ? 'emergency' : 'loader',
    };
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
      useCache = true,
      // Launch hardening:
      // - Prevent recursive preloading from expanding to all 365 days.
      // - Only the "primary" (non-preload) request should schedule adjacent preloads.
      preloadAdjacent = true,
      _isPreload = false,
    } = options;
    
    const normalizedArchetype = this.normalizeArchetype(archetype);
    const targetRegion = region || this.ageToRegion(age);
    const dayNum = Math.max(1, Math.min(365, parseInt(dayNumber) || 1));
    
    const cacheKey = `${dayNum}-${normalizedArchetype}-${targetRegion}`;
    
    // Check cache
    if (useCache && this.cache.has(cacheKey)) {
      __kellyLoaderDebugLog(`📦 Cache hit: Day ${dayNum}`);
      return this.cache.get(cacheKey);
    }
    
    __kellyLoaderDebugLog(`🔍 Loading Day ${dayNum} for ${normalizedArchetype} (${targetRegion})`);
    
    if (!this.supabase) {
      __kellyLoaderDebugWarn('⚠️ Supabase not initialized, trying Cloudflare D1...');
      
      // Try Cloudflare D1 directly when Supabase is not available
      try {
        const d1Data = await this.tryCloudflareD1(dayNum, normalizedArchetype, targetRegion);
        if (d1Data?.lesson) {
          const result = this.formatD1Response(d1Data, dayNum, normalizedArchetype, targetRegion);
          this.cache.set(cacheKey, result);
          if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
          return result;
        }
      } catch (d1Error) {
        __kellyLoaderDebugWarn(`⚠️ Cloudflare D1 failed:`, d1Error.message);
      }

      // Try local Vercel API fallback (guaranteed to exist)
      try {
        const local = await this.tryLocalApi(dayNum, normalizedArchetype, targetRegion);
        if (local?.lesson) {
          const result = this.buildResult(local.lesson, local.atoms || [], local.shards || [], {
            dayNumber: dayNum,
            archetype: normalizedArchetype,
            region: targetRegion,
            _source: 'vercel-api'
          });
          this.cache.set(cacheKey, result);
          if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
          return result;
        }
      } catch (apiError) {
        __kellyLoaderDebugWarn(`⚠️ Local API fallback failed:`, apiError.message);
      }
      
      return await this.getFallback(dayNum);
    }
    
    try {
      // Wrap Supabase calls in timeout (NEVER hang forever)
      const supabasePromise = this.fetchFromSupabaseWithTimeout(dayNum, normalizedArchetype, targetRegion, options.age);
      
      // Fetch base lesson from core_lessons (with timeout)
      const supabaseResult = await Promise.race([
        supabasePromise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Supabase timeout')), this.SUPABASE_TIMEOUT)
        )
      ]);
      
      if (supabaseResult) {
        this.cache.set(cacheKey, supabaseResult);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        return supabaseResult;
      }
      
      throw new Error('Supabase returned no data');
      
    } catch (supabaseError) {
      __kellyLoaderDebugWarn(`⚠️ [L1] Supabase failed: ${supabaseError.message}`);
    }
    
    // LAYER 2: D1 Mirror
    try {
      __kellyLoaderDebugLog(`🔄 [L2] Trying Cloudflare D1 for day ${dayNum}...`);
      const d1Data = await this.tryCloudflareD1(dayNum, normalizedArchetype, targetRegion);
      if (d1Data?.lesson) {
        const result = this.formatD1Response(d1Data, dayNum, normalizedArchetype, targetRegion);
        this.cache.set(cacheKey, result);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        __kellyLoaderDebugLog(`✅ [L2] D1 success`);
        return result;
      }
    } catch (d1Error) {
      __kellyLoaderDebugWarn(`⚠️ [L2] Cloudflare D1 failed: ${d1Error.message}`);
    }

    // LAYER 2.5: Local Vercel API fallback
    try {
      __kellyLoaderDebugLog(`🔄 [L2.5] Trying local API fallback for day ${dayNum}...`);
      const local = await this.tryLocalApi(dayNum, normalizedArchetype, targetRegion);
      if (local?.lesson) {
        const result = this.buildResult(local.lesson, local.atoms || [], local.shards || [], {
          dayNumber: dayNum,
          archetype: normalizedArchetype,
          region: targetRegion,
          _source: 'vercel-api'
        });
        this.cache.set(cacheKey, result);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        __kellyLoaderDebugLog(`✅ [L2.5] Local API success`);
        return result;
      }
    } catch (apiError) {
      __kellyLoaderDebugWarn(`⚠️ [L2.5] Local API failed: ${apiError.message}`);
    }
    
    // LAYER 3: Static JSON
    try {
      __kellyLoaderDebugLog(`🔄 [L3] Trying Static JSON for day ${dayNum}...`);
      const staticResult = await this.tryStaticJSON(dayNum, normalizedArchetype, targetRegion);
      if (staticResult) {
        this.cache.set(cacheKey, staticResult);
        __kellyLoaderDebugLog(`✅ [L3] Static JSON success`);
        return staticResult;
      }
    } catch (staticError) {
      __kellyLoaderDebugWarn(`⚠️ [L3] Static JSON failed: ${staticError.message}`);
    }
    
    // LAYER 4: Emergency Fallback (NEVER FAILS)
    __kellyLoaderDebugLog(`🚨 [L4] Emergency Fallback for day ${dayNum}`);
    return await this.getFallback(dayNum);
  },
  
  /**
   * Fetch from Supabase with all data
   */
  async fetchFromSupabaseWithTimeout(dayNum, archetype, region, age) {
    // Fetch base lesson from core_lessons
    const { data: lesson, error: lessonError } = await this.supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNum)
      .single();
    
    if (lessonError || !lesson) {
      throw new Error(`Lesson ${dayNum} not found: ${lessonError?.message}`);
    }
    
    // Fetch atoms and shards in parallel for speed
    const [atomsResult, shardsResult] = await Promise.all([
      this.supabase
        .from('lesson_atoms')
        .select('*')
        .eq('core_lesson_id', lesson.id)
        .eq('archetype', archetype)
        .order('phase'),
      this.supabase
        .from('lesson_shards')
        .select('*')
        .eq('core_lesson_id', lesson.id)
        .eq('archetype', archetype)
    ]);
    
    const atoms = atomsResult.data || [];
    const shards = shardsResult.data || [];
    
    // Filter shards by region
    let matchedShards = shards;
    if (region && shards.length > 0) {
      const regionFiltered = shards.filter(s => 
        s.region === region || 
        s.region === 'en' ||
        this.shardMatchesAge(s, age)
      );
      if (regionFiltered.length > 0) {
        matchedShards = regionFiltered;
      }
    }
    
    // Build the result object
    return this.buildResult(lesson, atoms, matchedShards, {
      dayNumber: dayNum,
      archetype: archetype,
      region: region,
      _source: 'supabase'
    });
  },
  
  /**
   * Try fetching from static JSON files
   */
  async tryStaticJSON(dayNum, archetype, region) {
    const paddedDay = String(dayNum).padStart(3, '0');
    const jsonUrl = `/generated/lessons/day-${paddedDay}.json`;
    
    const response = await fetch(jsonUrl, {
      signal: AbortSignal.timeout(this.STATIC_TIMEOUT)
    });
    
    if (!response.ok) {
      throw new Error(`Static JSON returned ${response.status}`);
    }
    
    const data = await response.json();
    const ageBucket = this.regionToAgeBucket(region);
    const ageVariant = data.ageVariants?.[ageBucket] || data.ageVariants?.['18-35'] || Object.values(data.ageVariants || {})[0];
    
    if (!ageVariant) {
      throw new Error('No age variant found in static JSON');
    }
    
    // Build lesson from static format
    const lesson = {
      id: `static-${dayNum}`,
      day_number: dayNum,
      topic: data.meta?.topic || `Day ${dayNum} Lesson`,
      universal_truth: data.meta?.universalTruth || '',
      marketing_headline: data.meta?.topic || '',
      marketing_tagline: ''
    };
    
    // Build atoms from static phases
    const atoms = this.buildAtomsFromStatic(ageVariant, dayNum);
    
    return this.buildResult(lesson, atoms, [], {
      dayNumber: dayNum,
      archetype: archetype,
      region: region,
      _source: 'static'
    });
  },
  
  /**
   * Convert region to age bucket for static JSON
   */
  regionToAgeBucket(region) {
    const mapping = {
      'kid': '2-5',
      'teen': '13-17',
      'adult': '18-35',
      'mature': '36-60',
      'elder': '61-102'
    };
    return mapping[region] || '18-35';
  },
  
  /**
   * Build atoms from static JSON phases
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
   * Try fetching lesson from Cloudflare D1 (mirror)
   * @param {number} dayNumber - Day number (1-365)
   * @param {string} archetype - Archetype name (e.g., 'The Scientist')
   * @param {string} region - Age region (e.g., 'adult')
   * @returns {Object|null} - Lesson data or null if failed
   */
  async tryCloudflareD1(dayNumber, archetype, region) {
    try {
      const url = `${this.D1_API_URL}/lesson/${dayNumber}?archetype=${encodeURIComponent(archetype)}&region=${encodeURIComponent(region)}`;
      
      const response = await fetch(url, { 
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (response.ok) {
        const data = await response.json();
        __kellyLoaderDebugLog(`✅ Loaded day ${dayNumber} from Cloudflare D1`);
        return data;
      }
      
      __kellyLoaderDebugWarn(`⚠️ Cloudflare D1 returned ${response.status}`);
      return null;
    } catch (error) {
      __kellyLoaderDebugWarn('⚠️ Cloudflare D1 failed:', error.message);
      return null;
    }
  },

  /**
   * Format D1 response to match expected lesson structure
   */
  formatD1Response(d1Data, dayNumber, archetype, region) {
    const { lesson, atoms = [], shards = [] } = d1Data;
    
    // Build greeting from atoms
    const greetingAtom = atoms.find(a => 
      a.phase?.toLowerCase().includes('welcome') || 
      a.phase?.toLowerCase().includes('intro')
    );
    
    // Build script from shards or atoms
    let script = '';
    if (shards.length > 0) {
      script = shards.map(s => {
        const content = s.script_content;
        if (typeof content === 'string') return content;
        if (content?.text) return content.text;
        if (content?.script) return content.script;
        return '';
      }).filter(Boolean).join('\n\n');
    }
    
    if (!script && atoms.length > 0) {
      script = atoms.map(a => {
        const content = a.content;
        if (typeof content === 'string') return content;
        if (content?.script) return content.script;
        if (content?.text) return content.text;
        return '';
      }).filter(Boolean).join('\n\n');
    }
    
    let greeting = '';
    if (greetingAtom?.content) {
      greeting = greetingAtom.content.script || greetingAtom.content.text || '';
    }
    if (!greeting) {
      greeting = `Let's learn about ${lesson.topic || 'something new'}!`;
    }
    
    return {
      lesson,
      atoms,
      shards,
      dayNumber,
      archetype,
      region,
      _source: 'cloudflare-d1',
      
      get id() { return lesson.id || `d1-${dayNumber}`; },
      get title() { return lesson.topic || lesson.title || 'Daily Discovery'; },
      get subtitle() { return lesson.marketing_tagline || lesson.subtitle || ''; },
      get topic() { return lesson.topic; },
      get universalTruth() { return lesson.universal_truth || ''; },
      get marketingHeadline() { return lesson.marketing_headline || ''; },
      get marketingPitch() { return lesson.marketing_pitch || ''; },
      get greeting() { return greeting; },
      get script() { return script || lesson.universal_truth || ''; },
      
      get imageUrl() {
        if (lesson.hero_image_url) return lesson.hero_image_url;
        if (lesson.thumbnail_url) return lesson.thumbnail_url;
        const paddedDay = String(dayNumber).padStart(3, '0');
        return `/generated-assets/day-${paddedDay}/infographic.png`;
      },
      
      get audioUrl() { return lesson.audio_url || null; },
      get quickQuiz() { return lesson.quick_quiz_questions || []; },
      get reflectionPrompts() { return lesson.reflection_prompts || []; },
      get masteryCriteria() { return lesson.mastery_criteria || ''; },
      
      getPhase(phaseName) {
        return atoms.find(a => 
          a.phase?.toLowerCase() === phaseName.toLowerCase() ||
          a.phase?.toLowerCase().includes(phaseName.toLowerCase())
        );
      },
      
      getPhases() {
        const phaseOrder = ['welcome', 'fact1', 'fact2', 'fact3', 'wisdom'];
        return phaseOrder.map(p => this.getPhase(p)).filter(Boolean);
      }
    };
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
        __kellyLoaderDebugWarn('⚠️ Emergency lessons script failed to load');
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
    
    __kellyLoaderDebugLog(`📅 Today is Day ${dayNumber} of the year`);
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
      this.getLesson(prev, { archetype, region, useCache: true, _isPreload: true, preloadAdjacent: false })
        .then(() => this.preloadQueue.delete(prevKey))
        .catch(() => this.preloadQueue.delete(prevKey));
    }
    
    if (!this.cache.has(nextKey) && !this.preloadQueue.has(nextKey)) {
      this.preloadQueue.add(nextKey);
      this.getLesson(next, { archetype, region, useCache: true, _isPreload: true, preloadAdjacent: false })
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
        __kellyLoaderDebugWarn('Kellys table not found, using defaults');
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
      // This repo no longer ships `/data/lessons/day-XXX.json`.
      // Use the local API fallback instead, which is always present.
      const response = await fetch(`${this.LOCAL_API_ENDPOINT}/${dayNumber}?archetype=${encodeURIComponent(archetype)}&ageBucket=${encodeURIComponent(region)}`);
      
      if (response.ok) {
        const data = await response.json();
        __kellyLoaderDebugLog(`✅ Loaded day ${dayNumber} from static files`);
        
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
      __kellyLoaderDebugWarn(`⚠️ Static files failed for day ${dayNumber}:`, error.message);
    }
    return null;
  },

  /**
   * Try local Vercel API fallback (serverless function)
   * GET /api/lessons/:dayNumber?archetype=...&ageBucket=...
   */
  async tryLocalApi(dayNumber, archetype, region) {
    const url = `${this.LOCAL_API_ENDPOINT}/${dayNumber}?archetype=${encodeURIComponent(archetype)}&ageBucket=${encodeURIComponent(region)}`;
    const response = await fetch(url, { signal: AbortSignal.timeout(this.D1_TIMEOUT) });
    if (!response.ok) throw new Error(`Local API returned ${response.status}`);
    return await response.json();
  },

  /**
   * Fallback for missing lessons - tries static files first, then emergency
   */
  async getFallback(dayNumber) {
    __kellyLoaderDebugWarn('⚠️ Primary data source failed, trying fallbacks...');
    
    // Try static files first
    const staticResult = await this.tryStaticFiles(dayNumber, 'The Scientist', 'adult');
    if (staticResult) {
      return staticResult;
    }
    
    // Use emergency lessons as last resort
    __kellyLoaderDebugLog(`🆘 Using emergency fallback for day ${dayNumber}`);
    
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
    __kellyLoaderDebugLog('🗑️ Lesson cache cleared');
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

__kellyLoaderDebugLog('📚 Kelly Lesson Loader ready');

