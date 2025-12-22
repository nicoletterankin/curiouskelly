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

  // Canonical "seed" lessons shipped with the app (365 days).
  // These are the MVP backbone: 7 phases × 2 choices.
  SEED_LESSONS_BASE_URL: '/lessons',

  // On-demand lesson generation (client-side, deterministic).
  ON_DEMAND_TOPIC_STORAGE_KEY: 'kellyOnDemandTopic',
  ON_DEMAND_TOPIC_MAX_LEN: 80,
  ON_DEMAND_TOPIC_MIN_LEN: 3,

  // MVP completeness target for the lesson player.
  MVP_PHASE_ORDER: ['Hook', 'Cliff', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'Outro'],
  MVP_PHASE_KEY_MAP: {
    hook: 'Hook',
    cliff: 'Cliff',
    q1: 'Fact1',
    q2: 'Fact2',
    q3: 'Fact3',
    wisdom: 'Wisdom',
    outro: 'Outro',
  },
  MVP_DEFAULT_VISUAL_URL: '/images/kelly-hero-4k.png',
  
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
      track = 'learn', // 'learn' (traditional) or 'grow' (AI fluency)
    } = options;

    const normalizedArchetype = this.normalizeArchetype(archetype);
    const targetRegion = region || this.ageToRegion(age);
    const normalizedTrack = (track === 'grow') ? 'grow' : 'learn';
    const dayNum = Math.max(1, Math.min(365, parseInt(dayNumber) || 1));

    const paddedDay = String(dayNum).padStart(3, '0');
    const packKey = `day-${paddedDay}`;

    // On-demand overrides everything (Grow track / URL param).
    try {
      const onDemand = this.getOnDemandTopic();
      if (onDemand) {
        const generated = this.buildOnDemandLesson(dayNum, normalizedArchetype, targetRegion, onDemand);
        const processed = this.ensureMvpLessonShape(generated, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        return {
          lesson: processed?.lesson || null,
          atoms: processed?.atoms || [],
          shards: processed?.shards || [],
          source: 'on-demand',
        };
      }
    } catch (_) {}

    // Priority 1: Local Pack (deterministic, offline-ready)
    // NOTE: Local packs only exist for Learn track. Skip for Grow.
    if (normalizedTrack === 'learn') {
      try {
        // Check both string key ("day-351") and numeric key (351)
        const localPacks = window?.CURIOUS_KELLY?.LOCAL_PACKS;
        const localPack = localPacks?.[packKey] || localPacks?.[dayNum] || localPacks?.[String(dayNum)];
        if (localPack && (localPack.lesson || localPack.atoms)) {
          const rawAtoms = Array.isArray(localPack.atoms) ? localPack.atoms : [];
          const atoms = rawAtoms.filter((a) => !a?.archetype || a.archetype === normalizedArchetype);

          // OFFLINE-FIRST: Always use LOCAL_PACKS content - skeletons have real lessons!
          // This ensures lessons work without internet. Network enrichment is optional.
          __kellyLoaderDebugLog(`[Loader] Using local pack for day ${dayNum} (offline-first)`);
          const tmp = { lesson: localPack.lesson || null, atoms, shards: [] };
          const processed = this.ensureMvpLessonShape(tmp, { dayNum, archetype: normalizedArchetype, region: targetRegion });
          return {
            lesson: processed?.lesson || localPack.lesson || null,
            atoms: processed?.atoms || atoms,
            shards: [],
            source: 'local_pack',
          };
        }
      } catch (_) {
        // Non-fatal: fall through to normal loader logic.
      }
    } else {
      __kellyLoaderDebugLog(`[Loader] Skipping local pack for Grow track - using Supabase`);
    }

    // Priority 2+: Existing cascading loader logic (passes track through)
    const result = await this.getLesson(dayNum, { archetype: normalizedArchetype, age, region: targetRegion, track: normalizedTrack });
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
      track = 'learn', // 'learn' (traditional) or 'grow' (AI fluency)
      useCache = true,
      // Launch hardening:
      // - Prevent recursive preloading from expanding to all 365 days.
      // - Only the "primary" (non-preload) request should schedule adjacent preloads.
      preloadAdjacent = true,
      _isPreload = false,
    } = options;
    
    const normalizedArchetype = this.normalizeArchetype(archetype);
    const targetRegion = region || this.ageToRegion(age);
    const normalizedTrack = (track === 'grow') ? 'grow' : 'learn';
    const dayNum = Math.max(1, Math.min(365, parseInt(dayNumber) || 1));
    
    const cacheKey = `${normalizedTrack}-${dayNum}-${normalizedArchetype}-${targetRegion}`;
    
    // Check cache
    if (useCache && this.cache.has(cacheKey)) {
      __kellyLoaderDebugLog(`📦 Cache hit: Day ${dayNum}`);
      return this.cache.get(cacheKey);
    }
    
    __kellyLoaderDebugLog(`🔍 Loading Day ${dayNum} for ${normalizedArchetype} (${targetRegion})`);

    // ============================================================
    // ON-DEMAND (INSTANT) LESSONS
    // - Used for "Create a new lesson: <topic>" from search panel.
    // - By default, this runs in the Grow track; Learn stays day-based.
    // ============================================================
    try {
      const onDemand = this.getOnDemandTopic();
      if (onDemand) {
        const generated = this.buildOnDemandLesson(dayNum, normalizedArchetype, targetRegion, onDemand);
        const processed = this.ensureMvpLessonShape(generated, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        this.cache.set(cacheKey, processed);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        return processed;
      }
    } catch (e) {
      __kellyLoaderDebugWarn('⚠️ On-demand lesson generation failed, falling back to day lesson:', e?.message || e);
    }

    // ============================================================
    // SEED (LOCAL) LESSONS — 365-day MVP backbone
    // This is intentionally *before* Supabase/D1 so the app is
    // resilient and fast even when networks are flaky.
    //
    // To opt out: set window.KELLY_CONFIG.preferSeedLessons = false
    // NOTE: Seed lessons only exist for Learn track. Grow track must use Supabase.
    // ============================================================
    const preferSeedLessons =
      (typeof window !== 'undefined' &&
        window.KELLY_CONFIG &&
        window.KELLY_CONFIG.preferSeedLessons === false)
        ? false
        : true;
    // Skip seed lessons for Grow track - they only exist for Learn
    if (preferSeedLessons && normalizedTrack === 'learn') {
      try {
        const seedResult = await this.trySeedLessons(dayNum, normalizedArchetype, targetRegion);
        if (seedResult) {
          const processed = this.ensureMvpLessonShape(seedResult, { dayNum, archetype: normalizedArchetype, region: targetRegion });
          this.cache.set(cacheKey, processed);
          if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
          return processed;
        }
      } catch (seedErr) {
        __kellyLoaderDebugWarn(`⚠️ Seed lessons failed for day ${dayNum}:`, seedErr?.message || seedErr);
      }
    }
    
    if (!this.supabase) {
      __kellyLoaderDebugWarn('⚠️ Supabase not initialized, trying Cloudflare D1...');
      
      // Try Cloudflare D1 directly when Supabase is not available
      try {
        const d1Data = await this.tryCloudflareD1(dayNum, normalizedArchetype, targetRegion);
        if (d1Data?.lesson) {
          const result = this.formatD1Response(d1Data, dayNum, normalizedArchetype, targetRegion);
          const processed = this.ensureMvpLessonShape(result, { dayNum, archetype: normalizedArchetype, region: targetRegion });
          this.cache.set(cacheKey, processed);
          if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
          return processed;
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
          const processed = this.ensureMvpLessonShape(result, { dayNum, archetype: normalizedArchetype, region: targetRegion });
          this.cache.set(cacheKey, processed);
          if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
          return processed;
        }
      } catch (apiError) {
        __kellyLoaderDebugWarn(`⚠️ Local API fallback failed:`, apiError.message);
      }
      
      return await this.getFallback(dayNum);
    }
    
    try {
      // Wrap Supabase calls in timeout (NEVER hang forever)
      const supabasePromise = this.fetchFromSupabaseWithTimeout(dayNum, normalizedArchetype, targetRegion, options.age, normalizedTrack);
      
      // Fetch base lesson from core_lessons (with timeout)
      const supabaseResult = await Promise.race([
        supabasePromise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Supabase timeout')), this.SUPABASE_TIMEOUT)
        )
      ]);
      
      if (supabaseResult) {
        const processed = this.ensureMvpLessonShape(supabaseResult, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        this.cache.set(cacheKey, processed);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        return processed;
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
        const processed = this.ensureMvpLessonShape(result, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        this.cache.set(cacheKey, processed);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        __kellyLoaderDebugLog(`✅ [L2] D1 success`);
        return processed;
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
        const processed = this.ensureMvpLessonShape(result, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        this.cache.set(cacheKey, processed);
        if (preloadAdjacent && !_isPreload) this.preloadAdjacent(dayNum, normalizedArchetype, targetRegion);
        __kellyLoaderDebugLog(`✅ [L2.5] Local API success`);
        return processed;
      }
    } catch (apiError) {
      __kellyLoaderDebugWarn(`⚠️ [L2.5] Local API failed: ${apiError.message}`);
    }
    
    // LAYER 3: Static JSON
    try {
      __kellyLoaderDebugLog(`🔄 [L3] Trying Static JSON for day ${dayNum}...`);
      const staticResult = await this.tryStaticJSON(dayNum, normalizedArchetype, targetRegion);
      if (staticResult) {
        const processed = this.ensureMvpLessonShape(staticResult, { dayNum, archetype: normalizedArchetype, region: targetRegion });
        this.cache.set(cacheKey, processed);
        __kellyLoaderDebugLog(`✅ [L3] Static JSON success`);
        return processed;
      }
    } catch (staticError) {
      __kellyLoaderDebugWarn(`⚠️ [L3] Static JSON failed: ${staticError.message}`);
    }

    // LAYER 3.5: Seed lessons bundled with the app (dash/underscore variants)
    if (normalizedTrack === 'learn') {
      try {
        __kellyLoaderDebugLog(`🔄 [L3.5] Trying bundled seed lesson for day ${dayNum}...`);
        const seedResult = await this.trySeedLessons(dayNum, normalizedArchetype, targetRegion);
        if (seedResult) {
          const processed = this.ensureMvpLessonShape(seedResult, { dayNum, archetype: normalizedArchetype, region: targetRegion });
          this.cache.set(cacheKey, processed);
          __kellyLoaderDebugLog(`✅ [L3.5] Seed lesson success`);
          return processed;
        }
      } catch (seedErr) {
        __kellyLoaderDebugWarn(`⚠️ [L3.5] Seed lesson failed: ${seedErr?.message || seedErr}`);
      }
    }
    
    // LAYER 4: Emergency Fallback (NEVER FAILS)
    __kellyLoaderDebugLog(`🚨 [L4] Emergency Fallback for day ${dayNum}`);
    return await this.getFallback(dayNum);
  },
  
  /**
   * Fetch from Supabase with all data
   * @param {number} dayNum - Day number (1-365)
   * @param {string} archetype - Kelly persona
   * @param {string} region - Age region
   * @param {number} age - User age
   * @param {string} track - 'learn' or 'grow'
   */
  async fetchFromSupabaseWithTimeout(dayNum, archetype, region, age, track = 'learn') {
    // Fetch base lesson from core_lessons (with track filter)
    const { data: lesson, error: lessonError } = await this.supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNum)
      .eq('track', track)
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
    // Prefer canonical /public/lessons seed packs (365 days).
    const seed = await this.trySeedLessons(dayNum, archetype, region);
    if (seed) return seed;

    // Best-effort fallback: legacy generated/lessons format (may not exist).
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
      topic: data.meta?.topic || data.topic || 'Loading...',
      universal_truth: data.meta?.universalTruth || '',
      marketing_headline: data.meta?.topic || data.topic || '',
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
   * Try bundled seed lessons directly from /public/lessons
   */
  async trySeedLesson(dayNumber) {
    const base = this.SEED_LESSONS_BASE_URL || '/lessons';
    const paths = [
      `${base}/day-${dayNumber}.json`,
      `${base}/day_${dayNumber}.json`,
      `${base}/${dayNumber}.json`,
    ];
    
    for (const path of paths) {
      try {
        const resp = await fetch(path);
        if (resp.ok) {
          const data = await resp.json();
          __kellyLoaderDebugLog(`[Seed] Loaded from ${path}`);
          return data;
        }
      } catch (_) {
        __kellyLoaderDebugLog(`[Seed] ${path} failed`);
      }
    }
    return null;
  },

  /**
   * Load canonical seed lesson JSON shipped with the app:
   *   GET /lessons/day-<N>.json
   *
   * This is the reliable 365-day MVP source of truth.
   */
  async trySeedLessons(dayNumber, archetype, region) {
    try {
      const seed = await this.trySeedLesson(dayNumber);
      if (!seed) return null;
      const lesson = this.seedToLesson(seed, dayNumber);
      const atoms = this.seedToAtoms(seed, dayNumber, archetype, region);

      return this.buildResult(lesson, atoms, [], {
        dayNumber,
        archetype,
        region,
        _source: 'seed-lessons'
      });
    } catch (e) {
      return null;
    }
  },

  seedToLesson(seed, dayNumber) {
    const topicEn = seed?.meta?.topic?.en || seed?.meta?.topic || seed?.topic?.en || seed?.topic || 'Loading...';
    const truthEn = seed?.universal_truth?.en || seed?.universal_truth || seed?.meta?.universalTruth || '';
    const headlineEn = seed?.headline?.en || seed?.headline || '';
    return {
      id: `seed-${dayNumber}`,
      day_number: dayNumber,
      topic: topicEn,
      universal_truth: truthEn,
      marketing_headline: headlineEn || topicEn,
      marketing_tagline: '',
      category: seed?.meta?.category || '',
      emoji: seed?.meta?.emoji || '📚',
      // Preserve full phases object for UI access to phase titles
      phases: seed?.phases || null,
      // Preserve growTrack data for Grow track UI
      growTrack: seed?.growTrack || null,
    };
  },

  seedToAtoms(seed, dayNumber, archetype, region) {
    const phases = seed?.phases || {};
    const pickLang = (region === 'es' || region === 'pt') ? region : 'en';
    const lang = (pickLang === 'pt') ? 'pt' : (pickLang === 'es') ? 'es' : 'en';

    const getText = (node) => {
      if (!node) return '';
      if (typeof node === 'string') return node;
      if (node?.[lang] && node[lang] !== '[NEEDS TRANSLATION]') return node[lang];
      if (node?.en) return node.en;
      return '';
    };

    // Make archetype feel distinct without needing separate DB rows.
    const voiceWrap = (phaseKey, base) => {
      const t = String(base || '').trim();
      if (!t) return '';
      if (String(archetype) === 'The Explorer') {
        if (phaseKey === 'hook') return `Adventure time. ${t}`;
        if (phaseKey === 'cliff') return `Two paths. ${t}`;
        return t;
      }
      if (String(archetype) === 'The Rebel') {
        if (phaseKey === 'hook') return `No fluff. ${t}`;
        if (phaseKey === 'cliff') return `Choose your move. ${t}`;
        return t;
      }
      // Scientist default
      return t;
    };

    const phaseEntries = [
      ['hook', phases.hook],
      ['cliff', phases.cliff],
      ['q1', phases.q1],
      ['q2', phases.q2],
      ['q3', phases.q3],
      ['wisdom', phases.wisdom],
      ['outro', phases.outro],
    ];

    return phaseEntries.map(([phaseKey, phaseNode]) => {
      const phaseName = this.MVP_PHASE_KEY_MAP[String(phaseKey)] || String(phaseKey);

      // Extract phase title (topic-specific, e.g., "The 66-Day Truth" instead of generic "Fact 1")
      const title = getText(phaseNode?.title);
      const script = voiceWrap(String(phaseKey), getText(phaseNode?.script));
      const prompt = getText(phaseNode?.prompt);
      const options = Array.isArray(phaseNode?.options) ? phaseNode.options : [];

      const normalizedOptions = (options.length >= 2)
        ? options.slice(0, 2).map((opt, idx) => ({
          letter: opt?.letter || (idx === 0 ? 'A' : 'B'),
          icon: this.defaultIconForOption(phaseName, archetype, idx),
          text: getText(opt?.text) || (idx === 0 ? 'Option A' : 'Option B'),
          quality: opt?.quality || 'good',
          response: getText(opt?.response) || 'Nice choice.',
        }))
        : this.buildDefaultOptionsForPhase(phaseName, archetype);

      return {
        id: `seed-${dayNumber}-${phaseKey}-${String(archetype).replace(/\s+/g, '-')}`,
        phase: phaseName,
        archetype: archetype,
        content: {
          title: title || undefined, // Topic-specific phase title from lesson JSON
          script: script || `Today: ${seed?.meta?.topic?.en || seed?.meta?.topic || 'a new idea'}.`,
          prompt: prompt || undefined,
          options: normalizedOptions,
        },
        hd_video_url: null,
        visual_url: this.MVP_DEFAULT_VISUAL_URL
      };
    });
  },

  buildDefaultOptionsForPhase(phaseName, archetype) {
    const p = String(phaseName || '').toLowerCase();
    if (p === 'hook') {
      return [
        { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'Teach me', quality: 'good', response: "Love it. Let's go." },
        { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'Make it practical', quality: 'good', response: "Perfect. We'll keep it useful." },
      ];
    }
    if (p === 'cliff') {
      return [
        { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'Go deeper', quality: 'good', response: 'Great. Depth first.' },
        { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'Keep it simple', quality: 'good', response: 'Great. Simple is powerful.' },
      ];
    }
    if (p === 'fact1' || p === 'fact2' || p === 'fact3') {
      return [
        { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'Give me an example', quality: 'good', response: 'Example coming up.' },
        { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'Show me the idea', quality: 'good', response: "Got it. Here's the core idea." },
      ];
    }
    if (p === 'wisdom') {
      return [
        { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'What should I do?', quality: 'good', response: "Let's turn this into action." },
        { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'What should I remember?', quality: 'good', response: "Here's the takeaway." },
      ];
    }
    if (p === 'outro') {
      return [
        { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'Lock it in', quality: 'good', response: 'Done. You’ve got it.' },
        { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'Come back tomorrow', quality: 'good', response: 'See you tomorrow.' },
      ];
    }
    return [
      { letter: 'A', icon: this.defaultIconForOption(phaseName, archetype, 0), text: 'A', quality: 'good', response: 'Nice.' },
      { letter: 'B', icon: this.defaultIconForOption(phaseName, archetype, 1), text: 'B', quality: 'good', response: 'Nice.' },
    ];
  },

  defaultIconForOption(phaseName, archetype, idx) {
    const a = String(archetype || '').toLowerCase();
    const p = String(phaseName || '').toLowerCase();
    if (a.includes('explorer')) return idx === 0 ? '🧭' : '🗺️';
    if (a.includes('rebel')) return idx === 0 ? '⚡' : '🔥';
    if (p === 'wisdom') return idx === 0 ? '✅' : '💎';
    return idx === 0 ? '📌' : '💡';
  },

  // Minimal, practical topic safety gate (client-side).
  validateOnDemandTopic(raw) {
    const topic = String(raw || '').trim();
    if (topic.length < this.ON_DEMAND_TOPIC_MIN_LEN) return { ok: false, reason: 'too_short' };
    if (topic.length > this.ON_DEMAND_TOPIC_MAX_LEN) return { ok: false, reason: 'too_long' };
    const blocked = /(porn|sex|rape|nud(e|ity)|bestial|incest|fetish|suicide|self[-\s]?harm|kill yourself|how to (make|build) (a )?(bomb|weapon)|explosive|meth|cocaine|heroin)/i;
    if (blocked.test(topic)) return { ok: false, reason: 'blocked' };
    const allowed = /^[\p{L}\p{N}\s,'’".:;?!()&\-]+$/u;
    if (!allowed.test(topic)) return { ok: false, reason: 'bad_chars' };
    return { ok: true, topic };
  },

  getOnDemandTopic() {
    // URL param takes precedence.
    try {
      if (typeof location !== 'undefined' && location.search) {
        const p = new URLSearchParams(location.search);
        const fromUrl = p.get('gen') || p.get('topic');
        if (fromUrl) {
          const v = this.validateOnDemandTopic(fromUrl);
          return v.ok ? v.topic : null;
        }
      }
    } catch (_) {}

    // Otherwise use localStorage only when the user is in Grow track.
    try {
      if (typeof localStorage === 'undefined') return null;
      const rawState = localStorage.getItem('kellyState');
      const state = rawState ? JSON.parse(rawState) : {};
      if (String(state?.track || 'learn') !== 'grow') return null;

      const stored = localStorage.getItem(this.ON_DEMAND_TOPIC_STORAGE_KEY);
      if (!stored) return null;
      const v = this.validateOnDemandTopic(stored);
      return v.ok ? v.topic : null;
    } catch (_) {
      return null;
    }
  },

  buildOnDemandLesson(dayNum, archetype, region, topic) {
    const lesson = {
      id: `ondemand-${Date.now()}`,
      day_number: dayNum,
      topic,
      universal_truth: 'A good question is a doorway. Today we walk through it.',
      marketing_headline: topic,
      marketing_tagline: '',
      category: 'On Demand',
      emoji: '✨',
    };

    const phaseScripts = {
      Hook: `Today, you picked: "${topic}". We'll keep it simple, clear, and useful.`,
      Cliff: `Quick choice: do you want the practical path, or the deep path?`,
      Fact1: `First: the simplest truth about "${topic}".`,
      Fact2: `Second: the thing most people miss about "${topic}".`,
      Fact3: `Third: the "aha" connection that makes it stick.`,
      Wisdom: `Now the takeaway: one sentence you can use today.`,
      Outro: `That's it. Small lesson, big leverage. Want another?`,
    };

    const atoms = this.MVP_PHASE_ORDER.map((phaseName) => ({
      id: `ondemand-${dayNum}-${phaseName}-${String(archetype).replace(/\s+/g, '-')}`,
      phase: phaseName,
      archetype,
      content: {
        script: phaseScripts[phaseName] || '',
        options: this.buildDefaultOptionsForPhase(phaseName, archetype),
      },
      hd_video_url: null,
      visual_url: this.MVP_DEFAULT_VISUAL_URL
    }));

    return this.buildResult(lesson, atoms, [], {
      dayNumber: dayNum,
      archetype,
      region,
      _source: 'on-demand'
    });
  },

  ensureMvpLessonShape(result, ctx) {
    // Normalize atoms into the shape learn.html expects: content.script + content.options[2] + visual_url.
    if (!result || !Array.isArray(result.atoms)) return result;

    const { dayNum } = ctx || {};
    const fallbackVisual = this.MVP_DEFAULT_VISUAL_URL;

    result.atoms = result.atoms.map((atom) => {
      const phase = atom?.phase || 'Hook';
      const archetype = atom?.archetype || (ctx?.archetype || 'The Scientist');

      const content = (typeof atom?.content === 'object' && atom.content) ? atom.content : { script: String(atom?.content || '') };
      const script = (typeof content.script === 'string') ? content.script : (typeof content.text === 'string') ? content.text : '';
      let options = Array.isArray(content.options) ? content.options : [];

      if (options.length < 2) options = this.buildDefaultOptionsForPhase(phase, archetype);
      if (options.length > 2) options = options.slice(0, 2);

      const normalizedOptions = options.map((opt, idx) => ({
        letter: opt?.letter || (idx === 0 ? 'A' : 'B'),
        icon: opt?.icon || this.defaultIconForOption(phase, archetype, idx),
        text: opt?.text || (idx === 0 ? 'Option A' : 'Option B'),
        quality: opt?.quality || 'good',
        response: opt?.response || 'Nice choice.',
      }));

      return {
        ...atom,
        phase,
        archetype,
        content: {
          ...content,
          script: script || `Day ${dayNum || ''}: ${result.lesson?.topic || 'Today'}.`,
          options: normalizedOptions,
        },
        visual_url: atom?.visual_url || fallbackVisual,
        hd_video_url: atom?.hd_video_url ?? null,
      };
    });

    // Ensure all MVP phases exist (gap-filler).
    try {
      const have = new Set(result.atoms.map(a => a?.phase).filter(Boolean));
      const topic = result.lesson?.topic || 'Today';
      const archetype = ctx?.archetype || 'The Scientist';
      for (const phaseName of this.MVP_PHASE_ORDER) {
        if (have.has(phaseName)) continue;
        result.atoms.push({
          id: `mvpfill-${dayNum || 'x'}-${phaseName}-${String(archetype).replace(/\\s+/g, '-')}`,
          phase: phaseName,
          archetype,
          content: {
            script: `Today: ${topic}.`,
            options: this.buildDefaultOptionsForPhase(phaseName, archetype),
          },
          hd_video_url: null,
          visual_url: fallbackVisual,
        });
      }
    } catch (_) {}

    return result;
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
      script.src = '/data/support-lessons.js';
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
    
    // Try curriculum files (BULLETPROOF - always has correct titles)
    try {
      const curriculumResult = await this.tryCurriculumFiles(dayNumber);
      if (curriculumResult) {
        __kellyLoaderDebugLog(`✅ Using curriculum file for day ${dayNumber}`);
        return curriculumResult;
      }
    } catch (e) {
      __kellyLoaderDebugWarn(`⚠️ Curriculum files failed: ${e.message}`);
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
   * Try loading lesson topic from curriculum files (year1-foundations or year2-ai-fluency)
   * This is the most reliable source for lesson titles.
   */
  async tryCurriculumFiles(dayNumber, track = 'learn') {
    const months = ['january', 'february', 'march', 'april', 'may', 'june', 
                    'july', 'august', 'september', 'october', 'november', 'december'];
    
    // Calculate month from day number
    const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let dayCount = 0;
    let monthIndex = 0;
    for (let i = 0; i < 12; i++) {
      if (dayNumber <= dayCount + daysInMonth[i]) {
        monthIndex = i;
        break;
      }
      dayCount += daysInMonth[i];
    }
    
    const monthKey = months[monthIndex];
    const basePath = track === 'grow' 
      ? '/data/curriculum/year2-ai-fluency' 
      : '/data/curriculum/year1-foundations';
    
    const res = await fetch(`${basePath}/${monthKey}_curriculum.json`, {
      signal: AbortSignal.timeout(2000)
    });
    if (!res.ok) return null;
    
    const data = await res.json();
    const lesson = data?.days?.find(d => d.day === dayNumber);
    
    if (!lesson) return null;
    
    // Build a proper lesson result from curriculum data
    const lessonObj = {
      id: `curriculum-${dayNumber}`,
      day_number: dayNumber,
      topic: lesson.title,
      universal_truth: lesson.learning_objective || lesson.title,
      marketing_headline: lesson.title,
      marketing_tagline: lesson.category || '',
      emoji: lesson.icon || '📚'
    };
    
    // Build minimal atoms for display
    const atoms = this.MVP_PHASE_ORDER.map(phase => ({
      id: `curriculum-${dayNumber}-${phase}`,
      phase,
      archetype: 'The Scientist',
      content: {
        script: `Today we explore: ${lesson.title}`,
        options: this.buildDefaultOptionsForPhase(phase, 'The Scientist')
      },
      visual_url: this.MVP_DEFAULT_VISUAL_URL
    }));
    
    return this.buildResult(lessonObj, atoms, [], {
      dayNumber,
      archetype: 'The Scientist',
      region: 'adult',
      _source: 'curriculum'
    });
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

// ============================================================
// Right-panel Search: "Create a new lesson: <query>"
// - No learn.html edits required: we inject a normal search result item.
// - We store the topic in localStorage and switch to Grow track.
// ============================================================
(function setupOnDemandSearchInjection() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;

  let __bound = false;
  let __boundInput = null;
  let __boundContainer = null;
  let __retryTimer = null;
  let __rootObserver = null;
  let __valuePollTimer = null;

  const tryBind = () => {
    const input = document.getElementById('panel-search-input');
    const container = document.getElementById('panel-search-results');

    // If DOM isn't ready yet (or panels mount late), keep retrying.
    if (!input || !container) return false;

    // Idempotent: if we're already bound to the current nodes, do nothing.
    if (__bound && __boundInput === input && __boundContainer === container) return true;

    // If we previously bound to a different input/container, stop old polling.
    if (__valuePollTimer) {
      clearInterval(__valuePollTimer);
      __valuePollTimer = null;
    }

    const removeExisting = () => {
      const existing = document.getElementById('kelly-search-generate');
      if (existing) existing.remove();
    };

    const getListEl = () => {
      const list = container.querySelector('.search-results-list');
      if (list) return list;
      // If the search rendered a "no results" <p>, create a list below it.
      const newList = document.createElement('div');
      newList.className = 'search-results-list';
      container.appendChild(newList);
      return newList;
    };

    const render = () => {
      removeExisting();

      const query = String(input.value || '').trim();
      if (query.length < 3) return;

      const validator = window.KellyLessonLoader?.validateOnDemandTopic?.(query);
      const ok = !!validator?.ok;
      __kellyLoaderDebugLog('[OnDemandSearch] render', { query, ok, reason: validator?.reason || null });

      const item = document.createElement('div');
      item.className = 'search-result-item';
      item.id = 'kelly-search-generate';
      item.setAttribute('role', 'button');
      item.setAttribute('tabindex', '0');

      const title = ok ? `Create lesson: ${validator.topic}` : `Can't create lesson: "${query}"`;
      const meta = ok
        ? 'Instant • Safe defaults • Uses Grow track'
        : `Blocked (${validator?.reason || 'invalid'})`;

      item.innerHTML = `
        <span class="search-result-emoji">✨</span>
        <div class="search-result-info">
          <div class="search-result-topic">${title}</div>
          <div class="search-result-meta">${meta}</div>
        </div>
      `;

      const activate = () => {
        if (!ok) return;
        try {
          // Store topic for loader (Grow track only)
          localStorage.setItem(window.KellyLessonLoader.ON_DEMAND_TOPIC_STORAGE_KEY, validator.topic);

          // Force Grow track in persisted state so the loader will use the on-demand topic.
          const raw = localStorage.getItem('kellyState');
          const next = raw ? JSON.parse(raw) : {};
          next.track = 'grow';
          localStorage.setItem('kellyState', JSON.stringify(next));

          // Update UI state if the toggle exists.
          const growBtn = document.getElementById('track-grow');
          if (growBtn) growBtn.click();

          const day = (window.state && window.state.currentDay) ? window.state.currentDay : 1;
          if (typeof window.loadLessonRuntime === 'function') {
            window.loadLessonRuntime(day);
            if (typeof window.closeUnifiedPanel === 'function') window.closeUnifiedPanel();
            return;
          }
        } catch (_) {}

        // Hard fallback: reload with URL param for the loader.
        try {
          window.location.href = `/learn.html?gen=${encodeURIComponent(validator.topic)}`;
        } catch (_) {}
      };

      item.addEventListener('click', activate);
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          activate();
        }
      });

      const list = getListEl();
      list.appendChild(item);
      __kellyLoaderDebugLog('[OnDemandSearch] injected row');
    };

    // Render after each search render pass.
    input.addEventListener('input', () => setTimeout(render, 0));
    input.addEventListener('change', () => setTimeout(render, 0));
    input.addEventListener('keyup', () => setTimeout(render, 0));

    // Also observe changes to the container (search re-renders innerHTML).
    const obs = new MutationObserver(() => setTimeout(render, 0));
    obs.observe(container, { childList: true, subtree: true });

    __bound = true;
    __boundInput = input;
    __boundContainer = container;
    __kellyLoaderDebugLog('[OnDemandSearch] bound to #panel-search-input and #panel-search-results');

    // First render (covers the case where input already has value)
    setTimeout(render, 0);

    // Value-poll fallback: catches cases where code sets input.value without firing events.
    let last = String(input.value || '');
    __valuePollTimer = setInterval(() => {
      const next = String(input.value || '');
      if (next !== last) {
        last = next;
        render();
      }
    }, 120);

    return true;
  };

  const ensureBound = () => {
    if (tryBind()) {
      if (__retryTimer) {
        clearInterval(__retryTimer);
        __retryTimer = null;
      }
      if (__rootObserver) {
        try { __rootObserver.disconnect(); } catch (_) {}
        __rootObserver = null;
      }
      return;
    }

    // Polling retry (simple + reliable)
    if (!__retryTimer) {
      let attempts = 0;
      __retryTimer = setInterval(() => {
        attempts += 1;
        if (tryBind()) {
          clearInterval(__retryTimer);
          __retryTimer = null;
        } else if (attempts >= 80) { // ~20s @ 250ms
          clearInterval(__retryTimer);
          __retryTimer = null;
        }
      }, 250);
    }

    // MutationObserver retry (covers late-mount DOM)
    if (!__rootObserver && document.documentElement) {
      __rootObserver = new MutationObserver(() => {
        if (tryBind()) {
          try { __rootObserver.disconnect(); } catch (_) {}
          __rootObserver = null;
        }
      });
      __rootObserver.observe(document.documentElement, { childList: true, subtree: true });
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensureBound);
  else ensureBound();
})();

