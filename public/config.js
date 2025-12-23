// Kelly OS Configuration
// BULLETPROOF LESSON LOADING - THE LESSON ALWAYS PLAYS
//
// Cascading fallback system:
//   1. Supabase (Primary) - 5s timeout
//   2. Cloudflare D1 (Mirror) - 3s timeout  
//   3. Static JSON (Pre-exported) - 2s timeout
//   4. Emergency Fallback (Hardcoded) - instant

// Supabase credentials (client-side safe - anon key only)
// This is the CORRECT project with 365 lessons in core_lessons table
window.KELLY_CONFIG = {
  supabaseUrl: 'https://tvjalxxsyryjphkforjv.supabase.co',
  // IMPORTANT: keep in sync with Supabase project's current anon key
  // (If this is wrong, the frontend will 401 with "Invalid API key")
  supabaseKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI',
  
  // CRITICAL: Enable browser-direct Supabase reads
  // The /api/ serverless fallback was failing because SUPABASE_SERVICE_ROLE_KEY
  // might not be set in Vercel. Browser-direct uses the anon key (above) which is safer.
  enableSupabaseClient: true,
  
  // Fallback endpoints
  d1ApiUrl: '/api/lessons',           // Cloudflare D1 mirror (or local API fallback)
  staticLessonsUrl: '/generated/lessons',  // Pre-exported static JSON

  // TTS endpoint
  // NOTE: This repo’s `vercel.json` disables `/api/*` in static deployments, so
  // runtime voice must NOT depend on `/api/tts` in production.
  // Use the Cloudflare Worker route (see `infrastructure/cloudflare/tts-worker/`).
  ttsEndpoint: 'https://tts.curiouskelly.com/tts',
  
  // Timeout configuration (never hang forever)
  timeouts: {
    supabase: 5000,  // 5 seconds
    d1: 3000,        // 3 seconds  
    static: 2000     // 2 seconds
  },
  
  // PRODUCTION MODE: No gating, ever.
  // "Never gate anyone.. ever.." - The contribution model is OPTIONAL.
  // All 730 lessons (Learn + Grow tracks) are available to everyone.
  testingMode: false,
  disablePaywall: true,  // ALL LESSONS ACCESSIBLE - contribution is optional
  
  // PAYWALL DELAY: N/A since paywall is disabled
  // Kept for backwards compatibility
  paywallDelayMs: 0,

  // VISUALS (INFOGRAPHIC POPUPS)
  // Disabled until the on-brand infographic pipeline passes QA.
  // Prevents low-quality/incorrect visuals from showing in production.
  visualsEnabled: false,
  
  // ACCESS MODEL: NEVER GATE, EVER.
  // - ALL 730 lessons (Learn + Grow) are available to everyone, always
  // - Contribution options exist for those who want to support:
  //   1. Sponsor a Learner - Help someone else access Kelly
  //   2. Annual Supporter - Contribute to platform growth
  //   3. Lifetime Founding Member - Join the founding circle
  //   4. BYOK Credits - Contribute AI credits for community videos
  //
  // Education is priceless, not worthless. We don't gate. We invite.
  accessModel: {
    todayIsFree: true,              // All days are free
    enablePayPerLesson: false,      // DISABLED - no gating
    enableSubscription: true,       // Optional contribution tiers
    emergencyLessonsCount: 40       // Bonus content for supporters
  }
};

// Expose as global variables for legacy compatibility (learn.html, index.html, etc.)
window.SUPABASE_URL = window.KELLY_CONFIG.supabaseUrl;
window.SUPABASE_ANON_KEY = window.KELLY_CONFIG.supabaseKey;
window.MANIFEST_URL = '/assets/kelly/kelly-personas-manifest.json';
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
window.D1_API_URL = window.KELLY_CONFIG.d1ApiUrl;


