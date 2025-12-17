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
  
  // Timeout configuration (never hang forever)
  timeouts: {
    supabase: 5000,  // 5 seconds
    d1: 3000,        // 3 seconds  
    static: 2000     // 2 seconds
  },
  
  // DEVELOPMENT MODE: Paywall disabled for testing
  // TODO: Set back to false before production deploy
  testingMode: true,
  disablePaywall: true,
  
  // PAYWALL DELAY: Seconds to wait before showing paywall (let users preview)
  // Set to 0 for instant paywall (old behavior)
  paywallDelayMs: 5000,

  // VISUALS (INFOGRAPHIC POPUPS)
  // Disabled until the on-brand infographic pipeline passes QA.
  // Prevents low-quality/incorrect visuals from showing in production.
  visualsEnabled: false,
  
  // ACCESS MODEL:
  // - "Today's lesson" is ALWAYS included for everyone, forever
  // - Pay-per-lesson for past/future lessons
  // - Subscription unlocks all 365 + emergency lessons
  //
  // "Today" = the lesson for the current calendar day (Day of Year 1-365)
  // This creates urgency: learn today or pay for it later
  accessModel: {
    todayIsFree: true,              // The core promise
    enablePayPerLesson: true,       // Buy individual lessons
    enableSubscription: true,       // Monthly/annual/lifetime access
    emergencyLessonsCount: 40       // Bonus lessons for subscribers
  }
};

// Expose as global variables for legacy compatibility (learn.html, index.html, etc.)
window.SUPABASE_URL = window.KELLY_CONFIG.supabaseUrl;
window.SUPABASE_ANON_KEY = window.KELLY_CONFIG.supabaseKey;
window.MANIFEST_URL = '/assets/kelly/kelly-personas-manifest.json';
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
window.D1_API_URL = window.KELLY_CONFIG.d1ApiUrl;


