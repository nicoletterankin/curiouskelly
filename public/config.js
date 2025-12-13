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
  supabaseKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4NjI0NzgsImV4cCI6MjA0OTQzODQ3OH0.qfTs_t0tLmVHFNlKlOqXxvbmEgUEZpHdnVAFbQdJv1c',
  
  // Fallback endpoints
  d1ApiUrl: '/api/lessons',           // Cloudflare D1 mirror (or local API fallback)
  staticLessonsUrl: '/generated/lessons',  // Pre-exported static JSON
  
  // Timeout configuration (never hang forever)
  timeouts: {
    supabase: 5000,  // 5 seconds
    d1: 3000,        // 3 seconds  
    static: 2000     // 2 seconds
  }
};

// Expose as global variables for legacy compatibility (learn.html, index.html, etc.)
window.SUPABASE_URL = window.KELLY_CONFIG.supabaseUrl;
window.SUPABASE_ANON_KEY = window.KELLY_CONFIG.supabaseKey;
window.MANIFEST_URL = '/assets/kelly/kelly-personas-manifest.json';
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';
window.D1_API_URL = window.KELLY_CONFIG.d1ApiUrl;

