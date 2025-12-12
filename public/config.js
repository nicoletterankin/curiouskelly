// Kelly OS Configuration
// Supabase credentials (client-side safe - anon key only)
// This is the CORRECT project with 365 lessons in core_lessons table
window.KELLY_CONFIG = {
  supabaseUrl: 'https://tvjalxxsyryjphkforjv.supabase.co',
  supabaseKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzM4NjI0NzgsImV4cCI6MjA0OTQzODQ3OH0.qfTs_t0tLmVHFNlKlOqXxvbmEgUEZpHdnVAFbQdJv1c'
};

// Expose as global variables for legacy compatibility (learn.html, index.html, etc.)
window.SUPABASE_URL = window.KELLY_CONFIG.supabaseUrl;
window.SUPABASE_ANON_KEY = window.KELLY_CONFIG.supabaseKey;
window.MANIFEST_URL = '/assets/kelly/kelly-personas-manifest.json';
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0';

