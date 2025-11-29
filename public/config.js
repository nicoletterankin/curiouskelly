// Curious Kelly - Client Configuration
// These values are injected at build time or loaded from environment

window.SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
window.SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

// ElevenLabs is now handled server-side via /api/tts for security
// Voice ID is still used client-side for API requests
window.ELEVENLABS_VOICE_ID = 'wAdymQH5YucAkXwmrdL0'; // Kelly's voice ID

window.STRIPE_PUBLISHABLE_KEY = 'pk_live_51SXAYMEs6ql8qYcKCMClObrDq0eFVaKrhlEypQEVQHbFBfVloknFTitYLHn6TTWjPwMqWZfGT66iGycNiLLqnPQp004efrYmjm'; // For checkout

// Feature flags
window.FEATURES = {
  unity3D: true,
  voiceGeneration: true,
  offlineMode: false
};
