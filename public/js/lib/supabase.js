/**
 * Supabase singleton (browser)
 *
 * Goal:
 * - Guarantee a single Supabase client instance per browser context
 *   to avoid: "Multiple GoTrueClient instances detected..."
 *
 * Usage:
 *   const supabase = window.getSupabase();
 *   window.supabaseClient === supabase
 */
(function () {
  /** @type {import('@supabase/supabase-js').SupabaseClient | null} */
  let supabaseInstance = null;
  let initializedWith = null;

  function resolveSupabaseConfig() {
    // Prefer explicitly provided config objects used across the repo.
    const url =
      (window.KELLY_CONFIG && window.KELLY_CONFIG.supabaseUrl) ||
      (window.CONFIG && window.CONFIG.SUPABASE_URL) ||
      window.SUPABASE_URL ||
      null;

    const key =
      (window.KELLY_CONFIG && window.KELLY_CONFIG.supabaseKey) ||
      (window.CONFIG && window.CONFIG.SUPABASE_ANON_KEY) ||
      window.SUPABASE_ANON_KEY ||
      null;

    return { url, key };
  }

  /**
   * Get the singleton Supabase client.
   * @param {object} [options]
   * @param {object} [options.auth] - Supabase auth options (only used on first init)
   */
  function getSupabase(options = {}) {
    if (supabaseInstance) return supabaseInstance;

    const { url, key } = resolveSupabaseConfig();
    const createClient = window.supabase && window.supabase.createClient;

    if (!createClient || !url || !key) {
      // Keep this silent by default; callers can treat null as "no supabase".
      return null;
    }

    const defaultAuth = {
      autoRefreshToken: true,
      persistSession: true,
      detectSessionInUrl: true,
      storageKey: 'curious-kelly-auth',
      flowType: 'pkce',
    };

    initializedWith = { url, key, auth: options.auth || defaultAuth };
    supabaseInstance = createClient(url, key, { auth: initializedWith.auth });

    // Provide predictable globals for non-module scripts.
    window.supabaseClient = supabaseInstance;
    return supabaseInstance;
  }

  // Expose globally
  window.getSupabase = getSupabase;
  window.__supabaseSingletonInfo = () => ({ initializedWith, hasInstance: !!supabaseInstance });
})();

