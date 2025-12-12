/**
 * Curious Kelly - Production Auth System
 * Designed for scale: millions of learners, zero friction
 * 
 * Supports:
 * - Google OAuth (primary, one-click)
 * - Apple OAuth (for iOS ecosystem)
 * - Magic Link (email fallback)
 * - Guest Mode (instant access, upgrade later)
 */

import { createClient, type SupabaseClient, type User, type Session } from '@supabase/supabase-js';

// ============================================
// CONFIGURATION
// ============================================

const SUPABASE_URL = import.meta.env.PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.PUBLIC_SUPABASE_ANON_KEY;

// Validate configuration at module load
if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('⚠️ Supabase not configured. Set PUBLIC_SUPABASE_URL and PUBLIC_SUPABASE_ANON_KEY');
}

// ============================================
// TYPES
// ============================================

export interface AuthState {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  isGuest: boolean;
  error: string | null;
}

export interface AuthCallbacks {
  onSignIn?: (user: User) => void;
  onSignOut?: () => void;
  onError?: (error: string) => void;
  onSessionChange?: (session: Session | null) => void;
}

export type AuthProvider = 'google' | 'apple' | 'github' | 'facebook';

// ============================================
// SINGLETON CLIENT
// ============================================

let supabaseClient: SupabaseClient | null = null;

function getClient(): SupabaseClient {
  if (!supabaseClient && SUPABASE_URL && SUPABASE_ANON_KEY) {
    supabaseClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true,
        // Storage key for session (allows multiple apps on same domain)
        storageKey: 'curious-kelly-auth',
        // Flow type for OAuth
        flowType: 'pkce',
      },
      // Connection pooling for scale
      global: {
        headers: {
          'x-client-info': 'curious-kelly-web',
        },
      },
    });
  }
  
  if (!supabaseClient) {
    throw new Error('Supabase client not initialized. Check environment variables.');
  }
  
  return supabaseClient;
}

// ============================================
// AUTH CLASS - For component integration
// ============================================

export class CuriousKellyAuth {
  private client: SupabaseClient;
  private callbacks: AuthCallbacks = {};
  private state: AuthState = {
    user: null,
    session: null,
    isLoading: true,
    isGuest: false,
    error: null,
  };

  constructor(callbacks?: AuthCallbacks) {
    this.client = getClient();
    if (callbacks) {
      this.callbacks = callbacks;
    }
    this.initializeAuth();
  }

  // ----------------------------------------
  // Initialization
  // ----------------------------------------

  private async initializeAuth() {
    try {
      // Check for existing session
      const { data: { session }, error } = await this.client.auth.getSession();
      
      if (error) {
        console.error('Session check error:', error);
        this.state.error = error.message;
      } else {
        this.state.session = session;
        this.state.user = session?.user ?? null;
        this.state.isGuest = !session;
      }

      // Listen for auth changes
      this.client.auth.onAuthStateChange((event, session) => {
        console.log('Auth event:', event);
        
        this.state.session = session;
        this.state.user = session?.user ?? null;
        this.state.isGuest = !session;
        this.state.isLoading = false;

        // Trigger callbacks
        if (event === 'SIGNED_IN' && session?.user) {
          this.callbacks.onSignIn?.(session.user);
          this.ensureUserRecord(session.user);
        } else if (event === 'SIGNED_OUT') {
          this.callbacks.onSignOut?.();
        }
        
        this.callbacks.onSessionChange?.(session);
      });

    } catch (err) {
      console.error('Auth initialization error:', err);
      this.state.error = err instanceof Error ? err.message : 'Unknown error';
    } finally {
      this.state.isLoading = false;
    }
  }

  // ----------------------------------------
  // OAuth Sign In (Google, Apple, etc.)
  // ----------------------------------------

  async signInWithOAuth(provider: AuthProvider, options?: {
    redirectTo?: string;
    scopes?: string;
  }): Promise<{ success: boolean; error?: string }> {
    try {
      this.state.isLoading = true;
      this.state.error = null;

      // Determine redirect URL
      const redirectTo = options?.redirectTo || this.getRedirectUrl();

      const { data, error } = await this.client.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo,
          scopes: options?.scopes,
          queryParams: provider === 'google' ? {
            access_type: 'offline',
            prompt: 'consent',
          } : undefined,
        },
      });

      if (error) {
        this.state.error = error.message;
        this.callbacks.onError?.(error.message);
        return { success: false, error: error.message };
      }

      // OAuth redirects, so if we get here without error, it's working
      return { success: true };

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'OAuth sign in failed';
      this.state.error = errorMsg;
      this.callbacks.onError?.(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      this.state.isLoading = false;
    }
  }

  // Convenience methods
  async signInWithGoogle(redirectTo?: string) {
    return this.signInWithOAuth('google', { redirectTo });
  }

  async signInWithApple(redirectTo?: string) {
    return this.signInWithOAuth('apple', { redirectTo });
  }

  // ----------------------------------------
  // Magic Link Sign In
  // ----------------------------------------

  async signInWithMagicLink(email: string, options?: {
    redirectTo?: string;
  }): Promise<{ success: boolean; error?: string }> {
    try {
      this.state.isLoading = true;
      this.state.error = null;

      const { error } = await this.client.auth.signInWithOtp({
        email: email.toLowerCase().trim(),
        options: {
          emailRedirectTo: options?.redirectTo || this.getRedirectUrl(),
        },
      });

      if (error) {
        this.state.error = error.message;
        this.callbacks.onError?.(error.message);
        return { success: false, error: error.message };
      }

      return { success: true };

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Magic link failed';
      this.state.error = errorMsg;
      this.callbacks.onError?.(errorMsg);
      return { success: false, error: errorMsg };
    } finally {
      this.state.isLoading = false;
    }
  }

  // ----------------------------------------
  // Sign Out
  // ----------------------------------------

  async signOut(): Promise<{ success: boolean; error?: string }> {
    try {
      const { error } = await this.client.auth.signOut();
      
      if (error) {
        this.state.error = error.message;
        return { success: false, error: error.message };
      }

      this.state.user = null;
      this.state.session = null;
      this.state.isGuest = true;

      return { success: true };

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Sign out failed';
      return { success: false, error: errorMsg };
    }
  }

  // ----------------------------------------
  // Guest Mode
  // ----------------------------------------

  enterGuestMode() {
    this.state.isGuest = true;
    this.state.user = null;
    this.state.session = null;
    
    // Store guest state
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('curious-kelly-guest', 'true');
      localStorage.setItem('curious-kelly-guest-started', new Date().toISOString());
    }
  }

  isGuestMode(): boolean {
    if (this.state.user) return false;
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem('curious-kelly-guest') === 'true';
    }
    return this.state.isGuest;
  }

  // ----------------------------------------
  // User Record Management
  // ----------------------------------------

  private async ensureUserRecord(user: User) {
    try {
      // Check if user exists in public.users
      const { data: existingUser, error: lookupError } = await this.client
        .from('users')
        .select('id')
        .eq('id', user.id)
        .maybeSingle();

      if (lookupError && lookupError.code !== 'PGRST116') {
        console.error('User lookup error:', lookupError);
        return;
      }

      if (!existingUser) {
        // Create user record
        const { error: insertError } = await this.client
          .from('users')
          .insert({
            id: user.id,
            email: user.email?.toLowerCase(),
            name: user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0],
            subscription_tier: 'free',
            subscription_status: 'inactive',
            current_day: 1,
            streak_days: 0,
          });

        if (insertError) {
          // Might fail due to RLS or trigger - that's OK
          console.warn('Could not create user record (may be created by trigger):', insertError.message);
        } else {
          console.log('✅ User record created for:', user.email);
        }
      }
    } catch (err) {
      console.error('ensureUserRecord error:', err);
    }
  }

  // ----------------------------------------
  // Getters
  // ----------------------------------------

  getState(): AuthState {
    return { ...this.state };
  }

  getUser(): User | null {
    return this.state.user;
  }

  getSession(): Session | null {
    return this.state.session;
  }

  isAuthenticated(): boolean {
    return !!this.state.user;
  }

  isLoading(): boolean {
    return this.state.isLoading;
  }

  // ----------------------------------------
  // Utilities
  // ----------------------------------------

  private getRedirectUrl(): string {
    if (typeof window === 'undefined') {
      return 'https://curiouskelly.com';
    }
    // Redirect to root - the OS will handle showing the right view
    return `${window.location.origin}/`;
  }

  getClient(): SupabaseClient {
    return this.client;
  }
}

// ============================================
// STANDALONE FUNCTIONS (for simple usage)
// ============================================

export async function signInWithGoogle(redirectTo?: string) {
  const client = getClient();
  const redirect = redirectTo || (typeof window !== 'undefined' ? `${window.location.origin}/` : 'https://curiouskelly.com');
  
  return client.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: redirect,
      queryParams: {
        access_type: 'offline',
        prompt: 'consent',
      },
    },
  });
}

export async function signInWithApple(redirectTo?: string) {
  const client = getClient();
  const redirect = redirectTo || (typeof window !== 'undefined' ? `${window.location.origin}/` : 'https://curiouskelly.com');
  
  return client.auth.signInWithOAuth({
    provider: 'apple',
    options: {
      redirectTo: redirect,
    },
  });
}

export async function signInWithMagicLink(email: string, redirectTo?: string) {
  const client = getClient();
  const redirect = redirectTo || (typeof window !== 'undefined' ? `${window.location.origin}/` : 'https://curiouskelly.com');
  
  return client.auth.signInWithOtp({
    email: email.toLowerCase().trim(),
    options: {
      emailRedirectTo: redirect,
    },
  });
}

export async function signOut() {
  const client = getClient();
  return client.auth.signOut();
}

export async function getSession() {
  const client = getClient();
  return client.auth.getSession();
}

export async function getUser() {
  const client = getClient();
  const { data: { user } } = await client.auth.getUser();
  return user;
}

export function onAuthStateChange(callback: (event: string, session: Session | null) => void) {
  const client = getClient();
  return client.auth.onAuthStateChange(callback);
}

// ============================================
// RATE LIMITING (Client-side)
// ============================================

const rateLimitMap = new Map<string, number[]>();

export function checkRateLimit(key: string, maxRequests: number = 5, windowMs: number = 60000): boolean {
  const now = Date.now();
  const timestamps = rateLimitMap.get(key) || [];
  
  // Remove old timestamps
  const validTimestamps = timestamps.filter(t => now - t < windowMs);
  
  if (validTimestamps.length >= maxRequests) {
    return false; // Rate limited
  }
  
  validTimestamps.push(now);
  rateLimitMap.set(key, validTimestamps);
  return true; // OK to proceed
}

// ============================================
// EXPORT DEFAULT INSTANCE
// ============================================

let defaultAuthInstance: CuriousKellyAuth | null = null;

export function getAuth(callbacks?: AuthCallbacks): CuriousKellyAuth {
  if (!defaultAuthInstance) {
    defaultAuthInstance = new CuriousKellyAuth(callbacks);
  }
  return defaultAuthInstance;
}

export { getClient as getSupabaseClient };

