// ============================================
// CURIOUS KELLY - PRODUCTION AUTH
// ============================================

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm'

// Initialize Supabase client
const supabase = createClient(
  'https://tvjalxxsyryjphkforjv.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcnp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5NjI3NTcsImV4cCI6MjA0NzUzODc1N30.kLMlC14ckEp-XoL8RX5liw_cMdGs8lR'
)

// ============================================
// AUTH FUNCTIONS
// ============================================

export async function signInWithGoogle() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/dashboard.html`,
      queryParams: {
        access_type: 'offline',
        prompt: 'consent',
      }
    }
  })
  
  if (error) {
    console.error('Google sign-in error:', error)
    throw error
  }
  
  return data
}

export async function signInWithApple() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'apple',
    options: {
      redirectTo: `${window.location.origin}/dashboard.html`
    }
  })
  
  if (error) {
    console.error('Apple sign-in error:', error)
    throw error
  }
  
  return data
}

export async function signInWithGitHub() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: `${window.location.origin}/dashboard.html`
    }
  })
  
  if (error) {
    console.error('GitHub sign-in error:', error)
    throw error
  }
  
  return data
}

export async function signInWithFacebook() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'facebook',
    options: {
      redirectTo: `${window.location.origin}/dashboard.html`,
      scopes: 'public_profile,email' 
    }
  })
  
  if (error) {
    console.error('Facebook sign-in error:', error)
    throw error
  }
  
  return data
}

export async function signInWithOpenAI() {
  // Note: OpenAI login usually requires a custom OIDC provider configuration in Supabase
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'oidc', // or specific OpenAI provider key if configured
    options: {
      redirectTo: `${window.location.origin}/dashboard.html`,
      scopes: 'openid profile email'
    }
  })
  
  if (error) {
    console.error('OpenAI sign-in error:', error)
    throw error
  }
  
  return data
}

export async function signOut() {
  const { error } = await supabase.auth.signOut()
  
  if (error) {
    console.error('Sign-out error:', error)
    throw error
  }
  
  // Redirect to homepage
  window.location.href = '/'
}

export async function getSession() {
  const { data: { session }, error } = await supabase.auth.getSession()
  
  if (error) {
    console.error('Get session error:', error)
    return null
  }
  
  return session
}

export async function getUser() {
  const { data: { user }, error } = await supabase.auth.getUser()
  
  if (error) {
    console.error('Get user error:', error)
    return null
  }
  
  return user
}

// ============================================
// SESSION MANAGEMENT
// ============================================

export async function requireAuth() {
  const session = await getSession()
  
  if (!session) {
    // Redirect to login
    window.location.href = '/index.html'
    return null
  }
  
  return session
}

export async function checkAuth() {
  const session = await getSession()
  return !!session
}

// ============================================
// AUTH STATE LISTENER
// ============================================

export function onAuthStateChange(callback) {
  return supabase.auth.onAuthStateChange((event, session) => {
    console.log('Auth state changed:', event, session)
    callback(event, session)
  })
}

// ============================================
// EXPORT SUPABASE CLIENT
// ============================================

export { supabase }



