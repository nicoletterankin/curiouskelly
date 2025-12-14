// ============================================
// CURIOUS KELLY - PRODUCTION AUTH
// ============================================

// Use the browser singleton to prevent multiple GoTrue clients.
// Pages that import this module MUST include the Supabase SDK script tag + /js/lib/supabase.js first.
const supabase =
  (typeof window !== 'undefined' && typeof window.getSupabase === 'function'
    ? window.getSupabase({
        auth: {
          autoRefreshToken: true,
          persistSession: true,
          detectSessionInUrl: true,
          storageKey: 'curious-kelly-auth',
          flowType: 'pkce',
        },
      })
    : null);

// ============================================
// AUTH FUNCTIONS
// ============================================

export async function signInWithGoogle() {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/learn.html`,
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
      redirectTo: `${window.location.origin}/learn.html`
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
      redirectTo: `${window.location.origin}/learn.html`
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
      redirectTo: `${window.location.origin}/learn.html`,
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
      redirectTo: `${window.location.origin}/learn.html`,
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
  return supabase.auth.onAuthStateChange(async (event, session) => {
    console.log('Auth state changed:', event, session)
    
    // EARN TO LEARN: Process referral conversion on signup
    if (event === 'SIGNED_IN' && session?.user) {
      await processReferralConversion(session.user.id, session.access_token);
    }
    
    callback(event, session)
  })
}

// ============================================
// REFERRAL CONVERSION (Earn to Learn)
// ============================================

/**
 * Process referral conversion when user signs up
 * Links the referral click to the new user account
 */
async function processReferralConversion(userId, accessToken) {
  try {
    // Check if there's a stored referral code (from affiliate-tracking.js)
    let referralCode = null;
    
    // Try the global function first
    if (typeof window.getReferralCode === 'function') {
      referralCode = window.getReferralCode();
    }
    
    // Fallback to parsing localStorage directly
    if (!referralCode) {
      try {
        const stored = localStorage.getItem('kelly_referral');
        if (stored) {
          const data = JSON.parse(stored);
          referralCode = data.code;
        }
      } catch (e) {
        console.log('[Referral] Could not parse stored referral data');
      }
    }
    
    if (!referralCode) {
      console.log('[Referral] No referral code stored');
      return;
    }
    
    // Check if user is already referred (avoid duplicate processing)
    const { data: userData } = await supabase
      .from('users')
      .select('referred_by_user_id')
      .eq('id', userId)
      .single();
    
    if (userData?.referred_by_user_id) {
      console.log('[Referral] User already has referrer, skipping');
      return;
    }
    
    // Get full tracking data
    const trackingData = window.getReferralTrackingData?.() || {};
    
    console.log('[Referral] Processing conversion for:', referralCode);
    
    // Call the convert endpoint
    const response = await fetch('/api/referral/convert', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify({
        userId: userId,
        referralCode: referralCode,
        clickId: trackingData.clickId || null,
        conversionType: 'signup'
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('[Referral] ✅ Conversion successful!', result);
      
      // Fire analytics event
      if (typeof gtag !== 'undefined') {
        gtag('event', 'referral_signup', {
          event_category: 'referral',
          event_label: referralCode
        });
      }
    } else {
      console.warn('[Referral] Conversion failed:', result.message);
    }
    
  } catch (error) {
    console.error('[Referral] Error processing conversion:', error);
  }
}

// ============================================
// EXPORT SUPABASE CLIENT
// ============================================

export { supabase }



