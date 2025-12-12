/**
 * Supabase Server Client
 * For server-side operations (API routes, webhooks)
 * 
 * Project: Curious Kelly (tvjalxxsyryjphkforjv)
 */

import { createClient, type SupabaseClient } from '@supabase/supabase-js';

// Types for our database
export interface UserRecord {
  id: string;
  email: string;
  name?: string;
  subscription_tier: 'free' | 'annual' | 'gift' | 'enterprise';
  subscription_status: 'active' | 'inactive' | 'cancelled' | 'expired' | 'trialing';
  subscription_started_at?: string;
  subscription_expires_at?: string;
  stripe_customer_id?: string;
  current_day: number;
  streak_days: number;
  last_lesson_at?: string;
  created_at?: string;
  updated_at?: string;
}

// Lazy-initialized client
let supabaseClient: SupabaseClient | null = null;

/**
 * Get Supabase admin client (for server-side operations)
 * Uses service role key for full access to bypass RLS
 */
export function getSupabaseAdmin(): SupabaseClient | null {
  // Use service role for admin operations (webhook handlers)
  const supabaseUrl = import.meta.env.PUBLIC_SUPABASE_URL;
  const serviceRoleKey = import.meta.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !serviceRoleKey) {
    console.warn('Supabase admin not configured - missing URL or service role key');
    return null;
  }

  if (!supabaseClient) {
    supabaseClient = createClient(supabaseUrl, serviceRoleKey, {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    });
  }

  return supabaseClient;
}

/**
 * Create or update a user record after successful checkout
 */
export async function upsertUserFromCheckout(data: {
  email: string;
  name?: string;
  stripeCustomerId: string;
  plan: 'monthly' | 'annual' | 'gift';
  trialEndDate?: Date;
}): Promise<{ success: boolean; error?: string }> {
  const supabase = getSupabaseAdmin();
  
  if (!supabase) {
    return { success: false, error: 'Supabase not configured' };
  }

  try {
    // Calculate subscription dates
    const now = new Date();
    const subscriptionStarted = now.toISOString();
    
    // For trial: expires at trial end. For gift: 1 year from now
    let subscriptionExpires: string | undefined;
    if (data.plan === 'gift') {
      const expiryDate = new Date();
      expiryDate.setFullYear(expiryDate.getFullYear() + 1);
      subscriptionExpires = expiryDate.toISOString();
    } else if (data.trialEndDate) {
      subscriptionExpires = data.trialEndDate.toISOString();
    }

    // First, check if user exists by email
    const { data: existingUser, error: lookupError } = await supabase
      .from('users')
      .select('id, email')
      .eq('email', data.email.toLowerCase())
      .maybeSingle();

    if (lookupError && lookupError.code !== 'PGRST116') {
      console.error('Error looking up user:', lookupError);
      return { success: false, error: lookupError.message };
    }

    if (existingUser) {
      // Update existing user
      const { error: updateError } = await supabase
        .from('users')
        .update({
          name: data.name || existingUser.email?.split('@')[0],
          subscription_tier: data.plan === 'gift' ? 'gift' : 'annual',
          subscription_status: data.plan === 'gift' ? 'active' : 'trialing',
          subscription_started_at: subscriptionStarted,
          subscription_expires_at: subscriptionExpires,
          stripe_customer_id: data.stripeCustomerId,
          updated_at: now.toISOString(),
        })
        .eq('id', existingUser.id);

      if (updateError) {
        console.error('Error updating user:', updateError);
        return { success: false, error: updateError.message };
      }
    } else {
      // Create new user record
      // Note: We can't create in public.users without auth.users entry
      // So we'll use a service function or just log for now
      console.log('New user checkout - will need auth signup:', data.email);
      
      // Try to insert anyway (will work if auth user exists)
      const { error: insertError } = await supabase
        .from('users')
        .insert({
          email: data.email.toLowerCase(),
          name: data.name || data.email.split('@')[0],
          subscription_tier: data.plan === 'gift' ? 'gift' : 'annual',
          subscription_status: data.plan === 'gift' ? 'active' : 'trialing',
          subscription_started_at: subscriptionStarted,
          subscription_expires_at: subscriptionExpires,
          stripe_customer_id: data.stripeCustomerId,
          current_day: 1,
          streak_days: 0,
        })
        .select()
        .single();

      if (insertError) {
        // This is expected if auth.users entry doesn't exist
        // The trigger will create it on first login
        console.warn('Could not insert user (expected if not signed up yet):', insertError.message);
      }
    }

    console.log(`✅ User record processed for ${data.email}`);
    return { success: true };
  } catch (error) {
    console.error('Error in upsertUserFromCheckout:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Update subscription status when payment succeeds
 */
export async function activateSubscription(data: {
  stripeCustomerId: string;
  plan: 'monthly' | 'annual' | 'gift';
  subscriptionId: string;
}): Promise<{ success: boolean; error?: string }> {
  const supabase = getSupabaseAdmin();
  
  if (!supabase) {
    return { success: false, error: 'Supabase not configured' };
  }

  try {
    // Calculate expiry based on plan
    const now = new Date();
    let subscriptionExpires: string;
    
    if (data.plan === 'annual' || data.plan === 'gift') {
      const expiryDate = new Date();
      expiryDate.setFullYear(expiryDate.getFullYear() + 1);
      subscriptionExpires = expiryDate.toISOString();
    } else {
      // Monthly
      const expiryDate = new Date();
      expiryDate.setMonth(expiryDate.getMonth() + 1);
      subscriptionExpires = expiryDate.toISOString();
    }

    const { error } = await supabase
      .from('users')
      .update({
        subscription_status: 'active',
        subscription_expires_at: subscriptionExpires,
        updated_at: now.toISOString(),
      })
      .eq('stripe_customer_id', data.stripeCustomerId);

    if (error) {
      console.error('Error activating subscription:', error);
      return { success: false, error: error.message };
    }

    console.log(`✅ Subscription activated for Stripe customer ${data.stripeCustomerId}`);
    return { success: true };
  } catch (error) {
    console.error('Error in activateSubscription:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Update subscription status (past_due, cancelled, etc.)
 */
export async function updateSubscriptionStatus(data: {
  stripeCustomerId: string;
  status: 'active' | 'inactive' | 'cancelled' | 'expired' | 'trialing';
}): Promise<{ success: boolean; error?: string }> {
  const supabase = getSupabaseAdmin();
  
  if (!supabase) {
    return { success: false, error: 'Supabase not configured' };
  }

  try {
    const { error } = await supabase
      .from('users')
      .update({
        subscription_status: data.status,
        updated_at: new Date().toISOString(),
      })
      .eq('stripe_customer_id', data.stripeCustomerId);

    if (error) {
      console.error('Error updating subscription status:', error);
      return { success: false, error: error.message };
    }

    console.log(`✅ Subscription status updated to ${data.status} for ${data.stripeCustomerId}`);
    return { success: true };
  } catch (error) {
    console.error('Error in updateSubscriptionStatus:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Get user by Stripe customer ID
 */
export async function getUserByStripeCustomerId(stripeCustomerId: string): Promise<UserRecord | null> {
  const supabase = getSupabaseAdmin();
  
  if (!supabase) {
    return null;
  }

  try {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .eq('stripe_customer_id', stripeCustomerId)
      .maybeSingle();

    if (error) {
      console.error('Error getting user by Stripe ID:', error);
      return null;
    }

    return data as UserRecord | null;
  } catch (error) {
    console.error('Error in getUserByStripeCustomerId:', error);
    return null;
  }
}

/**
 * Get user by email
 */
export async function getUserByEmail(email: string): Promise<UserRecord | null> {
  const supabase = getSupabaseAdmin();
  
  if (!supabase) {
    return null;
  }

  try {
    const { data, error } = await supabase
      .from('users')
      .select('*')
      .eq('email', email.toLowerCase())
      .maybeSingle();

    if (error) {
      console.error('Error getting user by email:', error);
      return null;
    }

    return data as UserRecord | null;
  } catch (error) {
    console.error('Error in getUserByEmail:', error);
    return null;
  }
}

