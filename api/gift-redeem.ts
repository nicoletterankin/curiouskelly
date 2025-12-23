/**
 * Gift Redemption API
 * 
 * POST /api/gift-redeem
 * 
 * Redeems a gift code and creates/updates the user account.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
// NOTE: `.js` extension required for Vercel's ESM output.
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase.js';

interface RedeemRequest {
  code: string;
  email: string;
}

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim().toLowerCase());
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric', 
    year: 'numeric' 
  });
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Parse body (defensive: Vercel may throw on invalid JSON)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let body: any = {};
  try {
    body = req.body as RedeemRequest;
  } catch {
    return res.status(400).json({ error: 'invalid_json' });
  }
  
  // Validate inputs
  if (!body.code || body.code.length !== 12) {
    return res.status(400).json({ error: 'invalid_code' });
  }
  
  if (!body.email || !isValidEmail(body.email)) {
    return res.status(400).json({ error: 'invalid_email' });
  }

  const code = body.code.toUpperCase();
  const email = body.email.trim().toLowerCase();

  if (!isSupabaseConfigured()) {
    // Return mock success for development
    return res.status(200).json({
      success: true,
      planName: '12 Month Access',
      gifterName: 'A friend',
      expiresAt: formatDate(new Date(Date.now() + 365 * 24 * 60 * 60 * 1000)),
      message: 'Enjoy learning with Kelly! 🎉'
    });
  }

  try {
    const supabase = getSupabaseAdmin();

    // Look up the gift code
    const { data: gift, error: giftError } = await supabase
      .from('gift_codes')
      .select('*')
      .eq('code', code)
      .single();

    if (giftError || !gift) {
      return res.status(404).json({ error: 'gift_not_found' });
    }

    if (gift.redeemed_at) {
      return res.status(400).json({ error: 'gift_already_redeemed' });
    }

    if (gift.expires_at && new Date(gift.expires_at) < new Date()) {
      return res.status(400).json({ error: 'gift_expired' });
    }

    // Calculate subscription dates
    const now = new Date();
    const durationMonths = gift.duration_months || 12;
    const expiresAt = new Date(now);
    expiresAt.setMonth(expiresAt.getMonth() + durationMonths);

    // Check if user exists
    const { data: existingUser } = await supabase
      .from('users')
      .select('id, subscription_tier, subscription_expires_at')
      .eq('email', email)
      .single();

    if (existingUser) {
      // Extend existing subscription
      const currentExpiry = existingUser.subscription_expires_at 
        ? new Date(existingUser.subscription_expires_at) 
        : now;
      const newExpiry = new Date(Math.max(currentExpiry.getTime(), now.getTime()));
      newExpiry.setMonth(newExpiry.getMonth() + durationMonths);

      await supabase
        .from('users')
        .update({
          subscription_tier: 'gift',
          subscription_status: 'active',
          subscription_expires_at: newExpiry.toISOString(),
          updated_at: now.toISOString()
        })
        .eq('id', existingUser.id);

      // Mark gift as redeemed
      await supabase
        .from('gift_codes')
        .update({
          redeemed_at: now.toISOString(),
          redeemed_by_email: email,
          redeemed_by_user_id: existingUser.id
        })
        .eq('id', gift.id);

      // Log event
      try {
        await supabase.from('user_events').insert({
          user_id: existingUser.id,
          event_type: 'gift.redeemed',
          event_category: 'learner_action',
          payload: {
            gift_id: gift.id,
            duration_months: durationMonths,
            new_expiry: newExpiry.toISOString(),
          },
        });
      } catch {
        // Non-critical event logging (network issues, etc.) should not block redemption.
      }

      return res.status(200).json({
        success: true,
        planName: `${durationMonths} Month Access`,
        gifterName: gift.gifter_name || 'A friend',
        expiresAt: formatDate(newExpiry),
        message: gift.message || null,
        accountExists: true
      });

    } else {
      // Create new user via Supabase Auth
      const { data: authData, error: authError } = await supabase.auth.admin.createUser({
        email,
        email_confirm: true, // Auto-confirm email
        user_metadata: {
          source: 'gift_redemption',
          gift_code: code
        }
      });

      if (authError) {
        console.error('Auth error:', authError);
        return res.status(500).json({ error: 'account_creation_failed' });
      }

      const userId = authData.user.id;

      // Create user profile
      await supabase.from('users').insert({
        id: userId,
        email,
        subscription_tier: 'gift',
        subscription_status: 'active',
        subscription_started_at: now.toISOString(),
        subscription_expires_at: expiresAt.toISOString(),
        acquisition_source: 'gift'
      });

      // Mark gift as redeemed
      await supabase
        .from('gift_codes')
        .update({
          redeemed_at: now.toISOString(),
          redeemed_by_email: email,
          redeemed_by_user_id: userId
        })
        .eq('id', gift.id);

      // Log event
      try {
        await supabase.from('user_events').insert({
          user_id: userId,
          event_type: 'gift.redeemed',
          event_category: 'learner_action',
          payload: {
            gift_id: gift.id,
            duration_months: durationMonths,
            new_account: true,
          },
        });
      } catch {
        // Non-critical event logging
      }

      return res.status(200).json({
        success: true,
        planName: `${durationMonths} Month Access`,
        gifterName: gift.gifter_name || 'A friend',
        expiresAt: formatDate(expiresAt),
        message: gift.message || null,
        accountCreated: true
      });
    }

  } catch (error) {
    console.error('Gift redemption error:', error);
    return res.status(500).json({ 
      error: 'redemption_failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
