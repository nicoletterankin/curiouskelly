/**
 * Referral Conversion API
 * POST /api/referral/convert
 * 
 * Called when a referred visitor creates an account or subscribes.
 * Links the referral_click to the new user and updates referrer stats.
 * 
 * LIFETIME ATTRIBUTION: Once attributed, the relationship persists forever.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface ConvertRequest {
  userId: string;           // The new user who signed up
  referralCode: string;     // The code they used
  clickId?: string;         // Optional: specific click to attribute
  conversionType: 'signup' | 'subscription' | 'gift';
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  if (req.method !== 'POST') {
    res.status(405).json({ success: false, message: 'Method not allowed' });
    return;
  }

  try {
    const { userId, referralCode, clickId, conversionType } = req.body as ConvertRequest;

    if (!userId || !referralCode) {
      res.status(400).json({
        success: false,
        message: 'userId and referralCode are required'
      });
      return;
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Get the referrer
    const { data: referrer, error: referrerError } = await supabase
      .from('users')
      .select('id, email, total_referrals, active_referrals')
      .eq('referral_code', referralCode.toLowerCase())
      .single();

    if (referrerError || !referrer) {
      res.status(404).json({
        success: false,
        message: 'Referral code not found'
      });
      return;
    }

    // Get the new user to check for self-referral
    const { data: newUser, error: userError } = await supabase
      .from('users')
      .select('id, email, referred_by_user_id')
      .eq('id', userId)
      .single();

    if (userError || !newUser) {
      res.status(404).json({
        success: false,
        message: 'User not found'
      });
      return;
    }

    // Prevent self-referral
    if (referrer.id === userId) {
      res.status(400).json({
        success: false,
        message: 'Self-referral is not allowed'
      });
      return;
    }

    // Check if user is already referred
    if (newUser.referred_by_user_id) {
      res.status(400).json({
        success: false,
        message: 'User already has a referrer'
      });
      return;
    }

    // Find the click record to update (if clickId provided, use that; otherwise find by code)
    let clickToUpdate = clickId;
    
    if (!clickToUpdate) {
      // Find the most recent unattributed click for this code
      const { data: click } = await supabase
        .from('referral_clicks')
        .select('id')
        .eq('referral_code', referralCode.toLowerCase())
        .is('converted_to_user_id', null)
        .order('clicked_at', { ascending: false })
        .limit(1)
        .single();
      
      if (click) {
        clickToUpdate = click.id;
      }
    }

    // Update the click record if found
    if (clickToUpdate) {
      await supabase
        .from('referral_clicks')
        .update({
          converted_to_user_id: userId,
          converted_at: new Date().toISOString(),
          conversion_type: conversionType
        })
        .eq('id', clickToUpdate);
    }

    // Update the new user with referral info
    const { error: updateUserError } = await supabase
      .from('users')
      .update({
        referred_by_user_id: referrer.id,
        referred_at: new Date().toISOString()
      })
      .eq('id', userId);

    if (updateUserError) {
      console.error('[Referral Convert] Failed to update user:', updateUserError);
      res.status(500).json({
        success: false,
        message: 'Failed to update user referral info'
      });
      return;
    }

    // Update referrer stats
    const { error: updateReferrerError } = await supabase
      .from('users')
      .update({
        total_referrals: (referrer.total_referrals || 0) + 1,
        active_referrals: (referrer.active_referrals || 0) + 1
      })
      .eq('id', referrer.id);

    if (updateReferrerError) {
      console.error('[Referral Convert] Failed to update referrer stats:', updateReferrerError);
    }

    console.log(`[Referral] Conversion: ${referralCode} -> User ${userId} (${conversionType})`);

    res.status(200).json({
      success: true,
      message: 'Referral conversion recorded',
      referrerId: referrer.id
    });

  } catch (error) {
    console.error('[Referral Convert] Error:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
}

