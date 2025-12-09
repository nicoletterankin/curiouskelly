/**
 * Payout Request API
 * 
 * Allows eligible users to request a payout of their available earnings.
 * 
 * COMPLIANCE:
 * - Minors (under 18) CANNOT request payouts directly
 * - Minimum payout: $50
 * - Tax forms required for payouts over $600/year
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface PayoutRequest {
  amount?: number; // If not specified, request all available
  method: 'paypal' | 'stripe';
  paypalEmail?: string;
}

interface ApiResponse {
  success: boolean;
  message: string;
  payoutId?: string;
  amount?: number;
  estimatedArrival?: string;
}

const MINIMUM_PAYOUT = 50;
const TAX_FORM_THRESHOLD = 600;

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ success: false, message: 'Method not allowed' });
    return;
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, message: 'Unauthorized' });
    return;
  }

  const token = authHeader.replace('Bearer ', '');
  const { amount, method, paypalEmail } = req.body as PayoutRequest;

  // Validate method
  if (!method || !['paypal', 'stripe'].includes(method)) {
    res.status(400).json({ success: false, message: 'Invalid payout method. Use "paypal" or "stripe".' });
    return;
  }

  if (method === 'paypal' && !paypalEmail) {
    res.status(400).json({ success: false, message: 'PayPal email is required' });
    return;
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const supabaseAnonKey = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseServiceKey || !supabaseAnonKey) {
    res.status(500).json({ success: false, message: 'Server configuration error' });
    return;
  }

  const supabaseAuth = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Get user from token
    const { data: { user }, error: authError } = await supabaseAuth.auth.getUser(token);
    
    if (authError || !user) {
      res.status(401).json({ success: false, message: 'Invalid token' });
      return;
    }

    // COMPLIANCE CHECK: Get user's eligibility
    const { data: eligibilityData } = await supabaseAdmin
      .rpc('can_user_earn', { user_uuid: user.id });

    const eligibility = eligibilityData?.[0];

    // Also get user age directly as backup
    const { data: userData } = await supabaseAdmin
      .from('users_with_age')
      .select(`
        calculated_age,
        is_minor,
        available_earnings,
        lifetime_earnings,
        tax_form_status,
        payout_method,
        payout_details
      `)
      .eq('id', user.id)
      .single();

    if (!userData) {
      res.status(404).json({ success: false, message: 'User not found' });
      return;
    }

    // COMPLIANCE: Block minors from requesting payouts
    const isMinor = userData.is_minor || (userData.calculated_age !== null && userData.calculated_age < 18);
    
    if (isMinor) {
      // Log compliance event
      await supabaseAdmin
        .from('earnings_compliance_log')
        .insert({
          user_id: user.id,
          event_type: 'payout_blocked_minor',
          details: {
            age: userData.calculated_age,
            attempted_amount: amount || userData.available_earnings,
            reason: 'Minors cannot request payouts directly. Earnings are held until age 18 or can be claimed by a parent.'
          },
          ip_address: req.headers['x-forwarded-for']?.toString().split(',')[0] || null
        });

      res.status(403).json({
        success: false,
        message: userData.calculated_age !== null && userData.calculated_age < 13
          ? 'Users under 13 cannot participate in payouts. Ask a parent to link your account to their family.'
          : 'Users under 18 cannot request payouts. Your earnings are being held until you turn 18, or a parent can claim them if your account is linked.'
      });
      return;
    }

    // Check available earnings
    const availableEarnings = Number(userData.available_earnings) || 0;
    const requestedAmount = amount || availableEarnings;

    if (requestedAmount > availableEarnings) {
      res.status(400).json({
        success: false,
        message: `Requested amount ($${requestedAmount.toFixed(2)}) exceeds available earnings ($${availableEarnings.toFixed(2)})`
      });
      return;
    }

    // Check minimum payout
    if (requestedAmount < MINIMUM_PAYOUT) {
      res.status(400).json({
        success: false,
        message: `Minimum payout is $${MINIMUM_PAYOUT}. You have $${availableEarnings.toFixed(2)} available.`
      });
      return;
    }

    // Check tax form status for large payouts
    const lifetimeEarnings = Number(userData.lifetime_earnings) || 0;
    if (lifetimeEarnings + requestedAmount >= TAX_FORM_THRESHOLD && userData.tax_form_status !== 'completed') {
      res.status(400).json({
        success: false,
        message: `Payouts exceeding $${TAX_FORM_THRESHOLD}/year require a W-9 (US) or W-8BEN (non-US) tax form. Please complete your tax information first.`
      });
      return;
    }

    // Check for pending payout
    const { data: pendingPayouts } = await supabaseAdmin
      .from('payouts')
      .select('id')
      .eq('user_id', user.id)
      .eq('status', 'pending')
      .limit(1);

    if (pendingPayouts && pendingPayouts.length > 0) {
      res.status(400).json({
        success: false,
        message: 'You already have a pending payout request. Please wait for it to be processed.'
      });
      return;
    }

    // Create payout request
    const { data: payout, error: payoutError } = await supabaseAdmin
      .from('payouts')
      .insert({
        user_id: user.id,
        amount: requestedAmount,
        currency: 'USD',
        method,
        status: 'pending',
        notes: method === 'paypal' ? `PayPal: ${paypalEmail}` : 'Stripe Connect'
      })
      .select('id')
      .single();

    if (payoutError || !payout) {
      console.error('Failed to create payout:', payoutError);
      res.status(500).json({ success: false, message: 'Failed to create payout request' });
      return;
    }

    // Update user's payout method preference
    await supabaseAdmin
      .from('users')
      .update({
        payout_method: method,
        payout_details: method === 'paypal' ? { email: paypalEmail } : null,
        available_earnings: availableEarnings - requestedAmount,
        pending_earnings: (Number(userData.available_earnings) || 0) + requestedAmount
      })
      .eq('id', user.id);

    // Calculate estimated arrival (3-5 business days)
    const estimatedDate = new Date();
    estimatedDate.setDate(estimatedDate.getDate() + 5);
    
    res.status(200).json({
      success: true,
      message: `Payout request for $${requestedAmount.toFixed(2)} has been submitted`,
      payoutId: payout.id,
      amount: requestedAmount,
      estimatedArrival: estimatedDate.toISOString().split('T')[0]
    });

  } catch (error) {
    console.error('Payout error:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
}


