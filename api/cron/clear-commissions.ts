/**
 * Commission Clearing Cron Job
 * 
 * Runs daily to move commissions from 'pending' to 'approved' status
 * after the 7-day refund window has passed.
 * 
 * Also transfers approved amounts to available_earnings.
 * 
 * Vercel Cron: 0 6 * * * (6 AM UTC daily)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const REFUND_WINDOW_DAYS = 7;

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // Verify cron secret
  const authHeader = req.headers.authorization;
  const cronSecret = process.env.CRON_SECRET;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    res.status(500).json({ error: 'Missing Supabase configuration' });
    return;
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    console.log('[Commission Clearing] Starting...');
    
    // Calculate cutoff date (7 days ago)
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - REFUND_WINDOW_DAYS);
    
    // Find pending commissions older than 7 days
    const { data: pendingCommissions, error: fetchError } = await supabase
      .from('commission_transactions')
      .select('id, referrer_id, commission_amount')
      .eq('status', 'pending')
      .lt('created_at', cutoffDate.toISOString());

    if (fetchError) {
      throw fetchError;
    }

    if (!pendingCommissions || pendingCommissions.length === 0) {
      console.log('[Commission Clearing] No pending commissions to clear');
      res.status(200).json({ 
        success: true, 
        message: 'No pending commissions to clear',
        cleared: 0 
      });
      return;
    }

    console.log(`[Commission Clearing] Found ${pendingCommissions.length} commissions to clear`);

    // Group by referrer to batch updates
    const referrerAmounts: Record<string, number> = {};
    const commissionIds: string[] = [];

    for (const commission of pendingCommissions) {
      if (!referrerAmounts[commission.referrer_id]) {
        referrerAmounts[commission.referrer_id] = 0;
      }
      referrerAmounts[commission.referrer_id] += Number(commission.commission_amount);
      commissionIds.push(commission.id);
    }

    // Update commission statuses to approved
    const { error: updateCommissionsError } = await supabase
      .from('commission_transactions')
      .update({ 
        status: 'approved',
        approved_at: new Date().toISOString()
      })
      .in('id', commissionIds);

    if (updateCommissionsError) {
      throw updateCommissionsError;
    }

    // Update each referrer's earnings (move from pending to available)
    let updatedCount = 0;
    let totalAmount = 0;

    for (const [referrerId, amount] of Object.entries(referrerAmounts)) {
      // Get current earnings
      const { data: user, error: userError } = await supabase
        .from('users')
        .select('pending_earnings, available_earnings')
        .eq('id', referrerId)
        .single();

      if (userError || !user) {
        console.error(`[Commission Clearing] Failed to get user ${referrerId}:`, userError);
        continue;
      }

      const currentPending = Number(user.pending_earnings) || 0;
      const currentAvailable = Number(user.available_earnings) || 0;

      // Move amount from pending to available
      const newPending = Math.max(0, currentPending - amount);
      const newAvailable = currentAvailable + amount;

      const { error: updateError } = await supabase
        .from('users')
        .update({
          pending_earnings: newPending,
          available_earnings: newAvailable
        })
        .eq('id', referrerId);

      if (updateError) {
        console.error(`[Commission Clearing] Failed to update user ${referrerId}:`, updateError);
        continue;
      }

      updatedCount++;
      totalAmount += amount;
      console.log(`[Commission Clearing] Cleared $${amount.toFixed(2)} for user ${referrerId}`);
    }

    console.log(`[Commission Clearing] Complete: ${pendingCommissions.length} commissions, ${updatedCount} users, $${totalAmount.toFixed(2)} total`);

    res.status(200).json({
      success: true,
      message: 'Commissions cleared successfully',
      cleared: pendingCommissions.length,
      usersUpdated: updatedCount,
      totalAmount: totalAmount.toFixed(2)
    });

  } catch (error) {
    console.error('[Commission Clearing] Error:', error);
    res.status(500).json({
      error: 'Failed to clear commissions',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

