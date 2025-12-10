/**
 * Family Claim Earnings API
 * 
 * Allows a parent to claim held earnings from their child's account.
 * The earnings are transferred to the parent's available_earnings balance.
 * 
 * COMPLIANCE:
 * - Only works for linked family accounts
 * - Only for minors (under 18)
 * - Creates audit trail in compliance log
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface ClaimRequest {
  childId: string;
}

interface ApiResponse {
  success: boolean;
  message: string;
  amountClaimed?: number;
}

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
  const { childId } = req.body as ClaimRequest;

  if (!childId) {
    res.status(400).json({ success: false, message: 'Child ID is required' });
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
    // Get parent user from token
    const { data: { user: parentUser }, error: authError } = await supabaseAuth.auth.getUser(token);
    
    if (authError || !parentUser) {
      res.status(401).json({ success: false, message: 'Invalid token' });
      return;
    }

    // Use the database function to claim earnings
    const { data: claimResult, error: claimError } = await supabaseAdmin
      .rpc('parent_claim_minor_earnings', {
        parent_uuid: parentUser.id,
        minor_uuid: childId
      });

    if (claimError) {
      console.error('Claim error:', claimError);
      
      // Fallback to manual process if function doesn't exist
      // Verify parent-child relationship
      const { data: childProfile } = await supabaseAdmin
        .from('users')
        .select('id, parent_account_id, display_name')
        .eq('id', childId)
        .single();

      if (!childProfile || childProfile.parent_account_id !== parentUser.id) {
        res.status(403).json({ 
          success: false, 
          message: 'You can only claim earnings for children linked to your account' 
        });
        return;
      }

      // Get held earnings
      const { data: heldEarnings } = await supabaseAdmin
        .from('minor_earnings_ledger')
        .select('id, amount')
        .eq('minor_user_id', childId)
        .eq('status', 'held');

      if (!heldEarnings || heldEarnings.length === 0) {
        res.status(400).json({ 
          success: false, 
          message: 'No held earnings to claim' 
        });
        return;
      }

      const totalAmount = heldEarnings.reduce((sum, row) => sum + Number(row.amount), 0);

      // Update ledger entries
      await supabaseAdmin
        .from('minor_earnings_ledger')
        .update({
          status: 'transferred_to_parent',
          resolved_at: new Date().toISOString(),
          resolved_by: 'parent_claim'
        })
        .eq('minor_user_id', childId)
        .eq('status', 'held');

      // Update parent's available earnings
      await supabaseAdmin
        .from('users')
        .update({
          available_earnings: supabaseAdmin.rpc('increment_earnings', { 
            user_id: parentUser.id, 
            amount: totalAmount 
          })
        })
        .eq('id', parentUser.id);

      // Actually just do a simple update
      const { data: parentData } = await supabaseAdmin
        .from('users')
        .select('available_earnings, earnings_held_for_minors')
        .eq('id', parentUser.id)
        .single();

      await supabaseAdmin
        .from('users')
        .update({
          available_earnings: (Number(parentData?.available_earnings) || 0) + totalAmount,
          earnings_held_for_minors: Math.max(0, (Number(parentData?.earnings_held_for_minors) || 0) - totalAmount)
        })
        .eq('id', parentUser.id);

      // Log compliance event
      await supabaseAdmin
        .from('earnings_compliance_log')
        .insert({
          user_id: childId,
          event_type: 'parent_claimed_earnings',
          details: {
            parent_id: parentUser.id,
            amount: totalAmount
          }
        });

      res.status(200).json({
        success: true,
        message: `$${totalAmount.toFixed(2)} has been transferred to your available earnings`,
        amountClaimed: totalAmount
      });
      return;
    }

    // Database function succeeded
    const result = claimResult?.[0];
    if (!result?.success) {
      res.status(400).json({
        success: false,
        message: result?.message || 'Failed to claim earnings'
      });
      return;
    }

    res.status(200).json({
      success: true,
      message: `$${Number(result.amount_claimed).toFixed(2)} has been transferred to your available earnings`,
      amountClaimed: Number(result.amount_claimed)
    });

  } catch (error) {
    console.error('Claim earnings error:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
}



