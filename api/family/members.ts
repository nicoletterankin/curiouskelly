/**
 * Family Members API
 * 
 * Returns list of family members and their earnings status.
 * For parents/family admins to manage their family's Share & Earn.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface FamilyMember {
  id: string;
  displayName: string | null;
  email: string;
  age: number | null;
  isMinor: boolean;
  heldEarnings: number;
  totalReferrals: number;
  commissionTier: string;
}

interface ApiResponse {
  success: boolean;
  isFamilyAdmin: boolean;
  familyMembers?: FamilyMember[];
  totalHeldEarnings?: number;
  message?: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'GET') {
    res.status(405).json({ success: false, isFamilyAdmin: false, message: 'Method not allowed' });
    return;
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, isFamilyAdmin: false, message: 'Unauthorized' });
    return;
  }

  const token = authHeader.replace('Bearer ', '');

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const supabaseAnonKey = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseServiceKey || !supabaseAnonKey) {
    res.status(500).json({ success: false, isFamilyAdmin: false, message: 'Server configuration error' });
    return;
  }

  const supabaseAuth = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Get user from token
    const { data: { user }, error: authError } = await supabaseAuth.auth.getUser(token);
    
    if (authError || !user) {
      res.status(401).json({ success: false, isFamilyAdmin: false, message: 'Invalid token' });
      return;
    }

    // Check if user is a family admin
    const { data: userProfile } = await supabaseAdmin
      .from('users')
      .select('is_family_admin, earnings_held_for_minors')
      .eq('id', user.id)
      .single();

    if (!userProfile?.is_family_admin) {
      // Check if they have any linked children (might not have is_family_admin set)
      const { data: children } = await supabaseAdmin
        .from('users')
        .select('id')
        .eq('parent_account_id', user.id)
        .limit(1);

      if (!children || children.length === 0) {
        res.status(200).json({
          success: true,
          isFamilyAdmin: false,
          familyMembers: [],
          totalHeldEarnings: 0,
          message: 'No family members linked. Add family members to manage their earnings.'
        });
        return;
      }

      // They have children, set them as family admin
      await supabaseAdmin
        .from('users')
        .update({ is_family_admin: true })
        .eq('id', user.id);
    }

    // Get all family members (children linked to this parent)
    const { data: members, error: membersError } = await supabaseAdmin
      .from('users_with_age')
      .select(`
        id,
        display_name,
        email,
        calculated_age,
        is_minor,
        total_referrals,
        commission_tier
      `)
      .eq('parent_account_id', user.id);

    if (membersError) {
      console.error('Failed to get family members:', membersError);
      res.status(500).json({ success: false, isFamilyAdmin: true, message: 'Failed to load family members' });
      return;
    }

    // Get held earnings for each minor
    const familyMembers: FamilyMember[] = [];
    let totalHeldEarnings = 0;

    for (const member of members || []) {
      // Get held earnings
      const { data: heldData } = await supabaseAdmin
        .from('minor_earnings_ledger')
        .select('amount')
        .eq('minor_user_id', member.id)
        .eq('status', 'held');

      const heldEarnings = heldData?.reduce((sum, row) => sum + Number(row.amount), 0) || 0;
      totalHeldEarnings += heldEarnings;

      familyMembers.push({
        id: member.id,
        displayName: member.display_name,
        email: member.email,
        age: member.calculated_age,
        isMinor: member.is_minor,
        heldEarnings,
        totalReferrals: member.total_referrals || 0,
        commissionTier: member.commission_tier || 'new_learner'
      });
    }

    res.status(200).json({
      success: true,
      isFamilyAdmin: true,
      familyMembers,
      totalHeldEarnings
    });

  } catch (error) {
    console.error('Family members error:', error);
    res.status(500).json({ success: false, isFamilyAdmin: false, message: 'Internal server error' });
  }
}


