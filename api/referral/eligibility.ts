/**
 * Referral Eligibility API
 * 
 * Checks if a user can participate in the Share & Earn program
 * based on age, parental consent, and family account status.
 * 
 * COPPA Compliance:
 * - Under 13: Requires parental consent, earnings go to parent
 * - 13-17: Can participate, earnings held until 18 or parent claims
 * - 18+: Full access
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface EligibilityResult {
  canSeeReferralLink: boolean;
  canShare: boolean;
  canAccumulateEarnings: boolean;
  canRequestPayout: boolean;
  earningsDestination: 'self' | 'parent' | 'parent_or_held' | 'held_until_18' | 'none';
  reason: string;
  userAge: number | null;
  isMinor: boolean;
  hasParentAccount: boolean;
  parentalConsentGiven: boolean;
  heldEarnings?: number;
}

interface ApiResponse {
  success: boolean;
  eligibility?: EligibilityResult;
  message?: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'GET') {
    res.status(405).json({ success: false, message: 'Method not allowed' });
    return;
  }

  // Get auth token from header
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, message: 'Unauthorized' });
    return;
  }

  const token = authHeader.replace('Bearer ', '');

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    res.status(500).json({ success: false, message: 'Server configuration error' });
    return;
  }

  const supabase = createClient(supabaseUrl, supabaseKey);

  try {
    // Get user from token
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    
    if (authError || !user) {
      res.status(401).json({ success: false, message: 'Invalid token' });
      return;
    }

    // Get user's eligibility from the database function
    const { data: eligibilityData, error: eligibilityError } = await supabase
      .rpc('can_user_earn', { user_uuid: user.id });

    if (eligibilityError) {
      console.error('Eligibility check error:', eligibilityError);
      // Fallback to manual calculation if function doesn't exist
      const { data: userData, error: userError } = await supabase
        .from('users')
        .select(`
          id,
          age,
          birthday,
          birth_year,
          parent_account_id,
          parental_consent_for_earnings,
          pending_earnings,
          available_earnings
        `)
        .eq('id', user.id)
        .single();

      if (userError || !userData) {
        res.status(404).json({ success: false, message: 'User not found' });
        return;
      }

      // Calculate age
      let userAge: number | null = null;
      if (userData.birthday) {
        const birthDate = new Date(userData.birthday);
        const today = new Date();
        userAge = today.getFullYear() - birthDate.getFullYear();
        const monthDiff = today.getMonth() - birthDate.getMonth();
        if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
          userAge--;
        }
      } else if (userData.birth_year) {
        userAge = new Date().getFullYear() - userData.birth_year;
      } else if (userData.age) {
        userAge = userData.age;
      }

      const hasParent = !!userData.parent_account_id;
      const hasConsent = !!userData.parental_consent_for_earnings;
      const isMinor = userAge !== null && userAge < 18;
      const isUnder13 = userAge !== null && userAge < 13;

      // Build eligibility response
      let eligibility: EligibilityResult;

      if (userAge === null) {
        // Age unknown - default to adult behavior
        eligibility = {
          canSeeReferralLink: true,
          canShare: true,
          canAccumulateEarnings: true,
          canRequestPayout: true,
          earningsDestination: 'self',
          reason: 'Age unknown - please update your profile',
          userAge: null,
          isMinor: false,
          hasParentAccount: hasParent,
          parentalConsentGiven: hasConsent
        };
      } else if (isUnder13) {
        if (hasParent && hasConsent) {
          eligibility = {
            canSeeReferralLink: true,
            canShare: true,
            canAccumulateEarnings: true,
            canRequestPayout: false,
            earningsDestination: 'parent',
            reason: 'Under 13 with parental consent - earnings go to parent account',
            userAge,
            isMinor: true,
            hasParentAccount: true,
            parentalConsentGiven: true
          };
        } else {
          eligibility = {
            canSeeReferralLink: false,
            canShare: false,
            canAccumulateEarnings: false,
            canRequestPayout: false,
            earningsDestination: 'none',
            reason: 'Under 13 - parent account with consent required to participate in Share & Earn',
            userAge,
            isMinor: true,
            hasParentAccount: hasParent,
            parentalConsentGiven: hasConsent
          };
        }
      } else if (isMinor) {
        // 13-17
        eligibility = {
          canSeeReferralLink: true,
          canShare: true,
          canAccumulateEarnings: true,
          canRequestPayout: false,
          earningsDestination: hasParent ? 'parent_or_held' : 'held_until_18',
          reason: 'Ages 13-17 - earnings are held until you turn 18, or your parent can claim them',
          userAge,
          isMinor: true,
          hasParentAccount: hasParent,
          parentalConsentGiven: hasConsent
        };
      } else {
        // 18+
        eligibility = {
          canSeeReferralLink: true,
          canShare: true,
          canAccumulateEarnings: true,
          canRequestPayout: true,
          earningsDestination: 'self',
          reason: 'Full access to Share & Earn',
          userAge,
          isMinor: false,
          hasParentAccount: hasParent,
          parentalConsentGiven: hasConsent
        };
      }

      // Get held earnings for minors
      if (isMinor) {
        const { data: heldData } = await supabase
          .from('minor_earnings_ledger')
          .select('amount')
          .eq('minor_user_id', user.id)
          .eq('status', 'held');
        
        if (heldData) {
          eligibility.heldEarnings = heldData.reduce((sum, row) => sum + Number(row.amount), 0);
        }
      }

      res.status(200).json({ success: true, eligibility });
      return;
    }

    // Use database function result
    const dbResult = eligibilityData?.[0];
    if (!dbResult) {
      res.status(500).json({ success: false, message: 'Failed to check eligibility' });
      return;
    }

    // Get additional user info for the response
    const { data: userData } = await supabase
      .from('users_with_age')
      .select('calculated_age, is_minor, parent_account_id, parental_consent_for_earnings')
      .eq('id', user.id)
      .single();

    const eligibility: EligibilityResult = {
      canSeeReferralLink: dbResult.can_see_referral_link,
      canShare: dbResult.can_share,
      canAccumulateEarnings: dbResult.can_accumulate_earnings,
      canRequestPayout: dbResult.can_request_payout,
      earningsDestination: dbResult.earnings_destination as EligibilityResult['earningsDestination'],
      reason: dbResult.reason,
      userAge: userData?.calculated_age || null,
      isMinor: userData?.is_minor || false,
      hasParentAccount: !!userData?.parent_account_id,
      parentalConsentGiven: !!userData?.parental_consent_for_earnings
    };

    // Get held earnings for minors
    if (eligibility.isMinor) {
      const { data: heldData } = await supabase
        .from('minor_earnings_ledger')
        .select('amount')
        .eq('minor_user_id', user.id)
        .eq('status', 'held');
      
      if (heldData) {
        eligibility.heldEarnings = heldData.reduce((sum, row) => sum + Number(row.amount), 0);
      }
    }

    res.status(200).json({ success: true, eligibility });

  } catch (error) {
    console.error('Eligibility API error:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
}

