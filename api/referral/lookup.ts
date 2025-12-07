/**
 * Referral Code Lookup API
 * GET /api/referral/lookup?code=XYZ
 * 
 * Validates a referral code and returns public referrer info.
 * Used by the share pages to show "Referred by [Name]" messaging.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface LookupResponse {
  valid: boolean;
  referrer?: {
    displayName?: string;
    tier?: string;
    lessonsCompleted?: number;
  };
  message?: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'GET') {
    res.status(405).json({ 
      valid: false, 
      message: 'Method not allowed. Use GET.' 
    });
    return;
  }

  try {
    const code = req.query.code as string;

    if (!code) {
      res.status(400).json({
        valid: false,
        message: 'Code parameter is required'
      });
      return;
    }

    // Validate code format
    const codePattern = /^[A-Za-z0-9_-]{3,30}$/;
    if (!codePattern.test(code)) {
      res.status(400).json({
        valid: false,
        message: 'Invalid code format'
      });
      return;
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Look up the referrer
    const { data: referrer, error } = await supabase
      .from('users')
      .select('display_name, commission_tier, total_lessons_completed')
      .eq('referral_code', code.toLowerCase())
      .single();

    if (error || !referrer) {
      res.status(200).json({
        valid: false,
        message: 'Referral code not found'
      });
      return;
    }

    // Return public info only
    const response: LookupResponse = {
      valid: true,
      referrer: {
        displayName: referrer.display_name || 'A Curious Learner',
        tier: referrer.commission_tier || 'new_learner',
        lessonsCompleted: referrer.total_lessons_completed || 0
      }
    };

    // Cache for 5 minutes
    res.setHeader('Cache-Control', 'public, max-age=300');
    res.status(200).json(response);

  } catch (error) {
    console.error('[Referral Lookup] Error:', error);
    res.status(500).json({
      valid: false,
      message: 'Internal server error'
    });
  }
}

