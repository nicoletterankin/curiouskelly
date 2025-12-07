/**
 * Referral Link Tracking API
 * POST /api/referral/track
 * 
 * Records referral clicks with LIFETIME attribution.
 * Every click is tracked - attribution NEVER expires.
 * 
 * Philosophy: "You introduced them to Kelly. You deserve credit forever."
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import crypto from 'crypto';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface TrackRequest {
  referralCode: string;
  visitorFingerprint?: string;
  visitorEmail?: string;
  sourceUrl?: string;
  landingPage?: string;
  utmSource?: string;
  utmMedium?: string;
  utmCampaign?: string;
  utmContent?: string;
  utmTerm?: string;
}

interface TrackResponse {
  success: boolean;
  message: string;
  clickId?: string;
  referrerInfo?: {
    displayName?: string;
    commissionTier?: string;
  };
}

// Hash IP for privacy-respecting tracking
function hashIP(ip: string): string {
  return crypto.createHash('sha256').update(ip + 'kelly-salt-2025').digest('hex').substring(0, 32);
}

// Extract client IP from request
function getClientIP(req: VercelRequest): string {
  const forwardedFor = req.headers['x-forwarded-for'];
  if (typeof forwardedFor === 'string') {
    return forwardedFor.split(',')[0].trim();
  }
  if (Array.isArray(forwardedFor)) {
    return forwardedFor[0];
  }
  return req.socket?.remoteAddress || 'unknown';
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // CORS headers for client-side tracking
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ 
      success: false, 
      message: 'Method not allowed. Use POST.' 
    });
    return;
  }

  try {
    const body = req.body as TrackRequest;
    const { 
      referralCode,
      visitorFingerprint,
      visitorEmail,
      sourceUrl,
      landingPage,
      utmSource,
      utmMedium,
      utmCampaign,
      utmContent,
      utmTerm
    } = body;

    // Validate referral code
    if (!referralCode) {
      res.status(400).json({
        success: false,
        message: 'Referral code is required'
      });
      return;
    }

    // Validate code format (alphanumeric, underscores, 3-30 chars)
    const codePattern = /^[A-Za-z0-9_-]{3,30}$/;
    if (!codePattern.test(referralCode)) {
      res.status(400).json({
        success: false,
        message: 'Invalid referral code format'
      });
      return;
    }

    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Look up the referrer by referral code
    const { data: referrer, error: lookupError } = await supabase
      .from('users')
      .select('id, display_name, email, commission_tier, referral_code')
      .eq('referral_code', referralCode.toLowerCase())
      .single();

    if (lookupError || !referrer) {
      // Code not found - could be old link or typo
      // Log for analytics but don't create click record
      console.warn(`[Referral] Invalid code attempted: ${referralCode}`);
      
      res.status(404).json({
        success: false,
        message: 'Referral code not found'
      });
      return;
    }

    // Get and hash visitor IP for privacy
    const clientIP = getClientIP(req);
    const visitorIpHash = hashIP(clientIP);

    // Check for self-referral (if visitor email matches referrer email)
    if (visitorEmail && visitorEmail.toLowerCase() === referrer.email?.toLowerCase()) {
      res.status(400).json({
        success: false,
        message: 'Self-referral is not allowed'
      });
      return;
    }

    // Create the referral click record
    // LIFETIME ATTRIBUTION - attribution_expires_at is NULL (never expires)
    const { data: click, error: insertError } = await supabase
      .from('referral_clicks')
      .insert({
        referrer_id: referrer.id,
        referral_code: referralCode.toLowerCase(),
        visitor_fingerprint: visitorFingerprint || null,
        visitor_ip_hash: visitorIpHash,
        visitor_email: visitorEmail?.toLowerCase() || null,
        source_url: sourceUrl || null,
        landing_page: landingPage || null,
        utm_source: utmSource || null,
        utm_medium: utmMedium || null,
        utm_campaign: utmCampaign || null,
        utm_content: utmContent || null,
        utm_term: utmTerm || null,
        attribution_expires_at: null, // LIFETIME - never expires!
        clicked_at: new Date().toISOString()
      })
      .select('id')
      .single();

    if (insertError) {
      console.error('[Referral] Failed to record click:', insertError);
      res.status(500).json({
        success: false,
        message: 'Failed to record referral click'
      });
      return;
    }

    // Success response with referrer info (for display purposes)
    const response: TrackResponse = {
      success: true,
      message: 'Referral tracked successfully',
      clickId: click.id,
      referrerInfo: {
        displayName: referrer.display_name || undefined,
        commissionTier: referrer.commission_tier || undefined
      }
    };

    console.log(`[Referral] Click tracked: ${referralCode} -> ${click.id}`);

    res.status(200).json(response);

  } catch (error) {
    console.error('[Referral] Error:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
}

