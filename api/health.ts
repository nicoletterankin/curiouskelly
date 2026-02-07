/**
 * Health Check Endpoint
 * 
 * GET /api/health
 * 
 * Checks:
 * - API is responding
 * - Database is connected
 * - Email service is configured
 * - Video providers (optional, with ?providers=true)
 * 
 * Last updated: 2026-02-01 - Sprint hardening
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { cors } from '../lib/cors';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const resendApiKey = process.env.RESEND_API_KEY;

// Package version from environment or fallback
const VERSION = process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 7) || '2026.02.01';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Public CORS for health checks
  if (!cors(req, res, { allowAllOrigins: true })) return;
  
  const checks: Record<string, { status: 'ok' | 'error' | 'degraded'; message?: string; latency_ms?: number }> = {};
  let allOk = true;
  let hasDegraded = false;

  // Check Supabase connection
  if (supabaseUrl && supabaseServiceKey) {
    try {
      const supabase = createClient(supabaseUrl, supabaseServiceKey);
      const { count, error } = await supabase
        .from('lessons')
        .select('*', { count: 'exact', head: true });
      
      if (error) {
        checks.database = { status: 'error', message: error.message };
        allOk = false;
      } else {
        checks.database = { status: 'ok', message: `${count} lessons` };
      }
    } catch (e) {
      checks.database = { status: 'error', message: 'Connection failed' };
      allOk = false;
    }
  } else {
    checks.database = { status: 'error', message: 'Not configured' };
    allOk = false;
  }

  // Check Resend configuration
  if (resendApiKey) {
    checks.email = { status: 'ok' };
  } else {
    checks.email = { status: 'error', message: 'Not configured' };
    allOk = false;
  }
  
  // Check Stripe configuration
  if (process.env.STRIPE_SECRET_KEY) {
    checks.stripe = { status: 'ok' };
  } else {
    checks.stripe = { status: 'error', message: 'Not configured' };
    // Stripe is critical for revenue
    allOk = false;
  }
  
  // Check ElevenLabs (for audio generation)
  if (process.env.ELEVENLABS_API_KEY) {
    checks.elevenlabs = { status: 'ok' };
  } else {
    checks.elevenlabs = { status: 'degraded', message: 'Not configured - audio generation disabled' };
    hasDegraded = true;
  }
  
  // Check video providers (optional, expensive so only when requested)
  if (req.query.providers === 'true') {
    try {
      const { getEngineStatus } = await import('../lib/engines');
      const engineStatus = await getEngineStatus();
      
      let availableCount = 0;
      for (const [name, status] of Object.entries(engineStatus)) {
        if (status.available) availableCount++;
      }
      
      if (availableCount === 0) {
        checks.video_providers = { status: 'error', message: 'No providers available' };
        allOk = false;
      } else if (availableCount < 2) {
        checks.video_providers = { status: 'degraded', message: `Only ${availableCount} provider available` };
        hasDegraded = true;
      } else {
        checks.video_providers = { status: 'ok', message: `${availableCount} providers available` };
      }
    } catch (e) {
      checks.video_providers = { status: 'error', message: 'Failed to check providers' };
    }
  }

  // Overall status
  let overallStatus: 'healthy' | 'degraded' | 'unhealthy';
  let httpStatus: number;
  
  if (!allOk) {
    overallStatus = 'unhealthy';
    httpStatus = 503;
  } else if (hasDegraded) {
    overallStatus = 'degraded';
    httpStatus = 200;
  } else {
    overallStatus = 'healthy';
    httpStatus = 200;
  }
  
  return res.status(httpStatus).json({
    status: overallStatus,
    timestamp: new Date().toISOString(),
    version: VERSION,
    checks,
  });
}
