/**
 * Health Check Endpoint
 * 
 * GET /api/health
 * 
 * Checks:
 * - API is responding
 * - Database is connected
 * - Email service is configured
 * 
 * Last updated: 2025-12-20 - Force redeploy
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const resendApiKey = process.env.RESEND_API_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const checks: Record<string, { status: 'ok' | 'error'; message?: string }> = {};
  let allOk = true;

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

  // Overall status
  const status = allOk ? 200 : 503;
  
  return res.status(status).json({
    status: allOk ? 'healthy' : 'degraded',
    timestamp: new Date().toISOString(),
    checks
  });
}
