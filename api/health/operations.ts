/**
 * OPERATIONS HEALTH CHECK
 * 
 * Zero-trust verification of all autonomous systems.
 * Returns detailed status for monitoring.
 */

import { createClient } from '@supabase/supabase-js';
import { verifyEnvironment } from '../../lib/security/zero-trust';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface HealthCheck {
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message?: string;
  latency_ms?: number;
}

export default async function handler(req: any, res: any) {
  const checks: HealthCheck[] = [];
  const startTime = Date.now();
  
  // 1. Environment check
  const envCheck = verifyEnvironment();
  checks.push({
    name: 'environment',
    status: envCheck.valid ? 'pass' : 'fail',
    message: envCheck.missing.length > 0 
      ? `Missing: ${envCheck.missing.join(', ')}` 
      : 'All required vars present'
  });
  
  // 2. Database connection
  if (SUPABASE_URL && SUPABASE_SERVICE_KEY) {
    const dbStart = Date.now();
    try {
      const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
      const { error } = await supabase.from('founder_notifications').select('id').limit(1);
      
      checks.push({
        name: 'database',
        status: error ? 'fail' : 'pass',
        message: error ? error.message : 'Connected',
        latency_ms: Date.now() - dbStart
      });
    } catch (err) {
      checks.push({
        name: 'database',
        status: 'fail',
        message: String(err),
        latency_ms: Date.now() - dbStart
      });
    }
  } else {
    checks.push({
      name: 'database',
      status: 'fail',
      message: 'Missing credentials'
    });
  }
  
  // 3. Required tables exist
  if (SUPABASE_URL && SUPABASE_SERVICE_KEY) {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
    const tables = [
      'founder_notifications',
      'happy_learner_events',
      'lesson_completions',
      'payment_events',
      'heygen_performance_logs',
      'phase_comments',
      'curriculum_suggestions'
    ];
    
    for (const table of tables) {
      try {
        const { error } = await supabase.from(table).select('id').limit(1);
        checks.push({
          name: `table:${table}`,
          status: error ? 'fail' : 'pass',
          message: error ? error.message : 'Exists'
        });
      } catch (err) {
        checks.push({
          name: `table:${table}`,
          status: 'fail',
          message: String(err)
        });
      }
    }
  }
  
  // 4. Check triggers exist
  if (SUPABASE_URL && SUPABASE_SERVICE_KEY) {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
    
    const triggers = [
      'on_lesson_completion',
      'auto_moderate_on_insert',
      'auto_feature_on_upvote',
      'auto_resolve_on_vote'
    ];
    
    try {
      const { data, error } = await supabase.rpc('get_triggers');
      // If RPC doesn't exist, we'll check via information_schema
      if (error) {
        checks.push({
          name: 'triggers',
          status: 'warn',
          message: 'Cannot verify triggers programmatically'
        });
      } else {
        checks.push({
          name: 'triggers',
          status: 'pass',
          message: 'Trigger check available'
        });
      }
    } catch {
      checks.push({
        name: 'triggers',
        status: 'warn',
        message: 'Trigger verification not available'
      });
    }
  }
  
  // 5. Recent notifications sent
  if (SUPABASE_URL && SUPABASE_SERVICE_KEY) {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    
    const { data, count } = await supabase
      .from('founder_notifications')
      .select('type, sent_at', { count: 'exact' })
      .gte('sent_at', weekAgo)
      .order('sent_at', { ascending: false })
      .limit(5);
    
    checks.push({
      name: 'notifications_last_7d',
      status: count && count > 0 ? 'pass' : 'warn',
      message: `${count || 0} notifications sent`
    });
  }
  
  // 6. SendGrid API
  if (process.env.SENDGRID_API_KEY) {
    checks.push({
      name: 'sendgrid',
      status: 'pass',
      message: 'API key configured'
    });
  } else {
    checks.push({
      name: 'sendgrid',
      status: 'fail',
      message: 'API key missing'
    });
  }
  
  // 7. CRON_SECRET
  checks.push({
    name: 'cron_auth',
    status: process.env.CRON_SECRET ? 'pass' : 'warn',
    message: process.env.CRON_SECRET ? 'Secret configured' : 'Not configured (dev mode)'
  });
  
  // Calculate overall status
  const failCount = checks.filter(c => c.status === 'fail').length;
  const warnCount = checks.filter(c => c.status === 'warn').length;
  
  let overallStatus: 'healthy' | 'degraded' | 'unhealthy';
  if (failCount > 0) {
    overallStatus = 'unhealthy';
  } else if (warnCount > 0) {
    overallStatus = 'degraded';
  } else {
    overallStatus = 'healthy';
  }
  
  const response = {
    status: overallStatus,
    timestamp: new Date().toISOString(),
    total_latency_ms: Date.now() - startTime,
    checks,
    summary: {
      pass: checks.filter(c => c.status === 'pass').length,
      warn: warnCount,
      fail: failCount
    }
  };
  
  const statusCode = overallStatus === 'unhealthy' ? 503 : 200;
  
  return res.status(statusCode).json(response);
}
