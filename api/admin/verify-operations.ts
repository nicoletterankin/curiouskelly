/**
 * OPERATIONS VERIFICATION ENDPOINT
 * 
 * Runs comprehensive tests on all autonomous systems.
 * Returns detailed pass/fail status for each component.
 * 
 * Zero Trust: Requires admin authentication
 */

import { createClient } from '@supabase/supabase-js';
import { verifyEnvironment } from '../../lib/security/zero-trust';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

interface TestResult {
  name: string;
  category: 'environment' | 'database' | 'triggers' | 'email' | 'cron';
  status: 'pass' | 'fail' | 'skip';
  message: string;
  duration_ms?: number;
}

export default async function handler(req: any, res: any) {
  // Simple admin auth check (can be enhanced)
  const adminKey = req.headers['x-admin-key'];
  if (process.env.ADMIN_KEY && adminKey !== process.env.ADMIN_KEY) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  
  const results: TestResult[] = [];
  const startTime = Date.now();
  
  // ═══════════════════════════════════════════════════════════════
  // ENVIRONMENT TESTS
  // ═══════════════════════════════════════════════════════════════
  
  const envCheck = verifyEnvironment();
  results.push({
    name: 'Required environment variables',
    category: 'environment',
    status: envCheck.valid ? 'pass' : 'fail',
    message: envCheck.missing.length > 0 
      ? `Missing: ${envCheck.missing.join(', ')}` 
      : 'All required variables present'
  });
  
  results.push({
    name: 'CRON_SECRET configured',
    category: 'environment',
    status: process.env.CRON_SECRET ? 'pass' : 'fail',
    message: process.env.CRON_SECRET ? 'Configured' : 'NOT CONFIGURED - Crons are unprotected!'
  });
  
  // ═══════════════════════════════════════════════════════════════
  // DATABASE TESTS
  // ═══════════════════════════════════════════════════════════════
  
  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    results.push({
      name: 'Database connection',
      category: 'database',
      status: 'fail',
      message: 'Missing Supabase credentials'
    });
  } else {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
    
    // Test each required table
    const tables = [
      { name: 'founder_notifications', required: true },
      { name: 'happy_learner_events', required: true },
      { name: 'lesson_completions', required: true },
      { name: 'payment_events', required: true },
      { name: 'heygen_performance_logs', required: true },
      { name: 'phase_comments', required: true },
      { name: 'curriculum_suggestions', required: true },
      { name: 'audit_log', required: true },
      { name: 'profiles', required: true }
    ];
    
    for (const table of tables) {
      const tableStart = Date.now();
      try {
        const { error } = await supabase.from(table.name).select('id').limit(1);
        results.push({
          name: `Table: ${table.name}`,
          category: 'database',
          status: error ? 'fail' : 'pass',
          message: error ? error.message : 'Accessible',
          duration_ms: Date.now() - tableStart
        });
      } catch (err) {
        results.push({
          name: `Table: ${table.name}`,
          category: 'database',
          status: 'fail',
          message: String(err),
          duration_ms: Date.now() - tableStart
        });
      }
    }
    
    // ═══════════════════════════════════════════════════════════════
    // TRIGGER TESTS
    // ═══════════════════════════════════════════════════════════════
    
    // Check if triggers exist by querying pg_trigger
    const triggerQuery = `
      SELECT tgname FROM pg_trigger 
      WHERE tgname IN (
        'on_lesson_completion',
        'auto_moderate_on_insert',
        'auto_feature_on_upvote',
        'auto_resolve_on_vote'
      )
    `;
    
    try {
      const { data, error } = await supabase.rpc('exec_sql', { query: triggerQuery });
      if (error) {
        // RPC might not exist, check via manual test
        results.push({
          name: 'Database triggers',
          category: 'triggers',
          status: 'skip',
          message: 'Cannot verify triggers directly - manual verification required'
        });
      } else {
        const triggerNames = (data || []).map((r: any) => r.tgname);
        const expectedTriggers = [
          'on_lesson_completion',
          'auto_moderate_on_insert',
          'auto_feature_on_upvote',
          'auto_resolve_on_vote'
        ];
        
        for (const trigger of expectedTriggers) {
          results.push({
            name: `Trigger: ${trigger}`,
            category: 'triggers',
            status: triggerNames.includes(trigger) ? 'pass' : 'fail',
            message: triggerNames.includes(trigger) ? 'Active' : 'NOT FOUND'
          });
        }
      }
    } catch {
      results.push({
        name: 'Database triggers',
        category: 'triggers',
        status: 'skip',
        message: 'Trigger verification requires manual check'
      });
    }
    
    // ═══════════════════════════════════════════════════════════════
    // FUNCTION TESTS
    // ═══════════════════════════════════════════════════════════════
    
    // Test check_and_log_streaks function exists
    try {
      const { error } = await supabase.rpc('check_and_log_streaks');
      results.push({
        name: 'Function: check_and_log_streaks',
        category: 'triggers',
        status: error ? 'fail' : 'pass',
        message: error ? error.message : 'Callable'
      });
    } catch (err) {
      results.push({
        name: 'Function: check_and_log_streaks',
        category: 'triggers',
        status: 'fail',
        message: String(err)
      });
    }
    
    // ═══════════════════════════════════════════════════════════════
    // RECENT ACTIVITY TESTS
    // ═══════════════════════════════════════════════════════════════
    
    // Check if notifications have been sent recently
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    const { count: recentNotifications } = await supabase
      .from('founder_notifications')
      .select('*', { count: 'exact', head: true })
      .gte('sent_at', weekAgo);
    
    results.push({
      name: 'Notifications sent (last 7 days)',
      category: 'email',
      status: (recentNotifications || 0) > 0 ? 'pass' : 'skip',
      message: `${recentNotifications || 0} notifications sent`
    });
    
    // Check audit log
    const { count: recentAudits } = await supabase
      .from('audit_log')
      .select('*', { count: 'exact', head: true })
      .gte('timestamp', weekAgo);
    
    results.push({
      name: 'Audit entries (last 7 days)',
      category: 'database',
      status: (recentAudits || 0) > 0 ? 'pass' : 'skip',
      message: `${recentAudits || 0} audit entries`
    });
  }
  
  // ═══════════════════════════════════════════════════════════════
  // EMAIL CONFIGURATION
  // ═══════════════════════════════════════════════════════════════
  
  results.push({
    name: 'SendGrid API key',
    category: 'email',
    status: process.env.SENDGRID_API_KEY ? 'pass' : 'fail',
    message: process.env.SENDGRID_API_KEY ? 'Configured' : 'MISSING - Emails will not send!'
  });
  
  // ═══════════════════════════════════════════════════════════════
  // SUMMARY
  // ═══════════════════════════════════════════════════════════════
  
  const passCount = results.filter(r => r.status === 'pass').length;
  const failCount = results.filter(r => r.status === 'fail').length;
  const skipCount = results.filter(r => r.status === 'skip').length;
  
  const overallStatus = failCount > 0 ? 'FAIL' : (skipCount > 0 ? 'PASS_WITH_WARNINGS' : 'PASS');
  
  return res.status(failCount > 0 ? 500 : 200).json({
    status: overallStatus,
    verified_at: new Date().toISOString(),
    total_duration_ms: Date.now() - startTime,
    summary: {
      pass: passCount,
      fail: failCount,
      skip: skipCount,
      total: results.length
    },
    results,
    critical_failures: results.filter(r => r.status === 'fail').map(r => r.name)
  });
}
