/**
 * API Endpoint: /api/commons-slop-report
 * 
 * Returns content quality issues for the Commons dashboard.
 * Public endpoint showing transparent quality monitoring.
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || '';

export default async function handler(req: any, res: any) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

    // Get unresolved issues grouped by severity (public only)
    const { data: issues, error: issuesError } = await supabase
      .from('content_validation_results')
      .select('*')
      .is('resolved_at', null)
      .eq('is_public', true)
      .order('severity', { ascending: true })
      .order('detected_at', { ascending: false })
      .limit(100);

    if (issuesError) {
      console.error('Error fetching issues:', issuesError);
      return res.status(500).json({ error: 'Failed to fetch issues' });
    }

    // Get summary stats
    const { data: criticalCount } = await supabase
      .from('content_validation_results')
      .select('id', { count: 'exact', head: true })
      .is('resolved_at', null)
      .eq('severity', 'critical');

    const { data: warningCount } = await supabase
      .from('content_validation_results')
      .select('id', { count: 'exact', head: true })
      .is('resolved_at', null)
      .eq('severity', 'warning');

    const { data: infoCount } = await supabase
      .from('content_validation_results')
      .select('id', { count: 'exact', head: true })
      .is('resolved_at', null)
      .eq('severity', 'info');

    // Get resolution stats for the past 7 days
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
    
    const { data: recentResolved } = await supabase
      .from('content_validation_results')
      .select('id', { count: 'exact', head: true })
      .gte('resolved_at', sevenDaysAgo);

    // Group issues by type
    const issuesByType: Record<string, number> = {};
    for (const issue of issues || []) {
      issuesByType[issue.issue_type] = (issuesByType[issue.issue_type] || 0) + 1;
    }

    const stats = {
      critical: criticalCount || 0,
      warning: warningCount || 0,
      info: infoCount || 0,
      total: (criticalCount || 0) + (warningCount || 0) + (infoCount || 0),
      resolvedThisWeek: recentResolved || 0,
      byType: issuesByType
    };

    return res.status(200).json({
      success: true,
      issues: issues || [],
      stats,
      lastAudit: issues?.[0]?.detected_at || null
    });

  } catch (error) {
    console.error('API Error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}



