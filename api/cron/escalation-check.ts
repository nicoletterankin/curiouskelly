/**
 * ESCALATION CHECK CRON
 * 
 * Checks for issues that need founder attention:
 * - Unresolved moderation issues > 48 hours
 * - Curriculum suggestions with high upvotes but no response
 * - Payment failures
 * - Bug reports
 * 
 * Only sends email if there are actual issues.
 * 
 * Schedule: 0 9 * * * (9 AM daily)
 */

import { createClient } from '@supabase/supabase-js';
import { sendFounderEmail, alertEmail } from '../../lib/notifications/founder-alerts';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export default async function handler(req: any, res: any) {
  const authHeader = req.headers.authorization;
  if (process.env.CRON_SECRET && authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  const issues: { title: string; detail: string; actionUrl?: string }[] = [];
  
  const fortyEightHoursAgo = new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString();
  
  // 1. Check for hidden/pending comments that are old (need moderation)
  const { data: pendingComments } = await supabase
    .from('phase_comments')
    .select('id, content, lesson_day, phase, created_at')
    .eq('moderation_status', 'pending')
    .lt('created_at', fortyEightHoursAgo)
    .limit(10);
  
  if (pendingComments && pendingComments.length > 0) {
    issues.push({
      title: `${pendingComments.length} comments pending moderation > 48h`,
      detail: `Oldest: Day ${pendingComments[0].lesson_day}, ${pendingComments[0].phase} phase`,
      actionUrl: 'https://curiouskelly.com/admin?tab=moderation'
    });
  }
  
  // 2. Check for high-voted suggestions without response
  const { data: hotSuggestions } = await supabase
    .from('curriculum_suggestions')
    .select('id, suggestion_type, content, lesson_day, phase, upvotes, created_at')
    .eq('status', 'open')
    .gte('upvotes', 5) // 5+ upvotes = community wants this
    .lt('created_at', fortyEightHoursAgo)
    .limit(10);
  
  if (hotSuggestions && hotSuggestions.length > 0) {
    for (const s of hotSuggestions.slice(0, 3)) {
      issues.push({
        title: `Popular suggestion: ${s.suggestion_type} (${s.upvotes} upvotes)`,
        detail: `Day ${s.lesson_day}: "${s.content.substring(0, 80)}..."`,
        actionUrl: `https://curiouskelly.com/admin?tab=suggestions&id=${s.id}`
      });
    }
  }
  
  // 3. Check for payment failures (from stripe events)
  const { data: paymentIssues } = await supabase
    .from('payment_events')
    .select('*')
    .eq('event_type', 'payment_failed')
    .eq('resolved', false)
    .limit(5);
  
  if (paymentIssues && paymentIssues.length > 0) {
    issues.push({
      title: `${paymentIssues.length} unresolved payment failure${paymentIssues.length > 1 ? 's' : ''}`,
      detail: 'Learners may need help with billing',
      actionUrl: 'https://dashboard.stripe.com/payments?status=failed'
    });
  }
  
  // 4. Check for bug reports
  const { data: bugReports } = await supabase
    .from('curriculum_suggestions')
    .select('id, content, lesson_day, phase, created_at')
    .eq('suggestion_type', 'bug')
    .eq('status', 'open')
    .limit(5);
  
  if (bugReports && bugReports.length > 0) {
    issues.push({
      title: `${bugReports.length} open bug report${bugReports.length > 1 ? 's' : ''}`,
      detail: `Most recent: Day ${bugReports[0].lesson_day} - "${bugReports[0].content.substring(0, 60)}..."`,
      actionUrl: 'https://curiouskelly.com/admin?tab=bugs'
    });
  }
  
  // Only send if there are issues
  if (issues.length === 0) {
    console.log('No escalations needed - platform running smoothly!');
    return res.status(200).json({ sent: false, reason: 'No issues found' });
  }
  
  // Determine severity
  const severity = issues.some(i => i.title.includes('payment')) ? 'critical' : 'warning';
  
  await sendFounderEmail(alertEmail({
    severity,
    headline: `${issues.length} item${issues.length > 1 ? 's' : ''} need${issues.length === 1 ? 's' : ''} attention`,
    issues,
    context: 'These issues were escalated because the community couldn\'t resolve them within 48 hours.'
  }));
  
  // Log escalation
  await supabase.from('founder_notifications').insert({
    type: 'escalation_digest',
    data: { issues_count: issues.length, severity },
    sent_at: new Date().toISOString()
  });

  return res.status(200).json({ 
    sent: true, 
    issues: issues.length,
    severity
  });
}
