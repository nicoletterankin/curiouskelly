/**
 * WEEKLY DIGEST CRON
 * 
 * Comprehensive weekly summary email every Sunday at 6 PM.
 * The one email you actually want to read each week.
 * 
 * Schedule: 0 18 * * 0 (Sunday 6 PM)
 * 
 * ZERO TRUST: Auth verified, rate limited, circuit breaker enabled
 */

import { createClient } from '@supabase/supabase-js';
import { sendFounderEmail, weeklyDigestEmail } from '../../lib/notifications/founder-alerts';
import {
  verifyCronAuth,
  checkEmailRateLimit,
  checkCircuit,
  recordSuccess,
  recordFailure,
  logAudit
} from '../../lib/security/zero-trust';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const CRON_NAME = 'weekly-digest';

function getWeekNumber(date: Date): number {
  const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
  const pastDaysOfYear = (date.getTime() - firstDayOfYear.getTime()) / 86400000;
  return Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
}

export default async function handler(req: any, res: any) {
  // Zero Trust: Verify authentication
  const auth = verifyCronAuth(req);
  if (!auth.authorized) {
    return res.status(401).json({ error: 'Unauthorized', reason: auth.reason });
  }
  
  // Zero Trust: Check circuit breaker
  if (!checkCircuit(CRON_NAME)) {
    return res.status(503).json({ error: 'Circuit open', message: 'Too many recent failures' });
  }
  
  // Zero Trust: Rate limit emails
  if (!checkEmailRateLimit()) {
    return res.status(429).json({ error: 'Rate limited', message: 'Too many emails sent recently' });
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  
  const now = new Date();
  const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  const weekNumber = getWeekNumber(now);
  
  // Gather all stats
  const [
    { count: newLearners },
    { count: lessonsCompleted },
    { count: commentsPosted },
    { count: suggestionsReceived },
    { count: issuesResolved },
    { count: issuesPending }
  ] = await Promise.all([
    supabase.from('profiles').select('*', { count: 'exact', head: true })
      .gte('created_at', weekAgo.toISOString()),
    supabase.from('lesson_completions').select('*', { count: 'exact', head: true })
      .gte('completed_at', weekAgo.toISOString()),
    supabase.from('phase_comments').select('*', { count: 'exact', head: true })
      .gte('created_at', weekAgo.toISOString()),
    supabase.from('curriculum_suggestions').select('*', { count: 'exact', head: true })
      .gte('created_at', weekAgo.toISOString()),
    supabase.from('curriculum_suggestions').select('*', { count: 'exact', head: true })
      .eq('status', 'resolved')
      .gte('updated_at', weekAgo.toISOString()),
    supabase.from('curriculum_suggestions').select('*', { count: 'exact', head: true })
      .eq('status', 'open')
  ]);
  
  // Get featured/helpful comments (top moments)
  const { data: featuredComments } = await supabase
    .from('phase_comments')
    .select('content, lesson_day, phase, upvotes')
    .eq('moderation_status', 'featured')
    .gte('created_at', weekAgo.toISOString())
    .order('upvotes', { ascending: false })
    .limit(3);
  
  // Get happy events
  const { data: happyEvents } = await supabase
    .from('happy_learner_events')
    .select('type, detail')
    .gte('created_at', weekAgo.toISOString())
    .order('created_at', { ascending: false });
  
  // Build top moments
  const topMoments: { title: string; detail: string }[] = [];
  
  // Count streaks
  const streakCounts = (happyEvents || []).reduce((acc, e) => {
    acc[e.type] = (acc[e.type] || 0) + 1;
    return acc;
  }, {} as { [key: string]: number });
  
  if (streakCounts.streak_100) {
    topMoments.push({
      title: `🏆 ${streakCounts.streak_100} learner${streakCounts.streak_100 > 1 ? 's' : ''} hit 100-day streak`,
      detail: 'True dedication to lifelong learning'
    });
  }
  
  if (streakCounts.streak_30) {
    topMoments.push({
      title: `🔥 ${streakCounts.streak_30} learner${streakCounts.streak_30 > 1 ? 's' : ''} hit 30-day streak`,
      detail: 'Habit formation in action'
    });
  }
  
  if (featuredComments && featuredComments.length > 0) {
    topMoments.push({
      title: `💬 ${featuredComments.length} featured comment${featuredComments.length > 1 ? 's' : ''} in Commons`,
      detail: `Best: "${featuredComments[0].content.substring(0, 60)}..."`
    });
  }
  
  // Build needs attention
  const needsAttention: { title: string; detail: string }[] = [];
  
  if ((issuesPending || 0) > 5) {
    needsAttention.push({
      title: `${issuesPending} open suggestions waiting`,
      detail: 'Consider reviewing in Mission Control'
    });
  }
  
  // Get any stuck moderation
  const { count: stuckModeration } = await supabase
    .from('phase_comments')
    .select('*', { count: 'exact', head: true })
    .eq('moderation_status', 'pending')
    .lt('created_at', new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString());
  
  if ((stuckModeration || 0) > 0) {
    needsAttention.push({
      title: `${stuckModeration} comments pending > 72 hours`,
      detail: 'May need manual review'
    });
  }
  
  // Send the digest
  await sendFounderEmail(weeklyDigestEmail({
    weekNumber,
    periodStart: weekAgo.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    periodEnd: now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    stats: {
      newLearners: newLearners || 0,
      lessonsCompleted: lessonsCompleted || 0,
      commentsPosted: commentsPosted || 0,
      suggestionsReceived: suggestionsReceived || 0,
      issuesResolved: issuesResolved || 0,
      issuesPending: issuesPending || 0
    },
    topMoments,
    needsAttention
  }));
  
  // Log
  await supabase.from('founder_notifications').insert({
    type: 'weekly_digest',
    data: { 
      week: weekNumber,
      stats: { newLearners, lessonsCompleted, commentsPosted }
    },
    sent_at: new Date().toISOString()
  });

  return res.status(200).json({ 
    sent: true, 
    week: weekNumber,
    newLearners,
    lessonsCompleted
  });
}
