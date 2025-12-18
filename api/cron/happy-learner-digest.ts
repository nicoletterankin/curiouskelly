/**
 * HAPPY LEARNER DIGEST CRON
 * 
 * Daily email at 8 PM with positive moments from the platform.
 * Only sends if there are genuinely happy events to share.
 * 
 * Schedule: 0 20 * * * (8 PM daily)
 */

import { createClient } from '@supabase/supabase-js';
import { 
  sendFounderEmail, 
  celebrationEmail,
  getHappyEventsSince 
} from '../../lib/notifications/founder-alerts';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export default async function handler(req: any, res: any) {
  const authHeader = req.headers.authorization;
  if (process.env.CRON_SECRET && authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  
  // Get events from last 24 hours
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000);
  const events = await getHappyEventsSince(supabase, since);
  
  // Get today's stats
  const today = new Date().toISOString().split('T')[0];
  
  const [
    { count: newLearners },
    { count: lessonsCompleted },
    { count: commentsPosted }
  ] = await Promise.all([
    supabase.from('profiles').select('*', { count: 'exact', head: true })
      .gte('created_at', `${today}T00:00:00Z`),
    supabase.from('lesson_completions').select('*', { count: 'exact', head: true })
      .gte('completed_at', `${today}T00:00:00Z`),
    supabase.from('phase_comments').select('*', { count: 'exact', head: true })
      .gte('created_at', `${today}T00:00:00Z`)
  ]);

  // Count happy events by type
  const eventCounts = events.reduce((acc, e) => {
    acc[e.type] = (acc[e.type] || 0) + 1;
    return acc;
  }, {} as { [key: string]: number });

  // Only send if there's something to celebrate
  const totalHappy = events.length;
  const totalActivity = (newLearners || 0) + (lessonsCompleted || 0);
  
  if (totalHappy === 0 && totalActivity < 5) {
    console.log('No significant happy events today');
    return res.status(200).json({ sent: false, reason: 'No significant events' });
  }

  // Build highlights
  const highlights: { title: string; detail: string }[] = [];
  
  if (eventCounts.streak_30) {
    highlights.push({
      title: `🔥 ${eventCounts.streak_30} learner${eventCounts.streak_30 > 1 ? 's' : ''} hit 30-day streaks!`,
      detail: 'A full month of daily learning. That\'s habit formation.'
    });
  }
  
  if (eventCounts.streak_100) {
    highlights.push({
      title: `🏆 ${eventCounts.streak_100} learner${eventCounts.streak_100 > 1 ? 's' : ''} hit 100-day streaks!`,
      detail: 'This is life-changing dedication.'
    });
  }
  
  if (eventCounts.completed_track) {
    highlights.push({
      title: `🎓 ${eventCounts.completed_track} learner${eventCounts.completed_track > 1 ? 's' : ''} completed a full track!`,
      detail: '365 days of learning. A full year of growth.'
    });
  }
  
  if (eventCounts.first_lesson) {
    highlights.push({
      title: `👋 ${eventCounts.first_lesson} new learner${eventCounts.first_lesson > 1 ? 's' : ''} started their journey`,
      detail: 'Every expert was once a beginner.'
    });
  }
  
  if (eventCounts.helpful_comment) {
    highlights.push({
      title: `💬 ${eventCounts.helpful_comment} helpful comment${eventCounts.helpful_comment > 1 ? 's' : ''} in the Commons`,
      detail: 'Learners helping learners. Community working.'
    });
  }

  // Send the email
  await sendFounderEmail(celebrationEmail({
    headline: totalHappy > 10 ? 'Big Day for Curious Kelly!' : 'Today\'s Happy Moments',
    subhead: `${new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' })}`,
    stats: [
      { label: 'New Learners', value: newLearners || 0 },
      { label: 'Lessons Completed', value: lessonsCompleted || 0 },
      { label: 'Happy Events', value: totalHappy },
      { label: 'Comments', value: commentsPosted || 0 }
    ],
    highlights
  }));

  // Log that we sent
  await supabase.from('founder_notifications').insert({
    type: 'happy_digest',
    data: { events_count: totalHappy, stats: { newLearners, lessonsCompleted, commentsPosted } },
    sent_at: new Date().toISOString()
  });

  return res.status(200).json({ 
    sent: true, 
    events: totalHappy,
    highlights: highlights.length
  });
}
