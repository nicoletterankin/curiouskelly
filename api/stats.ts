/**
 * Stats Endpoint (Protected)
 * 
 * GET /api/stats
 * 
 * Returns key metrics for monitoring.
 * Requires CRON_SECRET for access.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Require auth
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Total users
    const { count: totalUsers } = await supabase
      .from('users')
      .select('*', { count: 'exact', head: true });

    // Subscribed users
    const { count: subscribedUsers } = await supabase
      .from('users')
      .select('*', { count: 'exact', head: true })
      .eq('email_daily_lesson', true)
      .is('email_unsubscribed_at', null);

    // Total lessons
    const { count: totalLessons } = await supabase
      .from('lessons')
      .select('*', { count: 'exact', head: true });

    // Completions today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const { count: completionsToday } = await supabase
      .from('lesson_completions')
      .select('*', { count: 'exact', head: true })
      .gte('completed_at', today.toISOString());

    // Top streaks
    const { data: topStreaks } = await supabase
      .from('users')
      .select('display_name, current_streak')
      .order('current_streak', { ascending: false })
      .limit(5);

    return res.status(200).json({
      timestamp: new Date().toISOString(),
      users: {
        total: totalUsers || 0,
        subscribed: subscribedUsers || 0
      },
      lessons: {
        total: totalLessons || 0
      },
      activity: {
        completionsToday: completionsToday || 0
      },
      leaderboard: topStreaks || []
    });

  } catch (error) {
    console.error('Stats error:', error);
    return res.status(500).json({ error: 'Failed to get stats' });
  }
}

