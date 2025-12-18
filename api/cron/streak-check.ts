/**
 * STREAK CHECK CRON
 * 
 * Runs the streak checking function to log happy events for milestone streaks.
 * Runs daily at 11 PM after most learners have completed their daily lesson.
 * 
 * Schedule: 0 23 * * * (11 PM daily)
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export default async function handler(req: any, res: any) {
  const authHeader = req.headers.authorization;
  if (process.env.CRON_SECRET && authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  
  try {
    // Call the streak checking function
    const { error } = await supabase.rpc('check_and_log_streaks');
    
    if (error) {
      console.error('Streak check error:', error);
      return res.status(500).json({ success: false, error: error.message });
    }
    
    // Count how many streak events were logged today
    const today = new Date().toISOString().split('T')[0];
    const { count } = await supabase
      .from('happy_learner_events')
      .select('*', { count: 'exact', head: true })
      .in('type', ['streak_7', 'streak_30', 'streak_100', 'streak_365'])
      .gte('created_at', `${today}T00:00:00Z`);
    
    console.log(`✅ Streak check complete. ${count || 0} new streak milestones today.`);
    
    return res.status(200).json({ 
      success: true, 
      streaks_logged_today: count || 0
    });
  } catch (err) {
    console.error('Streak check failed:', err);
    return res.status(500).json({ success: false, error: String(err) });
  }
}
