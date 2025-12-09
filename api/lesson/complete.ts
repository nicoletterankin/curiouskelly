/**
 * Lesson Completion API
 * 
 * Records when a user completes a lesson and updates their streak.
 * 
 * POST /api/lesson/complete
 * Body: { day_number: number }
 * Auth: Bearer token (Supabase JWT)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Server configuration error' });
  }

  // Get user from auth header
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const token = authHeader.replace('Bearer ', '');
  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Verify the JWT and get user
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    
    if (authError || !user) {
      return res.status(401).json({ error: 'Invalid token' });
    }

    const { day_number } = req.body;
    
    if (!day_number || typeof day_number !== 'number' || day_number < 1 || day_number > 365) {
      return res.status(400).json({ error: 'Invalid day_number' });
    }

    // Get current user data
    const { data: userData, error: userError } = await supabase
      .from('users')
      .select('id, current_streak, longest_streak, total_lessons_completed, last_lesson_at')
      .eq('id', user.id)
      .single();

    if (userError || !userData) {
      return res.status(404).json({ error: 'User not found' });
    }

    const now = new Date();
    const lastLessonDate = userData.last_lesson_at ? new Date(userData.last_lesson_at) : null;
    
    // Calculate if this continues the streak
    let newStreak = userData.current_streak || 0;
    
    if (lastLessonDate) {
      const daysSinceLastLesson = Math.floor(
        (now.getTime() - lastLessonDate.getTime()) / (1000 * 60 * 60 * 24)
      );
      
      if (daysSinceLastLesson === 0) {
        // Same day - no streak change
      } else if (daysSinceLastLesson === 1) {
        // Next day - continue streak
        newStreak += 1;
      } else {
        // Gap - reset streak
        newStreak = 1;
      }
    } else {
      // First lesson ever
      newStreak = 1;
    }

    const newLongestStreak = Math.max(userData.longest_streak || 0, newStreak);
    const newTotalLessons = (userData.total_lessons_completed || 0) + 1;

    // Record completion
    const { error: completionError } = await supabase
      .from('lesson_completions')
      .upsert({
        user_id: user.id,
        day_number,
        completed_at: now.toISOString()
      }, {
        onConflict: 'user_id,day_number'
      });

    if (completionError) {
      console.error('Completion insert error:', completionError);
    }

    // Update user stats
    const { error: updateError } = await supabase
      .from('users')
      .update({
        current_streak: newStreak,
        longest_streak: newLongestStreak,
        total_lessons_completed: newTotalLessons,
        last_lesson_at: now.toISOString()
      })
      .eq('id', user.id);

    if (updateError) {
      console.error('User update error:', updateError);
      return res.status(500).json({ error: 'Failed to update progress' });
    }

    return res.status(200).json({
      success: true,
      streak: newStreak,
      longest_streak: newLongestStreak,
      total_lessons: newTotalLessons,
      day_number
    });

  } catch (error) {
    console.error('Lesson completion error:', error);
    return res.status(500).json({ error: 'Something went wrong' });
  }
}


