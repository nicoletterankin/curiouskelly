/**
 * GET /api/student/:id/progress
 * Returns completed days/phases, streak count, next recommended lesson
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!
);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
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
    const { id } = req.query;
    const studentId = id as string;
    
    if (!studentId) {
      return res.status(400).json({ error: 'Missing student ID' });
    }
    
    // Get completed lessons from feedback
    const { data: completedFeedback } = await supabase
      .from('student_feedback')
      .select('lesson_day, phase')
      .eq('student_id', studentId)
      .eq('completed', true)
      .order('lesson_day', { ascending: true });
    
    // Calculate completed days (all 5 phases completed)
    const phasesByDay = new Map<number, Set<string>>();
    completedFeedback?.forEach(f => {
      if (!phasesByDay.has(f.lesson_day)) {
        phasesByDay.set(f.lesson_day, new Set());
      }
      phasesByDay.get(f.lesson_day)!.add(f.phase);
    });
    
    const completedDays: number[] = [];
    const REQUIRED_PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom'];
    
    phasesByDay.forEach((phases, day) => {
      const hasAllPhases = REQUIRED_PHASES.every(p => phases.has(p));
      if (hasAllPhases) {
        completedDays.push(day);
      }
    });
    
    // Calculate streak
    const today = new Date();
    const startOfYear = new Date(today.getFullYear(), 0, 1);
    const dayOfYear = Math.ceil((today.getTime() - startOfYear.getTime()) / (1000 * 60 * 60 * 24));
    
    let streak = 0;
    const sortedDays = [...completedDays].sort((a, b) => b - a);
    
    for (let i = 0; i < sortedDays.length; i++) {
      const expectedDay = dayOfYear - i;
      if (sortedDays[i] === expectedDay) {
        streak++;
      } else {
        break;
      }
    }
    
    // Get next recommended lesson
    const lastCompleted = sortedDays[0] || 0;
    const nextDay = Math.min(lastCompleted + 1, 365);
    
    // Get topic for next lesson
    const { data: nextLesson } = await supabase
      .from('core_lessons')
      .select('topic, icon_emoji')
      .eq('day_number', nextDay)
      .single();
    
    // Get user stats
    const { data: user } = await supabase
      .from('users')
      .select('current_day, streak_days, subscription_tier')
      .eq('id', studentId)
      .single();
    
    return res.status(200).json({
      student_id: studentId,
      
      // Progress
      completed_days: completedDays.length,
      completed_days_list: completedDays,
      total_days: 365,
      progress_percent: Math.round((completedDays.length / 365) * 100 * 10) / 10,
      
      // Streak
      current_streak: streak,
      longest_streak: user?.streak_days || streak,
      
      // Next lesson
      next_lesson: {
        day: nextDay,
        topic: nextLesson?.topic || 'Unknown',
        icon: nextLesson?.icon_emoji || '📚',
      },
      
      // Subscription
      tier: user?.subscription_tier || 'free',
      
      // Phases by day (for partial progress)
      phases_completed: Object.fromEntries(phasesByDay),
    });
    
  } catch (err) {
    console.error('Progress handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
