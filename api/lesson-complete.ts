import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface LessonCompleteRequest {
  lessonDay: number;
  answers?: Record<string, string>;
  notes?: string;
  timeSpentSeconds?: number;
  layer?: string;
}

interface Milestone {
  type: string;
  achievedAt: string;
  metadata?: Record<string, unknown>;
}

interface LessonCompleteResponse {
  success: boolean;
  viewNumber: number;
  isFirstTime: boolean;
  newMilestones: Milestone[];
  streak: {
    current: number;
    longest: number;
    isNewRecord: boolean;
  };
  yearProgress: {
    completed: number;
    remaining: number;
    percentComplete: number;
  };
}

// Milestone definitions
const STREAK_MILESTONES = [7, 30, 100, 365, 1000];
const LESSON_MILESTONES = [50, 100, 200, 365];

async function checkAndAwardMilestones(
  supabase: ReturnType<typeof createClient>,
  userId: string,
  currentStreak: number,
  uniqueLessons: number,
  yearsCompleted: number
): Promise<Milestone[]> {
  const newMilestones: Milestone[] = [];
  const now = new Date().toISOString();
  
  // Check streak milestones
  for (const streak of STREAK_MILESTONES) {
    if (currentStreak >= streak) {
      const milestoneType = `streak_${streak}`;
      const { data: existing } = await supabase
        .from('milestones')
        .select('id')
        .eq('user_id', userId)
        .eq('milestone_type', milestoneType)
        .single();
      
      if (!existing) {
        await supabase.from('milestones').insert({
          user_id: userId,
          milestone_type: milestoneType,
          metadata: { streak_count: currentStreak }
        });
        newMilestones.push({ type: milestoneType, achievedAt: now, metadata: { streak_count: currentStreak } });
      }
    }
  }
  
  // Check lesson count milestones
  for (const count of LESSON_MILESTONES) {
    if (uniqueLessons >= count) {
      const milestoneType = `lessons_${count}`;
      const { data: existing } = await supabase
        .from('milestones')
        .select('id')
        .eq('user_id', userId)
        .eq('milestone_type', milestoneType)
        .single();
      
      if (!existing) {
        await supabase.from('milestones').insert({
          user_id: userId,
          milestone_type: milestoneType,
          metadata: { lesson_count: uniqueLessons }
        });
        newMilestones.push({ type: milestoneType, achievedAt: now, metadata: { lesson_count: uniqueLessons } });
      }
    }
  }
  
  // Check year complete milestone
  if (uniqueLessons >= 365 && yearsCompleted >= 1) {
    const milestoneType = `year_complete_${yearsCompleted}`;
    const { data: existing } = await supabase
      .from('milestones')
      .select('id')
      .eq('user_id', userId)
      .eq('milestone_type', milestoneType)
      .single();
    
    if (!existing) {
      await supabase.from('milestones').insert({
        user_id: userId,
        milestone_type: milestoneType,
        metadata: { years_completed: yearsCompleted }
      });
      newMilestones.push({ type: milestoneType, achievedAt: now, metadata: { years_completed: yearsCompleted } });
    }
  }
  
  // Check first lesson milestone
  const { data: firstLessonMilestone } = await supabase
    .from('milestones')
    .select('id')
    .eq('user_id', userId)
    .eq('milestone_type', 'first_lesson')
    .single();
  
  if (!firstLessonMilestone) {
    await supabase.from('milestones').insert({
      user_id: userId,
      milestone_type: 'first_lesson'
    });
    newMilestones.push({ type: 'first_lesson', achievedAt: now });
  }
  
  return newMilestones;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Get auth token
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing authorization' });
  }
  
  const token = authHeader.substring(7);
  
  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Server configuration error' });
  }
  
  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  
  try {
    // Verify token and get user
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    
    if (authError || !user) {
      return res.status(401).json({ error: 'Invalid token' });
    }
    
    // Parse request body
    const { lessonDay, answers, notes, timeSpentSeconds, layer }: LessonCompleteRequest = req.body;
    
    if (!lessonDay || lessonDay < 1 || lessonDay > 366) {
      return res.status(400).json({ error: 'Invalid lesson day' });
    }
    
    // Get user profile
    const { data: profile } = await supabase
      .from('users')
      .select('birth_year, age, streak_days, longest_streak, total_lessons_completed, unique_lessons_completed, years_completed, last_lesson_date, first_lesson_at, kelly_remembers')
      .eq('id', user.id)
      .single();
    
    const currentYear = new Date().getFullYear();
    const today = new Date().toISOString().split('T')[0];
    const userAge = profile?.age || (profile?.birth_year ? currentYear - profile.birth_year : null);
    
    // Calculate view number for this lesson
    const { count: previousViews } = await supabase
      .from('lesson_history')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', user.id)
      .eq('lesson_day', lessonDay);
    
    const viewNumber = (previousViews || 0) + 1;
    const isFirstTime = viewNumber === 1;
    
    // Check if already completed this lesson this year
    const { data: existingThisYear } = await supabase
      .from('lesson_history')
      .select('id')
      .eq('user_id', user.id)
      .eq('lesson_day', lessonDay)
      .eq('year_completed', currentYear)
      .single();
    
    // Record lesson history (if kelly_remembers is true or not set)
    if (profile?.kelly_remembers !== false) {
      if (existingThisYear) {
        // Update existing entry for this year
        await supabase
          .from('lesson_history')
          .update({
            answers: answers || {},
            notes: notes || null,
            time_spent_seconds: timeSpentSeconds || 0,
            layer: layer || 'foundation',
            completed_at: new Date().toISOString()
          })
          .eq('id', existingThisYear.id);
      } else {
        // Insert new entry
        await supabase.from('lesson_history').insert({
          user_id: user.id,
          lesson_day: lessonDay,
          year_completed: currentYear,
          view_number: viewNumber,
          answers: answers || {},
          notes: notes || null,
          time_spent_seconds: timeSpentSeconds || 0,
          layer: layer || 'foundation',
          user_age_at_completion: userAge
        });
      }
      
      // Update commons aggregates
      if (answers) {
        for (const [questionId, answerValue] of Object.entries(answers)) {
          await supabase.rpc('increment_commons_answer', {
            p_lesson_day: lessonDay,
            p_question_id: questionId,
            p_answer_value: answerValue,
            p_year: currentYear
          });
        }
      }
    }
    
    // Calculate streak
    const lastLessonDate = profile?.last_lesson_date;
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().split('T')[0];
    
    let newStreak = profile?.streak_days || 0;
    
    if (lastLessonDate === today) {
      // Already did a lesson today, streak unchanged
    } else if (lastLessonDate === yesterdayStr) {
      // Continuing streak
      newStreak += 1;
    } else {
      // Streak broken or first lesson
      newStreak = 1;
    }
    
    const longestStreak = Math.max(profile?.longest_streak || 0, newStreak);
    const isNewRecord = newStreak > (profile?.longest_streak || 0);
    
    // Calculate unique lessons completed
    const { count: uniqueCount } = await supabase
      .from('lesson_history')
      .select('lesson_day', { count: 'exact', head: true })
      .eq('user_id', user.id);
    
    // Actually count distinct lessons
    const { data: distinctLessons } = await supabase
      .from('lesson_history')
      .select('lesson_day')
      .eq('user_id', user.id);
    
    const uniqueLessons = new Set(distinctLessons?.map(l => l.lesson_day)).size;
    
    // Check if year is complete (365 unique lessons)
    let yearsCompleted = profile?.years_completed || 0;
    if (uniqueLessons >= 365 && !existingThisYear) {
      // Check if this completion pushed them to a new year
      const prevUniqueLessons = uniqueLessons - (isFirstTime ? 1 : 0);
      if (prevUniqueLessons < 365) {
        yearsCompleted += 1;
      }
    }
    
    // Update user profile
    await supabase
      .from('users')
      .update({
        streak_days: newStreak,
        longest_streak: longestStreak,
        last_lesson_date: today,
        total_lessons_completed: (profile?.total_lessons_completed || 0) + 1,
        unique_lessons_completed: uniqueLessons,
        years_completed: yearsCompleted,
        first_lesson_at: profile?.first_lesson_at || new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('id', user.id);
    
    // Check and award milestones
    const newMilestones = await checkAndAwardMilestones(
      supabase,
      user.id,
      newStreak,
      uniqueLessons,
      yearsCompleted
    );
    
    const response: LessonCompleteResponse = {
      success: true,
      viewNumber,
      isFirstTime,
      newMilestones,
      streak: {
        current: newStreak,
        longest: longestStreak,
        isNewRecord
      },
      yearProgress: {
        completed: uniqueLessons,
        remaining: Math.max(0, 365 - uniqueLessons),
        percentComplete: Math.round((uniqueLessons / 365) * 100)
      }
    };
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Error in lesson-complete:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}


