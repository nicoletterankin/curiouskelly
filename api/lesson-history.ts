import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface LessonHistoryEntry {
  year: number;
  completedAt: string;
  answers: Record<string, string>;
  notes: string | null;
  ageAtCompletion: number | null;
  layer: string;
  viewNumber: number;
}

interface LessonHistoryResponse {
  hasSeenBefore: boolean;
  viewCount: number;
  history: LessonHistoryEntry[];
  recommendedLayer: 'foundation' | 'exploration' | 'mastery' | 'teaching';
  isBirthdayLesson: boolean;
  birthdayMessage?: string;
}

function getRecommendedLayer(
  viewCount: number,
  userAge: number | null
): 'foundation' | 'exploration' | 'mastery' | 'teaching' {
  const age = userAge || 18;
  
  // Teaching layer: 10+ views
  if (viewCount >= 10) return 'teaching';
  
  // Mastery layer: 5+ views OR 18+ and 3+ views
  if (viewCount >= 5 || (age >= 18 && viewCount >= 3)) return 'mastery';
  
  // Exploration layer: 2+ views OR 13+
  if (viewCount >= 2 || age >= 13) return 'exploration';
  
  // Foundation: default
  return 'foundation';
}

function isTodayUserBirthday(birthday: string | null): boolean {
  if (!birthday) return false;
  
  const today = new Date();
  const bday = new Date(birthday);
  
  return today.getMonth() === bday.getMonth() && today.getDate() === bday.getDate();
}

function getBirthdayLessonDay(birthday: string | null): number | null {
  if (!birthday) return null;
  
  const bday = new Date(birthday);
  const startOfYear = new Date(bday.getFullYear(), 0, 0);
  const diff = bday.getTime() - startOfYear.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Get lesson day from query
  const { day } = req.query;
  const lessonDay = parseInt(day as string, 10);
  
  if (isNaN(lessonDay) || lessonDay < 1 || lessonDay > 366) {
    return res.status(400).json({ error: 'Invalid lesson day' });
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
    
    // Get user profile
    const { data: profile } = await supabase
      .from('users')
      .select('birth_year, birthday, kelly_remembers, age')
      .eq('id', user.id)
      .single();
    
    // If kelly_remembers is false, return minimal response
    if (profile && profile.kelly_remembers === false) {
      return res.status(200).json({
        hasSeenBefore: false,
        viewCount: 0,
        history: [],
        recommendedLayer: 'foundation',
        isBirthdayLesson: false
      } as LessonHistoryResponse);
    }
    
    // Get lesson history for this user and lesson
    const { data: history, error: historyError } = await supabase
      .from('lesson_history')
      .select('*')
      .eq('user_id', user.id)
      .eq('lesson_day', lessonDay)
      .order('year_completed', { ascending: false });
    
    if (historyError) {
      console.error('Error fetching history:', historyError);
      return res.status(500).json({ error: 'Failed to fetch history' });
    }
    
    const viewCount = history?.length || 0;
    const userAge = profile?.age || (profile?.birth_year ? new Date().getFullYear() - profile.birth_year : null);
    
    // Check if this is the user's birthday lesson
    const birthdayLessonDay = getBirthdayLessonDay(profile?.birthday);
    const isBirthdayLesson = birthdayLessonDay === lessonDay;
    const isActualBirthday = isTodayUserBirthday(profile?.birthday);
    
    // Format history entries
    const formattedHistory: LessonHistoryEntry[] = (history || []).map(h => ({
      year: h.year_completed,
      completedAt: h.completed_at,
      answers: h.answers || {},
      notes: h.notes,
      ageAtCompletion: h.user_age_at_completion,
      layer: h.layer,
      viewNumber: h.view_number
    }));
    
    const response: LessonHistoryResponse = {
      hasSeenBefore: viewCount > 0,
      viewCount,
      history: formattedHistory,
      recommendedLayer: getRecommendedLayer(viewCount, userAge),
      isBirthdayLesson
    };
    
    // Add birthday message if applicable
    if (isBirthdayLesson && isActualBirthday) {
      response.birthdayMessage = viewCount > 1 
        ? `Happy birthday! This is your lesson. You've learned it ${viewCount} times now.`
        : `Happy birthday! This lesson is yours. It always will be.`;
    }
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Error in lesson-history:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}



