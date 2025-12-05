import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface ReflectionEntry {
  year: number;
  age: number | null;
  answers: Record<string, string>;
  completedAt: string;
}

interface ReflectionResponse {
  canReflect: boolean;
  viewCount: number;
  timeline: ReflectionEntry[];
  insights: string[];
}

function generateInsights(timeline: ReflectionEntry[]): string[] {
  const insights: string[] = [];
  
  if (timeline.length < 2) return insights;
  
  // Compare first and most recent
  const oldest = timeline[timeline.length - 1];
  const newest = timeline[0];
  
  // Check for answer changes
  const questions = new Set([
    ...Object.keys(oldest.answers || {}),
    ...Object.keys(newest.answers || {})
  ]);
  
  for (const q of questions) {
    const oldAnswer = oldest.answers?.[q];
    const newAnswer = newest.answers?.[q];
    
    if (oldAnswer && newAnswer && oldAnswer !== newAnswer) {
      const yearSpan = newest.year - oldest.year;
      insights.push(
        `Your answer to ${q.toUpperCase()} changed from "${oldAnswer}" to "${newAnswer}" over ${yearSpan} year${yearSpan > 1 ? 's' : ''}.`
      );
    }
  }
  
  // Age-based insight
  if (oldest.age && newest.age && newest.age > oldest.age) {
    const ageDiff = newest.age - oldest.age;
    insights.push(
      `You first learned this at age ${oldest.age}. You're ${newest.age} now. ${ageDiff} years of growth.`
    );
  }
  
  // Consistency insight
  const allSameAnswers = Array.from(questions).every(q => {
    const answers = timeline.map(t => t.answers?.[q]).filter(Boolean);
    return new Set(answers).size === 1;
  });
  
  if (allSameAnswers && timeline.length >= 3) {
    insights.push(
      `Your perspective has been consistent across ${timeline.length} years. You know what you believe.`
    );
  }
  
  // Evolution insight
  if (insights.length === 0 && timeline.length >= 2) {
    insights.push(
      `You've explored this lesson ${timeline.length} times. Each time with fresh eyes.`
    );
  }
  
  return insights;
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
    
    // Check kelly_remembers
    const { data: profile } = await supabase
      .from('users')
      .select('kelly_remembers')
      .eq('id', user.id)
      .single();
    
    if (profile?.kelly_remembers === false) {
      return res.status(200).json({
        canReflect: false,
        viewCount: 0,
        timeline: [],
        insights: []
      } as ReflectionResponse);
    }
    
    // Get all history for this lesson
    const { data: history, error: historyError } = await supabase
      .from('lesson_history')
      .select('year_completed, user_age_at_completion, answers, completed_at')
      .eq('user_id', user.id)
      .eq('lesson_day', lessonDay)
      .order('year_completed', { ascending: false });
    
    if (historyError) {
      console.error('Error fetching history:', historyError);
      return res.status(500).json({ error: 'Failed to fetch history' });
    }
    
    const viewCount = history?.length || 0;
    const canReflect = viewCount >= 2;
    
    const timeline: ReflectionEntry[] = (history || []).map(h => ({
      year: h.year_completed,
      age: h.user_age_at_completion,
      answers: h.answers || {},
      completedAt: h.completed_at
    }));
    
    const insights = canReflect ? generateInsights(timeline) : [];
    
    return res.status(200).json({
      canReflect,
      viewCount,
      timeline,
      insights
    } as ReflectionResponse);
    
  } catch (error) {
    console.error('Error in reflection:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

