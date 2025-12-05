import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface AnswerDistribution {
  [answer: string]: number;
}

interface YearData {
  [questionId: string]: AnswerDistribution;
}

interface CommonsResponse {
  currentYear: YearData;
  historical: {
    [year: number]: YearData;
  };
  userVsCommons?: {
    [questionId: string]: {
      userAnswer: string;
      popularAnswer: string;
      userPercentile: number;
    };
  };
  totalResponses: number;
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
  
  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Server configuration error' });
  }
  
  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  
  try {
    const currentYear = new Date().getFullYear();
    
    // Get all commons answers for this lesson
    const { data: answers, error: fetchError } = await supabase
      .from('commons_answers')
      .select('*')
      .eq('lesson_day', lessonDay)
      .order('year', { ascending: false });
    
    if (fetchError) {
      console.error('Error fetching commons:', fetchError);
      return res.status(500).json({ error: 'Failed to fetch commons data' });
    }
    
    // Organize by year
    const historical: { [year: number]: YearData } = {};
    let totalResponses = 0;
    
    for (const answer of (answers || [])) {
      const year = answer.year;
      
      if (!historical[year]) {
        historical[year] = {};
      }
      
      if (!historical[year][answer.question_id]) {
        historical[year][answer.question_id] = {};
      }
      
      historical[year][answer.question_id][answer.answer_value] = answer.count;
      totalResponses += answer.count;
    }
    
    // Calculate percentages for current year
    const currentYearData = historical[currentYear] || {};
    for (const questionId of Object.keys(currentYearData)) {
      const total = Object.values(currentYearData[questionId]).reduce((a, b) => a + b, 0);
      if (total > 0) {
        for (const answer of Object.keys(currentYearData[questionId])) {
          currentYearData[questionId][answer] = Math.round((currentYearData[questionId][answer] / total) * 100);
        }
      }
    }
    
    // Check for user's answers if authenticated
    let userVsCommons: CommonsResponse['userVsCommons'] | undefined;
    
    const authHeader = req.headers.authorization;
    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      const { data: { user } } = await supabase.auth.getUser(token);
      
      if (user) {
        // Get user's most recent answers for this lesson
        const { data: userHistory } = await supabase
          .from('lesson_history')
          .select('answers')
          .eq('user_id', user.id)
          .eq('lesson_day', lessonDay)
          .order('completed_at', { ascending: false })
          .limit(1)
          .single();
        
        if (userHistory?.answers) {
          userVsCommons = {};
          
          for (const [questionId, userAnswer] of Object.entries(userHistory.answers as Record<string, string>)) {
            const questionData = currentYearData[questionId];
            if (questionData) {
              // Find most popular answer
              const popularAnswer = Object.entries(questionData).reduce(
                (max, [answer, count]) => (count as number) > (max.count as number) ? { answer, count } : max,
                { answer: '', count: 0 }
              ).answer;
              
              // Calculate user's percentile
              const userAnswerPercentage = (questionData[userAnswer] as number) || 0;
              
              userVsCommons[questionId] = {
                userAnswer,
                popularAnswer,
                userPercentile: userAnswerPercentage
              };
            }
          }
        }
      }
    }
    
    const response: CommonsResponse = {
      currentYear: currentYearData,
      historical,
      totalResponses
    };
    
    if (userVsCommons) {
      response.userVsCommons = userVsCommons;
    }
    
    return res.status(200).json(response);
    
  } catch (error) {
    console.error('Error in commons:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

