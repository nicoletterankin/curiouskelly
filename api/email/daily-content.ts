/**
 * Daily Email Content Extraction API
 * 
 * Extracts full lesson content for email delivery
 * Combines LEARN (core lesson) + GROW (AI fluency) tracks
 * 
 * GET /api/email/daily-content?day=17
 * Returns: Full email-ready content for both tracks
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_ANON_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '';

interface EmailContent {
  day: number;
  date: string;
  learn: {
    topic: string;
    headline: string;
    universal_truth: string;
    hook: string;
    facts: string[];
    wisdom: string;
    question: string;
  };
  grow: {
    topic: string;
    objective: string;
    content: string;
  };
  combined_wisdom: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    const dayParam = req.query.day;
    const dayNumber = dayParam ? parseInt(String(dayParam), 10) : getTodayDayNumber();
    
    if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
      res.status(400).json({ error: 'Invalid day number (1-365)' });
      return;
    }

    // Get LEARN track content from Supabase
    const learnContent = await getLearnContent(dayNumber);
    
    // Get GROW track content from curriculum files
    const growContent = await getGrowContent(dayNumber);
    
    // Get date for this day
    const date = getDayDate(dayNumber);
    
    const emailContent: EmailContent = {
      day: dayNumber,
      date,
      learn: learnContent,
      grow: growContent,
      combined_wisdom: generateCombinedWisdom(learnContent, growContent)
    };

    res.status(200).json(emailContent);
  } catch (error) {
    console.error('[email/daily-content] Error:', error);
    res.status(500).json({ error: 'Failed to extract email content' });
  }
}

async function getLearnContent(dayNumber: number) {
  const defaultContent = {
    topic: `Day ${dayNumber} Lesson`,
    headline: 'Today\'s learning adventure',
    universal_truth: 'Every day brings new knowledge.',
    hook: '',
    facts: [],
    wisdom: '',
    question: ''
  };

  if (!supabaseUrl || !supabaseKey) {
    return defaultContent;
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    // Get core lesson
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNumber)
      .single();

    if (!lesson) {
      return defaultContent;
    }

    // Get lesson atoms for full content
    const { data: atoms } = await supabase
      .from('lesson_atoms')
      .select('phase, content')
      .eq('core_lesson_id', lesson.id)
      .order('phase');

    const hook = atoms?.find(a => a.phase === 'Hook')?.content?.script || '';
    const facts = atoms
      ?.filter(a => a.phase?.startsWith('Fact'))
      ?.map(a => a.content?.script)
      ?.filter(Boolean) || [];
    const wisdom = atoms?.find(a => a.phase === 'Wisdom')?.content?.script || '';
    const cliff = atoms?.find(a => a.phase === 'Cliff')?.content;
    const question = cliff?.cliffPrompt || '';

    return {
      topic: lesson.topic || defaultContent.topic,
      headline: lesson.marketing_headline || lesson.topic || defaultContent.headline,
      universal_truth: lesson.universal_truth || defaultContent.universal_truth,
      hook,
      facts,
      wisdom,
      question
    };
  } catch (error) {
    console.error('[getLearnContent] Error:', error);
    return defaultContent;
  }
}

async function getGrowContent(dayNumber: number) {
  // Map day to month
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const months = ['january', 'february', 'march', 'april', 'may', 'june', 
                  'july', 'august', 'september', 'october', 'november', 'december'];
  
  let monthIndex = 0;
  let cumulative = 0;
  for (let i = 0; i < 12; i++) {
    cumulative += monthDays[i];
    if (dayNumber <= cumulative) {
      monthIndex = i;
      break;
    }
  }

  const defaultContent = {
    topic: 'Learning Skills',
    objective: 'Become a better learner today.',
    content: ''
  };

  try {
    // Try to fetch from curriculum files
    const monthKey = months[monthIndex];
    const curriculumUrl = `https://www.curiouskelly.com/data/curriculum/year2-ai-fluency/${monthKey}_curriculum.json`;
    
    const response = await fetch(curriculumUrl);
    if (!response.ok) {
      return defaultContent;
    }

    const curriculum = await response.json();
    const dayData = curriculum.days?.find((d: any) => d.day === dayNumber);

    if (!dayData) {
      return defaultContent;
    }

    return {
      topic: dayData.title || defaultContent.topic,
      objective: dayData.learning_objective || defaultContent.objective,
      content: dayData.content || ''
    };
  } catch (error) {
    console.error('[getGrowContent] Error:', error);
    return defaultContent;
  }
}

function getTodayDayNumber(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

function getDayDate(dayNumber: number): string {
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December'];
  
  let remaining = dayNumber;
  for (let i = 0; i < 12; i++) {
    if (remaining <= monthDays[i]) {
      return `${months[i]} ${remaining}`;
    }
    remaining -= monthDays[i];
  }
  return `December ${remaining}`;
}

function generateCombinedWisdom(learn: any, grow: any): string {
  if (learn.universal_truth && grow.objective) {
    return `Today you learned: ${learn.universal_truth} And you grew by understanding: ${grow.objective}`;
  }
  return learn.universal_truth || grow.objective || 'Stay curious!';
}
