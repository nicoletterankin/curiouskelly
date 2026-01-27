/**
 * POST /api/feedback/vote
 * Records True/False vote and auto-updates lesson_impacts via trigger
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
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const { lesson_day, phase, student_id, vote } = req.body;
    
    // Validate
    if (!lesson_day || !phase || !vote) {
      return res.status(400).json({ 
        error: 'Missing required fields: lesson_day, phase, vote' 
      });
    }
    
    if (!['true', 'false', 'skip'].includes(vote)) {
      return res.status(400).json({ 
        error: 'Vote must be: true, false, or skip' 
      });
    }
    
    // Insert feedback (trigger auto-updates lesson_impacts)
    const { data, error } = await supabase
      .from('student_feedback')
      .insert({
        lesson_day: parseInt(lesson_day),
        phase,
        student_id: student_id || null,
        vote,
        completed: vote !== 'skip',
      })
      .select()
      .single();
    
    if (error) {
      console.error('Vote insert error:', error);
      return res.status(500).json({ error: error.message });
    }
    
    return res.status(200).json({ 
      success: true,
      feedback_id: data.id,
      message: `Vote recorded: ${vote} for Day ${lesson_day} ${phase}`
    });
    
  } catch (err) {
    console.error('Vote handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
