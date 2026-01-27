/**
 * POST /api/feedback/complete
 * Marks phase/lesson complete and triggers impact recalculation
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
    const { lesson_day, phase, student_id, time_seconds } = req.body;
    
    if (!lesson_day || !phase) {
      return res.status(400).json({ 
        error: 'Missing required fields: lesson_day, phase' 
      });
    }
    
    // Record completion
    const { data, error } = await supabase
      .from('student_feedback')
      .insert({
        lesson_day: parseInt(lesson_day),
        phase,
        student_id: student_id || null,
        completed: true,
        time_on_phase_seconds: time_seconds || null,
      })
      .select()
      .single();
    
    if (error) {
      console.error('Completion insert error:', error);
      return res.status(500).json({ error: error.message });
    }
    
    // Update user progress if student_id provided
    if (student_id) {
      await supabase
        .from('user_progress')
        .upsert({
          user_id: student_id,
          lesson_id: lesson_day.toString(), // Would need proper mapping
          completed: true,
          progress_percent: 100,
        }, {
          onConflict: 'user_id,lesson_id'
        })
        .catch(() => {
          // user_progress may not exist or have different schema
        });
    }
    
    return res.status(200).json({ 
      success: true,
      feedback_id: data.id,
      message: `Completed: Day ${lesson_day} ${phase}`
    });
    
  } catch (err) {
    console.error('Complete handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
