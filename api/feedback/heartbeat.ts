/**
 * POST /api/feedback/heartbeat
 * Tracks time-on-phase (called every 10 seconds while watching)
 * Detects dropoff points
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!
);

// In-memory store for active sessions (use Redis in production)
const activeSessions = new Map<string, { start: number; lastBeat: number }>();

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
    const { lesson_day, phase, student_id, event } = req.body;
    
    if (!lesson_day || !phase || !event) {
      return res.status(400).json({ 
        error: 'Missing required fields: lesson_day, phase, event' 
      });
    }
    
    const sessionKey = `${student_id || 'anon'}-${lesson_day}-${phase}`;
    const now = Date.now();
    
    if (event === 'start') {
      // Start new session
      activeSessions.set(sessionKey, { start: now, lastBeat: now });
      
      return res.status(200).json({ 
        success: true,
        session_started: true,
        session_key: sessionKey
      });
    }
    
    if (event === 'watching') {
      // Update heartbeat
      const session = activeSessions.get(sessionKey);
      
      if (session) {
        session.lastBeat = now;
        const elapsed = Math.floor((now - session.start) / 1000);
        
        return res.status(200).json({ 
          success: true,
          elapsed_seconds: elapsed
        });
      } else {
        // Session not found, start new one
        activeSessions.set(sessionKey, { start: now, lastBeat: now });
        return res.status(200).json({ 
          success: true,
          session_restarted: true
        });
      }
    }
    
    if (event === 'end' || event === 'leave') {
      // End session, record total time
      const session = activeSessions.get(sessionKey);
      
      if (session) {
        const totalSeconds = Math.floor((now - session.start) / 1000);
        activeSessions.delete(sessionKey);
        
        // Update avg_time in lesson_impacts
        await supabase.rpc('update_avg_time', {
          p_lesson_day: parseInt(lesson_day),
          p_phase: phase,
          p_seconds: totalSeconds
        }).catch(() => {
          // Function may not exist yet, that's ok
        });
        
        // If they left early (< 30 seconds), record as dropout
        if (totalSeconds < 30 && event === 'leave') {
          await supabase
            .from('student_feedback')
            .insert({
              lesson_day: parseInt(lesson_day),
              phase,
              student_id: student_id || null,
              vote: null,
              completed: false,
              time_on_phase_seconds: totalSeconds,
              dropped_at_timestamp: new Date().toISOString(),
            });
        }
        
        return res.status(200).json({ 
          success: true,
          total_seconds: totalSeconds,
          session_ended: true
        });
      }
    }
    
    return res.status(200).json({ success: true });
    
  } catch (err) {
    console.error('Heartbeat handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
