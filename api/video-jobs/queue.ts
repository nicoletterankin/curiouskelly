/**
 * Video Jobs Queue API
 * 
 * POST /api/video-jobs/queue
 * Body: { day_of_year, phase, age_category, engine, input_payload, priority? }
 * Creates a new job in 'queued' status
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin } from '../lib/supabase';
import type { QueueJobRequest, EngineType, Phase, AgeCategory } from '../../lib/engines/types';
import { ENGINE_TYPES } from '../../lib/engines';

const VALID_PHASES: Phase[] = ['hook', 'story', 'wonder', 'action', 'wisdom'];
const VALID_AGES: AgeCategory[] = ['toddler', 'child', 'teen', 'young_adult', 'adult', 'elder'];

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
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
    const body = req.body as QueueJobRequest;
    
    // Validate required fields
    const { day_of_year, phase, age_category, engine, input_payload, priority = 5 } = body;
    
    if (!day_of_year || !phase || !age_category || !engine || !input_payload) {
      return res.status(400).json({
        error: 'Missing required fields',
        required: ['day_of_year', 'phase', 'age_category', 'engine', 'input_payload'],
        received: { day_of_year, phase, age_category, engine, has_payload: !!input_payload },
      });
    }
    
    // Validate values
    if (day_of_year < 1 || day_of_year > 365) {
      return res.status(400).json({ error: 'day_of_year must be 1-365' });
    }
    
    if (!VALID_PHASES.includes(phase)) {
      return res.status(400).json({ error: `Invalid phase. Must be one of: ${VALID_PHASES.join(', ')}` });
    }
    
    if (!VALID_AGES.includes(age_category)) {
      return res.status(400).json({ error: `Invalid age_category. Must be one of: ${VALID_AGES.join(', ')}` });
    }
    
    if (!ENGINE_TYPES.includes(engine)) {
      return res.status(400).json({ error: `Invalid engine. Must be one of: ${ENGINE_TYPES.join(', ')}` });
    }
    
    const supabase = getSupabaseAdmin();
    
    // Check for existing job with same parameters
    const { data: existing } = await supabase
      .from('video_jobs')
      .select('id, status')
      .eq('day_of_year', day_of_year)
      .eq('phase', phase)
      .eq('age_category', age_category)
      .eq('engine', engine)
      .in('status', ['queued', 'submitted', 'processing'])
      .single();
    
    if (existing) {
      return res.status(409).json({
        error: 'Job already exists',
        existing_job: existing,
        hint: 'Use the existing job or wait for it to complete/fail',
      });
    }
    
    // Create new job
    const { data: job, error } = await supabase
      .from('video_jobs')
      .insert({
        day_of_year,
        phase,
        age_category,
        language: body.language || 'en',
        engine,
        status: 'queued',
        input_payload,
        priority,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .select()
      .single();
    
    if (error) {
      console.error('Queue job error:', error);
      return res.status(500).json({ error: 'Failed to create job', details: error.message });
    }
    
    return res.status(201).json({
      id: job.id,
      status: 'queued',
      message: 'Job queued successfully',
      job,
    });
    
  } catch (error) {
    console.error('Queue job error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
