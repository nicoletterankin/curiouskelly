/**
 * Kelly Generation Jobs API
 * 
 * Tracks and manages batch generation jobs.
 * 
 * GET /api/kelly-jobs - List all jobs
 * GET /api/kelly-jobs?id=xxx - Get specific job
 * POST /api/kelly-jobs - Create new job
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method === 'GET') {
    const { id } = req.query;
    
    if (id) {
      // Get specific job
      const { data, error } = await supabase
        .from('kelly_generation_jobs')
        .select('*')
        .eq('id', id)
        .single();
      
      if (error) {
        return res.status(404).json({ error: 'Job not found' });
      }
      
      return res.status(200).json(data);
    }
    
    // List all jobs
    const { data: jobs, error } = await supabase
      .from('kelly_generation_jobs')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(50);
    
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    
    // Calculate totals
    const running = jobs?.filter(j => j.status === 'running').length || 0;
    const completed = jobs?.filter(j => j.status === 'completed').length || 0;
    const totalCost = jobs?.reduce((sum, j) => sum + (j.actual_cost_usd || 0), 0) || 0;
    
    return res.status(200).json({
      summary: {
        total: jobs?.length || 0,
        running,
        completed,
        total_cost_usd: totalCost.toFixed(2)
      },
      jobs
    });
  }
  
  if (req.method === 'POST') {
    const { job_type, day_start, day_end, quality_tier = 'standard' } = req.body;
    
    if (!job_type || !day_start || !day_end) {
      return res.status(400).json({
        error: 'Missing required fields',
        required: ['job_type', 'day_start', 'day_end']
      });
    }
    
    // Estimate items and cost
    const days = day_end - day_start + 1;
    const phases = 5;
    let total_items = days * phases;
    let estimated_cost = 0;
    
    switch (job_type) {
      case 'image_batch':
        estimated_cost = total_items * 0.003;
        break;
      case 'animation_batch':
        estimated_cost = total_items * 0.05;
        break;
      case 'audio_batch':
        total_items = days * phases * 12; // archetypes
        estimated_cost = total_items * 0.002;
        break;
      case 'lipsync_batch':
        total_items = days * phases * 12;
        estimated_cost = total_items * 0.02;
        break;
    }
    
    const { data, error } = await supabase
      .from('kelly_generation_jobs')
      .insert({
        job_type,
        day_start,
        day_end,
        status: 'pending',
        total_items,
        estimated_cost_usd: estimated_cost,
        quality_tier
      })
      .select()
      .single();
    
    if (error) {
      return res.status(500).json({ error: error.message });
    }
    
    return res.status(201).json({
      message: 'Job created',
      job: data,
      estimated: {
        items: total_items,
        cost_usd: estimated_cost.toFixed(2)
      }
    });
  }
  
  return res.status(405).json({ error: 'Method not allowed' });
}



