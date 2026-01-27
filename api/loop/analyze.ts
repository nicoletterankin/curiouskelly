/**
 * POST /api/loop/analyze
 * Triggered daily - calculates all impact metrics, detects incoherencies, queues improvements
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
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Optional: Verify cron secret
  const cronSecret = req.headers['x-cron-secret'];
  if (process.env.CRON_SECRET && cronSecret !== process.env.CRON_SECRET) {
    // Allow without secret for now, but log warning
    console.warn('Missing or invalid cron secret');
  }
  
  const results = {
    impacts_updated: 0,
    incoherencies_detected: 0,
    improvements_queued: 0,
    errors: [] as string[],
  };
  
  try {
    // 1. Recalculate all impact metrics
    console.log('Step 1: Calculating impacts...');
    const { data: impactData, error: impactError } = await supabase
      .rpc('calculate_all_impacts');
    
    if (impactError) {
      results.errors.push(`Impact calculation: ${impactError.message}`);
    } else {
      results.impacts_updated = impactData?.[0]?.updated_count || 0;
    }
    
    // 2. Detect incoherencies
    console.log('Step 2: Detecting incoherencies...');
    const { data: incoherenceData, error: incoherenceError } = await supabase
      .rpc('detect_incoherencies');
    
    if (incoherenceError) {
      results.errors.push(`Incoherence detection: ${incoherenceError.message}`);
    } else {
      results.incoherencies_detected = incoherenceData?.[0]?.detected || 0;
    }
    
    // 3. Generate improvement suggestions
    console.log('Step 3: Generating improvements...');
    const { data: improvementData, error: improvementError } = await supabase
      .rpc('generate_improvements');
    
    if (improvementError) {
      results.errors.push(`Improvement generation: ${improvementError.message}`);
    } else {
      results.improvements_queued = improvementData?.[0]?.queued || 0;
    }
    
    // 4. Calculate effectiveness rankings
    console.log('Step 4: Updating rankings...');
    await supabase.rpc('update_effectiveness_rankings').catch((err) => {
      // Function may not exist yet
      console.log('Rankings function not available:', err.message);
    });
    
    // 5. Get summary stats
    const { data: summaryData } = await supabase
      .from('lesson_impacts')
      .select('lesson_day, phase, completion_rate, engagement_score')
      .order('engagement_score', { ascending: true })
      .limit(5);
    
    const { count: totalIncoherencies } = await supabase
      .from('incoherencies')
      .select('*', { count: 'exact', head: true })
      .eq('fixed', false);
    
    const { count: pendingImprovements } = await supabase
      .from('improvement_queue')
      .select('*', { count: 'exact', head: true })
      .eq('deployed', false);
    
    console.log('Analysis complete:', results);
    
    return res.status(200).json({
      success: true,
      timestamp: new Date().toISOString(),
      results,
      summary: {
        lowest_performing: summaryData,
        total_open_incoherencies: totalIncoherencies || 0,
        pending_improvements: pendingImprovements || 0,
      }
    });
    
  } catch (err) {
    console.error('Analyze handler error:', err);
    return res.status(500).json({ 
      error: 'Internal server error',
      results 
    });
  }
}
