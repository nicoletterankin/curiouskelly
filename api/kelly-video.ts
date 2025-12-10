/**
 * Kelly Video Delivery API
 * 
 * Serves Kelly video assets for lesson playback.
 * 
 * GET /api/kelly-video?day=1&phase=hook&type=video
 * GET /api/kelly-video?day=1&phase=hook&type=video&age=6-8&lang=es
 * GET /api/kelly-video/progress - Get production progress
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const { day, phase, type, age, lang, action } = req.query;
  
  // Progress endpoint
  if (action === 'progress') {
    return getProgress(res);
  }
  
  // Stats endpoint
  if (action === 'stats') {
    return getStats(res);
  }
  
  // Asset lookup
  if (!day || !phase) {
    return res.status(400).json({
      error: 'Missing required parameters',
      usage: '/api/kelly-video?day=1&phase=hook&type=video',
      parameters: {
        day: 'Lesson day (1-365)',
        phase: 'hook | q1 | q2 | q3 | wisdom',
        type: 'image | animation | video | audio (default: video)',
        age: 'Age bucket (e.g., 6-8)',
        lang: 'Language code (default: en)'
      }
    });
  }
  
  const dayNum = parseInt(day as string);
  const phaseStr = phase as string;
  const assetType = (type as string) || 'video';
  const ageBucket = age as string;
  const language = (lang as string) || 'en';
  
  try {
    let query = supabase
      .from('kelly_video_assets')
      .select('*')
      .eq('day_number', dayNum)
      .eq('phase', phaseStr)
      .eq('asset_type', assetType);
    
    // For variant-specific assets (audio, video)
    if (ageBucket && ['audio', 'video', 'video_4k'].includes(assetType)) {
      query = query.eq('age_bucket', ageBucket);
    }
    if (language && ['audio', 'video', 'video_4k'].includes(assetType)) {
      query = query.eq('language', language);
    }
    
    const { data, error } = await query.single();
    
    if (error || !data) {
      // Check what's available
      const { data: available } = await supabase
        .from('kelly_video_assets')
        .select('asset_type, age_bucket, language')
        .eq('day_number', dayNum)
        .eq('phase', phaseStr);
      
      return res.status(404).json({
        error: 'Asset not found',
        requested: { day: dayNum, phase: phaseStr, type: assetType, age: ageBucket, lang: language },
        available: available || [],
        hint: assetType === 'video' && available?.some(a => a.asset_type === 'image') 
          ? 'Image exists but video not yet generated. Use /api/kelly-generate to create.'
          : undefined
      });
    }
    
    return res.status(200).json({
      url: data.public_url,
      day: data.day_number,
      phase: data.phase,
      type: data.asset_type,
      template: data.template,
      resolution: data.resolution,
      duration: data.duration_seconds,
      quality: data.quality_tier,
      validated: data.face_audit_passed,
      created: data.created_at
    });
    
  } catch (error) {
    console.error('Kelly video API error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function getProgress(res: VercelResponse) {
  try {
    // Get counts by asset type
    const { data: counts } = await supabase
      .from('kelly_video_assets')
      .select('asset_type')
      .then(({ data }) => {
        const counts: Record<string, number> = {};
        data?.forEach(d => {
          counts[d.asset_type] = (counts[d.asset_type] || 0) + 1;
        });
        return { data: counts };
      });
    
    // Get validated count
    const { count: validated } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      .eq('face_audit_passed', true);
    
    // Get days with complete base assets
    const { data: progress } = await supabase
      .from('kelly_production_progress')
      .select('*');
    
    return res.status(200).json({
      assets: counts,
      validated,
      days: progress?.length || 0,
      progress: progress?.slice(0, 10) // First 10 days
    });
    
  } catch (error) {
    console.error('Progress API error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function getStats(res: VercelResponse) {
  try {
    const { count: totalAssets } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true });
    
    const { count: images } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      .eq('asset_type', 'image');
    
    const { count: animations } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      .eq('asset_type', 'animation');
    
    const { count: videos } = await supabase
      .from('kelly_video_assets')
      .select('*', { count: 'exact', head: true })
      .eq('asset_type', 'video');
    
    // Calculate progress percentages
    const totalDays = 365;
    const totalPhases = 5;
    const expectedImages = totalDays * totalPhases;
    const expectedAnimations = expectedImages;
    
    return res.status(200).json({
      total_assets: totalAssets,
      images: {
        count: images,
        expected: expectedImages,
        percent: ((images || 0) / expectedImages * 100).toFixed(1)
      },
      animations: {
        count: animations,
        expected: expectedAnimations,
        percent: ((animations || 0) / expectedAnimations * 100).toFixed(1)
      },
      videos: {
        count: videos
      },
      storage: {
        bucket: 'kelly-templates',
        paths: ['production/images', 'production/animations', 'production/videos']
      }
    });
    
  } catch (error) {
    console.error('Stats API error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}



