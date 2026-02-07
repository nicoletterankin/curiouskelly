#!/usr/bin/env npx tsx
/**
 * Quick check of video assets status for Day 1-7
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function main() {
  console.log('Checking video assets for Day 1-7...\n');
  
  // Check kelly_video_assets
  const { data: videoAssets, count: videoCount, error: videoError } = await supabase
    .from('kelly_video_assets')
    .select('lesson_day, phase, status, quality_tier', { count: 'exact' })
    .gte('lesson_day', 1)
    .lte('lesson_day', 7);
  
  if (videoError) {
    console.log('kelly_video_assets table may not exist or error:', videoError.message);
  } else {
    console.log(`kelly_video_assets count: ${videoCount || 0}`);
    
    // Group by day
    const byDay: Record<number, number> = {};
    const byStatus: Record<string, number> = {};
    
    for (const v of videoAssets || []) {
      byDay[v.lesson_day] = (byDay[v.lesson_day] || 0) + 1;
      byStatus[v.status || 'unknown'] = (byStatus[v.status || 'unknown'] || 0) + 1;
    }
    
    console.log('\nVideos by Day:', JSON.stringify(byDay));
    console.log('Videos by Status:', JSON.stringify(byStatus));
  }
  
  // Check lesson_atoms with hd_video_url
  const { data: atoms, error: atomError } = await supabase
    .from('lesson_atoms')
    .select('id, phase, archetype, hd_video_url, core_lesson_id')
    .not('hd_video_url', 'is', null);
  
  // Get lesson day numbers
  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('id, day_number')
    .gte('day_number', 1)
    .lte('day_number', 7);
  
  const lessonIdToDay = new Map((lessons || []).map(l => [l.id, l.day_number]));
  const atomsWithVideo = (atoms || []).filter(a => {
    const day = lessonIdToDay.get(a.core_lesson_id);
    return day && day >= 1 && day <= 7;
  });
  
  console.log(`\nlesson_atoms with hd_video_url (Day 1-7): ${atomsWithVideo.length}`);
  
  // Count by day
  const atomsByDay: Record<number, number> = {};
  for (const a of atomsWithVideo) {
    const day = lessonIdToDay.get(a.core_lesson_id) || 0;
    atomsByDay[day] = (atomsByDay[day] || 0) + 1;
  }
  console.log('Atoms with video by Day:', JSON.stringify(atomsByDay));
  
  // Check kelly_lesson_assets
  const { data: assets, count: assetCount } = await supabase
    .from('kelly_lesson_assets')
    .select('day_number, phase, status, video_url', { count: 'exact' })
    .gte('day_number', 1)
    .lte('day_number', 7);
  
  const assetsWithVideo = (assets || []).filter(a => a.video_url);
  console.log(`\nkelly_lesson_assets total: ${assetCount || 0}`);
  console.log(`kelly_lesson_assets with video_url: ${assetsWithVideo.length}`);
  
  const assetsByStatus: Record<string, number> = {};
  for (const a of assets || []) {
    assetsByStatus[a.status || 'unknown'] = (assetsByStatus[a.status || 'unknown'] || 0) + 1;
  }
  console.log('Assets by Status:', JSON.stringify(assetsByStatus));
  
  console.log('\nDone.');
}

main().catch(console.error);
