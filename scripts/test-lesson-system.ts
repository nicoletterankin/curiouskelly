#!/usr/bin/env npx tsx
/**
 * 🧪 LESSON PLAYER SYSTEM TESTS
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Get correct service role key
const envContent = fs.readFileSync(path.join(process.cwd(), '.env'), 'utf-8');
let serviceRoleKey = '';
for (const line of envContent.split('\n')) {
  if (line.startsWith('SUPABASE_SERVICE_ROLE_KEY=') && !serviceRoleKey) {
    serviceRoleKey = line.split('=')[1].trim();
    break;
  }
}

const supabase = createClient(process.env.PUBLIC_SUPABASE_URL!, serviceRoleKey);

async function test() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🧪 LESSON PLAYER SYSTEM TESTS                             ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  let passed = 0;
  let failed = 0;

  // Test 1: DB connection
  console.log('TEST 1: Database Connectivity');
  const { count, error } = await supabase.from('kelly_video_assets').select('*', { count: 'exact', head: true });
  if (error) {
    console.log('  [FAIL]', error.message);
    failed++;
  } else {
    console.log('  [PASS] Connected. Total assets:', count);
    passed++;
  }

  // Test 2: Day 1 video count  
  console.log('\nTEST 2: Day 1 Video Count');
  const { data: d1videos } = await supabase.from('kelly_video_assets').select('id').eq('day_number', 1).eq('asset_type', 'video');
  const videoCount = d1videos?.length || 0;
  console.log('  [INFO] Day 1 videos:', videoCount);
  if (videoCount >= 40) {
    console.log('  [PASS] Sufficient videos synced');
    passed++;
  } else {
    console.log('  [WARN] Expected 40+, got', videoCount);
  }

  // Test 3: Phase coverage
  // NOTE: cliff maps to 'hook' dbName, outro maps to 'wisdom' dbName
  console.log('\nTEST 3: Phase Coverage');
  const phases = [
    { phase: 'hook', template: 'explorer', displayName: 'hook' },
    { phase: 'hook', template: 'strategist', displayName: 'cliff (via hook)' },
    { phase: 'q1', template: 'scientist', displayName: 'fact1 (q1)' },
    { phase: 'q2', template: 'architect', displayName: 'fact2 (q2)' },
    { phase: 'q3', template: 'macgyver', displayName: 'fact3 (q3)' },
    { phase: 'wisdom', template: 'mystic', displayName: 'wisdom' },
    { phase: 'wisdom', template: 'storyteller', displayName: 'outro (via wisdom)' }
  ];
  
  let covered = 0;
  for (const p of phases) {
    const { data } = await supabase.from('kelly_video_assets')
      .select('public_url')
      .eq('day_number', 1)
      .eq('phase', p.phase)
      .eq('template', p.template)
      .eq('age_bucket', 'adult')
      .eq('asset_type', 'video')
      .limit(1);
    
    if (data?.length) {
      console.log(`  [PASS] ${p.displayName || p.phase}/${p.template}`);
      covered++;
    } else {
      console.log(`  [MISS] ${p.displayName || p.phase}/${p.template} (TTS fallback)`);
    }
  }
  console.log(`  Coverage: ${covered}/7 phases`);
  if (covered >= 5) passed++;
  else failed++;

  // Test 4: Video URL accessibility
  console.log('\nTEST 4: Video URL Accessibility');
  const { data: testVideo } = await supabase.from('kelly_video_assets')
    .select('public_url')
    .eq('day_number', 1)
    .eq('phase', 'q1')
    .eq('template', 'scientist')
    .limit(1);
  
  if (testVideo?.length) {
    const url = testVideo[0].public_url;
    console.log('  [INFO] Testing:', url.substring(0, 70) + '...');
    try {
      const res = await fetch(url, { method: 'HEAD' });
      if (res.ok) {
        console.log('  [PASS] Video accessible, status:', res.status);
        passed++;
      } else {
        console.log('  [FAIL] Video returned status:', res.status);
        failed++;
      }
    } catch (e: any) {
      console.log('  [FAIL] Video not accessible:', e.message);
      failed++;
    }
  } else {
    console.log('  [FAIL] No test video found');
    failed++;
  }

  // Test 5: learn.html check
  console.log('\nTEST 5: learn.html Syntax');
  const learnHtml = fs.readFileSync('public/learn.html', 'utf-8');
  const checks = [
    { pattern: /PHASE_CONFIG/, name: 'PHASE_CONFIG present' },
    { pattern: /kelly_video_assets/, name: 'kelly_video_assets query' },
    { pattern: /getVideoUrl/, name: 'getVideoUrl function' },
    { pattern: /dbName:\s*['"]q1['"]/, name: 'q1 dbName mapping' },
    { pattern: /videoHasAudio/, name: 'Audio decision logic' }
  ];
  
  let syntaxPassed = 0;
  for (const c of checks) {
    if (c.pattern.test(learnHtml)) {
      console.log(`  [PASS] ${c.name}`);
      syntaxPassed++;
    } else {
      console.log(`  [WARN] Missing: ${c.name}`);
    }
  }
  if (syntaxPassed >= 4) passed++;
  else failed++;

  console.log('\n════════════════════════════════════════════════════════════');
  console.log(`🏁 TESTS COMPLETE: ${passed} passed, ${failed} failed`);
  console.log('════════════════════════════════════════════════════════════');
  
  process.exit(failed > 0 ? 1 : 0);
}

test().catch(console.error);
