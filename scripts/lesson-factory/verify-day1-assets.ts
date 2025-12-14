#!/usr/bin/env npx tsx
/**
 * 🔍 DAY 1 ASSET VERIFICATION
 * 
 * Confirms all 200 videos exist in Supabase for Day 1:
 * - 10 archetypes × 5 phases × 4 videos (1 main + 3 responses) = 200 videos
 * 
 * Usage:
 *   npx tsx scripts/lesson-factory/verify-day1-assets.ts
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const ARCHETYPES = [
  'explorer', 'rebel', 'scientist', 'architect', 'diplomat',
  'empath', 'macgyver', 'mystic', 'storyteller', 'survivor'
];

const PHASES = ['hook', 'fact1', 'fact2', 'fact3', 'wisdom'];

interface VideoCheck {
  archetype: string;
  phase: string;
  videoType: string;
  expectedPath: string;
  exists: boolean;
}

async function verifyDay1Assets(): Promise<void> {
  console.log('\n' + '═'.repeat(70));
  console.log('🔍 DAY 1 ASSET VERIFICATION');
  console.log('═'.repeat(70) + '\n');

  // Fetch all Day 1 videos from storage
  const { data: files, error } = await supabase.storage
    .from('kelly-videos')
    .list('day-001', { limit: 500 });

  if (error) {
    console.error('❌ Failed to list storage:', error.message);
    process.exit(1);
  }

  // Get files from subdirectories (archetypes)
  const allVideos: string[] = [];
  
  for (const item of files || []) {
    if (!item.name.includes('.')) {
      // It's a directory (archetype folder)
      const { data: archetypeFiles } = await supabase.storage
        .from('kelly-videos')
        .list(`day-001/${item.name}`, { limit: 100 });
      
      for (const file of archetypeFiles || []) {
        if (file.name.endsWith('.mp4')) {
          allVideos.push(`day-001/${item.name}/${file.name}`);
        }
      }
    }
  }

  console.log(`📁 Found ${allVideos.length} videos in storage\n`);

  // Build expected video list
  const checks: VideoCheck[] = [];
  
  for (const archetype of ARCHETYPES) {
    for (const phase of PHASES) {
      // Main video
      checks.push({
        archetype,
        phase,
        videoType: 'main',
        expectedPath: `day-001/${archetype}/${phase}.mp4`,
        exists: false
      });
      
      // Response videos
      for (const response of ['a', 'b', 'c']) {
        checks.push({
          archetype,
          phase,
          videoType: `response_${response}`,
          expectedPath: `day-001/${archetype}/${phase}_response_${response}.mp4`,
          exists: false
        });
      }
    }
  }

  // Check existence
  for (const check of checks) {
    check.exists = allVideos.includes(check.expectedPath);
  }

  // Generate report
  const missing: VideoCheck[] = checks.filter(c => !c.exists);
  const existing = checks.filter(c => c.exists);

  // Per-archetype summary
  console.log('📊 ARCHETYPE SUMMARY');
  console.log('─'.repeat(50));
  
  for (const archetype of ARCHETYPES) {
    const archetypeChecks = checks.filter(c => c.archetype === archetype);
    const archetypeExisting = archetypeChecks.filter(c => c.exists).length;
    const total = archetypeChecks.length;
    const status = archetypeExisting === total ? '✅' : (archetypeExisting > 0 ? '⚠️' : '❌');
    const displayName = archetype.charAt(0).toUpperCase() + archetype.slice(1);
    console.log(`  ${status} The ${displayName}: ${archetypeExisting}/${total} videos`);
  }

  console.log('\n' + '═'.repeat(70));
  console.log('📈 OVERALL STATUS');
  console.log('═'.repeat(70));
  console.log(`  Total expected: 200 videos`);
  console.log(`  Total found: ${existing.length} videos`);
  console.log(`  Missing: ${missing.length} videos`);
  
  if (missing.length === 0) {
    console.log('\n✅ DAY 1 IS COMPLETE! All 200 videos verified.\n');
  } else {
    console.log('\n⚠️ MISSING VIDEOS:');
    console.log('─'.repeat(50));
    
    // Group by archetype
    const missingByArchetype: Record<string, VideoCheck[]> = {};
    for (const m of missing) {
      if (!missingByArchetype[m.archetype]) {
        missingByArchetype[m.archetype] = [];
      }
      missingByArchetype[m.archetype].push(m);
    }
    
    for (const [arch, videos] of Object.entries(missingByArchetype)) {
      console.log(`\n  The ${arch.charAt(0).toUpperCase() + arch.slice(1)}:`);
      for (const v of videos.slice(0, 5)) {
        console.log(`    - ${v.phase}/${v.videoType}`);
      }
      if (videos.length > 5) {
        console.log(`    ... and ${videos.length - 5} more`);
      }
    }
    
    console.log('\n📝 To fix, run the affected archetypes:');
    const affectedArchetypes = Object.keys(missingByArchetype);
    for (const arch of affectedArchetypes) {
      const displayName = 'The ' + arch.charAt(0).toUpperCase() + arch.slice(1);
      console.log(`  npx tsx scripts/lesson-factory/unified-factory.ts --day 1 --archetype "${displayName}"`);
    }
  }
  
  console.log('');
}

verifyDay1Assets().catch(console.error);





