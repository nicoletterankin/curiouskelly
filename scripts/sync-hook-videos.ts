#!/usr/bin/env npx tsx
/**
 * 🔄 SYNC HOOK VIDEOS FROM day1_results.json
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Parse .env file manually to get the CORRECT service role key (first occurrence)
const envContent = fs.readFileSync(path.join(process.cwd(), '.env'), 'utf-8');
const envLines = envContent.split('\n');
let serviceRoleKey = '';
for (const line of envLines) {
  if (line.startsWith('SUPABASE_SERVICE_ROLE_KEY=') && !serviceRoleKey) {
    serviceRoleKey = line.split('=')[1].trim();
    break;
  }
}

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: serviceRoleKey || process.env.SUPABASE_SERVICE_ROLE_KEY!,
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🔄 SYNC HOOK VIDEOS TO DATABASE                           ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  // Load day1_results.json (hook videos)
  const resultsPath = path.join(process.cwd(), 'generated-videos', 'heygen-production', 'day1_results.json');
  const hookVideos: Record<string, string> = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
  
  let inserted = 0;
  let skipped = 0;
  let failed = 0;
  
  for (const [archetype, url] of Object.entries(hookVideos)) {
    if (url === 'FAILED' || !url.startsWith('http')) {
      console.log(`⏭️ Skipping ${archetype} (failed/invalid)`);
      skipped++;
      continue;
    }
    
    // Normalize archetype name
    const template = archetype.toLowerCase().replace(/\s+/g, '_');
    
    // Extract storage path from URL
    const urlParts = url.split('/public/');
    const storagePath = urlParts.length > 1 ? urlParts[1] : `production/day_001/${template}_hook.mp4`;
    
    const record = {
      day_number: 1,
      phase: 'hook',
      template: template,
      age_bucket: 'adult',
      asset_type: 'video',
      language: 'en',
      public_url: url,
      storage_path: storagePath,
      status: 'validated',
    };
    
    // Check if already exists
    const { data: existing } = await supabase
      .from('kelly_video_assets')
      .select('id')
      .eq('day_number', record.day_number)
      .eq('phase', record.phase)
      .eq('template', record.template)
      .eq('age_bucket', record.age_bucket)
      .eq('asset_type', 'video')
      .eq('language', record.language)
      .limit(1);
    
    if (existing && existing.length > 0) {
      console.log(`⏭️ Already exists: hook/${template}`);
      skipped++;
      continue;
    }
    
    // Insert
    const { error } = await supabase.from('kelly_video_assets').insert(record);
    
    if (error) {
      console.log(`❌ Failed: hook/${template} - ${error.message}`);
      failed++;
    } else {
      console.log(`✅ Inserted: hook/${template}`);
      inserted++;
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log(`📊 SUMMARY: ${inserted} inserted, ${skipped} skipped, ${failed} failed`);
  console.log('═'.repeat(60));
}

main().catch(console.error);
