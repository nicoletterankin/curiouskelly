#!/usr/bin/env npx tsx
/**
 * 📤 UPLOAD HD VIDEOS TO SUPABASE STORAGE
 * 
 * Uploads generated HD videos to Supabase Storage and updates lesson_atoms table
 * 
 * Usage:
 *   npx tsx scripts/upload-hd-videos-to-supabase.ts --day 1
 *   npx tsx scripts/upload-hd-videos-to-supabase.ts --from 1 --to 7
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  SUPABASE_URL: process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  SUPABASE_KEY: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  VIDEO_DIR: path.join(process.cwd(), 'generated-videos', 'golden-lesson-hd'),
  BUCKET_NAME: 'kelly-videos',
  ARCHETYPES: ['The Explorer', 'The Architect', 'The Diplomat', 'The Empath', 'The Rebel'],
  PHASES: ['Hook', 'Fact1', 'Fact2', 'Fact3', 'Wisdom'],
};

const supabase = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_KEY);

// =============================================================================
// UPLOAD FUNCTIONS
// =============================================================================

async function uploadVideo(dayNumber: number, archetype: string, phase: string): Promise<string | null> {
  const videoDir = path.join(
    CONFIG.VIDEO_DIR,
    `day_${dayNumber.toString().padStart(3, '0')}_${phase}_${archetype.replace(/ /g, '_')}`
  );
  
  const videoPath = path.join(videoDir, 'final_hd.mp4');
  
  if (!fs.existsSync(videoPath)) {
    console.log(`   ⚠️ Video not found: ${videoPath}`);
    return null;
  }
  
  const fileBuffer = fs.readFileSync(videoPath);
  const fileSize = (fileBuffer.length / 1024 / 1024).toFixed(2);
  
  // Storage path: day-001/explorer/hook.mp4
  const storagePath = `day-${dayNumber.toString().padStart(3, '0')}/${archetype.toLowerCase().replace(/the /g, '').replace(/ /g, '-')}/${phase.toLowerCase()}.mp4`;
  
  console.log(`   📤 Uploading ${fileSize} MB to ${storagePath}...`);
  
  const { data, error } = await supabase.storage
    .from(CONFIG.BUCKET_NAME)
    .upload(storagePath, fileBuffer, {
      contentType: 'video/mp4',
      upsert: true, // Overwrite if exists
    });
  
  if (error) {
    console.error(`   ❌ Upload failed:`, error.message);
    return null;
  }
  
  // Get public URL
  const { data: urlData } = supabase.storage
    .from(CONFIG.BUCKET_NAME)
    .getPublicUrl(storagePath);
  
  console.log(`   ✅ Uploaded: ${urlData.publicUrl}`);
  return urlData.publicUrl;
}

async function updateDatabase(dayNumber: number, archetype: string, phase: string, videoUrl: string): Promise<boolean> {
  // Find the lesson atom
  const { data: coreLessons, error: coreLessonError } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
  
  if (coreLessonError || !coreLessons) {
    console.error(`   ❌ Core lesson not found for day ${dayNumber}`);
    return false;
  }
  
  // Update the lesson atom
  const { error: updateError } = await supabase
    .from('lesson_atoms')
    .update({ hd_video_url: videoUrl })
    .eq('core_lesson_id', coreLessons.id)
    .eq('archetype', archetype)
    .eq('phase', phase);
  
  if (updateError) {
    console.error(`   ❌ Database update failed:`, updateError.message);
    return false;
  }
  
  console.log(`   ✅ Database updated`);
  return true;
}

async function uploadDayVideos(dayNumber: number): Promise<void> {
  console.log(`\n${'═'.repeat(72)}`);
  console.log(`  📅 DAY ${dayNumber}: Uploading HD Videos`);
  console.log('═'.repeat(72));
  
  let uploaded = 0;
  let failed = 0;
  
  for (const archetype of CONFIG.ARCHETYPES) {
    for (const phase of CONFIG.PHASES) {
      console.log(`\n[${uploaded + failed + 1}] ${archetype} - ${phase}`);
      
      const videoUrl = await uploadVideo(dayNumber, archetype, phase);
      
      if (videoUrl) {
        const success = await updateDatabase(dayNumber, archetype, phase, videoUrl);
        if (success) {
          uploaded++;
        } else {
          failed++;
        }
      } else {
        failed++;
      }
    }
  }
  
  console.log(`\n${'─'.repeat(72)}`);
  console.log(`✅ Uploaded: ${uploaded}`);
  console.log(`❌ Failed: ${failed}`);
  console.log('─'.repeat(72));
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  let dayNumber = 1;
  let startDay: number | undefined;
  let endDay: number | undefined;
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--day':
        dayNumber = parseInt(args[++i]);
        break;
      case '--from':
        startDay = parseInt(args[++i]);
        break;
      case '--to':
        endDay = parseInt(args[++i]);
        break;
      case '--help':
        console.log(`
📤 Upload HD Videos to Supabase Storage

Usage:
  npx tsx scripts/upload-hd-videos-to-supabase.ts [options]

Options:
  --day <number>    Upload videos for specific day (default: 1)
  --from <day>      Start day for batch upload
  --to <day>        End day for batch upload
  --help            Show this help

Examples:
  npx tsx scripts/upload-hd-videos-to-supabase.ts --day 1
  npx tsx scripts/upload-hd-videos-to-supabase.ts --from 1 --to 7
`);
        process.exit(0);
    }
  }
  
  // Batch mode
  if (startDay && endDay) {
    for (let day = startDay; day <= endDay; day++) {
      await uploadDayVideos(day);
    }
    return;
  }
  
  // Single day
  await uploadDayVideos(dayNumber);
}

main().catch(console.error);






