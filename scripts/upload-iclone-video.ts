#!/usr/bin/env npx tsx
/**
 * Upload iClone-generated video to Supabase and update database
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function main() {
  const videoPath = 'C:\\Users\\user\\UI-TARS-desktop\\generated-videos\\iclone-lipsync\\day1_hook_final_1765360626285.mp4';
  
  console.log('📤 Uploading iClone video to Supabase...');
  
  // Read and upload
  const videoBuffer = fs.readFileSync(videoPath);
  const remotePath = 'production/videos/iclone/day_001_Hook_iclone.mp4';
  
  const { error: uploadError } = await supabase.storage
    .from('kelly-videos')
    .upload(remotePath, videoBuffer, { 
      upsert: true,
      contentType: 'video/mp4'
    });
  
  if (uploadError) {
    console.error('Upload error:', uploadError);
    return;
  }
  
  const { data } = supabase.storage.from('kelly-videos').getPublicUrl(remotePath);
  console.log('✅ Uploaded:', data.publicUrl);
  
  // Update lesson_atoms for The Scientist Hook
  console.log('\n📝 Updating database...');
  
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();
  
  if (!lesson) {
    console.error('Lesson not found');
    return;
  }
  
  const { error: updateError } = await supabase
    .from('lesson_atoms')
    .update({ hd_video_url: data.publicUrl })
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', 'The Scientist')
    .eq('phase', 'Hook');
  
  if (updateError) {
    console.error('Update error:', updateError);
    return;
  }
  
  console.log('✅ Database updated!');
  console.log('\n🎉 Day 1 Hook now uses iClone video!');
  console.log('   Test at: http://localhost:3000/learn?day=1&clearcache=1');
}

main().catch(console.error);

