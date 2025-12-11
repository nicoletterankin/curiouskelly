#!/usr/bin/env npx tsx
/**
 * 🚀 UPLOAD DAY 1 DYNAMIC LESSONS
 * 
 * Uploads compiled dynamic videos to Supabase and updates lesson_atoms.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// CONFIG
const DAY_NUMBER = 1;
const INPUT_DIR = path.join(process.cwd(), `generated-videos/compiled/day_${DAY_NUMBER.toString().padStart(3, '0')}`);
const BUCKET = 'kelly-videos';
const STORAGE_PATH = `production/day_${DAY_NUMBER.toString().padStart(3, '0')}_final`;

const supabase = createClient(process.env.PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!);

// MAPPINGS
const ARCHETYPE_DB_MAP: Record<string, string> = {
  "architect": "The Architect",
  "empath": "The Empath",
  "explorer": "The Explorer",
  "macgyver": "The MacGyver",
  "mystic": "The Mystic",
  "rebel": "The Rebel",
  "scientist": "The Scientist",
  "storyteller": "The Storyteller",
  "survivor": "The Survivor",
  "provider": "The Provider",
  "strategist": "The Strategist",
  "neutral": "The Diplomat",
  "diplomat": "The Diplomat" 
};

const PHASE_DB_MAP: Record<string, string> = {
  "hook": "Hook",
  "fact1": "Fact1",
  "fact2": "Fact2",
  "fact3": "Fact3",
  "wisdom": "Wisdom"
};

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  ☁️  UPLOADING DAY 1 DYNAMIC LESSONS                        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // 1. Get Core Lesson ID
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', DAY_NUMBER)
    .single();

  if (lessonError || !lesson) {
    console.error('❌ Could not find Core Lesson for Day 1:', lessonError);
    process.exit(1);
  }
  console.log(`✅ Found Core Lesson ID: ${lesson.id}`);

  // 2. Scan Directory
  if (!fs.existsSync(INPUT_DIR)) {
    console.error(`❌ Input directory not found: ${INPUT_DIR}`);
    process.exit(1);
  }

  const files = fs.readdirSync(INPUT_DIR).filter(f => f.endsWith('.mp4'));
  console.log(`📋 Found ${files.length} compiled videos.`);

  for (const file of files) {
    // Format: day_001_fact1_architect_dynamic.mp4
    const parts = file.split('_');
    if (parts.length < 5) {
      console.warn(`⚠️ Skipping invalid filename: ${file}`);
      continue;
    }

    const phaseKey = parts[2]; // fact1
    const archKey = parts[3];  // architect
    
    const dbPhase = PHASE_DB_MAP[phaseKey];
    const dbArchetype = ARCHETYPE_DB_MAP[archKey];

    if (!dbPhase || !dbArchetype) {
      console.warn(`⚠️ Skipping unknown key mapping: Phase=${phaseKey}, Arch=${archKey} (${file})`);
      continue;
    }

    console.log(`\n📤 Processing: ${dbArchetype} - ${dbPhase}`);

    // 3. Upload to Storage
    const filePath = path.join(INPUT_DIR, file);
    const fileBuffer = fs.readFileSync(filePath);
    const remotePath = `${STORAGE_PATH}/${file}`;

    const { error: uploadError } = await supabase.storage
      .from(BUCKET)
      .upload(remotePath, fileBuffer, { upsert: true, contentType: 'video/mp4' });

    if (uploadError) {
      console.error(`  ❌ Upload Failed: ${uploadError.message}`);
      continue;
    }

    const { data: publicUrlData } = supabase.storage.from(BUCKET).getPublicUrl(remotePath);
    const publicUrl = publicUrlData.publicUrl;
    console.log(`  ☁️ Uploaded: ${publicUrl}`);

    // 4. Update Database
    const { error: dbError } = await supabase
      .from('lesson_atoms')
      .update({ hd_video_url: publicUrl })
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', dbArchetype)
      .eq('phase', dbPhase);

    if (dbError) {
      console.error(`  ❌ Database Update Failed: ${dbError.message}`);
    } else {
      console.log(`  ✅ Database Updated!`);
    }
  }

  console.log('\n🎉 ALL DONE!');
}

main().catch(console.error);
