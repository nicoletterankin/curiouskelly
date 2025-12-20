#!/usr/bin/env npx ts-node
/**
 * Sync Supabase to Cloudflare D1 Mirror
 * 
 * This script syncs lesson data from Supabase to D1 for redundancy.
 * Run daily or after major content updates.
 * 
 * Usage: npx ts-node scripts/sync-d1-mirror.ts
 * 
 * Requires:
 * - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
 * - wrangler CLI installed and configured
 */

import { createClient } from '@supabase/supabase-js';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// Load environment
require('dotenv').config();

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const D1_DATABASE_NAME = 'lessons-db';
const BATCH_SIZE = 50;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

interface CoreLesson {
  id: number;
  day_number: number;
  topic: string;
  universal_truth?: string;
  marketing_headline?: string;
  marketing_tagline?: string;
  marketing_pitch?: string;
  hook_question?: string;
  quick_quiz_questions?: any;
  reflection_prompts?: any;
  mastery_criteria?: string;
  hero_image_url?: string;
  thumbnail_url?: string;
  audio_url?: string;
}

interface LessonAtom {
  id: number;
  core_lesson_id: number;
  archetype: string;
  phase: string;
  content: any;
  hd_video_url?: string;
  visual_url?: string;
}

interface LessonShard {
  id: number;
  core_lesson_id: number;
  archetype: string;
  region?: string;
  age?: number;
  tone?: string;
  birth_year?: number;
  script_content: any;
}

function escapeSQL(value: any): string {
  if (value === null || value === undefined) return 'NULL';
  if (typeof value === 'number') return String(value);
  if (typeof value === 'object') {
    return `'${JSON.stringify(value).replace(/'/g, "''")}'`;
  }
  return `'${String(value).replace(/'/g, "''")}'`;
}

async function fetchAllLessons(): Promise<CoreLesson[]> {
  console.log('📚 Fetching core_lessons from Supabase...');
  
  const { data, error } = await supabase
    .from('core_lessons')
    .select('*')
    .order('day_number');
  
  if (error) {
    console.error('❌ Error fetching lessons:', error);
    throw error;
  }
  
  console.log(`✅ Fetched ${data?.length || 0} lessons`);
  return data || [];
}

async function fetchAllAtoms(): Promise<LessonAtom[]> {
  console.log('🔬 Fetching lesson_atoms from Supabase...');
  
  // Fetch in batches to avoid timeout
  let allAtoms: LessonAtom[] = [];
  let offset = 0;
  const limit = 1000;
  
  while (true) {
    const { data, error } = await supabase
      .from('lesson_atoms')
      .select('*')
      .range(offset, offset + limit - 1);
    
    if (error) {
      console.error('❌ Error fetching atoms:', error);
      throw error;
    }
    
    if (!data || data.length === 0) break;
    
    allAtoms = allAtoms.concat(data);
    console.log(`  ... fetched ${allAtoms.length} atoms`);
    
    if (data.length < limit) break;
    offset += limit;
  }
  
  console.log(`✅ Fetched ${allAtoms.length} atoms`);
  return allAtoms;
}

async function fetchAllShards(): Promise<LessonShard[]> {
  console.log('🧩 Fetching lesson_shards from Supabase...');
  
  let allShards: LessonShard[] = [];
  let offset = 0;
  const limit = 1000;
  
  while (true) {
    const { data, error } = await supabase
      .from('lesson_shards')
      .select('*')
      .range(offset, offset + limit - 1);
    
    if (error) {
      console.error('❌ Error fetching shards:', error);
      throw error;
    }
    
    if (!data || data.length === 0) break;
    
    allShards = allShards.concat(data);
    console.log(`  ... fetched ${allShards.length} shards`);
    
    if (data.length < limit) break;
    offset += limit;
  }
  
  console.log(`✅ Fetched ${allShards.length} shards`);
  return allShards;
}

function generateInsertSQL(lessons: CoreLesson[], atoms: LessonAtom[], shards: LessonShard[]): string {
  let sql = '';
  
  // Clear existing data
  sql += '-- Clear existing data\n';
  sql += 'DELETE FROM lesson_shards;\n';
  sql += 'DELETE FROM lesson_atoms;\n';
  sql += 'DELETE FROM core_lessons;\n\n';
  
  // Insert lessons
  sql += '-- Insert core_lessons\n';
  for (const lesson of lessons) {
    sql += `INSERT INTO core_lessons (id, day_number, topic, universal_truth, marketing_headline, marketing_tagline, marketing_pitch, hook_question, quick_quiz_questions, reflection_prompts, mastery_criteria, hero_image_url, thumbnail_url, audio_url) VALUES (`;
    sql += [
      lesson.id,
      lesson.day_number,
      escapeSQL(lesson.topic),
      escapeSQL(lesson.universal_truth),
      escapeSQL(lesson.marketing_headline),
      escapeSQL(lesson.marketing_tagline),
      escapeSQL(lesson.marketing_pitch),
      escapeSQL(lesson.hook_question),
      escapeSQL(lesson.quick_quiz_questions),
      escapeSQL(lesson.reflection_prompts),
      escapeSQL(lesson.mastery_criteria),
      escapeSQL(lesson.hero_image_url),
      escapeSQL(lesson.thumbnail_url),
      escapeSQL(lesson.audio_url)
    ].join(', ');
    sql += ');\n';
  }
  
  // Insert atoms
  sql += '\n-- Insert lesson_atoms\n';
  for (const atom of atoms) {
    sql += `INSERT INTO lesson_atoms (id, core_lesson_id, archetype, phase, content, hd_video_url, visual_url) VALUES (`;
    sql += [
      atom.id,
      atom.core_lesson_id,
      escapeSQL(atom.archetype),
      escapeSQL(atom.phase),
      escapeSQL(atom.content),
      escapeSQL(atom.hd_video_url),
      escapeSQL(atom.visual_url)
    ].join(', ');
    sql += ');\n';
  }
  
  // Insert shards
  sql += '\n-- Insert lesson_shards\n';
  for (const shard of shards) {
    sql += `INSERT INTO lesson_shards (id, core_lesson_id, archetype, region, age, tone, birth_year, script_content) VALUES (`;
    sql += [
      shard.id,
      shard.core_lesson_id,
      escapeSQL(shard.archetype),
      escapeSQL(shard.region),
      shard.age ?? 'NULL',
      escapeSQL(shard.tone),
      shard.birth_year ?? 'NULL',
      escapeSQL(shard.script_content)
    ].join(', ');
    sql += ');\n';
  }
  
  // Update sync status
  sql += '\n-- Update sync status\n';
  sql += `INSERT OR REPLACE INTO sync_status (id, table_name, last_sync, row_count, checksum) VALUES \n`;
  sql += `  (1, 'core_lessons', datetime('now'), ${lessons.length}, '${Date.now()}'),\n`;
  sql += `  (2, 'lesson_atoms', datetime('now'), ${atoms.length}, '${Date.now()}'),\n`;
  sql += `  (3, 'lesson_shards', datetime('now'), ${shards.length}, '${Date.now()}');\n`;
  
  return sql;
}

async function main() {
  console.log('🔄 Starting D1 Mirror Sync\n');
  console.log('='.repeat(50));
  
  try {
    // Fetch all data from Supabase
    const lessons = await fetchAllLessons();
    const atoms = await fetchAllAtoms();
    const shards = await fetchAllShards();
    
    console.log('\n📊 Summary:');
    console.log(`   - ${lessons.length} lessons`);
    console.log(`   - ${atoms.length} atoms`);
    console.log(`   - ${shards.length} shards`);
    
    // Generate SQL
    console.log('\n📝 Generating SQL...');
    const sql = generateInsertSQL(lessons, atoms, shards);
    
    // Write to file
    const sqlPath = path.join(__dirname, '..', 'sql', 'd1-sync-data.sql');
    fs.writeFileSync(sqlPath, sql);
    console.log(`✅ SQL written to ${sqlPath}`);
    console.log(`   File size: ${(fs.statSync(sqlPath).size / 1024 / 1024).toFixed(2)} MB`);
    
    // Optionally run wrangler to sync
    if (process.argv.includes('--execute')) {
      console.log('\n🚀 Executing sync with wrangler...');
      try {
        execSync(`wrangler d1 execute ${D1_DATABASE_NAME} --file=${sqlPath}`, {
          stdio: 'inherit'
        });
        console.log('✅ D1 sync complete!');
      } catch (e) {
        console.error('❌ Wrangler execution failed:', e);
        console.log('   Run manually: wrangler d1 execute lessons-db --file=./sql/d1-sync-data.sql');
      }
    } else {
      console.log('\n💡 To apply to D1, run:');
      console.log(`   wrangler d1 execute ${D1_DATABASE_NAME} --file=./sql/d1-sync-data.sql`);
      console.log('   Or run this script with --execute flag');
    }
    
    console.log('\n✅ D1 Mirror Sync preparation complete!');
    
  } catch (error) {
    console.error('\n❌ Sync failed:', error);
    process.exit(1);
  }
}

main();












