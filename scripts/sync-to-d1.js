/**
 * Sync Supabase data to Cloudflare D1
 * 
 * This script syncs lesson data from Supabase to D1 for redundancy.
 * Run daily via cron job or GitHub Actions.
 * 
 * Required environment variables:
 *   SUPABASE_URL - Your Supabase project URL
 *   SUPABASE_SERVICE_KEY - Supabase service role key
 *   CF_ACCOUNT_ID - Cloudflare account ID
 *   CF_API_TOKEN - Cloudflare API token with D1 permissions
 *   D1_DATABASE_ID - The D1 database ID
 * 
 * Usage:
 *   node scripts/sync-to-d1.js
 */

const { createClient } = require('@supabase/supabase-js');

// Load environment variables
require('dotenv').config();

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
const CF_ACCOUNT_ID = process.env.CF_ACCOUNT_ID || process.env.CLOUDFLARE_ACCOUNT_ID;
const CF_API_TOKEN = process.env.CF_API_TOKEN || process.env.CLOUDFLARE_API_TOKEN;
const D1_DATABASE_ID = process.env.D1_DATABASE_ID;

// Validate required env vars
function validateEnv() {
  const missing = [];
  if (!SUPABASE_URL) missing.push('SUPABASE_URL');
  if (!SUPABASE_KEY) missing.push('SUPABASE_SERVICE_KEY');
  if (!CF_ACCOUNT_ID) missing.push('CF_ACCOUNT_ID');
  if (!CF_API_TOKEN) missing.push('CF_API_TOKEN');
  if (!D1_DATABASE_ID) missing.push('D1_DATABASE_ID');
  
  if (missing.length > 0) {
    console.error('❌ Missing required environment variables:', missing.join(', '));
    console.error('\nSet them in .env or export them:');
    missing.forEach(v => console.error(`  export ${v}=your_value`));
    process.exit(1);
  }
}

// Initialize Supabase client
function getSupabase() {
  return createClient(SUPABASE_URL, SUPABASE_KEY);
}

// Execute D1 query via Cloudflare API
async function executeD1(sql, params = []) {
  const response = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/d1/database/${D1_DATABASE_ID}/query`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CF_API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sql, params })
    }
  );
  
  const result = await response.json();
  
  if (!response.ok || !result.success) {
    const errorMsg = result.errors?.[0]?.message || 'Unknown D1 error';
    throw new Error(`D1 Error: ${errorMsg}`);
  }
  
  return result;
}

// Batch execute multiple D1 statements
async function executeBatch(statements) {
  const response = await fetch(
    `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/d1/database/${D1_DATABASE_ID}/query`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CF_API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(statements)
    }
  );
  
  const result = await response.json();
  
  if (!response.ok) {
    throw new Error(`D1 Batch Error: ${JSON.stringify(result.errors)}`);
  }
  
  return result;
}

// Safe JSON stringify for storing in D1
function safeJsonStringify(obj) {
  if (obj === null || obj === undefined) return null;
  if (typeof obj === 'string') return obj;
  try {
    return JSON.stringify(obj);
  } catch {
    return null;
  }
}

// Main sync function
async function sync() {
  console.log('🔄 Starting Supabase → D1 sync...');
  console.log(`   Supabase URL: ${SUPABASE_URL}`);
  console.log(`   D1 Database: ${D1_DATABASE_ID.substring(0, 8)}...`);
  
  const startTime = Date.now();
  const supabase = getSupabase();
  
  // Fetch all data from Supabase
  console.log('\n📥 Fetching data from Supabase...');
  
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select('*')
    .order('day_number');
  
  if (lessonsError) {
    throw new Error(`Failed to fetch lessons: ${lessonsError.message}`);
  }
  
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*');
  
  if (atomsError) {
    console.warn(`⚠️ Warning: lesson_atoms fetch failed: ${atomsError.message}`);
  }
  
  const { data: shards, error: shardsError } = await supabase
    .from('lesson_shards')
    .select('*');
  
  if (shardsError) {
    console.warn(`⚠️ Warning: lesson_shards fetch failed: ${shardsError.message}`);
  }
  
  console.log(`📦 Fetched: ${lessons.length} lessons, ${(atoms || []).length} atoms, ${(shards || []).length} shards`);
  
  // Clear D1 tables
  console.log('\n🗑️  Clearing D1 tables...');
  await executeD1('DELETE FROM shards');
  await executeD1('DELETE FROM atoms');
  await executeD1('DELETE FROM lessons');
  
  // Insert lessons
  console.log('\n📤 Inserting lessons...');
  let lessonCount = 0;
  
  for (const lesson of lessons) {
    await executeD1(
      `INSERT INTO lessons (
        day_number, title, topic, subtitle, marketing_hook, marketing_headline, 
        marketing_tagline, marketing_pitch, hook_question, universal_truth, 
        content, category, difficulty, duration_estimate, hero_image_url, 
        thumbnail_url, audio_url, video_url, quick_quiz_questions, 
        reflection_prompts, mastery_criteria, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        lesson.day_number,
        lesson.title || lesson.topic || 'Daily Discovery',
        lesson.topic,
        lesson.subtitle || lesson.marketing_tagline,
        lesson.marketing_hook,
        lesson.marketing_headline,
        lesson.marketing_tagline,
        lesson.marketing_pitch,
        lesson.hook_question,
        lesson.universal_truth,
        safeJsonStringify(lesson.content),
        lesson.category,
        lesson.difficulty || 'beginner',
        lesson.duration_estimate || 5,
        lesson.hero_image_url,
        lesson.thumbnail_url,
        lesson.audio_url,
        lesson.video_url,
        safeJsonStringify(lesson.quick_quiz_questions),
        safeJsonStringify(lesson.reflection_prompts),
        lesson.mastery_criteria,
        lesson.created_at || new Date().toISOString(),
        lesson.updated_at || new Date().toISOString()
      ]
    );
    lessonCount++;
    
    // Progress indicator every 50 lessons
    if (lessonCount % 50 === 0) {
      console.log(`   ✓ ${lessonCount}/${lessons.length} lessons`);
    }
  }
  console.log(`✅ Inserted ${lessonCount} lessons`);
  
  // Insert atoms
  if (atoms && atoms.length > 0) {
    console.log('\n📤 Inserting atoms...');
    let atomCount = 0;
    
    for (const atom of atoms) {
      // Find the day_number for this atom
      const lessonDay = lessons.find(l => l.id === atom.core_lesson_id)?.day_number;
      if (!lessonDay) continue;
      
      await executeD1(
        `INSERT INTO atoms (
          lesson_day, core_lesson_id, archetype, phase, dialog_type, content,
          kelly_script, kelly_pose, kelly_emotion, trigger_context, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          lessonDay,
          atom.core_lesson_id,
          atom.archetype,
          atom.phase,
          atom.dialog_type,
          safeJsonStringify(atom.content),
          atom.kelly_script,
          atom.kelly_pose,
          atom.kelly_emotion,
          atom.trigger_context,
          atom.is_active ? 1 : 0
        ]
      );
      atomCount++;
      
      if (atomCount % 500 === 0) {
        console.log(`   ✓ ${atomCount}/${atoms.length} atoms`);
      }
    }
    console.log(`✅ Inserted ${atomCount} atoms`);
  }
  
  // Insert shards
  if (shards && shards.length > 0) {
    console.log('\n📤 Inserting shards...');
    let shardCount = 0;
    
    for (const shard of shards) {
      // Find the day_number for this shard
      const lessonDay = lessons.find(l => l.id === shard.core_lesson_id)?.day_number;
      if (!lessonDay) continue;
      
      await executeD1(
        `INSERT INTO shards (
          lesson_day, core_lesson_id, archetype, region, age, script_content, diff_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [
          lessonDay,
          shard.core_lesson_id,
          shard.archetype,
          shard.region || 'adult',
          shard.age,
          safeJsonStringify(shard.script_content),
          shard.diff_type
        ]
      );
      shardCount++;
      
      if (shardCount % 500 === 0) {
        console.log(`   ✓ ${shardCount}/${shards.length} shards`);
      }
    }
    console.log(`✅ Inserted ${shardCount} shards`);
  }
  
  // Update sync metadata
  const duration = Date.now() - startTime;
  await executeD1(
    `UPDATE sync_metadata SET 
      last_sync_at = ?,
      lessons_count = ?,
      atoms_count = ?,
      shards_count = ?,
      sync_source = 'supabase',
      sync_duration_ms = ?
    WHERE id = 1`,
    [
      new Date().toISOString(),
      lessonCount,
      (atoms || []).length,
      (shards || []).length,
      duration
    ]
  );
  
  console.log(`\n🎉 Sync complete in ${(duration / 1000).toFixed(1)}s!`);
  console.log(`   Lessons: ${lessonCount}`);
  console.log(`   Atoms: ${(atoms || []).length}`);
  console.log(`   Shards: ${(shards || []).length}`);
}

// Verify sync by checking counts
async function verify() {
  console.log('\n🔍 Verifying sync...');
  
  const lessonsResult = await executeD1('SELECT COUNT(*) as count FROM lessons');
  const atomsResult = await executeD1('SELECT COUNT(*) as count FROM atoms');
  const shardsResult = await executeD1('SELECT COUNT(*) as count FROM shards');
  
  const counts = {
    lessons: lessonsResult.result?.[0]?.results?.[0]?.count || 0,
    atoms: atomsResult.result?.[0]?.results?.[0]?.count || 0,
    shards: shardsResult.result?.[0]?.results?.[0]?.count || 0
  };
  
  console.log(`   D1 contains: ${counts.lessons} lessons, ${counts.atoms} atoms, ${counts.shards} shards`);
  
  if (counts.lessons === 0) {
    console.warn('⚠️ Warning: No lessons in D1 after sync!');
  }
  
  return counts;
}

// Main entry point
async function main() {
  try {
    validateEnv();
    await sync();
    await verify();
  } catch (error) {
    console.error('\n❌ Sync failed:', error.message);
    process.exit(1);
  }
}

main();


