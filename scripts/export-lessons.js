/**
 * Export Lessons to Static JSON Files
 * 
 * Run with: node scripts/export-lessons.js
 * Requires: npm install @supabase/supabase-js
 * 
 * Environment variables:
 *   SUPABASE_URL - Supabase project URL (optional, has default)
 *   SUPABASE_SERVICE_KEY - Service key with full access (required)
 * 
 * This script exports all lessons from Supabase to static JSON files
 * that can be used as fallback when Supabase is unavailable.
 */

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const SUPABASE_URL = process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY;

if (!SUPABASE_KEY) {
  console.error('❌ SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY environment variable required');
  console.error('');
  console.error('Usage:');
  console.error('  $env:SUPABASE_SERVICE_KEY="your-key-here"');
  console.error('  node scripts/export-lessons.js');
  console.error('');
  console.error('Or set in .env file and run:');
  console.error('  node -r dotenv/config scripts/export-lessons.js');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function exportAllLessons() {
  const outputDir = path.join(__dirname, '..', 'public', 'data', 'lessons');
  
  // Create directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`📁 Created directory: ${outputDir}`);
  }
  
  console.log('📥 Fetching lessons from Supabase...');
  console.log(`   URL: ${SUPABASE_URL}`);
  
  // Fetch all lessons
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select('*')
    .order('day_number');
  
  if (lessonsError) {
    console.error('❌ Failed to fetch lessons:', lessonsError.message);
    return;
  }
  
  if (!lessons || lessons.length === 0) {
    console.error('❌ No lessons found in database');
    return;
  }
  
  console.log(`📦 Found ${lessons.length} lessons`);
  
  // Fetch all atoms
  console.log('📥 Fetching atoms...');
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*');
  
  if (atomsError) {
    console.warn('⚠️ Failed to fetch atoms:', atomsError.message);
  }
  console.log(`📦 Found ${atoms?.length || 0} atoms`);
  
  // Fetch all shards
  console.log('📥 Fetching shards...');
  const { data: shards, error: shardsError } = await supabase
    .from('lesson_shards')
    .select('*');
  
  if (shardsError) {
    console.warn('⚠️ Failed to fetch shards:', shardsError.message);
  }
  console.log(`📦 Found ${shards?.length || 0} shards`);
  
  // Group atoms and shards by lesson
  const atomsByLessonId = {};
  const shardsByLessonId = {};
  
  atoms?.forEach(atom => {
    const lessonId = atom.core_lesson_id;
    if (!atomsByLessonId[lessonId]) atomsByLessonId[lessonId] = [];
    atomsByLessonId[lessonId].push(atom);
  });
  
  shards?.forEach(shard => {
    const lessonId = shard.core_lesson_id;
    if (!shardsByLessonId[lessonId]) shardsByLessonId[lessonId] = [];
    shardsByLessonId[lessonId].push(shard);
  });
  
  // Export each lesson
  let exportedCount = 0;
  for (const lesson of lessons) {
    const day = lesson.day_number;
    const lessonId = lesson.id;
    const filename = `day-${String(day).padStart(3, '0')}.json`;
    
    const fullLesson = {
      lesson,
      atoms: atomsByLessonId[lessonId] || [],
      shards: shardsByLessonId[lessonId] || [],
      exported_at: new Date().toISOString(),
      version: '1.0'
    };
    
    fs.writeFileSync(
      path.join(outputDir, filename),
      JSON.stringify(fullLesson, null, 2)
    );
    exportedCount++;
    
    if (exportedCount % 50 === 0) {
      console.log(`   Exported ${exportedCount}/${lessons.length}...`);
    }
  }
  
  // Create index file
  const index = {
    lessons: lessons.map(l => ({
      day: l.day_number,
      title: l.topic || l.title,
      category: l.category,
      id: l.id
    })),
    total: lessons.length,
    exported_at: new Date().toISOString(),
    version: '1.0'
  };
  
  fs.writeFileSync(
    path.join(outputDir, 'index.json'),
    JSON.stringify(index, null, 2)
  );
  
  console.log('');
  console.log(`✅ Exported ${exportedCount} lessons to ${outputDir}`);
  console.log('');
  console.log('📋 Files created:');
  console.log(`   - ${exportedCount} individual lesson files (day-001.json through day-${String(exportedCount).padStart(3, '0')}.json)`);
  console.log('   - index.json (lesson index)');
  console.log('');
  console.log('💡 These files will be used as fallback when Supabase is unavailable.');
}

// Run the export
exportAllLessons()
  .then(() => {
    console.log('🎉 Export complete!');
    process.exit(0);
  })
  .catch(error => {
    console.error('❌ Export failed:', error);
    process.exit(1);
  });

