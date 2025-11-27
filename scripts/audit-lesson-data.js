/**
 * Audit Lesson Data Structure
 * Queries Day 330 "Rights" lesson and logs complete data structure
 */

import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

// Create require to load from daily-lesson-marketing node_modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');

// Load environment variables from project root .env
const envPath = resolve(__dirname, '..', '.env');
dotenv.config({ path: envPath });

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('❌ Missing Supabase credentials in .env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function auditLessonData() {
  console.log('🔍 AUDITING LESSON DATA STRUCTURE\n');
  console.log('='.repeat(80));
  
  const dayNumber = 330;
  
  try {
    // 1. Query core_lessons table
    console.log(`\n📚 STEP 1: Querying core_lessons for Day ${dayNumber}...`);
    const { data: coreLesson, error: coreError } = await supabase
      .from('core_lessons')
      .select('*')
      .eq('day_number', dayNumber)
      .single();
    
    if (coreError) {
      console.error('❌ Error querying core_lessons:', coreError);
      return;
    }
    
    if (!coreLesson) {
      console.log(`⚠️  No lesson found for Day ${dayNumber}`);
      return;
    }
    
    console.log('\n✅ CORE_LESSONS DATA:');
    console.log(JSON.stringify(coreLesson, null, 2));
    console.log('\n📋 Available columns:', Object.keys(coreLesson));
    
    // 2. Query lesson_atoms for this lesson
    console.log(`\n\n🧩 STEP 2: Querying lesson_atoms for lesson_id: ${coreLesson.id}...`);
    const { data: atoms, error: atomsError } = await supabase
      .from('lesson_atoms')
      .select('*')
      .eq('core_lesson_id', coreLesson.id);
    
    if (atomsError) {
      console.error('❌ Error querying lesson_atoms:', atomsError);
    } else {
      console.log(`\n✅ Found ${atoms?.length || 0} lesson_atoms`);
      
      if (atoms && atoms.length > 0) {
        console.log('\n📋 LESSON_ATOMS STRUCTURE (First Atom):');
        const atom = atoms[0];
        console.log(JSON.stringify(atom, null, 2));
        console.log('\nAtom Columns:', Object.keys(atom));
        
        // Group by phase
        const phasesByArchetype = {};
        atoms.forEach(atom => {
          if (!phasesByArchetype[atom.archetype]) {
            phasesByArchetype[atom.archetype] = [];
          }
          phasesByArchetype[atom.archetype].push(atom.phase);
        });
        
        console.log('\n📊 PHASES BY ARCHETYPE:');
        console.log(JSON.stringify(phasesByArchetype, null, 2));
      }
    }
    
    // 3. Check for looms_shards
    console.log(`\n\n🔮 STEP 3: Checking for looms_shards table...`);
    const { data: shards, error: shardsError } = await supabase
      .from('looms_shards')
      .select('*')
      .limit(5);
    
    if (shardsError) {
      console.log(`ℹ️  looms_shards table check failed: ${shardsError.message}`);
    } else {
      console.log(`\n✅ Found ${shards?.length || 0} looms_shards records`);
      if (shards && shards.length > 0) {
        console.log('\n📋 LOOMS_SHARDS STRUCTURE:');
        console.log(JSON.stringify(shards[0], null, 2));
        console.log('\nShards Columns:', Object.keys(shards[0]));
      }
    }

    // 4. Check for lesson_shards (alternative name)
    console.log(`\n\n🔮 STEP 4: Checking for lesson_shards table...`);
    const { data: lessonShards, error: lessonShardsError } = await supabase
      .from('lesson_shards')
      .select('*')
      .limit(5);
    
    if (lessonShardsError) {
      console.log(`ℹ️  lesson_shards table check failed: ${lessonShardsError.message}`);
    } else {
      console.log(`\n✅ Found ${lessonShards?.length || 0} lesson_shards records`);
      if (lessonShards && lessonShards.length > 0) {
        console.log('\n📋 LESSON_SHARDS STRUCTURE:');
        console.log(JSON.stringify(lessonShards[0], null, 2));
        console.log('\nLesson Shards Columns:', Object.keys(lessonShards[0]));
      }
    }
    
    console.log('\n\n' + '='.repeat(80));
    console.log('📊 SUMMARY COMPLETE');
    
  } catch (error) {
    console.error('❌ Fatal error:', error);
  }
}

auditLessonData();
