/**
 * Audit Lesson Data Structure
 * Queries Day 330 "Rights" lesson and logs complete data structure
 */

import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = process.env.PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

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
        console.log('\n📋 LESSON_ATOMS STRUCTURE:');
        atoms.forEach((atom, index) => {
          console.log(`\n--- Atom ${index + 1} ---`);
          console.log(`Archetype: ${atom.archetype}`);
          console.log(`Phase: ${atom.phase}`);
          console.log(`Content type: ${typeof atom.content}`);
          if (typeof atom.content === 'object') {
            console.log(`Content keys:`, Object.keys(atom.content));
            console.log('Content preview:', JSON.stringify(atom.content, null, 2).substring(0, 500));
          }
        });
        
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
    
    // 3. Check for looms_shards (if exists)
    console.log(`\n\n🔮 STEP 3: Checking for looms_shards table...`);
    const { data: shards, error: shardsError } = await supabase
      .from('looms_shards')
      .select('*')
      .eq('core_lesson_id', coreLesson.id)
      .limit(5);
    
    if (shardsError) {
      console.log(`ℹ️  looms_shards table may not exist: ${shardsError.message}`);
    } else {
      console.log(`\n✅ Found ${shards?.length || 0} looms_shards`);
      if (shards && shards.length > 0) {
        console.log('\n📋 LOOMS_SHARDS STRUCTURE:');
        console.log(JSON.stringify(shards[0], null, 2));
      }
    }
    
    // 4. Summary
    console.log('\n\n' + '='.repeat(80));
    console.log('📊 SUMMARY');
    console.log('='.repeat(80));
    console.log(`Day ${dayNumber} Lesson: ${coreLesson.topic || 'N/A'}`);
    console.log(`Core Lesson ID: ${coreLesson.id}`);
    console.log(`Available Atoms: ${atoms?.length || 0}`);
    console.log(`Available Shards: ${shards?.length || 0}`);
    
    if (atoms && atoms.length > 0) {
      const uniquePhases = [...new Set(atoms.map(a => a.phase))];
      const uniqueArchetypes = [...new Set(atoms.map(a => a.archetype))];
      console.log(`Unique Phases: ${uniquePhases.join(', ')}`);
      console.log(`Unique Archetypes: ${uniqueArchetypes.join(', ')}`);
    }
    
  } catch (error) {
    console.error('❌ Fatal error:', error);
  }
}

auditLessonData();

