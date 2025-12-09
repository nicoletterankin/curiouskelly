/**
 * Inspect Database Sample
 * Shows exactly what's in the database for Day 1
 */

import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function inspectDay1() {
  console.log('🔍 INSPECTING DAY 1 DATABASE CONTENT');
  console.log('=' .repeat(60));
  console.log('');

  // Get Day 1 core lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', 1)
    .single();

  if (lessonError) {
    console.error('❌ Error fetching Day 1:', lessonError);
    return;
  }

  console.log('📚 CORE LESSON:');
  console.log(`   Day: ${lesson.day_number}`);
  console.log(`   Topic: ${lesson.topic}`);
  console.log(`   Universal Truth: ${lesson.universal_truth}`);
  console.log(`   ID: ${lesson.id}`);
  console.log('');

  // Get atoms for Day 1
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);

  if (atomsError) {
    console.error('❌ Error fetching atoms:', atomsError);
    return;
  }

  console.log(`📝 ATOMS: ${atoms ? atoms.length : 0} found`);
  console.log('');

  if (atoms && atoms.length > 0) {
    atoms.forEach((atom, i) => {
      console.log(`   Atom ${i + 1}:`);
      console.log(`      Phase: ${atom.phase}`);
      console.log(`      Archetype: ${atom.archetype}`);
      console.log(`      Content keys: ${Object.keys(atom.content || {}).join(', ')}`);
      if (atom.content?.text) {
        console.log(`      Text: ${atom.content.text.substring(0, 100)}...`);
      }
      if (atom.content?.choices) {
        console.log(`      Has choices: Yes`);
      }
      console.log('');
    });
  } else {
    console.log('   ❌ NO ATOMS FOUND');
    console.log('');
    console.log('   This means:');
    console.log('   - No Hook (welcome) phase');
    console.log('   - No Fact1-3 (question) phases');
    console.log('   - No Wisdom (conclusion) phase');
    console.log('   - No content for ANY archetype (Sage/Jester/Ruler)');
    console.log('');
    console.log('   The lesson exists as metadata only!');
  }

  console.log('=' .repeat(60));
}

inspectDay1().catch(console.error);











