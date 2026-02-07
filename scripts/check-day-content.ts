#!/usr/bin/env npx tsx
/**
 * Quick check of Day 1-7 content in Supabase
 */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

async function main() {
  console.log('Checking Day 1-7 content in Supabase...\n');
  
  // Get core lessons
  const { data: lessons, error: lessonsError } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic')
    .gte('day_number', 1)
    .lte('day_number', 7)
    .order('day_number');
  
  if (lessonsError) {
    console.error('Error fetching lessons:', lessonsError.message);
    process.exit(1);
  }
  
  console.log(`Found ${lessons?.length || 0} core lessons for Day 1-7:\n`);
  
  for (const lesson of lessons || []) {
    // Get atoms for this lesson
    const { data: atoms, count } = await supabase
      .from('lesson_atoms')
      .select('phase, archetype, content', { count: 'exact' })
      .eq('core_lesson_id', lesson.id);
    
    const hasScripts = atoms?.some(a => a.content?.script);
    
    console.log(`Day ${lesson.day_number}: "${lesson.topic}"`);
    console.log(`  - Atoms: ${count || 0}`);
    console.log(`  - Has scripts: ${hasScripts ? '✅' : '❌'}`);
    
    // Count by phase
    const phaseCount: Record<string, number> = {};
    for (const atom of atoms || []) {
      phaseCount[atom.phase] = (phaseCount[atom.phase] || 0) + 1;
    }
    if (Object.keys(phaseCount).length > 0) {
      console.log(`  - Phases: ${JSON.stringify(phaseCount)}`);
    }
    console.log('');
  }
  
  console.log('Done.');
}

main().catch(console.error);
