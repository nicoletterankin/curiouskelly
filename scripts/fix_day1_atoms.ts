
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function fixAtoms() {
  console.log("🛠️ Fixing Day 1 Atoms...");

  // 1. Get Core Lesson ID
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();

  if (!lesson) {
    console.error("❌ Lesson not found");
    return;
  }

  // 2. Get Template (The Scientist)
  const { data: templateAtoms } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', 'The Scientist');

  if (!templateAtoms || templateAtoms.length === 0) {
    console.error("❌ Template atoms not found");
    return;
  }

  // 3. Insert Missing Archetypes
  const missingArchetypes = ['The Provider', 'The Strategist'];

  for (const targetArch of missingArchetypes) {
    console.log(`Processing ${targetArch}...`);
    
    // Check if exists
    const { data: existing } = await supabase
        .from('lesson_atoms')
        .select('id')
        .eq('core_lesson_id', lesson.id)
        .eq('archetype', targetArch);
        
    if (existing && existing.length > 0) {
        console.log(`  ✅ ${targetArch} already exists.`);
        continue;
    }

    // Clone
    const newAtoms = templateAtoms.map(atom => ({
        core_lesson_id: lesson.id,
        archetype: targetArch,
        phase: atom.phase,
        content: atom.content, // Cloning content for now
        // hd_video_url: null // Reset video
    }));

    const { error } = await supabase
        .from('lesson_atoms')
        .insert(newAtoms);

    if (error) {
        console.error(`  ❌ Error inserting ${targetArch}:`, error);
    } else {
        console.log(`  ✅ Inserted ${newAtoms.length} atoms for ${targetArch}`);
    }
  }
}

fixAtoms().catch(console.error);
















