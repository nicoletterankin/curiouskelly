
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function patchGaps() {
  console.log("🩹 Operation Seamless Patch: Filling gaps with sibling assets...");

  // 1. Get Day 1 Lesson ID
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', 1)
    .single();

  if (!lesson) { console.error("❌ Lesson 1 not found"); return; }

  // 2. Identify the Gaps (Hardcoded based on audit)
  const gaps = [
    { archetype: 'The Provider', phase: 'Fact2', fallbackPhase: 'Fact1' },
    { archetype: 'The Strategist', phase: 'Fact2', fallbackPhase: 'Fact1' },
    { archetype: 'The Strategist', phase: 'Wisdom', fallbackPhase: 'Fact3' } // Use Fact3 or Hook? Fact3 is closer.
  ];

  for (const gap of gaps) {
    // Get the Fallback Asset URL
    const { data: fallback } = await supabase
      .from('lesson_atoms')
      .select('hd_video_url')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', gap.archetype)
      .eq('phase', gap.fallbackPhase)
      .single();

    if (!fallback?.hd_video_url) {
      console.warn(`⚠️ Could not find fallback (${gap.fallbackPhase}) for ${gap.archetype}`);
      continue;
    }

    console.log(`✅ Found fallback for ${gap.archetype} ${gap.phase}: Using ${gap.fallbackPhase}`);

    // Patch the Target
    const { error } = await supabase
      .from('lesson_atoms')
      .update({ hd_video_url: fallback.hd_video_url })
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', gap.archetype)
      .eq('phase', gap.phase);

    if (error) {
      console.error(`  ❌ Patch failed: ${error.message}`);
    } else {
      console.log(`  🩹 Patched ${gap.archetype} ${gap.phase} successfully.`);
    }
  }
}

patchGaps().catch(console.error);

















