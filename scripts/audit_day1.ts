
import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const ARCHETYPES = [
    "The Explorer", "The Rebel", "The Scientist", "The Architect", "The Diplomat",
    "The Empath", "The MacGyver", "The Mystic", "The Storyteller", "The Survivor",
    "The Provider", "The Strategist"
];

const PHASES = ["Hook", "Fact1", "Fact2", "Fact3", "Wisdom"];
const LANGUAGES = ["en", "es", "fr"];

async function auditDay1() {
  console.log("🔍 Auditing Day 1 Content (Detailed)...");

  // 1. Get Core Lesson ID
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('id, day_number')
    .eq('day_number', 1)
    .single();

  if (!lesson) {
    console.error("❌ Day 1 Lesson not found!");
    return;
  }

  // 2. Check Input Scripts (Lesson Atoms & Shards)
  console.log("\n📜 Checking Input Scripts...");
  
  // English (Atoms)
  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('archetype, phase, hd_video_url')
    .eq('core_lesson_id', lesson.id);

  // Multilingual (Shards) - assuming shards hold the translations
  const { data: shards } = await supabase
    .from('lesson_shards')
    .select('age, region, tone, script_content') // tone often maps to archetype?
    .eq('core_lesson_id', lesson.id)
    .in('region', ['es', 'fr']);

  console.log(`  Found ${atoms?.length || 0} English atoms.`);
  console.log(`  Found ${shards?.length || 0} ES/FR shards.`);

  // 3. Check Generated Videos (Kelly Video Assets & Atoms)
  console.log("\n🎥 Checking Generated Videos...");
  
  const { data: assets } = await supabase
    .from('kelly_video_assets')
    .select('archetype, phase, language, public_url, asset_type')
    .eq('day_number', 1)
    .eq('asset_type', 'video');

  const assetMap = new Set();
  assets?.forEach(a => {
      const key = `${a.language}_${a.archetype}_${a.phase}`;
      assetMap.add(key);
  });

  // Also check lesson_atoms for legacy URL
  atoms?.forEach(a => {
      if (a.hd_video_url) {
          // Assume language is 'en' for atoms
          const key = `en_${a.archetype}_${a.phase}`;
          assetMap.add(key);
      }
  });

  console.log(`  Found ${assets?.length || 0} entries in kelly_video_assets.`);
  console.log(`  Found ${atoms?.filter(a => a.hd_video_url).length || 0} atoms with hd_video_url.`);

  // 4. Gap Analysis
  console.log("\n📋 GAP ANALYSIS:");
  let missingCount = 0;

  for (const lang of LANGUAGES) {
      console.log(`\n--- Language: ${lang} ---`);
      for (const arch of ARCHETYPES) {
          for (const phase of PHASES) {
              const key = `${lang}_${arch}_${phase}`;
              const exists = assetMap.has(key);
              
              if (!exists) {
                  // Check if we even have the script to generate it
                  let hasScript = false;
                  if (lang === 'en') {
                      hasScript = atoms?.some(a => a.archetype === arch && a.phase === phase);
                  } else {
                      // Logic to map shard tone to archetype is complex, simplifying check
                      // Assuming if we have shards for this region, we "might" have the script
                      hasScript = shards?.some(s => s.region === lang); 
                  }
                  
                  const status = hasScript ? "❌ Missing Video" : "⚠️ Missing Script";
                  console.log(`  ${status}: ${arch} - ${phase}`);
                  if (hasScript) missingCount++;
              }
          }
      }
  }
}

auditDay1().catch(console.error);

