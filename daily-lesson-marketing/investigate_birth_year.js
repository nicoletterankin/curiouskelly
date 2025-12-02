/**
 * Investigate birth_year usage in lesson_shards
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function investigate() {
  console.log('╔═══════════════════════════════════════════════════════════╗');
  console.log('║        🔍 BIRTH YEAR INVESTIGATION                        ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');

  // 1. Get unique birth years
  console.log('1️⃣  Unique birth_year values in lesson_shards:\n');
  const { data: birthYears, error: byErr } = await supabase
    .from('lesson_shards')
    .select('birth_year')
    .order('birth_year');
  
  if (byErr) {
    console.log('Error:', byErr.message);
    return;
  }
  
  const uniqueYears = [...new Set(birthYears.map(b => b.birth_year))].sort((a,b) => a-b);
  console.log('   Years found:', uniqueYears.length > 20 ? `${uniqueYears.slice(0,10).join(', ')}... (${uniqueYears.length} total)` : uniqueYears.join(', '));
  console.log('   Range:', uniqueYears[0], 'to', uniqueYears[uniqueYears.length - 1]);
  
  // 2. Count shards per birth year
  console.log('\n2️⃣  Shards per birth_year:');
  const yearCounts = {};
  birthYears.forEach(b => {
    yearCounts[b.birth_year] = (yearCounts[b.birth_year] || 0) + 1;
  });
  
  Object.entries(yearCounts).slice(0, 10).forEach(([year, count]) => {
    console.log(`   ${year}: ${count} shards`);
  });
  if (Object.keys(yearCounts).length > 10) {
    console.log(`   ... and ${Object.keys(yearCounts).length - 10} more years`);
  }
  
  // 3. Sample a shard to see the structure
  console.log('\n3️⃣  Sample shard structure:\n');
  const { data: sample, error: sampleErr } = await supabase
    .from('lesson_shards')
    .select('*')
    .limit(1)
    .single();
  
  if (sample) {
    console.log('   Columns:', Object.keys(sample).join(', '));
    console.log('   Sample birth_year:', sample.birth_year);
    console.log('   Sample age:', sample.age);
    console.log('   Sample tone:', sample.tone);
    console.log('   Sample region:', sample.region);
    console.log('   Script content preview:', 
      typeof sample.script_content === 'string' 
        ? sample.script_content.substring(0, 100) + '...'
        : JSON.stringify(sample.script_content).substring(0, 100) + '...'
    );
  }
  
  // 4. Check combination of age + birth_year
  console.log('\n4️⃣  Age + birth_year combinations (first 20):');
  const { data: combos, error: comboErr } = await supabase
    .from('lesson_shards')
    .select('age, birth_year')
    .limit(100);
  
  const uniqueCombos = [...new Set(combos.map(c => `age=${c.age}, birth_year=${c.birth_year}`))];
  uniqueCombos.slice(0, 20).forEach(c => console.log(`   ${c}`));
  
  // 5. Check how many shards per lesson
  console.log('\n5️⃣  Checking shard distribution:');
  const { data: lesson1Shards } = await supabase
    .from('lesson_shards')
    .select('id, age, birth_year, tone, region')
    .eq('core_lesson_id', (await supabase.from('core_lessons').select('id').eq('day_number', 1).single()).data.id);
  
  if (lesson1Shards) {
    console.log(`   Day 1 has ${lesson1Shards.length} shards`);
    console.log('   Unique ages in Day 1:', [...new Set(lesson1Shards.map(s => s.age))].sort((a,b) => a-b).join(', '));
    console.log('   Unique birth_years in Day 1:', [...new Set(lesson1Shards.map(s => s.birth_year))].sort((a,b) => a-b).join(', '));
    console.log('   Unique tones in Day 1:', [...new Set(lesson1Shards.map(s => s.tone))].join(', '));
    console.log('   Unique regions in Day 1:', [...new Set(lesson1Shards.map(s => s.region))].join(', '));
  }
  
  // 6. Calculate the actual variant space
  console.log('\n6️⃣  VARIANT SPACE ANALYSIS:');
  const totalShards = birthYears.length;
  const totalYears = uniqueYears.length;
  const avgShardsPerYear = Math.round(totalShards / totalYears);
  
  console.log(`   Total shards: ${totalShards.toLocaleString()}`);
  console.log(`   Unique birth years: ${totalYears}`);
  console.log(`   Avg shards per birth year: ${avgShardsPerYear}`);
  
  // If we have 365 lessons × N birth years × 3 tones × 3 languages
  const estimatedStructure = 365 * totalYears * 3 * 3;
  console.log(`   Expected if 365 × ${totalYears} × 3 × 3: ${estimatedStructure.toLocaleString()}`);
}

investigate().catch(console.error);






