
import { createRequire } from 'module';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(resolve(__dirname, '../daily-lesson-marketing/package.json'));

const { createClient } = require('@supabase/supabase-js');
const dotenv = require('dotenv');

const envPath = resolve(__dirname, '..', '.env');
dotenv.config({ path: envPath });

const supabase = createClient(process.env.PUBLIC_SUPABASE_URL, process.env.PUBLIC_SUPABASE_ANON_KEY);

async function checkShardsForDay330() {
  const dayNumber = 330;
  
  const { data: coreLesson } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .single();
    
  console.log(`Day ${dayNumber} ID: ${coreLesson.id}`);
  
  const { data: shards, error } = await supabase
    .from('lesson_shards')
    .select('count', { count: 'exact' })
    .eq('core_lesson_id', coreLesson.id);
    
  console.log(`Shards for Day 330: ${shards?.length || 0} (Error: ${error?.message || 'None'})`);
  
  if (shards?.length === 0) {
      // Check if ANY shards exist for ANY lesson
      const { count: totalShards } = await supabase
        .from('lesson_shards')
        .select('*', { count: 'exact', head: true });
      console.log(`Total shards in DB: ${totalShards}`);
  }
}

checkShardsForDay330();


