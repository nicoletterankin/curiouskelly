/**
 * Push age hooks to Supabase
 * Creates a new table 'lesson_age_hooks' with all 2,196 hooks
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function pushHooks() {
  console.log('📦 Loading age hooks...');
  const hooksData = JSON.parse(fs.readFileSync('age_hooks.json', 'utf8'));
  
  console.log(`Found ${hooksData.hooks.length} lessons with hooks`);
  console.log('Age buckets:', hooksData.age_buckets);
  
  // Flatten to individual records: one row per day + age_bucket combination
  const records = [];
  for (const lesson of hooksData.hooks) {
    for (const [ageBucket, hook] of Object.entries(lesson.hooks)) {
      records.push({
        day_number: lesson.day,
        topic: lesson.topic,
        age_bucket: ageBucket,
        hook: hook
      });
    }
  }
  
  console.log(`\n📊 Total records to insert: ${records.length}`);
  
  // Insert in batches of 500
  const BATCH_SIZE = 500;
  let inserted = 0;
  let errors = 0;
  
  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    
    const { data, error } = await supabase
      .from('lesson_age_hooks')
      .upsert(batch, { 
        onConflict: 'day_number,age_bucket',
        ignoreDuplicates: false 
      });
    
    if (error) {
      console.log(`❌ Batch ${Math.floor(i/BATCH_SIZE) + 1} error:`, error.message);
      errors++;
      
      // If table doesn't exist, we need to create it first
      if (error.message.includes('does not exist')) {
        console.log('\n⚠️  Table lesson_age_hooks does not exist!');
        console.log('Creating table via SQL...');
        
        // Try to create the table
        const createTableSQL = `
          CREATE TABLE IF NOT EXISTS lesson_age_hooks (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            day_number INTEGER NOT NULL,
            topic TEXT NOT NULL,
            age_bucket TEXT NOT NULL,
            hook TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(day_number, age_bucket)
          );
          
          CREATE INDEX IF NOT EXISTS idx_lesson_age_hooks_day ON lesson_age_hooks(day_number);
          CREATE INDEX IF NOT EXISTS idx_lesson_age_hooks_bucket ON lesson_age_hooks(age_bucket);
        `;
        
        console.log('\n📝 SQL to create table:');
        console.log(createTableSQL);
        console.log('\n⚠️  Please create this table in Supabase Dashboard, then run this script again.');
        return;
      }
    } else {
      inserted += batch.length;
      console.log(`✅ Batch ${Math.floor(i/BATCH_SIZE) + 1}: ${batch.length} records`);
    }
  }
  
  console.log('\n=== COMPLETE ===');
  console.log(`✅ Inserted: ${inserted} records`);
  console.log(`❌ Errors: ${errors} batches`);
  
  // Verify
  console.log('\n=== VERIFICATION ===');
  const { data: sample, error: sampleError } = await supabase
    .from('lesson_age_hooks')
    .select('*')
    .eq('day_number', 1)
    .order('age_bucket');
  
  if (sample) {
    console.log('Day 1 hooks:');
    sample.forEach(h => console.log(`  ${h.age_bucket}: "${h.hook}"`));
  }
}

pushHooks().catch(console.error);










