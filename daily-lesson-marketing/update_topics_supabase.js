import { createClient } from '@supabase/supabase-js';
import fs from 'fs';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function updateTopics() {
  console.log('Loading enhanced topics...');
  const enhancedTopics = JSON.parse(fs.readFileSync('enhanced_topics.json', 'utf8'));
  
  console.log(`Found ${enhancedTopics.topics.length} topics to update\n`);
  
  let successCount = 0;
  let errorCount = 0;
  const errors = [];
  
  for (const topic of enhancedTopics.topics) {
    const { data, error } = await supabase
      .from('core_lessons')
      .update({ topic: topic.new_topic })
      .eq('day_number', topic.day);
    
    if (error) {
      errorCount++;
      errors.push({ day: topic.day, error: error.message });
      console.log(`❌ Day ${topic.day}: ${error.message}`);
    } else {
      successCount++;
      if (successCount % 50 === 0) {
        console.log(`✓ Updated ${successCount} topics...`);
      }
    }
  }
  
  console.log('\n=== UPDATE COMPLETE ===');
  console.log(`✓ Success: ${successCount}`);
  console.log(`✗ Errors: ${errorCount}`);
  
  if (errors.length > 0) {
    console.log('\nErrors:');
    errors.forEach(e => console.log(`  Day ${e.day}: ${e.error}`));
  }
  
  // Verify a few updates
  console.log('\n=== VERIFICATION ===');
  const verifyDays = [1, 50, 100, 200, 300, 365];
  for (const day of verifyDays) {
    const { data } = await supabase
      .from('core_lessons')
      .select('day_number, topic')
      .eq('day_number', day)
      .single();
    
    if (data) {
      console.log(`Day ${day}: "${data.topic}"`);
    }
  }
}

updateTopics().catch(console.error);

