/**
 * Query all lessons marketing fields
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function queryAllLessons() {
  console.error('Querying all lessons...');
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline, marketing_pitch, sample_testimonial, success_metric')
    .order('day_number', { ascending: true });

  if (error) {
    console.error('Error:', error);
    return;
  }

  console.error(`Found ${lessons.length} lessons`);
  console.log(JSON.stringify(lessons, null, 2));
}

queryAllLessons().catch(console.error);






