/**
 * Quick fix for critical day alignments
 * Swap topics to make special days meaningful
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Critical swaps - move topics to where they belong
const swaps = [
  // Day 1 (New Year) should be about NEW BEGINNINGS
  // Swap Day 1 (Leaves) with Day 364 (New Beginnings)
  { from: 1, to: 274 }, // Move Leaves to October (Fall)
  
  // Day 14 (Valentine's) - currently "Curiosity", should be about connection
  // Day 8 is "What Makes a Real Friend" - perfect for Valentine's
  { from: 14, to: 8 }, // Swap Curiosity and Friendship
  
  // Day 111 (Earth Day) - currently "Tools", should be about environment
  // Day 235 is "Protecting What We Have" (Conservation) - perfect!
  { from: 111, to: 235 },
  
  // Day 304 (Halloween) - currently "Muscles", should be about fear/imagination
  // Day 317 is "Your Body's Alarm System" (Fear) - spooky!
  { from: 304, to: 317 }
];

async function fixDays() {
  console.log('🔧 Fixing critical day alignments...\n');
  
  // First, let's see what's at Day 1 vs Day 364
  const { data: day1 } = await supabase.from('core_lessons').select('*').eq('day_number', 1).single();
  const { data: day364 } = await supabase.from('core_lessons').select('*').eq('day_number', 364).single();
  
  console.log('Current Day 1:', day1?.topic);
  console.log('Current Day 364:', day364?.topic);
  
  // Day 364 is "Starting Fresh" - PERFECT for Day 1!
  // Let's just update Day 1's topic to be about new beginnings
  
  console.log('\n📝 Updating Day 1 to be about New Beginnings...');
  
  const { error } = await supabase
    .from('core_lessons')
    .update({ topic: 'Starting Fresh' })
    .eq('day_number', 1);
  
  if (error) {
    console.log('❌ Error:', error.message);
  } else {
    console.log('✅ Day 1 updated to "Starting Fresh"');
  }
  
  // Update Day 364 to the old Day 1 topic (Leaves)
  const { error: error2 } = await supabase
    .from('core_lessons')
    .update({ topic: 'How Leaves Feed the World' })
    .eq('day_number', 274); // Put leaves in October
  
  if (!error2) {
    console.log('✅ Day 274 (October) now has "How Leaves Feed the World"');
  }
  
  // Verify
  console.log('\n=== VERIFICATION ===');
  const { data: verify1 } = await supabase.from('core_lessons').select('day_number, topic').eq('day_number', 1).single();
  console.log(`Day 1: "${verify1?.topic}"`);
}

fixDays().catch(console.error);








