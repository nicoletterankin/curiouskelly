/**
 * VALIDATE VISUAL QUALITY
 * Check actual generated visuals and their academic appropriateness
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function main() {
  console.log('═'.repeat(70));
  console.log('VISUAL QUALITY VALIDATION');
  console.log('═'.repeat(70));
  
  // Sample Days 6-11 (Gemini generated)
  const sampleDays = [6, 7, 8, 9, 10, 11];
  
  for (const day of sampleDays) {
    const paddedDay = String(day).padStart(3, '0');
    
    // Get lesson info
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('day_number, topic, universal_truth')
      .eq('day_number', day)
      .single();
    
    if (!lesson) continue;
    
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`DAY ${day}: ${lesson.topic}`);
    console.log(`Universal Truth: ${lesson.universal_truth}`);
    console.log(`${'─'.repeat(70)}`);
    
    // Check what visuals exist
    const baseUrl = `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-${paddedDay}`;
    
    const assets = [
      { name: 'Thumbnail', url: `${baseUrl}/thumbnail.png` },
      { name: 'Illustration', url: `${baseUrl}/illustration.png` },
      { name: 'Infographic 1', url: `${baseUrl}/infographic-1.png` },
      { name: 'Infographic 2', url: `${baseUrl}/infographic-2.png` },
      { name: 'Infographic 3', url: `${baseUrl}/infographic-3.png` },
    ];
    
    console.log('\nGenerated Assets:');
    for (const asset of assets) {
      try {
        const response = await fetch(asset.url, { method: 'HEAD' });
        if (response.ok) {
          console.log(`  ✅ ${asset.name}`);
          console.log(`     ${asset.url}`);
        }
      } catch (e) {
        // Skip missing
      }
    }
    
    // Get the visual plan that was generated
    const { data: visualData } = await supabase
      .from('lesson_visuals')
      .select('*')
      .eq('day_number', day)
      .maybeSingle();
    
    if (visualData) {
      console.log('\nGeneration Status:', visualData.status);
      if (visualData.error) {
        console.log('Error:', visualData.error);
      }
    }
  }
  
  console.log('\n' + '═'.repeat(70));
  console.log('QUALITY ASSESSMENT NEEDED');
  console.log('═'.repeat(70));
  console.log(`
To validate academic quality, manually review these URLs in browser:

Day 6 (What's Inside a Seed):
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-006/infographic-1.png
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-006/infographic-2.png
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-006/illustration.png

Day 7 (What Stars Are Made Of):
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-007/infographic-1.png
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-007/infographic-2.png

Day 8 (What Makes a Real Friend):
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-008/infographic-1.png
  https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/lesson-visuals/day-008/infographic-2.png

QUALITY CRITERIA:
✓ Age-appropriate (5-12 year olds)
✓ Scientifically accurate
✓ Clear visual metaphors
✓ No text/captions (visual only)
✓ 16:9 aspect ratio
✓ Professional, clean design
✓ Supports learning objective
  `);
}

main().catch(console.error);
