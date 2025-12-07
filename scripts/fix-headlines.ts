/**
 * EMERGENCY HEADLINE FIXER
 * 
 * Fixes 251 lessons where headlines and universal truths are mismatched with topics.
 * Uses Claude to regenerate content that actually matches the topic.
 * 
 * Usage:
 *   npm run headlines:preview          # Preview 5 sample fixes (SAFE)
 *   npm run headlines:preview-10       # Preview 10 fixes (SAFE)
 *   npm run headlines:fix              # FIX ALL 251 MISMATCHED LESSONS
 *   npm run headlines:fix-day 57       # Fix a single day
 */

import { createClient } from '@supabase/supabase-js';
import Anthropic from '@anthropic-ai/sdk';

// Configuration
const SUPABASE_URL = process.env.SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || '';
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';

if (!SUPABASE_SERVICE_KEY) {
  console.error('❌ Missing SUPABASE_SERVICE_KEY environment variable');
  process.exit(1);
}

if (!ANTHROPIC_API_KEY) {
  console.error('❌ Missing ANTHROPIC_API_KEY environment variable');
  console.error('   Set it in .env.local or export ANTHROPIC_API_KEY=sk-ant-...');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

// Types
interface LessonToFix {
  id: string;
  day_number: number;
  topic: string;
  marketing_headline: string | null;
  universal_truth: string | null;
}

interface FixResult {
  id: string;
  day_number: number;
  topic: string;
  old_headline: string | null;
  new_headline: string;
  old_truth: string | null;
  new_truth: string;
  success: boolean;
  error?: string;
}

// ============================================
// HEADLINE GENERATION
// ============================================

/**
 * Generate a fresh headline that actually matches the topic
 * NO PUNS. NO EXCLAMATION MARKS. NO "UNLOCK/DISCOVER/UNLEASH"
 */
async function generateHeadline(topic: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 100,
    messages: [{
      role: 'user',
      content: `Generate a marketing headline for this educational lesson for kids/families:

Topic: "${topic}"

RULES:
- The headline MUST directly reference the topic
- NO puns (absolutely no "Leaf It to Nature" style wordplay)
- NO exclamation marks
- NO clichés like "Unlock", "Discover", "Unleash", "Master", "Secrets of"
- Write like a smart friend, not a marketing brochure
- Make someone genuinely curious
- 8-15 words max
- Can use a dash or colon for structure

GOOD examples:
- "You walk past 10,000 leaves a day—here's why one might save your life"
- "The reason your heart beats 100,000 times today without being asked"
- "What happens in your brain the moment before you remember something"
- "Every lake you've ever swam in started as rain with nowhere to go"

BAD examples (DO NOT USE THESE PATTERNS):
- "Unlock the Secrets of Your Heart!"
- "Master the Art of Memory!"
- "Leaf It to Nature: Discover the Magic!"
- "Dive Deep into the World of..."

Return ONLY the headline, nothing else.`
    }]
  });

  const text = (response.content[0] as { type: string; text: string }).text.trim();
  // Remove quotes if Claude wrapped it
  return text.replace(/^["']|["']$/g, '');
}

/**
 * Generate a universal truth that matches the topic
 */
async function generateUniversalTruth(topic: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 80,
    messages: [{
      role: 'user',
      content: `Write a single-sentence universal truth about "${topic}" for an educational lesson.

RULES:
- Must be specific to this exact topic
- Not a generic platitude that could apply to anything
- Should create curiosity or reveal something surprising
- State a fact, not a command
- 10-25 words max
- No exclamation marks

GOOD examples:
- "Leaves are the only reason you can breathe right now."
- "Lakes form when water collects faster than it can drain away."
- "Deserts aren't empty—they're selective about who survives."

BAD examples:
- "Nature renews itself through cycles of growth and decay."
- "Learning is a journey of discovery."
- "Everything is connected in beautiful ways."

Return ONLY the truth statement, nothing else.`
    }]
  });

  const text = (response.content[0] as { type: string; text: string }).text.trim();
  return text.replace(/^["']|["']$/g, '');
}

// ============================================
// FIX FUNCTIONS
// ============================================

/**
 * Fix a single lesson by day number
 */
async function fixSingleDay(dayNumber: number): Promise<FixResult | null> {
  console.log(`\n🔧 Fixing Day ${dayNumber}...`);

  const { data: lesson, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, marketing_headline, universal_truth')
    .eq('day_number', dayNumber)
    .single();

  if (error || !lesson) {
    console.error(`❌ Could not find lesson for day ${dayNumber}`);
    return null;
  }

  console.log(`   Topic: ${lesson.topic}`);
  console.log(`   Current headline: ${lesson.marketing_headline}`);
  console.log(`   Current truth: ${lesson.universal_truth}`);

  try {
    const [newHeadline, newTruth] = await Promise.all([
      generateHeadline(lesson.topic),
      generateUniversalTruth(lesson.topic)
    ]);

    console.log(`   ✨ New headline: ${newHeadline}`);
    console.log(`   ✨ New truth: ${newTruth}`);

    // Update the database
    const { error: updateError } = await supabase
      .from('core_lessons')
      .update({
        marketing_headline: newHeadline,
        universal_truth: newTruth,
        updated_at: new Date().toISOString()
      })
      .eq('id', lesson.id);

    if (updateError) {
      console.error(`   ❌ Database update failed: ${updateError.message}`);
      return {
        id: lesson.id,
        day_number: dayNumber,
        topic: lesson.topic,
        old_headline: lesson.marketing_headline,
        new_headline: newHeadline,
        old_truth: lesson.universal_truth,
        new_truth: newTruth,
        success: false,
        error: updateError.message
      };
    }

    // Mark as resolved in validation table
    await supabase
      .from('content_validation_results')
      .update({
        resolved_at: new Date().toISOString(),
        resolved_by: 'fix-headlines.ts',
        resolution_notes: `Auto-fixed. Old headline: "${lesson.marketing_headline}"`
      })
      .eq('day_number', dayNumber)
      .eq('issue_type', 'topic_headline_mismatch');

    console.log(`   ✅ Day ${dayNumber} fixed successfully!`);

    return {
      id: lesson.id,
      day_number: dayNumber,
      topic: lesson.topic,
      old_headline: lesson.marketing_headline,
      new_headline: newHeadline,
      old_truth: lesson.universal_truth,
      new_truth: newTruth,
      success: true
    };

  } catch (err) {
    console.error(`   ❌ Generation failed: ${err}`);
    return {
      id: lesson.id,
      day_number: dayNumber,
      topic: lesson.topic,
      old_headline: lesson.marketing_headline,
      new_headline: '',
      old_truth: lesson.universal_truth,
      new_truth: '',
      success: false,
      error: String(err)
    };
  }
}

/**
 * Fix all mismatched lessons
 */
async function fixAllMismatchedLessons() {
  console.log('🔧 HEADLINE FIXER - Starting bulk fix...\n');
  console.log('=' .repeat(60));

  // Get all mismatched lessons from validation table
  const { data: issues, error: issuesError } = await supabase
    .from('content_validation_results')
    .select('day_number')
    .eq('issue_type', 'topic_headline_mismatch')
    .eq('severity', 'critical')
    .is('resolved_at', null)
    .order('day_number');

  if (issuesError || !issues || issues.length === 0) {
    console.log('✅ No mismatched lessons found!');
    return;
  }

  const dayNumbers = [...new Set(issues.map(i => i.day_number))];
  console.log(`Found ${dayNumbers.length} days to fix\n`);

  // Process in batches of 5 to avoid rate limits
  const batchSize = 5;
  let fixed = 0;
  let failed = 0;
  const results: FixResult[] = [];

  for (let i = 0; i < dayNumbers.length; i += batchSize) {
    const batch = dayNumbers.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(dayNumbers.length / batchSize);

    console.log(`\n📦 Batch ${batchNum}/${totalBatches} (Days ${batch.join(', ')})`);
    console.log('-'.repeat(40));

    // Process batch sequentially to be safe with rate limits
    for (const dayNum of batch) {
      const result = await fixSingleDay(dayNum);
      if (result) {
        results.push(result);
        if (result.success) {
          fixed++;
        } else {
          failed++;
        }
      }
      
      // Small delay between each call
      await new Promise(r => setTimeout(r, 500));
    }

    // Longer delay between batches
    if (i + batchSize < dayNumbers.length) {
      console.log(`\n⏳ Waiting 3s before next batch...`);
      await new Promise(r => setTimeout(r, 3000));
    }
  }

  // Final summary
  console.log('\n' + '='.repeat(60));
  console.log('🎯 HEADLINE FIX COMPLETE');
  console.log('='.repeat(60));
  console.log(`   ✅ Fixed: ${fixed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   📊 Total processed: ${results.length}`);
  console.log('='.repeat(60));

  // Show failed ones
  const failures = results.filter(r => !r.success);
  if (failures.length > 0) {
    console.log('\n⚠️ Failed days:');
    failures.forEach(f => {
      console.log(`   Day ${f.day_number}: ${f.error}`);
    });
  }
}

/**
 * Preview mode - show sample fixes without saving
 */
async function previewFixes(count: number = 5) {
  console.log(`🔍 PREVIEW MODE - Showing ${count} sample fixes\n`);
  console.log('⚠️  This is READ-ONLY. No changes will be saved.\n');
  console.log('='.repeat(60));

  // Get some known bad days
  const { data: issues } = await supabase
    .from('content_validation_results')
    .select('day_number')
    .eq('issue_type', 'topic_headline_mismatch')
    .is('resolved_at', null)
    .order('day_number')
    .limit(count);

  if (!issues || issues.length === 0) {
    console.log('✅ No mismatched lessons found to preview!');
    return;
  }

  const dayNumbers = issues.map(i => i.day_number);

  const { data: lessons } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, marketing_headline, universal_truth')
    .in('day_number', dayNumbers)
    .order('day_number');

  if (!lessons) {
    console.log('❌ Could not fetch lessons');
    return;
  }

  for (const lesson of lessons) {
    console.log(`\n📅 Day ${lesson.day_number}: ${lesson.topic}`);
    console.log('-'.repeat(50));
    
    console.log(`\n   ❌ CURRENT (WRONG):`);
    console.log(`      Headline: ${lesson.marketing_headline}`);
    console.log(`      Truth: ${lesson.universal_truth}`);

    try {
      const [newHeadline, newTruth] = await Promise.all([
        generateHeadline(lesson.topic),
        generateUniversalTruth(lesson.topic)
      ]);

      console.log(`\n   ✅ PROPOSED (NEW):`);
      console.log(`      Headline: ${newHeadline}`);
      console.log(`      Truth: ${newTruth}`);
    } catch (err) {
      console.log(`\n   ⚠️ Generation error: ${err}`);
    }

    // Rate limit
    await new Promise(r => setTimeout(r, 1000));
  }

  console.log('\n' + '='.repeat(60));
  console.log('Preview complete. Run with --fix to apply changes.');
  console.log('='.repeat(60));
}

// ============================================
// CLI HANDLING
// ============================================

const args = process.argv.slice(2);

if (args.includes('--preview')) {
  const countArg = args.find(a => a.startsWith('--count='));
  const count = countArg ? parseInt(countArg.split('=')[1]) : 5;
  previewFixes(count).catch(console.error);

} else if (args.includes('--fix-day')) {
  const dayIndex = args.indexOf('--fix-day');
  const dayNum = parseInt(args[dayIndex + 1]);
  if (isNaN(dayNum)) {
    console.error('Usage: --fix-day <day_number>');
    process.exit(1);
  }
  fixSingleDay(dayNum).catch(console.error);

} else if (args.includes('--fix')) {
  // Confirm before bulk fix
  console.log('⚠️  WARNING: This will modify 251 lessons in the database!');
  console.log('   Press Ctrl+C within 5 seconds to cancel...\n');
  
  setTimeout(() => {
    fixAllMismatchedLessons().catch(console.error);
  }, 5000);

} else {
  console.log(`
🔧 HEADLINE FIXER - Emergency Content Repair Tool

Usage:
  npx ts-node scripts/fix-headlines.ts --preview              # Preview 5 sample fixes (SAFE)
  npx ts-node scripts/fix-headlines.ts --preview --count=10   # Preview 10 fixes (SAFE)
  npx ts-node scripts/fix-headlines.ts --fix-day 57           # Fix a single day
  npx ts-node scripts/fix-headlines.ts --fix                  # FIX ALL 251 LESSONS

Environment Variables Required:
  SUPABASE_URL           (or uses default)
  SUPABASE_SERVICE_KEY   (required)
  ANTHROPIC_API_KEY      (required)
  `);
}

export { fixSingleDay, fixAllMismatchedLessons, previewFixes };


