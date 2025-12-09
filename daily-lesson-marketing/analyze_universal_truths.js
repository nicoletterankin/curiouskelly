/**
 * Analyze Universal Truths
 * Flags generic platitudes that don't create curiosity or tension
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Patterns that indicate generic platitudes
const genericPatterns = [
  /^[\w\s]+ is important/i,
  /^[\w\s]+ matters/i,
  /^[\w\s]+ helps us/i,
  /^[\w\s]+ connects/i,
  /^[\w\s]+ brings people together/i,
  /^[\w\s]+ enriches/i,
  /^[\w\s]+ enhances/i,
  /^[\w\s]+ improves/i,
  /^[\w\s]+ empowers/i,
  /^[\w\s]+ enables/i,
  /^[\w\s]+ fosters/i,
  /^[\w\s]+ promotes/i,
  /^[\w\s]+ encourages/i,
  /^[\w\s]+ supports/i,
  /^[\w\s]+ strengthens/i,
  /^[\w\s]+ builds/i,
  /^[\w\s]+ creates/i,
  /^[\w\s]+ develops/i,
  /^[\w\s]+ cultivates/i,
  /essential for/i,
  /key to success/i,
  /vital for/i,
  /crucial for/i,
  /necessary for/i,
  /foundation of/i,
  /power of/i,
];

// Words that signal vagueness
const vagueWords = [
  'important', 'valuable', 'meaningful', 'significant', 'essential',
  'vital', 'crucial', 'necessary', 'fundamental', 'key',
  'powerful', 'transformative', 'life-changing', 'impactful',
  'better', 'positive', 'good', 'beneficial', 'helpful',
  'success', 'growth', 'potential', 'possibilities',
];

// Structure patterns that are boring
const boringStructures = [
  /^[\w\s]+ (?:is|are) [\w\s]+\.$/,  // "X is Y." - simple definition
  /^[\w\s]+ (?:can|helps|allows) [\w\s]+\.$/,  // "X helps Y." - obvious benefit
];

function analyzeUniversalTruth(truth, topic) {
  const issues = [];
  const lowerTruth = truth.toLowerCase();
  
  // Check for generic patterns
  for (const pattern of genericPatterns) {
    if (pattern.test(truth)) {
      issues.push('GENERIC_PATTERN');
      break;
    }
  }
  
  // Count vague words
  let vagueCount = 0;
  for (const word of vagueWords) {
    if (lowerTruth.includes(word)) {
      vagueCount++;
    }
  }
  if (vagueCount >= 2) {
    issues.push('TOO_VAGUE');
  }
  
  // Check if it's just a definition (doesn't create tension)
  if (truth.match(/^[\w\s]+ (?:is|are) (?:a |an |the )?[\w\s]+\.?$/)) {
    // Simple "X is Y" structure - might be okay if Y is surprising
    const parts = truth.split(/\s+(?:is|are)\s+/);
    if (parts.length === 2 && parts[1].split(' ').length < 6) {
      issues.push('SIMPLE_DEFINITION');
    }
  }
  
  // Check if topic word is missing (might be too generic)
  const topicWords = topic.toLowerCase().split(/[\s-]+/);
  let hasTopicReference = false;
  for (const word of topicWords) {
    if (word.length > 3 && lowerTruth.includes(word)) {
      hasTopicReference = true;
      break;
    }
  }
  if (!hasTopicReference && topic.length > 4) {
    issues.push('NO_TOPIC_REFERENCE');
  }
  
  // Check for "could apply to anything" patterns
  const anyTopicPatterns = [
    /understanding .* leads to/i,
    /learning about .* helps/i,
    /knowing .* is important/i,
    /appreciating .* enriches/i,
    /exploring .* opens/i,
  ];
  for (const pattern of anyTopicPatterns) {
    if (pattern.test(truth)) {
      issues.push('APPLIES_TO_ANYTHING');
      break;
    }
  }
  
  // Check length - too short often means too simple
  if (truth.length < 40) {
    issues.push('TOO_SHORT');
  }
  
  // Check for lack of specific/surprising detail
  const surprisingIndicators = [
    /\d+/, // numbers
    /million|billion|thousand/i,
    /only|just|merely/i,
    /actually|surprisingly|remarkably/i,
    /despite|although|even though/i,
    /but|however|yet/i,
    /more than|less than/i,
    /before|after|during/i,
    /without|never|always/i,
  ];
  
  let hasSurprise = false;
  for (const pattern of surprisingIndicators) {
    if (pattern.test(truth)) {
      hasSurprise = true;
      break;
    }
  }
  
  // Final verdict
  let verdict = 'OK';
  let severity = 0;
  
  if (issues.includes('APPLIES_TO_ANYTHING')) {
    verdict = 'BAD';
    severity = 3;
  } else if (issues.includes('GENERIC_PATTERN') && issues.includes('TOO_VAGUE')) {
    verdict = 'BAD';
    severity = 3;
  } else if (issues.includes('GENERIC_PATTERN') || issues.includes('TOO_VAGUE')) {
    verdict = 'WEAK';
    severity = 2;
  } else if (issues.includes('SIMPLE_DEFINITION') && !hasSurprise) {
    verdict = 'WEAK';
    severity = 2;
  } else if (issues.length >= 2) {
    verdict = 'WEAK';
    severity = 1;
  }
  
  return {
    verdict,
    severity,
    issues,
    hasSurprise,
  };
}

async function main() {
  console.error('Fetching all lessons...');
  
  const { data: lessons, error } = await supabase
    .from('core_lessons')
    .select('id, day_number, topic, universal_truth')
    .order('day_number', { ascending: true });

  if (error) {
    console.error('Error:', error);
    return;
  }

  console.error(`Analyzing ${lessons.length} universal truths...`);
  
  const results = {
    total: lessons.length,
    bad: [],
    weak: [],
    ok: [],
  };
  
  for (const lesson of lessons) {
    const analysis = analyzeUniversalTruth(lesson.universal_truth, lesson.topic);
    
    const entry = {
      day_number: lesson.day_number,
      topic: lesson.topic,
      universal_truth: lesson.universal_truth,
      issues: analysis.issues,
    };
    
    if (analysis.verdict === 'BAD') {
      results.bad.push(entry);
    } else if (analysis.verdict === 'WEAK') {
      results.weak.push(entry);
    } else {
      results.ok.push(entry);
    }
  }
  
  // Output summary
  const output = {
    summary: {
      total: results.total,
      bad_count: results.bad.length,
      weak_count: results.weak.length,
      ok_count: results.ok.length,
      needs_rewrite_percentage: Math.round((results.bad.length + results.weak.length) / results.total * 100),
    },
    bad_universal_truths: results.bad,
    weak_universal_truths: results.weak,
    // ok_universal_truths: results.ok,  // Uncomment to see good examples
  };
  
  console.log(JSON.stringify(output, null, 2));
}

main().catch(console.error);









