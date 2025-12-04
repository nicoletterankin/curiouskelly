/**
 * Kelly Lip-Sync System Test Script
 * 
 * Tests all components of the lip-sync pipeline:
 * 1. Phoneme-to-viseme mapping
 * 2. Timeline generation
 * 3. Real-time analysis
 * 4. Orchestrator integration
 * 
 * Run: node scripts/test-kelly-lipsync.js
 */

import {
  ARPABET_PHONEMES,
  PHONEME_TO_BLENDSHAPES,
  getBlendshapesForPhoneme,
  getVisemeCategory,
  interpolateBlendshapes,
  generateBlendshapeTimeline,
  applyCoarticulation,
} from '../app/lipsync/phoneme-viseme-map.js';

import { KellyLipSyncOrchestrator } from '../app/lipsync/kelly-lipsync-orchestrator.js';

// =============================================================================
// TEST UTILITIES
// =============================================================================

let testsPassed = 0;
let testsFailed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    testsPassed++;
  } catch (error) {
    console.log(`  ✗ ${name}`);
    console.log(`    Error: ${error.message}`);
    testsFailed++;
  }
}

function assertEqual(actual, expected, message = '') {
  if (actual !== expected) {
    throw new Error(`${message} Expected ${expected}, got ${actual}`);
  }
}

function assertExists(value, message = '') {
  if (value === undefined || value === null) {
    throw new Error(`${message} Value is undefined or null`);
  }
}

function assertRange(value, min, max, message = '') {
  if (value < min || value > max) {
    throw new Error(`${message} Expected ${value} to be between ${min} and ${max}`);
  }
}

// =============================================================================
// PHONEME-VISEME MAP TESTS
// =============================================================================

console.log('\n🔤 PHONEME-VISEME MAPPING TESTS\n');

test('All ARPAbet phonemes have blendshape mappings', () => {
  for (const phoneme of Object.keys(ARPABET_PHONEMES)) {
    const blendshapes = getBlendshapesForPhoneme(phoneme);
    assertExists(blendshapes, `No mapping for ${phoneme}`);
  }
});

test('Vowels have jawOpen > 0', () => {
  const vowels = ['AA', 'AE', 'AH', 'AO', 'EH', 'IH', 'IY', 'UH', 'UW'];
  for (const vowel of vowels) {
    const blendshapes = getBlendshapesForPhoneme(vowel);
    if (blendshapes.jawOpen <= 0) {
      throw new Error(`${vowel} should have jawOpen > 0, got ${blendshapes.jawOpen}`);
    }
  }
});

test('Bilabials (P, B, M) have mouthClose = 100', () => {
  const bilabials = ['P', 'B', 'M'];
  for (const phoneme of bilabials) {
    const blendshapes = getBlendshapesForPhoneme(phoneme);
    assertEqual(blendshapes.mouthClose, 100, `${phoneme} mouthClose`);
  }
});

test('Silence has minimal mouth movement', () => {
  const silBlendshapes = getBlendshapesForPhoneme('SIL');
  assertRange(silBlendshapes.jawOpen, 0, 5, 'SIL jawOpen');
  assertRange(silBlendshapes.mouthOpen, 0, 5, 'SIL mouthOpen');
});

test('Viseme categories are correctly assigned', () => {
  assertEqual(getVisemeCategory('AA'), 'A');
  assertEqual(getVisemeCategory('IY'), 'I');
  assertEqual(getVisemeCategory('UW'), 'U');
  assertEqual(getVisemeCategory('M'), 'M');
  assertEqual(getVisemeCategory('SIL'), 'REST');
});

test('Stress markers are stripped from phonemes', () => {
  const aa0 = getBlendshapesForPhoneme('AA0');
  const aa1 = getBlendshapesForPhoneme('AA1');
  const aa = getBlendshapesForPhoneme('AA');
  
  assertEqual(aa0.jawOpen, aa.jawOpen, 'AA0 should match AA');
  assertEqual(aa1.jawOpen, aa.jawOpen, 'AA1 should match AA');
});

// =============================================================================
// INTERPOLATION TESTS
// =============================================================================

console.log('\n🔀 INTERPOLATION TESTS\n');

test('Interpolation at t=0 returns from values', () => {
  const from = { jawOpen: 0, smile: 50 };
  const to = { jawOpen: 100, smile: 0 };
  const result = interpolateBlendshapes(from, to, 0);
  
  assertEqual(result.jawOpen, 0);
  assertEqual(result.smile, 50);
});

test('Interpolation at t=1 returns to values', () => {
  const from = { jawOpen: 0, smile: 50 };
  const to = { jawOpen: 100, smile: 0 };
  const result = interpolateBlendshapes(from, to, 1);
  
  assertEqual(result.jawOpen, 100);
  assertEqual(result.smile, 0);
});

test('Interpolation at t=0.5 returns midpoint', () => {
  const from = { jawOpen: 0 };
  const to = { jawOpen: 100 };
  const result = interpolateBlendshapes(from, to, 0.5);
  
  assertEqual(result.jawOpen, 50);
});

test('Interpolation handles missing keys', () => {
  const from = { jawOpen: 50 };
  const to = { smile: 100 };
  const result = interpolateBlendshapes(from, to, 0.5);
  
  assertEqual(result.jawOpen, 25);  // 50 → 0
  assertEqual(result.smile, 50);    // 0 → 100
});

// =============================================================================
// TIMELINE GENERATION TESTS
// =============================================================================

console.log('\n📊 TIMELINE GENERATION TESTS\n');

test('Timeline generates frames for phoneme sequence', () => {
  const phoneSequence = [
    { phone: 'HH', start: 0.0, end: 0.1 },
    { phone: 'AH', start: 0.1, end: 0.2 },
    { phone: 'L', start: 0.2, end: 0.3 },
    { phone: 'OW', start: 0.3, end: 0.5 },
  ];
  
  const timeline = generateBlendshapeTimeline(phoneSequence, 30);
  
  // At 30fps, 0.5 seconds = ~15 frames
  if (timeline.length < 10 || timeline.length > 20) {
    throw new Error(`Expected 10-20 frames, got ${timeline.length}`);
  }
});

test('Timeline frames have timestamps and blendshapes', () => {
  const phoneSequence = [
    { phone: 'AA', start: 0.0, end: 0.2 },
  ];
  
  const timeline = generateBlendshapeTimeline(phoneSequence, 30);
  
  for (const frame of timeline) {
    assertExists(frame.timestamp, 'Frame should have timestamp');
    assertExists(frame.blendshapes, 'Frame should have blendshapes');
  }
});

test('Timeline handles empty sequence', () => {
  const timeline = generateBlendshapeTimeline([], 30);
  assertEqual(timeline.length, 0);
});

// =============================================================================
// COARTICULATION TESTS
// =============================================================================

console.log('\n🔗 COARTICULATION TESTS\n');

test('Coarticulation adjusts anticipatory shapes', () => {
  const sequence = [
    { phone: 'T', start: 0.0, end: 0.1, blendshapes: { mouthStretchLeft: 20 } },
    { phone: 'UW', start: 0.1, end: 0.3, blendshapes: { mouthFunnel: 60 } },
  ];
  
  const adjusted = applyCoarticulation(sequence);
  
  // T should anticipate UW's rounding
  if (adjusted[0].blendshapes.mouthFunnel <= 0) {
    throw new Error('Expected anticipatory mouthFunnel on T before UW');
  }
});

// =============================================================================
// ORCHESTRATOR TESTS
// =============================================================================

console.log('\n🎭 ORCHESTRATOR TESTS\n');

test('Orchestrator initializes with default config', () => {
  const orchestrator = new KellyLipSyncOrchestrator();
  assertExists(orchestrator.config);
  assertEqual(orchestrator.config.fps, 30);
});

test('Orchestrator factory methods work', () => {
  const lessonsOrch = KellyLipSyncOrchestrator.forLessons();
  assertEqual(lessonsOrch.config.preferredMethod, 'alignment');
  
  const convOrch = KellyLipSyncOrchestrator.forConversation();
  assertEqual(convOrch.config.preferredMethod, 'realtime');
  
  const streamOrch = KellyLipSyncOrchestrator.forStreaming();
  assertEqual(streamOrch.config.preferredMethod, 'streaming');
});

test('Orchestrator estimates alignment from text', () => {
  const orchestrator = new KellyLipSyncOrchestrator();
  const alignment = orchestrator.estimateAlignment('Hello world');
  
  assertEqual(alignment.words.length, 2);
  assertExists(alignment.phones);
  assertEqual(alignment.method, 'estimation');
});

test('Orchestrator smooths blendshapes', () => {
  const orchestrator = new KellyLipSyncOrchestrator({ smoothTransitions: true });
  
  const from = { jawOpen: 0 };
  const to = { jawOpen: 100 };
  const smoothed = orchestrator.smoothBlendshapes(from, to);
  
  // Should be between 0 and 100 due to smoothing
  assertRange(smoothed.jawOpen, 0, 100);
  // Should NOT be exactly 100 due to smoothing
  if (smoothed.jawOpen === 100) {
    throw new Error('Smoothing should prevent instant jump to target');
  }
});

// =============================================================================
// INTEGRATION TEST
// =============================================================================

console.log('\n🔧 INTEGRATION TEST\n');

test('Full pipeline: text → alignment → timeline → blendshapes', () => {
  const orchestrator = new KellyLipSyncOrchestrator();
  
  // Step 1: Estimate alignment from text
  const transcript = 'Hello, I am Kelly!';
  const alignment = orchestrator.estimateAlignment(transcript);
  
  // Step 2: Generate timeline
  const timeline = generateBlendshapeTimeline(alignment.phones, 30);
  
  // Step 3: Verify output
  if (timeline.length === 0) {
    throw new Error('Timeline should not be empty');
  }
  
  // Step 4: Verify blendshapes are valid
  for (const frame of timeline) {
    if (frame.blendshapes.jawOpen < 0 || frame.blendshapes.jawOpen > 100) {
      throw new Error(`Invalid jawOpen: ${frame.blendshapes.jawOpen}`);
    }
  }
  
  console.log(`    Generated ${timeline.length} frames from "${transcript}"`);
});

// =============================================================================
// SUMMARY
// =============================================================================

console.log('\n' + '='.repeat(60));
console.log('📋 TEST SUMMARY');
console.log('='.repeat(60));
console.log(`  ✓ Passed: ${testsPassed}`);
console.log(`  ✗ Failed: ${testsFailed}`);
console.log('='.repeat(60) + '\n');

if (testsFailed > 0) {
  console.log('❌ Some tests failed! Please fix the issues above.\n');
  process.exit(1);
} else {
  console.log('✅ All tests passed! Kelly lip-sync system is ready.\n');
  process.exit(0);
}

