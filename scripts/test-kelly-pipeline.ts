#!/usr/bin/env npx ts-node
/**
 * Kelly Pipeline Integration Test
 * 
 * Verifies all pipeline components are working:
 * 1. Eval gates
 * 2. Engine adapters
 * 3. Provider availability
 * 4. Email alerts (console only)
 * 
 * Usage: npx ts-node scripts/test-kelly-pipeline.ts
 */

import * as dotenv from 'dotenv';
dotenv.config();

// Test imports using require (CommonJS compatible)
console.log('╔══════════════════════════════════════════════════════════════╗');
console.log('║       KELLY PIPELINE INTEGRATION TEST                        ║');
console.log('╚══════════════════════════════════════════════════════════════╝\n');

let passed = 0;
let failed = 0;

function test(name: string, condition: boolean): void {
  if (condition) {
    console.log(`✅ ${name}`);
    passed++;
  } else {
    console.log(`❌ ${name}`);
    failed++;
  }
}

async function runTests() {
  console.log('🧪 Running tests...\n');
  
  // Test 1: Import eval gates
  console.log('─── Eval Gates ───');
  try {
    const evalGates = require('../lib/eval-gates');
    test('Eval gates import', true);
    test('SLOP_PATTERNS defined', evalGates.SLOP_PATTERNS?.length > 0);
    test('evaluateContent function', typeof evalGates.evaluateContent === 'function');
    test('evaluateAudio function', typeof evalGates.evaluateAudio === 'function');
    test('evaluateVideo function', typeof evalGates.evaluateVideo === 'function');
    
    // Test good content
    const goodResult = evalGates.evaluateContent({ text: "Hi, I'm Kelly. Today we're learning about compound interest." });
    test('Good content passes', goodResult.passed === true);
    
    // Test slop detection
    const slopResult = evalGates.evaluateContent({ text: "Let's dive deep into the secrets of success!" });
    test('Slop content fails', slopResult.passed === false);
    test('Slop issues detected', slopResult.issues.length > 0);
    
  } catch (e: any) {
    console.log(`❌ Eval gates import failed: ${e.message}`);
    failed += 8;
  }
  
  // Test 2: Import engine types
  console.log('\n─── Engine Types ───');
  try {
    const engineTypes = require('../lib/engines/types');
    test('Engine types import', true);
    test('PROVIDER_FALLBACK_CHAIN defined', engineTypes.PROVIDER_FALLBACK_CHAIN?.length >= 4);
    console.log(`   Fallback order: ${engineTypes.PROVIDER_FALLBACK_CHAIN?.join(' → ')}`);
  } catch (e: any) {
    console.log(`❌ Engine types import failed: ${e.message}`);
    failed += 2;
  }
  
  // Test 3: Import engine adapters
  console.log('\n─── Engine Adapters ───');
  try {
    const engines = require('../lib/engines');
    test('Engines import', true);
    test('HeyGen adapter', engines.heygenAdapter !== undefined);
    test('Sync.so adapter', engines.syncSoAdapter !== undefined);
    test('Fal LatentSync adapter', engines.falLatentsyncAdapter !== undefined);
    test('Replicate adapter', engines.replicateAdapter !== undefined);
    test('getEngine function', typeof engines.getEngine === 'function');
  } catch (e: any) {
    console.log(`❌ Engine adapters import failed: ${e.message}`);
    failed += 6;
  }
  
  // Test 4: Import fallback queue
  console.log('\n─── Fallback Queue ───');
  try {
    const fallbackQueue = require('../lib/fallback-queue');
    test('Fallback queue import', true);
    test('getAvailableProviders function', typeof fallbackQueue.getAvailableProviders === 'function');
    test('submitWithFallback function', typeof fallbackQueue.submitWithFallback === 'function');
    test('runProcessingCycle function', typeof fallbackQueue.runProcessingCycle === 'function');
  } catch (e: any) {
    console.log(`❌ Fallback queue import failed: ${e.message}`);
    failed += 4;
  }
  
  // Test 5: Import email alerts
  console.log('\n─── Email Alerts ───');
  try {
    const emailAlerts = require('../lib/email-alerts');
    test('Email alerts import', true);
    test('sendAlert function', typeof emailAlerts.sendAlert === 'function');
    test('notifyEvalFailure function', typeof emailAlerts.notifyEvalFailure === 'function');
    test('notifyJobFailure function', typeof emailAlerts.notifyJobFailure === 'function');
    
    // Test alert formatting (console only)
    console.log('\n   Testing alert format (console only):');
    await emailAlerts.sendAlert({
      type: 'pipeline_error',
      subject: '[TEST] Pipeline Test Alert',
      body: 'This is a test alert from the pipeline integration test.',
    });
    test('Alert sent (console)', true);
    
  } catch (e: any) {
    console.log(`❌ Email alerts import failed: ${e.message}`);
    failed += 5;
  }
  
  // Test 6: Environment configuration
  console.log('\n─── Environment ───');
  test('ELEVENLABS_API_KEY', !!process.env.ELEVENLABS_API_KEY);
  test('At least one video provider', 
    !!process.env.HEYGEN_API_KEY || 
    !!process.env.SYNC_LABS_API_KEY || 
    !!process.env.FAL_KEY || 
    !!process.env.REPLICATE_API_TOKEN
  );
  
  // Print summary
  console.log('\n' + '═'.repeat(60));
  console.log(`TEST RESULTS: ${passed} passed, ${failed} failed`);
  console.log('═'.repeat(60));
  
  if (failed > 0) {
    console.log('\n⚠️  Some tests failed. Check configuration.');
    process.exit(1);
  } else {
    console.log('\n✅ All tests passed! Pipeline ready.');
    process.exit(0);
  }
}

runTests().catch(e => {
  console.error('Test runner error:', e);
  process.exit(1);
});
