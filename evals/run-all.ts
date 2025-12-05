#!/usr/bin/env npx ts-node
/**
 * Master Eval Runner
 * Runs all evaluation suites
 * 
 * Usage: npx ts-node evals/run-all.ts
 * Or:    npm run eval
 */

import { runEvals as runVoiceEvals } from './kelly-voice-eval.js';
import { runEvals as runLifetimeEvals } from './lifetime-experience-eval.js';

async function main() {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║                 CURIOUS KELLY EVAL SUITE                     ║');
  console.log('║                                                              ║');
  console.log('║   Testing Kelly\'s Voice + Lifetime Experience System        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('\n');
  
  let hasFailures = false;
  
  // Run Kelly Voice Evals
  console.log('━'.repeat(60));
  console.log('  SUITE 1: KELLY VOICE');
  console.log('━'.repeat(60));
  
  try {
    runVoiceEvals();
  } catch (e) {
    hasFailures = true;
    console.log('Voice evals had failures');
  }
  
  // Run Lifetime Experience Evals
  console.log('\n');
  console.log('━'.repeat(60));
  console.log('  SUITE 2: LIFETIME EXPERIENCE');
  console.log('━'.repeat(60));
  
  try {
    await runLifetimeEvals();
  } catch (e) {
    hasFailures = true;
    console.log('Lifetime evals had failures');
  }
  
  // Final summary
  console.log('\n');
  console.log('╔══════════════════════════════════════════════════════════════╗');
  if (hasFailures) {
    console.log('║           ❌ SOME EVALS FAILED - SEE ABOVE                   ║');
  } else {
    console.log('║           ✅ ALL EVALS PASSED                                ║');
  }
  console.log('╚══════════════════════════════════════════════════════════════╝');
  console.log('\n');
  
  if (hasFailures) {
    process.exit(1);
  }
}

main().catch(e => {
  console.error('Eval runner error:', e);
  process.exit(1);
});

