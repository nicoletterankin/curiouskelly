#!/usr/bin/env node
/**
 * Lesson Test Harness
 *
 * Simulates the lesson player flow for verification.
 * Tests each phase transition and choice interaction.
 *
 * Usage:
 *   node test-lesson-harness.js path/to/lesson-dna.json [--age 18-35]
 *   node test-lesson-harness.js --all [--summary]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================================
// TEST HARNESS
// ============================================================================

class LessonTestHarness {
  constructor() {
    this.results = [];
  }

  /**
   * Test a lesson for a specific age bucket
   */
  testLesson(lesson, ageBucket = '18-35') {
    const result = {
      lessonId: lesson.id,
      ageBucket,
      phases: [],
      passed: true,
      errors: [],
    };

    const variant = lesson.ageVariants?.[ageBucket];
    if (!variant) {
      result.passed = false;
      result.errors.push(`No variant for age bucket ${ageBucket}`);
      return result;
    }

    if (!variant.phases || variant.phases.length === 0) {
      result.passed = false;
      result.errors.push('No phases defined');
      return result;
    }

    // Simulate phase flow
    const phaseOrder = ['welcome', 'teaching', 'practice', 'wisdom'];

    for (const expectedPhase of phaseOrder) {
      const phase = variant.phases.find(p => p.id === expectedPhase);

      if (!phase) {
        result.passed = false;
        result.errors.push(`Missing phase: ${expectedPhase}`);
        continue;
      }

      const phaseResult = {
        id: phase.id,
        contentLength: phase.content?.length || 0,
        hasContent: !!phase.content && phase.content.length >= 10,
        choices: [],
        passed: true,
      };

      // Test content
      if (!phaseResult.hasContent) {
        phaseResult.passed = false;
        result.errors.push(`${expectedPhase}: Content too short or missing`);
      }

      // Test choices for interactive phases
      if (expectedPhase === 'teaching' || expectedPhase === 'practice') {
        if (!phase.choices || phase.choices.length < 2) {
          phaseResult.passed = false;
          result.errors.push(`${expectedPhase}: Must have 2 choices`);
        } else {
          for (let i = 0; i < phase.choices.length; i++) {
            const choice = phase.choices[i];
            const choiceResult = {
              text: choice.text?.substring(0, 40) + '...',
              hasText: !!choice.text && choice.text.length >= 5,
              hasResponse: !!choice.response && choice.response.length >= 10,
              passed: true,
            };

            if (!choiceResult.hasText) {
              choiceResult.passed = false;
              result.errors.push(`${expectedPhase}.choice[${i}]: Text too short`);
            }
            if (!choiceResult.hasResponse) {
              choiceResult.passed = false;
              result.errors.push(`${expectedPhase}.choice[${i}]: Response too short`);
            }

            phaseResult.choices.push(choiceResult);
          }
        }
      }

      if (!phaseResult.passed) {
        result.passed = false;
      }

      result.phases.push(phaseResult);
    }

    this.results.push(result);
    return result;
  }

  /**
   * Print detailed test results
   */
  printResult(result, verbose = true) {
    const status = result.passed ? '✅' : '❌';
    console.log(`\n${status} ${result.lessonId} [${result.ageBucket}]`);

    if (verbose && result.phases.length > 0) {
      console.log('   Phase Flow:');
      for (const phase of result.phases) {
        const phaseStatus = phase.passed ? '✓' : '✗';
        console.log(`   ${phaseStatus} ${phase.id} (${phase.contentLength} chars)`);

        if (phase.choices.length > 0) {
          for (const choice of phase.choices) {
            const choiceStatus = choice.passed ? '✓' : '✗';
            console.log(`     ${choiceStatus} Choice: ${choice.text}`);
          }
        }
      }
    }

    if (result.errors.length > 0) {
      console.log('   Errors:');
      result.errors.forEach(e => console.log(`   - ${e}`));
    }
  }

  /**
   * Print summary of all results
   */
  printSummary() {
    const passed = this.results.filter(r => r.passed).length;
    const failed = this.results.filter(r => !r.passed).length;
    const total = this.results.length;

    console.log('\n' + '='.repeat(60));
    console.log('TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total lessons tested: ${total}`);
    console.log(`Passed: ${passed} (${Math.round(passed / total * 100)}%)`);
    console.log(`Failed: ${failed}`);

    if (failed > 0) {
      console.log('\nFailed lessons:');
      this.results
        .filter(r => !r.passed)
        .forEach(r => console.log(`  - ${r.lessonId} [${r.ageBucket}]: ${r.errors[0]}`));
    }
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  const testAll = args.includes('--all');
  const summaryOnly = args.includes('--summary');
  const specificFile = args.find(a => a.endsWith('.json'));
  const ageArg = args.find(a => a.startsWith('--age='));
  const ageBucket = ageArg ? ageArg.split('=')[1] : '18-35';

  console.log('='.repeat(60));
  console.log('LESSON TEST HARNESS');
  console.log('='.repeat(60));

  const harness = new LessonTestHarness();
  let filesToTest = [];

  if (specificFile) {
    filesToTest = [path.resolve(specificFile)];
  } else if (testAll) {
    const dirs = [
      path.join(__dirname, '..', 'lessons'),
      path.join(__dirname, '..', '_archive', 'curious-kellly', 'backend', 'config', 'lessons'),
    ];

    for (const dir of dirs) {
      if (fs.existsSync(dir)) {
        const files = fs.readdirSync(dir)
          .filter(f => f.endsWith('-dna.json') || f.endsWith('_dna.json'))
          .map(f => path.join(dir, f));
        filesToTest.push(...files);
      }
    }

    // Remove duplicates
    const seen = new Set();
    filesToTest = filesToTest.filter(f => {
      const name = path.basename(f);
      if (seen.has(name)) return false;
      seen.add(name);
      return true;
    });
  } else {
    console.log('\nUsage:');
    console.log('  node test-lesson-harness.js path/to/lesson.json');
    console.log('  node test-lesson-harness.js --all [--summary]');
    console.log('  node test-lesson-harness.js lesson.json --age=6-12');
    process.exit(0);
  }

  for (const file of filesToTest) {
    try {
      const content = fs.readFileSync(file, 'utf8');
      const lesson = JSON.parse(content);
      const result = harness.testLesson(lesson, ageBucket);

      if (!summaryOnly) {
        harness.printResult(result, !testAll);
      }
    } catch (error) {
      console.log(`\n❌ ${path.basename(file)}: ${error.message}`);
    }
  }

  harness.printSummary();

  const allPassed = harness.results.every(r => r.passed);
  process.exit(allPassed ? 0 : 1);
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(2);
});
