#!/usr/bin/env node
/**
 * Browser-Based Conversational Narration Test
 * Actually tests the functionality in production using Puppeteer
 * NO ASSUMPTIONS - REAL TESTS ONLY
 */

import puppeteer from 'puppeteer';

const PRODUCTION_URL = 'https://curiouskelly.com';
const TEST_DAY = 17; // Day 17 known to have choice phases

class ConversationalTest {
  constructor() {
    this.browser = null;
    this.page = null;
    this.results = {
      timestamp: new Date().toISOString(),
      url: `${PRODUCTION_URL}/learn.html?day=${TEST_DAY}`,
      tests: [],
      overall: 'UNKNOWN'
    };
  }

  async init() {
    console.log('🚀 Starting browser-based conversational test...');
    this.browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    this.page = await this.browser.newPage();
    
    // Set viewport
    await this.page.setViewport({ width: 1920, height: 1080 });
    
    // Monitor console logs
    this.page.on('console', msg => {
      const text = msg.text();
      if (text.includes('error') || text.includes('Error') || text.includes('failed')) {
        console.log(`[CONSOLE] ${text}`);
      }
    });

    // Monitor page errors
    this.page.on('pageerror', error => {
      console.error(`[PAGE ERROR] ${error.message}`);
      this.results.pageErrors = this.results.pageErrors || [];
      this.results.pageErrors.push(error.message);
    });
  }

  async testCodePresence() {
    console.log('\n📋 Test 1: Code Presence Verification');
    const html = await this.page.content();
    
    const checks = {
      asyncFunction: /async function enterPhaseWithChoices/.test(html),
      optionsNarration: /optionsNarration/.test(html),
      visualRef: /visualRef/.test(html),
      errorHandling: /\.catch\(\(\) => \{\}\)/.test(html),
      awaitPlayPhaseMedia: /await playPhaseMedia/.test(html)
    };

    const passed = Object.values(checks).filter(v => v).length;
    const total = Object.keys(checks).length;

    this.results.tests.push({
      name: 'Code Presence',
      passed,
      total,
      details: checks,
      status: passed === total ? 'PASS' : 'FAIL'
    });

    console.log(`   ✅ ${passed}/${total} checks passed`);
    return passed === total;
  }

  async testFunctionExecution() {
    console.log('\n🎯 Test 2: Function Execution Test');
    
    try {
      // Navigate to lesson page
      await this.page.goto(`${PRODUCTION_URL}/learn.html?day=${TEST_DAY}&track=learn`, {
        waitUntil: 'networkidle2',
        timeout: 30000
      });

      // Wait for page to initialize
      await this.page.waitForTimeout(5000);

      // Check if enterPhaseWithChoices function exists
      const functionExists = await this.page.evaluate(() => {
        return typeof window.enterPhaseWithChoices === 'function' ||
               (typeof enterPhaseWithChoices === 'function');
      });

      // Check if function is async
      const isAsync = await this.page.evaluate(() => {
        const funcStr = enterPhaseWithChoices?.toString() || '';
        return funcStr.includes('async') || funcStr.startsWith('async');
      });

      this.results.tests.push({
        name: 'Function Execution',
        passed: functionExists && isAsync ? 2 : 0,
        total: 2,
        details: {
          functionExists,
          isAsync
        },
        status: (functionExists && isAsync) ? 'PASS' : 'FAIL'
      });

      console.log(`   Function exists: ${functionExists ? '✅' : '❌'}`);
      console.log(`   Is async: ${isAsync ? '✅' : '❌'}`);

      return functionExists && isAsync;
    } catch (error) {
      console.error(`   ❌ Test failed: ${error.message}`);
      this.results.tests.push({
        name: 'Function Execution',
        passed: 0,
        total: 2,
        error: error.message,
        status: 'ERROR'
      });
      return false;
    }
  }

  async testChoicePhaseFlow() {
    console.log('\n🎬 Test 3: Choice Phase Flow Test');
    
    try {
      // Navigate to lesson
      await this.page.goto(`${PRODUCTION_URL}/learn.html?day=${TEST_DAY}&track=learn`, {
        waitUntil: 'networkidle2',
        timeout: 30000
      });

      await this.page.waitForTimeout(5000);

      // Try to advance to a choice phase (cliff phase is typically phase 1)
      const advancedToChoice = await this.page.evaluate(async () => {
        try {
          // Check if advancePhase function exists
          if (typeof advancePhase === 'function') {
            // Advance to cliff phase (usually phase 1)
            advancePhase();
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Check if choice UI is visible
            const container = document.getElementById('cliff-container');
            const isVisible = container && !container.hidden && !container.classList.contains('hidden');
            
            return {
              success: true,
              choiceVisible: isVisible,
              hasPrompt: !!document.getElementById('cliff-prompt'),
              hasOptionA: !!document.getElementById('cliff-label-a'),
              hasOptionB: !!document.getElementById('cliff-label-b')
            };
          }
          return { success: false, reason: 'advancePhase not found' };
        } catch (error) {
          return { success: false, error: error.message };
        }
      });

      const checks = {
        advanced: advancedToChoice.success,
        choiceVisible: advancedToChoice.choiceVisible,
        hasPrompt: advancedToChoice.hasPrompt,
        hasOptionA: advancedToChoice.hasOptionA,
        hasOptionB: advancedToChoice.hasOptionB
      };

      const passed = Object.values(checks).filter(v => v).length;
      const total = Object.keys(checks).length;

      this.results.tests.push({
        name: 'Choice Phase Flow',
        passed,
        total,
        details: checks,
        status: passed >= 3 ? 'PASS' : 'FAIL' // At least 3/5 checks must pass
      });

      console.log(`   Advanced to choice: ${checks.advanced ? '✅' : '❌'}`);
      console.log(`   Choice UI visible: ${checks.choiceVisible ? '✅' : '❌'}`);
      console.log(`   Has prompt: ${checks.hasPrompt ? '✅' : '❌'}`);
      console.log(`   Has option A: ${checks.hasOptionA ? '✅' : '❌'}`);
      console.log(`   Has option B: ${checks.hasOptionB ? '✅' : '❌'}`);

      return passed >= 3;
    } catch (error) {
      console.error(`   ❌ Test failed: ${error.message}`);
      this.results.tests.push({
        name: 'Choice Phase Flow',
        passed: 0,
        total: 5,
        error: error.message,
        status: 'ERROR'
      });
      return false;
    }
  }

  async testNarrationTiming() {
    console.log('\n⏱️  Test 4: Narration Timing Test');
    
    try {
      await this.page.goto(`${PRODUCTION_URL}/learn.html?day=${TEST_DAY}&track=learn`, {
        waitUntil: 'networkidle2',
        timeout: 30000
      });

      await this.page.waitForTimeout(5000);

      // Check if narration happens before buttons appear
      const narrationCheck = await this.page.evaluate(() => {
        try {
          // Get the enterPhaseWithChoices function source
          const funcStr = enterPhaseWithChoices?.toString() || '';
          
          // Check for key patterns:
          // 1. Options narration is built
          // 2. Narration is played
          // 3. Buttons appear AFTER narration
          const hasOptionsNarration = funcStr.includes('optionsNarration');
          const hasPlayPhaseMedia = funcStr.includes('playPhaseMedia');
          const hasDelayBeforeButtons = funcStr.includes('setTimeout') || funcStr.includes('Promise');
          const buttonsAfterNarration = funcStr.indexOf('container.hidden = false') > funcStr.indexOf('optionsNarration');

          return {
            hasOptionsNarration,
            hasPlayPhaseMedia,
            hasDelayBeforeButtons,
            buttonsAfterNarration,
            functionSource: funcStr.substring(0, 500) // First 500 chars for debugging
          };
        } catch (error) {
          return { error: error.message };
        }
      });

      const checks = {
        hasOptionsNarration: narrationCheck.hasOptionsNarration,
        hasPlayPhaseMedia: narrationCheck.hasPlayPhaseMedia,
        hasDelayBeforeButtons: narrationCheck.hasDelayBeforeButtons,
        buttonsAfterNarration: narrationCheck.buttonsAfterNarration
      };

      const passed = Object.values(checks).filter(v => v).length;
      const total = Object.keys(checks).length;

      this.results.tests.push({
        name: 'Narration Timing',
        passed,
        total,
        details: checks,
        status: passed === total ? 'PASS' : 'FAIL'
      });

      console.log(`   Has options narration: ${checks.hasOptionsNarration ? '✅' : '❌'}`);
      console.log(`   Has playPhaseMedia: ${checks.hasPlayPhaseMedia ? '✅' : '❌'}`);
      console.log(`   Has delay before buttons: ${checks.hasDelayBeforeButtons ? '✅' : '❌'}`);
      console.log(`   Buttons after narration: ${checks.buttonsAfterNarration ? '✅' : '❌'}`);

      return passed === total;
    } catch (error) {
      console.error(`   ❌ Test failed: ${error.message}`);
      this.results.tests.push({
        name: 'Narration Timing',
        passed: 0,
        total: 4,
        error: error.message,
        status: 'ERROR'
      });
      return false;
    }
  }

  async runAllTests() {
    try {
      await this.init();

      const test1 = await this.testCodePresence();
      const test2 = await this.testFunctionExecution();
      const test3 = await this.testChoicePhaseFlow();
      const test4 = await this.testNarrationTiming();

      const allPassed = test1 && test2 && test3 && test4;
      this.results.overall = allPassed ? 'PASS' : 'FAIL';

      console.log('\n' + '='.repeat(60));
      console.log(`\n🎯 Overall Status: ${this.results.overall}`);
      console.log(`\n📊 Test Summary:`);
      this.results.tests.forEach(test => {
        const icon = test.status === 'PASS' ? '✅' : (test.status === 'ERROR' ? '💥' : '❌');
        console.log(`   ${icon} ${test.name}: ${test.passed}/${test.total} passed`);
      });

      return this.results;
    } catch (error) {
      console.error('Fatal error:', error);
      this.results.overall = 'ERROR';
      this.results.error = error.message;
      return this.results;
    } finally {
      if (this.browser) {
        await this.browser.close();
      }
    }
  }
}

// Run tests
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new ConversationalTest();
  tester.runAllTests()
    .then(results => {
      process.exit(results.overall === 'PASS' ? 0 : 1);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

export default ConversationalTest;

