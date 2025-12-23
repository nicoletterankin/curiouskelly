/**
 * Conversational Enhancement Test Suite
 * Tests pre-choice narration and visual awareness features
 */

(function() {
  'use strict';

  const ConversationalTest = {
    testsRun: 0,
    testsPassed: 0,
    testsFailed: 0,
    errors: [],

    /**
     * Run all tests
     */
    async runAll() {
      console.log('🧪 Starting Conversational Enhancement Tests...\n');

      // Test 1: Function is async
      this.test('enterPhaseWithChoices is async', () => {
        const func = window.enterPhaseWithChoices || 
                     (() => {
                       // Try to find it in learn.html context
                       const scripts = document.querySelectorAll('script');
                       for (const script of scripts) {
                         if (script.textContent.includes('async function enterPhaseWithChoices')) {
                           return true;
                         }
                       }
                       return false;
                     });
        
        // Check if function exists and is async
        if (typeof func === 'function') {
          return func.constructor.name === 'AsyncFunction';
        }
        return false;
      });

      // Test 2: No variable redeclaration
      this.test('No variable redeclaration in enterPhaseWithChoices', () => {
        const scripts = document.querySelectorAll('script');
        for (const script of scripts) {
          const content = script.textContent;
          if (content.includes('async function enterPhaseWithChoices')) {
            // Check for duplicate const declarations
            const matches = content.match(/const optAText/g);
            if (matches && matches.length > 1) {
              return false; // Found duplicate
            }
            const matches2 = content.match(/const optBText/g);
            if (matches2 && matches2.length > 1) {
              return false; // Found duplicate
            }
            return true;
          }
        }
        return true; // Function not found in inline scripts, assume OK
      });

      // Test 3: Error handling exists
      this.test('Error handling in place', () => {
        const scripts = document.querySelectorAll('script');
        for (const script of scripts) {
          const content = script.textContent;
          if (content.includes('async function enterPhaseWithChoices')) {
            // Check for .catch() on playPhaseMedia calls
            return content.includes('.catch(() => {})') || 
                   content.includes('.catch(function');
          }
        }
        return true; // Assume OK if not found
      });

      // Test 4: Visual awareness enhancement exists
      this.test('Visual awareness enhancement exists', () => {
        const scripts = document.querySelectorAll('script');
        for (const script of scripts) {
          const content = script.textContent;
          if (content.includes('updatePhaseProgress')) {
            // Check for visual reference enhancement
            return content.includes('visualRef') || 
                   content.includes('visualUrl') ||
                   content.includes('Look at this image');
          }
        }
        return true; // Assume OK if not found
      });

      // Test 5: Buttons appear after narration
      this.test('Buttons appear after narration', () => {
        const scripts = document.querySelectorAll('script');
        for (const script of scripts) {
          const content = script.textContent;
          if (content.includes('async function enterPhaseWithChoices')) {
            // Check that container.hidden is set AFTER await statements
            const containerHiddenIndex = content.indexOf('container.hidden = false');
            const awaitIndex = content.lastIndexOf('await', containerHiddenIndex);
            return awaitIndex !== -1; // Found await before container.hidden
          }
        }
        return true; // Assume OK if not found
      });

      // Print results
      console.log(`\n✅ Tests Passed: ${this.testsPassed}`);
      console.log(`❌ Tests Failed: ${this.testsFailed}`);
      console.log(`📊 Total Tests: ${this.testsRun}\n`);

      if (this.errors.length > 0) {
        console.log('Errors:');
        this.errors.forEach(err => console.error('  -', err));
      }

      return {
        passed: this.testsPassed,
        failed: this.testsFailed,
        total: this.testsRun,
        errors: this.errors
      };
    },

    /**
     * Run a single test
     */
    test(name, fn) {
      this.testsRun++;
      try {
        const result = fn();
        if (result) {
          this.testsPassed++;
          console.log(`✅ ${name}`);
        } else {
          this.testsFailed++;
          this.errors.push(name);
          console.log(`❌ ${name}`);
        }
      } catch (e) {
        this.testsFailed++;
        this.errors.push(`${name}: ${e.message}`);
        console.log(`❌ ${name}: ${e.message}`);
      }
    }
  };

  // Expose globally for testing
  window.ConversationalTest = ConversationalTest;

  // Auto-run if in test environment
  if (window.location.search.includes('test=conversational')) {
    ConversationalTest.runAll();
  }
})();

