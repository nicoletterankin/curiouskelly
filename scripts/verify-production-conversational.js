#!/usr/bin/env node
/**
 * Enterprise-Grade Production Verification Script
 * Hard-coded verification of conversational narration functionality
 * NO WISHFUL THINKING - ACTUAL TESTS ONLY
 */

import https from 'https';
import { URL } from 'url';

const PRODUCTION_URL = 'https://curiouskelly.com';
const TEST_DAYS = [1, 17, 100, 200, 351]; // Days known to have choice phases

const VERIFICATION_CHECKS = {
  codePresence: {
    name: 'Code Presence Verification',
    critical: true,
    checks: [
      'async function enterPhaseWithChoices',
      'optionsNarration',
      'visualRef',
      'await playPhaseMedia',
      '.catch(() => {})'
    ]
  },
  functionSignature: {
    name: 'Function Signature Verification',
    critical: true,
    checks: [
      'async function enterPhaseWithChoices\\(atom\\)',
      'const optAText =',
      'const optBText =',
      'optionsNarration =',
      'await new Promise\\(resolve => setTimeout'
    ]
  },
  errorHandling: {
    name: 'Error Handling Verification',
    critical: true,
    checks: [
      '\\.catch\\(\\(\\) => \\{\\}\\)',
      'try.*catch',
      'if \\(.*\\) \\{.*\\} catch'
    ]
  },
  visualAwareness: {
    name: 'Visual Awareness Verification',
    critical: false,
    checks: [
      'visualRef',
      'atom\\?\\.visualUrl',
      'LessonVisualDisplay\\.show'
    ]
  }
};

class ProductionVerifier {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      url: PRODUCTION_URL,
      checks: {},
      overall: 'UNKNOWN',
      criticalFailures: []
    };
  }

  async fetchHTML(url, followRedirects = true, maxRedirects = 5) {
    return new Promise((resolve, reject) => {
      if (maxRedirects <= 0) {
        reject(new Error('Too many redirects'));
        return;
      }

      const parsedUrl = new URL(url);
      const options = {
        hostname: parsedUrl.hostname,
        path: parsedUrl.pathname + parsedUrl.search,
        method: 'GET',
        headers: {
          'User-Agent': 'Production-Verifier/1.0'
        }
      };

      const req = https.request(options, (res) => {
        // Handle redirects
        if (followRedirects && (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307 || res.statusCode === 308)) {
          const location = res.headers.location;
          if (location) {
            const redirectUrl = location.startsWith('http') ? location : `https://${parsedUrl.hostname}${location}`;
            return this.fetchHTML(redirectUrl, followRedirects, maxRedirects - 1).then(resolve).catch(reject);
          }
        }

        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolve(data);
          } else {
            reject(new Error(`HTTP ${res.statusCode}`));
          }
        });
      });

      req.on('error', reject);
      req.setTimeout(10000, () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });
      req.end();
    });
  }

  verifyCodePresence(html) {
    const check = VERIFICATION_CHECKS.codePresence;
    const results = {
      name: check.name,
      critical: check.critical,
      passed: 0,
      failed: 0,
      details: []
    };

    check.checks.forEach(pattern => {
      // Special handling for .catch pattern - check for variations
      let found = false;
      if (pattern === '.catch(() => {})') {
        // Check for various whitespace variations
        found = /\.catch\s*\(\s*\(\)\s*=>\s*\{\s*\}\)/.test(html) ||
                /\.catch\s*\(\s*\(\)\s*=>\s*\{\}\)/.test(html) ||
                /\.catch\s*\(\(\)\s*=>\s*\{\}\)/.test(html);
      } else {
        const regex = new RegExp(pattern, 'i');
        found = regex.test(html);
      }
      
      if (found) {
        results.passed++;
        results.details.push({ pattern, status: 'PASS' });
      } else {
        results.failed++;
        results.details.push({ pattern, status: 'FAIL' });
        if (check.critical) {
          this.results.criticalFailures.push(`${check.name}: Missing ${pattern}`);
        }
      }
    });

    return results;
  }

  verifyFunctionSignature(html) {
    const check = VERIFICATION_CHECKS.functionSignature;
    const results = {
      name: check.name,
      critical: check.critical,
      passed: 0,
      failed: 0,
      details: []
    };

    check.checks.forEach(pattern => {
      const regex = new RegExp(pattern);
      const found = regex.test(html);
      if (found) {
        results.passed++;
        results.details.push({ pattern, status: 'PASS' });
      } else {
        results.failed++;
        results.details.push({ pattern, status: 'FAIL' });
        if (check.critical) {
          this.results.criticalFailures.push(`${check.name}: Missing ${pattern}`);
        }
      }
    });

    return results;
  }

  verifyErrorHandling(html) {
    const check = VERIFICATION_CHECKS.errorHandling;
    const results = {
      name: check.name,
      critical: check.critical,
      passed: 0,
      failed: 0,
      details: []
    };

    // Check for error handling around playPhaseMedia calls
    const playPhaseMediaPattern = /playPhaseMedia\([^)]+\)\.catch\(\(\) => \{\}\)/g;
    const matches = html.match(playPhaseMediaPattern);
    const count = matches ? matches.length : 0;

    if (count >= 2) { // Should have at least 2 error-handled calls
      results.passed++;
      results.details.push({ 
        check: 'playPhaseMedia error handling', 
        status: 'PASS', 
        count 
      });
    } else {
      results.failed++;
      results.details.push({ 
        check: 'playPhaseMedia error handling', 
        status: 'FAIL', 
        count,
        expected: '>= 2'
      });
      if (check.critical) {
        this.results.criticalFailures.push(`${check.name}: Insufficient error handling`);
      }
    }

    return results;
  }

  verifyVisualAwareness(html) {
    const check = VERIFICATION_CHECKS.visualAwareness;
    const results = {
      name: check.name,
      critical: check.critical,
      passed: 0,
      failed: 0,
      details: []
    };

    check.checks.forEach(pattern => {
      const regex = new RegExp(pattern);
      const found = regex.test(html);
      if (found) {
        results.passed++;
        results.details.push({ pattern, status: 'PASS' });
      } else {
        results.failed++;
        results.details.push({ pattern, status: 'FAIL' });
      }
    });

    return results;
  }

  async verifyProduction() {
    console.log('🔍 Enterprise-Grade Production Verification');
    console.log('=' .repeat(60));
    console.log(`URL: ${PRODUCTION_URL}`);
    console.log(`Time: ${this.results.timestamp}`);
    console.log('');

    try {
      // Fetch production HTML
      console.log('📥 Fetching production HTML...');
      const html = await this.fetchHTML(`${PRODUCTION_URL}/learn.html`);
      console.log(`✅ Fetched ${(html.length / 1024).toFixed(2)} KB\n`);

      // Run all verification checks
      console.log('🔬 Running verification checks...\n');

      this.results.checks.codePresence = this.verifyCodePresence(html);
      this.results.checks.functionSignature = this.verifyFunctionSignature(html);
      this.results.checks.errorHandling = this.verifyErrorHandling(html);
      this.results.checks.visualAwareness = this.verifyVisualAwareness(html);

      // Calculate overall status
      const criticalChecks = Object.values(this.results.checks).filter(c => c.critical);
      const allCriticalPassed = criticalChecks.every(c => c.failed === 0);
      const anyCriticalFailed = criticalChecks.some(c => c.failed > 0);

      if (anyCriticalFailed) {
        this.results.overall = 'FAIL';
      } else if (allCriticalPassed) {
        this.results.overall = 'PASS';
      } else {
        this.results.overall = 'WARN';
      }

      // Print results
      this.printResults();

      return this.results;

    } catch (error) {
      console.error('❌ Verification failed:', error.message);
      this.results.overall = 'ERROR';
      this.results.error = error.message;
      return this.results;
    }
  }

  printResults() {
    console.log('📊 Verification Results');
    console.log('=' .repeat(60));

    Object.values(this.results.checks).forEach(check => {
      const status = check.failed === 0 ? '✅' : (check.critical ? '❌' : '⚠️');
      console.log(`\n${status} ${check.name}`);
      console.log(`   Passed: ${check.passed}, Failed: ${check.failed}`);
      
      if (check.failed > 0) {
        check.details.filter(d => d.status === 'FAIL').forEach(detail => {
          console.log(`   ❌ Missing: ${detail.pattern || detail.check}`);
        });
      }
    });

    console.log('\n' + '=' .repeat(60));
    console.log(`\n🎯 Overall Status: ${this.results.overall}`);

    if (this.results.criticalFailures.length > 0) {
      console.log('\n🚨 Critical Failures:');
      this.results.criticalFailures.forEach(failure => {
        console.log(`   ❌ ${failure}`);
      });
    }

    console.log('');
  }

  generateReport() {
    return {
      ...this.results,
      summary: {
        totalChecks: Object.values(this.results.checks).reduce((sum, c) => sum + c.passed + c.failed, 0),
        passedChecks: Object.values(this.results.checks).reduce((sum, c) => sum + c.passed, 0),
        failedChecks: Object.values(this.results.checks).reduce((sum, c) => sum + c.failed, 0),
        criticalFailures: this.results.criticalFailures.length
      }
    };
  }
}

// Run verification
const isMainModule = import.meta.url === `file://${process.argv[1]}` || 
                     import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/'));

if (isMainModule || process.argv[1]?.includes('verify-production-conversational')) {
  const verifier = new ProductionVerifier();
  verifier.verifyProduction()
    .then(results => {
      const report = verifier.generateReport();
      process.exit(report.overall === 'PASS' ? 0 : 1);
    })
    .catch(error => {
      console.error('Fatal error:', error);
      process.exit(1);
    });
}

export default ProductionVerifier;

