/**
 * Master Evaluation Runner
 * 
 * Runs all evaluations and generates a comprehensive report.
 * 
 * Usage:
 *   node scripts/eval-all.cjs                    - Full report
 *   node scripts/eval-all.cjs public/about.html  - Single page
 *   node scripts/eval-all.cjs --summary          - Quick summary only
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  marketingPages: [
    'about.html', 'accessibility.html', 'affiliates.html', 'ambassador.html',
    'api.html', 'careers.html', 'commons.html', 'contact.html',
    'curriculum.html', 'diversity.html', 'enterprise.html', 'gifts.html',
    'group.html', 'help.html', 'impact.html', 'join.html', 'missions.html',
    'newsroom.html', 'partner.html', 'perspectives.html', 'pricing.html',
    'privacy.html', 'social.html', 'terms.html', 'trust.html'
  ],
  
  passThresholds: {
    design: 85,
    links: 100, // All links must work
    accessibility: 70
  },
  
  scriptsDir: __dirname,
  publicDir: path.join(__dirname, '..', 'public')
};

// Note: We run sub-scripts via execSync rather than importing
// to avoid module conflicts

// ============================================
// RUN EVALUATIONS
// ============================================

function runDesignEval(filePath) {
  try {
    const result = execSync(`node "${path.join(CONFIG.scriptsDir, 'eval-migration.cjs')}" "${filePath}"`, {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Parse score from output
    const scoreMatch = result.match(/Score: (\d+)\/100/);
    const passed = result.includes('PASSED');
    
    return {
      score: scoreMatch ? parseInt(scoreMatch[1]) : 0,
      passed,
      output: result
    };
  } catch (e) {
    return {
      score: 0,
      passed: false,
      output: e.stdout || e.message
    };
  }
}

function runLinkEval(filePath) {
  try {
    const result = execSync(`node "${path.join(CONFIG.scriptsDir, 'eval-links.cjs')}" "${filePath}"`, {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    const passed = result.includes('PASSED') || result.includes('All links valid');
    const brokenMatch = result.match(/Broken Links Found: (\d+)/);
    
    return {
      brokenCount: brokenMatch ? parseInt(brokenMatch[1]) : 0,
      passed,
      output: result
    };
  } catch (e) {
    return {
      brokenCount: -1,
      passed: false,
      output: e.stdout || e.message
    };
  }
}

function runAccessibilityEval(filePath) {
  try {
    const result = execSync(`node "${path.join(CONFIG.scriptsDir, 'eval-accessibility.cjs')}" "${filePath}"`, {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    const scoreMatch = result.match(/Score: (\d+)\/100/);
    const passed = result.includes('PASSED');
    
    return {
      score: scoreMatch ? parseInt(scoreMatch[1]) : 0,
      passed,
      output: result
    };
  } catch (e) {
    return {
      score: 0,
      passed: false,
      output: e.stdout || e.message
    };
  }
}

// ============================================
// COMPREHENSIVE EVALUATION
// ============================================

function evaluatePage(pageName) {
  const filePath = path.join(CONFIG.publicDir, pageName);
  
  if (!fs.existsSync(filePath)) {
    return {
      page: pageName,
      exists: false,
      overall: 'MISSING'
    };
  }
  
  const design = runDesignEval(filePath);
  const links = runLinkEval(filePath);
  const a11y = runAccessibilityEval(filePath);
  
  // Calculate overall status
  const allPassed = design.passed && links.passed && a11y.passed;
  const overallScore = Math.round((design.score + a11y.score) / 2);
  
  return {
    page: pageName,
    exists: true,
    design: {
      score: design.score,
      passed: design.passed
    },
    links: {
      broken: links.brokenCount,
      passed: links.passed
    },
    accessibility: {
      score: a11y.score,
      passed: a11y.passed
    },
    overall: allPassed ? 'READY' : 'NEEDS WORK',
    overallScore
  };
}

// ============================================
// REPORTING
// ============================================

function generateReport(results, summaryOnly = false) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════════════╗');
  console.log('║         COMPREHENSIVE MIGRATION EVALUATION REPORT                  ║');
  console.log('╚════════════════════════════════════════════════════════════════════╝');
  console.log(`\n  Generated: ${new Date().toISOString()}`);
  
  // Overall stats
  const ready = results.filter(r => r.overall === 'READY');
  const needsWork = results.filter(r => r.overall === 'NEEDS WORK');
  const missing = results.filter(r => r.overall === 'MISSING');
  
  const avgDesign = results.filter(r => r.design).reduce((sum, r) => sum + r.design.score, 0) / 
                    results.filter(r => r.design).length || 0;
  const avgA11y = results.filter(r => r.accessibility).reduce((sum, r) => sum + r.accessibility.score, 0) / 
                  results.filter(r => r.accessibility).length || 0;
  
  console.log('\n' + '═'.repeat(70));
  console.log('  EXECUTIVE SUMMARY');
  console.log('═'.repeat(70));
  
  console.log(`
  📊 Overall Status:
     ┌─────────────────────────────────────────┐
     │  Ready for Production:  ${String(ready.length).padStart(2)} pages  ✅    │
     │  Needs Migration:       ${String(needsWork.length).padStart(2)} pages  ⚠️     │
     │  Missing Files:         ${String(missing.length).padStart(2)} pages  ❌    │
     └─────────────────────────────────────────┘
  
  📈 Average Scores:
     Design System Compliance: ${avgDesign.toFixed(1)}/100
     Accessibility:            ${avgA11y.toFixed(1)}/100
  
  🎯 Migration Progress: ${Math.round((ready.length / results.length) * 100)}%
  `);
  
  // Progress bar
  const progressWidth = 40;
  const filledWidth = Math.round((ready.length / results.length) * progressWidth);
  const progressBar = '█'.repeat(filledWidth) + '░'.repeat(progressWidth - filledWidth);
  console.log(`  [${progressBar}] ${ready.length}/${results.length}`);
  
  if (!summaryOnly) {
    // Detailed table
    console.log('\n' + '═'.repeat(70));
    console.log('  DETAILED RESULTS');
    console.log('═'.repeat(70));
    
    console.log('\n  | Page                      | Design | A11y  | Links | Status     |');
    console.log('  |---------------------------|--------|-------|-------|------------|');
    
    results
      .sort((a, b) => {
        // Sort: READY first, then NEEDS WORK, then MISSING
        const order = { 'READY': 0, 'NEEDS WORK': 1, 'MISSING': 2 };
        return order[a.overall] - order[b.overall];
      })
      .forEach(r => {
        if (r.overall === 'MISSING') {
          console.log(`  | ${r.page.padEnd(25)} |   --   |  --   |  --   | ❌ MISSING  |`);
        } else {
          const designIcon = r.design.passed ? '✅' : '❌';
          const a11yIcon = r.accessibility.passed ? '✅' : '⚠️';
          const linksIcon = r.links.passed ? '✅' : '❌';
          const statusIcon = r.overall === 'READY' ? '✅' : '⚠️';
          
          console.log(`  | ${r.page.padEnd(25)} | ${String(r.design.score).padStart(3)}/100${designIcon}| ${String(r.accessibility.score).padStart(2)}/100${a11yIcon}| ${linksIcon}     | ${statusIcon} ${r.overall.padEnd(9)} |`);
        }
      });
    
    // Pages needing work
    if (needsWork.length > 0) {
      console.log('\n' + '═'.repeat(70));
      console.log('  PAGES REQUIRING MIGRATION');
      console.log('═'.repeat(70));
      
      needsWork.forEach(r => {
        console.log(`\n  📄 ${r.page}`);
        if (!r.design.passed) {
          console.log(`     └─ Design: ${r.design.score}/100 (need ${CONFIG.passThresholds.design}+)`);
        }
        if (!r.accessibility.passed) {
          console.log(`     └─ Accessibility: ${r.accessibility.score}/100 (need ${CONFIG.passThresholds.accessibility}+)`);
        }
        if (!r.links.passed) {
          console.log(`     └─ Links: ${r.links.broken} broken links`);
        }
      });
    }
    
    // Ready pages
    if (ready.length > 0) {
      console.log('\n' + '═'.repeat(70));
      console.log('  ✅ PRODUCTION READY PAGES');
      console.log('═'.repeat(70));
      console.log(`\n  ${ready.map(r => r.page).join(', ')}`);
    }
  }
  
  // Next steps
  console.log('\n' + '═'.repeat(70));
  console.log('  RECOMMENDED NEXT STEPS');
  console.log('═'.repeat(70));
  
  if (needsWork.length > 0) {
    // Sort by combined score to prioritize easiest fixes
    const prioritized = needsWork
      .filter(r => r.design && r.accessibility)
      .sort((a, b) => b.overallScore - a.overallScore)
      .slice(0, 5);
    
    console.log('\n  Priority order (closest to passing):');
    prioritized.forEach((r, i) => {
      console.log(`     ${i + 1}. ${r.page} (${r.overallScore}/100)`);
    });
  } else {
    console.log('\n  🎉 All pages are ready! Run final QA checks:');
    console.log('     1. Visual comparison in browser');
    console.log('     2. Mobile responsiveness test');
    console.log('     3. Cross-browser check');
  }
  
  console.log('\n' + '═'.repeat(70) + '\n');
}

// ============================================
// MAIN
// ============================================

function main() {
  const args = process.argv.slice(2);
  const summaryOnly = args.includes('--summary');
  const singlePage = args.find(arg => arg.endsWith('.html'));
  
  console.log('\n🔍 Running comprehensive evaluation...\n');
  
  if (singlePage) {
    // Single page evaluation
    const pageName = path.basename(singlePage);
    const result = evaluatePage(pageName);
    
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log(`║  ${pageName.padEnd(54)} ║`);
    console.log('╚════════════════════════════════════════════════════════════╝');
    
    if (!result.exists) {
      console.log('\n  ❌ File not found\n');
      process.exit(1);
    }
    
    console.log(`
  Design System:  ${result.design.score}/100 ${result.design.passed ? '✅' : '❌'}
  Accessibility:  ${result.accessibility.score}/100 ${result.accessibility.passed ? '✅' : '⚠️'}
  Links:          ${result.links.passed ? 'All valid ✅' : `${result.links.broken} broken ❌`}
  
  Overall: ${result.overall} ${result.overall === 'READY' ? '✅' : '⚠️'}
    `);
    
    process.exit(result.overall === 'READY' ? 0 : 1);
  } else {
    // Full evaluation
    const results = [];
    let completed = 0;
    
    CONFIG.marketingPages.forEach(page => {
      process.stdout.write(`  Evaluating: ${page.padEnd(25)}`);
      const result = evaluatePage(page);
      results.push(result);
      completed++;
      
      const icon = result.overall === 'READY' ? '✅' : 
                   result.overall === 'MISSING' ? '❌' : '⚠️';
      console.log(icon);
    });
    
    generateReport(results, summaryOnly);
  }
}

main();

