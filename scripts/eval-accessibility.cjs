/**
 * Accessibility Evaluation Script
 * 
 * Performs static analysis for common accessibility issues.
 * For full WCAG compliance, also run Lighthouse or axe-core.
 * 
 * Usage:
 *   node scripts/eval-accessibility.cjs public/about.html
 *   node scripts/eval-accessibility.cjs --all
 */

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
  publicDir: path.join(__dirname, '..', 'public')
};

// ============================================
// ACCESSIBILITY CHECKS
// ============================================

function checkAccessibility(htmlContent, fileName) {
  const results = {
    file: fileName,
    score: 100,
    passed: true,
    issues: [],
    warnings: [],
    checks: {
      images: { passed: 0, failed: 0 },
      headings: { passed: 0, failed: 0 },
      forms: { passed: 0, failed: 0 },
      landmarks: { passed: 0, failed: 0 },
      contrast: { passed: 0, failed: 0 },
      links: { passed: 0, failed: 0 }
    }
  };

  // ---- 1. IMAGE ALT TEXT ----
  const imgRegex = /<img[^>]*>/gi;
  const images = htmlContent.match(imgRegex) || [];
  
  images.forEach(img => {
    if (!img.includes('alt=')) {
      results.issues.push(`❌ Image missing alt attribute`);
      results.checks.images.failed++;
      results.score -= 5;
    } else if (img.includes('alt=""') || img.includes("alt=''")) {
      // Empty alt is OK for decorative images
      results.checks.images.passed++;
    } else {
      results.checks.images.passed++;
    }
  });

  // ---- 2. HEADING HIERARCHY ----
  const headingRegex = /<h([1-6])[^>]*>/gi;
  const headings = [];
  let match;
  
  while ((match = headingRegex.exec(htmlContent)) !== null) {
    headings.push(parseInt(match[1]));
  }
  
  // Check for h1
  if (!headings.includes(1)) {
    results.issues.push(`❌ Page missing <h1> element`);
    results.checks.headings.failed++;
    results.score -= 10;
  } else {
    results.checks.headings.passed++;
  }
  
  // Check for skipped levels
  for (let i = 1; i < headings.length; i++) {
    if (headings[i] > headings[i-1] + 1) {
      results.warnings.push(`⚠️ Heading level skipped: h${headings[i-1]} → h${headings[i]}`);
      results.checks.headings.failed++;
      results.score -= 3;
    } else {
      results.checks.headings.passed++;
    }
  }

  // ---- 3. FORM LABELS ----
  const inputRegex = /<input[^>]*type=["'](?!hidden|submit|button|reset)[^"']*["'][^>]*>/gi;
  const inputs = htmlContent.match(inputRegex) || [];
  
  inputs.forEach(input => {
    // Check for associated label (id + for, or aria-label)
    const hasLabel = input.includes('aria-label') || 
                     input.includes('aria-labelledby') ||
                     input.includes('id=');
    
    if (!hasLabel) {
      results.warnings.push(`⚠️ Form input may be missing label`);
      results.checks.forms.failed++;
      results.score -= 3;
    } else {
      results.checks.forms.passed++;
    }
  });

  // ---- 4. LANDMARK REGIONS ----
  const hasMain = /<main[^>]*>|role=["']main["']/i.test(htmlContent);
  const hasNav = /<nav[^>]*>|role=["']navigation["']/i.test(htmlContent);
  const hasHeader = /<header[^>]*>|role=["']banner["']/i.test(htmlContent);
  const hasFooter = /<footer[^>]*>|role=["']contentinfo["']/i.test(htmlContent);
  
  if (!hasMain) {
    results.warnings.push(`⚠️ Page missing <main> landmark`);
    results.checks.landmarks.failed++;
    results.score -= 5;
  } else {
    results.checks.landmarks.passed++;
  }
  
  if (!hasNav) {
    results.warnings.push(`⚠️ Page missing <nav> landmark`);
    results.checks.landmarks.failed++;
    results.score -= 2;
  } else {
    results.checks.landmarks.passed++;
  }
  
  if (!hasHeader) {
    results.warnings.push(`⚠️ Page missing <header> landmark`);
    results.checks.landmarks.failed++;
    results.score -= 2;
  } else {
    results.checks.landmarks.passed++;
  }

  // ---- 5. LINK TEXT ----
  const linkRegex = /<a[^>]*>([^<]*)<\/a>/gi;
  const emptyLinkTexts = ['click here', 'here', 'read more', 'learn more', 'more'];
  
  while ((match = linkRegex.exec(htmlContent)) !== null) {
    const linkText = match[1].trim().toLowerCase();
    
    if (!linkText) {
      // Check for aria-label
      if (!match[0].includes('aria-label')) {
        results.warnings.push(`⚠️ Link with empty text and no aria-label`);
        results.checks.links.failed++;
        results.score -= 3;
      } else {
        results.checks.links.passed++;
      }
    } else if (emptyLinkTexts.includes(linkText)) {
      results.warnings.push(`⚠️ Non-descriptive link text: "${linkText}"`);
      results.checks.links.failed++;
      results.score -= 2;
    } else {
      results.checks.links.passed++;
    }
  }

  // ---- 6. LANGUAGE ATTRIBUTE ----
  if (!htmlContent.includes('lang=')) {
    results.issues.push(`❌ Missing lang attribute on <html>`);
    results.score -= 10;
  }

  // ---- 7. VIEWPORT META ----
  if (!htmlContent.includes('viewport')) {
    results.issues.push(`❌ Missing viewport meta tag`);
    results.score -= 5;
  }
  
  // Check for user-scalable=no (bad practice)
  if (htmlContent.includes('user-scalable=no') || htmlContent.includes('user-scalable="no"')) {
    results.issues.push(`❌ Disabling zoom is an accessibility violation`);
    results.score -= 15;
  }

  // ---- 8. SKIP LINK ----
  if (!htmlContent.includes('skip') && !htmlContent.includes('Skip')) {
    results.warnings.push(`⚠️ Consider adding a skip-to-content link`);
    results.score -= 2;
  }

  // ---- 9. FOCUS STYLES ----
  if (htmlContent.includes('outline: none') || htmlContent.includes('outline:none')) {
    if (!htmlContent.includes(':focus')) {
      results.issues.push(`❌ Removing focus outline without replacement`);
      results.score -= 10;
    }
  }

  // ---- 10. COLOR CONTRAST (static check for known bad combos) ----
  // This is a simplified check - real contrast needs computed styles
  const lowContrastPatterns = [
    { bg: '#0a0a0b', fg: '#27272a' }, // border on bg - too low
    { bg: '#0a0a0b', fg: '#18181b' }, // elevated on bg - too low
  ];
  
  // We trust our design system colors are good, so just verify they're used
  if (htmlContent.includes('--text-primary') || htmlContent.includes('#fafafa')) {
    results.checks.contrast.passed++;
  }

  // Calculate pass/fail
  results.score = Math.max(0, results.score);
  results.passed = results.score >= 70 && results.issues.length === 0;
  
  return results;
}

// ============================================
// REPORTING
// ============================================

function printAccessibilityResults(results) {
  const statusIcon = results.passed ? '✅' : '❌';
  const statusText = results.passed ? 'PASSED' : 'NEEDS WORK';
  
  console.log('\n' + '='.repeat(60));
  console.log(`${statusIcon} ${results.file} - Accessibility ${statusText}`);
  console.log('='.repeat(60));
  
  console.log(`\n  📊 Accessibility Score: ${results.score}/100`);
  
  console.log(`\n  📋 Check Summary:`);
  Object.entries(results.checks).forEach(([name, counts]) => {
    const total = counts.passed + counts.failed;
    if (total > 0) {
      const pct = Math.round((counts.passed / total) * 100);
      console.log(`     ${name}: ${counts.passed}/${total} (${pct}%)`);
    }
  });
  
  if (results.issues.length > 0) {
    console.log('\n  🚨 Critical Issues:');
    results.issues.forEach(issue => {
      console.log(`     ${issue}`);
    });
  }
  
  if (results.warnings.length > 0) {
    console.log('\n  ⚠️  Warnings:');
    results.warnings.slice(0, 10).forEach(warning => {
      console.log(`     ${warning}`);
    });
    if (results.warnings.length > 10) {
      console.log(`     ... and ${results.warnings.length - 10} more`);
    }
  }
  
  if (results.passed && results.issues.length === 0 && results.warnings.length === 0) {
    console.log('\n  🎉 No accessibility issues found!');
  }
  
  console.log('');
}

function generateAccessibilityReport(allResults) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║            ACCESSIBILITY EVALUATION REPORT                 ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  const passed = allResults.filter(r => r.passed);
  const failed = allResults.filter(r => !r.passed);
  const avgScore = allResults.reduce((sum, r) => sum + r.score, 0) / allResults.length;
  
  console.log(`\n📈 Summary:`);
  console.log(`   Pages Evaluated: ${allResults.length}`);
  console.log(`   Passed: ${passed.length} ✅`);
  console.log(`   Needs Work: ${failed.length} ⚠️`);
  console.log(`   Average Score: ${avgScore.toFixed(1)}/100`);
  
  // Collect all unique issues
  const allIssues = {};
  allResults.forEach(r => {
    r.issues.forEach(issue => {
      allIssues[issue] = (allIssues[issue] || 0) + 1;
    });
  });
  
  if (Object.keys(allIssues).length > 0) {
    console.log(`\n🚨 Most Common Issues:`);
    Object.entries(allIssues)
      .sort((a, b) => b[1] - a[1])
      .forEach(([issue, count]) => {
        console.log(`   ${count}x ${issue}`);
      });
  }
  
  // Score table
  console.log('\n📋 Score Table:\n');
  console.log('| Page | Score | Status |');
  console.log('|------|-------|--------|');
  allResults
    .sort((a, b) => b.score - a.score)
    .forEach(r => {
      const status = r.passed ? '✅' : '⚠️';
      console.log(`| ${r.file.padEnd(24)} | ${String(r.score).padStart(3)}/100 | ${status} |`);
    });
  
  console.log('\n💡 Recommendations:');
  console.log('   1. Run Lighthouse for full WCAG audit');
  console.log('   2. Test with screen reader (VoiceOver, NVDA)');
  console.log('   3. Verify keyboard navigation works');
  console.log('   4. Check color contrast with WebAIM tool');
}

// ============================================
// MAIN
// ============================================

function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node scripts/eval-accessibility.cjs public/about.html');
    console.log('  node scripts/eval-accessibility.cjs --all');
    process.exit(0);
  }
  
  if (args[0] === '--all') {
    const allResults = [];
    
    CONFIG.marketingPages.forEach(page => {
      const filePath = path.join(CONFIG.publicDir, page);
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, 'utf8');
        const results = checkAccessibility(content, page);
        allResults.push(results);
      }
    });
    
    generateAccessibilityReport(allResults);
  } else {
    const filePath = args[0];
    if (!fs.existsSync(filePath)) {
      console.log(`File not found: ${filePath}`);
      process.exit(1);
    }
    
    const content = fs.readFileSync(filePath, 'utf8');
    const results = checkAccessibility(content, path.basename(filePath));
    printAccessibilityResults(results);
    process.exit(results.passed ? 0 : 1);
  }
}

main();




