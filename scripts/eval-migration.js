/**
 * Migration Evaluation Script
 * 
 * Validates that a page conforms to the new design system.
 * 
 * Usage:
 *   node scripts/eval-migration.js public/about.html
 *   node scripts/eval-migration.js --all    (evaluates all marketing pages)
 *   node scripts/eval-migration.js --report (generates full report)
 */

const fs = require('fs');
const path = require('path');

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  requiredFonts: ['Instrument Sans', 'Newsreader'],
  bannedFonts: ['Times New Roman', 'Inter', 'Fraunces', 'DM Sans', 'Cormorant'],
  
  requiredColors: {
    '--bg-color': '#0a0a0b',
    '--bg-secondary': '#111113',
    '--bg-elevated': '#18181b',
    '--text-primary': '#fafafa',
    '--text-secondary': '#a1a1aa',
    '--accent-primary': '#3b82f6',
    '--border-color': '#27272a'
  },
  
  bannedColors: ['#0f0f11', '#f4f4f5', '#0a0a0c', '#12121a'],
  
  requiredElements: [
    'kelly-mark-circle',           // Logo
    'Lesson of the Day PBC',       // Company name
    'hello@curiouskelly.com',      // Contact email
  ],
  
  bannedEmails: ['team@', 'dev@', 'support@curiouskelly', 'social@'],
  
  // Pages that should NOT be evaluated (app pages, tests, etc.)
  excludePatterns: [
    'learn.html', 'learn-v1.html', 'learn-v2.html',
    'app.html', 'calendar.html', 'dashboard.html', 'hub.html',
    'kelly.html', 'lesson-detail.html', 'live.html', 'me.html',
    'player.html', 'settings.html', 'welcome.html',
    'test-', 'debug-', 'mockup', 'index-unified', 'index-production',
    'index-final', 'index-old', 'unity'
  ],
  
  // Marketing pages that MUST be migrated
  marketingPages: [
    'about.html', 'accessibility.html', 'affiliates.html', 'ambassador.html',
    'api.html', 'careers.html', 'commons.html', 'contact.html',
    'curriculum.html', 'diversity.html', 'enterprise.html', 'gifts.html',
    'group.html', 'help.html', 'impact.html', 'join.html', 'missions.html',
    'newsroom.html', 'partner.html', 'perspectives.html', 'pricing.html',
    'privacy.html', 'social.html', 'terms.html', 'trust.html'
  ]
};

// ============================================
// EVALUATION FUNCTIONS
// ============================================

function evaluatePage(filePath) {
  if (!fs.existsSync(filePath)) {
    return {
      file: path.basename(filePath),
      exists: false,
      score: 0,
      passed: false,
      issues: [`File not found: ${filePath}`]
    };
  }
  
  const content = fs.readFileSync(filePath, 'utf8');
  const fileName = path.basename(filePath);
  
  const results = {
    file: fileName,
    exists: true,
    scores: {
      typography: 0,
      colors: 0,
      branding: 0,
      structure: 0
    },
    issues: [],
    warnings: [],
    passed: true
  };
  
  // ---- TYPOGRAPHY (25 points) ----
  let typographyScore = 25;
  
  // Check for required fonts
  CONFIG.requiredFonts.forEach(font => {
    if (!content.includes(font)) {
      typographyScore -= 10;
      results.issues.push(`❌ Missing required font: ${font}`);
      results.passed = false;
    }
  });
  
  // Check for banned fonts
  CONFIG.bannedFonts.forEach(font => {
    // Case-insensitive check for font-family declarations
    const regex = new RegExp(`font-family[^;]*${font}`, 'i');
    if (regex.test(content)) {
      typographyScore -= 8;
      results.issues.push(`❌ Found banned font: ${font}`);
      results.passed = false;
    }
  });
  
  results.scores.typography = Math.max(0, typographyScore);
  
  // ---- COLORS (25 points) ----
  let colorScore = 25;
  
  // Check for banned colors
  CONFIG.bannedColors.forEach(color => {
    if (content.includes(color)) {
      colorScore -= 5;
      results.issues.push(`❌ Found old color: ${color}`);
      results.passed = false;
    }
  });
  
  // Check for required color variables
  let hasColorVars = 0;
  Object.entries(CONFIG.requiredColors).forEach(([variable, value]) => {
    if (content.includes(variable) && content.includes(value)) {
      hasColorVars++;
    }
  });
  
  if (hasColorVars < 4) {
    colorScore -= 5;
    results.warnings.push(`⚠️ Missing some color variables (found ${hasColorVars}/7)`);
  }
  
  results.scores.colors = Math.max(0, colorScore);
  
  // ---- BRANDING (25 points) ----
  let brandingScore = 25;
  
  // Check for Kelly logo
  if (!content.includes('kelly-mark-circle')) {
    brandingScore -= 8;
    results.warnings.push(`⚠️ May be missing Kelly logo image`);
  }
  
  // Check for correct company name
  if (!content.includes('Lesson of the Day PBC')) {
    brandingScore -= 5;
    results.warnings.push(`⚠️ Footer should reference "Lesson of the Day PBC"`);
  }
  
  // Check for correct email
  if (content.includes('@curiouskelly.com')) {
    if (!content.includes('hello@curiouskelly.com')) {
      // Check for banned email patterns
      CONFIG.bannedEmails.forEach(banned => {
        if (content.includes(banned)) {
          brandingScore -= 10;
          results.issues.push(`❌ Found unauthorized email pattern: ${banned}`);
          results.passed = false;
        }
      });
    }
  }
  
  results.scores.branding = Math.max(0, brandingScore);
  
  // ---- STRUCTURE (25 points) ----
  let structureScore = 25;
  
  // Check for proper HTML structure
  if (!content.includes('<!DOCTYPE html>')) {
    structureScore -= 5;
    results.warnings.push(`⚠️ Missing DOCTYPE declaration`);
  }
  
  // Check for viewport meta
  if (!content.includes('viewport')) {
    structureScore -= 5;
    results.issues.push(`❌ Missing viewport meta tag`);
    results.passed = false;
  }
  
  // Check for description meta
  if (!content.includes('meta name="description"') && !content.includes("meta name='description'")) {
    structureScore -= 3;
    results.warnings.push(`⚠️ Missing meta description`);
  }
  
  // Check for preconnect to Google Fonts
  if (!content.includes('preconnect') || !content.includes('fonts.googleapis.com')) {
    structureScore -= 3;
    results.warnings.push(`⚠️ Missing font preconnect for performance`);
  }
  
  // Check for header structure
  if (!content.includes('class="header"') && !content.includes("class='header'") && 
      !content.includes('<header')) {
    structureScore -= 5;
    results.warnings.push(`⚠️ May be missing header element`);
  }
  
  // Check for footer
  if (!content.includes('footer') && !content.includes('Footer')) {
    structureScore -= 5;
    results.warnings.push(`⚠️ May be missing footer element`);
  }
  
  results.scores.structure = Math.max(0, structureScore);
  
  // ---- CALCULATE TOTAL ----
  results.score = results.scores.typography + results.scores.colors + 
                  results.scores.branding + results.scores.structure;
  
  // Adjust pass threshold
  if (results.score < 85) {
    results.passed = false;
  }
  
  return results;
}

// ============================================
// REPORTING
// ============================================

function printResults(results) {
  const statusIcon = results.passed ? '✅' : '❌';
  const statusText = results.passed ? 'PASSED' : 'FAILED';
  
  console.log('\n' + '='.repeat(60));
  console.log(`${statusIcon} ${results.file} - ${statusText}`);
  console.log('='.repeat(60));
  
  if (!results.exists) {
    console.log('  File not found!');
    return;
  }
  
  console.log(`\n  📊 Score: ${results.score}/100`);
  console.log(`     Typography: ${results.scores.typography}/25`);
  console.log(`     Colors:     ${results.scores.colors}/25`);
  console.log(`     Branding:   ${results.scores.branding}/25`);
  console.log(`     Structure:  ${results.scores.structure}/25`);
  
  if (results.issues.length > 0) {
    console.log('\n  🚨 Issues (must fix):');
    results.issues.forEach(issue => {
      console.log(`     ${issue}`);
    });
  }
  
  if (results.warnings.length > 0) {
    console.log('\n  ⚠️  Warnings:');
    results.warnings.forEach(warning => {
      console.log(`     ${warning}`);
    });
  }
  
  if (results.passed && results.issues.length === 0 && results.warnings.length === 0) {
    console.log('\n  🎉 Perfect! No issues found.');
  }
  
  console.log('');
}

function generateReport(allResults) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║           DESIGN MIGRATION EVALUATION REPORT               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  const passed = allResults.filter(r => r.passed);
  const failed = allResults.filter(r => !r.passed);
  const avgScore = allResults.reduce((sum, r) => sum + r.score, 0) / allResults.length;
  
  console.log(`\n📈 Summary:`);
  console.log(`   Total Pages Evaluated: ${allResults.length}`);
  console.log(`   Passed: ${passed.length} ✅`);
  console.log(`   Failed: ${failed.length} ❌`);
  console.log(`   Average Score: ${avgScore.toFixed(1)}/100`);
  console.log(`   Pass Rate: ${((passed.length / allResults.length) * 100).toFixed(1)}%`);
  
  if (failed.length > 0) {
    console.log(`\n🚨 Failed Pages:`);
    failed.forEach(r => {
      console.log(`   - ${r.file} (${r.score}/100)`);
    });
  }
  
  console.log('\n' + '─'.repeat(60));
  console.log('Individual Results:');
  console.log('─'.repeat(60));
  
  allResults.forEach(printResults);
  
  // Summary table
  console.log('\n📋 Quick Reference Table:\n');
  console.log('| Page | Score | Typography | Colors | Branding | Structure | Status |');
  console.log('|------|-------|------------|--------|----------|-----------|--------|');
  allResults.forEach(r => {
    const status = r.passed ? '✅' : '❌';
    console.log(`| ${r.file.padEnd(20)} | ${String(r.score).padStart(3)}/100 | ${String(r.scores.typography).padStart(2)}/25 | ${String(r.scores.colors).padStart(2)}/25 | ${String(r.scores.branding).padStart(2)}/25 | ${String(r.scores.structure).padStart(2)}/25 | ${status} |`);
  });
}

// ============================================
// MAIN
// ============================================

function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node scripts/eval-migration.js public/about.html   - Evaluate single page');
    console.log('  node scripts/eval-migration.js --all               - Evaluate all marketing pages');
    console.log('  node scripts/eval-migration.js --report            - Generate full report');
    process.exit(0);
  }
  
  if (args[0] === '--all' || args[0] === '--report') {
    // Evaluate all marketing pages
    const publicDir = path.join(__dirname, '..', 'public');
    const allResults = [];
    
    CONFIG.marketingPages.forEach(page => {
      const filePath = path.join(publicDir, page);
      const results = evaluatePage(filePath);
      allResults.push(results);
    });
    
    generateReport(allResults);
  } else {
    // Evaluate single page
    const filePath = args[0];
    const results = evaluatePage(filePath);
    printResults(results);
    
    // Exit with error code if failed
    process.exit(results.passed ? 0 : 1);
  }
}

main();


