/**
 * Link Validation Script
 * 
 * Extracts and validates all links from HTML files.
 * 
 * Usage:
 *   node scripts/eval-links.cjs public/about.html
 *   node scripts/eval-links.cjs --all
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// ============================================
// CONFIGURATION
// ============================================

const CONFIG = {
  // Known good external domains (skip validation)
  trustedDomains: [
    'fonts.googleapis.com',
    'fonts.gstatic.com',
    'cdn.jsdelivr.net',
    'twitter.com',
    'instagram.com',
    'youtube.com',
    'linkedin.com',
    'github.com',
    'stripe.com',
    'buy.stripe.com',
    'supabase.co'
  ],
  
  // Internal pages that should exist
  requiredInternalLinks: [
    '/',
    '/privacy.html',
    '/terms.html',
    '/help.html',
    '/curriculum.html'
  ],
  
  // Marketing pages to check
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
// LINK EXTRACTION
// ============================================

function extractLinks(htmlContent) {
  const links = {
    internal: [],
    external: [],
    anchors: [],
    emails: [],
    broken: []
  };
  
  // Match href attributes
  const hrefRegex = /href\s*=\s*["']([^"']+)["']/gi;
  let match;
  
  while ((match = hrefRegex.exec(htmlContent)) !== null) {
    const href = match[1].trim();
    
    if (!href || href === '#') continue;
    
    if (href.startsWith('mailto:')) {
      links.emails.push(href);
    } else if (href.startsWith('#')) {
      links.anchors.push(href);
    } else if (href.startsWith('http://') || href.startsWith('https://')) {
      links.external.push(href);
    } else if (href.startsWith('//')) {
      links.external.push('https:' + href);
    } else {
      // Normalize internal links
      let normalized = href;
      if (!normalized.startsWith('/')) {
        normalized = '/' + normalized;
      }
      links.internal.push(normalized);
    }
  }
  
  // Deduplicate
  links.internal = [...new Set(links.internal)];
  links.external = [...new Set(links.external)];
  links.anchors = [...new Set(links.anchors)];
  links.emails = [...new Set(links.emails)];
  
  return links;
}

// ============================================
// LINK VALIDATION
// ============================================

function validateInternalLink(link, publicDir) {
  // Handle anchors on current page
  if (link.includes('#')) {
    link = link.split('#')[0];
  }
  
  if (!link || link === '/') {
    // Root is valid
    return { valid: true, path: link };
  }
  
  // Check if file exists
  const filePath = path.join(publicDir, link);
  const exists = fs.existsSync(filePath);
  
  // Also check without .html extension
  if (!exists && !link.endsWith('.html') && !link.endsWith('/')) {
    const withHtml = path.join(publicDir, link + '.html');
    if (fs.existsSync(withHtml)) {
      return { valid: true, path: link };
    }
  }
  
  // Check index.html in directory
  if (!exists && link.endsWith('/')) {
    const indexPath = path.join(publicDir, link, 'index.html');
    if (fs.existsSync(indexPath)) {
      return { valid: true, path: link };
    }
  }
  
  return { valid: exists, path: link };
}

function validateEmail(email) {
  const emailRegex = /^mailto:([^\?]+)/;
  const match = email.match(emailRegex);
  
  if (!match) return { valid: false, email };
  
  const address = match[1];
  
  // Check for authorized email
  if (address === 'hello@curiouskelly.com') {
    return { valid: true, email: address, authorized: true };
  }
  
  // Check for unauthorized emails
  if (address.includes('@curiouskelly.com')) {
    return { valid: false, email: address, reason: 'Unauthorized email address' };
  }
  
  return { valid: true, email: address };
}

async function checkExternalLink(url) {
  return new Promise((resolve) => {
    try {
      const urlObj = new URL(url);
      
      // Skip trusted domains
      if (CONFIG.trustedDomains.some(domain => urlObj.hostname.includes(domain))) {
        resolve({ valid: true, url, skipped: true });
        return;
      }
      
      const protocol = urlObj.protocol === 'https:' ? https : http;
      
      const req = protocol.request(url, { method: 'HEAD', timeout: 5000 }, (res) => {
        resolve({
          valid: res.statusCode >= 200 && res.statusCode < 400,
          url,
          statusCode: res.statusCode
        });
      });
      
      req.on('error', () => {
        resolve({ valid: false, url, error: 'Connection failed' });
      });
      
      req.on('timeout', () => {
        req.destroy();
        resolve({ valid: false, url, error: 'Timeout' });
      });
      
      req.end();
    } catch (e) {
      resolve({ valid: false, url, error: e.message });
    }
  });
}

// ============================================
// MAIN EVALUATION
// ============================================

async function evaluatePageLinks(filePath) {
  if (!fs.existsSync(filePath)) {
    return {
      file: path.basename(filePath),
      exists: false,
      passed: false,
      issues: ['File not found']
    };
  }
  
  const content = fs.readFileSync(filePath, 'utf8');
  const fileName = path.basename(filePath);
  const links = extractLinks(content);
  
  const results = {
    file: fileName,
    exists: true,
    passed: true,
    counts: {
      internal: links.internal.length,
      external: links.external.length,
      anchors: links.anchors.length,
      emails: links.emails.length
    },
    issues: [],
    warnings: [],
    details: {
      brokenInternal: [],
      brokenExternal: [],
      unauthorizedEmails: []
    }
  };
  
  // Validate internal links
  for (const link of links.internal) {
    const validation = validateInternalLink(link, CONFIG.publicDir);
    if (!validation.valid) {
      results.details.brokenInternal.push(link);
      results.issues.push(`❌ Broken internal link: ${link}`);
      results.passed = false;
    }
  }
  
  // Validate emails
  for (const email of links.emails) {
    const validation = validateEmail(email);
    if (!validation.valid) {
      results.details.unauthorizedEmails.push(validation.email);
      results.issues.push(`❌ ${validation.reason}: ${validation.email}`);
      results.passed = false;
    }
  }
  
  // Note: External link checking is slow, so we make it optional
  // Uncomment below to enable:
  /*
  for (const url of links.external.slice(0, 10)) { // Limit to first 10
    const validation = await checkExternalLink(url);
    if (!validation.valid && !validation.skipped) {
      results.details.brokenExternal.push(url);
      results.warnings.push(`⚠️ External link may be broken: ${url}`);
    }
  }
  */
  
  return results;
}

function printLinkResults(results) {
  const statusIcon = results.passed ? '✅' : '❌';
  const statusText = results.passed ? 'PASSED' : 'FAILED';
  
  console.log('\n' + '='.repeat(60));
  console.log(`${statusIcon} ${results.file} - Link Check ${statusText}`);
  console.log('='.repeat(60));
  
  if (!results.exists) {
    console.log('  File not found!');
    return;
  }
  
  console.log(`\n  📊 Link Counts:`);
  console.log(`     Internal: ${results.counts.internal}`);
  console.log(`     External: ${results.counts.external}`);
  console.log(`     Anchors:  ${results.counts.anchors}`);
  console.log(`     Emails:   ${results.counts.emails}`);
  
  if (results.issues.length > 0) {
    console.log('\n  🚨 Issues:');
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
  
  if (results.passed && results.issues.length === 0) {
    console.log('\n  🎉 All links valid!');
  }
  
  console.log('');
}

async function generateLinkReport(allResults) {
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║              LINK VALIDATION REPORT                        ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  
  const passed = allResults.filter(r => r.passed);
  const failed = allResults.filter(r => !r.passed);
  
  let totalInternal = 0;
  let totalExternal = 0;
  let totalBroken = 0;
  
  allResults.forEach(r => {
    totalInternal += r.counts?.internal || 0;
    totalExternal += r.counts?.external || 0;
    totalBroken += r.details?.brokenInternal?.length || 0;
  });
  
  console.log(`\n📈 Summary:`);
  console.log(`   Pages Checked: ${allResults.length}`);
  console.log(`   Passed: ${passed.length} ✅`);
  console.log(`   Failed: ${failed.length} ❌`);
  console.log(`   Total Internal Links: ${totalInternal}`);
  console.log(`   Total External Links: ${totalExternal}`);
  console.log(`   Broken Links Found: ${totalBroken}`);
  
  if (failed.length > 0) {
    console.log(`\n🚨 Pages with Broken Links:`);
    failed.forEach(r => {
      console.log(`   - ${r.file}`);
      r.details?.brokenInternal?.forEach(link => {
        console.log(`     └─ ${link}`);
      });
    });
  }
  
  // Collect all unique broken links
  const allBrokenLinks = new Set();
  allResults.forEach(r => {
    r.details?.brokenInternal?.forEach(link => allBrokenLinks.add(link));
  });
  
  if (allBrokenLinks.size > 0) {
    console.log(`\n📋 All Broken Links (unique):`);
    [...allBrokenLinks].sort().forEach(link => {
      console.log(`   ${link}`);
    });
  }
}

// ============================================
// MAIN
// ============================================

async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node scripts/eval-links.cjs public/about.html   - Check single page');
    console.log('  node scripts/eval-links.cjs --all               - Check all marketing pages');
    process.exit(0);
  }
  
  if (args[0] === '--all') {
    const allResults = [];
    
    for (const page of CONFIG.marketingPages) {
      const filePath = path.join(CONFIG.publicDir, page);
      const results = await evaluatePageLinks(filePath);
      allResults.push(results);
    }
    
    await generateLinkReport(allResults);
  } else {
    const filePath = args[0];
    const results = await evaluatePageLinks(filePath);
    printLinkResults(results);
    process.exit(results.passed ? 0 : 1);
  }
}

main();

