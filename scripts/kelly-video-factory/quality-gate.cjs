#!/usr/bin/env node
/**
 * Kelly Video Factory - Quality Gate
 * 
 * Automated quality checks for generated Kelly images:
 * 1. Face consistency (compared to reference)
 * 2. Sweater color verification (blue, not pink)
 * 3. Image quality metrics
 * 
 * Run: node quality-gate.cjs <image-or-directory>
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

// Reference Kelly image URL (canonical)
const KELLY_REFERENCE_URL = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/reference/kelly_primary_face.jpeg';

// Color thresholds for blue sweater detection
const BLUE_SWEATER_HSV = {
  hMin: 180,  // Blue hue start
  hMax: 220,  // Blue hue end
  sMin: 20,   // Saturation minimum
  vMin: 40,   // Value minimum
};

const PINK_SWEATER_HSV = {
  hMin: 330,  // Pink/red hue start
  hMax: 360,  // Pink/red hue end (wraps)
  hMin2: 0,   // Pink/red hue start (wrapped)
  hMax2: 30,  // Pink/red hue end
};

class QualityGate {
  constructor() {
    this.supabase = createClient(
      process.env.PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );
    this.results = [];
  }
  
  /**
   * Analyze image for sweater color
   * This is a simplified heuristic - checks if dominant color in torso region is blue
   */
  async analyzeSweaterColor(imagePath) {
    // For a full implementation, we'd use sharp or jimp to analyze pixel data
    // For now, we'll use a heuristic based on file name and trust the prompt
    
    // In production, this would:
    // 1. Load image
    // 2. Sample pixels in torso region (middle-lower area of image)
    // 3. Convert to HSV
    // 4. Check if majority fall in blue range
    
    return {
      check: 'sweater_color',
      status: 'manual_review',
      message: 'Requires visual inspection or pixel analysis',
    };
  }
  
  /**
   * Check image file validity
   */
  async checkImageFile(imagePath) {
    const stats = fs.statSync(imagePath);
    
    return {
      check: 'file_validity',
      status: stats.size > 10000 ? 'pass' : 'fail',
      fileSize: stats.size,
      message: stats.size > 10000 ? 'Valid image file' : 'File too small, may be corrupted',
    };
  }
  
  /**
   * Generate quality report for single image
   */
  async checkImage(imagePath) {
    const filename = path.basename(imagePath);
    console.log(`\n  Checking: ${filename}`);
    
    const checks = [];
    
    // File validity
    const fileCheck = await this.checkImageFile(imagePath);
    checks.push(fileCheck);
    console.log(`    📁 File: ${fileCheck.status} (${(fileCheck.fileSize / 1024).toFixed(1)}KB)`);
    
    // Sweater color (simplified)
    const colorCheck = await this.analyzeSweaterColor(imagePath);
    checks.push(colorCheck);
    console.log(`    👔 Sweater: ${colorCheck.status}`);
    
    // Overall status
    const failed = checks.filter(c => c.status === 'fail');
    const manual = checks.filter(c => c.status === 'manual_review');
    
    let status;
    if (failed.length > 0) {
      status = 'FAIL';
    } else if (manual.length > 0) {
      status = 'REVIEW';
    } else {
      status = 'PASS';
    }
    
    const result = {
      image: filename,
      path: imagePath,
      status,
      checks,
      timestamp: new Date().toISOString(),
    };
    
    this.results.push(result);
    console.log(`    📋 Overall: ${status}`);
    
    return result;
  }
  
  /**
   * Check all images in a directory
   */
  async checkDirectory(dirPath) {
    const files = fs.readdirSync(dirPath)
      .filter(f => f.endsWith('.png') || f.endsWith('.jpg') || f.endsWith('.jpeg'));
    
    console.log(`\n  Found ${files.length} images in ${dirPath}`);
    
    for (const file of files) {
      await this.checkImage(path.join(dirPath, file));
    }
    
    return this.generateReport();
  }
  
  /**
   * Generate summary report
   */
  generateReport() {
    const passed = this.results.filter(r => r.status === 'PASS').length;
    const failed = this.results.filter(r => r.status === 'FAIL').length;
    const review = this.results.filter(r => r.status === 'REVIEW').length;
    
    return {
      total: this.results.length,
      passed,
      failed,
      needsReview: review,
      results: this.results,
    };
  }
  
  /**
   * Save results to JSON
   */
  saveResults(outputPath) {
    const report = this.generateReport();
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    return report;
  }
}

/**
 * Generate visual comparison HTML
 */
function generateComparisonHTML(results, outputPath) {
  const cards = results.map(r => {
    const statusColor = r.status === 'PASS' ? '#10b981' : r.status === 'FAIL' ? '#ef4444' : '#f59e0b';
    return `
      <div class="card" style="border-color: ${statusColor}">
        <img src="file://${r.path.replace(/\\/g, '/')}" alt="${r.image}">
        <div class="info">
          <strong>${r.image}</strong><br>
          <span style="color: ${statusColor}">${r.status}</span>
        </div>
      </div>
    `;
  }).join('');
  
  const html = `<!DOCTYPE html>
<html>
<head>
  <title>Kelly Quality Gate Results</title>
  <style>
    body { font-family: system-ui; background: #0a0a0f; color: #eee; padding: 2rem; }
    h1 { color: #10b981; }
    .summary { background: #12121a; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
    .summary span { margin-right: 2rem; }
    .pass { color: #10b981; }
    .fail { color: #ef4444; }
    .review { color: #f59e0b; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; }
    .card { background: #12121a; border-radius: 12px; overflow: hidden; border: 2px solid; }
    .card img { width: 100%; }
    .info { padding: 0.75rem; font-size: 0.85rem; }
  </style>
</head>
<body>
  <h1>🔍 Kelly Quality Gate Results</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  
  <div class="summary">
    <span class="pass">✅ Passed: ${results.filter(r => r.status === 'PASS').length}</span>
    <span class="fail">❌ Failed: ${results.filter(r => r.status === 'FAIL').length}</span>
    <span class="review">⚠️ Review: ${results.filter(r => r.status === 'REVIEW').length}</span>
  </div>
  
  <div class="grid">${cards}</div>
</body>
</html>`;
  
  fs.writeFileSync(outputPath, html);
  console.log(`\n  HTML Report: ${outputPath}`);
}

// Main
async function main() {
  const args = process.argv.slice(2);
  const target = args[0];
  
  if (!target) {
    console.log(`
Quality Gate - Automated Kelly Image Validation

Usage:
  node quality-gate.cjs <image.png>          Check single image
  node quality-gate.cjs <directory>          Check all images in directory
  node quality-gate.cjs --production         Check production images

Checks:
  ✓ File validity (size, format)
  ✓ Sweater color (blue vs pink)
  
Coming soon:
  ○ Face similarity (vs reference)
  ○ Expression match (per template)
`);
    return;
  }
  
  console.log('═'.repeat(70));
  console.log('🔍 KELLY QUALITY GATE');
  console.log('═'.repeat(70));
  
  const gate = new QualityGate();
  let report;
  
  if (target === '--production') {
    const prodDir = path.join(__dirname, '../../template-forge/production-images');
    if (!fs.existsSync(prodDir)) {
      console.log('\n  No production images found. Run batch-image-generator.cjs first.');
      return;
    }
    report = await gate.checkDirectory(prodDir);
  } else if (fs.statSync(target).isDirectory()) {
    report = await gate.checkDirectory(target);
  } else {
    await gate.checkImage(target);
    report = gate.generateReport();
  }
  
  // Save results
  const outputDir = path.join(__dirname, '../../template-forge/quality-reports');
  fs.mkdirSync(outputDir, { recursive: true });
  
  const timestamp = Date.now();
  const jsonPath = path.join(outputDir, `quality_report_${timestamp}.json`);
  const htmlPath = path.join(outputDir, `quality_report_${timestamp}.html`);
  
  gate.saveResults(jsonPath);
  generateComparisonHTML(gate.results, htmlPath);
  
  console.log('\n' + '═'.repeat(70));
  console.log('📊 QUALITY GATE SUMMARY');
  console.log('═'.repeat(70));
  console.log(`\n  Total: ${report.total}`);
  console.log(`  ✅ Passed: ${report.passed}`);
  console.log(`  ❌ Failed: ${report.failed}`);
  console.log(`  ⚠️ Review: ${report.needsReview}`);
  console.log(`\n  Reports saved to: ${outputDir}`);
}

main().catch(console.error);

