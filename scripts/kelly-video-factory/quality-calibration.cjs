/**
 * Kelly Video Quality Calibration System
 * 
 * Systematically tests different parameters to find optimal settings.
 * Measures: Quality, Speed, Reliability, Cost
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const fs = require('fs');
const path = require('path');
const KellyVideoFactory = require('./factory.cjs');

const CALIBRATION_TESTS = {
  // LoRA scale tests (character consistency)
  lora_scale: [0.75, 0.80, 0.85, 0.90, 0.95],
  
  // Animation motion tests
  motion_bucket_id: [60, 80, 100, 120, 140],
  
  // Image megapixels
  megapixels: ['0.5', '1', '1.5'],
  
  // Lipsync models
  lipsync_models: ['wav2lip', 'sadtalker_hq'],
};

class QualityCalibration {
  constructor(options = {}) {
    this.outputDir = options.outputDir || path.join(__dirname, '../../template-forge/calibration');
    this.testScript = options.testScript || "Hello curious learner! Today we discover something amazing about the world around us.";
    this.testTemplate = options.template || 'explain';
    
    fs.mkdirSync(this.outputDir, { recursive: true });
    
    this.results = [];
    this.sessionId = Date.now();
  }
  
  async runLoRAScaleTests() {
    console.log('\n═══════════════════════════════════════════════════');
    console.log('🔬 LORA SCALE CALIBRATION');
    console.log('═══════════════════════════════════════════════════\n');
    
    const scales = CALIBRATION_TESTS.lora_scale;
    
    for (const scale of scales) {
      console.log(`\nTesting LoRA scale: ${scale}`);
      
      // Temporarily modify config
      const config = require('./config.cjs');
      const originalScale = config.lora.scale;
      config.lora.scale = scale;
      
      const factory = new KellyVideoFactory({
        quality: 'preview',
        outputDir: path.join(this.outputDir, `lora_scale_${scale}`),
      });
      
      const startTime = Date.now();
      const result = await factory.generate(this.testTemplate, this.testScript);
      const duration = (Date.now() - startTime) / 1000;
      
      this.results.push({
        test: 'lora_scale',
        value: scale,
        success: result.success,
        duration,
        imageUrl: result.steps?.image?.url,
        videoUrl: result.finalVideoUrl,
      });
      
      // Restore original
      config.lora.scale = originalScale;
    }
    
    this.saveResults();
  }
  
  async runComparisonTest() {
    console.log('\n═══════════════════════════════════════════════════');
    console.log('🎬 FULL COMPARISON TEST: All Templates + Quality Tiers');
    console.log('═══════════════════════════════════════════════════\n');
    
    const templates = ['welcome', 'explain', 'heartfelt', 'curious', 'excited', 'thoughtful'];
    const qualities = ['preview', 'standard'];
    
    const scripts = {
      welcome: "Hello curious learner! I'm so happy you're here today.",
      explain: "Did you know butterflies taste with their feet? Nature is amazing!",
      heartfelt: "Every time you learn something new, you become a little bit wiser.",
      curious: "Have you ever wondered why the sky is blue? Let's find out!",
      excited: "Wow! This is incredible! You're about to discover something amazing!",
      thoughtful: "Take a moment to think about what we learned today.",
    };
    
    for (const quality of qualities) {
      for (const template of templates) {
        console.log(`\n📹 ${quality.toUpperCase()} / ${template}`);
        
        const factory = new KellyVideoFactory({
          quality,
          outputDir: path.join(this.outputDir, `${quality}_${template}`),
        });
        
        const startTime = Date.now();
        const result = await factory.generate(template, scripts[template]);
        const duration = (Date.now() - startTime) / 1000;
        
        this.results.push({
          test: 'comparison',
          quality,
          template,
          success: result.success,
          duration,
          imageUrl: result.steps?.image?.url,
          videoUrl: result.finalVideoUrl,
          error: result.error,
        });
        
        // Brief pause
        await new Promise(r => setTimeout(r, 3000));
      }
    }
    
    this.saveResults();
    this.printSummary();
  }
  
  saveResults() {
    const resultsPath = path.join(this.outputDir, `calibration_${this.sessionId}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(this.results, null, 2));
    console.log(`\nResults saved to: ${resultsPath}`);
  }
  
  printSummary() {
    console.log('\n═══════════════════════════════════════════════════');
    console.log('📊 CALIBRATION SUMMARY');
    console.log('═══════════════════════════════════════════════════\n');
    
    // Group by test type
    const byTest = {};
    for (const r of this.results) {
      const key = r.test;
      if (!byTest[key]) byTest[key] = [];
      byTest[key].push(r);
    }
    
    for (const [test, results] of Object.entries(byTest)) {
      console.log(`\n${test.toUpperCase()}:`);
      const successful = results.filter(r => r.success);
      const avgDuration = successful.reduce((a, b) => a + b.duration, 0) / (successful.length || 1);
      console.log(`  Success: ${successful.length}/${results.length}`);
      console.log(`  Avg Duration: ${avgDuration.toFixed(1)}s`);
    }
  }
  
  async generateCalibrationReport() {
    const htmlPath = path.join(this.outputDir, `calibration_report_${this.sessionId}.html`);
    
    const videoGrid = this.results
      .filter(r => r.success && r.videoUrl)
      .map(r => `
        <div class="video-item">
          <video controls muted loop>
            <source src="${r.videoUrl}" type="video/mp4">
          </video>
          <div class="label">
            ${r.test}: ${r.value || r.template || ''}<br>
            ${r.quality || ''} - ${r.duration?.toFixed(1)}s
          </div>
        </div>
      `).join('');
    
    const html = `<!DOCTYPE html>
<html>
<head>
  <title>Kelly Calibration Report - ${new Date().toISOString().split('T')[0]}</title>
  <style>
    body { font-family: system-ui; background: #0a0a0f; color: #eee; padding: 2rem; }
    h1 { color: #10b981; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }
    .video-item { background: #12121a; border-radius: 12px; overflow: hidden; }
    .video-item video { width: 100%; }
    .label { padding: 0.5rem; text-align: center; font-size: 0.8rem; }
  </style>
</head>
<body>
  <h1>Kelly Calibration Report</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  <div class="grid">${videoGrid}</div>
</body>
</html>`;
    
    fs.writeFileSync(htmlPath, html);
    console.log(`\nHTML Report: ${htmlPath}`);
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  const calibration = new QualityCalibration();
  
  if (args.includes('--lora')) {
    await calibration.runLoRAScaleTests();
  } else if (args.includes('--full')) {
    await calibration.runComparisonTest();
    await calibration.generateCalibrationReport();
  } else {
    console.log(`
Quality Calibration System

Usage:
  node quality-calibration.cjs --lora    Run LoRA scale tests
  node quality-calibration.cjs --full    Run full comparison across all templates
`);
  }
}

main().catch(console.error);



