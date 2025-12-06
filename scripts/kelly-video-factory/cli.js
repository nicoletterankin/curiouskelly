#!/usr/bin/env node
/**
 * Kelly Video Factory CLI
 * 
 * Usage:
 *   node cli.js generate <template> "<script>" [--quality preview|standard|production]
 *   node cli.js batch <manifest.json> [--quality standard]
 *   node cli.js calibrate [--iterations 3]
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const KellyVideoFactory = require('./factory');
const fs = require('fs');
const path = require('path');

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];
  
  if (!command) {
    printUsage();
    return;
  }
  
  switch (command) {
    case 'generate':
      await handleGenerate(args.slice(1));
      break;
    case 'batch':
      await handleBatch(args.slice(1));
      break;
    case 'calibrate':
      await handleCalibrate(args.slice(1));
      break;
    case 'templates':
      listTemplates();
      break;
    default:
      console.log(`Unknown command: ${command}`);
      printUsage();
  }
}

function printUsage() {
  console.log(`
Kelly Video Factory CLI

Commands:
  generate <template> "<script>" [options]   Generate a single video
  batch <manifest.json> [options]            Generate videos from manifest
  calibrate [options]                        Run calibration tests
  templates                                  List available templates

Options:
  --quality <tier>   Quality tier: preview, standard, production (default: standard)
  --output <dir>     Output directory

Examples:
  node cli.js generate welcome "Hello curious learner!"
  node cli.js generate explain "Today we learn about butterflies" --quality production
  node cli.js batch lessons.json --quality standard
  node cli.js calibrate --iterations 3
`);
}

function listTemplates() {
  const config = require('./config');
  console.log('\nAvailable Templates:\n');
  for (const [key, template] of Object.entries(config.templates)) {
    console.log(`  ${key.padEnd(12)} - ${template.environment}, ${template.emotion}, ${template.action}`);
  }
  console.log('');
}

async function handleGenerate(args) {
  const template = args[0];
  const script = args[1];
  
  if (!template || !script) {
    console.log('Error: template and script are required');
    console.log('Usage: generate <template> "<script>"');
    return;
  }
  
  const qualityIndex = args.indexOf('--quality');
  const quality = qualityIndex > -1 ? args[qualityIndex + 1] : 'standard';
  
  const outputIndex = args.indexOf('--output');
  const outputDir = outputIndex > -1 ? args[outputIndex + 1] : undefined;
  
  const factory = new KellyVideoFactory({
    quality,
    outputDir,
  });
  
  await factory.generate(template, script);
}

async function handleBatch(args) {
  const manifestPath = args[0];
  
  if (!manifestPath || !fs.existsSync(manifestPath)) {
    console.log('Error: manifest file required');
    return;
  }
  
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
  
  const qualityIndex = args.indexOf('--quality');
  const quality = qualityIndex > -1 ? args[qualityIndex + 1] : 'standard';
  
  const factory = new KellyVideoFactory({ quality });
  
  console.log(`\n🎬 BATCH GENERATION: ${manifest.videos.length} videos\n`);
  
  const results = [];
  
  for (let i = 0; i < manifest.videos.length; i++) {
    const video = manifest.videos[i];
    console.log(`\n[${i + 1}/${manifest.videos.length}] ${video.template}: "${video.script.substring(0, 40)}..."\n`);
    
    const result = await factory.generate(video.template, video.script);
    results.push(result);
    
    // Brief pause between videos
    await new Promise(r => setTimeout(r, 2000));
  }
  
  // Summary
  const successful = results.filter(r => r.success).length;
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`BATCH COMPLETE: ${successful}/${results.length} successful`);
  console.log(`${'═'.repeat(70)}\n`);
  
  // Save batch results
  const batchResultPath = path.join(factory.outputDir, `batch_results_${Date.now()}.json`);
  fs.writeFileSync(batchResultPath, JSON.stringify(results, null, 2));
  console.log(`Results saved to: ${batchResultPath}`);
}

async function handleCalibrate(args) {
  const iterationsIndex = args.indexOf('--iterations');
  const iterations = iterationsIndex > -1 ? parseInt(args[iterationsIndex + 1]) : 2;
  
  console.log(`\n🔬 CALIBRATION MODE: Testing ${iterations} iterations\n`);
  
  const testScript = "Hello curious learner! Today we discover something amazing.";
  const testTemplate = 'explain';
  
  const results = [];
  
  // Test each quality tier
  for (const quality of ['preview', 'standard']) {
    console.log(`\n--- Testing ${quality} quality ---\n`);
    
    for (let i = 0; i < iterations; i++) {
      const factory = new KellyVideoFactory({ quality });
      const result = await factory.generate(testTemplate, testScript);
      
      results.push({
        quality,
        iteration: i + 1,
        success: result.success,
        duration: result.duration,
        imageUrl: result.steps?.image?.url,
        videoUrl: result.finalVideoUrl,
      });
    }
  }
  
  // Analysis
  console.log(`\n${'═'.repeat(70)}`);
  console.log('CALIBRATION RESULTS');
  console.log(`${'═'.repeat(70)}\n`);
  
  for (const quality of ['preview', 'standard']) {
    const qualityResults = results.filter(r => r.quality === quality);
    const successful = qualityResults.filter(r => r.success);
    const avgDuration = successful.reduce((a, b) => a + parseFloat(b.duration), 0) / successful.length;
    
    console.log(`${quality.toUpperCase()}:`);
    console.log(`  Success rate: ${successful.length}/${qualityResults.length}`);
    console.log(`  Avg duration: ${avgDuration.toFixed(1)}s`);
    console.log('');
  }
  
  // Save calibration results
  const calibrationPath = path.join(__dirname, '../../template-forge/factory-output', `calibration_${Date.now()}.json`);
  fs.mkdirSync(path.dirname(calibrationPath), { recursive: true });
  fs.writeFileSync(calibrationPath, JSON.stringify(results, null, 2));
  console.log(`Calibration data saved to: ${calibrationPath}`);
}

main().catch(console.error);

