#!/usr/bin/env node
/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * ██████╗  ██████╗ ██╗     ██████╗ ███████╗███╗   ██╗    ██╗   ██╗██████╗ 
 * ██╔════╝ ██╔═══██╗██║     ██╔══██╗██╔════╝████╗  ██║    ██║   ██║╚════██╗
 * ██║  ███╗██║   ██║██║     ██║  ██║█████╗  ██╔██╗ ██║    ██║   ██║ █████╔╝
 * ██║   ██║██║   ██║██║     ██║  ██║██╔══╝  ██║╚██╗██║    ╚██╗ ██╔╝██╔═══╝ 
 * ╚██████╔╝╚██████╔╝███████╗██████╔╝███████╗██║ ╚████║     ╚████╔╝ ███████╗
 *  ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝      ╚═══╝  ╚══════╝
 * ═══════════════════════════════════════════════════════════════════════════════
 * 
 * GOLDEN V2 - PRODUCTION ORCHESTRATOR
 * 
 * Master controller for the complete Kelly lesson production pipeline.
 * Orchestrates: Lesson DNA → Audio → Lip-Sync → Visuals → Package
 * 
 * Usage:
 *   node orchestrator.js --all                    # Run full pipeline
 *   node orchestrator.js --lessons                # Generate lesson DNA only
 *   node orchestrator.js --audio --day=1          # Generate audio for day 1
 *   node orchestrator.js --lipsync --start=1 --end=30
 *   node orchestrator.js --visuals
 *   node orchestrator.js --package --day=1        # Package day 1 for deployment
 * 
 * @version 2.0.0 - Golden V2
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIG = {
  // Base directories
  baseDir: __dirname,
  outputDir: path.join(__dirname, 'generated'),
  deployDir: path.join(__dirname, 'deploy'),
  
  // Sub-directories
  lessonsDir: path.join(__dirname, 'generated', 'lessons'),
  audioDir: path.join(__dirname, 'generated', 'audio'),
  lipsyncDir: path.join(__dirname, 'generated', 'lipsync'),
  visualsDir: path.join(__dirname, 'generated', 'visuals'),
  
  // Generation settings
  totalLessons: 365,
  defaultBatchSize: 30,
  
  // Feature flags
  enableAudio: true,
  enableLipsync: true,
  enableVisuals: true,
  
  // API settings (from environment)
  elevenLabsKey: process.env.ELEVENLABS_API_KEY,
  kellyVoiceId: process.env.KELLY_VOICE_ID
};

// ═══════════════════════════════════════════════════════════════════════════════
// BANNER
// ═══════════════════════════════════════════════════════════════════════════════

const BANNER = `
═══════════════════════════════════════════════════════════════════════════════
  ✨ CURIOUS KELLY - GOLDEN V2 PRODUCTION PIPELINE ✨
═══════════════════════════════════════════════════════════════════════════════
  
  "Quality education for anyone ages 2 to 102, anywhere in the world."
  
  Pipeline Stages:
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │   LESSONS   │ -> │    AUDIO    │ -> │  LIP-SYNC   │ -> │   VISUALS   │
  │   (DNA)     │    │ (ElevenLabs)│    │  (Visemes)  │    │  (Prompts)  │
  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                   │
                                   v
                           ┌─────────────┐
                           │   PACKAGE   │
                           │  (Deploy)   │
                           └─────────────┘
═══════════════════════════════════════════════════════════════════════════════
`;

// ═══════════════════════════════════════════════════════════════════════════════
// ORCHESTRATOR CLASS
// ═══════════════════════════════════════════════════════════════════════════════

class ProductionOrchestrator {
  constructor(config = CONFIG) {
    this.config = config;
    this.startTime = Date.now();
    this.stats = {
      lessonsGenerated: 0,
      audioGenerated: 0,
      lipsyncGenerated: 0,
      visualsGenerated: 0,
      errors: []
    };
  }
  
  /**
   * Initialize directory structure
   */
  initDirectories() {
    const dirs = [
      this.config.outputDir,
      this.config.lessonsDir,
      this.config.audioDir,
      this.config.lipsyncDir,
      this.config.visualsDir,
      this.config.deployDir
    ];
    
    for (const dir of dirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
    
    console.log('  📁 Directories initialized');
  }
  
  /**
   * Run lesson DNA generation
   */
  async runLessonGeneration() {
    console.log('\n  ═══ STAGE 1: LESSON DNA GENERATION ═══');
    
    try {
      const { LessonDNAGenerator } = await import('./lesson-dna-generator.js');
      const generator = new LessonDNAGenerator({
        outputDir: this.config.lessonsDir,
        totalLessons: this.config.totalLessons
      });
      
      await generator.generateAllLessons();
      this.stats.lessonsGenerated = this.config.totalLessons;
      
      console.log(`  ✅ Lesson DNA: ${this.stats.lessonsGenerated} lessons generated`);
    } catch (error) {
      this.stats.errors.push({ stage: 'lessons', error: error.message });
      console.error(`  ❌ Lesson generation failed: ${error.message}`);
    }
  }
  
  /**
   * Run audio generation
   */
  async runAudioGeneration(startDay = 1, endDay = 30, useMock = false) {
    console.log('\n  ═══ STAGE 2: AUDIO GENERATION ═══');
    
    if (!this.config.enableAudio) {
      console.log('  ⏭️ Audio generation disabled');
      return;
    }
    
    try {
      const { AudioGenerator, MockAudioGenerator } = await import('./audio-generator.js');
      const Generator = useMock ? MockAudioGenerator : AudioGenerator;
      
      const generator = new Generator({
        ...CONFIG,
        outputDir: this.config.audioDir,
        lessonsDir: this.config.lessonsDir
      });
      
      await generator.generateAllAudio(startDay, endDay);
      this.stats.audioGenerated = endDay - startDay + 1;
      
      console.log(`  ✅ Audio: ${this.stats.audioGenerated} days generated`);
    } catch (error) {
      this.stats.errors.push({ stage: 'audio', error: error.message });
      console.error(`  ❌ Audio generation failed: ${error.message}`);
    }
  }
  
  /**
   * Run lip-sync generation
   */
  async runLipSyncGeneration(startDay = 1, endDay = 30) {
    console.log('\n  ═══ STAGE 3: LIP-SYNC GENERATION ═══');
    
    if (!this.config.enableLipsync) {
      console.log('  ⏭️ Lip-sync generation disabled');
      return;
    }
    
    try {
      const { LipSyncGenerator } = await import('./lipsync-generator.js');
      const generator = new LipSyncGenerator({
        ...CONFIG,
        outputDir: this.config.lipsyncDir,
        lessonsDir: this.config.lessonsDir
      });
      
      await generator.generateAllLipSync(startDay, endDay);
      this.stats.lipsyncGenerated = endDay - startDay + 1;
      
      console.log(`  ✅ Lip-sync: ${this.stats.lipsyncGenerated} days generated`);
    } catch (error) {
      this.stats.errors.push({ stage: 'lipsync', error: error.message });
      console.error(`  ❌ Lip-sync generation failed: ${error.message}`);
    }
  }
  
  /**
   * Run visual prompt generation
   */
  async runVisualGeneration(startDay = 1, endDay = 365) {
    console.log('\n  ═══ STAGE 4: VISUAL GENERATION ═══');
    
    if (!this.config.enableVisuals) {
      console.log('  ⏭️ Visual generation disabled');
      return;
    }
    
    try {
      const { VisualGenerator } = await import('./visual-generator.js');
      const generator = new VisualGenerator({
        outputDir: this.config.visualsDir,
        lessonsDir: this.config.lessonsDir
      });
      
      await generator.generateAllVisuals(startDay, endDay);
      this.stats.visualsGenerated = endDay - startDay + 1;
      
      console.log(`  ✅ Visuals: ${this.stats.visualsGenerated} days generated`);
    } catch (error) {
      this.stats.errors.push({ stage: 'visuals', error: error.message });
      console.error(`  ❌ Visual generation failed: ${error.message}`);
    }
  }
  
  /**
   * Package a day's assets for deployment
   */
  async packageDay(dayNumber) {
    console.log(`\n  ═══ PACKAGING DAY ${dayNumber} ═══`);
    
    const paddedDay = String(dayNumber).padStart(3, '0');
    const packageDir = path.join(this.config.deployDir, `day-${paddedDay}`);
    
    // Create package directory
    if (!fs.existsSync(packageDir)) {
      fs.mkdirSync(packageDir, { recursive: true });
    }
    
    const packageManifest = {
      day: dayNumber,
      packagedAt: new Date().toISOString(),
      version: '2.0.0-golden',
      assets: {}
    };
    
    // Copy lesson DNA
    const lessonFile = path.join(this.config.lessonsDir, `day-${paddedDay}.json`);
    if (fs.existsSync(lessonFile)) {
      fs.copyFileSync(lessonFile, path.join(packageDir, 'lesson.json'));
      packageManifest.assets.lesson = 'lesson.json';
    }
    
    // Copy audio files
    const audioDir = path.join(this.config.audioDir, `day-${paddedDay}`);
    if (fs.existsSync(audioDir)) {
      const audioPackageDir = path.join(packageDir, 'audio');
      this.copyDirectoryRecursive(audioDir, audioPackageDir);
      packageManifest.assets.audio = 'audio/';
    }
    
    // Copy lip-sync data
    const lipsyncDir = path.join(this.config.lipsyncDir, `day-${paddedDay}`);
    if (fs.existsSync(lipsyncDir)) {
      const lipsyncPackageDir = path.join(packageDir, 'lipsync');
      this.copyDirectoryRecursive(lipsyncDir, lipsyncPackageDir);
      packageManifest.assets.lipsync = 'lipsync/';
    }
    
    // Copy visual prompts
    const visualsDir = path.join(this.config.visualsDir, `day-${paddedDay}`);
    if (fs.existsSync(visualsDir)) {
      const visualsPackageDir = path.join(packageDir, 'visuals');
      this.copyDirectoryRecursive(visualsDir, visualsPackageDir);
      packageManifest.assets.visuals = 'visuals/';
    }
    
    // Write package manifest
    fs.writeFileSync(
      path.join(packageDir, 'package.json'),
      JSON.stringify(packageManifest, null, 2)
    );
    
    console.log(`  ✅ Day ${dayNumber} packaged to: ${packageDir}`);
    return packageManifest;
  }
  
  /**
   * Copy directory recursively
   */
  copyDirectoryRecursive(src, dest) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    
    const entries = fs.readdirSync(src, { withFileTypes: true });
    
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      
      if (entry.isDirectory()) {
        this.copyDirectoryRecursive(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }
  
  /**
   * Run full production pipeline
   */
  async runFullPipeline(options = {}) {
    console.log(BANNER);
    
    const startDay = options.startDay || 1;
    const endDay = options.endDay || 30;
    const useMock = options.mock || false;
    
    console.log(`  📅 Processing days ${startDay}-${endDay}`);
    console.log(`  🎤 Audio: ${useMock ? 'MOCK' : 'ElevenLabs'}`);
    console.log('');
    
    this.initDirectories();
    
    // Stage 1: Lessons
    await this.runLessonGeneration();
    
    // Stage 2: Audio
    await this.runAudioGeneration(startDay, endDay, useMock);
    
    // Stage 3: Lip-sync
    await this.runLipSyncGeneration(startDay, endDay);
    
    // Stage 4: Visuals
    await this.runVisualGeneration(startDay, endDay);
    
    // Package each day
    console.log('\n  ═══ PACKAGING FOR DEPLOYMENT ═══');
    for (let day = startDay; day <= endDay; day++) {
      await this.packageDay(day);
    }
    
    // Final report
    this.printReport();
  }
  
  /**
   * Print final report
   */
  printReport() {
    const duration = Math.round((Date.now() - this.startTime) / 1000);
    
    console.log('\n═══════════════════════════════════════════════════════════════════════════════');
    console.log('  📊 PRODUCTION REPORT');
    console.log('═══════════════════════════════════════════════════════════════════════════════');
    console.log(`  ⏱️  Duration:        ${duration} seconds`);
    console.log(`  📚 Lessons:         ${this.stats.lessonsGenerated}`);
    console.log(`  🔊 Audio Days:      ${this.stats.audioGenerated}`);
    console.log(`  👄 Lip-sync Days:   ${this.stats.lipsyncGenerated}`);
    console.log(`  🎨 Visual Days:     ${this.stats.visualsGenerated}`);
    
    if (this.stats.errors.length > 0) {
      console.log(`  ⚠️  Errors:          ${this.stats.errors.length}`);
      for (const err of this.stats.errors) {
        console.log(`      - ${err.stage}: ${err.error}`);
      }
    }
    
    console.log('═══════════════════════════════════════════════════════════════════════════════');
    console.log('  ✨ GOLDEN V2 PRODUCTION COMPLETE');
    console.log('═══════════════════════════════════════════════════════════════════════════════');
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLI INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  const args = process.argv.slice(2);
  const orchestrator = new ProductionOrchestrator();
  
  // Parse arguments
  const getArg = (name) => {
    const arg = args.find(a => a.startsWith(`--${name}=`));
    return arg ? arg.split('=')[1] : null;
  };
  
  const hasFlag = (name) => args.includes(`--${name}`);
  
  const startDay = parseInt(getArg('start')) || 1;
  const endDay = parseInt(getArg('end')) || 30;
  const day = parseInt(getArg('day')) || 1;
  const useMock = hasFlag('mock');
  
  if (hasFlag('all') || args.length === 0) {
    // Full pipeline
    await orchestrator.runFullPipeline({ startDay, endDay, mock: useMock });
    
  } else if (hasFlag('lessons')) {
    // Lessons only
    orchestrator.initDirectories();
    await orchestrator.runLessonGeneration();
    orchestrator.printReport();
    
  } else if (hasFlag('audio')) {
    // Audio only
    orchestrator.initDirectories();
    await orchestrator.runAudioGeneration(startDay, endDay, useMock);
    orchestrator.printReport();
    
  } else if (hasFlag('lipsync')) {
    // Lip-sync only
    orchestrator.initDirectories();
    await orchestrator.runLipSyncGeneration(startDay, endDay);
    orchestrator.printReport();
    
  } else if (hasFlag('visuals')) {
    // Visuals only
    orchestrator.initDirectories();
    await orchestrator.runVisualGeneration(startDay, endDay);
    orchestrator.printReport();
    
  } else if (hasFlag('package')) {
    // Package only
    orchestrator.initDirectories();
    await orchestrator.packageDay(day);
    
  } else {
    console.log(`
Usage: node orchestrator.js [options]

Options:
  --all                 Run complete pipeline
  --lessons             Generate lesson DNA only
  --audio               Generate audio files
  --lipsync             Generate lip-sync data
  --visuals             Generate visual prompts
  --package             Package day for deployment

Parameters:
  --start=N             Start day (default: 1)
  --end=N               End day (default: 30)
  --day=N               Single day (for --package)
  --mock                Use mock audio (no API calls)

Examples:
  node orchestrator.js --all --mock               # Full pipeline with mock audio
  node orchestrator.js --lessons                  # Generate all 365 lesson DNAs
  node orchestrator.js --audio --start=1 --end=7  # Audio for first week
  node orchestrator.js --package --day=1          # Package day 1
    `);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

export { ProductionOrchestrator, CONFIG };
export default ProductionOrchestrator;

// Run if called directly
main().catch(console.error);







