#!/usr/bin/env node
/**
 * Kelly Production Orchestrator
 * 
 * Master script that coordinates all generation steps:
 * 1. Images (LoRA) → 2. Animations (SVD) → 3. Audio (ElevenLabs) → 4. Lipsync (Wav2Lip)
 * 
 * Run: node production-orchestrator.cjs --day 1
 *      node production-orchestrator.cjs --days 1-5
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const { spawn } = require('child_process');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

class ProductionOrchestrator {
  constructor(options = {}) {
    this.dayStart = options.dayStart || 1;
    this.dayEnd = options.dayEnd || 1;
    this.startTime = Date.now();
    this.stats = { images: 0, animations: 0, audio: 0, videos: 0, cost: 0 };
  }
  
  async run() {
    console.log('═'.repeat(70));
    console.log('🎬 KELLY PRODUCTION ORCHESTRATOR');
    console.log('   Full pipeline: Image → Animation → Audio → Lipsync');
    console.log('═'.repeat(70));
    console.log(`\n  Days: ${this.dayStart} to ${this.dayEnd}`);
    
    for (let day = this.dayStart; day <= this.dayEnd; day++) {
      await this.processDay(day);
    }
    
    this.printSummary();
  }
  
  async processDay(day) {
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`📅 DAY ${day}`);
    console.log('─'.repeat(70));
    
    const status = await this.getDayStatus(day);
    
    // Step 1: Images
    if (status.images < 5) {
      console.log(`\n  [1/4] 🎨 Generating ${5 - status.images} images...`);
      await this.runScript('batch-image-generator.cjs', ['--days', day.toString()]);
      this.stats.images += 5 - status.images;
    } else {
      console.log('\n  [1/4] ✅ Images complete');
    }
    
    // Step 2: Animations
    const newStatus = await this.getDayStatus(day);
    if (newStatus.animations < 5) {
      console.log(`\n  [2/4] 🎬 Generating ${5 - newStatus.animations} animations...`);
      await this.runScript('batch-animation-generator.cjs', ['--days', day.toString()]);
      this.stats.animations += 5 - newStatus.animations;
    } else {
      console.log('\n  [2/4] ✅ Animations complete');
    }
    
    // Step 3: Audio
    const audioStatus = await this.getDayStatus(day);
    const expectedAudio = await this.getExpectedAudioCount(day);
    if (audioStatus.audio < expectedAudio) {
      console.log(`\n  [3/4] 🎙️ Generating ${expectedAudio - audioStatus.audio} audio files...`);
      await this.runScript('generate-day-audio.cjs', ['--day', day.toString()]);
      this.stats.audio += expectedAudio - audioStatus.audio;
    } else {
      console.log(`\n  [3/4] ✅ Audio complete (${audioStatus.audio} files)`);
    }
    
    // Step 4: Lipsync
    const finalStatus = await this.getDayStatus(day);
    const expectedVideos = expectedAudio;
    if (finalStatus.videos < expectedVideos) {
      console.log(`\n  [4/4] 👄 Generating ${expectedVideos - finalStatus.videos} videos...`);
      await this.runScript('generate-day-lipsync.cjs', ['--day', day.toString()]);
      this.stats.videos += expectedVideos - finalStatus.videos;
    } else {
      console.log(`\n  [4/4] ✅ Videos complete (${finalStatus.videos} files)`);
    }
    
    console.log(`\n  ✅ Day ${day} complete!`);
  }
  
  async getDayStatus(day) {
    const { data } = await supabase
      .from('kelly_video_assets')
      .select('asset_type')
      .eq('day_number', day);
    
    const counts = { images: 0, animations: 0, audio: 0, videos: 0 };
    data?.forEach(d => {
      if (d.asset_type === 'image') counts.images++;
      if (d.asset_type === 'animation') counts.animations++;
      if (d.asset_type === 'audio') counts.audio++;
      if (d.asset_type === 'video') counts.videos++;
    });
    return counts;
  }
  
  async getExpectedAudioCount(day) {
    const { count } = await supabase
      .from('lesson_atoms')
      .select('*', { count: 'exact', head: true })
      .eq('core_lessons.day_number', day);
    return count || 75; // Default to 75 if can't fetch
  }
  
  runScript(script, args) {
    return new Promise((resolve, reject) => {
      const proc = spawn('node', [script, ...args], {
        cwd: __dirname,
        env: process.env,
        stdio: 'inherit'
      });
      
      proc.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`Script ${script} exited with code ${code}`));
      });
    });
  }
  
  printSummary() {
    const duration = ((Date.now() - this.startTime) / 1000 / 60).toFixed(1);
    
    console.log('\n' + '═'.repeat(70));
    console.log('📊 PRODUCTION COMPLETE');
    console.log('═'.repeat(70));
    console.log(`\n  Days processed: ${this.dayEnd - this.dayStart + 1}`);
    console.log(`  Duration: ${duration} minutes`);
    console.log(`\n  Generated:`);
    console.log(`    Images:     ${this.stats.images}`);
    console.log(`    Animations: ${this.stats.animations}`);
    console.log(`    Audio:      ${this.stats.audio}`);
    console.log(`    Videos:     ${this.stats.videos}`);
    
    // Estimate costs
    const cost = (
      this.stats.images * 0.003 +
      this.stats.animations * 0.05 +
      this.stats.audio * 0.002 +
      this.stats.videos * 0.02
    );
    console.log(`\n  Est. Cost: $${cost.toFixed(2)}`);
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  
  let dayStart = 1, dayEnd = 1;
  
  const dayIndex = args.indexOf('--day');
  if (dayIndex > -1) {
    dayStart = dayEnd = parseInt(args[dayIndex + 1]);
  }
  
  const daysIndex = args.indexOf('--days');
  if (daysIndex > -1) {
    const range = args[daysIndex + 1];
    if (range.includes('-')) {
      [dayStart, dayEnd] = range.split('-').map(Number);
    } else {
      dayStart = dayEnd = parseInt(range);
    }
  }
  
  if (args.includes('--help') || args.length === 0) {
    console.log(`
Kelly Production Orchestrator

Full pipeline automation: Images → Animations → Audio → Lipsync

Usage:
  node production-orchestrator.cjs --day 1        # Process Day 1
  node production-orchestrator.cjs --days 1-5    # Process Days 1-5
  
Resumes from where it left off - skips completed assets.
`);
    return;
  }
  
  const orchestrator = new ProductionOrchestrator({ dayStart, dayEnd });
  await orchestrator.run();
}

main().catch(console.error);



