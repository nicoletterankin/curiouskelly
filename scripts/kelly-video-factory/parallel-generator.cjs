#!/usr/bin/env node
/**
 * Kelly Parallel Generator
 * 
 * Runs multiple generation jobs in parallel for speed.
 * Uses worker pools for different stages of the pipeline.
 * 
 * Run: node parallel-generator.cjs --stage images --days 1-30 --workers 3
 */

require('dotenv').config({ path: require('path').join(__dirname, '../../.env') });

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const LOG_DIR = path.join(__dirname, '../../template-forge/logs');
fs.mkdirSync(LOG_DIR, { recursive: true });

class ParallelGenerator {
  constructor(options = {}) {
    this.maxWorkers = options.workers || 3;
    this.stage = options.stage || 'images';
    this.dayStart = options.dayStart || 1;
    this.dayEnd = options.dayEnd || 30;
    this.activeWorkers = 0;
    this.queue = [];
    this.results = { completed: 0, failed: 0, costs: 0 };
    this.startTime = Date.now();
    this.jobId = null;
  }
  
  async createJob() {
    const { data } = await supabase.from('kelly_generation_jobs').insert({
      job_type: `${this.stage}_batch`,
      day_start: this.dayStart,
      day_end: this.dayEnd,
      status: 'running',
      total_items: this.dayEnd - this.dayStart + 1,
      quality_tier: 'standard',
      started_at: new Date().toISOString(),
      config: { workers: this.maxWorkers }
    }).select().single();
    
    this.jobId = data?.id;
    return this.jobId;
  }
  
  async updateJob(updates) {
    if (!this.jobId) return;
    await supabase.from('kelly_generation_jobs').update(updates).eq('id', this.jobId);
  }
  
  async run() {
    console.log('═'.repeat(70));
    console.log('⚡ PARALLEL GENERATOR');
    console.log('═'.repeat(70));
    console.log(`\n  Stage: ${this.stage}`);
    console.log(`  Days: ${this.dayStart}-${this.dayEnd}`);
    console.log(`  Workers: ${this.maxWorkers}`);
    
    await this.createJob();
    console.log(`  Job ID: ${this.jobId}\n`);
    
    // Build queue
    for (let day = this.dayStart; day <= this.dayEnd; day++) {
      this.queue.push(day);
    }
    
    console.log(`  Queue: ${this.queue.length} items\n`);
    
    // Process queue
    const promises = [];
    for (let i = 0; i < this.maxWorkers; i++) {
      promises.push(this.worker(i));
    }
    
    await Promise.all(promises);
    
    // Complete job
    const duration = ((Date.now() - this.startTime) / 1000 / 60).toFixed(1);
    
    await this.updateJob({
      status: this.results.failed > 0 ? 'completed' : 'completed',
      completed_items: this.results.completed,
      failed_items: this.results.failed,
      actual_cost_usd: this.results.costs,
      completed_at: new Date().toISOString()
    });
    
    console.log('\n' + '═'.repeat(70));
    console.log('📊 PARALLEL GENERATION COMPLETE');
    console.log('═'.repeat(70));
    console.log(`\n  Completed: ${this.results.completed}`);
    console.log(`  Failed: ${this.results.failed}`);
    console.log(`  Duration: ${duration} minutes`);
    console.log(`  Cost: $${this.results.costs.toFixed(2)}`);
  }
  
  async worker(workerId) {
    while (this.queue.length > 0) {
      const day = this.queue.shift();
      if (!day) break;
      
      console.log(`  [Worker ${workerId}] Processing day ${day}...`);
      
      try {
        const result = await this.processDay(day, workerId);
        this.results.completed++;
        this.results.costs += result.cost || 0;
        
        await this.updateJob({
          completed_items: this.results.completed,
          failed_items: this.results.failed
        });
        
        console.log(`  [Worker ${workerId}] Day ${day} ✅`);
        
      } catch (error) {
        this.results.failed++;
        console.log(`  [Worker ${workerId}] Day ${day} ❌ ${error.message}`);
        
        await this.updateJob({
          completed_items: this.results.completed,
          failed_items: this.results.failed,
          last_error: error.message,
          error_count: this.results.failed
        });
      }
    }
  }
  
  async processDay(day, workerId) {
    const script = this.getScriptForStage();
    const logFile = path.join(LOG_DIR, `worker_${workerId}_day_${day}.log`);
    
    return new Promise((resolve, reject) => {
      const proc = spawn('node', [script, '--days', day.toString()], {
        cwd: path.join(__dirname),
        env: process.env,
        stdio: ['ignore', 'pipe', 'pipe']
      });
      
      let output = '';
      proc.stdout.on('data', (data) => { output += data; });
      proc.stderr.on('data', (data) => { output += data; });
      
      proc.on('close', (code) => {
        fs.writeFileSync(logFile, output);
        
        if (code === 0) {
          // Estimate cost based on stage
          const costPerDay = {
            'images': 0.015, // 5 images × $0.003
            'animations': 0.25, // 5 animations × $0.05
            'audio': 0.01, // ~60 audio files × minimal cost
            'lipsync': 0.60 // ~60 lipsync videos × $0.01
          };
          resolve({ cost: costPerDay[this.stage] || 0 });
        } else {
          reject(new Error(`Exit code ${code}`));
        }
      });
    });
  }
  
  getScriptForStage() {
    const scripts = {
      'images': 'batch-image-generator.cjs',
      'animations': 'batch-animation-generator.cjs',
      'audio': 'batch-audio-generator.cjs',
      'lipsync': 'batch-lipsync-generator.cjs'
    };
    return scripts[this.stage] || 'batch-image-generator.cjs';
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);
  
  const stageIndex = args.indexOf('--stage');
  const stage = stageIndex > -1 ? args[stageIndex + 1] : 'images';
  
  const daysIndex = args.indexOf('--days');
  let dayStart = 1, dayEnd = 30;
  if (daysIndex > -1) {
    const daysArg = args[daysIndex + 1];
    if (daysArg.includes('-')) {
      [dayStart, dayEnd] = daysArg.split('-').map(Number);
    } else {
      dayStart = dayEnd = parseInt(daysArg);
    }
  }
  
  const workersIndex = args.indexOf('--workers');
  const workers = workersIndex > -1 ? parseInt(args[workersIndex + 1]) : 3;
  
  if (args.includes('--help') || args.length === 0) {
    console.log(`
Parallel Generator

Usage:
  node parallel-generator.cjs --stage <stage> --days <range> --workers <n>

Stages:
  images      Generate Kelly LoRA images
  animations  Generate SVD animations
  audio       Generate ElevenLabs audio
  lipsync     Apply lipsync to animations

Examples:
  node parallel-generator.cjs --stage images --days 1-30 --workers 3
  node parallel-generator.cjs --stage animations --days 1-5 --workers 2
`);
    return;
  }
  
  const generator = new ParallelGenerator({
    stage,
    dayStart,
    dayEnd,
    workers
  });
  
  await generator.run();
}

main().catch(console.error);



