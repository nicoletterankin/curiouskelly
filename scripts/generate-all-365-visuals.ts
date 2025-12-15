#!/usr/bin/env npx tsx
/**
 * 🚀 GENERATE ALL 365 LESSON VISUALS
 * 
 * Master orchestrator that generates and links visuals for all 365 lessons.
 * Runs in batches to avoid rate limits and memory issues.
 * 
 * Usage:
 *   npx tsx scripts/generate-all-365-visuals.ts
 *   npx tsx scripts/generate-all-365-visuals.ts --start=51
 *   npx tsx scripts/generate-all-365-visuals.ts --batch-size=20
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const BATCH_SIZE = 40; // Generate 40 days at a time
const DELAY_BETWEEN_BATCHES = 5000; // 5 seconds between batches

interface BatchResult {
  start: number;
  end: number;
  success: boolean;
  error?: string;
}

function runCommand(command: string, args: string[]): Promise<{ success: boolean; output: string }> {
  return new Promise((resolve) => {
    console.log(`\n💻 Running: ${command} ${args.join(' ')}`);
    
    const proc = spawn(command, args, {
      cwd: process.cwd(),
      shell: true,
      stdio: 'inherit',
    });
    
    proc.on('close', (code) => {
      resolve({
        success: code === 0,
        output: '',
      });
    });
    
    proc.on('error', (err) => {
      resolve({
        success: false,
        output: err.message,
      });
    });
  });
}

async function generateBatch(start: number, end: number): Promise<boolean> {
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  📦 BATCH: Days ${start} to ${end}`);
  console.log(`${'█'.repeat(60)}`);
  
  // Step 1: Generate visuals
  const genResult = await runCommand('npx', [
    'tsx',
    'scripts/generate-lesson-visuals.ts',
    String(start),
    String(end),
    '--delay-ms=2500',
  ]);
  
  if (!genResult.success) {
    console.error(`❌ Generation failed for batch ${start}-${end}`);
    return false;
  }
  
  // Step 2: Link to atoms
  const linkResult = await runCommand('npx', [
    'tsx',
    'scripts/link-gemini-visuals-to-atoms.ts',
    `--range=${start}-${end}`,
  ]);
  
  if (!linkResult.success) {
    console.error(`❌ Linking failed for batch ${start}-${end}`);
    return false;
  }
  
  console.log(`✅ Batch ${start}-${end} complete`);
  return true;
}

async function main() {
  const args = process.argv.slice(2);
  const startArg = args.find(a => a.startsWith('--start='));
  const batchSizeArg = args.find(a => a.startsWith('--batch-size='));
  
  const startDay = startArg ? parseInt(startArg.split('=')[1]) : 1;
  const batchSize = batchSizeArg ? parseInt(batchSizeArg.split('=')[1]) : BATCH_SIZE;
  
  console.log(`\n${'█'.repeat(60)}`);
  console.log(`  🚀 GENERATE ALL 365 LESSON VISUALS`);
  console.log(`  Starting from Day ${startDay}`);
  console.log(`  Batch size: ${batchSize} days`);
  console.log(`${'█'.repeat(60)}`);
  
  const results: BatchResult[] = [];
  let currentDay = startDay;
  
  while (currentDay <= 365) {
    const batchEnd = Math.min(currentDay + batchSize - 1, 365);
    
    const success = await generateBatch(currentDay, batchEnd);
    
    results.push({
      start: currentDay,
      end: batchEnd,
      success,
    });
    
    if (!success) {
      console.error(`\n⚠️ Batch failed. Stopping at Day ${currentDay}`);
      break;
    }
    
    currentDay = batchEnd + 1;
    
    if (currentDay <= 365) {
      console.log(`\n⏳ Waiting ${DELAY_BETWEEN_BATCHES}ms before next batch...`);
      await new Promise(r => setTimeout(r, DELAY_BETWEEN_BATCHES));
    }
  }
  
  // Summary
  console.log(`\n${'═'.repeat(60)}`);
  console.log(`📊 FINAL SUMMARY`);
  console.log(`${'═'.repeat(60)}`);
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`✅ Successful batches: ${successful.length}`);
  console.log(`❌ Failed batches: ${failed.length}`);
  
  if (successful.length > 0) {
    const totalDays = successful.reduce((sum, r) => sum + (r.end - r.start + 1), 0);
    console.log(`📅 Total days processed: ${totalDays}`);
  }
  
  if (failed.length > 0) {
    console.log(`\n⚠️ Failed batches:`);
    failed.forEach(r => console.log(`   Days ${r.start}-${r.end}`));
  }
  
  // Save progress log
  const logPath = path.join(process.cwd(), 'visual-generation-log.json');
  fs.writeFileSync(logPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    startDay,
    batchSize,
    results,
  }, null, 2));
  
  console.log(`\n📝 Progress log saved: ${logPath}`);
}

main().catch(console.error);
