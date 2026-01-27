#!/usr/bin/env npx tsx
/**
 * 🚀 BULK LOADER: Days 31-365
 * Loads all JSON lesson files into lesson_shards
 * Runs autonomously until complete
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY!
);

const LESSONS_DIR = path.join(process.cwd(), 'public', 'lessons');

interface LessonContent {
  meta: { day: number; topic: { en: string; es: string }; emoji: string; category: string };
  phases: Record<string, { script: { en: string }; options?: any[] }>;
  headline: { en: string };
  universal_truth: { en: string };
  fun_facts: Array<{ en: string }>;
}

// Get core_lesson_id - prefer the natural science track
async function getCoreLesson(dayNumber: number, topic: string): Promise<string | null> {
  const { data } = await supabase
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', dayNumber);
  
  if (!data || data.length === 0) return null;
  
  // Try to match by topic first
  const match = data.find(d => d.topic.toLowerCase().includes(topic.toLowerCase().split(' ')[0]));
  return match?.id || data[0]?.id;
}

async function loadDay(dayNumber: number): Promise<{ success: boolean; topic?: string }> {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  
  if (!fs.existsSync(filePath)) {
    return { success: false };
  }
  
  try {
    const content: LessonContent = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const topic = content.meta.topic.en;
    
    const coreId = await getCoreLesson(dayNumber, topic);
    if (!coreId) {
      console.log(`  ⚠️  Day ${dayNumber}: No core_lesson found`);
      return { success: false };
    }
    
    // Build script_content JSONB with all phases
    const scriptContent = {
      topic: content.meta.topic,
      emoji: content.meta.emoji,
      category: content.meta.category,
      headline: content.headline?.en || '',
      universal_truth: content.universal_truth?.en || '',
      fun_facts: content.fun_facts?.map(f => f.en) || [],
      phases: {} as Record<string, any>
    };
    
    const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];
    for (const phase of PHASES) {
      if (content.phases[phase]) {
        scriptContent.phases[phase] = {
          script_en: content.phases[phase].script?.en || '',
          options: content.phases[phase].options || [],
        };
      }
    }
    
    // Insert/upsert adult shard
    const { error } = await supabase
      .from('lesson_shards')
      .upsert({
        core_lesson_id: coreId,
        age: 26,
        region: 'en',
        tone: 'curious',
        birth_year: 1999,
        script_content: scriptContent,
      }, {
        onConflict: 'core_lesson_id,age,region,tone,birth_year'
      });
    
    if (error) {
      console.log(`  ❌ Day ${dayNumber}: ${error.message}`);
      return { success: false };
    }
    
    return { success: true, topic };
  } catch (err) {
    console.log(`  ❌ Day ${dayNumber}: ${(err as Error).message}`);
    return { success: false };
  }
}

async function loadBatch(startDay: number, endDay: number): Promise<{ loaded: number; failed: number }> {
  let loaded = 0;
  let failed = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    const result = await loadDay(day);
    if (result.success) {
      loaded++;
      process.stdout.write(`\r  Loading... Day ${day}: ${result.topic?.substring(0, 30)}...`);
    } else {
      failed++;
    }
  }
  console.log('');
  return { loaded, failed };
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 31;
  let endDay = 365;
  let batchSize = 50;
  
  for (const arg of args) {
    if (arg.startsWith('--start=')) startDay = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--end=')) endDay = parseInt(arg.split('=')[1]);
    if (arg.startsWith('--batch=')) batchSize = parseInt(arg.split('=')[1]);
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║         🚀 BULK LOADER: Days ${startDay}-${endDay}                        ║
╚══════════════════════════════════════════════════════════════╝
`);
  
  let totalLoaded = 0;
  let totalFailed = 0;
  
  for (let batchStart = startDay; batchStart <= endDay; batchStart += batchSize) {
    const batchEnd = Math.min(batchStart + batchSize - 1, endDay);
    console.log(`\n📦 Batch: Days ${batchStart}-${batchEnd}`);
    
    const result = await loadBatch(batchStart, batchEnd);
    totalLoaded += result.loaded;
    totalFailed += result.failed;
    
    console.log(`  ✅ Loaded: ${result.loaded} | ❌ Failed: ${result.failed}`);
    console.log(`  📊 Progress: ${totalLoaded}/${endDay - startDay + 1} (${Math.round(totalLoaded / (endDay - startDay + 1) * 100)}%)`);
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                    📊 FINAL SUMMARY                          ║
╚══════════════════════════════════════════════════════════════╝

  ✅ Total Loaded: ${totalLoaded}
  ❌ Total Failed: ${totalFailed}
  📁 Days Processed: ${endDay - startDay + 1}
`);
}

main().catch(console.error);
