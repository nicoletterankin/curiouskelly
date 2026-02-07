#!/usr/bin/env npx tsx
/**
 * 🌱 BULK DATABASE SEEDER
 * Loads all 365 lesson JSON files into Supabase lesson_shards
 * 
 * Usage:
 *   npx tsx scripts/bulk-seed-database.ts --days 1-30
 *   npx tsx scripts/bulk-seed-database.ts --all
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
const PHASES = ['hook', 'cliff', 'q1', 'q2', 'q3', 'wisdom', 'outro'];

interface LessonFile {
  meta: { day: number; topic: { en: string; es: string } };
  phases: Record<string, { script: { en: string; es: string }; options: any[] }>;
}

async function getCoreLesson(dayNumber: number) {
  // Use limit(1) instead of single() to handle days with multiple tracks
  const { data } = await supabase
    .from('core_lessons')
    .select('id')
    .eq('day_number', dayNumber)
    .limit(1);
  return data?.[0]?.id;
}

async function seedDay(dayNumber: number): Promise<boolean> {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  
  if (!fs.existsSync(filePath)) {
    console.log(`  ⚠️  Day ${dayNumber}: File not found`);
    return false;
  }
  
  try {
    const content: LessonFile = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const coreId = await getCoreLesson(dayNumber);
    
    if (!coreId) {
      console.log(`  ⚠️  Day ${dayNumber}: No core_lesson found`);
      return false;
    }
    
    // Build script_content JSONB
    const scriptContent = {
      topic: content.meta.topic,
      phases: {} as Record<string, any>
    };
    
    for (const phase of PHASES) {
      if (content.phases[phase]) {
        scriptContent.phases[phase] = {
          script_en: content.phases[phase].script?.en || '',
          script_es: content.phases[phase].script?.es || '',
          options: content.phases[phase].options || [],
        };
      }
    }
    
    // Insert/upsert for adult age
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
      return false;
    }
    
    console.log(`  ✅ Day ${dayNumber}: ${content.meta.topic.en}`);
    return true;
  } catch (err) {
    console.log(`  ❌ Day ${dayNumber}: ${(err as Error).message}`);
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 1;
  let endDay = 365;
  
  for (const arg of args) {
    if (arg.startsWith('--days=')) {
      const range = arg.split('=')[1];
      if (range.includes('-')) {
        [startDay, endDay] = range.split('-').map(Number);
      } else {
        startDay = endDay = Number(range);
      }
    }
    if (arg === '--all') {
      startDay = 1;
      endDay = 365;
    }
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║         🌱 BULK DATABASE SEEDER                              ║
╚══════════════════════════════════════════════════════════════╝
`);
  console.log(`Seeding Days ${startDay} to ${endDay}...\n`);
  
  let success = 0;
  let failed = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    const result = await seedDay(day);
    if (result) success++; else failed++;
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝

  ✅ Succeeded: ${success}
  ❌ Failed: ${failed}
  📁 Total: ${success + failed}
`);
}

main().catch(console.error);
