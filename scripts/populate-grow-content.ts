#!/usr/bin/env npx tsx
/**
 * 🌱 POPULATE GROW TRACK CONTENT
 * 
 * Fills in fun_facts and reflection_prompts for the Grow (AI Fluency) track
 * based on the year2-ai-fluency curriculum JSON files.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Month curriculum files
const CURRICULUM_FILES = [
  'january_curriculum.json',
  'february_curriculum.json',
  'march_curriculum.json',
  'april_curriculum.json',
  'may_curriculum.json',
  'june_curriculum.json',
  'july_curriculum.json',
  'august_curriculum.json',
  'september_curriculum.json',
  'october_curriculum.json',
  'november_curriculum.json',
  'december_curriculum.json',
];

interface DayContent {
  day: number;
  date: string;
  title: string;
  learning_objective: string;
}

interface MonthCurriculum {
  year: number;
  month: string;
  theme: string;
  themeDescription: string;
  days: DayContent[];
}

/**
 * Generate fun facts from the learning objective
 */
function generateFunFacts(day: DayContent, theme: string): string[] {
  const title = day.title.split(' - ')[0];
  const objective = day.learning_objective;
  
  // Extract key concepts from the title and objective
  const facts: string[] = [
    `Today's topic "${title}" helps you understand AI better.`,
    objective.split(', ').slice(0, 1).join('').replace(/\.$/, '') + ' is a key skill for the future.',
    `This lesson is part of the "${theme}" theme in your AI fluency journey.`,
  ];
  
  return facts;
}

/**
 * Generate reflection prompts from the learning objective
 */
function generateReflectionPrompts(day: DayContent): string[] {
  const title = day.title.split(' - ')[0];
  
  return [
    `How does "${title}" change the way you think about AI?`,
    `What's one thing you'll do differently after learning about this?`,
  ];
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🌱 POPULATE GROW TRACK CONTENT                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  const basePath = path.join(process.cwd(), 'lessons', 'year2-ai-fluency');
  let totalUpdated = 0;
  let totalErrors = 0;

  for (const file of CURRICULUM_FILES) {
    const filePath = path.join(basePath, file);
    
    if (!fs.existsSync(filePath)) {
      console.log(`⚠️  Skipping ${file} (not found)`);
      continue;
    }

    console.log(`📖 Processing ${file}...`);
    
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const curriculum: MonthCurriculum = JSON.parse(content);
      
      console.log(`   Theme: ${curriculum.theme} (${curriculum.days.length} days)`);
      
      for (const day of curriculum.days) {
        const funFacts = generateFunFacts(day, curriculum.theme);
        const reflectionPrompts = generateReflectionPrompts(day);
        
        const { error } = await supabase
          .from('core_lessons')
          .update({
            fun_facts: funFacts,
            reflection_prompts: reflectionPrompts,
          })
          .eq('day_number', day.day)
          .eq('track', 'grow');

        if (error) {
          console.log(`   ❌ Day ${day.day}: ${error.message}`);
          totalErrors++;
        } else {
          totalUpdated++;
        }
      }
      
      console.log(`   ✅ Updated ${curriculum.days.length} days\n`);
      
    } catch (err: any) {
      console.log(`   ❌ Error: ${err.message}\n`);
      totalErrors++;
    }
  }

  console.log('═'.repeat(60));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(60));
  console.log(`✅ Updated: ${totalUpdated} lessons`);
  console.log(`❌ Errors: ${totalErrors}`);
  
  // Verify
  const { data, error } = await supabase
    .from('core_lessons')
    .select('day_number, fun_facts, reflection_prompts')
    .eq('track', 'grow')
    .not('fun_facts', 'is', null)
    .limit(5);
  
  if (data && data.length > 0) {
    console.log('\n📋 Sample populated lessons:');
    data.forEach(d => {
      console.log(`   Day ${d.day_number}: ${(d.fun_facts as any)?.length || 0} facts, ${(d.reflection_prompts as any)?.length || 0} reflections`);
    });
  }
}

main().catch(console.error);
