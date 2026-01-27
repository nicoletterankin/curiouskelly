#!/usr/bin/env npx tsx
/**
 * 👶👦🧑 AGE VARIANT GENERATOR
 * Generates YOUNG (4-6) and TEEN (13-17) script variants
 * 
 * YOUNG ADAPTATIONS:
 * - Shorter sentences (max 10 words)
 * - No statistics or abstract numbers
 * - Physical analogies: "Imagine you're a butterfly..."
 * - More questions, less exposition
 * - Celebratory tone: "You just thought like a scientist!"
 * 
 * TEEN ADAPTATIONS:
 * - Direct, not cute
 * - Reference their world: social media, school pressure
 * - Acknowledge complexity: "It's not that simple, is it?"
 * - Respect their skepticism
 * - Real-world examples they care about
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

// Transform adult script to young (4-6 year old)
function transformToYoung(adultScript: string, topic: string): string {
  // Simplify sentences
  let young = adultScript
    .replace(/\d+%/g, 'lots of')
    .replace(/\d+ percent/g, 'lots of')
    .replace(/research shows/gi, 'guess what?')
    .replace(/studies show/gi, 'did you know?')
    .replace(/scientists found/gi, 'people discovered')
    .replace(/billion/g, 'super super many')
    .replace(/million/g, 'super many')
    .replace(/thousand/g, 'lots and lots');
  
  // Shorten and simplify
  const sentences = young.split(/[.!?]+/).filter(s => s.trim());
  const simplified = sentences.map(s => {
    const words = s.trim().split(/\s+/);
    if (words.length > 12) {
      return words.slice(0, 10).join(' ') + '!';
    }
    return s.trim();
  });
  
  return simplified.slice(0, 3).join('. ') + '!';
}

// Transform adult script to teen (13-17 year old)
function transformToTeen(adultScript: string, topic: string): string {
  let teen = adultScript
    .replace(/Here's what's fascinating/gi, "Here's something that might surprise you")
    .replace(/Did you know/gi, "You probably heard")
    .replace(/Let me tell you/gi, "Check this out")
    .replace(/wonderful/gi, 'actually pretty cool')
    .replace(/amazing/gi, 'kind of wild')
    .replace(/Beautiful/gi, "That's real");
  
  // Add teen-relevant framing
  if (topic.toLowerCase().includes('listen')) {
    teen += " Think about your group chats - same thing applies there.";
  }
  if (topic.toLowerCase().includes('habit')) {
    teen += " This is why streaks on apps actually work.";
  }
  
  return teen;
}

async function generateVariantsForDay(dayNumber: number): Promise<{ young: boolean; teen: boolean }> {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  const result = { young: false, teen: false };
  
  if (!fs.existsSync(filePath)) return result;
  
  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const topic = content.meta.topic.en;
    
    // Get core_lesson_id
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id')
      .eq('day_number', dayNumber)
      .single();
    
    if (!lesson) return result;
    
    // Generate YOUNG variant
    const youngContent = {
      topic: { en: topic, es: content.meta.topic.es },
      phases: {} as Record<string, any>
    };
    
    for (const [phase, data] of Object.entries(content.phases || {})) {
      const phaseData = data as any;
      if (phaseData.script?.en) {
        youngContent.phases[phase] = {
          script_en: transformToYoung(phaseData.script.en, topic),
          script_es: phaseData.script.es || '',
          options: (phaseData.options || []).slice(0, 2).map((opt: any) => ({
            ...opt,
            text: { en: opt.text?.en?.split(' ').slice(0, 6).join(' '), es: opt.text?.es }
          }))
        };
      }
    }
    
    const { error: youngError } = await supabase
      .from('lesson_shards')
      .upsert({
        core_lesson_id: lesson.id,
        age: 5,
        region: 'en',
        tone: 'playful',
        birth_year: 2021,
        script_content: youngContent,
      }, { onConflict: 'core_lesson_id,age,region,tone,birth_year' });
    
    result.young = !youngError;
    
    // Generate TEEN variant
    const teenContent = {
      topic: { en: topic, es: content.meta.topic.es },
      phases: {} as Record<string, any>
    };
    
    for (const [phase, data] of Object.entries(content.phases || {})) {
      const phaseData = data as any;
      if (phaseData.script?.en) {
        teenContent.phases[phase] = {
          script_en: transformToTeen(phaseData.script.en, topic),
          script_es: phaseData.script.es || '',
          options: phaseData.options || []
        };
      }
    }
    
    const { error: teenError } = await supabase
      .from('lesson_shards')
      .upsert({
        core_lesson_id: lesson.id,
        age: 15,
        region: 'en',
        tone: 'direct',
        birth_year: 2011,
        script_content: teenContent,
      }, { onConflict: 'core_lesson_id,age,region,tone,birth_year' });
    
    result.teen = !teenError;
    
    return result;
  } catch (err) {
    console.error(`Day ${dayNumber} error:`, err);
    return result;
  }
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 1;
  let endDay = 30;
  
  for (const arg of args) {
    if (arg.startsWith('--days=')) {
      const range = arg.split('=')[1];
      if (range.includes('-')) {
        [startDay, endDay] = range.split('-').map(Number);
      }
    }
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║         👶👦🧑 AGE VARIANT GENERATOR                         ║
╚══════════════════════════════════════════════════════════════╝
`);
  console.log(`Generating variants for Days ${startDay}-${endDay}...\n`);
  
  let youngSuccess = 0, teenSuccess = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    const result = await generateVariantsForDay(day);
    if (result.young) youngSuccess++;
    if (result.teen) teenSuccess++;
    console.log(`  Day ${day}: Young ${result.young ? '✅' : '❌'} | Teen ${result.teen ? '✅' : '❌'}`);
  }
  
  console.log(`
📊 SUMMARY:
  Young variants: ${youngSuccess}/${endDay - startDay + 1}
  Teen variants:  ${teenSuccess}/${endDay - startDay + 1}
`);
}

main().catch(console.error);
