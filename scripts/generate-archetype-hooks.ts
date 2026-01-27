#!/usr/bin/env npx tsx
/**
 * 🎭 ARCHETYPE HOOKS GENERATOR
 * Generates archetype-specific SPARK hooks for Days 1-10
 * 
 * | Archetype | Hook Style |
 * |-----------|------------|
 * | 🧭 Explorer | "What if you could discover..." |
 * | 📐 Architect | "Here's the hidden structure behind..." |
 * | 💗 Empath | "Imagine how it felt when..." |
 * | ⚡ Rebel | "Everyone believes X. They're wrong..." |
 * | 🔮 Mystic | "There's a deeper pattern here..." |
 * | 🤲 Provider | "This could help everyone you know..." |
 * | 🤝 Diplomat | "Different people see this differently..." |
 * | 🔧 Maker | "Let's build something to prove this..." |
 * | 🔬 Scientist | "The evidence shows something surprising..." |
 * | ♟️ Strategist | "Here's how to use this to your advantage..." |
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

const ARCHETYPES = [
  { name: 'The Explorer', emoji: '🧭', prefix: "What if you could discover", style: 'adventure' },
  { name: 'The Architect', emoji: '📐', prefix: "Here's the hidden structure behind", style: 'analytical' },
  { name: 'The Empath', emoji: '💗', prefix: "Imagine how it feels when", style: 'emotional' },
  { name: 'The Rebel', emoji: '⚡', prefix: "Everyone believes this. But here's the truth", style: 'contrarian' },
  { name: 'The Mystic', emoji: '🔮', prefix: "There's a deeper pattern here", style: 'philosophical' },
  { name: 'The Provider', emoji: '🤲', prefix: "This could help everyone you care about", style: 'caring' },
  { name: 'The Diplomat', emoji: '🤝', prefix: "Different people see this differently", style: 'balanced' },
  { name: 'The Maker', emoji: '🔧', prefix: "Let's build something to prove", style: 'hands-on' },
  { name: 'The Scientist', emoji: '🔬', prefix: "The evidence shows something surprising", style: 'evidence' },
  { name: 'The Strategist', emoji: '♟️', prefix: "Here's how to use this to your advantage", style: 'tactical' },
  { name: 'The Storyteller', emoji: '📖', prefix: "Let me tell you about", style: 'narrative' },
  { name: 'The Survivor', emoji: '🛡️', prefix: "When things get tough, remember", style: 'resilient' },
];

function generateArchetypeHook(baseScript: string, archetype: typeof ARCHETYPES[0], topic: string): string {
  const opening = `${archetype.prefix} ${topic.toLowerCase()}... `;
  
  // Extract key insight from base script
  const sentences = baseScript.split(/[.!?]+/).filter(s => s.trim());
  const keyInsight = sentences.length > 1 ? sentences[1].trim() : sentences[0].trim();
  
  // Style-specific modifications
  let styled = keyInsight;
  switch (archetype.style) {
    case 'adventure':
      styled = keyInsight.replace(/you can/gi, "you'll discover how to");
      break;
    case 'contrarian':
      styled = "Most people think " + keyInsight.toLowerCase() + ". They're missing something.";
      break;
    case 'emotional':
      styled = "Think about " + keyInsight.toLowerCase() + ". How does that make you feel?";
      break;
    case 'tactical':
      styled = keyInsight + " And here's how you can use it.";
      break;
    case 'hands-on':
      styled = "We're going to prove " + keyInsight.toLowerCase() + " together.";
      break;
  }
  
  return opening + styled;
}

async function generateHooksForDay(dayNumber: number): Promise<number> {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  
  if (!fs.existsSync(filePath)) return 0;
  
  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const topic = content.meta.topic.en;
    const baseHook = content.phases?.hook?.script?.en || '';
    
    // Get core_lesson_id
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id')
      .eq('day_number', dayNumber)
      .single();
    
    if (!lesson) return 0;
    
    let success = 0;
    
    for (const archetype of ARCHETYPES) {
      const archetypeHook = generateArchetypeHook(baseHook, archetype, topic);
      
      const atomContent = {
        script: archetypeHook,
        kellyPose: 'explaining',
        kellyEmotion: archetype.style === 'emotional' ? 'empathetic' : 'curious',
        options: content.phases?.hook?.options || [],
      };
      
      const { error } = await supabase
        .from('lesson_atoms')
        .upsert({
          core_lesson_id: lesson.id,
          archetype: archetype.name,
          phase: 'Hook',
          content: atomContent,
        }, {
          onConflict: 'core_lesson_id,archetype,phase'
        });
      
      if (!error) success++;
    }
    
    return success;
  } catch (err) {
    console.error(`Day ${dayNumber} error:`, err);
    return 0;
  }
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 1;
  let endDay = 10;
  
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
║         🎭 ARCHETYPE HOOKS GENERATOR                         ║
╚══════════════════════════════════════════════════════════════╝
`);
  console.log(`Generating 12 archetype hooks for Days ${startDay}-${endDay}...\n`);
  
  let totalHooks = 0;
  
  for (let day = startDay; day <= endDay; day++) {
    const count = await generateHooksForDay(day);
    totalHooks += count;
    console.log(`  Day ${day}: ${count}/12 archetype hooks generated`);
  }
  
  const expected = (endDay - startDay + 1) * 12;
  console.log(`
📊 SUMMARY:
  Total hooks generated: ${totalHooks}/${expected}
  Days processed: ${endDay - startDay + 1}
`);
}

main().catch(console.error);
