/**
 * Populate Grow Track (AI Fluency) Lessons
 * 
 * Reads all year2-ai-fluency curriculum JSONs and inserts into core_lessons
 * with track = 'grow'
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

interface CurriculumDay {
  day: number;
  date: string;
  title: string;
  learning_objective: string;
}

interface CurriculumMonth {
  year: number;
  program: string;
  month: string;
  theme: string;
  themeDescription: string;
  days: CurriculumDay[];
}

const CURRICULUM_DIR = path.join(__dirname, '../lessons/year2-ai-fluency');

const MONTH_FILES = [
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

async function main() {
  console.log('🤖 Populating Grow Track (AI Fluency) lessons...\n');
  
  let totalInserted = 0;
  let totalSkipped = 0;
  
  for (const monthFile of MONTH_FILES) {
    const filePath = path.join(CURRICULUM_DIR, monthFile);
    
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠️ File not found: ${monthFile}`);
      continue;
    }
    
    const data: CurriculumMonth = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    console.log(`📅 ${data.month}: "${data.theme}" (${data.days.length} days)`);
    
    for (const day of data.days) {
      // Create lesson record
      const lesson = {
        day_number: day.day,
        track: 'grow',
        topic: day.title.split(' - ')[0], // First part before dash
        headline: day.title,
        universal_truth: day.learning_objective,
        category: data.theme,
        emoji: getEmojiForTheme(data.theme),
        fun_facts: [day.learning_objective],
        reflection_prompts: [`What did you learn about ${day.title.split(' - ')[0].toLowerCase()}?`],
      };
      
      // Upsert (insert or update)
      const { error } = await supabase
        .from('core_lessons')
        .upsert(lesson, { 
          onConflict: 'day_number,track',
          ignoreDuplicates: false 
        });
      
      if (error) {
        if (error.code === '23505') {
          totalSkipped++;
        } else {
          console.error(`  ❌ Day ${day.day}: ${error.message}`);
        }
      } else {
        totalInserted++;
      }
    }
  }
  
  console.log(`\n✅ Complete!`);
  console.log(`   Inserted/Updated: ${totalInserted}`);
  console.log(`   Skipped (exists): ${totalSkipped}`);
  console.log(`   Total: ${totalInserted + totalSkipped}`);
}

function getEmojiForTheme(theme: string): string {
  const emojiMap: Record<string, string> = {
    'Foundations': '🏗️',
    'Questioning': '❓',
    'Verification': '✅',
    'Memory & Learning': '🧠',
    'Creativity & AI': '🎨',
    'Communication': '💬',
    'Ethics & Responsibility': '⚖️',
    'Systems Thinking': '🔄',
    'Human Capabilities': '🌟',
    'Privacy & Digital Citizenship': '🔒',
    'Future of Learning': '🚀',
    'Integration & Reflection': '🔮',
  };
  return emojiMap[theme] || '🤖';
}

main().catch(console.error);
