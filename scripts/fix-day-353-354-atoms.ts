#!/usr/bin/env npx tsx
/**
 * 🔧 FIX DAY 353-354 ATOMS
 * 
 * Fixes lesson atoms to match LESSON_GENERATION_SPEC:
 * - Trims options to exactly 2 (removes 3rd option)
 * - Adds simulated comments (2-3 per phase)
 * - Generates missing Cliff and Outro phases
 * 
 * Usage:
 *   npx tsx scripts/fix-day-353-354-atoms.ts
 *   npx tsx scripts/fix-day-353-354-atoms.ts --day 353
 *   npx tsx scripts/fix-day-353-354-atoms.ts --dry-run
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

// =============================================================================
// CONFIGURATION
// =============================================================================

const DAYS_TO_FIX = [353, 354];

const ARCHETYPES = [
  'The Scientist', 'The Explorer', 'The Rebel', 'The Architect',
  'The Diplomat', 'The Empath', 'The MacGyver', 'The Mystic',
  'The Provider', 'The Storyteller', 'The Strategist', 'The Survivor'
];

const PHASES = ['Hook', 'Cliff', 'Fact1', 'Fact2', 'Fact3', 'Wisdom', 'Outro'];

// Simulated comment templates per phase (with ✨ Trust & Safety indicator)
const COMMENT_TEMPLATES: Record<string, Array<{emoji: string, text: string, author: string}>> = {
  Hook: [
    { emoji: '✨', text: 'Ooh this is going to be good!', author: 'curious_maya' },
    { emoji: '✨', text: 'I was just thinking about this!', author: 'wonder_kid_22' },
    { emoji: '✨', text: 'Kelly always picks the best topics 💜', author: 'daily_learner' },
  ],
  Cliff: [
    { emoji: '✨', text: 'Tough choice! I went with A', author: 'explorer_sam' },
    { emoji: '✨', text: 'Both sound interesting...', author: 'mindful_mia' },
    { emoji: '✨', text: 'I love when we get to choose!', author: 'path_finder' },
  ],
  Fact1: [
    { emoji: '✨', text: 'I did not know that!', author: 'science_lover' },
    { emoji: '✨', text: 'This connects to yesterday\'s lesson!', author: 'pattern_seeker' },
    { emoji: '✨', text: 'Writing this one down 📝', author: 'note_taker_101' },
  ],
  Fact2: [
    { emoji: '✨', text: 'Mind = blown 🤯', author: 'curious_cat_42' },
    { emoji: '✨', text: 'This changes how I think about it', author: 'deep_thinker' },
    { emoji: '✨', text: 'Kelly explains things so clearly!', author: 'grateful_learner' },
  ],
  Fact3: [
    { emoji: '✨', text: 'I\'m going to try this today!', author: 'action_taker' },
    { emoji: '✨', text: 'This reminds me of my own experience', author: 'personal_story' },
    { emoji: '✨', text: 'The pieces are coming together', author: 'puzzle_solver' },
  ],
  Wisdom: [
    { emoji: '✨', text: 'This really resonates with me 💜', author: 'reflective_soul' },
    { emoji: '✨', text: 'I needed to hear this today', author: 'grateful_heart' },
    { emoji: '✨', text: 'Saving this one for later', author: 'wisdom_collector' },
  ],
  Outro: [
    { emoji: '✨', text: 'See you tomorrow Kelly! 👋', author: 'daily_routine' },
    { emoji: '✨', text: 'Can\'t wait for the next one!', author: 'eager_learner' },
  ],
};

// Default Cliff content (archetype-neutral)
const DEFAULT_CLIFF_CONTENT = {
  script: "Now here's an interesting choice. Which path feels right to you today?",
  kellyPose: 'curious',
  kellyEmotion: 'warm',
  options: [
    {
      id: 'A',
      label: 'Deep dive',
      imageUrl: '',
      responseScript: 'Great choice! Let\'s explore this in depth together.',
      quality: 'best'
    },
    {
      id: 'B',
      label: 'Quick overview',
      imageUrl: '',
      responseScript: 'Perfect! Let me give you the key insights.',
      quality: 'good'
    }
  ],
  simulatedComments: COMMENT_TEMPLATES.Cliff.slice(0, 3)
};

// Default Outro content
const DEFAULT_OUTRO_CONTENT = {
  script: "And that's today's journey! You've learned something valuable. Take a moment to let it sink in, and I'll see you tomorrow for another adventure.",
  kellyPose: 'celebrating',
  kellyEmotion: 'warm',
  options: [
    {
      id: 'A',
      label: 'Preview tomorrow',
      imageUrl: '',
      responseScript: 'Here\'s a sneak peek at what we\'ll explore next!',
      quality: 'best'
    },
    {
      id: 'B',
      label: 'Reflect & save',
      imageUrl: '',
      responseScript: 'Take a moment to reflect on what you\'ve learned.',
      quality: 'good'
    }
  ],
  simulatedComments: COMMENT_TEMPLATES.Outro.slice(0, 2)
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function log(emoji: string, msg: string): void {
  console.log(`${emoji} ${msg}`);
}

function pickComments(phase: string, count: number = 3): Array<{emoji: string, text: string, author: string, timestamp: string}> {
  const templates = COMMENT_TEMPLATES[phase] || COMMENT_TEMPLATES.Hook;
  const selected = templates.slice(0, count);
  return selected.map((c, i) => ({
    ...c,
    timestamp: `${i + 1}m ago`
  }));
}

// =============================================================================
// FIX FUNCTIONS
// =============================================================================

async function fixOptionsTo2(atom: any): Promise<any> {
  const content = atom.content;
  if (!content.options || content.options.length === 0) {
    // No options - create default 2
    return {
      ...content,
      options: [
        { id: 'A', label: 'Option A', imageUrl: '', responseScript: 'Great choice!', quality: 'best' },
        { id: 'B', label: 'Option B', imageUrl: '', responseScript: 'Interesting perspective!', quality: 'good' }
      ]
    };
  }
  
  if (content.options.length === 2) {
    return content; // Already correct
  }
  
  if (content.options.length > 2) {
    // Keep only first 2 options
    return {
      ...content,
      options: content.options.slice(0, 2).map((opt: any, i: number) => ({
        ...opt,
        id: i === 0 ? 'A' : 'B',
        quality: i === 0 ? 'best' : 'good'
      }))
    };
  }
  
  return content;
}

function addSimulatedComments(content: any, phase: string): any {
  if (content.simulatedComments && content.simulatedComments.length >= 2) {
    return content; // Already has comments
  }
  
  const commentCount = phase === 'Outro' ? 2 : 3;
  return {
    ...content,
    simulatedComments: pickComments(phase, commentCount)
  };
}

// =============================================================================
// MAIN
// =============================================================================

async function fixDay(dayNumber: number, dryRun: boolean = false): Promise<void> {
  log('📅', `Processing Day ${dayNumber}...`);
  
  // Get core lesson
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', dayNumber)
    .single();
  
  if (lessonError || !lesson) {
    log('❌', `Lesson not found for day ${dayNumber}`);
    return;
  }
  
  log('📚', `Topic: ${lesson.topic}`);
  
  // Get all atoms for this lesson
  const { data: atoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id);
  
  if (atomsError || !atoms) {
    log('❌', `Failed to fetch atoms: ${atomsError?.message}`);
    return;
  }
  
  log('📊', `Found ${atoms.length} atoms`);
  
  // Track which phases exist per archetype
  const existingPhases = new Map<string, Set<string>>();
  for (const atom of atoms) {
    if (!existingPhases.has(atom.archetype)) {
      existingPhases.set(atom.archetype, new Set());
    }
    existingPhases.get(atom.archetype)!.add(atom.phase);
  }
  
  // 1. Fix existing atoms (options to 2, add comments)
  let updatedCount = 0;
  for (const atom of atoms) {
    let content = atom.content;
    let needsUpdate = false;
    
    // Fix options
    const optionCount = content.options?.length || 0;
    if (optionCount !== 2) {
      content = await fixOptionsTo2(atom);
      needsUpdate = true;
    }
    
    // Add comments
    if (!content.simulatedComments || content.simulatedComments.length < 2) {
      content = addSimulatedComments(content, atom.phase);
      needsUpdate = true;
    }
    
    if (needsUpdate) {
      if (dryRun) {
        log('🔍', `Would update ${atom.archetype} ${atom.phase}: options=${optionCount}→2`);
      } else {
        const { error } = await supabase
          .from('lesson_atoms')
          .update({ content })
          .eq('id', atom.id);
        
        if (error) {
          log('❌', `Failed to update ${atom.archetype} ${atom.phase}: ${error.message}`);
        } else {
          updatedCount++;
        }
      }
    }
  }
  
  log('✅', `Updated ${updatedCount} atoms`);
  
  // 2. Create missing Cliff and Outro phases
  let createdCount = 0;
  for (const archetype of ARCHETYPES) {
    const phases = existingPhases.get(archetype) || new Set();
    
    // Check for missing Cliff
    if (!phases.has('Cliff')) {
      if (dryRun) {
        log('🔍', `Would create Cliff for ${archetype}`);
      } else {
        const { error } = await supabase
          .from('lesson_atoms')
          .insert({
            core_lesson_id: lesson.id,
            archetype,
            phase: 'Cliff',
            content: DEFAULT_CLIFF_CONTENT
          });
        
        if (error) {
          log('❌', `Failed to create Cliff for ${archetype}: ${error.message}`);
        } else {
          createdCount++;
        }
      }
    }
    
    // Check for missing Outro
    if (!phases.has('Outro')) {
      if (dryRun) {
        log('🔍', `Would create Outro for ${archetype}`);
      } else {
        const { error } = await supabase
          .from('lesson_atoms')
          .insert({
            core_lesson_id: lesson.id,
            archetype,
            phase: 'Outro',
            content: DEFAULT_OUTRO_CONTENT
          });
        
        if (error) {
          log('❌', `Failed to create Outro for ${archetype}: ${error.message}`);
        } else {
          createdCount++;
        }
      }
    }
  }
  
  log('✅', `Created ${createdCount} missing atoms`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const dayArg = args.find(a => a.startsWith('--day='));
  
  const daysToProcess = dayArg 
    ? [parseInt(dayArg.split('=')[1])]
    : DAYS_TO_FIX;
  
  log('🚀', `Starting atom fix for days: ${daysToProcess.join(', ')}`);
  if (dryRun) log('⚠️', 'DRY RUN - no changes will be made');
  
  for (const day of daysToProcess) {
    await fixDay(day, dryRun);
  }
  
  log('🎉', 'Done!');
}

main().catch(console.error);
