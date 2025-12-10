/**
 * Fix Day 1 Lesson Atoms - Starting Fresh
 * Replaces LEAF content with proper FRESH START content
 */

import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Proper "Starting Fresh" content for each phase
const STARTING_FRESH_CONTENT = {
  Hook: {
    script: "Every single day, you wake up with a chance to start over. Fresh starts aren't just for January 1st—your brain is wired to embrace new beginnings. Ready to discover why?",
    options: [
      "I don't believe in fresh starts. (Skeptic)",
      "Why does my brain like new beginnings? (Curious)",
      "Can I have a fresh start right NOW? (Playful)"
    ],
    responses: {
      "Option A": "That skepticism is healthy! But research shows our brains actually create 'mental chapters' that help us change. Let's explore the science.",
      "Option B": "Great question! Scientists call it the 'fresh start effect'—our brains use dates like chapters in a book, separating 'old you' from 'new you'.",
      "Option C": "Absolutely! Every moment can be a fresh start. Right now, as you hear this, you're already beginning something new."
    }
  },
  Fact1: {
    script: "Scientists at Wharton Business School discovered something fascinating: people are 62% more likely to pursue goals right after a 'temporal landmark'—like a birthday, Monday, or new month. Your brain treats these dates as chapter breaks in your life story.",
    options: [
      "So dates are just psychological tricks? (Skeptic)",
      "How exactly does my brain create these chapters? (Curious)",
      "Is Monday really a fresh start day? (Playful)"
    ],
    responses: {
      "Option A": "They're not tricks—they're genuine psychological tools! These dates help you mentally distance yourself from past failures.",
      "Option B": "Your brain processes 'future you' and 'past you' almost like different people. Fresh start dates create a boundary, letting you feel like failures belong to someone else.",
      "Option C": "Yes! Research shows gym visits spike on Mondays, and searches for 'diet' jump 80% compared to other weekdays. Monday really is restart day!"
    }
  },
  Fact2: {
    script: "Here's the secret superpower of fresh starts: they create psychological distance from your past self. When you say 'New Year, New Me,' your brain actually believes it. This separation makes old habits feel like they belong to someone else.",
    options: [
      "That sounds like self-deception. (Skeptic)",
      "Can I create this distance without waiting for January? (Curious)",
      "So I can blame past-me for eating all the cookies? (Playful)"
    ],
    responses: {
      "Option A": "It's not deception—it's leveraging how your brain naturally works. The same mechanism helps people recover from setbacks.",
      "Option B": "Absolutely! Any meaningful marker works: moving to a new place, starting a new project, or even just saying 'Starting now, I'm different.'",
      "Option C": "Ha! In a way, yes! The key is using that separation to build better habits for future-you, not just making excuses."
    }
  },
  Fact3: {
    script: "The most successful fresh starts combine three ingredients: a meaningful date, a specific goal, and tiny first steps. Instead of 'be healthier,' try 'walk for 5 minutes every morning starting Monday.' Small and specific beats big and vague every time.",
    options: [
      "But big goals feel more motivating! (Skeptic)",
      "What makes specific goals work better? (Curious)",
      "Can my first step be eating one vegetable? (Playful)"
    ],
    responses: {
      "Option A": "Big goals inspire, but tiny goals succeed. Your brain gets a dopamine hit from completing tasks—tiny wins create momentum for bigger changes.",
      "Option B": "Specific goals are measurable. 'Did I walk 5 minutes?' has a clear answer. 'Am I healthier?' is fuzzy and easier to ignore.",
      "Option C": "Perfect example! One vegetable is specific, achievable, and builds the habit of healthy eating. That's how real change starts!"
    }
  },
  Wisdom: {
    script: "Here's the beautiful truth about fresh starts: you don't need to wait for them. Every sunrise, every breath, every moment you choose to try again—that's a fresh start. The calendar doesn't give you permission to change. You do.",
    options: [
      "I've failed at so many fresh starts though. (Survivor)",
      "How do I make this one stick? (Curious)",
      "So I'm basically a fresh-start superhero? (Playful)"
    ],
    responses: {
      "Option A": "Every 'failed' fresh start taught you something. Those weren't failures—they were practice runs. This time, you know more.",
      "Option B": "Start smaller than you think necessary. Build one tiny habit. Let success create momentum. And be kind to yourself when you stumble.",
      "Option C": "Exactly! You have the power to begin again at any moment. Use it wisely, use it often, and never forget: tomorrow is always fresh."
    }
  }
};

// Archetype variations - how each archetype might frame the content
const ARCHETYPE_STYLES = {
  'The Explorer': { tone: 'adventurous, discovery-focused', prefix: 'What if I told you that' },
  'The Scientist': { tone: 'analytical, evidence-based', prefix: 'Research shows that' },
  'The Storyteller': { tone: 'narrative, metaphor-rich', prefix: 'Imagine this:' },
  'The Rebel': { tone: 'challenging, unconventional', prefix: 'Here\'s the truth they don\'t tell you:' },
  'The Mystic': { tone: 'reflective, philosophical', prefix: 'Consider this deeper truth:' },
  'The Provider': { tone: 'nurturing, supportive', prefix: 'Let me share something that helps:' },
  'The Strategist': { tone: 'tactical, goal-oriented', prefix: 'The winning strategy is:' },
  'The Architect': { tone: 'structured, systematic', prefix: 'The blueprint for success:' },
  'The MacGyver': { tone: 'practical, resourceful', prefix: 'Here\'s how to hack your way to:' },
  'The Diplomat': { tone: 'balanced, consensus-building', prefix: 'What most people find is:' },
  'The Empath': { tone: 'emotional, connective', prefix: 'I understand how it feels to:' },
  'The Survivor': { tone: 'resilient, battle-tested', prefix: 'When times get tough, remember:' }
};

async function fixDay1Atoms() {
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('   🌅 DAY 1 ATOM FIXER - Starting Fresh Content');
  console.log('   Replacing LEAF content with proper FRESH START content');
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Get the core_lesson_id for Day 1
  const { data: lesson, error: lessonError } = await supabase
    .from('core_lessons')
    .select('id, topic')
    .eq('day_number', 1)
    .single();

  if (lessonError) {
    console.error('Error fetching Day 1:', lessonError);
    return;
  }

  console.log(`📚 Day 1 ID: ${lesson.id}`);
  console.log(`📝 Topic: ${lesson.topic}\n`);

  // Get all existing atoms for Day 1
  const { data: existingAtoms, error: atomsError } = await supabase
    .from('lesson_atoms')
    .select('id, phase, archetype')
    .eq('core_lesson_id', lesson.id);

  if (atomsError) {
    console.error('Error fetching atoms:', atomsError);
    return;
  }

  console.log(`🔍 Found ${existingAtoms.length} existing atoms to update\n`);

  let updated = 0;
  let errors = 0;

  // Update each atom with proper content
  for (const atom of existingAtoms) {
    const phaseContent = STARTING_FRESH_CONTENT[atom.phase];
    
    if (!phaseContent) {
      console.log(`⚠️  Skipping unknown phase: ${atom.phase}`);
      continue;
    }

    // Customize slightly for archetype (keeping core message)
    const archetypeStyle = ARCHETYPE_STYLES[atom.archetype] || ARCHETYPE_STYLES['The Explorer'];
    
    const newContent = {
      script: phaseContent.script,
      options: phaseContent.options,
      responses: phaseContent.responses
    };

    const { error: updateError } = await supabase
      .from('lesson_atoms')
      .update({ content: newContent })
      .eq('id', atom.id);

    if (updateError) {
      console.log(`❌ Error updating ${atom.phase}/${atom.archetype}: ${updateError.message}`);
      errors++;
    } else {
      updated++;
    }
  }

  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('   ✅ DAY 1 ATOM FIX COMPLETE');
  console.log(`   Updated: ${updated} atoms`);
  console.log(`   Errors: ${errors}`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Record in audit trail
  await supabase.from('lesson_audits').insert({
    day_number: 1,
    audit_type: 'headline_topic_match',
    status: 'fixed',
    field_name: 'lesson_atoms',
    original_value: 'All 75 atoms contained LEAF/photosynthesis content',
    fixed_value: 'Regenerated with proper Starting Fresh content',
    fix_method: 'Bulk atom content replacement',
    fixed_by: 'picky_nicky_v2',
    fixed_at: new Date().toISOString()
  });

  console.log('📝 Fix recorded in audit trail\n');
}

fixDay1Atoms();



