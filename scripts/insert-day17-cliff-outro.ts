#!/usr/bin/env npx tsx
/**
 * INSERT DAY 17 CLIFF AND OUTRO ATOMS
 * Bypasses RLS by using service role key properly
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.PUBLIC_SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
  },
});

const DAY_17_LESSON_ID = 'f1702ef6-c322-4fd9-b49f-1991bace4b99';

// All archetypes with Cliff/Outro content
const ATOMS = [
  // The Scientist
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Scientist',
    phase: 'Cliff',
    content: {
      script: "Now here's where it gets interesting - you have a choice. Would you like to explore this through data and evidence, or through experimentation and discovery? Both paths lead to understanding, but the journey is different.",
      kellyPose: 'contemplative',
      kellyEmotion: 'thoughtful',
      cliffPrompt: 'How do you want to explore?',
      options: [
        { letter: 'A', text: 'Show me the data', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Let me experiment', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Scientist',
    phase: 'Outro',
    content: {
      script: "And there you have it! You've just added a new piece to your understanding. I love how you approached this with curiosity. Tomorrow, we'll build on what you've learned. Keep questioning everything!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Explorer
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Explorer',
    phase: 'Cliff',
    content: {
      script: "Alright, adventurer! Two paths stretch before you. Do you want to take the scenic route and discover hidden connections, or blaze straight to the destination? Either way leads to treasure.",
      kellyPose: 'contemplative',
      kellyEmotion: 'excited',
      cliffPrompt: 'Which adventure calls to you?',
      options: [
        { letter: 'A', text: 'Scenic discovery', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Direct path', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Explorer',
    phase: 'Outro',
    content: {
      script: "What a journey we've been on today! You've explored new territory and discovered something wonderful. Tomorrow brings new horizons. Rest up, explorer - more adventures await!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Rebel
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Rebel',
    phase: 'Cliff',
    content: {
      script: "Here's the thing - you don't have to do this the conventional way. You could challenge the accepted wisdom, or you could work within the system to change it. Both are valid forms of rebellion.",
      kellyPose: 'contemplative',
      kellyEmotion: 'confident',
      cliffPrompt: 'How will you shake things up?',
      options: [
        { letter: 'A', text: 'Challenge it', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Transform it', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Rebel',
    phase: 'Outro',
    content: {
      script: "Now that's what I call thinking for yourself! You didn't just accept what you were told - you made it your own. Keep questioning, keep pushing boundaries. See you tomorrow, rebel.",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Architect
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Architect',
    phase: 'Cliff',
    content: {
      script: "Let's think systematically here. You can approach this by building on existing frameworks, or by designing something entirely new. Both require careful thought and planning.",
      kellyPose: 'contemplative',
      kellyEmotion: 'thoughtful',
      cliffPrompt: 'How will you build your understanding?',
      options: [
        { letter: 'A', text: 'Build on frameworks', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Design fresh', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Architect',
    phase: 'Outro',
    content: {
      script: "Excellent work! You've constructed a solid foundation of understanding today. Each piece fits together logically. Tomorrow we'll add another floor to this structure. Well built!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Diplomat
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Diplomat',
    phase: 'Cliff',
    content: {
      script: "There are different perspectives to consider here. Would you like to understand how this affects different people, or focus on finding common ground between viewpoints?",
      kellyPose: 'contemplative',
      kellyEmotion: 'thoughtful',
      cliffPrompt: 'How will you approach this?',
      options: [
        { letter: 'A', text: 'Explore perspectives', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Find common ground', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Diplomat',
    phase: 'Outro',
    content: {
      script: "You've done something wonderful today - you've grown your understanding while staying open to different views. That's the heart of wisdom. See you tomorrow for more meaningful exploration!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Empath
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Empath',
    phase: 'Cliff',
    content: {
      script: "I can sense you're ready to go deeper. Would you like to explore how this feels on a personal level, or understand how it affects the people around us?",
      kellyPose: 'contemplative',
      kellyEmotion: 'caring',
      cliffPrompt: 'Where does your heart lead?',
      options: [
        { letter: 'A', text: 'Personal journey', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Understanding others', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Empath',
    phase: 'Outro',
    content: {
      script: "I'm so proud of the connection you've made with this material today. You didn't just learn it - you felt it. That's what makes learning truly meaningful. Rest well, and I'll see you tomorrow.",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The MacGyver
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The MacGyver',
    phase: 'Cliff',
    content: {
      script: "Okay, here's where it gets practical. Do you want to figure out how to apply this right away, or first understand all the pieces so you can improvise later?",
      kellyPose: 'contemplative',
      kellyEmotion: 'practical',
      cliffPrompt: "What's your move?",
      options: [
        { letter: 'A', text: 'Apply it now', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Gather the pieces', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The MacGyver',
    phase: 'Outro',
    content: {
      script: "Now you've got another tool in your toolkit! The best part is you know how to use it. Tomorrow we'll add more skills to your repertoire. Keep tinkering!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Mystic
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Mystic',
    phase: 'Cliff',
    content: {
      script: "There's more here than meets the eye. Would you like to contemplate the deeper meaning, or see how this connects to the larger patterns of existence?",
      kellyPose: 'contemplative',
      kellyEmotion: 'serene',
      cliffPrompt: 'Which wisdom calls to you?',
      options: [
        { letter: 'A', text: 'Contemplate deeply', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'See the patterns', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Mystic',
    phase: 'Outro',
    content: {
      script: "The wisdom you've gained today goes beyond facts - it touches something deeper. Carry this insight with you. When we meet again tomorrow, we'll explore even further.",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Provider
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Provider',
    phase: 'Cliff',
    content: {
      script: "Now let's think about how this helps people. Do you want to focus on how you can use this to help those close to you, or how it could benefit your community?",
      kellyPose: 'contemplative',
      kellyEmotion: 'caring',
      cliffPrompt: 'Who will you help?',
      options: [
        { letter: 'A', text: 'Help loved ones', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Serve community', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Provider',
    phase: 'Outro',
    content: {
      script: "You've learned something valuable today - something you can share and use to help others. That's the best kind of knowledge. See you tomorrow, and keep caring for those around you!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Storyteller
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Storyteller',
    phase: 'Cliff',
    content: {
      script: "Every good story has a turning point. Do you want to see how this tale unfolds through personal narratives, or explore the bigger story that connects us all?",
      kellyPose: 'contemplative',
      kellyEmotion: 'intrigued',
      cliffPrompt: 'Which story draws you in?',
      options: [
        { letter: 'A', text: 'Personal tales', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'The bigger story', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Storyteller',
    phase: 'Outro',
    content: {
      script: "And that's today's chapter complete! What a story we've woven together. Every great tale needs a hero who keeps showing up - that's you. See you tomorrow for the next chapter!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Strategist
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Strategist',
    phase: 'Cliff',
    content: {
      script: "Let's be strategic about this. Do you want to focus on the immediate advantages this knowledge provides, or think about the long-term implications and possibilities?",
      kellyPose: 'contemplative',
      kellyEmotion: 'analytical',
      cliffPrompt: "What's your strategy?",
      options: [
        { letter: 'A', text: 'Immediate gains', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Long-term vision', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Strategist',
    phase: 'Outro',
    content: {
      script: "Strategic move, completing today's lesson! You now have knowledge that others don't - use it wisely. Tomorrow we'll add to your competitive advantage. Well played!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
  // The Survivor
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Survivor',
    phase: 'Cliff',
    content: {
      script: "Here's what matters - this knowledge can help you. Do you want to focus on how it applies to challenges you're facing now, or build resilience for whatever comes next?",
      kellyPose: 'contemplative',
      kellyEmotion: 'determined',
      cliffPrompt: 'What serves you best?',
      options: [
        { letter: 'A', text: 'Face current challenges', quality: 'good', response: "Great choice! Let's explore this path together." },
        { letter: 'B', text: 'Build for tomorrow', quality: 'good', response: "Excellent decision! This approach will serve you well." },
      ],
    },
  },
  {
    core_lesson_id: DAY_17_LESSON_ID,
    archetype: 'The Survivor',
    phase: 'Outro',
    content: {
      script: "You've added another skill to your survival toolkit today. Every piece of knowledge makes you stronger and more prepared. Keep showing up - you've got this. See you tomorrow!",
      kellyPose: 'celebratory',
      kellyEmotion: 'warm',
    },
  },
];

async function main() {
  console.log('═'.repeat(60));
  console.log('📝 INSERTING DAY 17 CLIFF & OUTRO ATOMS');
  console.log('═'.repeat(60));
  console.log(`Total atoms to insert: ${ATOMS.length}`);
  console.log('');

  // Check if any already exist
  const { data: existing } = await supabase
    .from('lesson_atoms')
    .select('archetype, phase')
    .eq('core_lesson_id', DAY_17_LESSON_ID)
    .in('phase', ['Cliff', 'Outro']);

  if (existing && existing.length > 0) {
    console.log(`⚠️ Found ${existing.length} existing Cliff/Outro atoms`);
    console.log('Existing:', existing.map(e => `${e.archetype} ${e.phase}`).join(', '));
    console.log('');
  }

  // Filter out existing
  const existingSet = new Set(existing?.map(e => `${e.archetype}|${e.phase}`) || []);
  const toInsert = ATOMS.filter(a => !existingSet.has(`${a.archetype}|${a.phase}`));

  if (toInsert.length === 0) {
    console.log('✅ All Cliff/Outro atoms already exist!');
    return;
  }

  console.log(`📝 Inserting ${toInsert.length} new atoms...`);

  // Insert in batches of 10
  const batchSize = 10;
  let inserted = 0;

  for (let i = 0; i < toInsert.length; i += batchSize) {
    const batch = toInsert.slice(i, i + batchSize);
    
    const { data, error } = await supabase
      .from('lesson_atoms')
      .insert(batch)
      .select('id, archetype, phase');

    if (error) {
      console.error(`❌ Batch ${i / batchSize + 1} failed:`, error.message);
      console.error('   Details:', error.details);
      console.error('   Hint:', error.hint);
    } else {
      inserted += batch.length;
      batch.forEach(a => console.log(`   ✅ ${a.archetype} ${a.phase}`));
    }
  }

  console.log('');
  console.log('═'.repeat(60));
  console.log(`📊 RESULT: ${inserted}/${toInsert.length} atoms inserted`);
  console.log('═'.repeat(60));
}

main().catch(console.error);
