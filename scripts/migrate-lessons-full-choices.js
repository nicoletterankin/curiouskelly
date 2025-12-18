#!/usr/bin/env node
/**
 * Migrate Lesson JSON Files to v5.0-full-choices Schema
 * 
 * This script:
 * 1. Adds prompt + options to ALL phases (hook, cliff, q1, q2, q3, wisdom, outro)
 * 2. Renames fact1/fact2/fact3 to q1/q2/q3
 * 3. Updates version to v5.0-full-choices
 * 
 * Run: node scripts/migrate-lessons-full-choices.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LESSONS_DIR = path.join(__dirname, '..', 'public', 'lessons');

// Templates for generating contextual choices based on phase type
function generateHookChoices(topic, script) {
  return {
    prompt: "What brings you here today?",
    options: [
      {
        text: "I want to learn something new",
        letter: "A",
        quality: "best",
        response: "Perfect. Curiosity is the first step to understanding. Let's explore this together."
      },
      {
        text: "I'm curious about this topic",
        letter: "B",
        quality: "best",
        response: "That curiosity is exactly what we need. Let's dive in and discover something amazing."
      }
    ]
  };
}

function generateQ1Choices(topic, title, script) {
  return {
    prompt: `What's your first impression of "${title}"?`,
    options: [
      {
        text: "This confirms what I already knew",
        letter: "A",
        quality: "good",
        response: "Great foundation! Now let's build on that knowledge with something you might not expect..."
      },
      {
        text: "This is new to me",
        letter: "B",
        quality: "best",
        response: "Perfect starting point! This insight will connect to even more surprising facts ahead."
      }
    ]
  };
}

function generateQ2Choices(topic, title, script) {
  return {
    prompt: `How does this connect to what you already know?`,
    options: [
      {
        text: "I see the connection now",
        letter: "A",
        quality: "best",
        response: "Exactly. These connections are how knowledge becomes understanding."
      },
      {
        text: "I need to think about this more",
        letter: "B",
        quality: "good",
        response: "That's a good instinct. Deep understanding takes time. The next fact will help clarify."
      }
    ]
  };
}

function generateQ3Choices(topic, title, script) {
  return {
    prompt: `What surprises you most about this?`,
    options: [
      {
        text: "I never thought about it this way",
        letter: "A",
        quality: "best",
        response: "That shift in perspective is exactly the point. New angles reveal new truths."
      },
      {
        text: "This changes my understanding",
        letter: "B",
        quality: "best",
        response: "That's learning in action. When understanding shifts, you're really getting it."
      }
    ]
  };
}

function generateWisdomChoices(topic, script) {
  return {
    prompt: "What resonates most with you today?",
    options: [
      {
        text: "The main insight",
        letter: "A",
        quality: "best",
        response: "That central truth will stay with you. Let it guide your thinking."
      },
      {
        text: "How I can apply this",
        letter: "B",
        quality: "best",
        response: "Application is where knowledge becomes wisdom. That's the real learning."
      }
    ]
  };
}

function generateOutroChoices(topic, script) {
  return {
    prompt: "Before you go—what will you take from today?",
    options: [
      {
        text: "I'll think about this more",
        letter: "A",
        quality: "best",
        response: "Reflection deepens learning. Let these ideas simmer in your mind."
      },
      {
        text: "I'll share this with someone",
        letter: "B",
        quality: "best",
        response: "Teaching others is the best way to learn. You've already begun the ripple."
      }
    ]
  };
}

function migrateLesson(lesson) {
  const topic = lesson.meta?.topic || 'this topic';
  const phases = lesson.phases || {};
  
  // Create new phases object with renamed keys and added choices
  const newPhases = {};
  
  // HOOK
  if (phases.hook) {
    newPhases.hook = {
      title: phases.hook.title || "Welcome",
      ...phases.hook,
      ...(phases.hook.prompt ? {} : generateHookChoices(topic, phases.hook.script))
    };
  }
  
  // CLIFF (usually already has options)
  if (phases.cliff) {
    newPhases.cliff = {
      title: phases.cliff.title || "The Question",
      ...phases.cliff
    };
    // Ensure it has options even if empty
    if (!newPhases.cliff.options || newPhases.cliff.options.length === 0) {
      const cliffChoices = {
        prompt: phases.cliff.prompt || "What do you think?",
        options: [
          {
            text: "Option A",
            letter: "A",
            quality: "good",
            response: "Interesting perspective. Let's explore further..."
          },
          {
            text: "Option B",
            letter: "B",
            quality: "best",
            response: "Great insight. This leads us to..."
          }
        ]
      };
      newPhases.cliff = { ...newPhases.cliff, ...cliffChoices };
    }
  }
  
  // Q1 (was fact1)
  const q1Source = phases.q1 || phases.fact1;
  if (q1Source) {
    newPhases.q1 = {
      title: q1Source.title || "Key Insight 1",
      script: q1Source.script,
      duration: q1Source.duration,
      ...(q1Source.prompt ? { prompt: q1Source.prompt, options: q1Source.options } : generateQ1Choices(topic, q1Source.title || "this fact", q1Source.script))
    };
  }
  
  // Q2 (was fact2)
  const q2Source = phases.q2 || phases.fact2;
  if (q2Source) {
    newPhases.q2 = {
      title: q2Source.title || "Key Insight 2",
      script: q2Source.script,
      duration: q2Source.duration,
      ...(q2Source.prompt ? { prompt: q2Source.prompt, options: q2Source.options } : generateQ2Choices(topic, q2Source.title || "this connection", q2Source.script))
    };
  }
  
  // Q3 (was fact3)
  const q3Source = phases.q3 || phases.fact3;
  if (q3Source) {
    newPhases.q3 = {
      title: q3Source.title || "Key Insight 3",
      script: q3Source.script,
      duration: q3Source.duration,
      ...(q3Source.prompt ? { prompt: q3Source.prompt, options: q3Source.options } : generateQ3Choices(topic, q3Source.title || "the surprise", q3Source.script))
    };
  }
  
  // WISDOM
  if (phases.wisdom) {
    newPhases.wisdom = {
      title: phases.wisdom.title || "Today's Wisdom",
      ...phases.wisdom,
      ...(phases.wisdom.prompt ? {} : generateWisdomChoices(topic, phases.wisdom.script))
    };
  }
  
  // OUTRO
  if (phases.outro) {
    newPhases.outro = {
      title: phases.outro.title || "See You Tomorrow",
      ...phases.outro,
      ...(phases.outro.prompt ? {} : generateOutroChoices(topic, phases.outro.script))
    };
  }
  
  // Update phaseOrder to use q1/q2/q3
  const newPhaseOrder = (lesson.phaseOrder || []).map(phase => {
    if (phase === 'fact1') return 'q1';
    if (phase === 'fact2') return 'q2';
    if (phase === 'fact3') return 'q3';
    return phase;
  });
  
  // Return migrated lesson
  return {
    ...lesson,
    meta: {
      ...lesson.meta,
      version: "v5.0-full-choices"
    },
    phases: newPhases,
    phaseOrder: newPhaseOrder
  };
}

async function main() {
  console.log('🚀 Starting lesson migration to v5.0-full-choices schema...\n');
  
  const files = fs.readdirSync(LESSONS_DIR).filter(f => f.endsWith('.json'));
  console.log(`📂 Found ${files.length} lesson files\n`);
  
  let migrated = 0;
  let skipped = 0;
  let errors = 0;
  
  for (const file of files) {
    const filePath = path.join(LESSONS_DIR, file);
    
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const lesson = JSON.parse(content);
      
      // Skip if already v5.0
      if (lesson.meta?.version === 'v5.0-full-choices') {
        console.log(`⏭️  ${file} - already v5.0, skipping`);
        skipped++;
        continue;
      }
      
      // Migrate
      const migratedLesson = migrateLesson(lesson);
      
      // Write back
      fs.writeFileSync(filePath, JSON.stringify(migratedLesson, null, 2) + '\n');
      console.log(`✅ ${file} - migrated to v5.0`);
      migrated++;
      
    } catch (err) {
      console.error(`❌ ${file} - ERROR: ${err.message}`);
      errors++;
    }
  }
  
  console.log('\n' + '='.repeat(50));
  console.log(`📊 Migration Complete`);
  console.log(`   ✅ Migrated: ${migrated}`);
  console.log(`   ⏭️  Skipped: ${skipped}`);
  console.log(`   ❌ Errors: ${errors}`);
  console.log('='.repeat(50));
}

main();
