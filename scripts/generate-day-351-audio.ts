#!/usr/bin/env npx tsx
/**
 * DAY 351 AUDIO GENERATOR - Full Phase Structure
 * 
 * Generates ALL 35 audio files for Day 351 using ElevenLabs.
 * 
 * Structure per phase:
 * - talk.mp3       (main teaching content)
 * - question.mp3   (Kelly asks learner)
 * - response_a.mp3 (feedback for option A)
 * - response_b.mp3 (feedback for option B)
 * - comment.mp3    (simulated student comment)
 * 
 * Usage:
 *   npx tsx scripts/generate-day-351-audio.ts
 *   npx tsx scripts/generate-day-351-audio.ts --dry-run
 *   npx tsx scripts/generate-day-351-audio.ts --phase=hook
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

// ═══════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const KELLY_VOICE_ID = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';

// For student comments, we might want a different voice
// For now, use Kelly for everything
const STUDENT_VOICE_ID = KELLY_VOICE_ID; // TODO: Consider different voice for students

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'audio', '351');
const DAY_351_DATA_PATH = path.join(process.cwd(), 'public', 'data', 'day-351-complete.js');

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function getArg(name: string): string | undefined {
  const arg = process.argv.find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : undefined;
}

function hasFlag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════
// ELEVENLABS TTS
// ═══════════════════════════════════════════════════════════════════

interface VoiceSettings {
  stability: number;
  similarity_boost: number;
  style?: number;
}

async function generateAudio(
  text: string, 
  voiceId: string = KELLY_VOICE_ID,
  settings: VoiceSettings = { stability: 0.5, similarity_boost: 0.75 }
): Promise<Buffer> {
  if (!ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY not set');
  }

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}?output_format=mp3_44100_192`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': ELEVENLABS_API_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: settings,
      }),
    }
  );

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`ElevenLabs error: ${response.status} - ${error.slice(0, 200)}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

// ═══════════════════════════════════════════════════════════════════
// DAY 351 CONTENT
// ═══════════════════════════════════════════════════════════════════

interface PhaseContent {
  talk: { script: string };
  question: { prompt: string };
  responses: {
    A: { script: string };
    B: { script: string };
  };
  studentComment: { text: string; name: string };
}

// Hard-coded from day-351-complete.js for reliability
const DAY_351_PHASES: Record<string, PhaseContent> = {
  hook: {
    talk: {
      script: "Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle. It's called visualization, and the science behind it might change how you think about learning itself."
    },
    question: {
      prompt: "Before we dive in—have you ever imagined doing something before you actually did it?"
    },
    responses: {
      A: { script: "You're already tapping into something powerful. Today you'll learn exactly why that works—and how to do it even better." },
      B: { script: "That's totally normal! Most people don't realize what they're missing. By the end of today, you might change your approach." }
    },
    studentComment: {
      name: "Jordan",
      text: "I always picture my presentations before giving them. Didn't know there was science behind it!"
    }
  },
  cliff: {
    talk: {
      script: "Here's where it gets interesting. When you vividly imagine doing something—really see it, feel it, experience it in your mind—your brain activates almost the same way as when you actually do it. The neurons fire. The pathways light up. But here's the question that puzzled scientists for years..."
    },
    question: {
      prompt: "Why would imagining something make you better at actually doing it?"
    },
    responses: {
      A: { script: "That's what researchers thought at first too! Confidence does play a role. But brain scans revealed something far more concrete happening inside the skull." },
      B: { script: "Exactly right. And not in some vague, mystical way—we're talking measurable, physical changes in neural structure. Let me show you the evidence." }
    },
    studentComment: {
      name: "Maya",
      text: "Wait, so daydreaming might actually be... productive?"
    }
  },
  fact1: {
    talk: {
      script: "When you imagine performing an action, your motor cortex—that's the part of your brain that controls movement—lights up almost identically to when you actually move. Brain scans show about 90% overlap. Ninety percent. Your brain literally cannot tell the difference between vividly imagining something and doing it. It's practicing either way."
    },
    question: {
      prompt: "What do you think this means for learning new skills?"
    },
    responses: {
      A: { script: "You've got it. On the bus, in bed, waiting in line—your brain doesn't care where your body is. It's ready to train." },
      B: { script: "Real practice is important, absolutely. But here's the thing—the best performers don't choose one or the other. They combine both. And the results are remarkable." }
    },
    studentComment: {
      name: "Alex",
      text: "90%?! That's insane. My brain's been lying to me this whole time."
    }
  },
  fact2: {
    talk: {
      script: "Let me tell you about a famous experiment. Researchers took people who had never played piano and divided them into three groups. Group one physically practiced a simple piece for five days. Group two only imagined practicing—same piece, same time, but never touched a key. Group three did nothing. After five days, they scanned everyone's brains. The results shocked the scientific community."
    },
    question: {
      prompt: "What do you think they found when comparing the imagination group to the physical practice group?"
    },
    responses: {
      A: { script: "That's the logical guess. But here's the twist—the imagination group's brains showed nearly identical changes to the physical practice group. Mental rehearsal created real, measurable neuroplastic changes." },
      B: { script: "Exactly. The brain regions responsible for piano playing grew in both groups. Imagination alone rewired their brains. Not as much as physical practice, but remarkably close." }
    },
    studentComment: {
      name: "Sam",
      text: "So I can tell my parents I'm practicing piano in my head?"
    }
  },
  fact3: {
    talk: {
      script: "This isn't just lab science. Elite performers have known this for decades. Olympic athletes spend up to 50% of their training time on mental rehearsal. Surgeons visualize entire procedures before making a single cut. Concert pianists play through pieces in their minds on the flight to performances. The key they all discovered: specificity. Vague daydreaming doesn't work. You need vivid, detailed, multi-sensory imagination."
    },
    question: {
      prompt: "What makes visualization most effective, based on what the pros do?"
    },
    responses: {
      A: { script: "Positive outcomes matter for motivation, but here's the secret the pros know: you have to visualize the process, not just the result. Feel the movements. See the environment. Hear the sounds. That's what triggers the neural overlap." },
      B: { script: "That's the key. The more senses you engage, the more your brain treats it as real practice. See it, feel it, hear it. First-person perspective. Every detail matters." }
    },
    studentComment: {
      name: "Riley",
      text: "50% of Olympic training is just... thinking? Mind equals blown."
    }
  },
  wisdom: {
    talk: {
      script: "Here's today's wisdom: Your imagination is a practice field. The mind that rehearses builds pathways the passive mind never develops. Every time you vividly imagine doing something, you're laying down the neural tracks that make it easier to do for real. This is one of the few truly free performance enhancers available to every human being."
    },
    question: {
      prompt: "What's one skill you'd like to practice in your mind this week?"
    },
    responses: {
      A: { script: "Perfect choice. Physical skills respond incredibly well to visualization. Tonight, before sleep, spend five minutes seeing yourself perform it perfectly. Feel every motion. You'll be surprised what happens." },
      B: { script: "Excellent. Visualization works for mental skills too—public speaking, difficult conversations, high-pressure decisions. Run through the scenario. See yourself handling it with grace. Your brain will be more prepared when it's real." }
    },
    studentComment: {
      name: "Taylor",
      text: "I'm going to try this before my job interview next week!"
    }
  },
  outro: {
    talk: {
      script: "That's today's lesson. Your brain is more trainable than you ever imagined—literally. Visualization isn't wishful thinking. It's cognitive rehearsal that primes your brain for performance. Tonight, give it a try. Close your eyes. Pick something you want to master. And practice it in the one gym that's always open—your mind."
    },
    question: {
      prompt: "Will you try visualization practice tonight?"
    },
    responses: {
      A: { script: "Love that energy! Remember: specific, vivid, multi-sensory. See you tomorrow with something new. Keep visualizing great things." },
      B: { script: "Take your time choosing. The right skill will come to you. When you're ready, your brain will be too. See you tomorrow!" }
    },
    studentComment: {
      name: "Casey",
      text: "Best 3 minutes I've spent today. Thanks, Kelly!"
    }
  }
};

// ═══════════════════════════════════════════════════════════════════
// MAIN GENERATION LOGIC
// ═══════════════════════════════════════════════════════════════════

interface AudioFile {
  filename: string;
  text: string;
  type: 'talk' | 'question' | 'response_a' | 'response_b' | 'comment';
  phase: string;
  voiceId: string;
}

function buildAudioList(): AudioFile[] {
  const files: AudioFile[] = [];
  
  for (const [phaseName, phase] of Object.entries(DAY_351_PHASES)) {
    // Main talk
    files.push({
      filename: `${phaseName}_talk.mp3`,
      text: phase.talk.script,
      type: 'talk',
      phase: phaseName,
      voiceId: KELLY_VOICE_ID
    });
    
    // Question
    files.push({
      filename: `${phaseName}_question.mp3`,
      text: phase.question.prompt,
      type: 'question',
      phase: phaseName,
      voiceId: KELLY_VOICE_ID
    });
    
    // Response A
    files.push({
      filename: `${phaseName}_response_a.mp3`,
      text: phase.responses.A.script,
      type: 'response_a',
      phase: phaseName,
      voiceId: KELLY_VOICE_ID
    });
    
    // Response B
    files.push({
      filename: `${phaseName}_response_b.mp3`,
      text: phase.responses.B.script,
      type: 'response_b',
      phase: phaseName,
      voiceId: KELLY_VOICE_ID
    });
    
    // Student comment (could use different voice in future)
    files.push({
      filename: `${phaseName}_comment.mp3`,
      text: phase.studentComment.text,
      type: 'comment',
      phase: phaseName,
      voiceId: STUDENT_VOICE_ID
    });
  }
  
  return files;
}

async function main() {
  const dryRun = hasFlag('dry-run');
  const phaseFilter = getArg('phase');
  
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎤 DAY 351 AUDIO GENERATOR                                    ║');
  console.log('║  Generating 35 audio files via ElevenLabs                      ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  
  if (!ELEVENLABS_API_KEY) {
    console.error('❌ ELEVENLABS_API_KEY not found in environment');
    process.exit(1);
  }
  
  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    console.log(`📁 Created output directory: ${OUTPUT_DIR}`);
  }
  
  // Build list of audio files to generate
  let audioFiles = buildAudioList();
  
  // Filter by phase if specified
  if (phaseFilter) {
    audioFiles = audioFiles.filter(f => f.phase === phaseFilter);
    console.log(`🔍 Filtering to phase: ${phaseFilter}`);
  }
  
  console.log(`📝 Files to generate: ${audioFiles.length}`);
  console.log('');
  
  if (dryRun) {
    console.log('🔍 DRY RUN MODE - No files will be generated');
    console.log('');
    for (const file of audioFiles) {
      console.log(`  ${file.filename}`);
      console.log(`    Type: ${file.type}`);
      console.log(`    Text: ${file.text.slice(0, 60)}...`);
      console.log('');
    }
    return;
  }
  
  // Generate each audio file
  let success = 0;
  let failed = 0;
  
  for (let i = 0; i < audioFiles.length; i++) {
    const file = audioFiles[i];
    const progress = `[${i + 1}/${audioFiles.length}]`;
    
    console.log(`${progress} Generating ${file.filename}...`);
    
    try {
      const buffer = await generateAudio(file.text, file.voiceId);
      const outputPath = path.join(OUTPUT_DIR, file.filename);
      fs.writeFileSync(outputPath, buffer);
      
      const sizeMB = (buffer.length / 1024 / 1024).toFixed(2);
      console.log(`  ✅ Saved (${sizeMB} MB)`);
      success++;
      
      // Rate limiting - ElevenLabs has limits
      await sleep(500);
      
    } catch (error) {
      console.error(`  ❌ Failed: ${error}`);
      failed++;
    }
  }
  
  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`✅ Success: ${success}/${audioFiles.length}`);
  if (failed > 0) {
    console.log(`❌ Failed: ${failed}`);
  }
  console.log(`📁 Output: ${OUTPUT_DIR}`);
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(console.error);
