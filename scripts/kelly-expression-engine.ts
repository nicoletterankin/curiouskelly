#!/usr/bin/env npx tsx
/**
 * 🎭 KELLY EXPRESSION ENGINE
 * 
 * Maps lesson phases to Kelly's facial expressions and generates
 * HeyGen-compatible motion/emotion prompts.
 * 
 * "Kelly should direct with her eyes and smile - pointing with her lips."
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

// Load expression phases config
const configPath = path.join(__dirname, '../config/kelly-expression-phases.json');
const expressionConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));

export type Phase = 'hook' | 'cliff' | 'fact1' | 'fact2' | 'fact3' | 'wisdom' | 'outro';

export interface PhaseExpression {
  emotion: string;
  energy: number;
  smile: number;
  eyebrows: string;
  gaze: {
    direction: string;
    intensity: number;
    saccades: boolean;
  };
  movement: {
    head_tilt: string;
    nod_frequency: string;
  };
  voice_modulation: {
    pitch_variance: number;
    speed: number;
    enthusiasm: number;
  };
  description: string;
}

/**
 * Get expression config for a lesson phase
 */
export function getPhaseExpression(phase: Phase): PhaseExpression {
  return expressionConfig.phases[phase];
}

/**
 * Generate a HeyGen-compatible motion prompt based on phase
 */
export function generateMotionPrompt(phase: Phase): string {
  const expr = getPhaseExpression(phase);
  
  const prompts: Record<Phase, string> = {
    hook: 'curious and excited, eyes bright, slight forward lean, welcoming smile',
    cliff: 'intrigued and questioning, one eyebrow raised, knowing look, building tension',
    fact1: 'confident and clear, direct eye contact, gentle nodding, explaining warmly',
    fact2: 'building momentum, growing excitement, engaged expression, connecting ideas',
    fact3: 'revealing discovery, enthusiastic lean forward, emphatic gestures, application focus',
    wisdom: 'warm and knowing, slow meaningful nods, deep eye contact, sharing something precious',
    outro: 'celebrating and proud, big genuine smile, high energy, pure warmth'
  };
  
  return prompts[phase] || 'friendly and engaged';
}

/**
 * Get best archetype for a topic category
 */
export function getBestArchetypeForTopic(topicCategory: string): string {
  const topicArchetypeMap: Record<string, string> = {
    'science': 'Scientist',
    'nature': 'Explorer',
    'emotions': 'Empath',
    'creativity': 'Storyteller',
    'ethics': 'Consultant',
    'philosophy': 'Mystic',
    'practical': 'MacGyver',
    'strategy': 'Strategist',
    'family': 'Provider',
    'independence': 'Rebel',
    'resilience': 'Survivor',
    'planning': 'Architect',
    'default': 'Scientist' // Warm, curious, approachable
  };
  
  const category = topicCategory.toLowerCase();
  return topicArchetypeMap[category] || topicArchetypeMap.default;
}

/**
 * Generate micro-expression config for natural animation
 */
export function getMicroExpressionConfig() {
  return expressionConfig.micro_expressions;
}

/**
 * Get HeyGen video generation config for a phase
 */
export function getHeyGenConfig(phase: Phase, avatarId: string, audioUrl: string) {
  const expr = getPhaseExpression(phase);
  const motion = generateMotionPrompt(phase);
  
  return {
    video_inputs: [{
      character: {
        type: 'talking_photo',
        talking_photo_id: avatarId,
      },
      voice: {
        type: 'audio',
        audio_url: audioUrl,
      },
      // Motion/emotion hints (if HeyGen supports them)
      // Note: Some of these may need to be adjusted based on HeyGen API capabilities
    }],
    dimension: { width: 1920, height: 1080 },
    // Store expression metadata for post-processing
    _expressionMeta: {
      phase,
      motion,
      energy: expr.energy,
      smile: expr.smile,
      gaze_intensity: expr.gaze.intensity
    }
  };
}

/**
 * KELLY'S GAZE DIRECTION GUIDE
 * 
 * Eyes: Should feel like she's finding YOU in a crowded room
 * Smile: Genuine Duchenne smile (eyes + mouth) = trust
 * Lips: Subtle direction during emphasis - pointing to key concepts
 * 
 * The goal: "She sees ME. She's teaching ME."
 */
export const KELLY_PRESENCE_GUIDELINES = {
  eyes: {
    target: 'camera_center',
    variance: 0.02, // micro-saccade range
    intensity_by_phase: {
      hook: 0.9,
      cliff: 0.85,
      fact1: 0.8,
      fact2: 0.85,
      fact3: 0.9,
      wisdom: 0.95, // Maximum connection during wisdom
      outro: 0.95
    }
  },
  smile: {
    type: 'duchenne', // Eyes and mouth together
    intensity_by_phase: {
      hook: 0.6,
      cliff: 0.3, // Less smile, more intrigue
      fact1: 0.5,
      fact2: 0.55,
      fact3: 0.65,
      wisdom: 0.75, // Warm knowing smile
      outro: 0.9 // Biggest smile - celebration
    }
  },
  lips: {
    emphasis_cues: [
      'slight_purse_on_key_words',
      'corners_up_on_questions',
      'neutral_on_transitions'
    ]
  }
};

// CLI usage
if (require.main === module) {
  const phase = (process.argv[2] as Phase) || 'hook';
  
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎭 KELLY EXPRESSION ENGINE                                ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
  
  const expr = getPhaseExpression(phase);
  const motion = generateMotionPrompt(phase);
  
  console.log(`📍 Phase: ${phase.toUpperCase()}\n`);
  console.log(`📝 Description: ${expr.description}\n`);
  console.log(`🎬 Motion Prompt: "${motion}"\n`);
  console.log('💫 Expression Config:');
  console.log(`   Energy: ${(expr.energy * 100).toFixed(0)}%`);
  console.log(`   Smile: ${(expr.smile * 100).toFixed(0)}%`);
  console.log(`   Gaze Intensity: ${(expr.gaze.intensity * 100).toFixed(0)}%`);
  console.log(`   Head Tilt: ${expr.movement.head_tilt}`);
  console.log(`   Nod Frequency: ${expr.movement.nod_frequency}`);
  
  console.log('\n🎯 Kelly Presence Guidelines:');
  console.log('   Eyes: "Should feel like she\'s finding YOU in a crowded room"');
  console.log('   Smile: "Genuine Duchenne smile = trust"');
  console.log('   Lips: "Subtle direction during emphasis"');
}
