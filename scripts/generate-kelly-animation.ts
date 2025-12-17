/**
 * Kelly Animation Generator
 * 
 * Generates Unity-ready animation JSON from:
 * - Audio file (for duration)
 * - Script text (for expression beats)
 * - Viseme data (from Rhubarb or iClone export)
 * 
 * Output format matches KellyAnimationPlayer.cs expectations.
 */

import * as fs from 'fs';
import * as path from 'path';

interface VisemeData {
  time: number;
  duration: number;
  viseme: string;
}

interface ExpressionFrame {
  frame: number;
  timestamp: number;
  mouthOpen: number;
  mouthWidth: number;
  smile: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
  leftBrowRaise: number;
  rightBrowRaise: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
}

interface KellyAnimationData {
  clipName: string;
  duration: number;
  fps: number;
  visemes: VisemeData[];
  expressions: ExpressionFrame[];
}

interface ExpressionBeat {
  time: number;
  duration: number;
  emotion: 'curious' | 'wonder' | 'reflective' | 'warm' | 'inviting' | 'neutral';
  intensity: number; // 0-1
}

// Expression presets for each emotion
const EMOTION_PRESETS: Record<string, Partial<ExpressionFrame>> = {
  neutral: {
    smile: 0.1,
    leftBrowRaise: 0.0,
    rightBrowRaise: 0.0,
    headPitch: 0,
    headYaw: 0,
  },
  curious: {
    smile: 0.2,
    leftBrowRaise: 0.4,
    rightBrowRaise: 0.4,
    headPitch: -3,  // slight head tilt forward
    headYaw: 2,     // slight turn
  },
  wonder: {
    smile: 0.3,
    leftBrowRaise: 0.6,
    rightBrowRaise: 0.6,
    headPitch: -5,
    headYaw: 0,
  },
  reflective: {
    smile: 0.15,
    leftBrowRaise: 0.2,
    rightBrowRaise: 0.2,
    headPitch: 5,   // looking slightly down
    headYaw: -3,
  },
  warm: {
    smile: 0.5,
    leftBrowRaise: 0.1,
    rightBrowRaise: 0.1,
    headPitch: -2,
    headYaw: 0,
  },
  inviting: {
    smile: 0.4,
    leftBrowRaise: 0.3,
    rightBrowRaise: 0.3,
    headPitch: -3,
    headYaw: 0,
  },
};

/**
 * Parse the Kelly intro script and generate expression beats
 */
function generateExpressionBeats(scriptText: string, totalDuration: number): ExpressionBeat[] {
  // For the intro script, we know the emotional beats:
  const beats: ExpressionBeat[] = [
    { time: 0, duration: 3, emotion: 'curious', intensity: 0.7 },      // "Octopuses have three hearts"
    { time: 3, duration: 3, emotion: 'curious', intensity: 0.8 },      // "Two for the gills..."
    { time: 6, duration: 3, emotion: 'wonder', intensity: 0.9 },       // "That third one stops"
    { time: 9, duration: 4, emotion: 'reflective', intensity: 0.7 },   // "I keep thinking about that"
    { time: 13, duration: 4, emotion: 'reflective', intensity: 0.6 },  // "What would it feel like"
    { time: 17, duration: 4, emotion: 'warm', intensity: 0.6 },        // "I'm Kelly"
    { time: 21, duration: 5, emotion: 'curious', intensity: 0.7 },     // "This is the kind of thing"
    { time: 26, duration: 4, emotion: 'neutral', intensity: 0.5 },     // "Five minutes. One thing."
    { time: 30, duration: 4, emotion: 'warm', intensity: 0.8 },        // "A year of that?"
    { time: 34, duration: 3, emotion: 'warm', intensity: 0.9 },        // "And so do you"
    { time: 37, duration: 2, emotion: 'inviting', intensity: 0.8 },    // "Want to jump in?"
  ];

  return beats.filter(b => b.time < totalDuration);
}

/**
 * Generate expression frames from beats
 */
function generateExpressionFrames(beats: ExpressionBeat[], duration: number, fps: number): ExpressionFrame[] {
  const totalFrames = Math.ceil(duration * fps);
  const frames: ExpressionFrame[] = [];

  for (let frame = 0; frame < totalFrames; frame++) {
    const timestamp = frame / fps;
    
    // Find the active beat
    let activeBeat = beats.find(b => timestamp >= b.time && timestamp < b.time + b.duration);
    if (!activeBeat) {
      activeBeat = { time: 0, duration: 1, emotion: 'neutral', intensity: 0.3 };
    }

    const preset = EMOTION_PRESETS[activeBeat.emotion] || EMOTION_PRESETS.neutral;
    const intensity = activeBeat.intensity;

    // Add natural variation
    const variation = Math.sin(timestamp * 2) * 0.05;
    
    // Random blinks (every 3-5 seconds on average)
    const shouldBlink = Math.random() < 0.01; // ~1% chance per frame at 25fps ≈ blink every 4s
    const blinkValue = shouldBlink ? 0.0 : 1.0;

    frames.push({
      frame,
      timestamp,
      mouthOpen: 0, // Handled by visemes
      mouthWidth: 0, // Handled by visemes
      smile: (preset.smile || 0) * intensity + variation,
      leftEyeOpen: blinkValue,
      rightEyeOpen: blinkValue,
      leftBrowRaise: (preset.leftBrowRaise || 0) * intensity,
      rightBrowRaise: (preset.rightBrowRaise || 0) * intensity,
      headYaw: (preset.headYaw || 0) * intensity,
      headPitch: (preset.headPitch || 0) * intensity,
      headRoll: Math.sin(timestamp * 0.5) * 2 * intensity, // Subtle head sway
    });
  }

  return frames;
}

/**
 * Generate placeholder visemes (should be replaced with Rhubarb output)
 */
function generatePlaceholderVisemes(duration: number): VisemeData[] {
  // This is a placeholder - real visemes should come from:
  // 1. Rhubarb lip sync tool
  // 2. iClone AccuLips export
  // 3. ElevenLabs timestamp data
  return [
    { time: 0, duration: duration, viseme: 'viseme_sil' }
  ];
}

/**
 * Main generation function
 */
export function generateKellyAnimation(
  clipName: string,
  duration: number,
  scriptText: string,
  visemes?: VisemeData[],
  fps: number = 25
): KellyAnimationData {
  const beats = generateExpressionBeats(scriptText, duration);
  const expressions = generateExpressionFrames(beats, duration, fps);
  
  return {
    clipName,
    duration,
    fps,
    visemes: visemes || generatePlaceholderVisemes(duration),
    expressions,
  };
}

// CLI usage
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('Usage: npx ts-node generate-kelly-animation.ts <clipName> <duration> [scriptFile]');
    process.exit(1);
  }

  const clipName = args[0];
  const duration = parseFloat(args[1]);
  const scriptFile = args[2];
  
  let scriptText = '';
  if (scriptFile && fs.existsSync(scriptFile)) {
    scriptText = fs.readFileSync(scriptFile, 'utf-8');
  }

  const animation = generateKellyAnimation(clipName, duration, scriptText);
  
  const outputPath = `${clipName}_unity.json`;
  fs.writeFileSync(outputPath, JSON.stringify(animation, null, 2));
  console.log(`Generated: ${outputPath}`);
}
