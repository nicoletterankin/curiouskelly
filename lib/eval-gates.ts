/**
 * Zero-Trust Eval Gates for Kelly Pipeline
 * 
 * Content, Audio, Video, and Deploy evaluation gates.
 * Blocks bad content before it ships.
 */

import type { EvalGateResult, VideoJob } from './engines/types';

// ============================================
// SLOP PATTERNS (Zero-Trust Content Check)
// ============================================

export const SLOP_PATTERNS = [
  // Generic AI fluff
  /dive\s*(into|deep)/i,
  /unlock\s*the\s*secrets?/i,
  /embark\s*on\s*(a\s*)?journey/i,
  /game[- ]?changer/i,
  /transformative\s*(experience|journey)/i,
  /unleash\s*(your|the)/i,
  /paradigm\s*shift/i,
  /revolutionary\s*(approach|method)/i,
  /cutting[- ]?edge/i,
  /next[- ]?level/i,
  /level\s*up/i,
  /supercharge/i,
  /skyrocket/i,
  /game[- ]?changing/i,
  /mind[- ]?blowing/i,
  /jaw[- ]?dropping/i,
  /groundbreaking/i,
  /world[- ]?class/i,
  /best[- ]?in[- ]?class/i,
  /state[- ]?of[- ]?the[- ]?art/i,
  
  // Salesy patterns
  /act\s*now/i,
  /limited\s*time/i,
  /don'?t\s*miss\s*(out|this)/i,
  /exclusive\s*(access|offer)/i,
  /hurry/i,
  /last\s*chance/i,
  /once[- ]?in[- ]?a[- ]?lifetime/i,
  
  // Filler words
  /at\s*the\s*end\s*of\s*the\s*day/i,
  /let'?s\s*face\s*it/i,
  /needless\s*to\s*say/i,
  /it\s*goes\s*without\s*saying/i,
  /in\s*this\s*day\s*and\s*age/i,
  
  // Overwrought phrasing
  /delve\s*(into|deep)/i,
  /plethora/i,
  /myriad/i,
  /plunge\s*(into|deep)/i,
  /treasure\s*trove/i,
];

// ============================================
// FORBIDDEN WORDS (Kelly Voice Violations)
// ============================================

export const FORBIDDEN_WORDS = [
  'user',           // Say "learner"
  'users',          // Say "learners"
  'unlock',         // Transactional
  'exclusive',      // Scarcity/FOMO
  'amazing',        // Overused
  'awesome',        // Overused
  'incredible',     // Overused
  'leverage',       // Corporate jargon
  'synergy',        // Corporate jargon
  'optimize',       // Corporate jargon
  'utilize',        // Corporate jargon
];

// ============================================
// EVAL GATE: CONTENT
// ============================================

export interface ContentEvalInput {
  text: string;
  phase?: string;
  day?: number;
}

export function evaluateContent(input: ContentEvalInput): EvalGateResult {
  const issues: string[] = [];
  const text = input.text;
  
  // Check slop patterns
  for (const pattern of SLOP_PATTERNS) {
    if (pattern.test(text)) {
      issues.push(`SLOP: "${text.match(pattern)?.[0]}" matches ${pattern.toString()}`);
    }
  }
  
  // Check forbidden words
  const lowerText = text.toLowerCase();
  for (const word of FORBIDDEN_WORDS) {
    if (lowerText.includes(word.toLowerCase())) {
      issues.push(`FORBIDDEN: contains "${word}"`);
    }
  }
  
  // Check excessive punctuation
  const exclamations = (text.match(/!/g) || []).length;
  if (exclamations > 2) {
    issues.push(`PUNCTUATION: ${exclamations} exclamation marks (max 2)`);
  }
  
  // Check emoji spam
  const emojiRegex = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}]/gu;
  const emojis = (text.match(emojiRegex) || []).length;
  if (emojis > 2) {
    issues.push(`EMOJI: ${emojis} emojis (max 2)`);
  }
  
  // Check length (too short = incomplete, too long = rambling)
  if (text.length < 20) {
    issues.push(`LENGTH: Too short (${text.length} chars, min 20)`);
  }
  if (text.length > 2000) {
    issues.push(`LENGTH: Too long (${text.length} chars, max 2000)`);
  }
  
  // Calculate score (start at 100, deduct per issue)
  const score = Math.max(0, 100 - issues.length * 20);
  
  return {
    passed: issues.length === 0,
    score,
    issues,
    retries: 0,
    timestamp: new Date().toISOString(),
  };
}

// ============================================
// EVAL GATE: AUDIO
// ============================================

export interface AudioEvalInput {
  url?: string;
  duration_seconds?: number;
  sample_rate?: number;
  file_size_bytes?: number;
}

export function evaluateAudio(input: AudioEvalInput): EvalGateResult {
  const issues: string[] = [];
  
  // URL check
  if (!input.url) {
    issues.push('MISSING: No audio URL provided');
  } else if (!input.url.startsWith('http')) {
    issues.push('INVALID: Audio URL must be HTTP/HTTPS');
  }
  
  // Duration check (Kelly lessons are typically 30s-5min)
  if (input.duration_seconds !== undefined) {
    if (input.duration_seconds < 5) {
      issues.push(`DURATION: Too short (${input.duration_seconds}s, min 5s)`);
    }
    if (input.duration_seconds > 600) {
      issues.push(`DURATION: Too long (${input.duration_seconds}s, max 600s)`);
    }
  }
  
  // Sample rate check (ElevenLabs outputs 22050 or 44100 Hz)
  if (input.sample_rate !== undefined) {
    if (input.sample_rate < 16000) {
      issues.push(`SAMPLE_RATE: Too low (${input.sample_rate} Hz, min 16000)`);
    }
  }
  
  // File size check
  if (input.file_size_bytes !== undefined) {
    if (input.file_size_bytes < 1000) {
      issues.push(`FILE_SIZE: Too small (${input.file_size_bytes} bytes)`);
    }
    if (input.file_size_bytes > 50_000_000) {
      issues.push(`FILE_SIZE: Too large (${input.file_size_bytes} bytes, max 50MB)`);
    }
  }
  
  const score = Math.max(0, 100 - issues.length * 25);
  
  return {
    passed: issues.length === 0,
    score,
    issues,
    retries: 0,
    timestamp: new Date().toISOString(),
  };
}

// ============================================
// EVAL GATE: VIDEO
// ============================================

export interface VideoEvalInput {
  url?: string;
  duration_seconds?: number;
  resolution?: { width: number; height: number };
  file_size_bytes?: number;
  format?: string;
}

export function evaluateVideo(input: VideoEvalInput): EvalGateResult {
  const issues: string[] = [];
  
  // URL check
  if (!input.url) {
    issues.push('MISSING: No video URL provided');
  } else if (!input.url.startsWith('http')) {
    issues.push('INVALID: Video URL must be HTTP/HTTPS');
  }
  
  // Duration alignment (video should roughly match audio)
  if (input.duration_seconds !== undefined) {
    if (input.duration_seconds < 3) {
      issues.push(`DURATION: Too short (${input.duration_seconds}s, min 3s)`);
    }
    if (input.duration_seconds > 600) {
      issues.push(`DURATION: Too long (${input.duration_seconds}s, max 600s)`);
    }
  }
  
  // Resolution check (minimum 720p for quality)
  if (input.resolution) {
    const minRes = 720;
    if (input.resolution.width < minRes && input.resolution.height < minRes) {
      issues.push(`RESOLUTION: Too low (${input.resolution.width}x${input.resolution.height}, min ${minRes}p)`);
    }
  }
  
  // Format check
  if (input.format && !['mp4', 'webm', 'mov'].includes(input.format.toLowerCase())) {
    issues.push(`FORMAT: Unsupported format "${input.format}" (use mp4, webm, or mov)`);
  }
  
  const score = Math.max(0, 100 - issues.length * 25);
  
  return {
    passed: issues.length === 0,
    score,
    issues,
    retries: 0,
    timestamp: new Date().toISOString(),
  };
}

// ============================================
// EVAL GATE: DEPLOY
// ============================================

export interface DeployEvalInput {
  content_eval?: EvalGateResult;
  audio_eval?: EvalGateResult;
  video_eval?: EvalGateResult;
  video_url?: string;
  storage_verified?: boolean;
}

export function evaluateDeploy(input: DeployEvalInput): EvalGateResult {
  const issues: string[] = [];
  
  // All prior gates must pass
  if (!input.content_eval?.passed) {
    issues.push('DEPENDENCY: Content eval did not pass');
  }
  if (!input.audio_eval?.passed) {
    issues.push('DEPENDENCY: Audio eval did not pass');
  }
  if (!input.video_eval?.passed) {
    issues.push('DEPENDENCY: Video eval did not pass');
  }
  
  // Video must be accessible
  if (!input.video_url) {
    issues.push('MISSING: No video URL for deployment');
  }
  
  // Storage must be verified
  if (!input.storage_verified) {
    issues.push('STORAGE: Video not verified in storage');
  }
  
  const score = Math.max(0, 100 - issues.length * 25);
  
  return {
    passed: issues.length === 0,
    score,
    issues,
    retries: 0,
    timestamp: new Date().toISOString(),
  };
}

// ============================================
// FULL PIPELINE EVAL
// ============================================

export interface PipelineEvalResult {
  content: EvalGateResult;
  audio: EvalGateResult;
  video: EvalGateResult;
  deploy: EvalGateResult;
  overall_passed: boolean;
  needs_human_review: boolean;
  total_retries: number;
}

export function evaluatePipeline(
  contentInput: ContentEvalInput,
  audioInput: AudioEvalInput,
  videoInput: VideoEvalInput,
  storageVerified: boolean = true
): PipelineEvalResult {
  const content = evaluateContent(contentInput);
  const audio = evaluateAudio(audioInput);
  const video = evaluateVideo(videoInput);
  const deploy = evaluateDeploy({
    content_eval: content,
    audio_eval: audio,
    video_eval: video,
    video_url: videoInput.url,
    storage_verified: storageVerified,
  });
  
  const total_retries = content.retries + audio.retries + video.retries + deploy.retries;
  
  return {
    content,
    audio,
    video,
    deploy,
    overall_passed: deploy.passed,
    needs_human_review: total_retries >= 3,
    total_retries,
  };
}

// ============================================
// TEST CONTENT
// ============================================

export function runEvalTests(): void {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║           EVAL GATES TEST SUITE                              ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
  
  const testCases = [
    {
      name: 'Good content',
      text: "Hi, I'm Kelly. Today we're learning about compound interest. It's a simple idea that can help you understand how money grows over time.",
      shouldPass: true,
    },
    {
      name: 'Slop: dive deep',
      text: "Let's dive deep into the world of finance and unlock the secrets of success!",
      shouldPass: false,
    },
    {
      name: 'Slop: embark on journey',
      text: "Today we embark on a journey to discover the game-changer concepts of science.",
      shouldPass: false,
    },
    {
      name: 'Forbidden word: user',
      text: "Hello user, welcome to your daily lesson.",
      shouldPass: false,
    },
    {
      name: 'Emoji spam',
      text: "Today's lesson is amazing! 🎉🎉🎉🎉 Let's learn!",
      shouldPass: false,
    },
    {
      name: 'Too short',
      text: "Hi Kelly.",
      shouldPass: false,
    },
  ];
  
  let passed = 0;
  let failed = 0;
  
  for (const tc of testCases) {
    const result = evaluateContent({ text: tc.text });
    const testPassed = result.passed === tc.shouldPass;
    
    if (testPassed) {
      passed++;
      console.log(`✅ ${tc.name}`);
    } else {
      failed++;
      console.log(`❌ ${tc.name}`);
      console.log(`   Expected: ${tc.shouldPass ? 'PASS' : 'FAIL'}, Got: ${result.passed ? 'PASS' : 'FAIL'}`);
      if (result.issues.length > 0) {
        result.issues.forEach(issue => console.log(`   - ${issue}`));
      }
    }
  }
  
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`Results: ${passed}/${testCases.length} passed`);
  
  if (failed > 0) {
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  runEvalTests();
}
