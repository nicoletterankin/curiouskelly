/**
 * Kelly Voice Evaluation System
 * Automated quality gate for all Kelly communications
 * 
 * Run: npx ts-node evals/kelly-voice-eval.ts
 */

// ============================================
// KELLY VOICE RULES (from docs/brand/KELLY_VOICE.md)
// ============================================

const FORBIDDEN_WORDS = [
  'user',        // Say "learner" instead
  'users',       // Say "learners" instead
  'unlock',      // Transactional
  'access',      // Transactional (context-dependent)
  'exclusive',   // Scarcity/FOMO
  'limited',     // Scarcity/FOMO
  "don't miss",  // FOMO
  'act now',     // Urgency
  'hurry',       // Urgency
  'last chance', // Scarcity
  'amazing',     // Overused, cheap
  'awesome',     // Overused, cheap
  'incredible',  // Overused, cheap
];

const FORBIDDEN_PATTERNS = [
  /!{2,}/,                    // Multiple exclamation marks
  /🎉.*🎉/,                   // Emoji sandwiches
  /[😀😃😄😁🤩🥳]{3,}/,        // Emoji spam
  /click here/i,              // Generic CTA
  /sign up now/i,             // Pushy
  /subscribe now/i,           // Pushy
  /get started/i,             // Generic (prefer "begin" or specific action)
];

const REQUIRED_QUALITIES = {
  humble: [
    "don't have all the answers",
    'love finding',
    'together',
    'with you',
    'alongside',
  ],
  warm: [
    'hi',
    'hello',
    'friend',
    'glad',
    'happy',
    'wonderful',
  ],
  inviting: [
    'want to',
    'would you like',
    'come along',
    'join',
    'welcome',
  ],
};

// ============================================
// SCORING FUNCTIONS
// ============================================

interface EvalResult {
  score: number;
  maxScore: number;
  passed: boolean;
  issues: string[];
  suggestions: string[];
}

interface DetailedEval {
  humility: EvalResult;
  warmth: EvalResult;
  simplicity: EvalResult;
  invitation: EvalResult;
  richness: EvalResult;
  collaboration: EvalResult;
  overall: {
    score: number;
    maxScore: number;
    passed: boolean;
    grade: string;
  };
}

function checkForbiddenWords(text: string): string[] {
  const issues: string[] = [];
  const lowerText = text.toLowerCase();
  
  for (const word of FORBIDDEN_WORDS) {
    if (lowerText.includes(word.toLowerCase())) {
      issues.push(`Contains forbidden word: "${word}"`);
    }
  }
  
  return issues;
}

function checkForbiddenPatterns(text: string): string[] {
  const issues: string[] = [];
  
  for (const pattern of FORBIDDEN_PATTERNS) {
    if (pattern.test(text)) {
      issues.push(`Matches forbidden pattern: ${pattern.toString()}`);
    }
  }
  
  return issues;
}

function countExclamationMarks(text: string): number {
  return (text.match(/!/g) || []).length;
}

function countEmojis(text: string): number {
  const emojiRegex = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu;
  return (text.match(emojiRegex) || []).length;
}

function hasQualityMarkers(text: string, markers: string[]): boolean {
  const lowerText = text.toLowerCase();
  return markers.some(marker => lowerText.includes(marker.toLowerCase()));
}

function evaluateHumility(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  // Check for arrogant language
  const arrogantPhrases = [
    'i will teach you',
    'you should',
    'you need to',
    'you must',
    "i've been waiting",
    'i have something to teach',
  ];
  
  const lowerText = text.toLowerCase();
  for (const phrase of arrogantPhrases) {
    if (lowerText.includes(phrase)) {
      issues.push(`Sounds superior: "${phrase}"`);
      score -= 1;
    }
  }
  
  // Check for humble markers
  if (!hasQualityMarkers(text, REQUIRED_QUALITIES.humble)) {
    suggestions.push('Consider adding humble language like "together" or "I don\'t have all the answers"');
    score -= 1;
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

function evaluateWarmth(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  // Check for cold/corporate language
  const coldPhrases = [
    'please note',
    'be advised',
    'this is to inform',
    'regarding your',
    'per our',
    'as per',
    'kindly',
  ];
  
  const lowerText = text.toLowerCase();
  for (const phrase of coldPhrases) {
    if (lowerText.includes(phrase)) {
      issues.push(`Sounds corporate: "${phrase}"`);
      score -= 1;
    }
  }
  
  // Check for warm markers
  if (!hasQualityMarkers(text, REQUIRED_QUALITIES.warm)) {
    suggestions.push('Add warmer language - start with "Hi" or reference shared experience');
    score -= 1;
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

function evaluateSimplicity(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  // Check sentence length (rough heuristic)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const avgWords = sentences.reduce((acc, s) => acc + s.trim().split(/\s+/).length, 0) / sentences.length;
  
  if (avgWords > 20) {
    issues.push(`Average sentence too long: ${avgWords.toFixed(1)} words`);
    suggestions.push('Break into shorter sentences');
    score -= 1;
  }
  
  // Check for jargon
  const jargon = [
    'leverage',
    'synergy',
    'optimize',
    'utilize',
    'facilitate',
    'implement',
    'strategize',
  ];
  
  const lowerText = text.toLowerCase();
  for (const word of jargon) {
    if (lowerText.includes(word)) {
      issues.push(`Contains jargon: "${word}"`);
      score -= 1;
    }
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

function evaluateInvitation(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  // Check for demanding language
  const demandingPhrases = [
    'you must',
    'you need to',
    'you should',
    'click now',
    'do this',
    'act now',
  ];
  
  const lowerText = text.toLowerCase();
  for (const phrase of demandingPhrases) {
    if (lowerText.includes(phrase)) {
      issues.push(`Sounds demanding: "${phrase}"`);
      score -= 1;
    }
  }
  
  // Check for inviting markers
  if (!hasQualityMarkers(text, REQUIRED_QUALITIES.inviting)) {
    suggestions.push('Use inviting language like "Want to come along?" instead of commands');
    score -= 1;
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

function evaluateRichness(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  // Check for cheap indicators
  const exclamations = countExclamationMarks(text);
  if (exclamations > 2) {
    issues.push(`Too many exclamation marks: ${exclamations}`);
    score -= 1;
  }
  
  const emojis = countEmojis(text);
  if (emojis > 2) {
    issues.push(`Too many emojis: ${emojis}`);
    score -= 1;
  }
  
  // Check for all-caps words
  const capsWords = text.match(/\b[A-Z]{3,}\b/g) || [];
  if (capsWords.length > 0) {
    issues.push(`Contains shouting (all caps): ${capsWords.join(', ')}`);
    score -= 1;
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

function evaluateCollaboration(text: string): EvalResult {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 5;
  
  const lowerText = text.toLowerCase();
  
  // Check for "I" focus vs "we/together" focus
  const iCount = (lowerText.match(/\bi\b/g) || []).length;
  const weCount = (lowerText.match(/\b(we|together|us)\b/g) || []).length;
  
  if (iCount > 5 && weCount === 0) {
    issues.push('Too focused on Kelly, not enough on learning together');
    suggestions.push('Add "together" or "we" language');
    score -= 2;
  }
  
  // Check for learner acknowledgment
  const learnerWords = ['you', 'your', 'learner'];
  const hasLearnerFocus = learnerWords.some(w => lowerText.includes(w));
  
  if (!hasLearnerFocus) {
    suggestions.push('Acknowledge the learner more directly');
    score -= 1;
  }
  
  return {
    score: Math.max(1, score),
    maxScore: 5,
    passed: score >= 4,
    issues,
    suggestions,
  };
}

// ============================================
// MAIN EVALUATION FUNCTION
// ============================================

export function evaluateKellyVoice(text: string): DetailedEval {
  // Run all evaluations
  const humility = evaluateHumility(text);
  const warmth = evaluateWarmth(text);
  const simplicity = evaluateSimplicity(text);
  const invitation = evaluateInvitation(text);
  const richness = evaluateRichness(text);
  const collaboration = evaluateCollaboration(text);
  
  // Check forbidden words/patterns
  const forbiddenWordIssues = checkForbiddenWords(text);
  const forbiddenPatternIssues = checkForbiddenPatterns(text);
  
  // Add forbidden issues to richness (affects overall quality)
  richness.issues.push(...forbiddenWordIssues, ...forbiddenPatternIssues);
  if (forbiddenWordIssues.length > 0 || forbiddenPatternIssues.length > 0) {
    richness.score = Math.max(1, richness.score - forbiddenWordIssues.length - forbiddenPatternIssues.length);
    richness.passed = richness.score >= 4;
  }
  
  // Calculate overall
  const totalScore = humility.score + warmth.score + simplicity.score + 
                     invitation.score + richness.score + collaboration.score;
  const maxScore = 30;
  
  // Grade
  let grade: string;
  if (totalScore >= 28) grade = 'A+ (Ship it)';
  else if (totalScore >= 25) grade = 'A (Minor polish)';
  else if (totalScore >= 22) grade = 'B (Needs work)';
  else if (totalScore >= 18) grade = 'C (Rewrite)';
  else grade = 'F (Start over)';
  
  // Pass if all individual scores >= 4 AND total >= 25
  const allPassed = [humility, warmth, simplicity, invitation, richness, collaboration]
    .every(e => e.passed);
  const passed = allPassed && totalScore >= 25;
  
  return {
    humility,
    warmth,
    simplicity,
    invitation,
    richness,
    collaboration,
    overall: {
      score: totalScore,
      maxScore,
      passed,
      grade,
    },
  };
}

// ============================================
// TEST CASES
// ============================================

const TEST_CASES = [
  {
    name: 'Perfect Kelly (Welcome Email)',
    text: `Hi — I'm Kelly.

I don't have all the answers. But I love finding them. And I think learning is better together.

Every day I find something wonderful and I can't wait to share it.

Want to come along?

— Kelly`,
    expectedPass: true,
  },
  {
    name: 'Good Kelly (Daily Lesson)',
    text: `Good morning.

I found something wonderful today: How Money Works

Five minutes. I think you'll love it.

Let's learn together.

— Kelly`,
    expectedPass: true,
  },
  {
    name: 'Bad Kelly (Superior)',
    text: `I've been waiting for you. I have something to teach you.

Curiosity is the most important thing. You should learn every day. Here's how.

Click here to unlock your learning potential!`,
    expectedPass: false,
  },
  {
    name: 'Bad Kelly (Salesy)',
    text: `🎉 Don't miss today's AMAZING lesson! Kelly has something INCREDIBLE to teach you! 

Unlock your learning potential NOW! 🚀🚀🚀

Sign up now for EXCLUSIVE access!`,
    expectedPass: false,
  },
  {
    name: 'Bad Kelly (Corporate)',
    text: `Dear User,

Please note that your account has been created. Per our terms, kindly verify your email to access our platform.

Thank you for your interest in our services.

Best regards,
The Kelly Team`,
    expectedPass: false,
  },
];

// ============================================
// RUN EVALS
// ============================================

function runEvals() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║           KELLY VOICE EVALUATION SYSTEM                      ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
  
  let passed = 0;
  let failed = 0;
  
  for (const testCase of TEST_CASES) {
    console.log(`\n📝 ${testCase.name}`);
    console.log('─'.repeat(60));
    
    const result = evaluateKellyVoice(testCase.text);
    
    // Print scores
    console.log(`  Humility:      ${result.humility.score}/5 ${result.humility.passed ? '✓' : '✗'}`);
    console.log(`  Warmth:        ${result.warmth.score}/5 ${result.warmth.passed ? '✓' : '✗'}`);
    console.log(`  Simplicity:    ${result.simplicity.score}/5 ${result.simplicity.passed ? '✓' : '✗'}`);
    console.log(`  Invitation:    ${result.invitation.score}/5 ${result.invitation.passed ? '✓' : '✗'}`);
    console.log(`  Richness:      ${result.richness.score}/5 ${result.richness.passed ? '✓' : '✗'}`);
    console.log(`  Collaboration: ${result.collaboration.score}/5 ${result.collaboration.passed ? '✓' : '✗'}`);
    console.log('─'.repeat(60));
    console.log(`  TOTAL:         ${result.overall.score}/${result.overall.maxScore}`);
    console.log(`  GRADE:         ${result.overall.grade}`);
    console.log(`  VERDICT:       ${result.overall.passed ? '✅ PASS' : '❌ FAIL'}`);
    
    // Print issues
    const allIssues = [
      ...result.humility.issues,
      ...result.warmth.issues,
      ...result.simplicity.issues,
      ...result.invitation.issues,
      ...result.richness.issues,
      ...result.collaboration.issues,
    ];
    
    if (allIssues.length > 0) {
      console.log('\n  Issues:');
      allIssues.forEach(issue => console.log(`    ⚠️  ${issue}`));
    }
    
    // Check expected result
    const testPassed = result.overall.passed === testCase.expectedPass;
    if (testPassed) {
      passed++;
      console.log(`\n  ✅ Test passed (expected ${testCase.expectedPass ? 'PASS' : 'FAIL'})`);
    } else {
      failed++;
      console.log(`\n  ❌ Test failed (expected ${testCase.expectedPass ? 'PASS' : 'FAIL'}, got ${result.overall.passed ? 'PASS' : 'FAIL'})`);
    }
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log(`EVAL RESULTS: ${passed}/${TEST_CASES.length} tests passed`);
  console.log('═'.repeat(60));
  
  if (failed > 0) {
    process.exit(1);
  }
}

export { runEvals, TEST_CASES };

// Run evals
runEvals();

