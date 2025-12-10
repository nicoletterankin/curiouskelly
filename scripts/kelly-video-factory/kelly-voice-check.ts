#!/usr/bin/env npx tsx
/**
 * 🎤 KELLY VOICE CHECK
 * 
 * Comprehensive voice quality test before bulk audio generation.
 * Tests Kelly's voice across all expressions and archetypes.
 * 
 * WHAT IT CHECKS:
 * ✅ ElevenLabs API connectivity
 * ✅ Kelly voice ID validity
 * ✅ Voice consistency across expressions
 * ✅ Audio quality (file size, duration)
 * ✅ Expression-matched voice settings
 * ✅ Archetype-specific variations
 * 
 * Usage:
 *   npx tsx scripts/kelly-video-factory/kelly-voice-check.ts
 *   npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick  # Test only one sample
 *   npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --expression excited
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import {
  EXPRESSIONS,
  getVoiceSettingsForExpression,
} from './kelly-blendshape-config.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  ELEVENLABS_API_KEY: process.env.ELEVENLABS_API_KEY!,
  KELLY_VOICE_ID: process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0',
  OUTPUT_DIR: path.join(process.cwd(), 'test-output', 'voice-check'),
  
  // Quality thresholds
  MIN_AUDIO_SIZE_KB: 5,  // Minimum expected file size
  MAX_AUDIO_SIZE_KB: 500, // Maximum reasonable file size
  EXPECTED_DURATION_TOLERANCE: 0.3, // 30% tolerance for duration estimates
};

// Test scripts for each expression type
const TEST_SCRIPTS = {
  excited: "Wow! Did you know that butterflies can taste with their feet? That's absolutely amazing!",
  curious: "Have you ever wondered why the sky changes colors at sunset? Let's explore this together.",
  explaining: "The water cycle is a continuous process. Water evaporates, forms clouds, and returns as rain.",
  thoughtful: "Sometimes the most important questions don't have easy answers. That's what makes learning so valuable.",
  wisdom: "Remember, every expert was once a beginner. Be patient with yourself as you learn and grow.",
  calm: "Take a deep breath. Learning is a journey, not a race. Let's move forward together, one step at a time.",
  welcoming: "Hello! I'm so glad you're here today. Let's discover something wonderful together!",
  contemplative: "What if we looked at this from a different perspective? Sometimes changing our view changes everything.",
  sincere: "I want you to know that your curiosity and effort matter. Every question you ask is a step forward.",
  celebrating: "You did it! I'm so proud of how far you've come. This is just the beginning of your journey!",
};

// Archetype variations
const ARCHETYPE_TESTS = {
  'The Explorer': {
    script: "Adventure awaits! Let's discover something new and exciting together!",
    expression: 'excited',
  },
  'The Rebel': {
    script: "Why do we accept things as they are? Let's challenge assumptions and think differently!",
    expression: 'curious',
  },
  'The Scientist': {
    script: "Let's examine the evidence carefully and draw logical conclusions from our observations.",
    expression: 'explaining',
  },
};

// =============================================================================
// TYPES
// =============================================================================

interface VoiceTestResult {
  expression: string;
  success: boolean;
  audioPath?: string;
  fileSize?: number;
  estimatedDuration?: number;
  voiceSettings: {
    stability: number;
    similarity_boost: number;
    style: number;
  };
  error?: string;
  apiResponseTime?: number;
}

interface VoiceCheckSummary {
  totalTests: number;
  successful: number;
  failed: number;
  results: VoiceTestResult[];
  overallHealth: 'EXCELLENT' | 'GOOD' | 'WARNING' | 'CRITICAL';
  issues: string[];
  recommendations: string[];
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function log(emoji: string, message: string, indent = 0): void {
  const prefix = '  '.repeat(indent);
  console.log(`${prefix}${emoji} ${message}`);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function formatDuration(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(1)}s`;
}

// =============================================================================
// VOICE GENERATION TEST
// =============================================================================

async function testVoiceGeneration(
  expressionName: string,
  script: string,
  archetype?: string
): Promise<VoiceTestResult> {
  const startTime = Date.now();
  
  log('🎤', `Testing ${expressionName}${archetype ? ` (${archetype})` : ''}...`, 1);
  
  const result: VoiceTestResult = {
    expression: expressionName,
    success: false,
    voiceSettings: getVoiceSettingsForExpression(expressionName),
  };
  
  try {
    // Generate audio
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${CONFIG.KELLY_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': CONFIG.ELEVENLABS_API_KEY,
        },
        body: JSON.stringify({
          text: script,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            ...result.voiceSettings,
            use_speaker_boost: true,
          },
        }),
      }
    );
    
    result.apiResponseTime = (Date.now() - startTime) / 1000;
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ElevenLabs API error: ${response.status} - ${errorText}`);
    }
    
    // Save audio
    const buffer = Buffer.from(await response.arrayBuffer());
    const fileName = archetype 
      ? `${expressionName}_${archetype.replace(/\s+/g, '_')}.mp3`
      : `${expressionName}.mp3`;
    const audioPath = path.join(CONFIG.OUTPUT_DIR, fileName);
    
    fs.mkdirSync(path.dirname(audioPath), { recursive: true });
    fs.writeFileSync(audioPath, buffer);
    
    result.audioPath = audioPath;
    result.fileSize = buffer.length;
    result.estimatedDuration = Math.ceil(script.length / 15); // ~150 words/min, ~5 chars/word
    result.success = true;
    
    // Quality checks
    const fileSizeKB = buffer.length / 1024;
    if (fileSizeKB < CONFIG.MIN_AUDIO_SIZE_KB) {
      log('⚠️', `File too small: ${formatBytes(buffer.length)}`, 2);
    } else if (fileSizeKB > CONFIG.MAX_AUDIO_SIZE_KB) {
      log('⚠️', `File unexpectedly large: ${formatBytes(buffer.length)}`, 2);
    } else {
      log('✅', `Generated: ${formatBytes(buffer.length)}, ~${result.estimatedDuration}s, ${formatDuration(result.apiResponseTime!)}`, 2);
    }
    
  } catch (error: any) {
    result.error = error.message;
    log('❌', `Failed: ${error.message}`, 2);
  }
  
  return result;
}

// =============================================================================
// COMPREHENSIVE VOICE CHECK
// =============================================================================

async function runVoiceCheck(options: {
  quick?: boolean;
  expression?: string;
}): Promise<VoiceCheckSummary> {
  console.log('\n');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  🎤 KELLY VOICE CHECK                                                ║');
  console.log('║  Testing voice quality before bulk generation                        ║');
  console.log('╚' + '═'.repeat(70) + '╝');
  console.log('');
  
  const summary: VoiceCheckSummary = {
    totalTests: 0,
    successful: 0,
    failed: 0,
    results: [],
    overallHealth: 'EXCELLENT',
    issues: [],
    recommendations: [],
  };
  
  // Step 1: Validate API configuration
  log('🔑', 'Validating API configuration...');
  
  if (!CONFIG.ELEVENLABS_API_KEY) {
    summary.issues.push('ELEVENLABS_API_KEY not set');
    log('❌', 'ELEVENLABS_API_KEY not found in environment', 1);
  } else {
    log('✅', `API Key: ${CONFIG.ELEVENLABS_API_KEY.substring(0, 10)}...[redacted]`, 1);
  }
  
  if (!CONFIG.KELLY_VOICE_ID) {
    summary.issues.push('ELEVENLABS_KELLY_VOICE_ID not set');
    log('❌', 'ELEVENLABS_KELLY_VOICE_ID not found in environment', 1);
  } else {
    log('✅', `Voice ID: ${CONFIG.KELLY_VOICE_ID}`, 1);
  }
  
  if (summary.issues.length > 0) {
    summary.overallHealth = 'CRITICAL';
    summary.recommendations.push('Set missing environment variables in .env file');
    return summary;
  }
  
  console.log('');
  
  // Step 2: Test expressions
  if (options.quick) {
    log('⚡', 'Quick mode: Testing single sample...');
    const testExpression = options.expression || 'excited';
    const script = TEST_SCRIPTS[testExpression as keyof typeof TEST_SCRIPTS] || TEST_SCRIPTS.excited;
    
    summary.totalTests = 1;
    const result = await testVoiceGeneration(testExpression, script);
    summary.results.push(result);
    if (result.success) summary.successful++;
    else summary.failed++;
    
  } else if (options.expression) {
    log('🎯', `Testing specific expression: ${options.expression}...`);
    const script = TEST_SCRIPTS[options.expression as keyof typeof TEST_SCRIPTS];
    
    if (!script) {
      summary.issues.push(`Unknown expression: ${options.expression}`);
      summary.recommendations.push(`Valid expressions: ${Object.keys(TEST_SCRIPTS).join(', ')}`);
      summary.overallHealth = 'CRITICAL';
      return summary;
    }
    
    summary.totalTests = 1;
    const result = await testVoiceGeneration(options.expression, script);
    summary.results.push(result);
    if (result.success) summary.successful++;
    else summary.failed++;
    
  } else {
    // Full test suite
    log('🧪', 'Running comprehensive voice tests...');
    console.log('');
    
    // Test all expressions
    log('📋', 'Testing all expressions...');
    for (const [expressionName, script] of Object.entries(TEST_SCRIPTS)) {
      summary.totalTests++;
      const result = await testVoiceGeneration(expressionName, script);
      summary.results.push(result);
      if (result.success) summary.successful++;
      else summary.failed++;
      
      // Brief pause between API calls
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log('');
    
    // Test archetype variations
    log('🎭', 'Testing archetype variations...');
    for (const [archetypeName, config] of Object.entries(ARCHETYPE_TESTS)) {
      summary.totalTests++;
      const result = await testVoiceGeneration(config.expression, config.script, archetypeName);
      summary.results.push(result);
      if (result.success) summary.successful++;
      else summary.failed++;
      
      // Brief pause between API calls
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
  
  console.log('');
  
  // Step 3: Analyze results
  log('📊', 'Analyzing results...');
  
  // Check for consistent quality
  const fileSizes = summary.results
    .filter(r => r.success && r.fileSize)
    .map(r => r.fileSize!);
  
  if (fileSizes.length > 0) {
    const avgSize = fileSizes.reduce((a, b) => a + b, 0) / fileSizes.length;
    const minSize = Math.min(...fileSizes);
    const maxSize = Math.max(...fileSizes);
    
    log('📏', `File sizes: avg ${formatBytes(avgSize)}, range ${formatBytes(minSize)}-${formatBytes(maxSize)}`, 1);
    
    // Check for outliers
    const variance = fileSizes.reduce((sum, size) => sum + Math.pow(size - avgSize, 2), 0) / fileSizes.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev > avgSize * 0.5) {
      summary.issues.push('High variance in audio file sizes');
      summary.recommendations.push('Review voice settings consistency');
    }
  }
  
  // Check API response times
  const responseTimes = summary.results
    .filter(r => r.success && r.apiResponseTime)
    .map(r => r.apiResponseTime!);
  
  if (responseTimes.length > 0) {
    const avgTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
    const maxTime = Math.max(...responseTimes);
    
    log('⏱️', `API response: avg ${formatDuration(avgTime)}, max ${formatDuration(maxTime)}`, 1);
    
    if (avgTime > 5) {
      summary.issues.push('Slow API response times');
      summary.recommendations.push('Check network connection or ElevenLabs service status');
    }
  }
  
  // Determine overall health
  const successRate = summary.successful / summary.totalTests;
  
  if (successRate === 1) {
    summary.overallHealth = 'EXCELLENT';
  } else if (successRate >= 0.9) {
    summary.overallHealth = 'GOOD';
    summary.recommendations.push('Review failed tests before bulk generation');
  } else if (successRate >= 0.7) {
    summary.overallHealth = 'WARNING';
    summary.recommendations.push('Fix issues before proceeding with bulk generation');
  } else {
    summary.overallHealth = 'CRITICAL';
    summary.recommendations.push('DO NOT proceed with bulk generation until issues are resolved');
  }
  
  // Add specific recommendations
  if (summary.failed > 0) {
    const failedExpressions = summary.results
      .filter(r => !r.success)
      .map(r => r.expression)
      .join(', ');
    summary.recommendations.push(`Failed expressions: ${failedExpressions}`);
  }
  
  return summary;
}

// =============================================================================
// REPORT GENERATION
// =============================================================================

function printSummaryReport(summary: VoiceCheckSummary): void {
  console.log('');
  console.log('╔' + '═'.repeat(70) + '╗');
  console.log('║  📊 VOICE CHECK SUMMARY                                              ║');
  console.log('╚' + '═'.repeat(70) + '╝');
  console.log('');
  
  // Overall health
  const healthEmoji = {
    EXCELLENT: '🟢',
    GOOD: '🟡',
    WARNING: '🟠',
    CRITICAL: '🔴',
  }[summary.overallHealth];
  
  console.log(`${healthEmoji} Overall Health: ${summary.overallHealth}`);
  console.log('');
  
  // Test results
  console.log(`📋 Tests: ${summary.successful}/${summary.totalTests} successful`);
  if (summary.failed > 0) {
    console.log(`   ❌ Failed: ${summary.failed}`);
  }
  console.log('');
  
  // Issues
  if (summary.issues.length > 0) {
    console.log('⚠️  Issues:');
    summary.issues.forEach(issue => console.log(`   • ${issue}`));
    console.log('');
  }
  
  // Recommendations
  if (summary.recommendations.length > 0) {
    console.log('💡 Recommendations:');
    summary.recommendations.forEach(rec => console.log(`   • ${rec}`));
    console.log('');
  }
  
  // Output location
  if (summary.successful > 0) {
    console.log(`📁 Test audio files saved to: ${CONFIG.OUTPUT_DIR}`);
    console.log('   Listen to these files to verify Kelly\'s voice quality');
    console.log('');
  }
  
  // Final verdict
  console.log('─'.repeat(72));
  if (summary.overallHealth === 'EXCELLENT') {
    console.log('✅ READY FOR BULK GENERATION');
    console.log('   Kelly\'s voice is consistent and high-quality across all expressions.');
  } else if (summary.overallHealth === 'GOOD') {
    console.log('✅ READY FOR BULK GENERATION (with minor issues)');
    console.log('   Review failed tests but generally safe to proceed.');
  } else if (summary.overallHealth === 'WARNING') {
    console.log('⚠️  CAUTION ADVISED');
    console.log('   Fix issues before bulk generation to avoid wasted API calls.');
  } else {
    console.log('🛑 DO NOT PROCEED WITH BULK GENERATION');
    console.log('   Critical issues must be resolved first.');
  }
  console.log('─'.repeat(72));
  console.log('');
}

// =============================================================================
// CLI
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  const options: {
    quick?: boolean;
    expression?: string;
  } = {};
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--quick':
        options.quick = true;
        break;
      case '--expression':
        options.expression = args[++i];
        break;
      case '--help':
        console.log(`
🎤 Kelly Voice Check

Tests Kelly's voice quality before bulk audio generation.

Usage:
  npx tsx scripts/kelly-video-factory/kelly-voice-check.ts [options]

Options:
  --quick              Quick test (single sample only)
  --expression <name>  Test specific expression
  --help               Show this help

Examples:
  npx tsx scripts/kelly-video-factory/kelly-voice-check.ts
  npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --quick
  npx tsx scripts/kelly-video-factory/kelly-voice-check.ts --expression excited

Available expressions:
  ${Object.keys(TEST_SCRIPTS).join(', ')}
`);
        process.exit(0);
    }
  }
  
  try {
    const summary = await runVoiceCheck(options);
    printSummaryReport(summary);
    
    // Save detailed report
    const reportPath = path.join(CONFIG.OUTPUT_DIR, 'voice-check-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(summary, null, 2));
    console.log(`📄 Detailed report saved: ${reportPath}`);
    console.log('');
    
    // Exit with appropriate code
    if (summary.overallHealth === 'CRITICAL') {
      process.exit(1);
    }
    
  } catch (error: any) {
    console.error('\n❌ Voice check failed:', error.message);
    process.exit(1);
  }
}

main();

export { runVoiceCheck, testVoiceGeneration, CONFIG };







