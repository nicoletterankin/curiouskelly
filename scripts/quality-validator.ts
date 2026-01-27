#!/usr/bin/env npx tsx
/**
 * ✅ QUALITY VALIDATOR
 * Validates all generated content for quality standards
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const LESSONS_DIR = path.join(process.cwd(), 'public', 'lessons');

const JARGON_LIST = ['synergy', 'paradigm', 'leverage', 'ideation', 'holistic', 'bandwidth'];

interface ValidationResult {
  day: number;
  phase: string;
  checks: Record<string, boolean>;
  passed: boolean;
  issues: string[];
}

function validateScript(script: string, phase: string): { passed: boolean; checks: Record<string, boolean>; issues: string[] } {
  const issues: string[] = [];
  
  const checks = {
    length: script.length >= 50 && script.length <= 2000,
    has_question: script.includes('?') || phase === 'wisdom' || phase === 'outro',
    no_jargon: !JARGON_LIST.some(word => script.toLowerCase().includes(word)),
    ends_properly: /[.!?]$/.test(script.trim()),
    no_placeholder: !script.includes('[NEEDS TRANSLATION]') && !script.includes('TODO'),
    has_kelly_voice: /you|your|we|us|let's|today/i.test(script),
    not_too_short: script.split(' ').length >= 10,
  };
  
  if (!checks.length) issues.push(`Length out of range: ${script.length} chars`);
  if (!checks.has_question && phase === 'hook') issues.push('Hook missing question');
  if (!checks.no_jargon) issues.push('Contains corporate jargon');
  if (!checks.ends_properly) issues.push('Does not end with punctuation');
  if (!checks.no_placeholder) issues.push('Contains placeholder text');
  if (!checks.has_kelly_voice) issues.push('Missing Kelly voice markers');
  if (!checks.not_too_short) issues.push('Script too short');
  
  return {
    passed: Object.values(checks).every(Boolean),
    checks,
    issues,
  };
}

async function validateDay(dayNumber: number): Promise<ValidationResult[]> {
  const filePath = path.join(LESSONS_DIR, `day-${dayNumber}.json`);
  const results: ValidationResult[] = [];
  
  if (!fs.existsSync(filePath)) {
    return [{
      day: dayNumber,
      phase: 'file',
      checks: { exists: false },
      passed: false,
      issues: ['File not found'],
    }];
  }
  
  try {
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    
    for (const [phase, data] of Object.entries(content.phases || {})) {
      const phaseData = data as any;
      const script = phaseData.script?.en || '';
      
      const validation = validateScript(script, phase);
      
      results.push({
        day: dayNumber,
        phase,
        checks: validation.checks,
        passed: validation.passed,
        issues: validation.issues,
      });
    }
    
    return results;
  } catch (err) {
    return [{
      day: dayNumber,
      phase: 'parse',
      checks: { valid_json: false },
      passed: false,
      issues: [(err as Error).message],
    }];
  }
}

async function main() {
  const args = process.argv.slice(2);
  let startDay = 1;
  let endDay = 30;
  
  for (const arg of args) {
    if (arg.startsWith('--days=')) {
      const range = arg.split('=')[1];
      if (range.includes('-')) {
        [startDay, endDay] = range.split('-').map(Number);
      }
    }
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║         ✅ QUALITY VALIDATOR                                 ║
╚══════════════════════════════════════════════════════════════╝
`);
  console.log(`Validating Days ${startDay}-${endDay}...\n`);
  
  let totalPassed = 0;
  let totalFailed = 0;
  const allIssues: { day: number; phase: string; issues: string[] }[] = [];
  
  for (let day = startDay; day <= endDay; day++) {
    const results = await validateDay(day);
    const dayPassed = results.every(r => r.passed);
    
    const passedCount = results.filter(r => r.passed).length;
    const failedCount = results.filter(r => !r.passed).length;
    
    totalPassed += passedCount;
    totalFailed += failedCount;
    
    const icon = dayPassed ? '✅' : '⚠️';
    console.log(`  ${icon} Day ${day}: ${passedCount} passed, ${failedCount} issues`);
    
    for (const result of results) {
      if (!result.passed) {
        allIssues.push({ day, phase: result.phase, issues: result.issues });
      }
    }
  }
  
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                        📊 SUMMARY                            ║
╚══════════════════════════════════════════════════════════════╝

  ✅ Passed: ${totalPassed}
  ⚠️  Failed: ${totalFailed}
  📁 Total checks: ${totalPassed + totalFailed}
`);
  
  if (allIssues.length > 0) {
    console.log('\n📋 ISSUES FOUND:\n');
    for (const issue of allIssues.slice(0, 20)) {
      console.log(`  Day ${issue.day} / ${issue.phase}:`);
      for (const i of issue.issues) {
        console.log(`    - ${i}`);
      }
    }
    if (allIssues.length > 20) {
      console.log(`\n  ... and ${allIssues.length - 20} more issues`);
    }
  }
}

main().catch(console.error);
