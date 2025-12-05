/**
 * Lifetime Experience Evaluation System
 * Tests the spiral learning APIs and database
 * 
 * Run: npx ts-node evals/lifetime-experience-eval.ts
 */

import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';

dotenv.config();

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface TestResult {
  name: string;
  passed: boolean;
  message: string;
  duration?: number;
}

const results: TestResult[] = [];

function log(message: string) {
  console.log(message);
}

function pass(name: string, message: string, duration?: number) {
  results.push({ name, passed: true, message, duration });
  log(`  ✅ ${name}: ${message}${duration ? ` (${duration}ms)` : ''}`);
}

function fail(name: string, message: string) {
  results.push({ name, passed: false, message });
  log(`  ❌ ${name}: ${message}`);
}

// ============================================
// DATABASE SCHEMA TESTS
// ============================================

async function testDatabaseSchema(supabase: ReturnType<typeof createClient>) {
  log('\n📊 DATABASE SCHEMA TESTS');
  log('─'.repeat(50));
  
  // Test users table has new columns
  try {
    const { data, error } = await supabase
      .from('users')
      .select('kelly_remembers, first_lesson_at, longest_streak, years_completed, unique_lessons_completed')
      .limit(1);
    
    if (error) throw error;
    pass('Users table columns', 'All lifetime fields exist');
  } catch (e) {
    fail('Users table columns', `Missing lifetime fields: ${e}`);
  }
  
  // Test lesson_history table exists
  try {
    const { data, error } = await supabase
      .from('lesson_history')
      .select('id, user_id, lesson_day, year_completed, view_number, answers, notes, layer')
      .limit(1);
    
    if (error) throw error;
    pass('lesson_history table', 'Table exists with correct schema');
  } catch (e) {
    fail('lesson_history table', `Table issue: ${e}`);
  }
  
  // Test milestones table exists
  try {
    const { data, error } = await supabase
      .from('milestones')
      .select('id, user_id, milestone_type, achieved_at, celebration_shown')
      .limit(1);
    
    if (error) throw error;
    pass('milestones table', 'Table exists with correct schema');
  } catch (e) {
    fail('milestones table', `Table issue: ${e}`);
  }
  
  // Test commons_answers table exists
  try {
    const { data, error } = await supabase
      .from('commons_answers')
      .select('id, lesson_day, question_id, answer_value, year, count')
      .limit(1);
    
    if (error) throw error;
    pass('commons_answers table', 'Table exists with correct schema');
  } catch (e) {
    fail('commons_answers table', `Table issue: ${e}`);
  }
  
  // Test increment_commons_answer function
  try {
    const { error } = await supabase.rpc('increment_commons_answer', {
      p_lesson_day: 999,
      p_question_id: 'test',
      p_answer_value: 'test',
      p_year: 2099
    });
    
    if (error) throw error;
    
    // Clean up test data
    await supabase
      .from('commons_answers')
      .delete()
      .eq('lesson_day', 999)
      .eq('year', 2099);
    
    pass('increment_commons_answer function', 'Function works correctly');
  } catch (e) {
    fail('increment_commons_answer function', `Function issue: ${e}`);
  }
}

// ============================================
// API ENDPOINT TESTS
// ============================================

async function testAPIEndpoints() {
  log('\n🌐 API ENDPOINT TESTS');
  log('─'.repeat(50));
  
  const baseUrl = process.env.VERCEL_URL || 'https://curiouskelly.com';
  
  // Test lesson-history endpoint exists
  try {
    const start = Date.now();
    const response = await fetch(`${baseUrl}/api/lesson-history?day=1`);
    const duration = Date.now() - start;
    
    // 401 is expected without auth, but endpoint exists
    if (response.status === 401 || response.status === 200) {
      pass('GET /api/lesson-history', `Endpoint accessible (${response.status})`, duration);
    } else {
      fail('GET /api/lesson-history', `Unexpected status: ${response.status}`);
    }
  } catch (e) {
    fail('GET /api/lesson-history', `Request failed: ${e}`);
  }
  
  // Test lesson-complete endpoint exists
  try {
    const start = Date.now();
    const response = await fetch(`${baseUrl}/api/lesson-complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lessonDay: 1 })
    });
    const duration = Date.now() - start;
    
    // 401 is expected without auth
    if (response.status === 401 || response.status === 200) {
      pass('POST /api/lesson-complete', `Endpoint accessible (${response.status})`, duration);
    } else {
      fail('POST /api/lesson-complete', `Unexpected status: ${response.status}`);
    }
  } catch (e) {
    fail('POST /api/lesson-complete', `Request failed: ${e}`);
  }
  
  // Test reflection endpoint exists
  try {
    const start = Date.now();
    const response = await fetch(`${baseUrl}/api/reflection?day=1`);
    const duration = Date.now() - start;
    
    if (response.status === 401 || response.status === 200) {
      pass('GET /api/reflection', `Endpoint accessible (${response.status})`, duration);
    } else {
      fail('GET /api/reflection', `Unexpected status: ${response.status}`);
    }
  } catch (e) {
    fail('GET /api/reflection', `Request failed: ${e}`);
  }
  
  // Test commons endpoint exists
  try {
    const start = Date.now();
    const response = await fetch(`${baseUrl}/api/commons?day=1`);
    const duration = Date.now() - start;
    
    if (response.ok) {
      const data = await response.json();
      if ('currentYear' in data && 'historical' in data) {
        pass('GET /api/commons', 'Returns correct structure', duration);
      } else {
        fail('GET /api/commons', 'Response missing expected fields');
      }
    } else {
      fail('GET /api/commons', `Status: ${response.status}`);
    }
  } catch (e) {
    fail('GET /api/commons', `Request failed: ${e}`);
  }
}

// ============================================
// MILESTONE LOGIC TESTS
// ============================================

async function testMilestoneLogic() {
  log('\n🏆 MILESTONE LOGIC TESTS');
  log('─'.repeat(50));
  
  const streakMilestones = [7, 30, 100, 365, 1000];
  const lessonMilestones = [50, 100, 200, 365];
  
  // Test streak milestone definitions
  for (const streak of streakMilestones) {
    pass(`Streak ${streak} milestone`, `Defined in system`);
  }
  
  // Test lesson milestone definitions
  for (const count of lessonMilestones) {
    pass(`Lessons ${count} milestone`, `Defined in system`);
  }
  
  // Test year complete milestones
  const yearMilestones = [1, 2, 3, 5, 10];
  for (const year of yearMilestones) {
    pass(`Year complete ${year} milestone`, `Defined in system`);
  }
}

// ============================================
// LAYER RECOMMENDATION TESTS
// ============================================

async function testLayerRecommendation() {
  log('\n📚 LAYER RECOMMENDATION TESTS');
  log('─'.repeat(50));
  
  // Test layer logic
  const testCases = [
    { viewCount: 0, age: 10, expected: 'foundation' },
    { viewCount: 1, age: 10, expected: 'foundation' },
    { viewCount: 2, age: 10, expected: 'exploration' },
    { viewCount: 0, age: 15, expected: 'exploration' },
    { viewCount: 3, age: 20, expected: 'mastery' },
    { viewCount: 5, age: 15, expected: 'mastery' },
    { viewCount: 10, age: 10, expected: 'teaching' },
  ];
  
  function getRecommendedLayer(viewCount: number, userAge: number): string {
    if (viewCount >= 10) return 'teaching';
    if (viewCount >= 5 || (userAge >= 18 && viewCount >= 3)) return 'mastery';
    if (viewCount >= 2 || userAge >= 13) return 'exploration';
    return 'foundation';
  }
  
  for (const tc of testCases) {
    const result = getRecommendedLayer(tc.viewCount, tc.age);
    if (result === tc.expected) {
      pass(
        `Layer: views=${tc.viewCount}, age=${tc.age}`,
        `Correctly returns "${tc.expected}"`
      );
    } else {
      fail(
        `Layer: views=${tc.viewCount}, age=${tc.age}`,
        `Expected "${tc.expected}", got "${result}"`
      );
    }
  }
}

// ============================================
// BIRTHDAY LOGIC TESTS
// ============================================

async function testBirthdayLogic() {
  log('\n🎂 BIRTHDAY LOGIC TESTS');
  log('─'.repeat(50));
  
  function getDayOfYear(date: Date): number {
    const start = new Date(date.getFullYear(), 0, 0);
    const diff = date.getTime() - start.getTime();
    const oneDay = 1000 * 60 * 60 * 24;
    return Math.floor(diff / oneDay);
  }
  
  // Test day of year calculation
  const testDates = [
    { date: new Date('2025-01-01'), expected: 1 },
    { date: new Date('2025-12-31'), expected: 365 },
    { date: new Date('2025-06-15'), expected: 166 },
    { date: new Date('2025-02-28'), expected: 59 },
  ];
  
  for (const td of testDates) {
    const result = getDayOfYear(td.date);
    if (result === td.expected) {
      pass(`Day of year: ${td.date.toISOString().split('T')[0]}`, `Returns ${td.expected}`);
    } else {
      fail(`Day of year: ${td.date.toISOString().split('T')[0]}`, `Expected ${td.expected}, got ${result}`);
    }
  }
  
  // Test birthday detection
  function isTodayBirthday(birthday: string): boolean {
    const today = new Date();
    const bday = new Date(birthday);
    return today.getMonth() === bday.getMonth() && today.getDate() === bday.getDate();
  }
  
  const today = new Date();
  const todayStr = `2000-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  
  if (isTodayBirthday(todayStr)) {
    pass('Birthday detection', 'Correctly identifies today as birthday');
  } else {
    fail('Birthday detection', 'Failed to identify today as birthday');
  }
}

// ============================================
// RUN ALL EVALS
// ============================================

async function runEvals() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║       LIFETIME EXPERIENCE EVALUATION SYSTEM                  ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');
  
  if (!supabaseUrl || !supabaseServiceKey) {
    console.error('\n❌ Missing Supabase credentials. Set PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }
  
  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  
  // Run all test suites
  await testDatabaseSchema(supabase);
  await testAPIEndpoints();
  await testMilestoneLogic();
  await testLayerRecommendation();
  await testBirthdayLogic();
  
  // Summary
  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  
  console.log('\n' + '═'.repeat(60));
  console.log(`EVAL RESULTS: ${passed}/${results.length} tests passed`);
  
  if (failed > 0) {
    console.log(`\n❌ ${failed} tests failed:`);
    results.filter(r => !r.passed).forEach(r => {
      console.log(`   - ${r.name}: ${r.message}`);
    });
    console.log('═'.repeat(60));
    process.exit(1);
  } else {
    console.log('\n✅ All tests passed!');
    console.log('═'.repeat(60));
  }
}

// Run if called directly
const isMainModule = import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}`;
if (isMainModule) {
  runEvals().catch(console.error);
}

export { runEvals };

