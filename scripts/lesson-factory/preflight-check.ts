#!/usr/bin/env npx tsx
/**
 * 🔍 PREFLIGHT CHECK
 * 
 * Verifies all systems are ready before running the Unified Lesson Factory.
 * Run this before overnight generation to catch any issues.
 * 
 * Usage:
 *   npx tsx scripts/lesson-factory/preflight-check.ts
 *   npx tsx scripts/lesson-factory/preflight-check.ts --day 1
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { S3Client, HeadBucketCommand } from '@aws-sdk/client-s3';

// Colors for terminal
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const BLUE = '\x1b[34m';
const RESET = '\x1b[0m';

interface CheckResult {
  name: string;
  status: 'pass' | 'fail' | 'warn';
  message: string;
  critical: boolean;
}

const results: CheckResult[] = [];

function pass(name: string, message: string, critical = true) {
  results.push({ name, status: 'pass', message, critical });
  console.log(`${GREEN}✅ ${name}${RESET}: ${message}`);
}

function fail(name: string, message: string, critical = true) {
  results.push({ name, status: 'fail', message, critical });
  console.log(`${RED}❌ ${name}${RESET}: ${message}`);
}

function warn(name: string, message: string) {
  results.push({ name, status: 'warn', message, critical: false });
  console.log(`${YELLOW}⚠️ ${name}${RESET}: ${message}`);
}

// =============================================================================
// CHECKS
// =============================================================================

async function checkEnvVars() {
  console.log(`\n${BLUE}📋 ENVIRONMENT VARIABLES${RESET}`);
  console.log('─'.repeat(50));
  
  const required = [
    { key: 'REPLICATE_API_TOKEN', name: 'Replicate' },
    { key: 'ELEVENLABS_API_KEY', name: 'ElevenLabs' },
    { key: 'PUBLIC_SUPABASE_URL', name: 'Supabase URL', alt: 'SUPABASE_URL' },
    { key: 'SUPABASE_SERVICE_ROLE_KEY', name: 'Supabase Key' },
  ];
  
  const optional = [
    { key: 'SYNC_LABS_API_KEY', name: 'Sync Labs (lipsync-2-pro)' },
    { key: 'CLOUDFLARE_ACCOUNT_ID', name: 'Cloudflare Account' },
    { key: 'CLOUDFLARE_R2_ACCESS_KEY_ID', name: 'R2 Access Key' },
    { key: 'CLOUDFLARE_R2_SECRET_ACCESS_KEY', name: 'R2 Secret Key' },
    { key: 'KELLY_ASSETS_BUCKET', name: 'R2 Bucket' },
    { key: 'GOOGLE_AI_API_KEY', name: 'Gemini API', alt: 'GEMINI_API_KEY' },
  ];
  
  for (const { key, name, alt } of required) {
    const value = process.env[key] || (alt ? process.env[alt] : undefined);
    if (value) {
      pass(name, `${key} configured (${value.substring(0, 8)}...)`);
    } else {
      fail(name, `${key} not set - REQUIRED`);
    }
  }
  
  console.log('');
  for (const { key, name, alt } of optional) {
    const value = process.env[key] || (alt ? process.env[alt] : undefined);
    if (value) {
      pass(name, `${key} configured`, false);
    } else {
      warn(name, `${key} not set - optional feature disabled`);
    }
  }
}

async function checkSupabase() {
  console.log(`\n${BLUE}🗄️ SUPABASE CONNECTION${RESET}`);
  console.log('─'.repeat(50));
  
  const url = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!url || !key) {
    fail('Connection', 'Missing credentials');
    return;
  }
  
  try {
    const supabase = createClient(url, key);
    
    // Test query
    const { data, error } = await supabase.from('core_lessons').select('count').limit(1);
    if (error) throw error;
    pass('Connection', 'Connected successfully');
    
    // Check tables
    const tables = ['core_lessons', 'lesson_atoms', 'lesson_assets'];
    for (const table of tables) {
      const { error: tableError } = await supabase.from(table).select('id').limit(1);
      if (tableError) {
        fail(`Table: ${table}`, tableError.message);
      } else {
        pass(`Table: ${table}`, 'Accessible', false);
      }
    }
    
    // Check storage buckets
    const buckets = ['kelly-videos', 'kelly-templates', 'lesson-visuals'];
    for (const bucket of buckets) {
      const { data: files, error: bucketError } = await supabase.storage.from(bucket).list('', { limit: 1 });
      if (bucketError) {
        warn(`Bucket: ${bucket}`, `Not accessible: ${bucketError.message}`);
      } else {
        pass(`Bucket: ${bucket}`, 'Accessible', false);
      }
    }
    
  } catch (error: any) {
    fail('Connection', error.message);
  }
}

async function checkReplicate() {
  console.log(`\n${BLUE}🎨 REPLICATE API${RESET}`);
  console.log('─'.repeat(50));
  
  if (!process.env.REPLICATE_API_TOKEN) {
    fail('API Key', 'Not configured');
    return;
  }
  
  try {
    const response = await fetch('https://api.replicate.com/v1/models/lucataco/flux-dev-lora', {
      headers: { 'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}` },
    });
    
    if (response.ok) {
      pass('Flux LoRA Model', 'Available');
    } else {
      fail('Flux LoRA Model', `HTTP ${response.status}`);
    }
    
    // Check MiniMax
    const minimaxRes = await fetch('https://api.replicate.com/v1/models/minimax/video-01', {
      headers: { 'Authorization': `Token ${process.env.REPLICATE_API_TOKEN}` },
    });
    
    if (minimaxRes.ok) {
      pass('MiniMax Model', 'Available');
    } else {
      warn('MiniMax Model', `HTTP ${minimaxRes.status} - video generation may fail`);
    }
    
  } catch (error: any) {
    fail('API Connection', error.message);
  }
}

async function checkElevenLabs() {
  console.log(`\n${BLUE}🎤 ELEVENLABS API${RESET}`);
  console.log('─'.repeat(50));
  
  if (!process.env.ELEVENLABS_API_KEY) {
    fail('API Key', 'Not configured');
    return;
  }
  
  try {
    const voiceId = process.env.ELEVENLABS_KELLY_VOICE_ID || 'wAdymQH5YucAkXwmrdL0';
    
    const response = await fetch(`https://api.elevenlabs.io/v1/voices/${voiceId}`, {
      headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
    });
    
    if (response.ok) {
      const voice = await response.json();
      pass('Kelly Voice', `Found: ${voice.name}`);
    } else if (response.status === 404) {
      fail('Kelly Voice', `Voice ID ${voiceId} not found`);
    } else {
      fail('Kelly Voice', `HTTP ${response.status}`);
    }
    
    // Check subscription/credits
    const userRes = await fetch('https://api.elevenlabs.io/v1/user/subscription', {
      headers: { 'xi-api-key': process.env.ELEVENLABS_API_KEY },
    });
    
    if (userRes.ok) {
      const user = await userRes.json();
      const remaining = user.character_limit - user.character_count;
      if (remaining > 10000) {
        pass('Credits', `${remaining.toLocaleString()} characters remaining`);
      } else {
        warn('Credits', `Only ${remaining.toLocaleString()} characters remaining`);
      }
    }
    
  } catch (error: any) {
    fail('API Connection', error.message);
  }
}

async function checkSyncLabs() {
  console.log(`\n${BLUE}👄 SYNC LABS API (lipsync-2-pro)${RESET}`);
  console.log('─'.repeat(50));
  
  if (!process.env.SYNC_LABS_API_KEY) {
    warn('API Key', 'Not configured - will use Wav2Lip fallback');
    return;
  }
  
  try {
    // No good health endpoint, just verify key format
    const key = process.env.SYNC_LABS_API_KEY;
    if (key.length > 20) {
      pass('API Key', 'Configured (format valid)');
      pass('Model', 'lipsync-2-pro (premium tier)');
    } else {
      warn('API Key', 'Key seems too short');
    }
  } catch (error: any) {
    warn('API Connection', error.message);
  }
}

async function checkCloudflareR2() {
  console.log(`\n${BLUE}☁️ CLOUDFLARE R2 BACKUP${RESET}`);
  console.log('─'.repeat(50));
  
  const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
  const accessKey = process.env.CLOUDFLARE_R2_ACCESS_KEY_ID;
  const secretKey = process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY;
  const bucket = process.env.KELLY_ASSETS_BUCKET;
  
  if (!accountId || !accessKey || !secretKey) {
    warn('Configuration', 'R2 not fully configured - backup disabled');
    return;
  }
  
  try {
    const s3 = new S3Client({
      region: 'auto',
      endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
      credentials: {
        accessKeyId: accessKey,
        secretAccessKey: secretKey,
      },
    });
    
    if (bucket) {
      await s3.send(new HeadBucketCommand({ Bucket: bucket }));
      pass('R2 Bucket', `${bucket} accessible`);
    } else {
      warn('R2 Bucket', 'KELLY_ASSETS_BUCKET not set');
    }
    
  } catch (error: any) {
    if (error.name === 'NotFound') {
      fail('R2 Bucket', `Bucket ${bucket} not found`);
    } else {
      warn('R2 Connection', error.message);
    }
  }
}

async function checkLessonData(dayNumber?: number) {
  console.log(`\n${BLUE}📚 LESSON DATA${RESET}`);
  console.log('─'.repeat(50));
  
  const url = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  
  if (!url || !key) {
    fail('Database', 'Cannot check - no credentials');
    return;
  }
  
  const supabase = createClient(url, key);
  
  if (dayNumber) {
    // Check specific day
    const { data: lesson, error } = await supabase
      .from('core_lessons')
      .select('id, topic, universal_truth')
      .eq('day_number', dayNumber)
      .single();
    
    if (error || !lesson) {
      fail(`Day ${dayNumber}`, 'Not found in database');
      return;
    }
    
    pass(`Day ${dayNumber}`, `"${lesson.topic}"`);
    
    // Check atoms
    const { data: atoms } = await supabase
      .from('lesson_atoms')
      .select('archetype, phase, content')
      .eq('core_lesson_id', lesson.id);
    
    if (atoms && atoms.length > 0) {
      const archetypes = new Set(atoms.map(a => a.archetype));
      const phases = new Set(atoms.map(a => a.phase));
      pass('Atoms', `${atoms.length} atoms (${archetypes.size} archetypes × ${phases.size} phases)`);
      
      // Check scripts
      const withScripts = atoms.filter(a => a.content?.script?.length > 0);
      if (withScripts.length === atoms.length) {
        pass('Scripts', `All ${atoms.length} atoms have scripts`);
      } else {
        warn('Scripts', `${withScripts.length}/${atoms.length} atoms have scripts`);
      }
      
      // Check options
      const withOptions = atoms.filter(a => a.content?.options?.length > 0);
      pass('Options', `${withOptions.length} atoms have options`, false);
      
    } else {
      fail('Atoms', 'No atoms found');
    }
    
  } else {
    // General count
    const { count } = await supabase
      .from('core_lessons')
      .select('*', { count: 'exact', head: true });
    
    pass('Total Lessons', `${count || 0} days in database`);
    
    const { count: atomCount } = await supabase
      .from('lesson_atoms')
      .select('*', { count: 'exact', head: true });
    
    pass('Total Atoms', `${atomCount || 0} atoms in database`, false);
  }
}

async function checkDiskSpace() {
  console.log(`\n${BLUE}💾 DISK SPACE${RESET}`);
  console.log('─'.repeat(50));
  
  // Estimate: ~50MB per archetype per day = 150MB per day
  // 7 days = ~1GB
  warn('Estimate', 'Each day generates ~150MB of assets');
  warn('Recommendation', 'Ensure at least 10GB free for overnight runs');
}

// =============================================================================
// MAIN
// =============================================================================

async function main() {
  console.log('\n');
  console.log('╔' + '═'.repeat(68) + '╗');
  console.log('║  🔍 UNIFIED LESSON FACTORY - PREFLIGHT CHECK'.padEnd(69) + '║');
  console.log('╚' + '═'.repeat(68) + '╝');
  
  // Parse args
  const args = process.argv.slice(2);
  let dayNumber: number | undefined;
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--day') {
      dayNumber = parseInt(args[++i]);
    }
  }
  
  // Run checks
  await checkEnvVars();
  await checkSupabase();
  await checkReplicate();
  await checkElevenLabs();
  await checkSyncLabs();
  await checkCloudflareR2();
  await checkLessonData(dayNumber);
  await checkDiskSpace();
  
  // Summary
  console.log('\n');
  console.log('═'.repeat(70));
  console.log(`${BLUE}📊 PREFLIGHT SUMMARY${RESET}`);
  console.log('═'.repeat(70));
  
  const passed = results.filter(r => r.status === 'pass').length;
  const failed = results.filter(r => r.status === 'fail').length;
  const warned = results.filter(r => r.status === 'warn').length;
  const criticalFails = results.filter(r => r.status === 'fail' && r.critical).length;
  
  console.log(`   ${GREEN}✅ Passed: ${passed}${RESET}`);
  console.log(`   ${YELLOW}⚠️ Warnings: ${warned}${RESET}`);
  console.log(`   ${RED}❌ Failed: ${failed}${RESET}`);
  
  if (criticalFails > 0) {
    console.log(`\n${RED}❌ CANNOT PROCEED - ${criticalFails} critical failures${RESET}`);
    console.log('\nFix these issues before running the factory:\n');
    results.filter(r => r.status === 'fail' && r.critical).forEach(r => {
      console.log(`   • ${r.name}: ${r.message}`);
    });
    process.exit(1);
  } else if (failed > 0) {
    console.log(`\n${YELLOW}⚠️ PROCEED WITH CAUTION - some checks failed${RESET}`);
    process.exit(0);
  } else {
    console.log(`\n${GREEN}✅ ALL SYSTEMS GO - Ready to generate!${RESET}`);
    console.log(`\nRun with:`);
    if (dayNumber) {
      console.log(`   npx tsx scripts/lesson-factory/unified-factory.ts --day ${dayNumber}`);
    } else {
      console.log(`   npx tsx scripts/lesson-factory/unified-factory.ts --day 1`);
    }
    process.exit(0);
  }
}

main().catch(error => {
  console.error(`\n${RED}❌ Preflight check error:${RESET}`, error);
  process.exit(1);
});
















