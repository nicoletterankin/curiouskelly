#!/usr/bin/env npx tsx
/**
 * Scans the kelly-videos storage bucket and guarantees every MP4 has a
 * corresponding kelly_video_assets row.
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY;
const KELLY_VIDEO_BUCKET = process.env.KELLY_VIDEO_BUCKET || 'kelly-videos';
const PAGE_SIZE = 1000;
const DRY_RUN = process.argv.includes('--dry-run');

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('❌ Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

type ParsedVideo = {
  storagePath: string;
  dayNumber: number;
  phase: string;
  archetypeSlug: string;
  template: string;
  language: string;
  ageBucket: string | null;
  qualityTier: string;
};

const ARCHETYPE_SLUG_TO_FULL: Record<string, string> = {
  scientist: 'The Scientist',
  explorer: 'The Explorer',
  rebel: 'The Rebel',
  architect: 'The Architect',
  diplomat: 'The Diplomat',
  empath: 'The Empath',
  macgyver: 'The MacGyver',
  mystic: 'The Mystic',
  provider: 'The Provider',
  storyteller: 'The Storyteller',
  strategist: 'The Strategist',
  survivor: 'The Survivor',
};

const ARCHETYPE_TEMPLATE_MAP: Record<string, string> = {
  'The Scientist': 'explaining',
  'The Explorer': 'curious',
  'The Rebel': 'confident',
  'The Architect': 'explaining',
  'The Diplomat': 'warm',
  'The Empath': 'warm',
  'The MacGyver': 'practical',
  'The Mystic': 'reflective',
  'The Provider': 'warm',
  'The Storyteller': 'expressive',
  'The Strategist': 'focused',
  'The Survivor': 'grounded',
};

const LANGUAGE_CODES = ['en', 'es', 'fr', 'pt', 'de', 'it', 'ja', 'ko'];
const AGE_BUCKETS = ['kid', 'teen', 'adult', 'mature', 'elder', 'young', 'young_adult'];

const PHASE_PATTERNS: Array<{ regex: RegExp; phase: string }> = [
  { regex: /hook/i, phase: 'hook' },
  { regex: /cliff/i, phase: 'cliff' },
  { regex: /fact1|q1/i, phase: 'q1' },
  { regex: /fact2|q2/i, phase: 'q2' },
  { regex: /fact3|q3/i, phase: 'q3' },
  { regex: /wisdom/i, phase: 'wisdom' },
  { regex: /outro|closing/i, phase: 'outro' },
];

async function listAllFiles(): Promise<string[]> {
  const results: string[] = [];
  const queue = [''];

  while (queue.length > 0) {
    const prefix = queue.pop()!;
    const { data, error } = await supabase.storage.from(KELLY_VIDEO_BUCKET).list(prefix, {
      limit: PAGE_SIZE,
      sortBy: { column: 'name', order: 'asc' },
    });

    if (error) {
      throw new Error(`Storage list failed for "${prefix}": ${error.message}`);
    }

    for (const entry of data || []) {
      if (!entry.name || entry.name === '.' || entry.name === '..') continue;
      const fullPath = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.metadata && typeof entry.metadata.size === 'number') {
        results.push(fullPath);
      } else {
        queue.push(fullPath);
      }
    }
  }

  return results;
}

function detectPhase(path: string): string | null {
  for (const pattern of PHASE_PATTERNS) {
    if (pattern.regex.test(path)) {
      return pattern.phase;
    }
  }
  return null;
}

function detectLanguage(path: string): string {
  for (const code of LANGUAGE_CODES) {
    const regex = new RegExp(`[-_.]${code}[-_.]`, 'i');
    if (regex.test(path)) {
      return code;
    }
  }
  return 'en';
}

function detectAgeBucket(path: string): string | null {
  for (const age of AGE_BUCKETS) {
    const regex = new RegExp(`[-_.]${age}[-_.]`, 'i');
    if (regex.test(path)) {
      if (age === 'young') return 'teen';
      return age === 'young_adult' ? 'adult' : age;
    }
  }
  if (/kid|child/i.test(path)) return 'kid';
  if (/elder/i.test(path)) return 'elder';
  return null;
}

function detectArchetype(path: string): string {
  const lower = path.toLowerCase();
  for (const slug of Object.keys(ARCHETYPE_SLUG_TO_FULL)) {
    if (lower.includes(slug)) {
      return slug;
    }
  }
  if (lower.includes('sadtalker')) return 'sadtalker';
  if (lower.includes('iclone')) return 'iclone';
  if (lower.includes('kelly')) return 'kelly';
  return 'default';
}

function slugToTemplate(slug: string): string {
  const full = ARCHETYPE_SLUG_TO_FULL[slug];
  if (full && ARCHETYPE_TEMPLATE_MAP[full]) {
    return ARCHETYPE_TEMPLATE_MAP[full];
  }
  return slug;
}

function detectDayNumber(path: string): number | null {
  const match = path.match(/day[-_]?([0-9]{1,3})/i);
  if (!match) return null;
  return parseInt(match[1], 10);
}

function determineQualityTier(path: string): string {
  if (path.includes('/raw/') || path.includes('/sadtalker/') || path.includes('/iclone/')) {
    return 'standard';
  }
  return 'production';
}

function parseVideo(storagePath: string): ParsedVideo | null {
  if (!storagePath.toLowerCase().endsWith('.mp4')) {
    return null;
  }

  const dayNumber = detectDayNumber(storagePath);
  const phase = detectPhase(storagePath);
  const archetypeSlug = detectArchetype(storagePath);

  if (!dayNumber || !phase) {
    return null;
  }

  const template = slugToTemplate(archetypeSlug);
  const language = detectLanguage(storagePath);
  const ageBucket = detectAgeBucket(storagePath) || (archetypeSlug === 'sadtalker' ? 'adult' : null);
  const qualityTier = determineQualityTier(storagePath);

  return {
    storagePath,
    dayNumber,
    phase,
    archetypeSlug,
    template,
    language,
    ageBucket,
    qualityTier,
  };
}

async function fetchExistingPaths(columns: string[]): Promise<Set<string>> {
  const existing = new Set<string>();
  let from = 0;

  while (true) {
    const { data, error } = await supabase
      .from('kelly_video_assets')
      .select(columns.join(', '))
      .eq('asset_type', 'video')
      .range(from, from + PAGE_SIZE - 1);

    if (error) {
      throw new Error(error.message);
    }

    if (!data || data.length === 0) {
      break;
    }

    for (const row of data) {
      const path = (row as any).storage_path || (row as any).video_storage_path;
      if (path) {
        existing.add(path);
      }
    }

    if (data.length < PAGE_SIZE) {
      break;
    }

    from += PAGE_SIZE;
  }

  return existing;
}

async function loadExistingStoragePaths(): Promise<Set<string>> {
  try {
    return await fetchExistingPaths(['storage_path', 'video_storage_path']);
  } catch (err: any) {
    const message = String(err?.message || err);
    if (message.includes('video_storage_path')) {
      console.warn('⚠️  video_storage_path column missing, falling back to storage_path only');
      return await fetchExistingPaths(['storage_path']);
    }
    throw new Error(`Failed to load existing assets: ${message}`);
  }
}

async function ensureVideoRegistered(meta: ParsedVideo, existingPaths: Set<string>) {
  if (existingPaths.has(meta.storagePath)) {
    return { status: 'existing' as const };
  }

  const publicUrl = supabase.storage.from(KELLY_VIDEO_BUCKET).getPublicUrl(meta.storagePath).data?.publicUrl;
  if (!publicUrl) {
    return { status: 'error' as const, reason: 'public_url_missing' };
  }

  if (DRY_RUN) {
    existingPaths.add(meta.storagePath);
    return { status: 'created' as const };
  }

  const insertPayload: Record<string, unknown> = {
    day_number: meta.dayNumber,
    phase: meta.phase,
    template: meta.template,
    asset_type: 'video',
    age_bucket: meta.ageBucket,
    language: meta.language,
    storage_bucket: KELLY_VIDEO_BUCKET,
    storage_path: meta.storagePath,
    public_url: publicUrl,
    status: 'validated',
    quality_tier: meta.qualityTier,
    resolution: '1920x1080',
  };

  const { error } = await supabase.from('kelly_video_assets').insert(insertPayload);
  if (error) {
    const message = error.message || '';
    if (message.includes('duplicate key value')) {
      existingPaths.add(meta.storagePath);
      return { status: 'existing' as const };
    }
    return { status: 'error' as const, reason: message };
  }

  existingPaths.add(meta.storagePath);
  return { status: 'created' as const };
}

async function main() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║  🔄 Sync kelly-videos bucket → kelly_video_assets     ║');
  console.log('╚══════════════════════════════════════════════════════╝');
  console.log(`Bucket: ${KELLY_VIDEO_BUCKET}`);
  console.log(`Mode: ${DRY_RUN ? 'DRY RUN' : 'LIVE'}`);

  const files = await listAllFiles();
  console.log(`Found ${files.length} objects in bucket`);

  const existingPaths = await loadExistingStoragePaths();
  console.log(`Existing video asset rows: ${existingPaths.size}`);

  let mp4Count = 0;
  let created = 0;
  let alreadyPresent = 0;
  let parseFailures = 0;
  let errors = 0;
  const failedSamples: string[] = [];

  for (const path of files) {
    if (!path.toLowerCase().endsWith('.mp4')) continue;
    mp4Count++;
    const parsed = parseVideo(path);
    if (!parsed) {
      parseFailures++;
      if (failedSamples.length < 10) failedSamples.push(path);
      continue;
    }

    const result = await ensureVideoRegistered(parsed, existingPaths);
    if (result.status === 'existing') {
      alreadyPresent++;
    } else if (result.status === 'created') {
      created++;
    } else {
      errors++;
      if (failedSamples.length < 10) failedSamples.push(`${path} → ${result.reason}`);
    }
  }

  console.log('\nSummary:');
  console.log(`  MP4 files scanned: ${mp4Count}`);
  console.log(`  New rows ${DRY_RUN ? 'pending' : 'created'}: ${created}`);
  console.log(`  Already registered: ${alreadyPresent}`);
  console.log(`  Unparsed files: ${parseFailures}`);
  console.log(`  Errors: ${errors}`);

  if (failedSamples.length > 0) {
    console.log('\nSamples:');
    for (const sample of failedSamples) {
      console.log(`  - ${sample}`);
    }
  }

  if (DRY_RUN) {
    console.log('\nRun without --dry-run to apply changes.');
  }
}

main().catch(err => {
  console.error('❌ Sync failed:', err);
  process.exit(1);
});
