#!/usr/bin/env npx tsx
/**
 * 📥 DOWNLOAD HEYGEN VIDEOS FROM SUPABASE STORAGE
 *
 * Mission: pull HeyGen MP4s locally so we can run motion extraction.
 *
 * Defaults:
 * - bucket: kelly-videos
 * - prefix: heygen
 * - output: ./local/heygen-videos
 *
 * Usage:
 *   npx tsx scripts/download-heygen-videos.ts
 *   npx tsx scripts/download-heygen-videos.ts --prefix day-001 --output ./local/day-001-videos
 *   npx tsx scripts/download-heygen-videos.ts --dry-run
 *
 * Required env:
 * - PUBLIC_SUPABASE_URL (or SUPABASE_URL)
 * - SUPABASE_SERVICE_ROLE_KEY
 */

import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

type CliOptions = {
  bucket: string;
  prefix: string;
  output: string;
  dryRun: boolean;
  overwrite: boolean;
  flat: boolean;
};

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    bucket: 'kelly-videos',
    prefix: 'heygen',
    output: path.join(process.cwd(), 'local', 'heygen-videos'),
    dryRun: false,
    overwrite: false,
    flat: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--bucket') opts.bucket = argv[++i] || opts.bucket;
    else if (a === '--prefix') opts.prefix = argv[++i] || opts.prefix;
    else if (a === '--output') opts.output = argv[++i] || opts.output;
    else if (a === '--dry-run') opts.dryRun = true;
    else if (a === '--overwrite') opts.overwrite = true;
    else if (a === '--flat') opts.flat = true;
    else if (a === '--help' || a === '-h') {
      console.log(`
📥 Download HeyGen videos from Supabase Storage

Usage:
  npx tsx scripts/download-heygen-videos.ts [options]

Options:
  --bucket <name>      Storage bucket (default: kelly-videos)
  --prefix <path>      Prefix to scan (default: heygen)
  --output <dir>       Local output directory (default: ./local/heygen-videos)
  --flat               Flatten filenames to avoid nested folders
  --overwrite          Overwrite existing local files
  --dry-run            List what would be downloaded
  --help               Show help
`);
      process.exit(0);
    }
  }

  // Normalize prefix: remove leading/trailing slashes
  opts.prefix = opts.prefix.replace(/^\/+/, '').replace(/\/+$/, '');
  return opts;
}

function ensureDir(p: string) {
  fs.mkdirSync(p, { recursive: true });
}

function safeFileName(s: string) {
  return s
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
    .replace(/\s+/g, '_')
    .slice(0, 240);
}

function isProbablyFolder(name: string) {
  // Supabase Storage list() returns folders as items without "." in the name (common convention).
  // Not perfect, but matches existing repo scripts.
  return !name.includes('.');
}

async function listAllRecursive(
  supabase: ReturnType<typeof createClient>,
  bucket: string,
  prefix: string
): Promise<string[]> {
  const files: string[] = [];

  async function walk(dir: string) {
    // Pagination: offset is supported by storage.list
    const limit = 1000;
    let offset = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { data, error } = await supabase.storage.from(bucket).list(dir, {
        limit,
        offset,
        sortBy: { column: 'name', order: 'asc' },
      });
      if (error) throw error;

      const items = data || [];
      for (const item of items) {
        const fullPath = dir ? `${dir}/${item.name}` : item.name;
        if (isProbablyFolder(item.name)) {
          await walk(fullPath);
        } else {
          files.push(fullPath);
        }
      }

      if (items.length < limit) break;
      offset += limit;
    }
  }

  await walk(prefix);
  return files;
}

async function downloadToFile(
  supabase: ReturnType<typeof createClient>,
  bucket: string,
  remotePath: string,
  localPath: string,
  overwrite: boolean
) {
  if (!overwrite && fs.existsSync(localPath)) return 'skipped_exists' as const;

  const { data, error } = await supabase.storage.from(bucket).download(remotePath);
  if (error) throw error;
  if (!data) throw new Error(`No data returned for download: ${remotePath}`);

  const buf = Buffer.from(await data.arrayBuffer());
  ensureDir(path.dirname(localPath));
  fs.writeFileSync(localPath, buf);
  return 'downloaded' as const;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !serviceKey) {
    console.error('❌ Missing Supabase credentials');
    console.error('   Required: PUBLIC_SUPABASE_URL (or SUPABASE_URL) + SUPABASE_SERVICE_ROLE_KEY');
    process.exit(1);
  }

  const supabase = createClient(supabaseUrl, serviceKey);

  console.log('═'.repeat(72));
  console.log('📥 DOWNLOAD HEYGEN VIDEOS');
  console.log('═'.repeat(72));
  console.log(`Bucket: ${opts.bucket}`);
  console.log(`Prefix: ${opts.prefix}`);
  console.log(`Output: ${opts.output}`);
  console.log(`Mode: ${opts.dryRun ? 'DRY RUN' : 'DOWNLOAD'}`);
  console.log(`Flat: ${opts.flat ? 'yes' : 'no'}`);
  console.log(`Overwrite: ${opts.overwrite ? 'yes' : 'no'}`);

  ensureDir(opts.output);

  console.log('\n📁 Listing files (recursive)...');
  const allPaths = await listAllRecursive(supabase, opts.bucket, opts.prefix);
  const mp4s = allPaths.filter((p) => p.toLowerCase().endsWith('.mp4'));

  console.log(`   Found ${allPaths.length} total objects`);
  console.log(`   Found ${mp4s.length} MP4 videos`);

  if (opts.dryRun) {
    console.log('\n🔎 DRY RUN (first 25 MP4s):');
    for (const p of mp4s.slice(0, 25)) console.log(`   - ${p}`);
    if (mp4s.length > 25) console.log(`   ... and ${mp4s.length - 25} more`);
    return;
  }

  let downloaded = 0;
  let skipped = 0;
  let failed = 0;

  for (let i = 0; i < mp4s.length; i++) {
    const remotePath = mp4s[i];

    const rel = remotePath.startsWith(`${opts.prefix}/`)
      ? remotePath.slice(opts.prefix.length + 1)
      : remotePath;

    const localFile = opts.flat
      ? safeFileName(rel.replaceAll('/', '__'))
      : rel.split('/').map(safeFileName).join(path.sep);

    const localPath = path.join(opts.output, localFile);

    try {
      const result = await downloadToFile(supabase, opts.bucket, remotePath, localPath, opts.overwrite);
      if (result === 'downloaded') {
        downloaded++;
        console.log(`✅ [${i + 1}/${mp4s.length}] ${remotePath} → ${path.relative(process.cwd(), localPath)}`);
      } else {
        skipped++;
        console.log(`⏭️  [${i + 1}/${mp4s.length}] exists, skipping → ${path.relative(process.cwd(), localPath)}`);
      }
    } catch (e: any) {
      failed++;
      console.error(`❌ [${i + 1}/${mp4s.length}] failed: ${remotePath}`);
      console.error(`   ${e?.message || e}`);
    }
  }

  console.log('\n' + '═'.repeat(72));
  console.log('📦 DOWNLOAD REPORT');
  console.log('═'.repeat(72));
  console.log(`✅ Downloaded: ${downloaded}`);
  console.log(`⏭️  Skipped:    ${skipped}`);
  console.log(`❌ Failed:     ${failed}`);
  console.log(`📁 Output dir: ${opts.output}`);
}

main().catch((e) => {
  console.error('❌ Fatal:', e?.message || e);
  process.exit(1);
});


