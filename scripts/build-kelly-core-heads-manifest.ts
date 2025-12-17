#!/usr/bin/env npx tsx
/**
 * Build a locked, checksummed manifest of the "Core 60" Kelly head assets.
 * Core 60 = 12 archetypes × 5 ages (kid, teen, adult, elder, super_elder)
 *
 * Output:
 *  - generated-images/kelly-archetypes-head-only/core-60.manifest.json
 */

import * as fs from 'fs';
import * as path from 'path';
import { createHash } from 'crypto';

const CDN_BASE = 'https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates';

const CORE_AGES = ['kid', 'teen', 'adult', 'elder', 'super_elder'] as const;
type CoreAge = (typeof CORE_AGES)[number];

const ARCHETYPES = [
  'scientist',
  'explorer',
  'rebel',
  'architect',
  'diplomat',
  'empath',
  'macgyver',
  'mystic',
  'provider',
  'storyteller',
  'strategist',
  'survivor'
] as const;
type Archetype = (typeof ARCHETYPES)[number];

const repoRoot = process.cwd();
const baseDir = path.join(repoRoot, 'generated-images', 'kelly-archetypes-head-only');

function sha256File(filePath: string): string {
  const buf = fs.readFileSync(filePath);
  return createHash('sha256').update(buf).digest('hex');
}

function getLocalPath(age: CoreAge, archetype: Archetype): string {
  // Adult assets are stored locally under age/adult for consistency.
  return path.join(baseDir, 'age', age, `kelly_${archetype}_head.png`);
}

function getSupabaseObjectPath(age: CoreAge, archetype: Archetype): string {
  // Adult is the historical/base path in Supabase (no /age/adult/).
  if (age === 'adult') return `heygen/archetypes-head-only/kelly_${archetype}_head.png`;
  return `heygen/archetypes-head-only/age/${age}/kelly_${archetype}_head.png`;
}

function assertExists(filePath: string) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing file: ${filePath}`);
  }
}

function main() {
  const generatedAt = new Date().toISOString();

  const entries: Array<{
    age: CoreAge;
    archetype: Archetype;
    localPath: string;
    bytes: number;
    sha256: string;
    supabaseObjectPath: string;
    url: string;
  }> = [];

  for (const age of CORE_AGES) {
    for (const archetype of ARCHETYPES) {
      const localPath = getLocalPath(age, archetype);
      assertExists(localPath);

      const stat = fs.statSync(localPath);
      const sha256 = sha256File(localPath);
      const supabaseObjectPath = getSupabaseObjectPath(age, archetype);
      const url = `${CDN_BASE}/${supabaseObjectPath}`;

      entries.push({
        age,
        archetype,
        localPath: path.relative(repoRoot, localPath).replace(/\\/g, '/'),
        bytes: stat.size,
        sha256,
        supabaseObjectPath,
        url
      });
    }
  }

  const outPath = path.join(baseDir, 'core-60.manifest.json');
  const payload = {
    version: 1,
    generatedAt,
    concept: 'Kelly head-only archetypes (locked framing, white studio).',
    coreAges: CORE_AGES,
    archetypes: ARCHETYPES,
    cdnBase: CDN_BASE,
    count: entries.length,
    entries
  };

  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2) + '\n', 'utf8');
  // eslint-disable-next-line no-console
  console.log(`Wrote ${entries.length} entries: ${outPath}`);
}

main();







