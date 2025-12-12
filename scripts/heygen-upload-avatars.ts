#!/usr/bin/env npx tsx
/**
 * 🎨 HEYGEN AVATAR UPLOADER
 * 
 * Uploads the 12 Kelly archetype images to HeyGen as Photo Avatars.
 * Returns the talking_photo_id for each, ready for video generation.
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const CONFIG = {
  HEYGEN_API_KEY: process.env.HEYGEN_API_KEY!,
};

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
  'survivor',
];

function titleCase(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function getArgValue(flag: string): string | null {
  const idx = process.argv.indexOf(flag);
  if (idx === -1) return null;
  const val = process.argv[idx + 1];
  return val && !val.startsWith('--') ? val : null;
}

function resolveImagesDir(): { imagesDir: string; label: string; isHeadOnly: boolean } {
  // Usage examples:
  // - Default (legacy): generated-images/kelly-archetypes-lora/kelly_archetype_<arch>.png
  //   npx tsx scripts/heygen-upload-avatars.ts
  //
  // - Head-only age variants (recommended for your 36-base-head system):
  //   npx tsx scripts/heygen-upload-avatars.ts --head-only --age kid
  //   npx tsx scripts/heygen-upload-avatars.ts --head-only --age teen
  //   npx tsx scripts/heygen-upload-avatars.ts --head-only --age adult
  //   npx tsx scripts/heygen-upload-avatars.ts --head-only --age elder
  //   npx tsx scripts/heygen-upload-avatars.ts --head-only --age super_elder
  //
  // - Explicit directory override:
  //   npx tsx scripts/heygen-upload-avatars.ts --dir "generated-images/kelly-archetypes-head-only/age/mature"
  const explicitDir = getArgValue('--dir') || getArgValue('--images-dir') || getArgValue('--imagesDir');
  const isHeadOnly = process.argv.includes('--head-only') || process.argv.includes('--headOnly');
  const age = getArgValue('--age');

  if (explicitDir) {
    return { imagesDir: path.isAbsolute(explicitDir) ? explicitDir : path.join(process.cwd(), explicitDir), label: 'custom', isHeadOnly };
  }

  if (isHeadOnly) {
    if (!age) {
      throw new Error('Missing --age when using --head-only (expected: kid|teen|adult|elder|super_elder)');
    }
    const allowed = new Set(['kid', 'teen', 'adult', 'elder', 'super_elder']);
    if (!allowed.has(age)) {
      throw new Error(`Invalid --age ${age} (expected: kid|teen|adult|elder|super_elder)`);
    }
    return {
      imagesDir: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-head-only', 'age', age),
      label: `age:${age}`,
      isHeadOnly: true,
    };
  }

  return {
    imagesDir: path.join(process.cwd(), 'generated-images', 'kelly-archetypes-lora'),
    label: 'lora',
    isHeadOnly: false,
  };
}

function getImagePath(imagesDir: string, archetype: string, isHeadOnly: boolean) {
  // legacy
  if (!isHeadOnly) return path.join(imagesDir, `kelly_archetype_${archetype}.png`);
  // head-only (age variant directories)
  return path.join(imagesDir, `kelly_${archetype}_head.png`);
}

function loadUrlManifest(imagesDir: string): Record<string, string> {
  // Prefer URL manifests that were created by our generators (avoid base64 uploads).
  const candidates = [
    path.join(imagesDir, 'archetype_head_urls.json'),
    path.join(imagesDir, 'archetype_urls.json'),
  ];

  for (const p of candidates) {
    if (!fs.existsSync(p)) continue;
    try {
      const raw = JSON.parse(fs.readFileSync(p, 'utf-8'));
      // head-only format: { archetype: { url, description } }
      if (raw && typeof raw === 'object') {
        const out: Record<string, string> = {};
        for (const [k, v] of Object.entries(raw)) {
          if (typeof v === 'string') {
            out[k] = v;
          } else if (v && typeof v === 'object' && typeof (v as any).url === 'string') {
            out[k] = (v as any).url;
          }
        }
        if (Object.keys(out).length > 0) return out;
      }
    } catch {
      // ignore and continue
    }
  }

  return {};
}

async function postJson(url: string, body: any) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'X-Api-Key': CONFIG.HEYGEN_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  let json: any = null;
  try {
    json = JSON.parse(text);
  } catch {
    // ignore
  }
  return { ok: response.ok, status: response.status, text, json };
}

async function uploadToHeyGen(imageUrl: string, name: string): Promise<string | null> {
  console.log(`\n📤 Uploading: ${name}`);
  
  // HeyGen APIs differ by account/version. We try multiple known-good endpoints.
  try {
    // 1) Most consistent: v1/talking_photo.add (expects image_url)
    {
      const res = await postJson('https://api.heygen.com/v1/talking_photo.add', {
        image_url: imageUrl,
        name,
      });
      if (res.ok) {
        const id = res.json?.data?.talking_photo_id || res.json?.data?.id || res.json?.data;
        console.log(`   ✅ Uploaded via v1/talking_photo.add! ID: ${id}`);
        return id || null;
      }
      console.log(`   ⚠️ v1/talking_photo.add failed (${res.status}): ${res.text.substring(0, 200)}`);
    }

    // 2) v2 photo avatar generate (also expects image_url on many accounts)
    {
      const res = await postJson('https://api.heygen.com/v2/photo_avatar/generate', {
        image_url: imageUrl,
        name,
      });
      if (res.ok) {
        const id = res.json?.data?.talking_photo_id || res.json?.data?.id || res.json?.data;
        console.log(`   ✅ Uploaded via v2/photo_avatar/generate! ID: ${id}`);
        return id || null;
      }
      console.log(`   ⚠️ v2/photo_avatar/generate failed (${res.status}): ${res.text.substring(0, 200)}`);
    }

    // 3) v2 talking photo (some accounts)
    {
      const res = await postJson('https://api.heygen.com/v2/talking_photo', {
        image_url: imageUrl,
        name,
      });
      if (res.ok) {
        const id = res.json?.data?.talking_photo_id || res.json?.data?.id || res.json?.data;
        console.log(`   ✅ Uploaded via v2/talking_photo! ID: ${id}`);
        return id || null;
      }
      console.log(`   ⚠️ v2/talking_photo failed (${res.status}): ${res.text.substring(0, 200)}`);
    }

  } catch (error: any) {
    console.error(`   ❌ Error: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  🎨 HEYGEN KELLY AVATAR UPLOADER                           ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  if (!CONFIG.HEYGEN_API_KEY) {
    console.error('❌ HEYGEN_API_KEY not found in environment');
    process.exit(1);
  }

  let imagesDirInfo: { imagesDir: string; label: string; isHeadOnly: boolean };
  try {
    imagesDirInfo = resolveImagesDir();
  } catch (e: any) {
    console.error(`❌ ${e.message}`);
    process.exit(1);
  }

  const { imagesDir, label, isHeadOnly } = imagesDirInfo;
  console.log(`\n📁 Images: ${imagesDir}`);
  console.log(`🏷️  Mode: ${label}${isHeadOnly ? ' (head-only)' : ''}`);

  // Check for images
  if (!fs.existsSync(imagesDir)) {
    console.error(`❌ Images directory not found: ${imagesDir}`);
    if (!isHeadOnly) console.error('Run generate-with-trained-lora.ts or generate-12-kellys-with-lora.ts first!');
    if (isHeadOnly) console.error('Run generate-12-kellys-head-accessories.ts --ages=mature,elder first!');
    process.exit(1);
  }

  const urlManifest = loadUrlManifest(imagesDir);
  const hasUrls = Object.keys(urlManifest).length > 0;
  if (!hasUrls) {
    console.warn('⚠️ No URL manifest found (archetype_head_urls.json / archetype_urls.json).');
    console.warn('   This uploader expects already-uploaded image URLs. Re-run the generator (it uploads to Supabase and writes a manifest).');
  }

  const results: Record<string, string | null> = {};

  for (const archetype of ARCHETYPES) {
    const imageUrl = urlManifest[archetype];
    if (!imageUrl || !imageUrl.startsWith('http')) {
      console.log(`⚠️ URL not found for: ${archetype}`);
      results[archetype] = null;
      continue;
    }

    const displayName = isHeadOnly
      ? `Kelly ${titleCase(archetype)} (${label})`
      : `Kelly ${titleCase(archetype)}`;
    const avatarId = await uploadToHeyGen(imageUrl, displayName);
    results[archetype] = avatarId;

    // Rate limit
    await new Promise(r => setTimeout(r, 2000));
  }

  // Summary
  console.log('\n\n' + '═'.repeat(60));
  console.log('📋 TALKING PHOTO IDS - Copy into pipeline mapping');
  console.log('═'.repeat(60));
  console.log(`\n// Source: ${label}`);
  console.log('const AVATAR_MAP: Record<string, string> = {');
  
  for (const [archetype, id] of Object.entries(results)) {
    const formattedName = `The ${titleCase(archetype)}`;
    if (id) {
      console.log(`  "${formattedName}": "${id}",`);
    } else {
      console.log(`  "${formattedName}": "UPLOAD_FAILED",`);
    }
  }
  
  console.log('};');

  // Save to file
  const outputPath = path.join(imagesDir, 'heygen_talking_photo_ids.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`\n💾 Saved to: ${outputPath}`);
}

main().catch(console.error);

