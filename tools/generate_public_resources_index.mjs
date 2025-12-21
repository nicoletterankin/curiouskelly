/**
 * Generate a searchable index of "listable" files under /public for the homepage Resource Explorer.
 *
 * Goals:
 * - Include useful resources (pages, dashboards, JSON, scripts, docs).
 * - Exclude heavy binaries (images/video/audio) to keep the JSON small.
 * - Extract <title> from HTML for nicer display.
 *
 * Run:
 *   node tools/generate_public_resources_index.mjs
 */

import fs from 'node:fs/promises';
import path from 'node:path';

const PUBLIC_DIR = path.resolve(process.cwd(), 'public');
const OUT_FILE = path.resolve(PUBLIC_DIR, 'resources-index.json');

/** @type {Set<string>} */
const ALLOWED_EXT = new Set([
  '.html',
  '.json',
  '.js',
  '.css',
  '.md',
  '.xml',
  '.txt',
  '.svg'
]);

/**
 * We still traverse the filesystem even for excluded extensions, so we must avoid
 * extremely large media folders (audio/video/images) for performance.
 * Everything else is fair game (admin dashboards, tools, JSON, HTML, etc).
 * @type {Set<string>}
 */
const IGNORE_DIRS = new Set([
  'audio',
  'video',
  'videos'
]);

function isAllowedFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return ALLOWED_EXT.has(ext);
}

function toPublicUrl(absPath) {
  const rel = path.relative(PUBLIC_DIR, absPath).split(path.sep).join('/');
  return '/' + rel;
}

function categorize(publicPath) {
  const p = publicPath.toLowerCase();
  if (p.startsWith('/admin/')) return 'admin';
  if (p.startsWith('/data/') || p.startsWith('/lessons/') || p.endsWith('.json')) return 'data';
  if (p.startsWith('/js/') || p.startsWith('/css/') || p.endsWith('.js') || p.endsWith('.css')) return 'code';
  if (p.endsWith('.html') || p.endsWith('/')) return 'pages';
  return 'other';
}

function inferTags(publicPath) {
  const tags = new Set();
  const p = publicPath.toLowerCase();
  if (p.includes('dashboard')) tags.add('dashboard');
  if (p.includes('test-') || p.includes('/test')) tags.add('test');
  if (p.includes('debug')) tags.add('debug');
  if (p.includes('golden')) tags.add('golden');
  if (p.includes('stripe')) tags.add('stripe');
  if (p.includes('supabase')) tags.add('supabase');
  if (p.includes('calendar')) tags.add('calendar');
  if (p.includes('lesson')) tags.add('lesson');
  if (p.includes('mission-control') || p.includes('command-center')) tags.add('ops');
  if (p.includes('translation')) tags.add('translation');
  return Array.from(tags);
}

async function extractHtmlTitle(absPath) {
  try {
    const raw = await fs.readFile(absPath, 'utf8');
    // very lightweight title extraction
    const m = raw.match(/<title>\s*([^<]+?)\s*<\/title>/i);
    if (!m) return null;
    return m[1].trim().replace(/\s+/g, ' ');
  } catch {
    return null;
  }
}

async function walk(dirAbs, relParts = []) {
  const entries = await fs.readdir(dirAbs, { withFileTypes: true });
  /** @type {string[]} */
  const files = [];

  for (const ent of entries) {
    const nextAbs = path.join(dirAbs, ent.name);
    const nextRelParts = [...relParts, ent.name];

    if (ent.isDirectory()) {
      // Skip ignored top-level dirs (but allow admin/data/etc)
      if (relParts.length === 0 && IGNORE_DIRS.has(ent.name)) continue;
      const nested = await walk(nextAbs, nextRelParts);
      files.push(...nested);
      continue;
    }

    if (ent.isFile()) {
      const fileRel = nextRelParts.join('/');
      files.push(fileRel);
    }
  }

  return files;
}

async function main() {
  const relFiles = await walk(PUBLIC_DIR);

  const absFiles = relFiles
    .map((rel) => path.join(PUBLIC_DIR, rel))
    .filter((abs) => isAllowedFile(abs));

  /** @type {{ path: string; title?: string; category: string; tags?: string[] }[]} */
  const items = [];

  for (const abs of absFiles) {
    const p = toPublicUrl(abs);
    const category = categorize(p);

    let title = undefined;
    if (p.toLowerCase().endsWith('.html')) {
      const t = await extractHtmlTitle(abs);
      if (t) title = t;
    }

    const tags = inferTags(p);
    items.push({
      path: p,
      title,
      category,
      tags: tags.length ? tags : undefined
    });
  }

  // Sort: category then title/path
  items.sort((a, b) => {
    const ac = a.category.localeCompare(b.category);
    if (ac !== 0) return ac;
    const at = (a.title || a.path).localeCompare(b.title || b.path);
    if (at !== 0) return at;
    return a.path.localeCompare(b.path);
  });

  const out = {
    version: '1.0.0',
    generatedAt: new Date().toISOString(),
    itemCount: items.length,
    items
  };

  await fs.writeFile(OUT_FILE, JSON.stringify(out, null, 2) + '\n', 'utf8');
  // eslint-disable-next-line no-console
  console.log(`Wrote ${items.length} items to ${path.relative(process.cwd(), OUT_FILE)}`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exitCode = 1;
});


