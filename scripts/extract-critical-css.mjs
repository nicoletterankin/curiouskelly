import { readdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import Critters from 'critters';

const distDir = path.resolve(process.cwd(), 'dist');

async function collectHtmlFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const resolved = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        return collectHtmlFiles(resolved);
      }

      if (entry.isFile() && entry.name.endsWith('.html')) {
        return [resolved];
      }

      return [];
    })
  );

  return files.flat();
}

async function inlineCriticalCss(file) {
  const critters = new Critters({
    path: distDir,
    pruneSource: false,
    preload: 'swap',
    inlineFonts: true,
    compress: true
  });
  const html = await readFile(file, 'utf-8');
  const transformed = await critters.process(html);
  await writeFile(file, transformed);
}

async function main() {
  try {
    const distStat = await stat(distDir);
    if (!distStat.isDirectory()) {
      console.warn('[critical-css] dist directory not found, skipping');
      return;
    }
  } catch {
    console.warn('[critical-css] dist directory missing, skipping');
    return;
  }

  const htmlFiles = await collectHtmlFiles(distDir);
  await Promise.all(htmlFiles.map((file) => inlineCriticalCss(file)));
  console.log(`[critical-css] Processed ${htmlFiles.length} HTML file(s)`);
}

main().catch((error) => {
  console.error('[critical-css] Failed to extract critical CSS', error);
  process.exitCode = 1;
});




