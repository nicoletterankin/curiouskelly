import { readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

const distDir = path.resolve(process.cwd(), 'dist');
const siteUrl = process.env.PUBLIC_SITE_URL ?? 'https://www.curiouskelly.com';
const locales = [
  { code: 'en-US', prefix: '' },
  { code: 'es-ES', prefix: 'es-es/' },
  { code: 'pt-BR', prefix: 'pt-br/' }
];

function resolveLocale(pathname) {
  const [, firstSegment] = pathname.split('/');
  const locale = locales.find((localeEntry) => localeEntry.prefix.replace(/\/$/, '') === firstSegment);
  return locale ?? locales[0];
}

function stripLocalePrefix(pathname) {
  for (const locale of locales) {
    if (locale.prefix && pathname.startsWith(`/${locale.prefix}`)) {
      return pathname.slice(locale.prefix.length + 1);
    }
  }
  return pathname.startsWith('/') ? pathname.slice(1) : pathname;
}

async function collectHtmlFiles(dir, base = '') {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const resolved = path.join(dir, entry.name);
    const relative = path.join(base, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectHtmlFiles(resolved, relative)));
      continue;
    }

    if (entry.isFile() && entry.name.endsWith('.html')) {
      files.push(relative);
    }
  }

  return files;
}

async function ensureDistDir() {
  try {
    const distStat = await stat(distDir);
    if (!distStat.isDirectory()) {
      throw new Error('dist is not a directory');
    }
  } catch (error) {
    console.warn('[sitemap] dist directory missing, skipping generation', error);
    process.exit(0);
  }
}

function buildUrl(relativeFile) {
  const cleanPath = relativeFile.replace(/index\.html$/, '');
  const withLeadingSlash = `/${cleanPath}`.replace(/\/+/g, '/');
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`;
}

async function main() {
  await ensureDistDir();
  const files = await collectHtmlFiles(distDir);

  const urlMap = new Map();

  for (const file of files) {
    const urlPath = buildUrl(file);
    const locale = resolveLocale(urlPath);
    const withoutLocale = stripLocalePrefix(urlPath);
    const baseKey = withoutLocale || '/';
    if (!urlMap.has(baseKey)) {
      urlMap.set(baseKey, new Map());
    }

    const localeMap = urlMap.get(baseKey);
    localeMap.set(locale.code, urlPath);
  }

  const urls = [];

  for (const [baseKey, localeMap] of urlMap.entries()) {
    const defaultLocalePath = localeMap.get('en-US') ?? localeMap.values().next().value;
    const defaultUrl = new URL(defaultLocalePath, siteUrl).toString();
    const links = locales
      .map((locale) => {
        const mapped = localeMap.get(locale.code);
        if (!mapped) {
          return null;
        }
        const href = new URL(mapped, siteUrl).toString();
        return `<xhtml:link rel="alternate" hreflang="${locale.code.toLowerCase()}" href="${href}"/>`;
      })
      .filter(Boolean)
      .join('');

    urls.push(
      `<url><loc>${defaultUrl}</loc>${links}<changefreq>weekly</changefreq><priority>0.8</priority></url>`
    );
  }

  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
${urls.join('\n')}
</urlset>`;

  await writeFile(path.join(distDir, 'sitemap.xml'), sitemap);
  console.log(`[sitemap] Generated sitemap with ${urls.length} entries`);
}

main().catch((error) => {
  console.error('[sitemap] Failed to generate sitemap', error);
  process.exitCode = 1;
});




