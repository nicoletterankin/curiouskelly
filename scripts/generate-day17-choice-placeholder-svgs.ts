import fs from 'node:fs';
import path from 'node:path';

type Variant = 'option' | 'success' | 'alt';

type Item = {
  file: string;
  phase: string;
  variant: Variant;
  letter: 'a' | 'b';
  title: string;
};

const OUT_DIR = path.join(process.cwd(), 'public', 'generated-images', 'day-017');

function ensureDir(p: string) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function getAccent(variant: Variant) {
  if (variant === 'success') return '#10b981'; // green
  if (variant === 'alt') return '#f59e0b'; // gold
  return '#2563eb'; // kelly blue
}

function svgPlaceholder(opts: { label: string; title: string; accent: string; variant: Variant }) {
  const { label, title, accent, variant } = opts;
  const subtitle = variant === 'option' ? 'Choose' : variant === 'success' ? 'Selected' : 'Alternate';

  // 400x300 (4:3)
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">`,
    `  <defs>`,
    `    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">`,
    `      <stop offset="0" stop-color="#09090b"/>`,
    `      <stop offset="1" stop-color="#18181b"/>`,
    `    </linearGradient>`,
    `    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">`,
    `      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="rgba(0,0,0,0.6)"/>`,
    `    </filter>`,
    `  </defs>`,
    `  <rect width="400" height="300" rx="18" fill="url(#bg)"/>`,
    `  <rect x="14" y="14" width="372" height="272" rx="14" fill="#0f0f13" stroke="rgba(255,255,255,0.10)"/>`,
    `  <rect x="14" y="14" width="372" height="54" rx="14" fill="rgba(255,255,255,0.03)"/>`,
    `  <circle cx="44" cy="41" r="12" fill="${accent}" filter="url(#shadow)"/>`,
    `  <text x="66" y="38" fill="#fafafa" font-size="14" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial" font-weight="700">${escapeXml(label)}</text>`,
    `  <text x="66" y="56" fill="#a1a1aa" font-size="12" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial" font-weight="500">${escapeXml(subtitle)}</text>`,
    `  <g transform="translate(0, 0)">`,
    `    <rect x="52" y="96" width="296" height="140" rx="16" fill="rgba(24,24,27,0.7)" stroke="rgba(255,255,255,0.08)"/>`,
    `    <path d="M92 166h216" stroke="${accent}" stroke-width="10" stroke-linecap="round" opacity="0.65"/>`,
    `    <circle cx="200" cy="166" r="44" fill="rgba(9,9,11,0.9)" stroke="${accent}" stroke-width="6"/>`,
    `    <text x="200" y="176" text-anchor="middle" fill="${accent}" font-size="28" font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial" font-weight="800">${escapeXml(title)}</text>`,
    `  </g>`,
    `</svg>`,
  ].join('\n');
}

function escapeXml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

const ITEMS: Item[] = [
  // Hook
  { file: 'hook_option_a.svg', phase: 'hook', variant: 'option', letter: 'a', title: 'A' },
  { file: 'hook_option_b.svg', phase: 'hook', variant: 'option', letter: 'b', title: 'B' },
  { file: 'hook_success_a.svg', phase: 'hook', variant: 'success', letter: 'a', title: 'A' },
  { file: 'hook_success_b.svg', phase: 'hook', variant: 'success', letter: 'b', title: 'B' },
  { file: 'hook_alt_a.svg', phase: 'hook', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'hook_alt_b.svg', phase: 'hook', variant: 'alt', letter: 'b', title: 'B' },

  // Cliff
  { file: 'cliff_option_a.svg', phase: 'cliff', variant: 'option', letter: 'a', title: 'A' },
  { file: 'cliff_option_b.svg', phase: 'cliff', variant: 'option', letter: 'b', title: 'B' },
  { file: 'cliff_success_a.svg', phase: 'cliff', variant: 'success', letter: 'a', title: 'A' },
  { file: 'cliff_success_b.svg', phase: 'cliff', variant: 'success', letter: 'b', title: 'B' },
  { file: 'cliff_alt_a.svg', phase: 'cliff', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'cliff_alt_b.svg', phase: 'cliff', variant: 'alt', letter: 'b', title: 'B' },

  // Fact1
  { file: 'fact1_option_a.svg', phase: 'fact1', variant: 'option', letter: 'a', title: 'A' },
  { file: 'fact1_option_b.svg', phase: 'fact1', variant: 'option', letter: 'b', title: 'B' },
  { file: 'fact1_success_a.svg', phase: 'fact1', variant: 'success', letter: 'a', title: 'A' },
  { file: 'fact1_success_b.svg', phase: 'fact1', variant: 'success', letter: 'b', title: 'B' },
  { file: 'fact1_alt_a.svg', phase: 'fact1', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'fact1_alt_b.svg', phase: 'fact1', variant: 'alt', letter: 'b', title: 'B' },

  // Fact2
  { file: 'fact2_option_a.svg', phase: 'fact2', variant: 'option', letter: 'a', title: 'A' },
  { file: 'fact2_option_b.svg', phase: 'fact2', variant: 'option', letter: 'b', title: 'B' },
  { file: 'fact2_success_a.svg', phase: 'fact2', variant: 'success', letter: 'a', title: 'A' },
  { file: 'fact2_success_b.svg', phase: 'fact2', variant: 'success', letter: 'b', title: 'B' },
  { file: 'fact2_alt_a.svg', phase: 'fact2', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'fact2_alt_b.svg', phase: 'fact2', variant: 'alt', letter: 'b', title: 'B' },

  // Fact3
  { file: 'fact3_option_a.svg', phase: 'fact3', variant: 'option', letter: 'a', title: 'A' },
  { file: 'fact3_option_b.svg', phase: 'fact3', variant: 'option', letter: 'b', title: 'B' },
  { file: 'fact3_success_a.svg', phase: 'fact3', variant: 'success', letter: 'a', title: 'A' },
  { file: 'fact3_success_b.svg', phase: 'fact3', variant: 'success', letter: 'b', title: 'B' },
  { file: 'fact3_alt_a.svg', phase: 'fact3', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'fact3_alt_b.svg', phase: 'fact3', variant: 'alt', letter: 'b', title: 'B' },

  // Wisdom
  { file: 'wisdom_option_a.svg', phase: 'wisdom', variant: 'option', letter: 'a', title: 'A' },
  { file: 'wisdom_option_b.svg', phase: 'wisdom', variant: 'option', letter: 'b', title: 'B' },
  { file: 'wisdom_success_a.svg', phase: 'wisdom', variant: 'success', letter: 'a', title: 'A' },
  { file: 'wisdom_success_b.svg', phase: 'wisdom', variant: 'success', letter: 'b', title: 'B' },
  { file: 'wisdom_alt_a.svg', phase: 'wisdom', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'wisdom_alt_b.svg', phase: 'wisdom', variant: 'alt', letter: 'b', title: 'B' },

  // Outro
  { file: 'outro_option_a.svg', phase: 'outro', variant: 'option', letter: 'a', title: 'A' },
  { file: 'outro_option_b.svg', phase: 'outro', variant: 'option', letter: 'b', title: 'B' },
  { file: 'outro_success_a.svg', phase: 'outro', variant: 'success', letter: 'a', title: 'A' },
  { file: 'outro_success_b.svg', phase: 'outro', variant: 'success', letter: 'b', title: 'B' },
  { file: 'outro_alt_a.svg', phase: 'outro', variant: 'alt', letter: 'a', title: 'A' },
  { file: 'outro_alt_b.svg', phase: 'outro', variant: 'alt', letter: 'b', title: 'B' },
];

function main() {
  ensureDir(OUT_DIR);

  for (const item of ITEMS) {
    const accent = getAccent(item.variant);
    const label = `Day 17 • ${item.phase.toUpperCase()} • ${item.variant.toUpperCase()} ${item.letter.toUpperCase()}`;
    const outPath = path.join(OUT_DIR, item.file);
    fs.writeFileSync(outPath, svgPlaceholder({ label, title: item.title, accent, variant: item.variant }));
  }

  // eslint-disable-next-line no-console
  console.log(`✅ Wrote ${ITEMS.length} SVG placeholders to ${OUT_DIR}`);
}

main();
