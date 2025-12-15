#!/usr/bin/env npx tsx
/**
 * ✅ GEMINI INFOGRAPHIC PROOF (LOCAL-ONLY)
 *
 * Goal: prove we can generate on-brand, production-quality infographics using Gemini,
 * without ever uploading to Supabase.
 *
 * Strategy:
 * - Gemini produces a *structured* infographic brief (no coordinates).
 * - We render a brand-locked SVG (crisp real text, consistent palette, deterministic layout).
 *
 * Usage:
 *   npx tsx scripts/gemini-infographic-proof.ts --day=7
 *   npx tsx scripts/gemini-infographic-proof.ts --day=7 --template=cross_section
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';

const BRAND = {
  bg: '#0a0a0b',
  card: '#18181b',
  border: '#27272a',
  text: '#f4f4f5',
  muted: '#a1a1aa',
  dim: '#71717a',
  accent: '#3b82f6',
  gold: '#fbbf24',
  ok: '#22c55e',
  warn: '#f59e0b',
};

type Template = 'cross_section' | 'process_flow' | 'compare';

type Brief = {
  template: Template;
  headline: string;
  subhead: string;
  // Up to 5 labeled callouts; labels must be short (<= 4 words)
  callouts: Array<{ label: string; detail: string; icon: 'dot' | 'spark' | 'arrow' | 'atom' | 'heart' | 'leaf' | 'wave' }>;
  // Optional: 3-step flow
  steps?: Array<{ label: string; detail: string; icon: Brief['callouts'][number]['icon'] }>;
  // Optional: compare panels
  left?: { label: string; bullets: string[] };
  right?: { label: string; bullets: string[] };
};

function arg(name: string): string | undefined {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.split('=')[1] : undefined;
}

function escapeXml(s: string) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function clampWords(s: string, maxWords: number) {
  const parts = s.trim().split(/\s+/);
  if (parts.length <= maxWords) return s.trim();
  return parts.slice(0, maxWords).join(' ');
}

function ensureBriefQuality(b: Brief): Brief {
  b.headline = clampWords(b.headline, 8);
  b.subhead = clampWords(b.subhead, 16);
  b.callouts = (b.callouts || []).slice(0, 5).map((c) => ({
    ...c,
    label: clampWords(c.label, 4),
    detail: clampWords(c.detail, 18),
  }));
  if (b.steps) {
    b.steps = b.steps.slice(0, 3).map((s) => ({
      ...s,
      label: clampWords(s.label, 4),
      detail: clampWords(s.detail, 14),
    }));
  }
  return b;
}

function svgHeader(width: number, height: number) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n` +
    `<defs>\n` +
    `<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">\n` +
    `<stop offset="0" stop-color="${BRAND.bg}"/>\n` +
    `<stop offset="1" stop-color="#101013"/>\n` +
    `</linearGradient>\n` +
    `<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">\n` +
    `<feDropShadow dx="0" dy="8" stdDeviation="14" flood-color="#000" flood-opacity="0.45"/>\n` +
    `</filter>\n` +
    `</defs>\n`;
}

function svgFooter() {
  return `</svg>\n`;
}

function renderCrossSection(b: Brief) {
  const W = 1344;
  const H = 768;
  const pad = 48;
  const headerH = 120;

  const cx = 420;
  const cy = 420;
  const rOuter = 230;
  const rMid = 160;
  const rCore = 80;

  const callouts = b.callouts.slice(0, 5);
  const rightX = 780;
  const cardW = 516;
  const cardH = 520;

  const icon = (kind: Brief['callouts'][number]['icon'], x: number, y: number) => {
    switch (kind) {
      case 'atom':
        return `
          <g opacity="0.95">
            <circle cx="${x}" cy="${y}" r="8" fill="${BRAND.accent}"/>
            <ellipse cx="${x}" cy="${y}" rx="18" ry="8" fill="none" stroke="${BRAND.accent}" stroke-width="2" opacity="0.8"/>
            <ellipse cx="${x}" cy="${y}" rx="8" ry="18" fill="none" stroke="${BRAND.accent}" stroke-width="2" opacity="0.55"/>
          </g>`;
      case 'spark':
        return `<path d="M ${x} ${y-12} L ${x+6} ${y-2} L ${x+18} ${y} L ${x+6} ${y+2} L ${x} ${y+12} L ${x-6} ${y+2} L ${x-18} ${y} L ${x-6} ${y-2} Z" fill="${BRAND.gold}"/>`;
      case 'leaf':
        return `<path d="M ${x} ${y} C ${x+16} ${y-18}, ${x+38} ${y-12}, ${x+42} ${y+6} C ${x+44} ${y+22}, ${x+22} ${y+32}, ${x+8} ${y+26} C ${x-8} ${y+18}, ${x-14} ${y+8}, ${x} ${y} Z" fill="${BRAND.ok}" opacity="0.9"/>`;
      case 'heart':
        return `<path d="M ${x} ${y+8} C ${x-18} ${y-8}, ${x-6} ${y-24}, ${x} ${y-10} C ${x+6} ${y-24}, ${x+18} ${y-8}, ${x} ${y+8} Z" fill="#ef4444" opacity="0.9"/>`;
      case 'wave':
        return `<path d="M ${x-22} ${y} C ${x-10} ${y-12}, ${x+10} ${y+12}, ${x+22} ${y}" fill="none" stroke="#60a5fa" stroke-width="3" stroke-linecap="round"/>`;
      case 'arrow':
        return `<path d="M ${x-18} ${y} H ${x+16} M ${x+16} ${y} l -8 -8 M ${x+16} ${y} l -8 8" stroke="${BRAND.accent}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`;
      default:
        return `<circle cx="${x}" cy="${y}" r="8" fill="${BRAND.accent}"/>`;
    }
  };

  const rows = callouts.map((c, i) => {
    const y = pad + headerH + 28 + i * 92;
    return `
      <g>
        ${icon(c.icon, rightX + 26, y)}
        <text x="${rightX + 60}" y="${y - 6}" fill="${BRAND.text}" font-size="20" font-weight="700" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(c.label)}</text>
        <text x="${rightX + 60}" y="${y + 22}" fill="${BRAND.muted}" font-size="15" font-weight="500" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(c.detail)}</text>
      </g>`;
  }).join('\n');

  return (
    svgHeader(W, H) +
    `<rect x="0" y="0" width="${W}" height="${H}" fill="url(#bg)"/>\n` +
    // Header
    `<text x="${pad}" y="${pad + 38}" fill="${BRAND.text}" font-size="34" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.headline)}</text>\n` +
    `<text x="${pad}" y="${pad + 72}" fill="${BRAND.muted}" font-size="18" font-weight="600" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.subhead)}</text>\n` +

    // Left diagram card
    `<g filter="url(#softShadow)">\n` +
    `<rect x="${pad}" y="${pad + headerH}" width="660" height="${H - pad - headerH - pad}" rx="18" fill="${BRAND.card}" stroke="${BRAND.border}"/>\n` +
    `</g>\n` +

    // Cross-section diagram
    `<g>
      <circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="#111827" stroke="${BRAND.border}" stroke-width="2"/>
      <circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="#f59e0b" opacity="0.15"/>
      <circle cx="${cx}" cy="${cy}" r="${rMid}" fill="#fbbf24" opacity="0.22"/>
      <circle cx="${cx}" cy="${cy}" r="${rCore}" fill="#fde68a" opacity="0.85"/>
      <circle cx="${cx}" cy="${cy}" r="${rCore}" fill="#ffffff" opacity="0.20"/>
      <path d="M ${cx-rOuter} ${cy} A ${rOuter} ${rOuter} 0 0 1 ${cx+rOuter} ${cy}" stroke="#60a5fa" stroke-width="3" opacity="0.55" fill="none"/>
      <path d="M ${cx-rMid} ${cy+14} A ${rMid} ${rMid} 0 0 0 ${cx+rMid} ${cy+14}" stroke="#3b82f6" stroke-width="3" opacity="0.45" fill="none"/>
    </g>\n` +

    // Right callouts card
    `<g filter="url(#softShadow)">\n` +
    `<rect x="${rightX}" y="${pad + headerH}" width="${cardW}" height="${cardH}" rx="18" fill="${BRAND.card}" stroke="${BRAND.border}"/>\n` +
    `</g>\n` +
    rows +

    // Footer badge
    `<g>
      <rect x="${pad}" y="${H - pad - 44}" width="260" height="34" rx="10" fill="#0b1220" stroke="${BRAND.border}"/>
      <circle cx="${pad + 18}" cy="${H - pad - 27}" r="6" fill="${BRAND.accent}"/>
      <text x="${pad + 32}" y="${H - pad - 22}" fill="${BRAND.muted}" font-size="13" font-weight="700" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">Curious Kelly • Infographic</text>
    </g>\n` +
    svgFooter()
  );
}

function renderProcessFlow(b: Brief) {
  const W = 1344;
  const H = 768;
  const pad = 48;
  const headerH = 120;

  const steps = (b.steps || []).slice(0, 3);
  while (steps.length < 3) {
    steps.push({ label: 'Step', detail: 'Add detail', icon: 'arrow' });
  }

  const boxW = 380;
  const boxH = 220;
  const gap = 30;
  const y = pad + headerH + 80;
  const x0 = pad;

  const icon = (kind: Brief['callouts'][number]['icon'], x: number, y: number) => {
    if (kind === 'spark') return `<circle cx="${x}" cy="${y}" r="10" fill="${BRAND.gold}"/>`;
    if (kind === 'leaf') return `<circle cx="${x}" cy="${y}" r="10" fill="${BRAND.ok}"/>`;
    if (kind === 'atom') return `<circle cx="${x}" cy="${y}" r="10" fill="${BRAND.accent}"/>`;
    return `<circle cx="${x}" cy="${y}" r="10" fill="${BRAND.accent}"/>`;
  };

  const stepBox = (i: number) => {
    const x = x0 + i * (boxW + gap);
    const s = steps[i];
    return `
      <g filter="url(#softShadow)">
        <rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" rx="18" fill="${BRAND.card}" stroke="${BRAND.border}"/>
      </g>
      <g>
        ${icon(s.icon, x + 34, y + 40)}
        <text x="${x + 58}" y="${y + 46}" fill="${BRAND.text}" font-size="22" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(s.label)}</text>
        <text x="${x + 34}" y="${y + 86}" fill="${BRAND.muted}" font-size="16" font-weight="600" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(s.detail)}</text>
        <path d="M ${x + 34} ${y + 120} h ${boxW - 68}" stroke="${BRAND.border}" stroke-width="2"/>
        <path d="M ${x + 34} ${y + 160} C ${x + 90} ${y + 130}, ${x + 190} ${y + 190}, ${x + 320} ${y + 156}" stroke="${BRAND.accent}" stroke-width="4" fill="none" opacity="0.75" stroke-linecap="round"/>
      </g>`;
  };

  const arrows = `
    <path d="M ${x0 + boxW + 10} ${y + boxH / 2} H ${x0 + boxW + gap - 10}" stroke="${BRAND.accent}" stroke-width="4" opacity="0.65"/>
    <path d="M ${x0 + boxW + gap - 10} ${y + boxH / 2} l -10 -10 M ${x0 + boxW + gap - 10} ${y + boxH / 2} l -10 10" stroke="${BRAND.accent}" stroke-width="4" opacity="0.65" stroke-linecap="round"/>

    <path d="M ${x0 + 2 * boxW + gap + 10} ${y + boxH / 2} H ${x0 + 2 * boxW + 2 * gap - 10}" stroke="${BRAND.accent}" stroke-width="4" opacity="0.65"/>
    <path d="M ${x0 + 2 * boxW + 2 * gap - 10} ${y + boxH / 2} l -10 -10 M ${x0 + 2 * boxW + 2 * gap - 10} ${y + boxH / 2} l -10 10" stroke="${BRAND.accent}" stroke-width="4" opacity="0.65" stroke-linecap="round"/>
  `;

  return (
    svgHeader(W, H) +
    `<rect x="0" y="0" width="${W}" height="${H}" fill="url(#bg)"/>\n` +
    `<text x="${pad}" y="${pad + 38}" fill="${BRAND.text}" font-size="34" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.headline)}</text>\n` +
    `<text x="${pad}" y="${pad + 72}" fill="${BRAND.muted}" font-size="18" font-weight="600" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.subhead)}</text>\n` +
    arrows +
    stepBox(0) +
    stepBox(1) +
    stepBox(2) +
    svgFooter()
  );
}

function renderCompare(b: Brief) {
  const W = 1344;
  const H = 768;
  const pad = 48;
  const headerH = 120;

  const left = b.left || { label: 'Left', bullets: [] };
  const right = b.right || { label: 'Right', bullets: [] };

  const cardY = pad + headerH;
  const cardH = H - pad - headerH - pad;
  const cardW = (W - pad * 2 - 24) / 2;

  const bullets = (items: string[], x: number, y: number) => {
    return items.slice(0, 4).map((t, i) => {
      const yy = y + i * 40;
      return `
        <g>
          <circle cx="${x}" cy="${yy - 6}" r="5" fill="${BRAND.gold}"/>
          <text x="${x + 16}" y="${yy}" fill="${BRAND.muted}" font-size="16" font-weight="600" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(clampWords(t, 10))}</text>
        </g>`;
    }).join('\n');
  };

  return (
    svgHeader(W, H) +
    `<rect x="0" y="0" width="${W}" height="${H}" fill="url(#bg)"/>\n` +
    `<text x="${pad}" y="${pad + 38}" fill="${BRAND.text}" font-size="34" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.headline)}</text>\n` +
    `<text x="${pad}" y="${pad + 72}" fill="${BRAND.muted}" font-size="18" font-weight="600" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(b.subhead)}</text>\n` +

    `<g filter="url(#softShadow)"><rect x="${pad}" y="${cardY}" width="${cardW}" height="${cardH}" rx="18" fill="${BRAND.card}" stroke="${BRAND.border}"/></g>\n` +
    `<g filter="url(#softShadow)"><rect x="${pad + cardW + 24}" y="${cardY}" width="${cardW}" height="${cardH}" rx="18" fill="${BRAND.card}" stroke="${BRAND.border}"/></g>\n` +

    `<text x="${pad + 24}" y="${cardY + 48}" fill="${BRAND.text}" font-size="22" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(clampWords(left.label, 5))}</text>\n` +
    `<text x="${pad + cardW + 48}" y="${cardY + 48}" fill="${BRAND.text}" font-size="22" font-weight="800" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI">${escapeXml(clampWords(right.label, 5))}</text>\n` +

    bullets(left.bullets || [], pad + 28, cardY + 96) +
    bullets(right.bullets || [], pad + cardW + 52, cardY + 96) +

    svgFooter()
  );
}

function renderSvg(b: Brief) {
  if (b.template === 'process_flow') return renderProcessFlow(b);
  if (b.template === 'compare') return renderCompare(b);
  return renderCrossSection(b);
}

async function loadLesson(day: number): Promise<{ topic: string; truth: string } | null> {
  const url = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;

  const sb = createClient(url, key);
  const { data } = await sb
    .from('core_lessons')
    .select('topic, universal_truth')
    .eq('day_number', day)
    .maybeSingle();

  if (!data?.topic) return null;
  return { topic: data.topic, truth: data.universal_truth || '' };
}

async function generateBrief(day: number, forcedTemplate?: Template): Promise<Brief> {
  const key = process.env.GEMINI_API_KEY || process.env.GOOGLE_AI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!key) throw new Error('Missing GEMINI_API_KEY/GOOGLE_AI_API_KEY');

  const lesson = await loadLesson(day);
  const title = lesson?.topic || `Day ${day}`;
  const objective = lesson?.truth || '';

  const genAI = new GoogleGenerativeAI(key);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

  const systemStyle = {
    palette: BRAND,
    typography: {
      headline: 'bold, modern, clean',
      body: 'simple, minimal, kid-friendly',
    },
    rules: [
      'Keep labels short (<= 4 words).',
      'No scientific misinformation.',
      'Avoid jargon; if needed, use simple words.',
      'No UI screenshots, no watermarks, no logos.',
      'Design must look like a premium education product.',
    ],
  };

  const templateList = ['cross_section', 'process_flow', 'compare'] as const;

  const prompt = `
You are Kelly's visual design lead.

Create a SINGLE infographic brief as JSON.

Brand style:
- Dark premium background, clean neon-accent lines.
- Palette (must use): ${JSON.stringify(systemStyle.palette)}
- Headline text is crisp and minimal. Avoid long text.

Hard constraints:
- Output MUST be valid JSON only.
- Labels must be <= 4 words.
- Details must be <= 18 words.
- No mention of "text inside image" (we render real text).

Lesson:
- Day: ${day}
- Topic: ${title}
- Universal truth/objective: ${objective}

Choose ONE template from: ${templateList.join(', ')}.
${forcedTemplate ? `Template MUST be: ${forcedTemplate}` : ''}

Return JSON with EXACT shape:
{
  "template": "cross_section" | "process_flow" | "compare",
  "headline": "...",
  "subhead": "...",
  "callouts": [ { "label": "...", "detail": "...", "icon": "dot"|"spark"|"arrow"|"atom"|"heart"|"leaf"|"wave" } ],
  "steps": [ { "label": "...", "detail": "...", "icon": "dot"|"spark"|"arrow"|"atom"|"heart"|"leaf"|"wave" } ],
  "left": { "label": "...", "bullets": ["...", "..."] },
  "right": { "label": "...", "bullets": ["...", "..."] }
}

Notes:
- Include only the fields that make sense for the chosen template.
- For cross_section: use callouts.
- For process_flow: use steps.
- For compare: use left/right.
`.trim();

  const res = await model.generateContent({
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { responseMimeType: 'application/json' },
  });

  const raw = res.response.text();
  const parsed = JSON.parse(raw) as Brief;
  return ensureBriefQuality(parsed);
}

async function main() {
  const day = Number(arg('day') || '7');
  const forcedTemplate = arg('template') as Template | undefined;

  const brief = await generateBrief(day, forcedTemplate);

  const outDir = path.join(process.cwd(), 'public', 'infographic-proof');
  fs.mkdirSync(outDir, { recursive: true });

  const svg = renderSvg(brief);
  const outPath = path.join(outDir, `day-${String(day).padStart(3, '0')}-${brief.template}.svg`);
  fs.writeFileSync(outPath, svg, 'utf8');

  const jsonPath = outPath.replace(/\.svg$/, '.brief.json');
  fs.writeFileSync(jsonPath, JSON.stringify(brief, null, 2), 'utf8');

  console.log('✅ Wrote:');
  console.log(`- ${outPath}`);
  console.log(`- ${jsonPath}`);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
