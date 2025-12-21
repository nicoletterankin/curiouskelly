/**
 * Dynamic Lesson Page
 * 
 * Serves lesson content for /day/{number}
 * Queries core_lessons table for proper content
 * Includes SEO-optimized meta tags and Schema.org JSON-LD
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from '../lib/supabase';
import { loadStaticLesson } from '../lib/static-lessons';

interface CoreLesson {
  id: string;
  day_number: number;
  topic: string;
  universal_truth: string;
  marketing_headline: string;
  marketing_tagline?: string;
  marketing_pitch?: string;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function getDayDate(dayNumber: number): string {
  // Convert day number to a date in 2025
  const date = new Date(2025, 0, dayNumber);
  return date.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
}

function generateLessonPage(lesson: CoreLesson, prevDay: number | null, nextDay: number | null): string {
  const topic = escapeHtml(lesson.topic || 'Daily Discovery');
  const headline = escapeHtml(lesson.marketing_headline || lesson.topic);
  const tagline = escapeHtml(lesson.marketing_tagline || '');
  const universalTruth = escapeHtml(lesson.universal_truth || '');
  const dateStr = getDayDate(lesson.day_number);
  
  const schemaOrg = {
    "@context": "https://schema.org",
    "@type": "LearningResource",
    "name": topic,
    "description": headline,
    "educationalLevel": "All ages",
    "learningResourceType": "Lesson",
    "timeRequired": "PT5M",
    "datePublished": dateStr,
    "provider": {
      "@type": "Organization",
      "name": "Curious Kelly",
      "url": "https://curiouskelly.com"
    },
    "url": `https://curiouskelly.com/day/${lesson.day_number}`
  };
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${topic} - Day ${lesson.day_number} | Curious Kelly</title>
  <meta name="description" content="${headline}">
  
  <!-- Open Graph -->
  <meta property="og:title" content="${topic} - Day ${lesson.day_number}">
  <meta property="og:description" content="${headline}">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://curiouskelly.com/day/${lesson.day_number}">
  <meta property="og:image" content="https://curiouskelly.com/api/og/${lesson.day_number}">
  <meta property="og:site_name" content="Curious Kelly">
  
  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${topic}">
  <meta name="twitter:description" content="${headline}">
  <meta name="twitter:image" content="https://curiouskelly.com/api/og/${lesson.day_number}">
  <meta name="twitter:site" content="@CuriousKelly">
  
  <link rel="canonical" href="https://curiouskelly.com/day/${lesson.day_number}">
  <link rel="icon" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
  
  <!-- Schema.org JSON-LD -->
  <script type="application/ld+json">${JSON.stringify(schemaOrg)}</script>
  
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    :root {
      --bg: #fafaf9;
      --text: #1c1917;
      --text-muted: #57534e;
      --accent: #3b82f6;
      --border: #e7e5e4;
    }
    
    body {
      font-family: 'Crimson Pro', Georgia, serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      min-height: 100vh;
    }
    
    .container {
      max-width: 640px;
      margin: 0 auto;
      padding: 60px 24px 80px;
    }
    
    .day-badge {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    
    .date-line {
      font-size: 15px;
      color: var(--text-muted);
      margin-bottom: 24px;
    }
    
    .lesson-icon {
      font-size: 64px;
      margin-bottom: 24px;
      display: block;
    }
    
    h1 {
      font-size: 36px;
      font-weight: 500;
      line-height: 1.3;
      margin-bottom: 16px;
      letter-spacing: -0.02em;
    }
    
    .tagline {
      font-size: 21px;
      color: var(--accent);
      font-weight: 500;
      margin-bottom: 24px;
    }
    
    .description {
      font-size: 21px;
      color: var(--text-muted);
      margin-bottom: 32px;
      font-style: italic;
    }
    
    .universal-truth {
      font-size: 19px;
      line-height: 1.9;
      padding: 24px;
      background: #f5f5f4;
      border-left: 4px solid var(--accent);
      border-radius: 0 8px 8px 0;
      margin-bottom: 32px;
    }
    
    .lesson-content {
      font-size: 19px;
      line-height: 1.9;
    }
    
    .lesson-content p {
      margin-bottom: 24px;
    }
    
    .cta {
      margin-top: 48px;
      padding: 32px;
      background: white;
      border: 1px solid var(--border);
      border-radius: 12px;
      text-align: center;
    }
    
    .cta p {
      font-size: 17px;
      color: var(--text-muted);
      margin-bottom: 16px;
    }
    
    .cta a {
      display: inline-block;
      background: var(--accent);
      color: white;
      padding: 14px 28px;
      border-radius: 8px;
      text-decoration: none;
      font-family: -apple-system, sans-serif;
      font-size: 15px;
      font-weight: 500;
      transition: background 0.2s;
    }
    
    .cta a:hover {
      background: #2563eb;
    }
    
    .nav {
      display: flex;
      justify-content: space-between;
      margin-top: 48px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font-family: -apple-system, sans-serif;
      font-size: 14px;
    }
    
    .nav a {
      color: var(--accent);
      text-decoration: none;
    }
    
    .nav a:hover {
      text-decoration: underline;
    }
    
    .footer {
      margin-top: 64px;
      text-align: center;
      font-size: 14px;
      color: var(--text-muted);
    }
    
    .footer a {
      color: var(--text-muted);
    }
    
    @media (max-width: 640px) {
      .container { padding: 40px 20px 60px; }
      h1 { font-size: 28px; }
      .tagline { font-size: 18px; }
      .description { font-size: 18px; }
    }
  </style>
</head>
<body>
  <main class="container">
    <div class="day-badge">Day ${lesson.day_number} of 365</div>
    <div class="date-line">${dateStr}</div>
    
    <h1>${topic}</h1>
    
    ${tagline ? `<p class="tagline">${tagline}</p>` : ''}
    
    ${headline ? `<p class="description">${headline}</p>` : ''}
    
    ${universalTruth ? `
    <div class="universal-truth">
      ${universalTruth}
    </div>
    ` : ''}
    
    <div class="cta">
      <p>Experience this lesson with Kelly, your AI learning companion</p>
      <a href="/learn.html?day=${lesson.day_number}">Start Learning →</a>
    </div>
    
    <nav class="nav">
      ${prevDay ? `<a href="/day/${prevDay}">← Day ${prevDay}</a>` : '<span></span>'}
      ${nextDay ? `<a href="/day/${nextDay}">Day ${nextDay} →</a>` : '<span></span>'}
    </nav>
    
    <footer class="footer">
      <a href="https://curiouskelly.com">✨ Curious Kelly</a> · 
      <a href="/sitemap.xml">Sitemap</a>
    </footer>
  </main>
  
  <script>
    // Auto-redirect for humans (bots stay for indexing)
    if (!/bot|crawl|spider|googlebot|bingbot|facebookexternalhit/i.test(navigator.userAgent)) {
      // Optional: Uncomment to auto-redirect after brief preview
      // setTimeout(() => { window.location.href = '/learn.html?day=${lesson.day_number}'; }, 3000);
    }
  </script>
</body>
</html>`;
}

function generate404Page(dayNumber: number): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lesson Not Found - Curious Kelly</title>
  <link rel="icon" href="/favicon.ico">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: Georgia, serif;
      background: #fafafa;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .container { max-width: 460px; text-align: center; }
    .icon { font-size: 48px; margin-bottom: 24px; }
    h1 { font-size: 24px; color: #1f2937; font-weight: 500; margin-bottom: 16px; }
    p { font-size: 17px; color: #4b5563; line-height: 1.8; margin-bottom: 24px; }
    a.button {
      display: inline-block;
      background: #3b82f6;
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      text-decoration: none;
      font-family: -apple-system, sans-serif;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">🔍</div>
    <h1>Day ${dayNumber} not found</h1>
    <p>Lessons are numbered 1 to 365. Try a different day?</p>
    <a href="/" class="button">Go Home</a>
  </div>
</body>
</html>`;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { number } = req.query;
  const dayNumber = parseInt(number as string, 10);

  if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
    res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=86400, stale-while-revalidate=604800');
    return res.status(404).send(generate404Page(dayNumber || 0));
  }

  const prevDay = dayNumber > 1 ? dayNumber - 1 : null;
  const nextDay = dayNumber < 365 ? dayNumber + 1 : null;

  // ---------------------------------------------------------------------------
  // PRIORITY 1: Static Files (Zero DB dependency)
  // ---------------------------------------------------------------------------
  try {
    const staticPack = loadStaticLesson(dayNumber);
    if (staticPack) {
      const lesson: CoreLesson = {
        id: `static-${dayNumber}`,
        day_number: dayNumber,
        topic: staticPack.lesson.topic,
        universal_truth: staticPack.lesson.universal_truth,
        marketing_headline: staticPack.lesson.headline,
        marketing_tagline: staticPack.lesson.category,
      };
      
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400');
      return res.status(200).send(generateLessonPage(lesson, prevDay, nextDay));
    }
  } catch (e) {
    console.warn('[api/day/:number] Static file load failed:', e);
  }

  // ---------------------------------------------------------------------------
  // PRIORITY 2: Supabase (fallback)
  // ---------------------------------------------------------------------------
  if (isSupabaseConfigured()) {
    try {
      const supabase = getSupabaseAdmin();
      // Default to 'learn' track to avoid multiple-row error
      const { data: lesson, error } = await supabase
        .from('core_lessons')
        .select('id, day_number, topic, universal_truth, marketing_headline, marketing_tagline, marketing_pitch')
        .eq('day_number', dayNumber)
        .eq('track', 'learn')
        .single();

      if (!error && lesson) {
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400');
        return res.status(200).send(generateLessonPage(lesson, prevDay, nextDay));
      }
    } catch (error) {
      console.error('Lesson page Supabase error:', error);
    }
  }

  // ---------------------------------------------------------------------------
  // PRIORITY 3: 404 (only if both static and DB fail)
  // ---------------------------------------------------------------------------
  res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=86400, stale-while-revalidate=604800');
  return res.status(404).send(generate404Page(dayNumber));
}
