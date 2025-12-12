/**
 * Dynamic Lesson Page
 * 
 * Serves lesson content for /day/{number}
 * Renders a beautiful, Kelly-style lesson page
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface Lesson {
  day_number: number;
  title: string;
  emoji: string;
  content: { description?: string };
}

function generateLessonPage(lesson: Lesson, prevDay: number | null, nextDay: number | null): string {
  const description = lesson.content?.description || '';
  
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${lesson.emoji} ${lesson.title} - Day ${lesson.day_number} | Curious Kelly</title>
  <meta name="description" content="${description}">
  
  <!-- Open Graph -->
  <meta property="og:title" content="${lesson.emoji} ${lesson.title}">
  <meta property="og:description" content="${description}">
  <meta property="og:type" content="article">
  <meta property="og:url" content="https://curiouskelly.com/day/${lesson.day_number}">
  <meta property="og:image" content="https://curiouskelly.com/og/day-${lesson.day_number}.png">
  
  <!-- Twitter -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${lesson.emoji} ${lesson.title}">
  <meta name="twitter:description" content="${description}">
  
  <link rel="icon" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
  
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    :root {
      --bg: #fafaf9;
      --text: #1c1917;
      --text-muted: #57534e;
      --accent: #2563eb;
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
      margin-bottom: 16px;
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
      margin-bottom: 24px;
      letter-spacing: -0.02em;
    }
    
    .description {
      font-size: 21px;
      color: var(--text-muted);
      margin-bottom: 48px;
      font-style: italic;
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
  </style>
</head>
<body>
  <main class="container">
    <div class="day-badge">Day ${lesson.day_number} of 365</div>
    
    <span class="lesson-icon">${lesson.emoji}</span>
    
    <h1>${lesson.title}</h1>
    
    <p class="description">${description}</p>
    
    <div class="lesson-content">
      <p>
        This lesson is coming soon. We're building something special — 365 days of wonder, one lesson at a time.
      </p>
      <p>
        Want to be notified when this lesson is ready? Sign up for daily emails and you'll never miss a thing.
      </p>
    </div>
    
    <div class="cta">
      <p>Get daily lessons delivered to your inbox</p>
      <a href="https://curiouskelly.com/#signup">Subscribe →</a>
    </div>
    
    <nav class="nav">
      ${prevDay ? `<a href="/day/${prevDay}">← Day ${prevDay}</a>` : '<span></span>'}
      ${nextDay ? `<a href="/day/${nextDay}">Day ${nextDay} →</a>` : '<span></span>'}
    </nav>
    
    <footer class="footer">
      <a href="https://curiouskelly.com">✨ Curious Kelly</a>
    </footer>
  </main>
</body>
</html>
  `.trim();
}

function generate404Page(dayNumber: number): string {
  return `
<!DOCTYPE html>
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
      background: #2563eb;
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
    <a href="https://curiouskelly.com" class="button">Go Home</a>
  </div>
</body>
</html>
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { number } = req.query;
  const dayNumber = parseInt(number as string, 10);

  if (isNaN(dayNumber) || dayNumber < 1 || dayNumber > 365) {
    res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=86400, stale-while-revalidate=604800');
    return res.status(404).send(generate404Page(dayNumber || 0));
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).send('Configuration error');
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const { data: lesson, error } = await supabase
      .from('lessons')
      .select('day_number, title, emoji, content')
      .eq('day_number', dayNumber)
      .single();

    if (error || !lesson) {
      res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=86400, stale-while-revalidate=604800');
      return res.status(404).send(generate404Page(dayNumber));
    }

    const prevDay = dayNumber > 1 ? dayNumber - 1 : null;
    const nextDay = dayNumber < 365 ? dayNumber + 1 : null;

    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Cache-Control', 'public, max-age=0, s-maxage=3600, stale-while-revalidate=86400');
    
    return res.status(200).send(generateLessonPage(lesson, prevDay, nextDay));

  } catch (error) {
    console.error('Lesson page error:', error);
    return res.status(500).send('Something went wrong');
  }
}

