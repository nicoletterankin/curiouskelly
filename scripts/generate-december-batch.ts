#!/usr/bin/env npx tsx
/**
 * DECEMBER BATCH GENERATOR
 * 
 * Generates watch pages, emails, and HeyGen videos for Days 353-365
 * 
 * Usage:
 *   npx tsx scripts/generate-december-batch.ts
 *   npx tsx scripts/generate-december-batch.ts --dry-run
 *   npx tsx scripts/generate-december-batch.ts --day 353
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const HEYGEN_API_KEY = process.env.HEYGEN_API_KEY!;
const KELLY_VOICE_ID = '0015ce4f932b405b9fc3a5e2f5e92c46';

interface LessonData {
  meta: { day: number; topic: string; emoji?: string; category?: string };
  headline?: string;
  universal_truth?: string;
  fun_facts?: string[];
  phases: {
    hook?: { script: string };
    fact1?: { script: string };
    fact2?: { script: string };
    wisdom?: { script: string };
    [key: string]: any;
  };
  growTrack?: {
    title: string;
    emoji?: string;
    learning_objective?: string;
    activity: string;
  };
}

function getDateForDay(day: number): { month: string; dayOfMonth: number; monthNum: number } {
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  const months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'];
  
  let remaining = day;
  let monthIndex = 0;
  
  while (remaining > monthDays[monthIndex]) {
    remaining -= monthDays[monthIndex];
    monthIndex++;
  }
  
  return { month: months[monthIndex], dayOfMonth: remaining, monthNum: monthIndex + 1 };
}

function loadLesson(day: number): LessonData | null {
  const lessonPath = path.join(process.cwd(), 'public', 'lessons', `day-${day}.json`);
  if (!fs.existsSync(lessonPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(lessonPath, 'utf-8'));
}

function generateWatchPage(lesson: LessonData): string {
  const { day, topic, emoji } = lesson.meta;
  const date = getDateForDay(day);
  const headline = lesson.headline || lesson.universal_truth || '';
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${day}: ${topic} — Curious Kelly</title>
  <meta name="description" content="${headline}. Learn in today's 100-second lesson.">
  
  <!-- Open Graph -->
  <meta property="og:title" content="Day ${day}: ${topic}">
  <meta property="og:description" content="${headline}">
  <meta property="og:type" content="video.other">
  <meta property="og:video" content="https://videos.curiouskelly.com/videos/summary/day-${day}.mp4">
  <meta property="og:image" content="https://videos.curiouskelly.com/videos/summary/day-${day}-thumb.jpg">
  
  <link rel="icon" href="/favicon.ico">
  
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      background: #0a0a0b;
      color: #fafafa;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    
    .video-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    
    video {
      max-width: 100%;
      max-height: 80vh;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    
    .info {
      padding: 24px;
      text-align: center;
      border-top: 1px solid #27272a;
    }
    
    .day-badge {
      font-size: 12px;
      color: #a1a1aa;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 8px;
    }
    
    h1 {
      font-size: 24px;
      font-weight: 500;
      margin-bottom: 8px;
      font-family: Georgia, serif;
    }
    
    .subtitle {
      font-size: 16px;
      color: #a1a1aa;
      margin-bottom: 20px;
      font-family: Georgia, serif;
      font-style: italic;
    }
    
    .cta {
      display: inline-block;
      background: #3b82f6;
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 500;
      transition: background 0.2s;
    }
    
    .cta:hover {
      background: #2563eb;
    }
    
    .footer {
      padding: 16px;
      text-align: center;
      font-size: 12px;
      color: #71717a;
    }
    
    .footer a {
      color: #71717a;
    }
  </style>
</head>
<body>
  <div class="video-container">
    <video 
      controls 
      autoplay 
      playsinline
      poster="https://videos.curiouskelly.com/videos/summary/day-${day}-thumb.jpg">
      <source src="https://videos.curiouskelly.com/videos/summary/day-${day}.mp4" type="video/mp4">
      Your browser doesn't support video playback.
    </video>
  </div>
  
  <div class="info">
    <div class="day-badge">${date.month} ${date.dayOfMonth} · Day ${day} of 365</div>
    <h1>${emoji || '📚'} ${topic}</h1>
    <p class="subtitle">${headline}</p>
    <a href="/learn.html?day=${day}" class="cta">Experience the full lesson →</a>
  </div>
  
  <div class="footer">
    <p>✨ <a href="https://curiouskelly.com">Curious Kelly</a> · Lesson of the Day PBC</p>
  </div>
</body>
</html>`;
}

function generateEmailHTML(lesson: LessonData): string {
  const { day, topic, emoji, category } = lesson.meta;
  const date = getDateForDay(day);
  const headline = lesson.headline || lesson.universal_truth || '';
  const hook = lesson.phases.hook?.script || '';
  const facts = lesson.fun_facts || [];
  const wisdom = lesson.phases.wisdom?.script?.match(/"([^"]+)"/)?.[1] || lesson.universal_truth || '';
  const grow = lesson.growTrack;
  
  // Pick gradient colors based on category
  const gradients: Record<string, { bg: string; color: string }> = {
    'Mindfulness': { bg: 'linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%)', color: '#1e40af' },
    'Meta-Learning': { bg: 'linear-gradient(135deg, #ede9fe 0%, #f3e8ff 100%)', color: '#5b21b6' },
    'Psychology': { bg: 'linear-gradient(135deg, #fce7f3 0%, #fdf2f8 100%)', color: '#9d174d' },
    'Technology': { bg: 'linear-gradient(135deg, #cffafe 0%, #e0f2fe 100%)', color: '#0e7490' },
    'Communication': { bg: 'linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%)', color: '#b45309' },
    'default': { bg: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)', color: '#0369a1' }
  };
  const grad = gradients[category || 'default'] || gradients.default;
  
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${emoji || '📚'} ${topic} | ${date.month} ${date.dayOfMonth} | Curious Kelly</title>
</head>
<body style="margin: 0; padding: 0; background: #fafafa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #fafafa;">
    <tr>
      <td align="center" style="padding: 24px 16px;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 520px;">
          
          <!-- Content -->
          <tr>
            <td style="padding: 32px 24px;">
              
              <!-- Kelly Avatar -->
              <p style="text-align: center; margin: 0 0 24px 0;">
                <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-128.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
              </p>
              
              <!-- Greeting -->
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 20px;">
                Good morning.
              </p>
              
              <!-- VIDEO THUMBNAIL -->
              <table width="100%" cellpadding="0" cellspacing="0" style="margin: 0 0 24px 0;">
                <tr>
                  <td style="position: relative;">
                    <a href="https://curiouskelly.com/watch/day-${day}.html" style="text-decoration: none; display: block;">
                      <img src="https://videos.curiouskelly.com/videos/summary/day-${day}-thumb.jpg" alt="Watch today's lesson" style="width: 100%; border-radius: 12px; display: block;">
                    </a>
                  </td>
                </tr>
              </table>
              
              <!-- Lesson Card -->
              <table width="100%" cellpadding="0" cellspacing="0" style="background: ${grad.bg}; border-radius: 12px; margin: 0 0 24px 0;">
                <tr>
                  <td style="padding: 24px;">
                    <p style="margin: 0 0 4px 0; color: ${grad.color}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-family: -apple-system, sans-serif;">
                      ${date.month} ${date.dayOfMonth} · ${category || 'Learning'}
                    </p>
                    <h2 style="margin: 0 0 12px 0; color: #1f2937; font-size: 24px; font-weight: 600; font-family: Georgia, serif;">
                      ${emoji || '📚'} ${topic}
                    </h2>
                    <p style="margin: 0; color: #374151; font-size: 16px; line-height: 1.6; font-family: Georgia, serif; font-style: italic;">
                      ${headline}
                    </p>
                  </td>
                </tr>
              </table>
              
              <!-- Hook -->
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
                ${hook.split('.').slice(0, 2).join('.')}.
              </p>
              
              <!-- Fun Facts Preview -->
              ${facts.length > 0 ? `<table width="100%" cellpadding="0" cellspacing="0" style="background: #f9fafb; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 0 0 24px 0;">
                <tr>
                  <td style="padding: 16px 20px;">
                    <p style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; font-weight: 600; font-family: -apple-system, sans-serif;">
                      💡 What you'll discover:
                    </p>
                    <ul style="margin: 0; padding: 0 0 0 18px; color: #4b5563; font-size: 14px; line-height: 1.7;">
                      ${facts.slice(0, 3).map(f => `<li>${f}</li>`).join('\n                      ')}
                    </ul>
                  </td>
                </tr>
              </table>` : ''}
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 28px;">
                Five minutes. I think you'll love it.
              </p>
              
              <!-- CTA Button -->
              <p style="text-align: center; margin: 0 0 28px 0;">
                <a href="https://curiouskelly.com/day/${day}" style="display: inline-block; background: #3b82f6; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 16px; font-weight: 600;">
                  Let's learn together →
                </a>
              </p>
              
              <!-- GROW Track Card -->
              ${grow ? `<table width="100%" cellpadding="0" cellspacing="0" style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 12px; margin: 0 0 24px 0;">
                <tr>
                  <td style="padding: 24px;">
                    <p style="margin: 0 0 4px 0; color: #047857; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-family: -apple-system, sans-serif;">
                      ${grow.emoji || '🎯'} Today's Growth Challenge
                    </p>
                    <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 18px; font-weight: 600; font-family: Georgia, serif;">
                      ${grow.title.replace(/^[^-]+ - /, '')}
                    </h3>
                    <p style="margin: 0; color: #374151; font-size: 15px; line-height: 1.6; font-family: Georgia, serif;">
                      ${grow.activity}
                    </p>
                  </td>
                </tr>
              </table>` : ''}
              
              <!-- Wisdom Preview -->
              ${wisdom ? `<table width="100%" cellpadding="0" cellspacing="0" style="background: #1f2937; border-radius: 12px; margin: 0 0 28px 0;">
                <tr>
                  <td style="padding: 20px 24px; text-align: center;">
                    <p style="margin: 0 0 8px 0; color: #9ca3af; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">
                      ✨ Today's Wisdom
                    </p>
                    <p style="margin: 0; color: #f9fafb; font-size: 16px; font-style: italic; line-height: 1.6; font-family: Georgia, serif;">
                      "${wisdom}"
                    </p>
                  </td>
                </tr>
              </table>` : ''}
              
              <!-- Sign off -->
              <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
                — Kelly
              </p>
              
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="font-family: -apple-system, sans-serif; font-size: 12px; color: #9ca3af; margin: 0 0 8px;">
                ${date.month} ${date.dayOfMonth} · <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a>
              </p>
              <p style="font-family: -apple-system, sans-serif; font-size: 11px; color: #9ca3af; margin: 0;">
                <a href="https://curiouskelly.com/api/unsubscribe?token={{UNSUBSCRIBE_TOKEN}}" style="color: #9ca3af;">Unsubscribe from daily emails</a>
              </p>
              <p style="font-family: -apple-system, sans-serif; font-size: 10px; color: #d1d5db; margin: 12px 0 0 0;">
                Lesson of the Day PBC · hello@curiouskelly.com
              </p>
            </td>
          </tr>
          
        </table>
      </td>
    </tr>
  </table>
</body>
</html>`;
}

async function main() {
  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📅 DECEMBER BATCH GENERATOR                                   ║');
  console.log('║  Creating watch pages, emails, and videos for Days 353-365     ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const singleDay = args.find(a => a.startsWith('--day='))?.split('=')[1];
  
  // Parse day range from args
  let days: number[];
  const rangeArg = args.find(a => a.startsWith('--range='))?.split('=')[1];
  if (singleDay) {
    days = [parseInt(singleDay)];
  } else if (rangeArg) {
    const [start, end] = rangeArg.split('-').map(Number);
    days = Array.from({ length: end - start + 1 }, (_, i) => start + i);
  } else {
    days = Array.from({ length: 13 }, (_, i) => 353 + i);
  }
  
  console.log(`\n🗓️  Processing days: ${days.join(', ')}`);
  if (dryRun) console.log('⚠️  DRY RUN - no files will be created');
  
  let created = 0;
  let skipped = 0;
  
  for (const day of days) {
    console.log(`\n━━━ Day ${day} ━━━`);
    
    const lesson = loadLesson(day);
    if (!lesson) {
      console.log(`   ❌ Lesson JSON not found`);
      skipped++;
      continue;
    }
    
    console.log(`   📚 ${lesson.meta.emoji || ''} ${lesson.meta.topic}`);
    
    const watchPagePath = path.join(process.cwd(), 'public', 'watch', `day-${day}.html`);
    const emailPath = path.join(process.cwd(), 'generated-emails', `day-${day}-email.html`);
    
    // Generate watch page
    if (!fs.existsSync(watchPagePath)) {
      const watchHtml = generateWatchPage(lesson);
      if (!dryRun) {
        fs.writeFileSync(watchPagePath, watchHtml);
      }
      console.log(`   ✅ Watch page created`);
    } else {
      console.log(`   ⏭️  Watch page exists`);
    }
    
    // Generate email
    if (!fs.existsSync(emailPath)) {
      const emailHtml = generateEmailHTML(lesson);
      if (!dryRun) {
        fs.writeFileSync(emailPath, emailHtml);
      }
      console.log(`   ✅ Email HTML created`);
    } else {
      console.log(`   ⏭️  Email exists`);
    }
    
    created++;
  }
  
  console.log('');
  console.log('════════════════════════════════════════════════════════════════');
  console.log(`✅ BATCH COMPLETE: ${created} days processed, ${skipped} skipped`);
  console.log('════════════════════════════════════════════════════════════════');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
